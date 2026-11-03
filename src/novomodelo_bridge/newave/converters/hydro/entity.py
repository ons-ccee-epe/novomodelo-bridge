"""Hydro entity-record construction: the ``hydros.json`` per-plant builder
``convert_hydros``, and its local leaf helpers (mirror unit groups, exph-derived
unit-ramp summaries) with no forward dependency on another seam.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, NamedTuple

import pandas as pd

from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    emit,
)
from novomodelo_bridge.core.diversion import (
    channel_flow_limit,
    is_unlimited_flow,
    unlimited_channels_diagnostic,
)
from novomodelo_bridge.core.hydro_units import build_mirror_unit_group, rated_capacity
from novomodelo_bridge.core.pandas_utils import is_na
from novomodelo_bridge.core.productivity import fpha_efficiency
from novomodelo_bridge.newave.case import NewaveCase
from novomodelo_bridge.newave.converters.hydro.bounds import (
    _operational_caps,
    _per_stage_turbined_envelope,
)
from novomodelo_bridge.newave.converters.hydro.geometry import (
    _EVAP_MONTHS,
    _expansion_configs,
    fpha_eligible_codes,
    operating_storage_range,
)
from novomodelo_bridge.newave.converters.hydro.overrides import (
    _apply_permanent_overrides,
    read_diversion_records,
)
from novomodelo_bridge.newave.converters.hydro.reference_volumes import (
    evaporation_reference_volumes,
)
from novomodelo_bridge.newave.filling import (
    ExpansionConfig,
    exph_unit_rows,
    filling_completion_date,
    filling_min_rate_m3s,
    filling_schedule,
)
from novomodelo_bridge.newave.horizon import (
    StudyHorizon,
    build_stage_dates,
    historical_start_date,
)
from novomodelo_bridge.newave.id_map import NewaveIdMap
from novomodelo_bridge.newave.plants import filling_hydro_codes
from novomodelo_bridge.newave.switches import switch_off_diagnostic
from novomodelo_bridge.novomodelo import schemas as novomodelo_schemas

if TYPE_CHECKING:
    from novomodelo_bridge.newave.converters.fict_cascade import (
        FictCascadeResolution,
    )

_LOG = logging.getLogger(__name__)


def _unit_ramp_summary(
    exph_df: pd.DataFrame,
    newave_code: int,
    start_year: int,
    start_month: int,
) -> str:
    """One-line summary of the units coming online for a filling plant.

    Renders ``"conj <c>: <n> unit(s) @ stage <s>"`` per conjunto (``<s>`` = the
    stage at which the conjunto is fully online) joined by ``"; "``, or ``"—"``
    when the plant has no parsable unit row.
    """
    by_conjunto: dict[int, list[int]] = {}
    for conjunto, stage in exph_unit_rows(
        exph_df, newave_code, start_year, start_month
    ):
        by_conjunto.setdefault(conjunto, []).append(stage)
    if not by_conjunto:
        return "—"
    parts = [
        f"conj {c}: {len(stages)} unit{'s' if len(stages) != 1 else ''} "
        f"@ stage {max(stages)}"
        for c, stages in sorted(by_conjunto.items())
    ]
    return "; ".join(parts)


def _diversion_channels(
    case: NewaveCase, id_map: NewaveIdMap, cadastro: pd.DataFrame
) -> dict[int, dict]:
    """``{plant code: hydros.json diversion channel}`` from ``CDESVIO`` records.

    Only a ``CDESVIO`` record declares a channel; the ``hidr.dat`` ``desvio``
    code alone does not. The channel carries the record's maximum flow, with
    the source model's "no limit" value converted to no limit. A record on a
    plant outside the case is ignored; one whose target is outside the case,
    or that gives no maximum flow, converts to no channel and is reported.
    """
    records = read_diversion_records(case)
    if not records:
        return {}
    switch = case.switches.canal_desvio
    if not switch.on:
        emit(switch_off_diagnostic(switch), logger=_LOG)
        return {}

    def name(code: int) -> str:
        if code in cadastro.index and "nome_usina" in cadastro.columns:
            return str(cadastro.loc[code, "nome_usina"]).strip()
        return ""

    in_case = set(id_map.all_hydro_codes)
    channels: dict[int, dict] = {}
    modeled: list[list[object]] = []
    unlimited: list[tuple[int, str, int, str, float]] = []
    dropped: list[list[object]] = []
    for code, record in sorted(records.items()):
        if code not in in_case:
            continue
        row: list[object] = [code, name(code), record.target, name(record.target)]
        if record.target not in in_case:
            dropped.append([*row, "target plant not in the case"])
        elif record.max_flow_m3s is None:
            dropped.append([*row, "no maximum flow given"])
        else:
            limit = channel_flow_limit(record.max_flow_m3s)
            channels[code] = {
                "downstream_id": id_map.hydro_id(record.target),
                "max_flow_m3s": limit,
            }
            if is_unlimited_flow(record.max_flow_m3s):
                unlimited.append(
                    (
                        code,
                        name(code),
                        record.target,
                        name(record.target),
                        record.max_flow_m3s,
                    )
                )
            flow = (
                "no limit" if limit is None else f"{limit:.2f}".rstrip("0").rstrip(".")
            )
            modeled.append([*row, flow])

    columns = ["Code", "Name", "Target code", "Target name"]
    if modeled:
        emit(
            Diagnostic(
                code="diversion-channels",
                severity=Severity.INFO,
                category="Diversion channels",
                title=f"Diversion channels ({len(modeled)})",
                summary=(
                    f"{len(modeled)} plant(s) divert water to the target plant "
                    "through a channel declared by CDESVIO in modif.dat, up to "
                    "the listed maximum flow."
                ),
                table=DiagnosticTable(
                    columns=[*columns, "Max flow (m³/s)"],
                    rows=modeled,
                    justify=["right", "left", "right", "left", "right"],
                ),
            ),
            logger=_LOG,
        )
    if unlimited:
        emit(
            unlimited_channels_diagnostic(unlimited, source="modif.dat CDESVIO"),
            logger=_LOG,
        )
    if dropped:
        emit(
            Diagnostic(
                code="diversion-channel-dropped",
                severity=Severity.WARNING,
                category="Diversion channels",
                title=f"Diversion channels not converted ({len(dropped)})",
                summary=(
                    f"{len(dropped)} CDESVIO record(s) in modif.dat convert to "
                    "no channel, so the listed plants release water only to "
                    "their downstream plant."
                ),
                table=DiagnosticTable(
                    columns=[*columns, "Reason"],
                    rows=dropped,
                    justify=["right", "left", "right", "left", "left"],
                ),
                remediation=(
                    "Give each listed record a target plant that is in the case "
                    "and the channel's maximum flow in m³/s after the target code."
                ),
            ),
            logger=_LOG,
        )
    return channels


@dataclass(frozen=True)
class _HydroContext:
    """The case-wide inputs every plant's ``hydros.json`` entry reads."""

    cadastro: pd.DataFrame
    diversion_channels: dict[int, dict]
    min_outflow_on: bool
    evaporation_references: dict[int, list[float]]
    exph_df: pd.DataFrame | None
    filling_codes: set[int]
    fict_cascade: dict[int, FictCascadeResolution]
    turbined_envelope: dict[int, float]
    expansion_configs: dict[int, ExpansionConfig]
    active_hydros: pd.DataFrame
    horizon: StudyHorizon
    stage_dates: list[date]
    ree_to_submercado: dict[int, int]
    fpha_codes: set[int]
    existing_op_date: str

    def stage_year_month(self, t: int) -> tuple[int, int]:
        d = self.stage_dates[t]
        return d.year, d.month


def _hydro_context(case: NewaveCase, id_map: NewaveIdMap) -> _HydroContext:
    hidr = case.hidr
    ree_file = case.ree

    cadastro = hidr.cadastro  # DataFrame indexed by codigo_usina (1-based)
    confhd_df = case.hydro_plants
    ree_df = ree_file.rees  # columns: codigo, nome, submercado, ...

    cadastro = _apply_permanent_overrides(cadastro, case)
    diversion_channels = _diversion_channels(case, id_map, cadastro)

    min_outflow_on = case.switches.min_outflow.on
    if not min_outflow_on:
        emit(switch_off_diagnostic(case.switches.min_outflow), logger=_LOG)

    evaporation_references = evaporation_reference_volumes(case, cadastro)

    # NE plants carrying an exph dead-volume filling row are admitted as real
    # downstream nodes (admission predicate, single source of truth). The
    # set is computed once here and threaded into the cascade walker so upstream
    # plants resolve to the filling plant; it also drives each filling plant's
    # FILLING contract (``_filling_phase``). Empty (byte-identical to the no-arg
    # call) when the case has no exph or no NE-with-filling plant.
    exph = case.exph
    exph_df = exph.expansoes if exph is not None else None
    filling_codes = filling_hydro_codes(confhd_df, exph_df)

    # Resolve the FICT-cascade for every real plant.  Provides the effective
    # next-real-plant downstream and the sum of any FICT-chain ρ_eq that must
    # be folded back into the upstream real plant's effective ρ_eq.  See
    # ``novomodelo_bridge.newave.converters.fict_cascade`` for the resolution rules.
    from novomodelo_bridge.newave.converters.fict_cascade import resolve_cascade

    fict_cascade = resolve_cascade(
        confhd_df,
        cadastro,
        fictitious=case.fictitious_codes,
        filling_codes=filling_codes,
    )

    # Per-hydro envelope over the per-stage turbined caps that
    # convert_per_stage_capacity_caps emits into hydro_bounds.parquet.
    # novomodelo rule 43 forbids any hydro_bounds row from raising max_turbined_m3s
    # above the plant's own declared value, so the reference-head value
    # (``_generation_caps``) must be raised to cover every emitted per-stage row.
    # Empty for a hydro with no per-stage turbined cap, which keeps its declared
    # value unchanged.
    turbined_envelope = _per_stage_turbined_envelope(case, id_map)

    expansion_configs = _expansion_configs(case, cadastro)

    active_hydros = case.active_hydros

    # Per-stage (year, month) calendar over the study horizon, built once and
    # passed into ``filling_min_rate_m3s`` for each filling plant.
    # The exph DataFrame is only consulted when ``filling_codes`` is non-empty,
    # which guarantees it is present there.
    horizon = case.horizon
    stage_dates = build_stage_dates(
        horizon.start_year, horizon.start_month, horizon.total_stages
    )

    ree_to_submercado: dict[int, int] = {}
    if ree_df is not None:
        for _, row in ree_df.iterrows():
            ree_to_submercado[int(row["codigo"])] = int(row["submercado"])

    # Plants emitted as FPHA (empty unless dger funcao_producao_uhe == 0).
    fpha_codes = fpha_eligible_codes(case)

    # Default operational_start_date for in-service plants: in service since the
    # historical record. Filling (NE) plants override it (``_filling_phase``) with
    # the calendar month they finish filling and enter operation.
    existing_op_date = historical_start_date(case.dger)

    return _HydroContext(
        cadastro=cadastro,
        diversion_channels=diversion_channels,
        min_outflow_on=min_outflow_on,
        evaporation_references=evaporation_references,
        exph_df=exph_df,
        filling_codes=filling_codes,
        fict_cascade=fict_cascade,
        turbined_envelope=turbined_envelope,
        expansion_configs=expansion_configs,
        active_hydros=active_hydros,
        horizon=horizon,
        stage_dates=stage_dates,
        ree_to_submercado=ree_to_submercado,
        fpha_codes=fpha_codes,
        existing_op_date=existing_op_date,
    )


class _FillingPhase(NamedTuple):
    """A plant's FILLING-contract fields and its filling-plant report row
    (``None`` for a plant that does not fill)."""

    entry_stage_id: int | None
    filling: dict | None
    operational_start_date: str
    report_row: list[object] | None


def _filling_phase(
    newave_code: int, name: str, vol_min: float, context: _HydroContext
) -> _FillingPhase:
    """FILLING phase: an admitted NE plant fills its dead
    volume from its seeded storage up to ``min_storage_hm3`` over the
    half-open window ``[start_sid, entry_sid)``, then enters operation at
    ``entry_sid``. EX plants never enter this branch (filling_codes is the
    NE-with-exph set), so their entry/exit/filling stay None.
    """
    exph_df = context.exph_df
    # filling_codes is empty without an exph (the admission predicate
    # requires one), so the exph_df test only narrows its type.
    if newave_code not in context.filling_codes or exph_df is None:
        return _FillingPhase(None, None, context.existing_op_date, None)
    horizon = context.horizon
    filling_row = exph_df.loc[
        (exph_df["codigo_usina"] == newave_code)
        & exph_df["data_inicio_enchimento"].notna()
    ].iloc[0]
    ts = filling_row["data_inicio_enchimento"]
    duracao = int(filling_row["duracao_enchimento"])
    volume_morto = float(filling_row["volume_morto"])
    start_sid, entry_sid = filling_schedule(
        ts.year, ts.month, duracao, horizon.start_year, horizon.start_month
    )
    # Operational start = the month filling completes and the plant enters
    # service, computed from the raw exph schedule so it is truthful even
    # when completion falls outside the study horizon.
    operational_start_date = filling_completion_date(
        ts.year, ts.month, duracao
    ).isoformat()
    # One INFO-diagnostic row per filling plant (``_report_filling_plants``):
    # its filling window, seeded dead volume, and unit-ramp summary. Built
    # for every filling plant, including the ``duracao == 0`` (no-filling-
    # block) case, so the user always sees the admitted plant.
    report_row: list[object] = [
        name,
        newave_code,
        start_sid,
        entry_sid,
        f"{volume_morto:.1f}",
        _unit_ramp_summary(
            exph_df,
            newave_code,
            horizon.start_year,
            horizon.start_month,
        ),
    ]
    filling: dict | None = None
    if entry_sid > start_sid:
        # JURUENA's single-stage window is anchor-insensitive: the soft
        # target is pinned to ``min_storage_hm3`` regardless of the rate,
        # so the exact per-stage ζ anchor (ζ_t vs ζ_{t+1}) does not matter.
        # TODO(multi-stage-anchor): verify the multi-stage anchor against
        # novomodelo's LP layout builder before shipping a non-trivial
        # (multi-stage) filling rate.
        #
        # Rate clamp: a plant whose filling completes past the
        # study horizon (``entry_sid > horizon.total_stages``) is a VALID
        # case — it fills but never operates within the study, yet is still
        # emitted with its true ``entry_stage_id``. The rate is summed only
        # over in-horizon stages, so clamp the entry passed to
        # ``filling_min_rate_m3s`` to the horizon: post-study stages have no
        # ``stage_dates`` entry (and thus no ζ to sum), and novomodelo handles the
        # remaining out-of-horizon fill. The true (unclamped) ``entry_sid``
        # stays on the hydro record. If ``start_sid >= total_stages``
        # too, ``rate_entry`` collapses to ``start_sid``, giving an empty
        # window → ``filling_min_rate_m3s`` returns 0.0 (no crash).
        rate_entry = min(entry_sid, horizon.total_stages)
        rate = filling_min_rate_m3s(
            vol_min,
            volume_morto,
            start_sid,
            rate_entry,
            context.stage_year_month,
        )
        filling = {
            "start_stage_id": start_sid,
            "filling_min_rate_m3s": rate,
        }
    # entry_sid == start_sid (duracao 0): keep filling None but still set
    # entry_stage_id — novomodelo rejects start_stage_id >= entry_stage_id, so
    # no degenerate filling block is emitted.
    return _FillingPhase(entry_sid, filling, operational_start_date, report_row)


class _GenerationCaps(NamedTuple):
    """A plant's declared generation caps and its expansion-plant report row
    (``None`` for a plant with no machine still to enter)."""

    max_turbined: float
    max_generation: float
    expansion_row: list[object] | None


def _generation_caps(
    newave_code: int,
    name: str,
    hreg: pd.Series,
    hydro_id: int,
    context: _HydroContext,
) -> _GenerationCaps:
    """Generation parameters. Productivity lives in
    ``hydro_production_models.json`` on novomodelo HEAD; callers that need
    the per-hydro base value call ``compute_base_productivities``.
    Turbined cap: the source model's engolimento, _operational_caps.
    Generation cap: rated installed power Σ n·p_nom = the FPHA GHmax the
    fit clamps at; the available power below it is a per-stage
    hydro_bounds row.

    The declared turbined cap is the ENVELOPE over that reference-head
    value and every per-stage cap this hydro emits into hydro_bounds: a
    plant with MODIF.DAT CFUGA/CMONT can turbine more at a higher-head
    stage than at the reference head, and novomodelo rule 43 forbids a
    hydro_bounds row from exceeding the plant's own declared value. A
    hydro with no per-stage turbined cap is absent from
    ``turbined_envelope`` and keeps its reference-head value unchanged.
    """
    config = context.expansion_configs.get(newave_code)
    hreg_declared = hreg if config is None else config.declared_hreg(hreg)
    max_turbined_reference = _operational_caps(hreg_declared, name)[0]
    envelope_value = context.turbined_envelope.get(hydro_id)
    max_turbined = (
        max_turbined_reference
        if envelope_value is None
        else max(max_turbined_reference, envelope_value)
    )
    max_generation = rated_capacity(hreg_declared)[1]
    if config is None:
        return _GenerationCaps(max_turbined, max_generation, None)

    start_generation = rated_capacity(config.hreg_at(hreg, 0))[1]
    expansion_row: list[object] = [
        name,
        newave_code,
        f"{start_generation:.1f}",
        f"{max_generation:.1f}",
        len(config.unit_rows),
        min(stage for _conjunto, stage in config.unit_rows),
    ]
    return _GenerationCaps(max_turbined, max_generation, expansion_row)


def _downstream_id(
    row: pd.Series,
    newave_code: int,
    fict_cascade: dict[int, FictCascadeResolution],
    id_map: NewaveIdMap,
) -> int | None:
    """Downstream cascade linkage.  Use the FICT-cascade resolver so that
    plants whose physical downstream is a fictitious plant (or whose
    confhd jusante is 0 but a name-matched FICT.<NAME> exists) end up
    wired to the next real plant in the cascade, not silently
    disconnected.  Fall back to the raw confhd link for plants the
    resolver did not classify (defensive).
    """
    resolution = fict_cascade.get(newave_code)
    if resolution is not None and resolution.downstream_code is not None:
        try:
            return id_map.hydro_id(resolution.downstream_code)
        except KeyError:
            return None
    if resolution is None:
        jusante_raw = row.get("codigo_usina_jusante")
        if jusante_raw is not None and not is_na(jusante_raw) and int(jusante_raw) != 0:
            try:
                return id_map.hydro_id(int(jusante_raw))
            except KeyError:
                pass
    return None


def _evaporation(hreg: pd.Series, reference_volumes: list[float] | None) -> dict | None:
    evap_coeffs = [float(hreg[f"evaporacao_{m}"]) for m in _EVAP_MONTHS]
    if not any(v != 0.0 for v in evap_coeffs):
        return None
    evaporation: dict = {"coefficients_mm": evap_coeffs}
    if reference_volumes is not None:
        evaporation["reference_volumes_hm3"] = reference_volumes
    return evaporation


def _hydraulic_losses(hreg: pd.Series) -> dict | None:
    """``tipo_perda`` 1 = a % of gross head (factor), 2 = constant metres;
    ``None`` for any other type or no loss."""
    tipo_perda = int(hreg.get("tipo_perda", 0) or 0)
    perdas_val = float(hreg.get("perdas", 0.0) or 0.0)
    if tipo_perda == 1 and perdas_val > 0 and not math.isnan(perdas_val):
        return {"type": "factor", "value": perdas_val / 100.0}
    if tipo_perda == 2 and perdas_val > 0 and not math.isnan(perdas_val):
        return {"type": "constant", "value_m": perdas_val}
    return None


def _specific_productivity(hreg: pd.Series) -> float | None:
    """Specific productivity ρ_esp [MW / ((m³/s)·m)] — feeds novomodelo's energy
    conversion pipeline (derives ρ_eq from VHA geometry)."""
    rho_esp_raw = hreg.get("produtibilidade_especifica")
    if rho_esp_raw is None:
        return None
    rho_esp = float(rho_esp_raw)
    if not math.isnan(rho_esp) and rho_esp > 0.0:
        return rho_esp
    return None


def _tailrace(hreg: pd.Series) -> dict | None:
    """Tailrace as a zero-order polynomial = canal_fuga_medio (constant). Novomodelo
    subtracts the tailrace level from the upstream head when deriving ρ_eq;
    without this source-model's productivity will not match."""
    cf_raw = hreg.get("canal_fuga_medio")
    if cf_raw is None:
        return None
    cf_val = float(cf_raw)
    if not math.isnan(cf_val) and cf_val > 0.0:
        return {"type": "polynomial", "coefficients": [cf_val]}
    return None


class _PlantEntry(NamedTuple):
    """A plant's ``hydros.json`` entry and its filling/expansion report rows."""

    entry: dict
    filling_row: list[object] | None
    expansion_row: list[object] | None


def _hydro_entry(
    row: pd.Series, context: _HydroContext, id_map: NewaveIdMap
) -> _PlantEntry:
    newave_code = int(row["codigo_usina"])
    name = str(row["nome_usina"]).strip()

    if newave_code not in context.cadastro.index:
        raise ValueError(
            f"Hydro plant '{name}' (code {newave_code}) from confhd.dat not found in hidr.dat"
        )

    hreg = context.cadastro.loc[newave_code]

    # Reservoir bounds. 'D'/'S' plants collapse to one volume: the source
    # model spills their inflow excess, and a usable range would let novomodelo
    # store it in a phantom buffer and shift it across stages.
    vol_min, vol_max = operating_storage_range(hreg)

    filling = _filling_phase(newave_code, name, vol_min, context)

    novomodelo_hydro_id = id_map.hydro_id(newave_code)
    caps = _generation_caps(newave_code, name, hreg, novomodelo_hydro_id, context)

    # Minimum outflow from historical minimum (may have been overridden by MODIF).
    vazao_min_hist = hreg.get("vazao_minima_historica")
    vazao_min_hist_val = float(vazao_min_hist) if vazao_min_hist else 0.0
    min_outflow = (
        vazao_min_hist_val if context.min_outflow_on and vazao_min_hist_val > 0 else 0.0
    )

    # VAZMINT overrides are per-stage bounds in hydro_bounds.parquet
    # (convert_storage_bounds).  The static min_outflow in hydros.json keeps the
    # hidr.dat / VAZMIN base value, which stages before the first VAZMINT record
    # use.  CFUGA/CMONT overrides belong to convert_production_models.

    # Static field is always zero — per-stage GHMIN values are
    # emitted in ``hydro_bounds.parquet:min_generation_mw`` by
    # :func:`convert_storage_bounds` so the LP sees the correct
    # bound at every stage.
    min_generation = 0.0

    downstream_id = _downstream_id(row, newave_code, context.fict_cascade, id_map)

    ree_code = int(row["ree"])
    subsystem_code = context.ree_to_submercado.get(ree_code)
    if subsystem_code is None:
        raise ValueError(
            f"Hydro plant '{name}' (code {newave_code}) has REE {ree_code}"
            f" which is not present in ree.dat"
        )
    bus_id = id_map.bus_id(subsystem_code)

    evaporation = _evaporation(hreg, context.evaporation_references.get(newave_code))
    hydraulic_losses = _hydraulic_losses(hreg)

    # Per-plant penalty overrides removed: newave converts PENALID R$/MWh
    # to rate units once using a system-average productivity and applies the
    # same value to all plants. The global defaults in penalties.json
    # (set by convert_penalties) already carry the converted values.
    penalties: dict | None = None

    rho_esp = _specific_productivity(hreg)

    # FPHA reservoirs: novomodelo fits the production function from geometry +
    # tailrace, so hand it the dimensionless turbine efficiency
    # (eta = rho_esp / K) and select the "fpha" generation model. Eligibility
    # guarantees rho_esp is present. Non-eligible plants (run-of-river, or
    # non-FPHA cases) keep the constant-productivity path with no efficiency.
    is_fpha = newave_code in context.fpha_codes
    efficiency: dict | None = None
    if is_fpha and rho_esp is not None:
        efficiency = {
            "type": "constant",
            "value": fpha_efficiency(rho_esp, name),
        }

    tailrace = _tailrace(hreg)

    entry: dict = {
        "id": novomodelo_hydro_id,
        "name": name,
        "operational_start_date": filling.operational_start_date,
        "downstream_id": downstream_id,
        "reservoir": {
            "min_storage_hm3": vol_min,
            "max_storage_hm3": vol_max,
        },
        "outflow": {
            "min_outflow_m3s": min_outflow,
            "max_outflow_m3s": None,
        },
        "generation": {
            "model": "fpha" if is_fpha else "constant_productivity",
            "min_turbined_m3s": 0.0,
            "max_turbined_m3s": caps.max_turbined,
            "min_generation_mw": min_generation,
            "max_generation_mw": caps.max_generation,
        },
        "unit_groups": [
            build_mirror_unit_group(
                name=name,
                bus_id=bus_id,
                min_generation_mw=min_generation,
                max_generation_mw=caps.max_generation,
                min_turbined_m3s=0.0,
                max_turbined_m3s=caps.max_turbined,
            )
        ],
        "specific_productivity_mw_per_m3s_per_m": rho_esp,
        "evaporation": evaporation,
        "tailrace": tailrace,
        "diversion": context.diversion_channels.get(newave_code),
        "filling": filling.filling,
        "efficiency": efficiency,
        "hydraulic_losses": hydraulic_losses,
        "penalties": penalties,
        "entry_stage_id": filling.entry_stage_id,
        "exit_stage_id": None,
    }
    return _PlantEntry(entry, filling.report_row, caps.expansion_row)


def _report_filling_plants(rows: list[list[object]]) -> None:
    if not rows:
        return
    emit(
        Diagnostic(
            code="ne-filling-plant",
            severity=Severity.INFO,
            category="Filling plants",
            title=f"Dead-volume filling plants admitted ({len(rows)})",
            summary=(
                f"{len(rows)} NE plant(s) admitted with a "
                "dead-volume filling schedule; each fills before operating "
                "and ramps capacity as units come online."
            ),
            table=DiagnosticTable(
                columns=[
                    "Plant",
                    "Code",
                    "Fill start",
                    "Operates from",
                    "Vol. morto %",
                    "Unit ramp",
                ],
                rows=rows,
                justify=["left", "right", "right", "right", "right", "left"],
                caption=("Stage ids are 0-based; capacity is 0 until 'Operates from'."),
            ),
        ),
        logger=_LOG,
    )


def _report_expansion_plants(rows: list[list[object]]) -> None:
    if not rows:
        return
    emit(
        Diagnostic(
            code="ee-expansion-ramped",
            severity=Severity.INFO,
            category="Expansion plants",
            title=f"Expansion plants ramped ({len(rows)})",
            summary=(
                f"{len(rows)} plant(s) marked "
                '"existente em expansão" operate from the first stage at the '
                "configuration modif.dat declares for the study start and "
                "reach the hidr.dat configuration as the exph.dat machines "
                "enter; the capacity before each entry is written as a "
                "per-stage bound."
            ),
            table=DiagnosticTable(
                columns=[
                    "Plant",
                    "Code",
                    "MW at start",
                    "MW when full",
                    "Machines entering",
                    "First entry stage",
                ],
                rows=rows,
                justify=["left", "right", "right", "right", "right", "right"],
                caption="Stage ids are 0-based.",
            ),
        ),
        logger=_LOG,
    )


def convert_hydros(case: NewaveCase, id_map: NewaveIdMap) -> dict:
    """Convert the source model hydro plant data to a Novomodelo ``hydros.json`` dict.

    Reads ``hidr.dat``, ``confhd.dat``, and ``ree.dat`` from *case*.
    Returns a dict with a ``"hydros"`` key containing a list of hydro
    entries sorted by Novomodelo 0-based ID.

    Also reads ``MODIF.DAT`` (if present) to apply permanent parameter
    overrides and extract temporal override metadata.  Reads ``GHMIN.DAT``
    (if present) to override computed minimum generation values.  Reads
    ``PENALID.DAT`` (if present) to populate per-plant penalty overrides.

    Parameters
    ----------
    case:
        Parsed the source model case.
    id_map:
        Pre-built ID mapping used for bus and downstream-hydro cross-
        references.

    Raises
    ------
    ValueError
        If a hydro in ``confhd.dat`` references a code not found in
        ``hidr.dat``.
    """
    context = _hydro_context(case, id_map)
    hydros: list[dict] = []
    filling_rows: list[list[object]] = []
    expansion_rows: list[list[object]] = []
    for _, row in context.active_hydros.iterrows():
        plant = _hydro_entry(row, context, id_map)
        hydros.append(plant.entry)
        if plant.filling_row is not None:
            filling_rows.append(plant.filling_row)
        if plant.expansion_row is not None:
            expansion_rows.append(plant.expansion_row)

    hydros.sort(key=lambda h: h["id"])

    # One INFO Diagnostic per plant kind (filling, expanding), each emitted only
    # when such a plant was seen. The de-dup in ``finalize_diagnostics`` keys on
    # ``(code, summary)``, so the per-plant detail rides on the table, not on N
    # separate diagnostics.
    _report_filling_plants(filling_rows)
    _report_expansion_plants(expansion_rows)

    return {
        "$schema": novomodelo_schemas.schema_url_for("system/hydros.json"),
        "hydros": hydros,
    }
