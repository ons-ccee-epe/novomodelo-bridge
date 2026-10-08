"""Hydro entity-record construction: the ``hydros.json`` per-plant builder
``convert_hydros``, and its local leaf helpers (mirror unit groups, exph-derived
unit-ramp summaries) with no forward dependency on another seam.
"""

from __future__ import annotations

import logging
import math

import pandas as pd

from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    emit,
)
from novomodelo_bridge.core.hydro_units import build_mirror_unit_group, rated_capacity
from novomodelo_bridge.core.pandas_utils import is_na
from novomodelo_bridge.core.productivity import fpha_efficiency
from novomodelo_bridge.newave.case import NewaveCase
from novomodelo_bridge.newave.converters.hydro.bounds import (
    _compute_max_turbined_head_corrected,
    _per_stage_turbined_envelope,
)
from novomodelo_bridge.newave.converters.hydro.geometry import (
    _EVAP_MONTHS,
    _expansion_configs,
    _read_volref_saz,
    fpha_eligible_codes,
)
from novomodelo_bridge.newave.converters.hydro.overrides import (
    _apply_permanent_overrides,
)
from novomodelo_bridge.newave.filling import (
    exph_unit_rows,
    filling_completion_date,
    filling_min_rate_m3s,
    filling_schedule,
)
from novomodelo_bridge.newave.horizon import build_stage_dates, historical_start_date
from novomodelo_bridge.newave.id_map import NewaveIdMap
from novomodelo_bridge.newave.plants import filling_hydro_codes
from novomodelo_bridge.newave.switches import switch_off_diagnostic
from novomodelo_bridge.novomodelo import schemas as novomodelo_schemas

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
    hidr = case.hidr
    confhd = case.confhd
    ree_file = case.ree

    cadastro = hidr.cadastro  # DataFrame indexed by codigo_usina (1-based)
    confhd_df = confhd.usinas
    ree_df = ree_file.rees  # columns: codigo, nome, submercado, ...

    cadastro = _apply_permanent_overrides(cadastro, case)

    min_outflow_on = case.switches.min_outflow.on
    if not min_outflow_on:
        emit(switch_off_diagnostic(case.switches.min_outflow), logger=_LOG)

    # Seasonal reference volumes per plant — when present, fed back into the evaporation
    # block as ``reference_volumes_hm3`` so novomodelo's evaporation linearization matches
    # the per-month reference the source model itself uses.
    seasonal_volref = _read_volref_saz(case)

    # NE plants carrying an exph dead-volume filling row are admitted as real
    # downstream nodes (admission predicate, single source of truth). The
    # set is computed once here and threaded into the cascade walker so upstream
    # plants resolve to the filling plant; it also drives each filling plant's
    # FILLING contract below. Empty (byte-identical to the no-arg call) when the
    # case has no exph or no NE-with-filling plant.
    filling_codes = filling_hydro_codes(
        confhd_df, case.exph.expansoes if case.exph is not None else None
    )

    # Resolve the FICT-cascade for every real plant.  Provides the effective
    # next-real-plant downstream and the sum of any FICT-chain ρ_eq that must
    # be folded back into the upstream real plant's effective ρ_eq.  See
    # ``novomodelo_bridge.newave.converters.fict_cascade`` for the resolution rules.
    from novomodelo_bridge.newave.converters.fict_cascade import resolve_cascade

    fict_cascade = resolve_cascade(confhd_df, cadastro, filling_codes=filling_codes)

    # Per-hydro envelope over the head-corrected per-stage turbined caps that
    # convert_turbined_bounds_head_corrected emits into hydro_bounds.parquet.
    # novomodelo rule 43 forbids any hydro_bounds row from raising max_turbined_m3s
    # above the plant's own declared value, so the reference-head value below
    # must be raised to cover every emitted per-stage row. Empty
    # for a hydro with no per-stage head variation, which keeps its declared
    # value unchanged.
    turbined_envelope = _per_stage_turbined_envelope(case, id_map)

    expansion_configs = _expansion_configs(case, cadastro)

    existing = case.active_hydros

    # Per-stage (year, month) closure over the study horizon, built once and
    # passed into ``filling_min_rate_m3s`` for each filling plant.
    # The exph DataFrame is only consulted when ``filling_codes`` is non-empty,
    # which guarantees ``case.exph`` is not None there.
    horizon = case.horizon
    stage_dates = build_stage_dates(
        horizon.start_year, horizon.start_month, horizon.total_stages
    )

    def stage_year_month(t: int) -> tuple[int, int]:
        d = stage_dates[t]
        return d.year, d.month

    ree_to_submercado: dict[int, int] = {}
    if ree_df is not None:
        for _, row in ree_df.iterrows():
            ree_to_submercado[int(row["codigo"])] = int(row["submercado"])

    # Plants emitted as FPHA (empty unless dger funcao_producao_uhe == 0).
    fpha_codes = fpha_eligible_codes(case)

    # Default operational_start_date for in-service plants: in service since the
    # historical record. Filling (NE) plants override it below with the calendar
    # month they finish filling and enter operation.
    existing_op_date = historical_start_date(case.dger)

    hydros: list[dict] = []
    # One diagnostic row per admitted filling plant, accumulated in the loop and
    # emitted as a single INFO Diagnostic after it (empty, emitting nothing, when
    # no filling plant is admitted).
    filling_diag_rows: list[list[object]] = []
    expansion_diag_rows: list[list[object]] = []
    for _, row in existing.iterrows():
        newave_code = int(row["codigo_usina"])
        name = str(row["nome_usina"]).strip()

        if newave_code not in cadastro.index:
            raise ValueError(
                f"Hydro plant '{name}' (code {newave_code}) from confhd.dat"
                f" not found in hidr.dat"
            )

        hreg = cadastro.loc[newave_code]

        # Reservoir bounds.
        vol_min = float(hreg["volume_minimo"])
        vol_max = float(hreg["volume_maximo"])

        # The source model treats Daily-regulation ('D') and run-of-river / fio-d'água
        # ('S') plants as fio-d'água — they can't accumulate water across stages, so the
        # useful volume is NOT a usable reservoir buffer. Collapse the active range to a
        # single point so Novomodelo's LP mirrors that; otherwise Novomodelo stores the inflow
        # excess in a phantom buffer and shifts it across stages, where the source
        # model simply spills it.
        #
        # Both 'D' and 'S' freeze at ``volume_referencia`` — the operating volume the
        # source model holds these plants at every stage. When
        # ``volume_referencia`` is absent/NaN the range is left as
        # ``[volume_minimo, volume_maximo]`` (no collapse).
        tipo_reg = str(hreg.get("tipo_regulacao", "")).strip()
        if tipo_reg in ("D", "S"):
            vol_ref_raw = hreg.get("volume_referencia")
            if vol_ref_raw is not None and not pd.isna(vol_ref_raw):
                vol_ref = float(vol_ref_raw)
                vol_min = vol_ref
                vol_max = vol_ref

        # FILLING phase: an admitted NE plant fills its dead
        # volume from its seeded storage up to ``min_storage_hm3`` over the
        # half-open window ``[start_sid, entry_sid)``, then enters operation at
        # ``entry_sid``. EX plants never enter this branch (filling_codes is the
        # NE-with-exph set), so their entry/exit/filling stay None.
        entry_stage_id: int | None = None
        filling: dict | None = None
        operational_start_date = existing_op_date
        if newave_code in filling_codes:
            # filling_codes is non-empty here ⇒ case.exph is not None (the
            # admission predicate required exph), so .expansoes is safe.
            exph_df = case.exph.expansoes
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
            entry_stage_id = entry_sid
            # Operational start = the month filling completes and the plant enters
            # service, computed from the raw exph schedule so it is truthful even
            # when completion falls outside the study horizon.
            operational_start_date = filling_completion_date(
                ts.year, ts.month, duracao
            ).isoformat()
            # One INFO-diagnostic row per filling plant (rendered after the loop):
            # its filling window, seeded dead volume, and unit-ramp summary. Built
            # for every filling plant, including the ``duracao == 0`` (no-filling-
            # block) case, so the user always sees the admitted plant.
            filling_diag_rows.append(
                [
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
            )
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
                # stays on the hydro record below. If ``start_sid >= total_stages``
                # too, ``rate_entry`` collapses to ``start_sid``, giving an empty
                # window → ``filling_min_rate_m3s`` returns 0.0 (no crash).
                rate_entry = min(entry_sid, horizon.total_stages)
                rate = filling_min_rate_m3s(
                    vol_min,
                    volume_morto,
                    start_sid,
                    rate_entry,
                    stage_year_month,
                )
                filling = {
                    "start_stage_id": start_sid,
                    "filling_min_rate_m3s": rate,
                }
            # entry_sid == start_sid (duracao 0): keep filling None but still set
            # entry_stage_id — novomodelo rejects start_stage_id >= entry_stage_id, so
            # no degenerate filling block is emitted.

        # Generation parameters. Productivity lives in
        # ``hydro_production_models.json`` on novomodelo HEAD; callers that need
        # the per-hydro base value call ``compute_base_productivities``.
        is_fpha = newave_code in fpha_codes
        # Turbined and generation caps are independent of the production
        # function:
        #   * turbined cap = head-corrected engolimento at the operating head
        #     (m³/s) — the operational dispatch cap; rated Σ n·q_nom overshoots it.
        #   * generation cap = rated installed power Σ n·p_nom = the FPHA GHmax.
        #     This loose ceiling never binds before the head-corrected turbined
        #     cap, so a plant reaches its turbined limit, not nameplate power.
        #
        # The declaration itself is the ENVELOPE over that reference-head value
        # and every per-stage head-corrected cap this hydro emits into
        # hydro_bounds: a plant with per-stage head variation
        # (MODIF.DAT CFUGA/CMONT or seasonal VOLREF_SAZ) can turbine more at a
        # higher-head stage than at the reference head, and novomodelo rule 43
        # forbids a hydro_bounds row from exceeding the plant's own declared
        # value. A hydro with no per-stage variation is absent from
        # ``turbined_envelope`` and keeps its reference-head value unchanged.
        novomodelo_hydro_id = id_map.hydro_id(newave_code)
        config = expansion_configs.get(newave_code)
        hreg_declared = hreg if config is None else config.declared_hreg(hreg)
        max_turbined_reference = _compute_max_turbined_head_corrected(
            hreg_declared, name
        )[0]
        envelope_value = turbined_envelope.get(novomodelo_hydro_id)
        max_turbined = (
            max_turbined_reference
            if envelope_value is None
            else max(max_turbined_reference, envelope_value)
        )
        max_generation = rated_capacity(hreg_declared)[1]

        if config is not None:
            start_generation = rated_capacity(config.hreg_at(hreg, 0))[1]
            expansion_diag_rows.append(
                [
                    name,
                    newave_code,
                    f"{start_generation:.1f}",
                    f"{max_generation:.1f}",
                    len(config.unit_rows),
                    min(stage for _conjunto, stage in config.unit_rows),
                ]
            )

        # Minimum outflow from historical minimum (may have been overridden by MODIF).
        vazao_min_hist = hreg.get("vazao_minima_historica")
        vazao_min_hist_val = float(vazao_min_hist) if vazao_min_hist else 0.0
        min_outflow = (
            vazao_min_hist_val if min_outflow_on and vazao_min_hist_val > 0 else 0.0
        )

        # VAZMINT temporal overrides are now emitted as per-stage bounds in
        # hydro_bounds.parquet via convert_storage_bounds().  The static
        # min_outflow in hydros.json keeps the hidr.dat / VAZMIN base value
        # so that stages before the first VAZMINT record use the correct default.

        # CFUGA/CMONT temporal overrides are handled by convert_production_models;
        # no warning needed here as per-stage productivity is now supported.

        # Static field is always zero — per-stage GHMIN values are
        # emitted in ``hydro_bounds.parquet:min_generation_mw`` by
        # :func:`convert_storage_bounds` so the LP sees the correct
        # bound at every stage.
        min_generation = 0.0

        # Downstream cascade linkage.  Use the FICT-cascade resolver so that
        # plants whose physical downstream is a fictitious plant (or whose
        # confhd jusante is 0 but a name-matched FICT.<NAME> exists) end up
        # wired to the next real plant in the cascade, not silently
        # disconnected.  Fall back to the raw confhd link for plants the
        # resolver did not classify (defensive).
        downstream_id: int | None = None
        resolution = fict_cascade.get(newave_code)
        if resolution is not None and resolution.downstream_code is not None:
            try:
                downstream_id = id_map.hydro_id(resolution.downstream_code)
            except KeyError:
                downstream_id = None
        elif resolution is None:
            jusante_raw = row.get("codigo_usina_jusante")
            if (
                jusante_raw is not None
                and not is_na(jusante_raw)
                and int(jusante_raw) != 0
            ):
                try:
                    downstream_id = id_map.hydro_id(int(jusante_raw))
                except KeyError:
                    pass

        ree_code = int(row["ree"])
        subsystem_code = ree_to_submercado.get(ree_code)
        if subsystem_code is None:
            raise ValueError(
                f"Hydro plant '{name}' (code {newave_code}) has REE {ree_code}"
                f" which is not present in ree.dat"
            )
        bus_id = id_map.bus_id(subsystem_code)

        evap_coeffs = [float(hreg[f"evaporacao_{m}"]) for m in _EVAP_MONTHS]
        has_evaporation = any(v != 0.0 for v in evap_coeffs)

        # Evaporation linearization points: when the plant has a seasonal row in
        # volref_saz.dat, emit one absolute hm³ value per calendar month (vmin +
        # useful_volume).  Missing months default to vmin (matching the source model's
        # "operate at vmin" semantics for zero entries). Clamped into [min_storage_hm3,
        # max_storage_hm3] so novomodelo's dimensional validator accepts every value even if
        # a permanent VOLMIN override raised vmin above what the file was written for.
        plant_seasonal_for_evap = seasonal_volref.get(newave_code)
        evap_reference_volumes: list[float] | None = None
        if has_evaporation and plant_seasonal_for_evap:
            evap_reference_volumes = [
                max(
                    vol_min,
                    min(
                        vol_max,
                        vol_min + plant_seasonal_for_evap.get(m, 0.0),
                    ),
                )
                for m in range(1, 13)
            ]

        tipo_perda = int(hreg.get("tipo_perda", 0) or 0)
        perdas_val = float(hreg.get("perdas", 0.0) or 0.0)
        if tipo_perda == 1 and perdas_val > 0 and not math.isnan(perdas_val):
            hydraulic_losses: dict | None = {
                "type": "factor",
                "value": perdas_val / 100.0,
            }
        elif tipo_perda == 2 and perdas_val > 0 and not math.isnan(perdas_val):
            hydraulic_losses = {"type": "constant", "value_m": perdas_val}
        else:
            hydraulic_losses = None

        # Per-plant penalty overrides removed: newave converts PENALID R$/MWh
        # to rate units once using a system-average productivity and applies the
        # same value to all plants. The global defaults in penalties.json
        # (set by convert_penalties) already carry the converted values.
        penalties: dict | None = None

        # Specific productivity ρ_esp [MW / ((m³/s)·m)] — feeds novomodelo's energy
        # conversion pipeline (derives ρ_eq from VHA geometry).
        rho_esp_raw = hreg.get("produtibilidade_especifica")
        rho_esp: float | None = None
        if rho_esp_raw is not None:
            rho_esp_f = float(rho_esp_raw)
            if not math.isnan(rho_esp_f) and rho_esp_f > 0.0:
                rho_esp = rho_esp_f

        # FPHA reservoirs: novomodelo fits the production function from geometry +
        # tailrace, so hand it the dimensionless turbine efficiency
        # (eta = rho_esp / K) and select the "fpha" generation model. Eligibility
        # guarantees rho_esp is present. Non-eligible plants (run-of-river, or
        # non-FPHA cases) keep the constant-productivity path with no efficiency.
        # ``is_fpha`` was computed above for the capacity-cap branch.
        efficiency: dict | None = None
        if is_fpha and rho_esp is not None:
            efficiency = {
                "type": "constant",
                "value": fpha_efficiency(rho_esp, name),
            }

        # Tailrace as a zero-order polynomial = canal_fuga_medio (constant). Novomodelo
        # subtracts the tailrace level from the upstream head when deriving ρ_eq;
        # without this source-model's productivity will not match.
        cf_raw = hreg.get("canal_fuga_medio")
        tailrace: dict | None = None
        if cf_raw is not None:
            cf_val = float(cf_raw)
            if not math.isnan(cf_val) and cf_val > 0.0:
                tailrace = {"type": "polynomial", "coefficients": [cf_val]}

        evaporation: dict | None = None
        if has_evaporation:
            evaporation = {"coefficients_mm": evap_coeffs}
            if evap_reference_volumes is not None:
                evaporation["reference_volumes_hm3"] = evap_reference_volumes

        hydro_entry: dict = {
            "id": novomodelo_hydro_id,
            "name": name,
            "operational_start_date": operational_start_date,
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
                "max_turbined_m3s": max_turbined,
                "min_generation_mw": min_generation,
                "max_generation_mw": max_generation,
            },
            "unit_groups": [
                build_mirror_unit_group(
                    name=name,
                    bus_id=bus_id,
                    min_generation_mw=min_generation,
                    max_generation_mw=max_generation,
                    min_turbined_m3s=0.0,
                    max_turbined_m3s=max_turbined,
                )
            ],
            "specific_productivity_mw_per_m3s_per_m": rho_esp,
            "evaporation": evaporation,
            "tailrace": tailrace,
            "diversion": None,
            "filling": filling,
            "efficiency": efficiency,
            "hydraulic_losses": hydraulic_losses,
            "penalties": penalties,
            "entry_stage_id": entry_stage_id,
            "exit_stage_id": None,
        }
        hydros.append(hydro_entry)

    hydros.sort(key=lambda h: h["id"])

    # One INFO Diagnostic per plant kind (filling, expanding), each emitted only
    # when such a plant was seen. The de-dup in ``finalize_diagnostics`` keys on
    # ``(code, summary)``, so the per-plant detail rides on the table, not on N
    # separate diagnostics.
    if filling_diag_rows:
        emit(
            Diagnostic(
                code="ne-filling-plant",
                severity=Severity.INFO,
                category="Filling plants",
                title=(
                    f"Dead-volume filling plants admitted ({len(filling_diag_rows)})"
                ),
                summary=(
                    f"{len(filling_diag_rows)} NE plant(s) admitted with a "
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
                    rows=filling_diag_rows,
                    justify=["left", "right", "right", "right", "right", "left"],
                    caption=(
                        "Stage ids are 0-based; capacity is 0 until 'Operates from'."
                    ),
                ),
            ),
            logger=_LOG,
        )

    if expansion_diag_rows:
        emit(
            Diagnostic(
                code="ee-expansion-ramped",
                severity=Severity.INFO,
                category="Expansion plants",
                title=f"Expansion plants ramped ({len(expansion_diag_rows)})",
                summary=(
                    f"{len(expansion_diag_rows)} plant(s) marked "
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
                    rows=expansion_diag_rows,
                    justify=["left", "right", "right", "right", "right", "right"],
                    caption="Stage ids are 0-based.",
                ),
            ),
            logger=_LOG,
        )

    return {
        "$schema": novomodelo_schemas.schema_url_for("system/hydros.json"),
        "hydros": hydros,
    }
