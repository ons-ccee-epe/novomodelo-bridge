"""Hydro capacity caps and the bounds converters they feed: a bound family
(turbined flow, generation, storage, outflow) emitted against a per-plant or
per-(hydro, stage) ceiling into ``hydros.json`` and ``hydro_bounds.parquet``.

Depends on ``.overrides``, ``.geometry``, and ``.productivity``; imports
nothing from ``.entity`` — kept acyclic so ``entity.convert_hydros`` can
import from here, never the reverse.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

import pandas as pd
import pyarrow as pa

from novomodelo_bridge.core.diagnostics import Diagnostic, Severity, emit
from novomodelo_bridge.core.pandas_utils import is_na
from novomodelo_bridge.core.productivity import (
    KTURB_BY_TIPO_TURBINA,
    apply_hydraulic_loss,
    evaluate_cota,
    mean_cota,
)
from novomodelo_bridge.core.tolerances import BIG_M
from novomodelo_bridge.core.turbine import MachineSet, power_limited_flow
from novomodelo_bridge.newave.case import NewaveCase
from novomodelo_bridge.newave.converters.hydro.geometry import _expansion_configs
from novomodelo_bridge.newave.converters.hydro.overrides import (
    _apply_permanent_overrides,
    _extract_temporal_overrides,
    _per_stage_drop_overrides,
    _read_ghmin_per_stage,
    percent_of_useful_volume,
    read_cadastro,
)
from novomodelo_bridge.newave.converters.hydro.productivity import _total_study_stages
from novomodelo_bridge.newave.filling import (
    exph_unit_rows,
    filling_schedule,
    hreg_with_machines,
    online_machines,
)
from novomodelo_bridge.newave.horizon import seasonal_step_function
from novomodelo_bridge.newave.id_map import NewaveIdMap
from novomodelo_bridge.newave.plants import filling_hydro_codes
from novomodelo_bridge.newave.switches import DgerSwitches, switch_off_diagnostic

_LOG = logging.getLogger(__name__)


def _clamp_outage_pct(value: float, label: str, plant_name: str) -> float:
    """Clamp TEIF/IP percentages into ``[0, 100]``, reporting an overshoot.

    :func:`_operational_caps` runs once per plant from ``convert_hydros`` and
    once per configuration from :func:`convert_per_stage_capacity_caps`, so the
    same plant+field can be clamped more than once in one run. Emitting per
    call and relying on ``finalize_diagnostics``' ``(code, summary)`` de-dup
    collapses that to a single diagnostic instead of threading an accumulator
    through every call site.
    """
    if math.isnan(value) or value < 0.0:
        return 0.0
    if value > 100.0:
        emit(
            Diagnostic(
                code="hydro-outage-rate-clamped",
                severity=Severity.WARNING,
                category="Hydro bounds",
                title="Outage rate exceeds 100%",
                summary=(
                    f"{label} exceeds 100% for plant {plant_name} "
                    f"({label.lower()}={value:.2f}); clamping to 100."
                ),
            ),
            logger=_LOG,
        )
        return 100.0
    return value


def _availability(hreg: pd.Series, name: str) -> float:
    """Joint TEIF/IP availability factor — both apply multiplicatively."""
    teif = _clamp_outage_pct(float(hreg.get("teif", 0.0) or 0.0), "teif", name)
    ip = _clamp_outage_pct(float(hreg.get("ip", 0.0) or 0.0), "ip", name)
    return ((100.0 - teif) / 100.0) * ((100.0 - ip) / 100.0)


def _operating_head(
    hreg: pd.Series,
    *,
    canal_fuga_override: float | None,
    cmont_override: float | None,
) -> float | None:
    """The net head (m) the source model rates the machines of *hreg* at, or
    ``None`` when the cota polynomial or the tailrace is missing.

    The forebay level is ``cmont_override`` when given, else the mean cota over
    ``[V_min, V_65]`` for a reservoir plant (``tipo_regulacao == 'M'``), else
    the cota at ``volume_referencia`` (the mean over ``[V_min, V_max]`` without
    one). The tailrace is ``canal_fuga_override`` when given, else
    ``canal_fuga_medio``. A seasonal reference volume never enters it.
    """
    if not all(f"a{i}_volume_cota" in hreg.index for i in range(5)):
        return None
    tailrace = (
        hreg.get("canal_fuga_medio")
        if canal_fuga_override is None
        else canal_fuga_override
    )
    if tailrace is None or is_na(tailrace):
        return None
    vol_min = float(hreg["volume_minimo"])
    vol_max = float(hreg["volume_maximo"])
    coeffs = [float(hreg[f"a{i}_volume_cota"]) for i in range(5)]
    vol_ref = hreg.get("volume_referencia")
    if cmont_override is not None:
        forebay = cmont_override
    elif str(hreg.get("tipo_regulacao", "")).strip() == "M":
        forebay = mean_cota(coeffs, vol_min, vol_min + 0.65 * (vol_max - vol_min))
    elif vol_ref is None or is_na(vol_ref) or float(vol_ref) <= 0.0:
        forebay = mean_cota(coeffs, vol_min, vol_max)
    else:
        forebay = evaluate_cota(coeffs, float(vol_ref))
    return apply_hydraulic_loss(
        forebay - float(tailrace),
        int(hreg.get("tipo_perda", 0) or 0),
        float(hreg.get("perdas", 0.0) or 0.0),
    )


def _operational_caps(
    hreg: pd.Series,
    name: str,
    *,
    canal_fuga_override: float | None = None,
    cmont_override: float | None = None,
) -> tuple[float, float]:
    """Return the ``(max_turbined, max_generation)`` the source model dispatches
    the machines of *hreg* against, under either production function.

    The generation cap is the available power ``Σ_c n_c · p_nom_c ·
    availability``, the TEIF/IP availability of :func:`_availability`. The
    turbined cap is the engolimento, the
    :func:`~novomodelo_bridge.core.turbine.power_limited_flow` with both heads at
    the :func:`_operating_head` ``h_op``: for each machine set *c* with ``n_c``
    units of nominal head ``h_nom_c`` and nominal flow ``q_nom_c``, the
    affinity-law flow ``q_eff_c = n_c · q_nom_c · (h_op / h_nom_c)^k`` (``k`` =
    0.5 Francis/Pelton, 0.2 Kaplan), and::

        max_turbined = min(Σ_c q_eff_c, p_inst / (ρ_esp · h_op)) · availability

    The turbined cap falls back to the available flow ``Σ_c n_c · q_nom_c ·
    availability`` when the head data is missing (``queda_nominal_conjunto_*``
    columns, cota polynomial, tailrace, positive ``ρ_esp``) or ``h_op`` is not
    positive.
    """
    sets = [
        (
            int(hreg[f"maquinas_conjunto_{i}"]),
            float(hreg[f"vazao_nominal_conjunto_{i}"]),
            float(hreg[f"potencia_nominal_conjunto_{i}"]),
        )
        for i in range(1, int(hreg["numero_conjuntos_maquinas"]) + 1)
    ]
    availability = _availability(hreg, name)
    rated_flow = sum(n * q for n, q, _p in sets)
    rated_power = sum(n * p for n, _q, p in sets)
    available_power = rated_power * availability

    rho_esp = hreg.get("produtibilidade_especifica")
    has_head_data = all(
        f"queda_nominal_conjunto_{i}" in hreg.index for i in range(1, 6)
    )
    h_op = (
        _operating_head(
            hreg,
            canal_fuga_override=canal_fuga_override,
            cmont_override=cmont_override,
        )
        if has_head_data
        and sum(n for n, _q, _p in sets) > 0
        and rho_esp is not None
        and not is_na(rho_esp)
        and float(rho_esp) > 0.0
        else None
    )
    if h_op is None or h_op <= 0.0:
        return rated_flow * availability, available_power

    engolimento = power_limited_flow(
        h_op,
        [
            MachineSet(
                units=n,
                flow_m3s=q,
                head_m=float(hreg[f"queda_nominal_conjunto_{i}"]),
                power_mw=p,
            )
            for i, (n, q, p) in enumerate(sets, start=1)
        ],
        generator_head_m=h_op,
        specific_productivity=float(rho_esp),
        exponent=KTURB_BY_TIPO_TURBINA.get(int(hreg.get("tipo_turbina", 0) or 0), 0.5),
    )
    return engolimento * availability, available_power


def _reduced_caps(
    hreg: pd.Series, online: dict[int, int], name: str
) -> tuple[float, float]:
    """Return ``(max_turbined, max_generation)`` over the online machine subset.

    Computes the :func:`_operational_caps` of a filling plant at a ramp stage
    where only ``online`` of its machines are in service. Rather than threading
    an override count through the load-bearing shared helpers, it copies the
    cadastro ``hreg`` Series and rewrites each ``maquinas_conjunto_{c}`` (for
    ``c`` in ``1..numero_conjuntos_maquinas``) to the online count (``0`` for any
    group absent from the dict), then re-runs the exact full-capacity helpers — so
    the all-online stage equals the base cap by construction. With an empty
    ``online`` dict every count is ``0`` and both caps are ``0.0`` (an explicit
    zero-capacity stage, not a skip).
    """
    n_sets = int(hreg["numero_conjuntos_maquinas"])
    hreg_copy = hreg_with_machines(
        hreg, {c: online.get(c, 0) for c in range(1, n_sets + 1)}
    )
    return _operational_caps(hreg_copy, name)


def convert_per_stage_capacity_caps(
    case: NewaveCase, id_map: NewaveIdMap
) -> pa.Table | None:
    """Per-stage :func:`_operational_caps` the static ``hydros.json``
    declaration cannot carry.

    Every plant gets its available power as ``max_generation_mw`` at every
    stage, below the rated power ``hydros.json`` declares as the FPHA fit's
    ceiling. A plant whose MODIF.DAT CFUGA/CMONT records move its operating head
    gets ``max_turbined_m3s`` at every stage, with the stage's tailrace and
    upstream level; an expanding plant is rated with the machines in service at
    each stage and, without CFUGA/CMONT, gets ``max_turbined_m3s`` only before
    its last machine enters.

    Returns a ``(hydro_id, stage_id, max_turbined_m3s, max_generation_mw)``
    table, or ``None`` when the case has no plant or no stage.
    """
    cadastro = _apply_permanent_overrides(case.hidr.cadastro, case)
    confhd_codes = [int(r["codigo_usina"]) for _, r in case.active_hydros.iterrows()]
    temporal_overrides = _extract_temporal_overrides(case, confhd_codes)
    expansion = _expansion_configs(case, cadastro)
    total_stages = _total_study_stages(case)
    if total_stages <= 0:
        return None

    hydro_ids: list[int] = []
    stage_ids: list[int] = []
    max_turbined_vals: list[float | None] = []
    max_generation_vals: list[float] = []

    for newave_code in sorted(confhd_codes):
        if newave_code not in cadastro.index:
            continue
        try:
            hydro_id = id_map.hydro_id(newave_code)
        except KeyError:
            continue
        hreg = cadastro.loc[newave_code]
        name = str(hreg.get("nome_usina", newave_code))
        config = expansion.get(newave_code)
        ramp_end = 0 if config is None else config.full_online_stage
        drop_overrides = [
            o
            for o in temporal_overrides.get(newave_code, [])
            if o["type"] in ("CFUGA", "CMONT")
        ]
        drops = _per_stage_drop_overrides(drop_overrides, case, total_stages)
        caps: dict[tuple[int, float | None, float | None], tuple[float, float]] = {}
        for stage_id, (cfuga, cmont) in enumerate(drops):
            key = (min(stage_id, ramp_end), cfuga, cmont)
            if key not in caps:
                caps[key] = _operational_caps(
                    hreg if config is None else config.hreg_at(hreg, stage_id),
                    name,
                    canal_fuga_override=cfuga,
                    cmont_override=cmont,
                )
            max_turbined, max_generation = caps[key]
            hydro_ids.append(hydro_id)
            stage_ids.append(stage_id)
            max_turbined_vals.append(
                max_turbined if drop_overrides or stage_id < ramp_end else None
            )
            max_generation_vals.append(max_generation)

    if not hydro_ids:
        return None

    return pa.table(
        {
            "hydro_id": pa.array(hydro_ids, type=pa.int32()),
            "stage_id": pa.array(stage_ids, type=pa.int32()),
            "max_turbined_m3s": pa.array(max_turbined_vals, type=pa.float64()),
            "max_generation_mw": pa.array(max_generation_vals, type=pa.float64()),
        }
    )


def _per_stage_turbined_envelope(
    case: NewaveCase, id_map: NewaveIdMap
) -> dict[int, float]:
    """Return ``{novomodelo_hydro_id: max per-stage max_turbined_m3s}`` from the
    per-stage capacity table.

    Delegates to :func:`convert_per_stage_capacity_caps` — the exact
    function the pipeline uses to emit the per-(hydro, stage)
    ``hydro_bounds`` rows — instead of re-deriving the per-stage head. This is
    what lets :func:`convert_hydros` raise its declared
    ``generation.max_turbined_m3s`` to cover every emitted per-stage row
    (novomodelo rule 43) with zero risk of the two ever drifting apart: they are
    two views of the same table, not two formulas that happen to agree today.

    Returns an empty dict when no plant has CFUGA/CMONT temporal overrides or
    a machine ramp, so callers must treat a missing key as "no per-stage
    variation", not as zero.
    """
    table = convert_per_stage_capacity_caps(case, id_map)
    if table is None:
        return {}
    envelope: dict[int, float] = {}
    for hydro_id, value in zip(
        table["hydro_id"].to_pylist(),
        table["max_turbined_m3s"].to_pylist(),
        strict=True,
    ):
        if value is None:
            continue
        current = envelope.get(hydro_id)
        if current is None or value > current:
            envelope[hydro_id] = value
    return envelope


def convert_water_withdrawal(case: NewaveCase, id_map: NewaveIdMap) -> pa.Table | None:
    """Convert the source model water withdrawal data to a hydro_bounds Parquet table.

    Reads ``dsvagua.dat`` (optional) from *case* and produces a
    ``pa.Table`` with columns ``(hydro_id: INT32, stage_id: INT32,
    water_withdrawal_m3s: DOUBLE)`` suitable for writing to
    ``constraints/hydro_bounds.parquet``.

    The ``codigo_usina`` field in ``dsvagua.dat`` is a source-model 1-based *plant* code
    (matching ``confhd``), not a posto. Each plant may contribute multiple rows per
    stage (one per consumptive-use or remaining-flow component) which are summed before
    the sign is negated to convert the source model's "withdrawal = negative valor"
    convention into Novomodelo's positive ``water_withdrawal_m3s``.

    Parameters
    ----------
    case:
        Parsed the source model case.
    id_map:
        Pre-built entity ID map.

    Returns
    -------
    pa.Table | None
        Table with schema ``(hydro_id: INT32, stage_id: INT32,
        water_withdrawal_m3s: DOUBLE)`` sorted by ``(hydro_id, stage_id)``,
        or ``None`` when ``dsvagua.dat`` is absent, empty, or yields no
        valid rows after filtering.
    """
    if case.files.dsvagua is None:
        _LOG.debug("dsvagua.dat not found; no water withdrawal.")
        return None

    # Read dger upfront so the ``outros_usos_da_agua`` switch can short-circuit before
    # any dsvagua I/O. The source model treats 0 as "ignore dsvagua.dat" — mirror that
    # here so Novomodelo's hydro_bounds match The source model's actual run instead of the
    # file contents.
    dger = case.dger
    if int(getattr(dger, "outros_usos_da_agua", 1) or 0) == 0:
        _LOG.info("dger.outros_usos_da_agua == 0; skipping dsvagua.dat conversion.")
        return None

    dsvagua = case.dsvagua
    if dsvagua is None:
        return None
    df = dsvagua.desvios
    if df is None or df.empty:
        return None

    horizon = case.horizon
    start_year = horizon.start_year
    start_month = horizon.start_month
    num_study_stages = horizon.study_months
    num_post_study_stages = horizon.pos_months
    num_total_stages = horizon.total_stages

    # Build a cascade map so NC (Não Construída) plant dsvagua entries can be propagated
    # to the immediately downstream EX plant — the source model applies NC withdrawals
    # to the next real plant in the cascade, while FICT entries are not propagated and
    # are silently dropped.
    confhd_df = case.confhd.usinas
    fict_set = case.fictitious_codes
    plant_downstream: dict[int, int] = {}
    for _, r in confhd_df.iterrows():
        code = int(r["codigo_usina"])
        jus = r["codigo_usina_jusante"]
        plant_downstream[code] = int(jus) if jus is not None and not pd.isna(jus) else 0

    def _resolve_to_hydro_id(start_code: int) -> int | None:
        """Walk downstream from *start_code* until reaching a plant present
        in ``id_map``. Fictitious plants are never propagated. Returns the
        hydro_id of the first reachable EX plant, or ``None`` if the chain hits
        a terminus or a fictitious node first.
        """
        visited: set[int] = set()
        cur = start_code
        while cur and cur not in visited:
            visited.add(cur)
            if cur in fict_set:
                return None
            try:
                return id_map.hydro_id(cur)
            except KeyError:
                pass
            cur = plant_downstream.get(cur, 0)
        return None

    grouped = df.groupby(["codigo_usina", "data"], as_index=False)["valor"].sum()

    # Accumulate by (hydro_id, stage_id) so propagated NC entries merge
    # cleanly with the downstream plant's own dsvagua row.
    accumulated: dict[tuple[int, int], float] = {}

    for _, row in grouped.iterrows():
        hydro_code = int(row["codigo_usina"])
        hydro_id = _resolve_to_hydro_id(hydro_code)
        if hydro_id is None:
            continue

        dt = row["data"]
        stage_id = (dt.year - start_year) * 12 + (dt.month - start_month)
        if stage_id < 0 or stage_id >= num_study_stages:
            continue

        # Negate: The source model negative valor = withdrawal; Novomodelo positive =
        # withdrawal.
        withdrawal = -float(row["valor"])
        key = (hydro_id, stage_id)
        accumulated[key] = accumulated.get(key, 0.0) + withdrawal

    hydro_ids: list[int] = [k[0] for k in accumulated]
    stage_ids: list[int] = [k[1] for k in accumulated]
    values: list[float] = list(accumulated.values())

    if not hydro_ids:
        return None

    # Extrapolate to post-study period by cycling the last calendar year's
    # Jan–Dec pattern from dsvagua.  The last year of dsvagua always contains
    # a full Jan–Dec cycle.  We build a template keyed by calendar month
    # (1–12) and map each post-study stage to its calendar month.
    if num_post_study_stages > 0:
        # Build per-plant template: hydro_id -> {calendar_month: value}
        # using the last calendar year present in dsvagua per plant.
        plant_by_date: dict[int, dict[tuple[int, int], float]] = {}
        for hid, sid, val in zip(hydro_ids, stage_ids, values):
            # Recover calendar (year, month) from stage_id.
            total_month = start_month + sid  # 1-based month offset from year start
            cal_year = start_year + (total_month - 1) // 12
            cal_month = (total_month - 1) % 12 + 1
            plant_by_date.setdefault(hid, {})[(cal_year, cal_month)] = val

        # Build per-plant set of stages that already have data.
        existing_stages: dict[int, set[int]] = {}
        for hid, sid in zip(hydro_ids, stage_ids):
            existing_stages.setdefault(hid, set()).add(sid)

        for hid, date_map in plant_by_date.items():
            last_year = max(y for y, _m in date_map)
            template: dict[int, float] = {}
            for m in range(1, 13):
                if (last_year, m) in date_map:
                    template[m] = date_map[(last_year, m)]

            if not template:
                continue

            have = existing_stages.get(hid, set())
            for s in range(num_total_stages):
                if s in have:
                    continue
                total_month = start_month + s
                cal_month = (total_month - 1) % 12 + 1
                val = template.get(cal_month)
                if val is not None:
                    hydro_ids.append(hid)
                    stage_ids.append(s)
                    values.append(val)

        _LOG.info(
            "Extrapolated water withdrawal to %d post-study stages "
            "(%d -> %d total stages).",
            num_post_study_stages,
            num_study_stages,
            num_total_stages,
        )

    table = pa.table(
        {
            "hydro_id": pa.array(hydro_ids, type=pa.int32()),
            "stage_id": pa.array(stage_ids, type=pa.int32()),
            "water_withdrawal_m3s": pa.array(values, type=pa.float64()),
        }
    )
    # Sort by (hydro_id, stage_id) for deterministic output.
    return table.sort_by([("hydro_id", "ascending"), ("stage_id", "ascending")])


_RECORD_SWITCH = {
    "TURBMAXT": "turbined_max",
    "TURBMINT": "turbined_min",
    "VAZMINT": "min_outflow",
}


def _drop_switched_off_records(
    overrides: dict[int, list[dict]], switches: DgerSwitches
) -> dict[int, list[dict]]:
    """MODIF.DAT dated records whose ``dger.dat`` switch is off, removed and
    reported once per record type."""
    dropped: set[str] = set()
    kept: dict[int, list[dict]] = {}
    for code, records in overrides.items():
        survivors = []
        for record in records:
            name = _RECORD_SWITCH.get(record["type"])
            if name is not None and not getattr(switches, name).on:
                dropped.add(name)
                continue
            survivors.append(record)
        if survivors:
            kept[code] = survivors
    for name in sorted(dropped):
        emit(switch_off_diagnostic(getattr(switches, name)), logger=_LOG)
    return kept


def convert_storage_bounds(
    case: NewaveCase,
    id_map: NewaveIdMap,
) -> pa.Table | None:
    """Build per-stage hydro bounds from MODIF.DAT temporal overrides.

    Handles four override types:

    - **VMAXT / VMINT**: storage volume overrides, in hm³ (unit ``h``) or as a
      percentage of the useful volume (unit ``%``, vol_max - vol_min).  Flood
      control and operational minimums.
    - **TURBMAXT / TURBMINT**: turbined flow overrides in absolute m³/s.
      Values of 99999 mean "no limit" (restore default).
    - **VAZMINT**: minimum outflow overrides in absolute m³/s.

    Each override acts as a step function: the value persists from its
    effective date until the next override for the same plant.

    Post-study extrapolation is per-quantity: VMINT/VMAXT repeat the last study
    year's seasonal pattern only when their dger ``sazonaliza_*`` flag is set
    (e.g. flood-control "volume de espera"); outflow (VAZMINT) and turbined
    (TURBMINT/TURBMAXT) have no seasonalize flag and freeze the last study
    stage value.

    Returns ``None`` if MODIF.DAT is absent or contains no relevant records.
    """
    dger = case.dger
    horizon = case.horizon
    start_year = horizon.start_year
    start_month = horizon.start_month
    study_months = horizon.study_months
    total_stages = horizon.total_stages

    # Post-study seasonalize flags (dger). VMINT/VMAXT repeat the last study
    # year's seasonal pattern only when their flag is set (e.g. flood-control
    # "volume de espera"); outflow (VAZMINT) and turbined (TURBMINT/TURBMAXT)
    # have no such flag and always freeze the last study stage value.
    sazonaliza_vmaxt = int(getattr(dger, "sazonaliza_vmaxt", 0) or 0) == 1
    sazonaliza_vmint = int(getattr(dger, "sazonaliza_vmint", 0) or 0) == 1

    # Read hidr.dat with permanent overrides for vol_min/vol_max.
    cadastro = read_cadastro(case)

    # Read confhd for the list of active plant codes.
    confhd_codes = case.active_hydro_codes

    exph_df = case.exph.expansoes if case.exph is not None else None
    filling_codes = filling_hydro_codes(case.confhd.usinas, exph_df)

    # Extract temporal overrides — empty dict when MODIF.DAT is absent,
    # which is fine because GHMIN.DAT alone can still produce per-stage
    # rows.
    switches = case.switches
    if case.files.modif is None:
        temporal_overrides: dict[int, list[dict]] = {}
    else:
        temporal_overrides = _drop_switched_off_records(
            _extract_temporal_overrides(case, confhd_codes), switches
        )

    def _build_step_function(
        recs: list[dict],
        transform: Callable[[float], float],
        *,
        seasonalize: bool,
    ) -> dict[int, float]:
        """Thin adapter over :func:`novomodelo_bridge.newave.horizon.seasonal_step_function`:
        dated records become ``(year, month, value)`` change-points and
        ``PRE``/``POS`` records ``(month, value)`` steps.
        """

        def steps(period: str) -> list[tuple[int, float]]:
            return [(r["month"], r["value"]) for r in recs if r["period"] == period]

        return seasonal_step_function(
            [(r["year"], r["month"], r["value"]) for r in recs if r["period"] is None],
            transform,
            seasonalize=seasonalize,
            horizon=horizon,
            pre_recs=steps("PRE"),
            pos_recs=steps("POS"),
        )

    # GHMIN.DAT per-stage minimums.  These are not MODIF.DAT overrides
    # but live alongside them at the per-(hydro, stage) granularity, so
    # they merge naturally into this parquet's row set.
    ghmin_by_plant_stage: dict[int, dict[int, float]] = {}
    if case.files.ghmin is not None and not switches.ghmin.on:
        emit(switch_off_diagnostic(switches.ghmin), logger=_LOG)
    else:
        ghmin_by_plant_stage = _read_ghmin_per_stage(
            case, start_year, start_month, study_months, total_stages
        )

    hydro_ids: list[int] = []
    stage_ids: list[int] = []
    min_storage_vals: list[float | None] = []
    max_storage_vals: list[float | None] = []
    min_turbined_vals: list[float | None] = []
    max_turbined_vals: list[float | None] = []
    min_outflow_vals: list[float | None] = []
    min_generation_vals: list[float | None] = []
    max_generation_vals: list[float | None] = []
    # Per-row origin tag: True for a filling-plant ramp row, False for a
    # MODIF/GHMIN row. Drives the ramp-wins de-dup pass before the table build.
    is_ramp_vals: list[bool] = []

    def _identity(val: float) -> float:
        return val

    plant_codes_with_data = set(temporal_overrides) | set(ghmin_by_plant_stage)
    for newave_code in sorted(plant_codes_with_data):
        overrides = temporal_overrides.get(newave_code, [])
        vmaxt = [o for o in overrides if o["type"] == "VMAXT"]
        vmint = [o for o in overrides if o["type"] == "VMINT"]
        turbmaxt = [o for o in overrides if o["type"] == "TURBMAXT"]
        turbmint = [o for o in overrides if o["type"] == "TURBMINT"]
        vazmint = [o for o in overrides if o["type"] == "VAZMINT"]
        ghmin_by_stage = ghmin_by_plant_stage.get(newave_code, {})

        if not any((vmaxt, vmint, turbmaxt, turbmint, vazmint, ghmin_by_stage)):
            continue

        try:
            hydro_id = id_map.hydro_id(newave_code)
        except KeyError:
            continue

        if newave_code not in cadastro.index:
            continue

        hreg = cadastro.loc[newave_code]
        vol_min = float(hreg["volume_minimo"])
        vol_max = float(hreg["volume_maximo"])
        useful = vol_max - vol_min

        def _storage_hm3(rec: dict) -> dict | None:
            """The record with its volume in hm³; ``None`` drops a percentage
            record on a plant without useful volume. A big-M sentinel passes
            through untouched so the step function still reads it as "restore"."""
            value = float(rec["value"])
            if value >= BIG_M or rec.get("unit", "%") == "h":
                return rec
            if useful <= 0:
                return None
            return {**rec, "value": percent_of_useful_volume(value, vol_min, vol_max)}

        vmaxt_hm3 = [r for r in map(_storage_hm3, vmaxt) if r is not None]
        vmint_hm3 = [r for r in map(_storage_hm3, vmint) if r is not None]
        # Seasonal post-study iff the corresponding dger flag is set; else freeze.
        vmaxt_by_stage = _build_step_function(
            vmaxt_hm3, lambda v: v, seasonalize=sazonaliza_vmaxt
        )
        vmint_by_stage = _build_step_function(
            vmint_hm3, lambda v: v, seasonalize=sazonaliza_vmint
        )

        # Turbined bounds (absolute m³/s) — no seasonalize flag → freeze.
        turbmaxt_by_stage = _build_step_function(turbmaxt, _identity, seasonalize=False)
        turbmint_by_stage = _build_step_function(turbmint, _identity, seasonalize=False)

        # Outflow bounds (absolute m³/s) — no seasonalize flag → freeze.
        vazmint_by_stage = _build_step_function(vazmint, _identity, seasonalize=False)

        all_stages = sorted(
            set(vmaxt_by_stage)
            | set(vmint_by_stage)
            | set(turbmaxt_by_stage)
            | set(turbmint_by_stage)
            | set(vazmint_by_stage)
            | set(ghmin_by_stage)
        )
        for stage_id in all_stages:
            hydro_ids.append(hydro_id)
            stage_ids.append(stage_id)
            max_storage_vals.append(vmaxt_by_stage.get(stage_id))
            min_storage_vals.append(vmint_by_stage.get(stage_id))
            max_turbined_vals.append(turbmaxt_by_stage.get(stage_id))
            min_turbined_vals.append(turbmint_by_stage.get(stage_id))
            min_outflow_vals.append(vazmint_by_stage.get(stage_id))
            min_generation_vals.append(ghmin_by_stage.get(stage_id))
            max_generation_vals.append(None)
            is_ramp_vals.append(False)

    # Filling-plant unit-ramp branch: a ``NE``-with-filling plant operates from
    # ``entry_sid`` but its turbine/generation capacity at each stage is whatever
    # generating units are online.  Export EXPLICIT 0/reduced caps over the FULL
    # pre-operating window ``[0, full_online_sid)`` (clamped to the in-study horizon)
    # so the parquet carries the true 0→full capacity profile.  ``online_machines``
    # clamps every unit's online stage up to ``entry_sid``, so ``[0, entry_sid)``
    # (PreFilling/Filling) gets ``(0, 0)`` caps and ``[entry_sid, full_online_sid)``
    # gets reduced caps; from ``full_online_sid`` the base ``hydros.json`` caps apply.
    # These rows are inert to novomodelo's own PreFilling/Filling forcing: its
    # ``hydro_bounds`` reader is a sparse override table with NO stage-window
    # validation, so a ``max=0`` row there leaves the simulation result UNCHANGED —
    # only the exported data gains the explicit 0-cap stages.  A ramp-window stage may
    # also carry a MODIF/GHMIN row; those append SEPARATE rows tagged ``is_ramp=True``,
    # and the de-dup pass below resolves any ``(hydro_id, stage_id)`` collision in
    # favour of the ramp row (the explicit 0-cap wins over a colliding MODIF/GHMIN
    # minimum), since novomodelo defers duplicate-pair handling.
    if filling_codes and case.exph is not None:
        exph_df = case.exph.expansoes
        for code in sorted(filling_codes):
            try:
                hydro_id = id_map.hydro_id(code)
            except KeyError:
                continue
            if code not in cadastro.index:
                continue
            hreg = cadastro.loc[code]
            name = str(hreg.get("nome_usina", "")).strip() or str(code)

            rows = exph_df.loc[exph_df["codigo_usina"] == code]
            fill_row = rows.loc[rows["data_inicio_enchimento"].notna()].iloc[0]
            ts = fill_row["data_inicio_enchimento"]
            duracao = int(fill_row["duracao_enchimento"])
            _start_sid, entry_sid = filling_schedule(
                ts.year,
                ts.month,
                duracao,
                start_year,
                start_month,
            )

            unit_rows = exph_unit_rows(exph_df, code, start_year, start_month)
            if not unit_rows:
                continue

            # All-units-online stage: the exported pre-operating window is
            # [0, full_online_sid). Clamp the upper bound to total_stages so the
            # loop never indexes past the horizon. A
            # plant whose entry is at/after the horizon end still emits explicit
            # 0-cap rows for every in-study stage [0, total_stages) — it never
            # operates in-study, so every stage is pre-operating.
            full_online_sid = max(max(usid, entry_sid) for _c, usid in unit_rows)
            for s in range(min(full_online_sid, total_stages)):
                online = online_machines(unit_rows, entry_sid, s)
                mt, mg = _reduced_caps(hreg, online, name)
                hydro_ids.append(hydro_id)
                stage_ids.append(s)
                min_storage_vals.append(None)
                max_storage_vals.append(None)
                min_turbined_vals.append(None)
                max_turbined_vals.append(mt)
                min_outflow_vals.append(None)
                min_generation_vals.append(0.0)
                max_generation_vals.append(mg)
                is_ramp_vals.append(True)

    if not hydro_ids:
        return None

    # Resolve duplicate ``(hydro_id, stage_id)`` pairs: a ramp row wins over a
    # MODIF/GHMIN row at the same key (the explicit 0-cap during filling must not
    # be undercut by a MODIF/GHMIN minimum). A case with no filling plant
    # produces no ramp rows (``is_ramp_vals`` all False), so every key is unique
    # and this is a no-op, keeping the regression-guard output byte-identical.
    chosen: dict[tuple[int, int], int] = {}
    for i, (h, s, ramp) in enumerate(zip(hydro_ids, stage_ids, is_ramp_vals)):
        key = (h, s)
        if key not in chosen or ramp:  # ramp overrides an earlier MODIF/GHMIN row
            chosen[key] = i
    keep = sorted(chosen.values())
    if len(keep) != len(hydro_ids):
        hydro_ids = [hydro_ids[i] for i in keep]
        stage_ids = [stage_ids[i] for i in keep]
        min_storage_vals = [min_storage_vals[i] for i in keep]
        max_storage_vals = [max_storage_vals[i] for i in keep]
        min_turbined_vals = [min_turbined_vals[i] for i in keep]
        max_turbined_vals = [max_turbined_vals[i] for i in keep]
        min_outflow_vals = [min_outflow_vals[i] for i in keep]
        min_generation_vals = [min_generation_vals[i] for i in keep]
        max_generation_vals = [max_generation_vals[i] for i in keep]
        is_ramp_vals = [is_ramp_vals[i] for i in keep]

    columns = {
        "hydro_id": pa.array(hydro_ids, type=pa.int32()),
        "stage_id": pa.array(stage_ids, type=pa.int32()),
        "min_storage_hm3": pa.array(min_storage_vals, type=pa.float64()),
        "max_storage_hm3": pa.array(max_storage_vals, type=pa.float64()),
        "min_turbined_m3s": pa.array(min_turbined_vals, type=pa.float64()),
        "max_turbined_m3s": pa.array(max_turbined_vals, type=pa.float64()),
        "min_outflow_m3s": pa.array(min_outflow_vals, type=pa.float64()),
        "min_generation_mw": pa.array(min_generation_vals, type=pa.float64()),
    }
    # Gate the column on a filling plant: a case without one keeps the 8-column
    # schema byte-identical (the regression guard depends on this); novomodelo's
    # parse_hydro_bounds tolerates the absent column.
    if filling_codes:
        columns["max_generation_mw"] = pa.array(max_generation_vals, type=pa.float64())
    return pa.table(columns).sort_by(
        [("hydro_id", "ascending"), ("stage_id", "ascending")]
    )
