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
from novomodelo_bridge.core.hydro_units import rated_capacity
from novomodelo_bridge.core.pandas_utils import is_na
from novomodelo_bridge.core.productivity import (
    KTURB_BY_TIPO_TURBINA,
    apply_hydraulic_loss,
    evaluate_cota,
    mean_cota,
)
from novomodelo_bridge.core.tolerances import BIG_M
from novomodelo_bridge.newave.case import NewaveCase
from novomodelo_bridge.newave.converters.hydro.geometry import (
    _expansion_configs,
    _read_volref_saz,
)
from novomodelo_bridge.newave.converters.hydro.overrides import (
    _apply_permanent_overrides,
    _extract_temporal_overrides,
    _read_ghmin_per_stage,
    percent_of_useful_volume,
    read_cadastro,
)
from novomodelo_bridge.newave.converters.hydro.productivity import (
    _compute_productivity,
    _per_stage_productivities,
    _total_study_stages,
)
from novomodelo_bridge.newave.filling import (
    exph_unit_rows,
    filling_schedule,
    hreg_with_machines,
    online_machines,
)
from novomodelo_bridge.newave.horizon import seasonal_step_function
from novomodelo_bridge.newave.id_map import NewaveIdMap
from novomodelo_bridge.newave.plants import fictitious_codes, filling_hydro_codes
from novomodelo_bridge.newave.switches import DgerSwitches, switch_off_diagnostic

_LOG = logging.getLogger(__name__)


def _clamp_outage_pct(value: float, label: str, plant_name: str) -> float:
    """Clamp TEIF/IP percentages into ``[0, 100]``, reporting an overshoot.

    A leaf helper on multiple call paths (``_compute_max_turbined_simple`` and
    ``_compute_max_turbined_head_corrected``, from both ``convert_hydros`` and
    :func:`convert_turbined_bounds_head_corrected`): the same plant+field can
    be clamped on more than one path in one run.  Emitting per-call and
    relying on ``finalize_diagnostics``' ``(code, summary)`` de-dup collapses
    that to a single diagnostic instead of threading an accumulator through
    every call site.
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


def _availability(teif: float, ip: float) -> float:
    """Joint TEIF/IP availability factor — both apply multiplicatively."""
    return ((100.0 - teif) / 100.0) * ((100.0 - ip) / 100.0)


def _compute_max_turbined_simple(hreg: pd.Series, name: str) -> tuple[float, float]:
    """Return ``(max_turbined, max_generation)`` using the simple sum-of-rated
    approach: ``Σ_c (n_c · q_nom_c)`` for flow and ``Σ_c (n_c · p_nom_c)`` for
    power, both derated by the combined TEIF/IP availability factor.

    This is **live code on two paths**: it is the fallback
    :func:`_compute_max_turbined_head_corrected` calls whenever a plant lacks the
    head data (``queda_nominal_conjunto_*`` / cota polynomial) the corrected
    formula needs, and it is the v0.6.x formulation the top-level caller can be
    switched back to if the head-corrected formula proves wrong.
    """
    n_sets = int(hreg["numero_conjuntos_maquinas"])
    max_turbined = 0.0
    max_generation = 0.0
    for i in range(1, n_sets + 1):
        n_machines = int(hreg[f"maquinas_conjunto_{i}"])
        q_nominal = float(hreg[f"vazao_nominal_conjunto_{i}"])
        p_nominal = float(hreg[f"potencia_nominal_conjunto_{i}"])
        max_turbined += q_nominal * n_machines
        max_generation += p_nominal * n_machines

    teif = _clamp_outage_pct(float(hreg.get("teif", 0.0) or 0.0), "teif", name)
    ip = _clamp_outage_pct(float(hreg.get("ip", 0.0) or 0.0), "ip", name)
    availability = _availability(teif, ip)
    return max_turbined * availability, max_generation * availability


def _compute_max_turbined_head_corrected(
    hreg: pd.Series, name: str, *, h_op_override: float | None = None
) -> tuple[float, float]:
    """Return ``(max_turbined, max_generation)`` using the head-corrected the
    source-model-style cap.

    ``h_op_override`` supplies the per-stage operating head directly (in metres),
    bypassing the static base-data head derivation below. It is the head that
    produces the stage's equivalent productivity (``ρ_eq / ρ_esp``), so the
    engolimento and the ``p_inst/prodt_eq`` cap track the per-stage CFUGA/CMONT
    overrides in lockstep with :func:`convert_hydro_energy_productivity`. Used by
    :func:`convert_turbined_bounds_head_corrected` to emit a per-stage
    ``max_turbined`` instead of a single static cap.

    This is the source model's **operational turbined cap** — the maximum turbinable
    flow (engolimento) at the plant's operating head, which is what binds in the
    dispatch LP. Verified against the source model on QUEBRA QUEIX, whose cap binds:
    head-corrected 113.006 m³/s == the source model's realized 113.01 (the rated ``Σ
    n·q_nom`` = 117.0 overshoots it by 3.5%). ``convert_hydros`` emits the flow value
    ``[0]`` as ``max_turbined``; the ``[1]`` it returns is the availability-derated
    power and is no longer used for the emitted ``max_generation`` (that comes from
    :func:`~novomodelo_bridge.core.hydro_units.rated_capacity`).

    For each machine set *c* with nominal head ``h_nom_c``, nominal flow
    ``q_nom_c`` and number of units ``n_c``, the effective rated flow at
    operating head ``h_op`` follows the affinity-law approximation::

        q_eff_c = (h_op / h_nom_c)^k_turb · q_nom_c

    where ``k_turb`` depends on the turbine family (0.5 for Francis/Pelton,
    0.2 for Kaplan).  The plant cap is then

        qtur_max = min(Σ_c n_c · q_eff_c , p_inst / prodt_eq) · availability

    with ``h_op`` being:

    * the volume-integrated net head from ``V_min`` to ``V_65``, i.e.
      ``mean_cota(V_min, V_65) - cota_jus - perdas``, for reservoir plants
      (``tipo_regulacao == 'M'``).  This is the source model ``h^{65%}`` symbol —
      despite the name, it is a *mean over the operating range* rather than the snapshot
      at V = V_65.  Reproduces the source model's actual cap exactly on the diagnostic
      case (M. DE MORAES) and matches the equivalent productivity ``prodt^M``
      denominator;
    * the machine-count-weighted average of nominal heads, for run-of-river /
      daily plants — they don't have a meaningful "operating volume".

    ``prodt_eq = ρ_esp · h_int`` integrates the head polynomial over
    ``[V_min, V_65]`` for M plants (so it equals ``ρ_esp · h_op``) and
    ``[V_min, V_max]`` for the others.

    Falls back to the simple Σ(n·q) when the inputs needed for the head
    correction are missing (``queda_nominal_conjunto_*`` columns absent,
    no polynomial coefficients, zero ρ_esp, etc.) — this keeps unit-test
    fixtures with partial schemas working.
    """
    n_sets = int(hreg["numero_conjuntos_maquinas"])
    available_cols = set(hreg.index)
    has_head_data = all(
        f"queda_nominal_conjunto_{i}" in available_cols for i in range(1, 6)
    ) and all(f"a{i}_volume_cota" in available_cols for i in range(5))
    if not has_head_data:
        return _compute_max_turbined_simple(hreg, name)

    sum_n_q = 0.0
    sum_n_p = 0.0
    sum_n_h = 0.0
    total_machines = 0
    for i in range(1, n_sets + 1):
        n_machines = int(hreg[f"maquinas_conjunto_{i}"])
        q_nominal = float(hreg[f"vazao_nominal_conjunto_{i}"])
        p_nominal = float(hreg[f"potencia_nominal_conjunto_{i}"])
        h_nominal = float(hreg[f"queda_nominal_conjunto_{i}"])
        sum_n_q += n_machines * q_nominal
        sum_n_p += n_machines * p_nominal
        sum_n_h += n_machines * h_nominal
        total_machines += n_machines

    teif = _clamp_outage_pct(float(hreg.get("teif", 0.0) or 0.0), "teif", name)
    ip = _clamp_outage_pct(float(hreg.get("ip", 0.0) or 0.0), "ip", name)
    availability = _availability(teif, ip)

    # Generation cap mirrors the legacy behavior — the source model caps power at rated
    # installed capacity, derated by availability.
    max_generation = sum_n_p * availability

    # Inputs needed for head correction.
    vol_min = float(hreg["volume_minimo"])
    vol_max = float(hreg["volume_maximo"])
    cf_raw = hreg.get("canal_fuga_medio")
    rho_esp_raw = hreg.get("produtibilidade_especifica")
    tipo_perda = int(hreg.get("tipo_perda", 0) or 0)
    perdas = float(hreg.get("perdas", 0.0) or 0.0)
    tipo_turbina = int(hreg.get("tipo_turbina", 0) or 0)
    tipo_reg = str(hreg.get("tipo_regulacao", "")).strip()

    # If we lack the inputs to compute the head correction, fall back to the
    # simple sum.  This protects against malformed/incomplete hidr rows.
    if (
        total_machines == 0
        or cf_raw is None
        or is_na(cf_raw)
        or rho_esp_raw is None
        or is_na(rho_esp_raw)
        or float(rho_esp_raw) <= 0.0
    ):
        return sum_n_q * availability, max_generation

    cf = float(cf_raw)
    rho_esp = float(rho_esp_raw)
    kturb = KTURB_BY_TIPO_TURBINA.get(tipo_turbina, 0.5)
    coeffs = [float(hreg[f"a{i}_volume_cota"]) for i in range(5)]

    if h_op_override is not None:
        # Per-stage caller supplies the operating head directly (= ρ_eq / ρ_esp for
        # the stage), so the cap tracks the per-stage CFUGA/CMONT head. h_int == h_op
        # holds in every branch below, so reuse it for the prodt_eq cap too.
        h_op = h_op_override
        h_int = h_op_override
    elif tipo_reg == "M":
        v65 = vol_min + 0.65 * (vol_max - vol_min)
        # The source model's ``h^{65%}`` is the *integrated* net head over [V_min,
        # V_65], not the snapshot at V = V_65.  Verified against M. DE MORAES (the
        # diagnostic case): with this interpretation the formula reproduces the observed
        # the source model cap of 1084.95 m³/s exactly.
        h_int_gross = mean_cota(coeffs, vol_min, v65) - cf
        h_op = apply_hydraulic_loss(h_int_gross, tipo_perda, perdas)
        # For M plants ``prodt^M`` integrates over the same range, so h_int
        # equals h_op — the same value is reused intentionally.
        h_int = h_op
    else:
        # Run-of-river ('S') and daily-regulated ('D') plants operate at a ~constant
        # reference volume (``volume_referencia``): operating head = forebay cota at
        # vol_ref minus tailrace and losses, used for BOTH the turbine affinity ratio
        # and prodt.  NOT the machine-weighted *nominal* head — that makes the affinity
        # ratio ``h_op / h_nominal`` ≈ 1, a silent no-op leaving ``max_turbined =
        # Σ(n·q)·availability`` uncorrected.  Matches the source model's operating head
        # ``(GHIDUH/QTURUH)/ρ_esp`` (ITAIPU 113.37 m vs nominal 117 m).
        vol_ref_raw = hreg.get("volume_referencia")
        if vol_ref_raw is None or is_na(vol_ref_raw) or float(vol_ref_raw) <= 0.0:
            h_op_gross = mean_cota(coeffs, vol_min, vol_max) - cf
        else:
            h_op_gross = evaluate_cota(coeffs, float(vol_ref_raw)) - cf
        h_op = apply_hydraulic_loss(h_op_gross, tipo_perda, perdas)
        h_int = h_op

    # Defensive: a negative or zero h_op means cota_jus is above the
    # forebay (data error or post-overhaul cota).  Fall back to the simple
    # Σ(n·q) to avoid emitting a meaningless bound.
    if h_op <= 0.0 or h_int <= 0.0:
        return sum_n_q * availability, max_generation

    # Affinity-law correction per machine set.
    sum_kt = 0.0
    for i in range(1, n_sets + 1):
        n_machines = int(hreg[f"maquinas_conjunto_{i}"])
        q_nominal = float(hreg[f"vazao_nominal_conjunto_{i}"])
        h_nominal = float(hreg[f"queda_nominal_conjunto_{i}"])
        if h_nominal <= 0.0:
            sum_kt += n_machines * q_nominal
            continue
        ratio = h_op / h_nominal
        # Use abs to guard against the rare case where h_op flips sign during
        # overhaul-only periods; the exponent < 1 keeps the result finite.
        sum_kt += n_machines * q_nominal * (ratio**kturb)

    # Installed-power-over-equivalent-productivity cap.
    prodt_eq = rho_esp * h_int
    cap_pinst = sum_n_p / prodt_eq if prodt_eq > 0.0 else math.inf

    max_turbined = min(sum_kt, cap_pinst) * availability
    return max_turbined, max_generation


def _reduced_caps(
    hreg: pd.Series, online: dict[int, int], name: str
) -> tuple[float, float]:
    """Return ``(max_turbined, max_generation)`` over the online machine subset.

    Computes the head-corrected turbined cap and the rated generation cap for a
    filling plant at a ramp stage where only ``online`` of its machines are in
    service. Rather than threading an override count through the load-bearing
    shared helpers, it copies the cadastro ``hreg`` Series and rewrites each
    ``maquinas_conjunto_{c}`` (for ``c`` in ``1..numero_conjuntos_maquinas``) to
    the online count (``0`` for any group absent from the dict), then re-runs the
    exact full-capacity helpers — so the all-online stage equals the base
    ``hydros.json`` cap by construction. With an empty ``online`` dict every
    count is ``0`` and both caps are ``0.0`` (an explicit zero-capacity stage, not
    a skip).
    """
    n_sets = int(hreg["numero_conjuntos_maquinas"])
    hreg_copy = hreg_with_machines(
        hreg, {c: online.get(c, 0) for c in range(1, n_sets + 1)}
    )
    max_turbined = _compute_max_turbined_head_corrected(hreg_copy, name)[0]
    max_generation = rated_capacity(hreg_copy)[1]
    return max_turbined, max_generation


def convert_turbined_bounds_head_corrected(
    case: NewaveCase, id_map: NewaveIdMap
) -> pa.Table | None:
    """Per-stage ``max_turbined_m3s`` for plants whose operating head varies by stage.

    The engolimento (max turbinable flow) depends on the operating head — both via
    the turbine affinity ratio and the ``p_inst / prodt_eq`` installed-power cap. For
    plants carrying MODIF.DAT CFUGA/CMONT temporal overrides (or a seasonal V_ref),
    that head changes stage-to-stage, exactly as the per-stage equivalent
    productivity does (:func:`convert_hydro_energy_productivity`). The static
    per-plant cap in ``hydros.json`` is computed at a single reference head, so at
    high-flow/low-head stages it under-caps turbining and forces spill (lost hydro →
    extra thermal). This emits a per-(hydro, stage) ``max_turbined`` override using
    the SAME per-stage head (``h = ρ_eq / ρ_esp``) that drives productivity, so the
    two stay consistent. An expanding plant is rated with the machines in service
    at each stage; one with no per-stage head gets rows only before its last
    machine enters.

    Returns a ``(hydro_id, stage_id, max_turbined_m3s)`` table for the affected
    plants/stages, or ``None`` when no plant has a per-stage head or a machine ramp.
    """
    cadastro = _apply_permanent_overrides(case.hidr.cadastro, case)
    confhd_codes = [int(r["codigo_usina"]) for _, r in case.active_hydros.iterrows()]

    temporal_overrides = _extract_temporal_overrides(case, confhd_codes)
    drop_overrides = {
        code: [o for o in overrides if o["type"] in ("CFUGA", "CMONT")]
        for code, overrides in temporal_overrides.items()
        if any(o["type"] in ("CFUGA", "CMONT") for o in overrides)
    }
    seasonal_volref = _read_volref_saz(case)
    expansion = _expansion_configs(case, cadastro)
    if not drop_overrides and not seasonal_volref and not expansion:
        return None

    total_stages = _total_study_stages(case)
    if total_stages <= 0:
        return None

    hydro_ids: list[int] = []
    stage_ids: list[int] = []
    max_turbined_vals: list[float] = []

    for newave_code in sorted(confhd_codes):
        overrides = drop_overrides.get(newave_code, [])
        plant_seasonal = seasonal_volref.get(newave_code)
        config = expansion.get(newave_code)
        if not overrides and not plant_seasonal and config is None:
            continue
        if newave_code not in cadastro.index:
            continue
        try:
            hydro_id = id_map.hydro_id(newave_code)
        except KeyError:
            continue
        hreg = cadastro.loc[newave_code]
        name = str(hreg.get("nome_usina", newave_code))
        if config is not None and not overrides and not plant_seasonal:
            # Rated at the reference head the declared value uses, never a
            # productivity-derived one, which rates the same machines differently.
            for stage_id in range(min(config.full_online_stage, total_stages)):
                hydro_ids.append(hydro_id)
                stage_ids.append(stage_id)
                max_turbined_vals.append(
                    _compute_max_turbined_head_corrected(
                        config.hreg_at(hreg, stage_id), name
                    )[0]
                )
            continue
        rho_esp_raw = hreg.get("produtibilidade_especifica")
        if rho_esp_raw is None or is_na(rho_esp_raw) or float(rho_esp_raw) <= 0.0:
            continue
        rho_esp = float(rho_esp_raw)

        legacy_base = _compute_productivity(hreg)
        per_stage_prod = _per_stage_productivities(
            hreg,
            legacy_base,
            overrides,
            case,
            total_stages,
            seasonal_volref_by_month=plant_seasonal,
        )
        for stage_id, prod in enumerate(per_stage_prod):
            if prod <= 0.0:
                continue
            h_op = prod / rho_esp
            hreg_stage = hreg if config is None else config.hreg_at(hreg, stage_id)
            max_turbined = _compute_max_turbined_head_corrected(
                hreg_stage, name, h_op_override=h_op
            )[0]
            hydro_ids.append(hydro_id)
            stage_ids.append(stage_id)
            max_turbined_vals.append(max_turbined)

    if not hydro_ids:
        return None

    return pa.table(
        {
            "hydro_id": pa.array(hydro_ids, type=pa.int32()),
            "stage_id": pa.array(stage_ids, type=pa.int32()),
            "max_turbined_m3s": pa.array(max_turbined_vals, type=pa.float64()),
        }
    )


def _per_stage_turbined_envelope(
    case: NewaveCase, id_map: NewaveIdMap
) -> dict[int, float]:
    """Return ``{novomodelo_hydro_id: max per-stage max_turbined_m3s}`` from the
    head-corrected per-stage table.

    Delegates to :func:`convert_turbined_bounds_head_corrected` — the exact
    function the pipeline uses to emit the per-(hydro, stage)
    ``hydro_bounds`` rows — instead of re-deriving the per-stage head. This is
    what lets :func:`convert_hydros` raise its declared
    ``generation.max_turbined_m3s`` to cover every emitted per-stage row
    (novomodelo rule 43) with zero risk of the two ever drifting apart: they are
    two views of the same table, not two formulas that happen to agree today.

    Returns an empty dict when no plant has a per-stage head (no CFUGA/CMONT
    temporal overrides and no seasonal ``VOLREF_SAZ`` row) or a machine ramp,
    so callers must treat a missing key as "no per-stage variation", not as zero.
    """
    table = convert_turbined_bounds_head_corrected(case, id_map)
    if table is None:
        return {}
    envelope: dict[int, float] = {}
    for hydro_id, value in zip(
        table["hydro_id"].to_pylist(),
        table["max_turbined_m3s"].to_pylist(),
        strict=True,
    ):
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
    fict_set = fictitious_codes(confhd_df, case.hidr.cadastro)
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
    expansion = _expansion_configs(case, cadastro)

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

    plant_codes_with_data = (
        set(temporal_overrides) | set(ghmin_by_plant_stage) | set(expansion)
    )
    for newave_code in sorted(plant_codes_with_data):
        config = expansion.get(newave_code)
        overrides = temporal_overrides.get(newave_code, [])
        vmaxt = [o for o in overrides if o["type"] == "VMAXT"]
        vmint = [o for o in overrides if o["type"] == "VMINT"]
        turbmaxt = [o for o in overrides if o["type"] == "TURBMAXT"]
        turbmint = [o for o in overrides if o["type"] == "TURBMINT"]
        vazmint = [o for o in overrides if o["type"] == "VAZMINT"]
        ghmin_by_stage = ghmin_by_plant_stage.get(newave_code, {})

        if (
            not any((vmaxt, vmint, turbmaxt, turbmint, vazmint, ghmin_by_stage))
            and config is None
        ):
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

        # An expanding plant's ramp caps land on the SAME ``(hydro, stage)`` row
        # as any MODIF/GHMIN bound below, never a competing one, so the de-dup
        # pass never has to choose between a ramp cap and a VMINT floor.
        ramp_generation: dict[int, float] = {}
        if config is not None:
            for ramp_stage in range(min(config.full_online_stage, total_stages)):
                ramp_generation[ramp_stage] = rated_capacity(
                    config.hreg_at(hreg, ramp_stage)
                )[1]

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
            | set(ramp_generation)
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
            max_generation_vals.append(ramp_generation.get(stage_id))
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
    # Gate the column on a filling or expanding plant: a case with neither keeps
    # the 8-column schema byte-identical (the regression guard depends on this);
    # novomodelo's parse_hydro_bounds tolerates the absent column.
    if filling_codes or expansion:
        columns["max_generation_mw"] = pa.array(max_generation_vals, type=pa.float64())
    return pa.table(columns).sort_by(
        [("hydro_id", "ascending"), ("stage_id", "ascending")]
    )
