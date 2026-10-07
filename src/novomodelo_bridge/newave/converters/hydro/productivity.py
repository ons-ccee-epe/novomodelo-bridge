"""Hydro productivity: per-stage/per-plant equivalent-productivity (rho_eq)
machinery, the production-model JSON, and the FPHA plane-reduction /
computed-config helpers.

All point / PRODT / integrated productivity math routes through the
presentation-free :mod:`cobre_bridge.core.productivity` domain module; this module
holds only the source-model-specific orchestration (temporal overrides,
seasonal reference volumes, FICT-cascade fold-in) around that math.
"""

from __future__ import annotations

import logging

import pandas as pd
import pyarrow as pa

from cobre_bridge.cobre import schemas as cobre_schemas
from cobre_bridge.core.diagnostics import Diagnostic, Severity, emit
from cobre_bridge.core.productivity import (
    compute_productivity,
    equivalent_productivity,
    integrated_productivity,
    stored_energy_productivity,
)
from cobre_bridge.newave.case import NewaveCase
from cobre_bridge.newave.converters.hydro.geometry import (
    _read_volref_saz,
    _seasonal_reference_volume,
    fpha_eligible_codes,
)
from cobre_bridge.newave.converters.hydro.overrides import (
    _apply_permanent_overrides,
    _extract_temporal_overrides,
    _per_stage_drop_overrides,
)
from cobre_bridge.newave.id_map import NewaveIdMap

_LOG = logging.getLogger(__name__)

# The point / PRODT / integrated productivity functions now live in the
# presentation-free ``cobre_bridge.core.productivity`` domain module. These aliases
# keep hydro.py's many internal callers (and tests that import the private
# names) working; external modules import the public names directly.
_compute_productivity = compute_productivity
_equivalent_productivity = equivalent_productivity
_compute_integrated_productivity = integrated_productivity
_stored_energy_productivity = stored_energy_productivity

# --- FPHA (hydro production function) emission ---------------------------------
#
# When the source model evaluates generation via FPHA (``dger.dat`` line 96,
# ``funcao_producao_uhe == 0``; see :attr:`NewaveCase.fpha_enabled`), reservoir plants
# are emitted with cobre's ``model: "fpha"`` so cobre fits the production function from
# the plant geometry + tailrace families instead of the bridge pre-baking a single
# constant productivity.

# The source model's tratamento-fpha distance method carries only a tolerance; cobre
# also requires a sample count for the mean-squared-distance estimate. The source model
# does not specify one, so we supply a reasonable default.
_FPHA_DISTANCE_N_SAMPLES = 100


def _parse_fpha_plane_reduction(case: NewaveCase) -> dict | None:
    """Parse ``tratamento-fpha`` into a cobre ``fpha_plane_reduction`` block.

    The source model's treatment file carries one active line (``&``-prefixed lines are
    comments) selecting either the angle or the distance plane-reduction method::

        HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES-ANGULO-PADRAO; 1.0
        HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES-DISTANCIA-PADRAO; 0.002

    Returns ``{"method": "angle", "tolerance_deg": v}`` or
    ``{"method": "distance", "tolerance_pct": v, "n_samples": N}``, or ``None``
    when no treatment file is present or no method line is active.
    """
    path = case.files.tratamento_fpha
    if path is None:
        return None

    methods: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("&"):
            continue
        parts = [part.strip() for part in line.split(";")]
        if len(parts) < 2:
            continue
        token = parts[0].upper()
        try:
            value = float(parts[1])
        except ValueError:
            continue
        if token.endswith("ANGULO-PADRAO"):
            methods.append({"method": "angle", "tolerance_deg": value})
        elif token.endswith("DISTANCIA-PADRAO"):
            methods.append(
                {
                    "method": "distance",
                    "tolerance_pct": value,
                    "n_samples": _FPHA_DISTANCE_N_SAMPLES,
                }
            )

    if not methods:
        return None
    if len(methods) > 1:
        emit(
            Diagnostic(
                code="fpha-plane-reduction-multiple-methods",
                severity=Severity.WARNING,
                category="Production model",
                title="Multiple active FPHA plane-reduction methods",
                summary=(
                    f"{path.name} has multiple active FPHA plane-reduction "
                    f"methods; using the first ({methods[0]['method']})."
                ),
            ),
            logger=_LOG,
        )
    return methods[0]


def compute_prodt_sin_mean(case: NewaveCase) -> float:
    """Return ``PROD_MEDIA_SIN`` = mean PRODT over existing plants (the source model).

    This is the system-average productivity the source model uses to convert the PENALID
    R$/MWh penalties to the flow/volume domain (pmo.dat "PENALIDADE POR VIOLACAO
    DE VAZAO MINIMA" etc.). Two conventions matter and are matched here:

    - the per-plant productivity is **PRODT** (:func:`_equivalent_productivity`),
      not the 65%-reference point value used for energy; and
    - the mean is over **all** existing non-fictitious plants **including** those
      with zero PRODT — no ``ρ > 0`` filter and no fictitious-cascade fold-in.

    On the example case this returns ≈ 0.6299, matching pmo's penalty-implied ρ
    (0.6294) to 0.06%; the legacy 65%-reference mean was 0.656 (~4% high).
    """
    cadastro = _apply_permanent_overrides(case.hidr.cadastro, case)

    existing = case.active_hydros

    prodt: list[float] = []
    for _, row in existing.iterrows():
        code = int(row["codigo_usina"])
        if code in cadastro.index:
            prodt.append(_equivalent_productivity(cadastro.loc[code]))
    return sum(prodt) / len(prodt) if prodt else 1.0


def compute_per_stage_prodt_sin_mean(case: NewaveCase) -> list[float]:
    """Return ``PROD_MEDIA_SIN[s]`` = mean PRODT per stage (CFUGA/CMONT-driven).

    Same plant set and convention as :func:`compute_prodt_sin_mean` (mean PRODT
    over all existing plants, zeros included, no FICT fold), but **per stage**:
    the handful of plants carrying MODIF.DAT CFUGA/CMONT temporal overrides make
    the SIN mean drift slightly per configuration — ≈0.15% on the example case,
    the decimal wiggle pmo.dat shows on the VAZMIN / TURBMN / TURBMX penalties
    (820.53…821.78). PRODT ignores VOLREF_SAZ, so only CFUGA/CMONT move it.

    Returns ``[]`` when the horizon can't be read (mocked-pipeline unit tests).
    """
    total_stages = _total_study_stages(case)
    if total_stages <= 0:
        return []

    cadastro = _apply_permanent_overrides(case.hidr.cadastro, case)
    existing = case.active_hydros
    codes = [int(r["codigo_usina"]) for _, r in existing.iterrows()]

    temporal = _extract_temporal_overrides(case, codes)
    drop_by_plant = {
        code: [o for o in ov if o["type"] in ("CFUGA", "CMONT")]
        for code, ov in temporal.items()
    }

    stage_sum = [0.0] * total_stages
    count = 0
    for code in codes:
        if code not in cadastro.index:
            continue
        hreg = cadastro.loc[code]
        series = _per_stage_equivalent_productivities(
            hreg,
            _equivalent_productivity(hreg),
            drop_by_plant.get(code, []),
            case,
            total_stages,
        )
        for s in range(total_stages):
            stage_sum[s] += series[s]
        count += 1
    return [v / count for v in stage_sum] if count else []


def _fpha_computed_config(hreg: pd.Series) -> dict:
    """``fpha_config`` for a computed-FPHA plant, with the source model's fitting
    window.

    The source model fits the FPHA volume axis over the *operative* storage range, which
    depends on the regulation type:

    - ``tipo_regulacao == "M"`` (monthly reservoir) → multi-volume over
      ``[volume_minimo, volume_maximo]``.
    - ``"D"`` / ``"S"`` (daily / run-of-river) → single-volume at
      ``volume_referencia`` (the reference operating volume).

    Passing this as ``fitting_window`` makes cobre fit the same volume grid. Without it
    cobre falls back to the full geometry span, which fits the "D" plants that carry a
    cadastro storage range (e.g. ITAIPU, JIRAU) as multi-volume even though the source
    model collapses them to a single volume.
    """
    reg = str(hreg["tipo_regulacao"]).strip()
    if reg == "M":
        vlo = float(hreg["volume_minimo"])
        vhi = float(hreg["volume_maximo"])
    else:
        vref = float(hreg["volume_referencia"])
        vlo = vhi = vref
    return {
        "source": "computed",
        "fitting_window": {"volume_min_hm3": vlo, "volume_max_hm3": vhi},
    }


def convert_production_models(case: NewaveCase, id_map: NewaveIdMap) -> dict:
    """Build ``hydro_production_models.json`` with per-hydro model selection.

    - **Non-FPHA cases (and run-of-river plants in FPHA cases):** a single
      ``stage_ranges`` entry with ``model: "constant_productivity"`` and no
      numeric value; ``productivity_mw_per_m3s`` is supplied per-(hydro, stage)
      in ``hydro_energy_productivity.parquet`` (see
      :func:`convert_hydro_energy_productivity`). Cross-file validation in cobre
      rejects double-supply (JSON + parquet) and coverage gaps, so keeping
      productivity strictly in the parquet eliminates the conflict surface.
    - **FPHA reservoirs** (``dger`` ``funcao_producao_uhe == 0`` and the plant
      has storage swing — see :func:`fpha_eligible_codes`): ``model: "fpha"``
      with ``fpha_config: {source: "computed"}``; cobre fits the production
      function from geometry + tailrace families, so no parquet productivity is
      emitted for these plants.

    **Reference volume.** In FPHA cases, ``reference_volume`` (V_ref) sets the FPHA
    backwater / tailrace level and the ρ_eq reference. The source model's per-month
    seasonal reference comes from ``volref_saz.dat``: a plant with a seasonal row is
    emitted in ``seasonal`` mode with one absolute ``reference_volume`` per season
    (``clamp(vmin + useful, vmin, vmax)``); a plant without one falls back to
    ``percentile 0.65`` (= cobre's default, the source model's altura_65). This is
    emitted for FPHA reservoirs **and** for any non-FPHA plant that has a seasonal row,
    because cobre reads a plant's *downstream* ``reference_volume`` (via
    ``downstream_id``) to set that plant's backwater — so each plant must be a correct
    reference for its upstream FPHA neighbour.

    When FPHA is active, a file-level ``fpha_plane_reduction`` block parsed from
    ``tratamento-fpha`` is added (see :func:`_parse_fpha_plane_reduction`).

    Parameters
    ----------
    case:
        Parsed the source model case.
    id_map:
        Pre-built entity ID map used to translate the source model plant codes to
        0-based Cobre hydro IDs.

    Returns
    -------
    dict
        A dict with a ``"production_models"`` key ready to serialise as
        ``system/hydro_production_models.json``.
    """
    existing = case.active_hydros
    confhd_codes = [int(r["codigo_usina"]) for _, r in existing.iterrows()]
    fpha_codes = fpha_eligible_codes(case)

    # Seasonal reference volumes (volref_saz.dat) drive the FPHA reference volume V_ref
    # — only relevant in FPHA cases. V_ref feeds both a plant's own rho_eq AND, via
    # cobre's downstream lookup (`downstream_id`), the backwater / tailrace level of the
    # plant ABOVE it. So we emit reference_volume for every plant that has a volref_saz
    # row (FPHA or not), so each plant is a correct backwater reference for its upstream
    # FPHA neighbour. Plants without a row fall through to cobre's 0.65 default fraction
    # (= the source model's altura_65).
    seasonal_volref = _read_volref_saz(case) if case.fpha_enabled else {}
    # Load the cadastro whenever per-plant volume info is needed — for the FPHA
    # fitting window (every FPHA plant) or the seasonal reference volumes.
    cadastro = (
        _apply_permanent_overrides(case.hidr.cadastro, case)
        if (seasonal_volref or fpha_codes)
        else None
    )

    production_models: list[dict] = []
    for newave_code in confhd_codes:
        try:
            hydro_id = id_map.hydro_id(newave_code)
        except KeyError:
            continue

        is_fpha = newave_code in fpha_codes
        monthly_useful = seasonal_volref.get(newave_code)
        hreg = (
            cadastro.loc[newave_code]
            if cadastro is not None and newave_code in cadastro.index
            else None
        )
        fpha_config = (
            _fpha_computed_config(hreg) if (is_fpha and hreg is not None) else None
        )

        if monthly_useful and hreg is not None:
            production_models.append(
                {
                    "hydro_id": hydro_id,
                    "selection_mode": "seasonal",
                    # All 12 seasons are listed below, so the default is never
                    # consulted; a config-less model keeps it valid.
                    "default_model": "constant_productivity",
                    "seasons": _seasonal_reference_volume(
                        monthly_useful,
                        float(hreg["volume_minimo"]),
                        float(hreg["volume_maximo"]),
                        is_fpha=is_fpha,
                        fpha_config=fpha_config,
                    ),
                }
            )
        elif is_fpha:
            # FPHA reservoir with no seasonal reference: fall back to V_65 (percentile
            # 0.65 = the source model's altura_65, and cobre's own default).
            production_models.append(
                {
                    "hydro_id": hydro_id,
                    "selection_mode": "stage_ranges",
                    "stage_ranges": [
                        {
                            "start_stage_id": 0,
                            "end_stage_id": None,
                            "model": "fpha",
                            "fpha_config": fpha_config or {"source": "computed"},
                            "reference_volume": {"percentile": 0.65},
                        }
                    ],
                }
            )
        else:
            production_models.append(
                {
                    "hydro_id": hydro_id,
                    "selection_mode": "stage_ranges",
                    "stage_ranges": [
                        {
                            "start_stage_id": 0,
                            "end_stage_id": None,
                            "model": "constant_productivity",
                        }
                    ],
                }
            )

    production_models.sort(key=lambda m: m["hydro_id"])

    result: dict = {
        "$schema": cobre_schemas.schema_url_for("system/hydro_production_models.json")
    }
    # File-level FPHA plane reduction (tratamento-fpha), applied to every FPHA
    # plant. Only meaningful when there is at least one FPHA plant.
    if fpha_codes:
        plane_reduction = _parse_fpha_plane_reduction(case)
        if plane_reduction is not None:
            result["fpha_plane_reduction"] = plane_reduction
    result["production_models"] = production_models
    return result


def _total_study_stages(case: NewaveCase) -> int:
    """Return the total number of stages in the study (including post-study)."""
    return case.horizon.total_stages


def _per_stage_integrated_productivities(
    hreg: pd.Series,
    base_integrated: float,
    drop_overrides: list[dict],
    case: NewaveCase,
    total_stages: int,
) -> list[float]:
    """Per-stage integrated productivity with CFUGA/CMONT step-function awareness.

    Recomputes the *integrated* productivity (volmin_volmax average) at each
    stage where canal_fuga or cmont state changes; stages with no active override
    return *base_integrated*. The per-stage ``(CFUGA, CMONT)`` step-function sweep
    is shared with :func:`_per_stage_equivalent_productivities` via
    :func:`_per_stage_drop_overrides` (the integrated variant differs only in
    applying :func:`_compute_integrated_productivity` instead of the point ρ).
    """
    if not drop_overrides:
        return [base_integrated] * total_stages

    drops = _per_stage_drop_overrides(drop_overrides, case, total_stages)
    return [
        base_integrated
        if cfuga is None and cmont is None
        else _stored_energy_productivity(
            hreg, canal_fuga_override=cfuga, cmont_override=cmont
        )
        for cfuga, cmont in drops
    ]


def compute_per_stage_own_integrated_productivities(
    case: NewaveCase,
) -> dict[int, list[float]]:
    """Return ``{plant_code: [own integrated ρ per stage]}`` for every existing plant.

    Companion to :func:`compute_per_stage_own_productivities` but with the
    EARM convention: ρ is the volume-integrated productivity (matching
    the source model's ``produtibilidade_equivalente_volmin_volmax``), not the point
    productivity at v_65.  Used by VminOP to override the ``rho_acum_h{id}`` scalar
    parameter so the constraint coefficient matches the source model's stored-energy
    accounting rather than the LP's gen = ρ·Q point coefficient.

    CFUGA/CMONT temporal overrides shift the integrand at every stage from the
    override's effective stage forward; FICT-cascade contribution is folded into the
    upstream real plant's own value so cascade traversal in the source-model-code space
    matches the rewired ``downstream_id`` in ``hydros.json``.
    """
    total_stages = _total_study_stages(case)
    if total_stages <= 0:
        return {}

    cadastro = _apply_permanent_overrides(case.hidr.cadastro, case)

    confhd_df = case.confhd.usinas
    existing = case.active_hydros
    confhd_codes = [int(r["codigo_usina"]) for _, r in existing.iterrows()]

    temporal_overrides = _extract_temporal_overrides(case, confhd_codes)
    plants_with_drop_overrides = {
        code: [o for o in overrides if o["type"] in ("CFUGA", "CMONT")]
        for code, overrides in temporal_overrides.items()
        if any(o["type"] in ("CFUGA", "CMONT") for o in overrides)
    }

    from cobre_bridge.newave.converters.fict_cascade import resolve_cascade

    fict_cascade = resolve_cascade(confhd_df, cadastro)

    result: dict[int, list[float]] = {}
    for plant_code in confhd_codes:
        if plant_code not in cadastro.index:
            continue
        hreg = cadastro.loc[plant_code]
        base = _stored_energy_productivity(hreg)
        resolution = fict_cascade.get(plant_code)
        fict_extra = resolution.fict_rho_sum if resolution is not None else 0.0
        overrides = plants_with_drop_overrides.get(plant_code, [])
        per_stage = _per_stage_integrated_productivities(
            hreg, base, overrides, case, total_stages
        )
        result[plant_code] = [v + fict_extra for v in per_stage]
    return result


def _per_stage_equivalent_productivities(
    hreg: pd.Series,
    base_prodt: float,
    drop_overrides: list[dict],
    case: NewaveCase,
    total_stages: int,
) -> list[float]:
    """Per-stage PRODT, applying CFUGA/CMONT temporal overrides.

    PRODT (vol_min→vol_max equivalent) is independent of the operating volume,
    so — unlike :func:`_per_stage_productivities` — VOLREF_SAZ is *not* applied;
    the only per-stage driver is the CFUGA/CMONT tailrace/forebay. Plants with no
    such override get a flat list at ``base_prodt``.
    """
    if not drop_overrides:
        return [base_prodt] * total_stages
    drops = _per_stage_drop_overrides(drop_overrides, case, total_stages)
    return [
        base_prodt
        if cfuga is None and cmont is None
        else _equivalent_productivity(
            hreg, canal_fuga_override=cfuga, cmont_override=cmont
        )
        for cfuga, cmont in drops
    ]


def _per_stage_productivities(
    hreg: pd.Series,
    base_productivity: float,
    drop_overrides: list[dict],
    case: NewaveCase,
    total_stages: int,
    seasonal_volref_by_month: dict[int, float] | None = None,
) -> list[float]:
    """Build per-stage productivity values from seasonal volref + CFUGA/CMONT.

    For each stage *s* (0-based), the productivity is computed as:

    1. Determine calendar month ``m = ((start_month − 1 + s) mod 12) + 1``.
    2. Pick reference useful volume: ``seasonal_volref_by_month[m]`` when
       present and positive, else fall back to the source model's altura_65 /
       volume_referencia convention (i.e. use *base_productivity*).
    3. Apply any active CFUGA/CMONT temporal override (step-function from its
       stage of effect forward until the next event of the same type).

    Returns a list of length *total_stages*.  If neither seasonal nor temporal
    overrides apply for a stage, that stage's value equals *base_productivity*.
    """
    has_seasonal = bool(seasonal_volref_by_month)
    if not drop_overrides and not has_seasonal:
        return [base_productivity] * total_stages

    start_month = int(case.dger.mes_inicio_estudo)
    drops = _per_stage_drop_overrides(drop_overrides, case, total_stages)
    seasonal = seasonal_volref_by_month or {}

    values: list[float] = []
    for stage_id in range(total_stages):
        active_cfuga, active_cmont = drops[stage_id]
        calendar_month = ((start_month - 1 + stage_id) % 12) + 1
        vol_useful = seasonal.get(calendar_month)

        if vol_useful is None and active_cfuga is None and active_cmont is None:
            values.append(base_productivity)
        else:
            values.append(
                _compute_productivity(
                    hreg,
                    canal_fuga_override=active_cfuga,
                    cmont_override=active_cmont,
                    useful_volume_override=vol_useful,
                )
            )

    # Post-study tail continues the seasonal cycle: VOLREF_SAZ is an always- seasonal
    # monthly reference volume, and CFUGA/CMONT repeat seasonally when
    # ``sazonaliza_cfuga_cmont == 1`` (handled above). The source model re-applies these
    # seasonal patterns every year, including post-study — only the quantities without a
    # seasonalize flag (outflow / turbined bounds) freeze.
    return values


def convert_hydro_energy_productivity(
    case: NewaveCase, id_map: NewaveIdMap
) -> pa.Table:
    """Build the per-(hydro, stage) equivalent-productivity (ρ_eq) parquet table.

    **Every** plant gets an ``equivalent_productivity_mw_per_m3s`` value here,
    FPHA and non-FPHA alike. `convert_production_models` emits no numeric
    productivity, so this parquet is the single ρ_eq source. Plants without
    CFUGA/CMONT/seasonal overrides emit one ``stage_id = NULL`` default row;
    plants with overrides emit one row per study stage.

    FPHA plants are **not** excluded: cobre's energy-conversion build derives
    their ρ_eq from this parquet override, because ``build_energy_and_templates``
    feeds the alternative "VHA geometry + ρ_esp" derivation path an *empty*
    geometry map — so the parquet override is the only working source. (The FPHA
    production function φ itself is fit separately by cobre from geometry +
    tailrace + efficiency.) Excluding FPHA plants makes cobre fail at load with
    "FPHA hydro '…' cannot derive ρ_eq".

    ``reference_outflow_m3s`` and ``specific_productivity_mw_per_m3s_per_m`` are
    left NULL. The reference volume V_ref is declared in
    ``hydro_production_models.json`` — see :func:`convert_production_models`.
    """
    cadastro = _apply_permanent_overrides(case.hidr.cadastro, case)

    confhd_df = case.confhd.usinas
    existing = case.active_hydros
    confhd_codes = [int(r["codigo_usina"]) for _, r in existing.iterrows()]

    temporal_overrides = _extract_temporal_overrides(case, confhd_codes)
    plants_with_drop_overrides = {
        code: [o for o in overrides if o["type"] in ("CFUGA", "CMONT")]
        for code, overrides in temporal_overrides.items()
        if any(o["type"] in ("CFUGA", "CMONT") for o in overrides)
    }

    seasonal_volref = _read_volref_saz(case)

    needs_per_stage = bool(plants_with_drop_overrides) or bool(seasonal_volref)
    total_stages = _total_study_stages(case) if needs_per_stage else 0

    # FICT-cascade: when a real plant's energy-cascade traverses fictitious plants,
    # fold those FICTs' ρ_eq into the upstream real plant's own ρ_eq so cobre's
    # per-plant cascade sum (from ``hydro_energy_productivity.parquet`` plus the
    # rewired ``downstream_id``) reproduces the source model's
    # ``produtibilidade_acumulada_calculo_earm``.  FICT plants have ρ_esp = 0 in the
    # bundled cases, so this is numerically a no-op there — a structural fix, robust
    # to non-zero FICT productivities.
    from cobre_bridge.newave.converters.fict_cascade import resolve_cascade

    fict_cascade = resolve_cascade(confhd_df, cadastro)

    hydro_ids: list[int] = []
    stage_ids: list[int | None] = []
    equiv_prods: list[float] = []

    for newave_code in sorted(confhd_codes):
        if newave_code not in cadastro.index:
            continue
        try:
            hydro_id = id_map.hydro_id(newave_code)
        except KeyError:
            continue
        hreg = cadastro.loc[newave_code]
        legacy_base = _compute_productivity(hreg)
        resolution = fict_cascade.get(newave_code)
        fict_extra = resolution.fict_rho_sum if resolution is not None else 0.0
        overrides = plants_with_drop_overrides.get(newave_code, [])
        plant_seasonal = seasonal_volref.get(newave_code)

        if not overrides and not plant_seasonal:
            hydro_ids.append(hydro_id)
            stage_ids.append(None)
            equiv_prods.append(legacy_base + fict_extra)
        else:
            per_stage = _per_stage_productivities(
                hreg,
                legacy_base,
                overrides,
                case,
                total_stages,
                seasonal_volref_by_month=plant_seasonal,
            )
            for stage_id, value in enumerate(per_stage):
                hydro_ids.append(hydro_id)
                stage_ids.append(stage_id)
                equiv_prods.append(value + fict_extra)

    nulls = [None] * len(hydro_ids)
    return pa.table(
        {
            "hydro_id": pa.array(hydro_ids, type=pa.int32()),
            "stage_id": pa.array(stage_ids, type=pa.int32()),
            "equivalent_productivity_mw_per_m3s": pa.array(
                equiv_prods, type=pa.float64()
            ),
            "reference_outflow_m3s": pa.array(nulls, type=pa.float64()),
            "specific_productivity_mw_per_m3s_per_m": pa.array(
                nulls, type=pa.float64()
            ),
        }
    )


def compute_per_stage_own_productivities(
    case: NewaveCase,
) -> dict[int, list[float]]:
    """Return ``{plant_code: [own ρ_eq per stage]}`` for every existing plant.

    Per-stage own productivity reflects MODIF.DAT CFUGA / CMONT temporal
    overrides — for stages before any override the value is the base
    polynomial-integrated productivity; for later stages it picks up the
    effective tailrace / forebay overrides as they take effect. Plants
    without any temporal overrides return a flat list of length
    ``total_stages``.

    Used by the VminOP RHS calculation so that the absolute bound
    ``(pct/100) × useful + dead`` is computed with the **same** per-stage
    ρ_acum that cobre uses to evaluate the LHS at solve time — otherwise
    the constraint silently drifts at every stage where overrides apply
    or for any plant upstream of an overridden plant in the cascade.

    Keys are the source model plant codes (not Cobre ids) since cascade traversal in
    ``compute_accumulated_productivities`` works in the source-model-code space.
    """
    total_stages = _total_study_stages(case)
    if total_stages <= 0:
        return {}

    cadastro = _apply_permanent_overrides(case.hidr.cadastro, case)

    confhd_df = case.confhd.usinas
    existing = case.active_hydros
    confhd_codes = [int(r["codigo_usina"]) for _, r in existing.iterrows()]

    temporal_overrides = _extract_temporal_overrides(case, confhd_codes)
    plants_with_drop_overrides = {
        code: [o for o in overrides if o["type"] in ("CFUGA", "CMONT")]
        for code, overrides in temporal_overrides.items()
        if any(o["type"] in ("CFUGA", "CMONT") for o in overrides)
    }

    seasonal_volref = _read_volref_saz(case)

    # FICT-cascade fold-in: per-stage ρ_eq must already include any FICT
    # contribution so that the per-stage ρ_acum used by VminOP and EARM
    # accounting matches the topology rewired into ``hydros.json``.
    from cobre_bridge.newave.converters.fict_cascade import resolve_cascade

    fict_cascade = resolve_cascade(confhd_df, cadastro)

    result: dict[int, list[float]] = {}
    for plant_code in confhd_codes:
        if plant_code not in cadastro.index:
            continue
        hreg = cadastro.loc[plant_code]
        legacy_base = _compute_productivity(hreg)
        resolution = fict_cascade.get(plant_code)
        fict_extra = resolution.fict_rho_sum if resolution is not None else 0.0
        overrides = plants_with_drop_overrides.get(plant_code, [])
        plant_seasonal = seasonal_volref.get(plant_code)
        per_stage = _per_stage_productivities(
            hreg,
            legacy_base,
            overrides,
            case,
            total_stages,
            seasonal_volref_by_month=plant_seasonal,
        )
        result[plant_code] = [v + fict_extra for v in per_stage]
    return result


def compute_base_productivities(
    case: NewaveCase, id_map: NewaveIdMap
) -> dict[int, float]:
    """Return ``{hydro_id: base_productivity_mw_per_m3s}`` for every hydro.

    The base productivity is the value `_compute_productivity` returns with no
    CFUGA/CMONT overrides applied — i.e. the productivity used when the case
    has no temporal overrides for that plant. Consumers that previously read
    ``hydros_dict[i]["generation"]["productivity_mw_per_m3s"]`` should call
    this instead now that productivity has moved out of `hydros.json`.
    """
    cadastro = _apply_permanent_overrides(case.hidr.cadastro, case)

    confhd_df = case.confhd.usinas
    existing = case.active_hydros

    # FICT-cascade fold-in — keep this in lockstep with the other productivity
    # helpers so every downstream consumer sees the same effective ρ_eq.
    from cobre_bridge.newave.converters.fict_cascade import resolve_cascade

    fict_cascade = resolve_cascade(confhd_df, cadastro)

    result: dict[int, float] = {}
    for _, row in existing.iterrows():
        newave_code = int(row["codigo_usina"])
        if newave_code not in cadastro.index:
            continue
        try:
            hydro_id = id_map.hydro_id(newave_code)
        except KeyError:
            continue
        base = _compute_productivity(cadastro.loc[newave_code])
        resolution = fict_cascade.get(newave_code)
        if resolution is not None:
            base += resolution.fict_rho_sum
        result[hydro_id] = base
    return result
