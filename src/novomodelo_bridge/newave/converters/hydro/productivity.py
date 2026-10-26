"""Hydro productivity: per-stage/per-plant equivalent-productivity (rho_eq)
machinery, the production-model JSON, and the FPHA plane-reduction /
computed-config helpers.

All point / PRODT / integrated productivity math routes through the
presentation-free :mod:`novomodelo_bridge.core.productivity` domain module; this module
holds only the source-model-specific orchestration (temporal overrides,
seasonal reference volumes, FICT-cascade fold-in) around that math.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet

import pandas as pd
import pyarrow as pa

from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    emit,
)
from novomodelo_bridge.core.pandas_utils import is_na
from novomodelo_bridge.core.productivity import (
    KTURB_BY_TIPO_TURBINA,
    apply_hydraulic_loss,
    compute_productivity,
    equivalent_productivity,
    evaluate_cota,
    fpha_spillage_fit_max_m3s,
    integrated_productivity,
    mean_cota,
    stored_energy_productivity,
)
from novomodelo_bridge.core.source_text import read_source_text
from novomodelo_bridge.core.tailrace import (
    TailraceFamily,
    tailrace_families_from_polinjus,
    tailrace_level_m,
)
from novomodelo_bridge.core.turbine import MachineSet, power_limited_flow, settled_flow
from novomodelo_bridge.newave.case import NewaveCase
from novomodelo_bridge.newave.converters.hydro.geometry import (
    _expansion_configs,
    fpha_eligible_codes,
    operating_storage_range,
)
from novomodelo_bridge.newave.converters.hydro.overrides import (
    _apply_permanent_overrides,
    _extract_temporal_overrides,
    _per_stage_drop_overrides,
)
from novomodelo_bridge.newave.converters.hydro.reference_volumes import (
    fpha_reference_volumes,
    seasonal_useful_volumes,
)
from novomodelo_bridge.newave.converters.stochastic import natural_monthly_means
from novomodelo_bridge.newave.horizon import stage_dates_for
from novomodelo_bridge.newave.id_map import NewaveIdMap
from novomodelo_bridge.novomodelo import schemas as novomodelo_schemas

_LOG = logging.getLogger(__name__)

# The point / PRODT / integrated productivity functions now live in the
# presentation-free ``novomodelo_bridge.core.productivity`` domain module. These aliases
# keep hydro.py's many internal callers (and tests that import the private
# names) working; external modules import the public names directly.
_compute_productivity = compute_productivity
_equivalent_productivity = equivalent_productivity
_compute_integrated_productivity = integrated_productivity

# --- FPHA (hydro production function) emission ---------------------------------
#
# When the source model evaluates generation via FPHA (``dger.dat`` line 96,
# ``funcao_producao_uhe == 0``; see :attr:`NewaveCase.fpha_enabled`), reservoir plants
# are emitted with novomodelo's ``model: "fpha"`` so novomodelo fits the production function from
# the plant geometry + tailrace families instead of the bridge pre-baking a single
# constant productivity.

# The source model's tratamento-fpha distance method carries only a tolerance; novomodelo
# also requires a sample count for the mean-squared-distance estimate. The source model
# does not specify one, so we supply a reasonable default.
_FPHA_DISTANCE_N_SAMPLES = 100

#: The source model prints its turbined fit range to 0.1 m³/s.
_FIT_FLOW_TOLERANCE_M3S = 1.0e-3
#: A safety bound only: the iteration converges in a few steps on real decks.
_FIT_FLOW_MAX_ITERATIONS = 100


def _parse_fpha_plane_reduction(case: NewaveCase) -> dict | None:
    """Parse ``tratamento-fpha`` into a novomodelo ``fpha_plane_reduction`` block.

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
    for raw in read_source_text(path).splitlines():
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


def _fpha_computed_config(
    hreg: pd.Series,
    spillage_fit_max_m3s: float | None = None,
    turbined_fit_max_m3s: float | None = None,
) -> dict:
    """``fpha_config`` for a computed-FPHA plant, with the source model's fitting
    window and, when given, its spillage fit range
    (:func:`~novomodelo_bridge.core.productivity.fpha_spillage_fit_max_m3s`) and
    turbined fit range (:func:`_turbined_fit_max`).

    The source model fits the FPHA volume axis over the plant's
    :func:`~novomodelo_bridge.newave.converters.hydro.geometry.operating_storage_range`:
    multi-volume over ``[volume_minimo, volume_maximo]`` for a monthly reservoir
    ('M'), single-volume at ``volume_referencia`` for a daily or run-of-river
    plant ('D' / 'S'). Passing it as ``fitting_window`` makes novomodelo fit the same
    volume grid.
    """
    vlo, vhi = operating_storage_range(hreg)
    config: dict = {
        "source": "computed",
        "fitting_window": {"volume_min_hm3": vlo, "volume_max_hm3": vhi},
    }
    if spillage_fit_max_m3s is not None:
        config["spillage_fit_max_m3s"] = spillage_fit_max_m3s
    if turbined_fit_max_m3s is not None:
        config["turbined_fit_max_m3s"] = turbined_fit_max_m3s
    return config


def _turbined_fit_max(
    hreg: pd.Series,
    families: Sequence[TailraceFamily],
    downstream_level_m: float | None,
    reference_volume_hm3: float,
) -> tuple[float | None, bool]:
    """The source model's FPHA turbined-flow range (m³/s) at a stage whose FPHA
    reference volume is *reference_volume_hm3*, and whether its fixed point
    settled.

    The range is the machine sets'
    :func:`~novomodelo_bridge.core.turbine.power_limited_flow`, with no TEIF/IP
    derating. The turbine head runs from the reference volume down to the
    tailrace at the turbined flow, on *families* at *downstream_level_m*
    (:func:`~novomodelo_bridge.core.turbine.settled_flow`), or down to
    ``canal_fuga_medio`` when *families* is empty. The generator head runs from
    the mean forebay over the plant's
    :func:`~novomodelo_bridge.newave.converters.hydro.geometry.operating_storage_range`
    down to ``canal_fuga_medio``. The range is ``None`` when
    ``produtibilidade_especifica`` or the net head is not positive.
    """
    rho_esp_raw = hreg.get("produtibilidade_especifica")
    cf_raw = hreg.get("canal_fuga_medio")
    if is_na(rho_esp_raw) or is_na(cf_raw):
        return None, True
    rho_esp = float(rho_esp_raw)
    if rho_esp <= 0.0:
        return None, True
    coeffs = [float(hreg[f"a{i}_volume_cota"]) for i in range(5)]
    forebay_m = evaluate_cota(coeffs, reference_volume_hm3)
    tipo_perda = int(hreg.get("tipo_perda", 0) or 0)
    perdas = float(hreg.get("perdas", 0.0) or 0.0)
    generator_head = apply_hydraulic_loss(
        mean_cota(coeffs, *operating_storage_range(hreg)) - float(cf_raw),
        tipo_perda,
        perdas,
    )
    sets = [
        MachineSet(
            units=int(hreg[f"maquinas_conjunto_{i}"]),
            flow_m3s=float(hreg[f"vazao_nominal_conjunto_{i}"]),
            head_m=float(hreg[f"queda_nominal_conjunto_{i}"]),
            power_mw=float(hreg[f"potencia_nominal_conjunto_{i}"]),
        )
        for i in range(1, int(hreg["numero_conjuntos_maquinas"]) + 1)
    ]
    exponent = KTURB_BY_TIPO_TURBINA.get(int(hreg.get("tipo_turbina", 0) or 0), 0.5)

    def flow_at(tailrace_m: float) -> float:
        return power_limited_flow(
            apply_hydraulic_loss(forebay_m - tailrace_m, tipo_perda, perdas),
            sets,
            generator_head_m=generator_head,
            specific_productivity=rho_esp,
            exponent=exponent,
        )

    flow, converged = flow_at(float(cf_raw)), True
    if families:
        flow, converged = settled_flow(
            lambda q: flow_at(tailrace_level_m(families, q, downstream_level_m)),
            flow,
            tolerance_m3s=_FIT_FLOW_TOLERANCE_M3S,
            max_iterations=_FIT_FLOW_MAX_ITERATIONS,
        )
    return (flow if flow > 0.0 else None), converged


def _downstream_level_m(
    cadastro: pd.DataFrame, hreg: pd.Series, confhd_codes: AbstractSet[int]
) -> float | None:
    """Forebay level (m) of the plant hidr names downstream of *hreg*, at the
    low end of its
    :func:`~novomodelo_bridge.newave.converters.hydro.geometry.operating_storage_range`;
    ``None`` when hidr names none or confhd does not list it."""
    downstream = hreg.get("codigo_usina_jusante")
    if is_na(downstream) or int(downstream) not in confhd_codes:
        return None
    if int(downstream) not in cadastro.index:
        return None
    below = cadastro.loc[int(downstream)]
    return evaluate_cota(
        [float(below[f"a{i}_volume_cota"]) for i in range(5)],
        operating_storage_range(below)[0],
    )


def _turbined_fit_maxima(
    case: NewaveCase,
    cadastro: pd.DataFrame,
    fpha_codes: AbstractSet[int],
    references: Mapping[int, Sequence[float]],
) -> dict[int, list[float | None]]:
    """``{plant code: turbined fit range (m³/s) of each study stage}`` for each
    FPHA plant with :func:`fpha_reference_volumes`, from
    :func:`_turbined_fit_max` at the stage's reference volume.

    An expanding plant is rated on the machines ``hydros.json`` declares. The
    tailrace family is read at :func:`_downstream_level_m`.
    """
    polinjus = case.polinjus
    families = (
        {}
        if polinjus is None
        else tailrace_families_from_polinjus(
            polinjus.hidreletrica_curvajusante(df=True),
            polinjus.hidreletrica_curvajusante_polinomio_segmento(df=True),
        )
    )
    confhd_codes = {int(code) for code in case.confhd.usinas["codigo_usina"]}
    expansion = _expansion_configs(case, cadastro)
    maxima: dict[int, list[float | None]] = {}
    unconverged: list[list[object]] = []
    for _, row in case.active_hydros.iterrows():
        code = int(row["codigo_usina"])
        if code not in fpha_codes or code not in cadastro.index:
            continue
        stage_volumes = references.get(code)
        if not stage_volumes:
            continue
        hreg = cadastro.loc[code]
        config = expansion.get(code)
        declared = hreg if config is None else config.declared_hreg(hreg)
        downstream_level = _downstream_level_m(cadastro, hreg, confhd_codes)
        by_volume: dict[float, float | None] = {}
        settled = True
        for volume in dict.fromkeys(stage_volumes):
            by_volume[volume], converged = _turbined_fit_max(
                declared, families.get(code, ()), downstream_level, volume
            )
            settled = settled and converged
        maxima[code] = [by_volume[volume] for volume in stage_volumes]
        if not settled:
            unconverged.append([code, str(hreg["nome_usina"]).strip()])
    if unconverged:
        emit(
            Diagnostic(
                code="fpha-turbined-fit-range-unconverged",
                severity=Severity.WARNING,
                category="Production model",
                title=(
                    f"FPHA turbined fit range did not converge ({len(unconverged)})"
                ),
                summary=(
                    "The tailrace-dependent turbined fit range of the listed "
                    f"plants did not settle within {_FIT_FLOW_MAX_ITERATIONS} "
                    "iterations; the last value is used."
                ),
                table=DiagnosticTable(columns=["Code", "Name"], rows=unconverged),
            ),
            logger=_LOG,
        )
    return maxima


def _spillage_fit_maxima(
    confhd: pd.DataFrame,
    fpha_codes: AbstractSet[int],
    natural_monthly_means: Mapping[int, Sequence[float]],
) -> dict[int, float]:
    """``{plant code: spillage fit range (m³/s)}`` for each FPHA plant whose
    ``confhd`` posto has one."""
    maxima: dict[int, float] = {}
    for _, row in confhd.iterrows():
        code = int(row["codigo_usina"])
        if code not in fpha_codes or pd.isna(row["posto"]):
            continue
        posto = int(row["posto"])
        value = fpha_spillage_fit_max_m3s(posto, natural_monthly_means.get(posto, ()))
        if value is not None:
            maxima[code] = value
    return maxima


def convert_production_models(case: NewaveCase, id_map: NewaveIdMap) -> dict:
    """Build ``hydro_production_models.json`` with per-hydro model selection.

    - **Plants outside the FPHA:** ``model: "constant_productivity"`` with no
      numeric value; ``productivity_mw_per_m3s`` is supplied per-(hydro, stage)
      in ``hydro_energy_productivity.parquet`` (see
      :func:`convert_hydro_energy_productivity`). Cross-file validation in novomodelo
      rejects double-supply (JSON + parquet) and coverage gaps, so keeping
      productivity strictly in the parquet eliminates the conflict surface.
    - **FPHA plants** (:func:`fpha_eligible_codes`): ``model: "fpha"`` with the
      :func:`_fpha_computed_config` of each stage; novomodelo fits the production
      function from geometry + tailrace families.

    **Reference volume.** In FPHA cases every plant declares its
    :func:`~novomodelo_bridge.newave.converters.hydro.reference_volumes.fpha_reference_volumes`
    as an absolute ``reference_volume``: novomodelo reads a plant's downstream
    ``reference_volume`` (via ``downstream_id``) for that plant's backwater, so
    each plant must be the correct reference for its upstream FPHA neighbour.
    The stages are grouped by :func:`_model_selection`.

    When FPHA is active, a file-level ``fpha_plane_reduction`` block parsed from
    ``tratamento-fpha`` is added (see :func:`_parse_fpha_plane_reduction`).

    Parameters
    ----------
    case:
        Parsed the source model case.
    id_map:
        Pre-built entity ID map used to translate the source model plant codes to
        0-based Novomodelo hydro IDs.

    Returns
    -------
    dict
        A dict with a ``"production_models"`` key ready to serialise as
        ``system/hydro_production_models.json``.
    """
    existing = case.active_hydros
    confhd_codes = [int(r["codigo_usina"]) for _, r in existing.iterrows()]
    fpha_codes = fpha_eligible_codes(case)

    cadastro = (
        _apply_permanent_overrides(case.hidr.cadastro, case)
        if case.fpha_enabled
        else None
    )
    references = fpha_reference_volumes(case, cadastro) if cadastro is not None else {}
    calendar_months = (
        [d.month for d in stage_dates_for(case.horizon)] if references else []
    )

    spillage_max_by_code = (
        _spillage_fit_maxima(existing, fpha_codes, natural_monthly_means(case))
        if fpha_codes
        else {}
    )
    turbined_max_by_code = (
        _turbined_fit_maxima(case, cadastro, fpha_codes, references)
        if cadastro is not None and fpha_codes
        else {}
    )

    production_models: list[dict] = []
    for newave_code in confhd_codes:
        try:
            hydro_id = id_map.hydro_id(newave_code)
        except KeyError:
            continue

        hreg = (
            cadastro.loc[newave_code]
            if cadastro is not None and newave_code in cadastro.index
            else None
        )
        is_fpha = newave_code in fpha_codes and hreg is not None
        volumes: Sequence[float | None] = references.get(newave_code) or [None]
        fit_maxima = turbined_max_by_code.get(newave_code) or [None] * len(volumes)
        stages = [
            _stage_model(
                volume,
                _fpha_computed_config(
                    hreg, spillage_max_by_code.get(newave_code), fit_max
                )
                if is_fpha and hreg is not None
                else None,
            )
            for volume, fit_max in zip(volumes, fit_maxima, strict=True)
        ]
        production_models.append(
            {"hydro_id": hydro_id, **_model_selection(stages, calendar_months)}
        )

    production_models.sort(key=lambda m: m["hydro_id"])

    result: dict = {
        "$schema": novomodelo_schemas.schema_url_for(
            "system/hydro_production_models.json"
        )
    }
    # File-level FPHA plane reduction (tratamento-fpha), applied to every FPHA
    # plant. Only meaningful when there is at least one FPHA plant.
    if fpha_codes:
        plane_reduction = _parse_fpha_plane_reduction(case)
        if plane_reduction is not None:
            result["fpha_plane_reduction"] = plane_reduction
    result["production_models"] = production_models
    return result


def _stage_model(volume_hm3: float | None, fpha_config: dict | None) -> dict:
    """One stage's production model: FPHA with *fpha_config* when given, else
    constant productivity, with *volume_hm3* as its reference volume."""
    stage: dict = {
        "model": "fpha" if fpha_config is not None else "constant_productivity"
    }
    if fpha_config is not None:
        stage["fpha_config"] = fpha_config
    if volume_hm3 is not None:
        stage["reference_volume"] = {"volume_hm3": volume_hm3}
    return stage


def _model_selection(stages: Sequence[dict], calendar_months: Sequence[int]) -> dict:
    """The model selection of a plant whose study stage *s* uses ``stages[s]``.

    One open stage range when every stage agrees; twelve seasons
    (``season_id = calendar_month - 1``, the ``stages.json`` season map) when
    each calendar month always gets the same model; otherwise one stage range
    per run of equal stages.
    """
    if all(stage == stages[0] for stage in stages):
        return {
            "selection_mode": "stage_ranges",
            "stage_ranges": [{"start_stage_id": 0, "end_stage_id": None, **stages[0]}],
        }
    by_month: dict[int, dict] = {}
    periodic = all(
        by_month.setdefault(month, stage) == stage
        for month, stage in zip(calendar_months, stages, strict=True)
    )
    if periodic and len(by_month) == 12:
        return {
            "selection_mode": "seasonal",
            # All 12 seasons are listed, so the default is never consulted; a
            # config-less model keeps it valid.
            "default_model": "constant_productivity",
            "seasons": [
                {"season_id": month - 1, **by_month[month]} for month in range(1, 13)
            ],
        }
    ranges: list[dict] = []
    for stage_id, stage in enumerate(stages):
        if ranges and stage == stages[stage_id - 1]:
            ranges[-1]["end_stage_id"] = stage_id
        else:
            ranges.append(
                {"start_stage_id": stage_id, "end_stage_id": stage_id, **stage}
            )
    ranges[-1]["end_stage_id"] = None
    return {"selection_mode": "stage_ranges", "stage_ranges": ranges}


def _total_study_stages(case: NewaveCase) -> int:
    """Return the total number of stages in the study (including post-study)."""
    return case.horizon.total_stages


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
    return _per_stage_drop_series(
        hreg, base_prodt, drop_overrides, case, total_stages, _equivalent_productivity
    )


def _per_stage_stored_energy_productivities(
    hreg: pd.Series,
    base: float,
    drop_overrides: list[dict],
    case: NewaveCase,
    total_stages: int,
) -> list[float]:
    """Per-stage :func:`stored_energy_productivity`, applying CFUGA/CMONT.

    Like PRODT it ignores VOLREF_SAZ: the stored-energy head does not depend on
    the operating volume.
    """
    return _per_stage_drop_series(
        hreg, base, drop_overrides, case, total_stages, stored_energy_productivity
    )


def _per_stage_drop_series(
    hreg: pd.Series,
    base: float,
    drop_overrides: list[dict],
    case: NewaveCase,
    total_stages: int,
    evaluate: Callable[..., float],
) -> list[float]:
    """*evaluate* at each stage's effective CFUGA/CMONT, *base* where none applies."""
    if not drop_overrides:
        return [base] * total_stages
    drops = _per_stage_drop_overrides(drop_overrides, case, total_stages)
    return [
        base
        if cfuga is None and cmont is None
        else evaluate(hreg, canal_fuga_override=cfuga, cmont_override=cmont)
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
    """Build the per-(hydro, stage) productivity parquet table.

    Every plant, FPHA and non-FPHA alike, gets two values:

    - ``equivalent_productivity_mw_per_m3s`` — the reference-point ρ_eq, the
      ``gen = ρ·Q`` coefficient at v_65 or the VOLREF_SAZ volume with any
      CFUGA/CMONT override. `convert_production_models` emits no numeric
      productivity, so this is the plant's single ρ_eq source. novomodelo could
      derive an FPHA plant's value from its VHA geometry, but not with the
      source model's seasonal reference volume or tailrace overrides.
    - ``integrated_equivalent_productivity_mw_per_m3s`` — the plant's own
      stored-energy productivity, :func:`stored_energy_productivity` with the
      CFUGA/CMONT overrides of each stage. novomodelo sums it down the cascade for
      the integrated tags and the maximum stored energy that VminOP reads. It
      is always supplied: novomodelo's own integral over the VHA geometry also
      integrates ``D``/``S`` plants (the source model takes their point head
      at ``volume_referencia``), evaluates the tailrace model at maximum
      turbined flow rather than the mean tailrace level, and applies no
      CFUGA/CMONT override.

    Both values carry the fictitious plants' ρ folded in. Plants without
    CFUGA/CMONT/seasonal overrides emit one ``stage_id = NULL`` default row;
    plants with overrides emit one row per study stage.

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

    seasonal_volref = seasonal_useful_volumes(case)

    needs_per_stage = bool(plants_with_drop_overrides) or bool(seasonal_volref)
    total_stages = _total_study_stages(case) if needs_per_stage else 0

    # FICT-cascade: when a real plant's energy-cascade traverses fictitious plants,
    # fold those FICTs' ρ_eq into the upstream real plant's own ρ_eq so novomodelo's
    # per-plant cascade sum (from ``hydro_energy_productivity.parquet`` plus the
    # rewired ``downstream_id``) reproduces the source model's
    # ``produtibilidade_acumulada_calculo_earm``.  FICT plants have ρ_esp = 0 in the
    # bundled cases, so this is numerically a no-op there — a structural fix, robust
    # to non-zero FICT productivities.
    from novomodelo_bridge.newave.converters.fict_cascade import resolve_cascade

    fict_cascade = resolve_cascade(
        confhd_df, cadastro, fictitious=case.fictitious_codes
    )

    hydro_ids: list[int] = []
    stage_ids: list[int | None] = []
    equiv_prods: list[float] = []
    integrated_prods: list[float] = []

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

        stored_base = stored_energy_productivity(hreg)

        if not overrides and not plant_seasonal:
            hydro_ids.append(hydro_id)
            stage_ids.append(None)
            equiv_prods.append(legacy_base + fict_extra)
            integrated_prods.append(stored_base + fict_extra)
        else:
            per_stage = _per_stage_productivities(
                hreg,
                legacy_base,
                overrides,
                case,
                total_stages,
                seasonal_volref_by_month=plant_seasonal,
            )
            per_stage_stored = _per_stage_stored_energy_productivities(
                hreg, stored_base, overrides, case, total_stages
            )
            for stage_id, (value, stored) in enumerate(
                zip(per_stage, per_stage_stored, strict=True)
            ):
                hydro_ids.append(hydro_id)
                stage_ids.append(stage_id)
                equiv_prods.append(value + fict_extra)
                integrated_prods.append(stored + fict_extra)

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
            "integrated_equivalent_productivity_mw_per_m3s": pa.array(
                integrated_prods, type=pa.float64()
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

    Keys are the source model plant codes (not Novomodelo ids) since cascade traversal in
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

    seasonal_volref = seasonal_useful_volumes(case)

    # FICT-cascade fold-in: per-stage ρ_eq must already include any FICT
    # contribution so that the per-stage ρ_acum used by VminOP and EARM
    # accounting matches the topology rewired into ``hydros.json``.
    from novomodelo_bridge.newave.converters.fict_cascade import resolve_cascade

    fict_cascade = resolve_cascade(
        confhd_df, cadastro, fictitious=case.fictitious_codes
    )

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
    from novomodelo_bridge.newave.converters.fict_cascade import resolve_cascade

    fict_cascade = resolve_cascade(
        confhd_df, cadastro, fictitious=case.fictitious_codes
    )

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
