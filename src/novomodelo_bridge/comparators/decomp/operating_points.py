"""Both models compared at the DECOMP per-block operating points.

Reads the source model's points and production model into the shapes
:mod:`novomodelo_bridge.comparators.operating_points` takes. Its conventions, each
checked against its own printed deviation table (``dec_desvfpha``):

- Its planes read ``GH <= fc · (rhs + cv · V_u + cq · Q + cs · S + cl ·
  Q_lat)``, ``V_u`` measured above the ``hidr.dat`` minimum and ``Q_lat`` the
  ``VU`` share of the influencing plants' turbined plus spilled flow in the
  same block. The ``VA`` gauge share is folded into ``rhs``.
- Its FPHA storage is the stage's mean, the effective minimum plus the mean of
  the initial and final useful volumes.
- The forebay comes from the effective cota polynomial. The tailrace is read
  at ``VL · (Q + S) + Q_lat + VA share``, with spill left out where
  ``hidr.dat`` says it does not reach the tailrace. ``G_max`` is the
  ``eco_fpha_`` generation window. The ``VA`` share is the factor times the
  gauge plant's first-stage incremental inflow, held over the horizon as the
  source model prints it.

Its points are flagged worthless on the submarket CMO; the source model
prints no water value.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import polars as pl

from novomodelo_bridge.comparators.analyze import build_fpha_comparison
from novomodelo_bridge.comparators.decomp.readers import (
    read_avl_cortesfpha,
    read_dec_desvfpha,
    read_dec_oper_sist,
    read_dec_oper_usih,
    read_eco_fpha,
)
from novomodelo_bridge.comparators.operating_points import (
    LINK_SCHEMA,
    POINT_KEYS,
    ZERO_PRICE,
    FitWindow,
    Hydraulics,
    OperatingPointComparison,
    SourceProductionModel,
    StagePhysics,
    evaluate_points,
    flag_ties,
    head,
    masked_hydro,
    novomodelo_points,
    productivity_frames,
    tie_share,
    weighted_mean,
    window_gh_max,
    with_envelopes,
)
from novomodelo_bridge.core.pandas_utils import as_frame
from novomodelo_bridge.decomp.converters.cadastro import (
    EffectiveCadastro,
    build_effective_cadastro,
    effective_storage_range,
)
from novomodelo_bridge.decomp.converters.hydro.turbine_limit import tailrace_families
from novomodelo_bridge.novomodelo import readers as novomodelo_readers

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from novomodelo_bridge.decomp.case import DecompCase

_LOG = logging.getLogger(__name__)

#: ``avl_cortesfpha`` coefficient -> the shared plane column.
_PLANE_COLUMNS = {
    "rhs": "gamma_0",
    "coeficiente_volume_util": "gamma_v",
    "coeficiente_vazao_turbinada": "gamma_q",
    "coeficiente_vazao_vertida": "gamma_s",
    "coeficiente_vazao_lateral": "gamma_lat",
}


def compare_operating_points(
    decomp_dir: Path,
    case: DecompCase,
    novomodelo_output_dir: Path,
    probabilities: pl.DataFrame,
    names: Mapping[int, str],
) -> OperatingPointComparison | None:
    """Compare both models at their per-block operating points.

    *probabilities* is the source model's ``(estagio, cenario,
    probabilidade)`` tree; when empty, each stage's scenarios weigh equally.
    *names* maps Novomodelo hydro ids to plant names. ``None`` when either run has
    no per-block hydro operation. The FPHA frames are empty when either model
    fitted no planes.
    """
    try:
        usih = read_dec_oper_usih(decomp_dir)
        sist = read_dec_oper_sist(decomp_dir)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No source-model per-block operation: %s", exc)
        return None
    eff, _ = build_effective_cadastro(case.dadger, case.hidr, case.calendar)
    source = _source_points(usih, sist, case, eff, probabilities)
    novomodelo = novomodelo_points(
        novomodelo_output_dir, dict(enumerate(case.id_map.hydro_codes))
    )
    if source.is_empty() or novomodelo.is_empty():
        return None
    source, novomodelo = flag_ties(source, novomodelo)
    return OperatingPointComparison(
        **_fpha_frames(
            decomp_dir,
            case,
            eff,
            novomodelo_output_dir,
            source,
            novomodelo,
            _first_stage_incremental(usih),
            names,
        ),
        tie_share=tie_share(source, novomodelo),
        masked_hydro=masked_hydro(source, novomodelo, names),
    )


def _source_points(
    usih: pl.DataFrame,
    sist: pl.DataFrame,
    case: DecompCase,
    eff: EffectiveCadastro,
    probabilities: pl.DataFrame,
) -> pl.DataFrame:
    """The source model's per-block operating points, every operated plant."""
    code_to_novomodelo = {c: case.id_map.hydro_id(c) for c in case.id_map.hydro_codes}
    points = (
        usih.filter(pl.col("patamar").is_not_null())
        .with_columns(
            pl.col("codigo_usina").cast(pl.Int64).alias("code"),
            (pl.col("estagio").cast(pl.Int64) - 1).alias("stage"),
            pl.col("cenario").cast(pl.Int64).alias("scenario"),
            (pl.col("patamar").cast(pl.Int64) - 1).alias("block"),
        )
        .with_columns(
            pl.col("code")
            .replace_strict(code_to_novomodelo, default=None, return_dtype=pl.Int64)
            .alias("novomodelo_id")
        )
        .filter(pl.col("novomodelo_id").is_not_null())
    )
    if points.is_empty():
        return pl.DataFrame()
    if probabilities.is_empty():
        points = points.with_columns(
            (1.0 / pl.col("scenario").n_unique().over("stage")).alias("probability")
        )
    else:
        points = points.join(
            probabilities.select(
                (pl.col("estagio").cast(pl.Int64) - 1).alias("stage"),
                pl.col("cenario").cast(pl.Int64).alias("scenario"),
                pl.col("probabilidade").cast(pl.Float64).alias("probability"),
            ),
            on=["stage", "scenario"],
            how="left",
        ).with_columns(pl.col("probability").fill_null(1.0))
    floors = pl.DataFrame(
        [
            (code, stage, effective_storage_range(eff, code, stage)[0])
            for code, stage in points.select("code", "stage").unique().iter_rows()
        ],
        schema={"code": pl.Int64, "stage": pl.Int64, "floor_hm3": pl.Float64},
        orient="row",
    )
    cmo = sist.filter(pl.col("patamar").is_not_null()).select(
        (pl.col("estagio").cast(pl.Int64) - 1).alias("stage"),
        pl.col("cenario").cast(pl.Int64).alias("scenario"),
        (pl.col("patamar").cast(pl.Int64) - 1).alias("block"),
        pl.col("nome_submercado").str.strip_chars().alias("submarket"),
        pl.col("cmo").cast(pl.Float64).alias("price"),
    )
    return (
        points.join(floors, on=["code", "stage"], how="left")
        .with_columns(pl.col("nome_submercado").str.strip_chars().alias("submarket"))
        .join(cmo, on=[*POINT_KEYS, "submarket"], how="left")
        .select(
            "code",
            "novomodelo_id",
            *POINT_KEYS,
            (pl.col("probability") * pl.col("duracao").cast(pl.Float64)).alias(
                "weight"
            ),
            (
                pl.col("floor_hm3")
                + (
                    pl.col("volume_util_inicial_hm3").fill_null(0.0)
                    + pl.col("volume_util_final_hm3").fill_null(0.0)
                )
                / 2.0
            ).alias("v_hm3"),
            pl.col("vazao_turbinada_m3s")
            .cast(pl.Float64)
            .fill_null(0.0)
            .alias("q_m3s"),
            pl.col("vazao_vertida_m3s").cast(pl.Float64).fill_null(0.0).alias("s_m3s"),
            pl.col("vazao_defluente_m3s").cast(pl.Float64).alias("outflow_m3s"),
            pl.col("geracao_MW").cast(pl.Float64).fill_null(0.0).alias("gh_mw"),
            (pl.col("price").abs() < ZERO_PRICE).fill_null(False).alias("worthless"),
        )
    )


def _first_stage_incremental(usih: pl.DataFrame) -> dict[int, float]:
    """``{code: incremental inflow}`` of the first stage's aggregate rows."""
    rows = usih.filter(
        pl.col("patamar").is_null()
        & (pl.col("estagio") == pl.col("estagio").min())
        & (pl.col("cenario") == pl.col("cenario").min())
    )
    return {
        int(code): float(value)
        for code, value in rows.select(
            "codigo_usina", "vazao_incremental_m3s"
        ).iter_rows()
        if value is not None
    }


def _fpha_frames(
    decomp_dir: Path,
    case: DecompCase,
    eff: EffectiveCadastro,
    novomodelo_output_dir: Path,
    source: pl.DataFrame,
    novomodelo: pl.DataFrame,
    first_incremental: Mapping[int, float],
    names: Mapping[int, str],
) -> dict[str, pl.DataFrame]:
    """The :class:`OperatingPointComparison` FPHA fields; empty when either
    model fitted no planes."""
    try:
        source_planes = read_avl_cortesfpha(decomp_dir)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No source-model FPHA planes: %s", exc)
        return {}
    novomodelo_planes = novomodelo_readers.read_novomodelo_fpha_planes(
        novomodelo_output_dir
    )
    if (
        novomodelo_planes is None
        or novomodelo_planes.is_empty()
        or source_planes.is_empty()
    ):
        return {}
    try:
        eco = read_eco_fpha(decomp_dir)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No source-model FPHA fit windows (eco_fpha_): %s", exc)
        eco = None

    code_to_novomodelo = {c: case.id_map.hydro_id(c) for c in case.id_map.hydro_codes}
    source_planes = source_planes.rename(_PLANE_COLUMNS).with_columns(
        pl.col("codigo_usina").alias("code"),
        (pl.col("estagio") - 1).alias("stage"),
    )
    source_by_key = {
        (int(k[0]), int(k[1])): v
        for k, v in source_planes.partition_by(["code", "stage"], as_dict=True).items()
    }
    novomodelo_by_key = {
        (int(k[0]), int(k[1])): v
        for k, v in novomodelo_planes.partition_by(
            ["hydro_id", "stage_id"], as_dict=True
        ).items()
    }
    pairs = {
        (code, stage)
        for code, stage in source_by_key
        if code in code_to_novomodelo
        and (code_to_novomodelo[code], stage) in novomodelo_by_key
    }
    hydraulics, links = _plant_hydraulics(
        case, {code for code, _ in pairs}, first_incremental
    )
    missing = sorted({code for code, _ in pairs if code not in hydraulics})
    if missing:
        _LOG.info(
            "No tailrace curve for plants %s; their FPHA is left out of the "
            "operating-point comparison.",
            missing,
        )
    pairs = {(code, stage) for code, stage in pairs if code in hydraulics}
    if not pairs:
        return {}

    windows = {} if eco is None else _fit_windows(eco)
    model = SourceProductionModel(
        planes={key: source_by_key[key] for key in pairs},
        windows=windows,
        hydraulics=hydraulics,
        physics={
            (code, stage): StagePhysics(
                cota=tuple(eff.cota_polynomial(code, stage)),
                rho_esp=eff.value(code, "produtibilidade_especifica", stage),
                losses=eff.value(code, "perdas", stage),
                max_generation_mw=window_gh_max(windows, code, stage),
            )
            for code, stage in pairs
        },
        downstream={
            (code, stage): eff.downstream_plant(code, stage) or None
            for code, stage in pairs
        },
        cota=eff.cota_polynomial,
        links=links,
    )
    source_eval = with_envelopes(
        evaluate_points(source, model), model, novomodelo_by_key
    )
    split, fit_inputs, slices = productivity_frames(
        source_eval,
        evaluate_points(novomodelo, model),
        model,
        novomodelo_by_key,
        novomodelo_output_dir,
        _exact_check(decomp_dir, eff, model),
        names,
    )
    metrics, surface, spill = build_fpha_comparison(
        _folded_source_planes(source_planes, source_eval, pairs),
        None if eco is None else _source_grid(eco, hydraulics),
        novomodelo_planes,
        code_to_novomodelo,
        names,
    )
    return {
        "split": split,
        "fit_inputs": fit_inputs,
        "slices": slices,
        "metrics": metrics,
        "surface": surface,
        "spill": spill,
    }


def _fit_windows(eco: pl.DataFrame) -> dict[tuple[int, int], FitWindow]:
    return {
        (int(r["codigo_usina"]), int(r["estagio"]) - 1): FitWindow(
            v_min_hm3=r["volume_armazenado_minimo"],
            v_max_hm3=r["volume_armazenado_maximo"],
            q_max_m3s=r["vazao_turbinada_maxima"],
            grid=(
                f"{int(r['numero_pontos_volume_armazenado'])}"
                f" × {int(r['numero_pontos_vazao_turbinada'])}"
            ),
            gh_max_mw=r["geracao_maxima"],
        )
        for r in eco.iter_rows(named=True)
    }


def _plant_hydraulics(
    case: DecompCase, codes: set[int], first_incremental: Mapping[int, float]
) -> tuple[dict[int, Hydraulics], pl.DataFrame]:
    """Each plant's exact-function inputs, and the ``VU`` links
    (:data:`~novomodelo_bridge.comparators.operating_points.LINK_SCHEMA`).

    A ``VA`` gauge with no operated plant on it contributes nothing and is
    logged.
    """
    hidr = case.hidr
    families = tailrace_families(case, case.id_map)
    dadger = case.dadger
    vl = as_frame(dadger.vl(df=True))
    vu = as_frame(dadger.vu(df=True))
    va = as_frame(dadger.va(df=True))
    outflow_factor: dict[int, float] = (
        {}
        if vl is None
        else {
            int(c): float(f)
            for c, f in zip(
                vl["codigo_usina_influenciada"],
                vl["fator_impacto_defluencia"],
                strict=True,
            )
        }
    )
    links = pl.DataFrame(
        []
        if vu is None
        else [
            (int(t), int(i), float(f))
            for t, i, f in zip(
                vu["codigo_usina_influenciada"],
                vu["codigo_usina_influenciadora"],
                vu["fator_impacto_defluencia"],
                strict=True,
            )
        ],
        schema=LINK_SCHEMA,
        orient="row",
    )
    operated = set(case.id_map.hydro_codes)
    plant_of_gauge = {
        int(p): int(c) for c, p in hidr["posto"].items() if int(c) in operated
    }
    gauge_share: dict[int, float] = defaultdict(float)
    if va is not None:
        for gauge, target, factor in zip(
            va["codigo_posto_influenciador"],
            va["codigo_usina_influenciada"],
            va["fator_impacto_incremental"],
            strict=True,
        ):
            plant = plant_of_gauge.get(int(gauge))
            if plant is None or plant not in first_incremental:
                _LOG.info(
                    "VA register of plant %d: no operated plant on gauge %d; its "
                    "share of the tail flow is left out of the exact production "
                    "function.",
                    int(target),
                    int(gauge),
                )
                continue
            gauge_share[int(target)] += float(factor) * first_incremental[plant]
    hydraulics = {
        code: Hydraulics(
            families=families[code],
            spill_in_tailrace=bool(
                int(hidr.loc[code, "influencia_vertimento_canal_fuga"])
            ),
            tipo_perda=int(hidr.loc[code, "tipo_perda"]),
            volume_floor_hm3=float(hidr.loc[code, "volume_minimo"]),
            outflow_factor=outflow_factor.get(code, 1.0),
            gauge_share_m3s=gauge_share.get(code, 0.0),
        )
        for code in codes
        if code in families
    }
    return hydraulics, links


def _exact_check(
    decomp_dir: Path, eff: EffectiveCadastro, model: SourceProductionModel
) -> dict[tuple[int, int], float]:
    """``{(code, stage): max |exact − printed FPH|}`` over the source model's
    printed deviation points, with its printed storage, flows, ``VU`` lateral
    flow and the downstream plant's printed forebay; empty without the table.
    """
    try:
        printed = read_dec_desvfpha(decomp_dir)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No source-model FPHA deviation table: %s", exc)
        return {}
    keys = ["estagio", "no", "patamar"]
    forebay = {
        (code, *key): level
        for code, *key, level in printed.select(
            "codigo_usina", *keys, "altura_montante"
        ).iter_rows()
    }
    worst: dict[tuple[int, int], float] = {}
    for code, estagio, no, patamar, v, q, s, lateral, fph in printed.select(
        "codigo_usina",
        *keys,
        "volume_total_hm3",
        "vazao_turbinada_m3s",
        "vazao_vertida_m3s",
        "vazao_lateral_usina_m3s",
        "geracao_hidraulica_fph",
    ).iter_rows():
        key = (code, estagio - 1)
        if key not in model.physics or fph is None:
            continue
        downstream = eff.downstream_plant(code, estagio - 1)
        point_head = head(
            model.hydraulics[code],
            model.physics[key],
            v=v,
            q=q,
            s=s,
            lateral_m3s=lateral or 0.0,
            downstream_level_m=forebay.get((downstream, estagio, no, patamar)),
        )
        worst[key] = max(worst.get(key, 0.0), abs(point_head.generation_mw - fph))
    return worst


def _folded_source_planes(
    planes: pl.DataFrame, source_eval: pl.DataFrame, pairs: set[tuple[int, int]]
) -> pl.DataFrame:
    """The source model's planes in the shared surface comparison's shape, the
    lateral term folded into the intercept at the stage's mean lateral flow."""
    lateral = source_eval.group_by("code", "stage").agg(
        weighted_mean("lateral_m3s").alias("_lateral")
    )
    keep = pl.DataFrame(
        sorted(pairs), schema={"code": pl.Int64, "stage": pl.Int64}, orient="row"
    )
    return (
        planes.join(keep, on=["code", "stage"])
        .join(lateral, on=["code", "stage"], how="left")
        .select(
            pl.col("code").alias("newave_code"),
            pl.col("estagio").alias("periodo"),
            (
                pl.col("gamma_0")
                + pl.col("gamma_lat") * pl.col("_lateral").fill_null(0.0)
            ).alias("gamma_0"),
            "gamma_v",
            "gamma_q",
            "gamma_s",
            "fator_correcao",
        )
    )


def _source_grid(
    eco: pl.DataFrame, hydraulics: Mapping[int, Hydraulics]
) -> pl.DataFrame:
    """The source model's fit windows in the shared surface comparison's shape."""
    offsets = pl.DataFrame(
        [(code, h.volume_floor_hm3) for code, h in hydraulics.items()],
        schema={"newave_code": pl.Int64, "volume_offset_hm3": pl.Float64},
        orient="row",
    )
    return eco.select(
        pl.col("codigo_usina").alias("newave_code"),
        pl.col("estagio").alias("periodo"),
        pl.col("volume_armazenado_minimo").alias("v_min_hm3"),
        pl.col("volume_armazenado_maximo").alias("v_max_hm3"),
        pl.col("vazao_turbinada_minima").alias("q_min_m3s"),
        pl.col("vazao_turbinada_maxima").alias("q_max_m3s"),
        pl.col("numero_pontos_volume_armazenado").cast(pl.Int64).alias("n_v"),
        pl.col("numero_pontos_vazao_turbinada").cast(pl.Int64).alias("n_q"),
    ).join(offsets, on="newave_code")
