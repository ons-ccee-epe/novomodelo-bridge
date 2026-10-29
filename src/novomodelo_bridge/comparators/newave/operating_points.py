"""Both models compared at the NEWAVE per-block operating points.

Reads the source model's points from the nwlistop plant tables and its
production model from the ``fpha`` reports into the shapes
:mod:`novomodelo_bridge.comparators.operating_points` takes. Its conventions, each
checked against its own printed heads (``hmont``, ``hjus``, ``hliq``) and
exact production (``gh_fphexat``):

- Its planes read ``GH <= fc · (γ0 + γv · V_u + γq · Q + γs · S)``, ``V_u``
  measured above the plant's minimum storage; period ``p`` is stage ``p``
  minus the study's first month.
- Its FPHA storage is the stage's mean: the minimum plus the mean of the
  initial and final useful storage (``varmuh``), the first stage starting
  from the ``confhd`` initial storage. A daily or run-of-river plant is held
  at its reference volume.
- The forebay comes from the cota polynomial. The tailrace is read at
  ``Q + S``, with spill left out where ``hidr.dat`` says it does not reach
  the tailrace, on the ``polinjus`` families (``canal_fuga_medio`` without
  them) interpolated at the downstream plant's forebay at the same point.
  ``G_max`` is the ``fpha_eco`` generation window.
- Generation is printed in MWmes per block: the block's mean MW times its
  share of the month.

Its points are flagged worthless on the submarket CMO (``cmarg``).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import polars as pl

from novomodelo_bridge.comparators.analyze import build_fpha_comparison
from novomodelo_bridge.comparators.newave.readers import (
    read_fpha_grid,
    read_fpha_planes,
    read_nwlistop_block_cmo,
    read_nwlistop_hydro_blocks,
    read_nwlistop_hydro_storage,
)
from novomodelo_bridge.comparators.operating_points import (
    PLANE_SCHEMA,
    ZERO_PRICE,
    FitWindow,
    Hydraulics,
    OperatingPointComparison,
    SourceProductionModel,
    StagePhysics,
    evaluate_points,
    flag_ties,
    masked_hydro,
    novomodelo_points,
    productivity_frames,
    tie_share,
    window_gh_max,
    with_envelopes,
)
from novomodelo_bridge.core.diagnostics import Diagnostic, Severity, emit
from novomodelo_bridge.core.pandas_utils import as_frame_or_empty, as_int, is_na
from novomodelo_bridge.core.tailrace import (
    TailraceFamily,
    TailraceSegment,
    tailrace_families_from_polinjus,
)
from novomodelo_bridge.newave.converters.hydro import read_cadastro
from novomodelo_bridge.newave.converters.hydro.geometry import held_volume
from novomodelo_bridge.newave.converters.hydro.reference_volumes import (
    initial_storage_hm3,
)
from novomodelo_bridge.newave.converters.temporal import block_fractions
from novomodelo_bridge.newave.filling import month_hours
from novomodelo_bridge.newave.horizon import stage_dates_for
from novomodelo_bridge.novomodelo import readers as novomodelo_readers

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import pandas as pd

    from novomodelo_bridge.comparators.newave.alignment import EntityAlignment
    from novomodelo_bridge.newave.case import NewaveCase

_LOG = logging.getLogger(__name__)


def compare_operating_points(
    case: NewaveCase, alignment: EntityAlignment, novomodelo_output_dir: Path
) -> OperatingPointComparison:
    """Compare both models' FPHA surfaces and, from the nwlistop plant
    tables, their per-block operating points.

    Without the plant tables only the surface frames are filled, with a note
    naming the ``nwlistop.dat`` plant list.
    """
    names = {h.novomodelo_id: h.name for h in alignment.hydros}
    code_to_novomodelo = {h.newave_code: h.novomodelo_id for h in alignment.hydros}
    newave_dir = case.files.directory
    novomodelo_planes = novomodelo_readers.read_novomodelo_fpha_planes(
        novomodelo_output_dir
    )
    source_planes = read_fpha_planes(newave_dir)
    grid = read_fpha_grid(newave_dir)
    if novomodelo_planes is None or source_planes is None or grid is None:
        return OperatingPointComparison()
    _LOG.info("Comparing fitted production functions (FPHA)...")
    metrics, surface, spill = build_fpha_comparison(
        source_planes, grid, novomodelo_planes, code_to_novomodelo, names
    )
    surfaces = {"metrics": metrics, "surface": surface, "spill": spill}

    horizon = case.horizon
    blocks = read_nwlistop_hydro_blocks(
        newave_dir, horizon.start_year, horizon.start_month
    )
    if blocks.is_empty():
        emit(
            Diagnostic(
                code="nwlistop-plant-tables-missing",
                severity=Severity.INFO,
                category="Compare data",
                title="No nwlistop plant tables",
                summary=(
                    "The source-model run has no per-plant nwlistop tables "
                    "(ghiduh<NNN>.out), so the Productivity tab leaves out "
                    "both models' operating points. Set the plant list in "
                    "nwlistop.dat to 999 (every plant) and rerun nwlistop to "
                    "add them."
                ),
            ),
            logger=_LOG,
        )
        return OperatingPointComparison(**surfaces)

    cadastro = read_cadastro(case)
    source = _source_points(case, cadastro, blocks, code_to_novomodelo)
    novomodelo = novomodelo_points(
        novomodelo_output_dir, {c: k for k, c in code_to_novomodelo.items()}
    )
    if source.is_empty() or novomodelo.is_empty():
        return OperatingPointComparison(**surfaces)
    source, novomodelo = flag_ties(source, novomodelo)
    return OperatingPointComparison(
        **surfaces,
        **_fpha_frames(
            case,
            cadastro,
            source,
            novomodelo,
            source_planes,
            grid,
            novomodelo_planes,
            code_to_novomodelo,
            novomodelo_output_dir,
            names,
        ),
        tie_share=tie_share(source, novomodelo),
        masked_hydro=masked_hydro(source, novomodelo, names),
    )


def _source_points(
    case: NewaveCase,
    cadastro: pd.DataFrame,
    blocks: pl.DataFrame,
    code_to_novomodelo: Mapping[int, int],
) -> pl.DataFrame:
    """The source model's per-block operating points of every plant aligned
    to a Novomodelo hydro, with ``printed_exact_mw`` from ``gh_fphexat``.

    ``v_hm3`` is null for a reservoir without a ``varmuh`` table.
    """
    horizon = case.horizon
    durations = pl.DataFrame(
        [
            (stage, block, fraction, fraction * month_hours(day.year, day.month))
            for stage, (day, fractions) in enumerate(
                zip(stage_dates_for(horizon), block_fractions(case), strict=True)
            )
            for block, fraction in enumerate(fractions)
        ],
        schema={
            "stage": pl.Int64,
            "block": pl.Int64,
            "fraction": pl.Float64,
            "hours": pl.Float64,
        },
        orient="row",
    )
    points = (
        blocks.with_columns(
            pl.col("newave_code").alias("code"),
            pl.col("newave_code")
            .replace_strict(
                dict(code_to_novomodelo), default=None, return_dtype=pl.Int64
            )
            .alias("novomodelo_id"),
        )
        .filter(pl.col("novomodelo_id").is_not_null())
        .join(durations, on=["stage", "block"])
    )
    if points.is_empty():
        return pl.DataFrame()
    held = {
        as_int(code): volume
        for code, hreg in cadastro.iterrows()
        if (volume := held_volume(hreg)) is not None
    }
    cmo = read_nwlistop_block_cmo(
        case.files.directory, horizon.start_year, horizon.start_month
    )
    return (
        points.join(
            _reservoir_storage(case, cadastro),
            on=["code", "stage", "scenario"],
            how="left",
        )
        .with_columns(
            pl.coalesce(
                pl.col("code").replace_strict(
                    held, default=None, return_dtype=pl.Float64
                ),
                pl.col("v_hm3"),
            ).alias("v_hm3"),
            pl.col("code")
            .replace_strict(
                _plant_submarkets(case), default=None, return_dtype=pl.Int64
            )
            .alias("submarket_code"),
        )
        .join(cmo, on=["submarket_code", "stage", "scenario", "block"], how="left")
        .select(
            "code",
            "novomodelo_id",
            "stage",
            "scenario",
            "block",
            (pl.col("hours") / pl.col("scenario").n_unique().over("stage")).alias(
                "weight"
            ),
            "v_hm3",
            pl.col("turbined_m3s").fill_null(0.0).alias("q_m3s"),
            pl.col("spillage_m3s").fill_null(0.0).alias("s_m3s"),
            (
                pl.col("turbined_m3s").fill_null(0.0)
                + pl.col("spillage_m3s").fill_null(0.0)
            ).alias("outflow_m3s"),
            (pl.col("generation_mwmes").fill_null(0.0) / pl.col("fraction")).alias(
                "gh_mw"
            ),
            (pl.col("exact_generation_mwmes") / pl.col("fraction")).alias(
                "printed_exact_mw"
            ),
            (pl.col("cmo").abs() < ZERO_PRICE).fill_null(False).alias("worthless"),
        )
    )


def _plant_submarkets(case: NewaveCase) -> dict[int, int]:
    """``{plant code: submarket code}`` through each plant's ``confhd`` REE."""
    rees = case.ree.rees
    ree_submarket = {
        int(code): int(submarket)
        for code, submarket in zip(rees["codigo"], rees["submercado"], strict=True)
    }
    plants = case.confhd.usinas
    return {
        int(code): ree_submarket[int(ree)]
        for code, ree in zip(plants["codigo_usina"], plants["ree"], strict=True)
        if int(ree) in ree_submarket
    }


def _reservoir_storage(case: NewaveCase, cadastro: pd.DataFrame) -> pl.DataFrame:
    """``(code, stage, scenario, v_hm3)``: each reservoir's minimum plus the
    mean of its initial and final useful storage over the stage."""
    horizon = case.horizon
    final = read_nwlistop_hydro_storage(
        case.files.directory, horizon.start_year, horizon.start_month
    )
    plants = case.confhd.usinas
    initial_percent = {
        int(code): float(percent)
        for code, percent in zip(
            plants["codigo_usina"], plants["volume_inicial_percentual"], strict=True
        )
    }
    floors = pl.DataFrame(
        [
            (
                as_int(code),
                float(hreg["volume_minimo"]),
                initial_storage_hm3(hreg, initial_percent[as_int(code)])
                - float(hreg["volume_minimo"]),
            )
            for code, hreg in cadastro.iterrows()
            if as_int(code) in initial_percent
        ],
        schema={"code": pl.Int64, "v_min": pl.Float64, "initial_useful": pl.Float64},
        orient="row",
    )
    return (
        final.rename({"newave_code": "code"})
        .join(floors, on="code")
        .sort("code", "scenario", "stage")
        .select(
            "code",
            "stage",
            "scenario",
            (
                pl.col("v_min")
                + (
                    pl.col("useful_storage_final_hm3")
                    .shift(1)
                    .over("code", "scenario")
                    .fill_null(pl.col("initial_useful"))
                    + pl.col("useful_storage_final_hm3")
                )
                / 2.0
            ).alias("v_hm3"),
        )
    )


def _fpha_frames(
    case: NewaveCase,
    cadastro: pd.DataFrame,
    source: pl.DataFrame,
    novomodelo: pl.DataFrame,
    source_planes: pl.DataFrame,
    grid: pl.DataFrame,
    novomodelo_planes: pl.DataFrame,
    code_to_novomodelo: Mapping[int, int],
    novomodelo_output_dir: Path,
    names: Mapping[int, str],
) -> dict[str, pl.DataFrame]:
    """The split, fit-input and slice frames over the (plant, stage) pairs
    both models fitted; empty when there are none."""
    first_month = case.horizon.start_month
    source_by_key = {
        (int(k[0]), int(k[1])): v.select(list(PLANE_SCHEMA))
        for k, v in source_planes.with_columns(
            (pl.col("periodo") - first_month).alias("stage")
        )
        .partition_by(["newave_code", "stage"], as_dict=True)
        .items()
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
        and code in cadastro.index
        and (code_to_novomodelo[code], stage) in novomodelo_by_key
    }
    fitted = {code for code, _ in pairs}
    unstored = sorted(
        fitted
        & set(source.filter(pl.col("v_hm3").is_null()).get_column("code").to_list())
    )
    if unstored:
        _LOG.info(
            "No nwlistop storage (varmuh) for plants %s; their FPHA is left out "
            "of the operating-point comparison.",
            unstored,
        )
    hydraulics = _plant_hydraulics(case, cadastro, fitted - set(unstored))
    pairs = {(code, stage) for code, stage in pairs if code in hydraulics}
    if not pairs:
        return {}

    windows = {
        (int(r["newave_code"]), int(r["periodo"]) - first_month): FitWindow(
            v_min_hm3=r["v_min_hm3"],
            v_max_hm3=r["v_max_hm3"],
            q_max_m3s=r["q_max_m3s"],
            grid=f"{int(r['n_v'])} × {int(r['n_q'])}",
            gh_max_mw=r["gh_max_mw"],
        )
        for r in grid.iter_rows(named=True)
    }
    cota = {
        as_int(code): tuple(float(hreg[f"a{i}_volume_cota"]) for i in range(5))
        for code, hreg in cadastro.iterrows()
    }
    model = SourceProductionModel(
        planes={key: source_by_key[key] for key in pairs},
        windows=windows,
        hydraulics=hydraulics,
        physics={
            (code, stage): StagePhysics(
                cota=cota[code],
                rho_esp=float(cadastro.loc[code, "produtibilidade_especifica"]),
                losses=float(cadastro.loc[code, "perdas"] or 0.0),
                max_generation_mw=window_gh_max(windows, code, stage),
            )
            for code, stage in pairs
        },
        downstream={
            (code, stage): _downstream(cadastro, code) for code, stage in pairs
        },
        cota=lambda code, _stage: cota[code],
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
        _exact_check(source_eval),
        names,
    )
    return {"split": split, "fit_inputs": fit_inputs, "slices": slices}


def _downstream(cadastro: pd.DataFrame, code: int) -> int | None:
    """The plant ``hidr.dat`` names downstream of *code*, ``None`` for none."""
    raw = cadastro.loc[code, "codigo_usina_jusante"]
    if is_na(raw) or int(raw) == 0 or int(raw) not in cadastro.index:
        return None
    return int(raw)


def _plant_hydraulics(
    case: NewaveCase, cadastro: pd.DataFrame, codes: set[int]
) -> dict[int, Hydraulics]:
    """Each plant's exact-function inputs; a plant with neither ``polinjus``
    families nor ``canal_fuga_medio`` is left out and logged."""
    polinjus = case.polinjus
    families = (
        {}
        if polinjus is None
        else tailrace_families_from_polinjus(
            as_frame_or_empty(polinjus.hidreletrica_curvajusante(df=True)),
            as_frame_or_empty(
                polinjus.hidreletrica_curvajusante_polinomio_segmento(df=True)
            ),
        )
    )
    hydraulics: dict[int, Hydraulics] = {}
    missing: list[int] = []
    for code in sorted(codes):
        hreg = cadastro.loc[code]
        plant_families = families.get(code) or _constant_tailrace(hreg)
        if plant_families is None:
            missing.append(code)
            continue
        hydraulics[code] = Hydraulics(
            families=plant_families,
            spill_in_tailrace=bool(int(hreg["influencia_vertimento_canal_fuga"])),
            tipo_perda=int(hreg.get("tipo_perda", 0) or 0),
            volume_floor_hm3=float(hreg["volume_minimo"]),
        )
    if missing:
        _LOG.info(
            "No tailrace curve for plants %s; their FPHA is left out of the "
            "operating-point comparison.",
            missing,
        )
    return hydraulics


def _constant_tailrace(hreg: pd.Series) -> tuple[TailraceFamily, ...] | None:
    """``canal_fuga_medio`` as one constant tailrace family."""
    level = hreg.get("canal_fuga_medio")
    if is_na(level):
        return None
    return (
        TailraceFamily(
            downstream_reference_level_m=math.nan,
            segments=(
                TailraceSegment(
                    outflow_max_m3s=math.inf,
                    coefficients=(float(level), 0.0, 0.0, 0.0, 0.0),
                ),
            ),
        ),
    )


def _exact_check(source_eval: pl.DataFrame) -> dict[tuple[int, int], float]:
    """``{(code, stage): max |exact − printed exact|}`` over the points that
    print ``gh_fphexat``."""
    return {
        (int(code), int(stage)): float(worst)
        for code, stage, worst in source_eval.filter(
            pl.col("printed_exact_mw").is_not_null()
        )
        .group_by("code", "stage")
        .agg((pl.col("exact_mw") - pl.col("printed_exact_mw")).abs().max())
        .iter_rows()
    }
