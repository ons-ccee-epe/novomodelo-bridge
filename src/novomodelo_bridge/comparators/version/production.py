"""Fitted production surface of a version comparison.

Both cases' hyperplanes are evaluated on one grid by the shared
``build_fpha_comparison``, which takes a source model's planes and fitting grid.
A Novomodelo baseline has neither, so its planes are re-encoded into the source
shape without loss and the grid is the intersection of the two cases' storage
and turbined-flow ranges. The frames keep the shared schemas, with ``source``
``newave`` for the baseline and ``novomodelo`` for the candidate, as the charts read
them. Plants that are not compared (fitted on one side only, unpaired, outside the
ranges or the stages) leave no row and no Diagnostic; reader Diagnostics reach the
enclosing sink tagged with the side.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from novomodelo_bridge.comparators.analyze import build_fpha_comparison
from novomodelo_bridge.comparators.version.alignment import AlignedPair
from novomodelo_bridge.comparators.version.sides import SideInputs, side_tagged
from novomodelo_bridge.novomodelo.readers import (
    read_novomodelo_fpha_planes,
    read_novomodelo_hydro_metadata,
)

_VOLUME_NODES = 5
_FLOW_NODES = 5

_GRID_SCHEMA: dict[str, type[pl.DataType]] = {
    "newave_code": pl.Int64,
    "periodo": pl.Int64,
    "v_min_hm3": pl.Float64,
    "v_max_hm3": pl.Float64,
    "q_min_m3s": pl.Float64,
    "q_max_m3s": pl.Float64,
    "n_v": pl.Int64,
    "n_q": pl.Int64,
    "volume_offset_hm3": pl.Float64,
}


@dataclass(frozen=True)
class PlantRange:
    """The static limits of one plant in one case.

    Attributes:
        v_min_hm3: Minimum absolute storage, hm3.
        v_max_hm3: Maximum absolute storage, hm3.
        q_max_m3s: Maximum turbined flow, m3/s.
    """

    v_min_hm3: float
    v_max_hm3: float
    q_max_m3s: float


def _compared_stages(
    baseline: pl.DataFrame, candidate: pl.DataFrame, stages: Sequence[int]
) -> list[int]:
    held = set(baseline["stage_id"]) & set(candidate["stage_id"])
    return sorted(held & set(stages))


def _by_rank(planes: pl.DataFrame, rank: Mapping[int, int]) -> pl.DataFrame:
    """The planes of the ranked stages, with ``stage_id`` replaced by its rank.

    The shared function pairs the k-th distinct ``periodo`` with candidate stage
    k, so both sides carry ranks, never real stage ids: a gap in the compared
    stages would otherwise pair the wrong stages.
    """
    return planes.filter(pl.col("stage_id").is_in(list(rank))).with_columns(
        pl.col("stage_id").replace_strict(rank, return_dtype=pl.Int64)
    )


def _as_reference_planes(ranked: pl.DataFrame) -> pl.DataFrame:
    """Novomodelo planes in the source shape: ``kappa`` scales the intercept only,
    so it folds into ``gamma_0`` and the correction factor is one."""
    return ranked.select(
        pl.col("hydro_id").cast(pl.Int64).alias("newave_code"),
        pl.col("stage_id").alias("periodo"),
        (pl.col("gamma_0") * pl.col("kappa")).alias("gamma_0"),
        "gamma_v",
        "gamma_q",
        "gamma_s",
        pl.lit(1.0).alias("fator_correcao"),
    )


def _grid(
    pairs: Sequence[AlignedPair],
    baseline_ranges: Mapping[int, PlantRange],
    candidate_ranges: Mapping[int, PlantRange],
    ranks: Sequence[int],
) -> pl.DataFrame:
    """One row per paired plant and rank, over the intersection of both ranges.

    ``volume_offset_hm3`` is 0.0 because the re-encoded planes take absolute
    storage; without the column the shared function would subtract ``v_min_hm3``
    from it and still evaluate. A plant missing from a range map, or whose
    intersection is empty, gets no row.
    """
    rows: list[dict[str, float | int]] = []
    for pair in pairs:
        baseline = baseline_ranges.get(pair.baseline_id)
        candidate = candidate_ranges.get(pair.candidate_id)
        if baseline is None or candidate is None:
            continue
        v_min = max(baseline.v_min_hm3, candidate.v_min_hm3)
        v_max = min(baseline.v_max_hm3, candidate.v_max_hm3)
        q_max = min(baseline.q_max_m3s, candidate.q_max_m3s)
        if v_max < v_min or q_max <= 0.0:
            continue
        rows.extend(
            {
                "newave_code": pair.baseline_id,
                "periodo": rank,
                "v_min_hm3": v_min,
                "v_max_hm3": v_max,
                "q_min_m3s": 0.0,
                "q_max_m3s": q_max,
                "n_v": _VOLUME_NODES if v_max > v_min else 1,
                "n_q": _FLOW_NODES,
                "volume_offset_hm3": 0.0,
            }
            for rank in ranks
        )
    return pl.DataFrame(rows, schema=_GRID_SCHEMA)


def _with_stage_ids(frame: pl.DataFrame, compared: Sequence[int]) -> pl.DataFrame:
    return frame.with_columns(
        pl.col("stage").replace_strict(dict(enumerate(compared)), return_dtype=pl.Int64)
    )


def surface_frames(
    baseline_planes: pl.DataFrame | None,
    candidate_planes: pl.DataFrame | None,
    baseline_ranges: Mapping[int, PlantRange],
    candidate_ranges: Mapping[int, PlantRange],
    hydro_pairs: Sequence[AlignedPair],
    stages: Sequence[int],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """The ``(metrics, surface, spill)`` frames of ``build_fpha_comparison`` for
    two cases' fitted planes, the ``stage`` column holding real stage ids.

    Each plane frame is what ``read_novomodelo_fpha_planes`` returns, or ``None``.
    The range maps are keyed by that side's hydro id; storage is in hm3 and flows
    in m3/s. A plant is compared at each stage of *stages* that both frames hold,
    on a grid whose storage and turbined-flow limits are the intersection of its
    two ranges. The frames are empty, with their schemas, when either side has no
    planes or no stage is compared. ``metrics`` columns ``nmae`` and ``bias`` are
    relative to the baseline, ``max_abs_dev`` is in MW and positive ``bias`` means
    the candidate produces more.
    """
    empty = build_fpha_comparison(None, None, None, {}, {})
    if baseline_planes is None or candidate_planes is None:
        return empty
    compared = _compared_stages(baseline_planes, candidate_planes, stages)
    if not compared:
        return empty
    rank = {stage: i for i, stage in enumerate(compared)}
    metrics, surface, spill = build_fpha_comparison(
        _as_reference_planes(_by_rank(baseline_planes, rank)),
        _grid(hydro_pairs, baseline_ranges, candidate_ranges, range(len(compared))),
        _by_rank(candidate_planes, rank),
        {pair.baseline_id: pair.candidate_id for pair in hydro_pairs},
        {pair.candidate_id: pair.name for pair in hydro_pairs},
    )
    return (
        _with_stage_ids(metrics, compared),
        _with_stage_ids(surface, compared),
        _with_stage_ids(spill, compared),
    )


def _read_planes_and_ranges(
    side: SideInputs,
) -> tuple[pl.DataFrame | None, dict[int, PlantRange]]:
    with side_tagged(side.side):
        planes = read_novomodelo_fpha_planes(side.output_dir)
        metadata = read_novomodelo_hydro_metadata(side.output_dir)
    ranges = {
        hydro_id: PlantRange(
            meta["min_storage_hm3"], meta["max_storage_hm3"], meta["max_turbined_m3s"]
        )
        for hydro_id, meta in metadata.items()
    }
    return planes, ranges


def fpha_surface_comparison(
    baseline: SideInputs,
    candidate: SideInputs,
    hydro_pairs: Sequence[AlignedPair],
    stages: Sequence[int],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Read each side's fitted planes and hydro ranges and return
    :func:`surface_frames` of them.

    A plane file that is present and unreadable raises
    :class:`~novomodelo_bridge.novomodelo.readers.NovomodeloReadError`.
    """
    baseline_planes, baseline_ranges = _read_planes_and_ranges(baseline)
    candidate_planes, candidate_ranges = _read_planes_and_ranges(candidate)
    return surface_frames(
        baseline_planes,
        candidate_planes,
        baseline_ranges,
        candidate_ranges,
        hydro_pairs,
        stages,
    )
