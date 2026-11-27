"""Operating points of a version comparison: where each case runs its hydro
plants and what each case's fitted planes make of the other's points.

The baseline plays the source role of the shared operating-point analysis and the
candidate the Novomodelo role, so every ``newave_*`` or ``source_*`` field holds
the baseline and every ``novomodelo_*`` field the candidate. The plant's exact
production function is not available between two Novomodelo outputs, so the
columns that need it (``rho_exact_*``, ``fit_error_*``) stay null and the fit
inputs and slice curves stay empty. Reader Diagnostics reach the enclosing sink
tagged with the side.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

import numpy as np
import polars as pl

from novomodelo_bridge.comparators.fpha import (
    FPHA_SPLIT_SCHEMA,
    dense_grid,
    novomodelo_plane_arrays,
    productivity_split,
)
from novomodelo_bridge.comparators.operating_points import (
    OperatingPointComparison,
    flag_ties,
    masked_hydro,
    novomodelo_points,
    tie_share,
)
from novomodelo_bridge.comparators.version.alignment import AlignedPair
from novomodelo_bridge.comparators.version.sides import SideInputs, side_tagged
from novomodelo_bridge.novomodelo.readers import read_novomodelo_fpha_planes

_NULL_COLUMNS = (
    "rho_exact_at_source",
    "rho_exact_at_novomodelo",
    "fit_error_source",
    "fit_error_novomodelo",
)


def _envelope(planes: pl.DataFrame, points: pl.DataFrame) -> np.ndarray:
    # No volume offset: both sides' gamma_v multiply absolute storage.
    return dense_grid(
        *novomodelo_plane_arrays(planes),
        points.get_column("v_hm3").to_numpy(),
        points.get_column("q_m3s").to_numpy(),
        points.get_column("s_m3s").to_numpy(),
    )


def split_at_points(
    source_points: pl.DataFrame,
    candidate_points: pl.DataFrame,
    baseline_planes: Mapping[tuple[int, int], pl.DataFrame],
    candidate_planes: Mapping[tuple[int, int], pl.DataFrame],
    names: Mapping[int, str],
) -> pl.DataFrame:
    """The productivity gap between the cases, one row per paired plant and stage,
    in ``FPHA_SPLIT_SCHEMA``.

    *source_points* are the baseline's points, ``code`` its hydro id and
    ``novomodelo_id`` the paired candidate id; *candidate_points* carry the
    candidate's id in ``novomodelo_id``; both hold the ``tie`` column of
    ``flag_ties``. The plane mappings are keyed by ``(hydro id, stage)`` with that
    side's own id; each value holds ``gamma_0``, ``gamma_v``, ``gamma_q``,
    ``gamma_s`` and ``kappa``. *names* maps the candidate id to the plant's name.

    ``d_total`` is the candidate's productivity over the baseline's minus one, so
    it is positive when the candidate turbines more generation per m3/s.
    ``d_model`` sets the candidate's planes against the baseline's realized
    generation at the baseline's points; ``d_operating_point`` sets where the
    candidate operates against where the baseline operates, under the candidate's
    planes. A plant-stage without planes on either side has no row.
    """
    parts = []
    for (code, candidate_id, stage), group in source_points.partition_by(
        ["code", "novomodelo_id", "stage"], as_dict=True
    ).items():
        baseline = baseline_planes.get((code, stage))
        candidate = candidate_planes.get((candidate_id, stage))
        if baseline is None or candidate is None:
            continue
        parts.append(
            group.with_columns(
                pl.Series("source_planes_mw", _envelope(baseline, group), pl.Float64),
                pl.Series(
                    "novomodelo_planes_mw", _envelope(candidate, group), pl.Float64
                ),
            )
        )
    if not parts:
        return pl.DataFrame(schema=FPHA_SPLIT_SCHEMA)
    no_exact = pl.lit(None, dtype=pl.Float64).alias("exact_mw")
    return (
        productivity_split(
            pl.concat(parts).with_columns(no_exact),
            candidate_points.with_columns(no_exact),
        )
        .with_columns(
            pl.col("novomodelo_id")
            .replace_strict(dict(names), default=None, return_dtype=pl.Utf8)
            .alias("plant_name"),
            # Overwrites the 0.0 that productivity_split gives rho_exact_* for an
            # all-null exact_mw: a zero would read as an exact function producing nothing.
            *(pl.lit(None, dtype=pl.Float64).alias(name) for name in _NULL_COLUMNS),
        )
        .select(list(FPHA_SPLIT_SCHEMA))
    )


def _read_points(
    side: SideInputs, to_code: Mapping[int, int]
) -> tuple[pl.DataFrame, dict[tuple[int, int], pl.DataFrame]]:
    with side_tagged(side.side):
        points = novomodelo_points(side.output_dir, to_code)
        planes = read_novomodelo_fpha_planes(side.output_dir)
    if planes is None:
        return points, {}
    return points, {
        (int(key[0]), int(key[1])): frame
        for key, frame in planes.partition_by(
            ["hydro_id", "stage_id"], as_dict=True
        ).items()
    }


def _in_stages(points: pl.DataFrame, stages: Collection[int]) -> pl.DataFrame:
    # An output without per-block hydro operation gives a frame with no columns.
    if points.is_empty():
        return points
    return points.filter(pl.col("stage").is_in(list(stages)))


def operating_point_comparison(
    baseline: SideInputs,
    candidate: SideInputs,
    hydro_pairs: Sequence[AlignedPair],
    stages: Collection[int],
) -> OperatingPointComparison:
    """The ``split``, ``tie_share`` and ``masked_hydro`` frames of the two cases'
    per-block hydro operation over the paired plants and the *stages* both hold.

    Every other frame of the result keeps its typed empty default. The comparison
    is empty when either case has no points for those plants and stages. ``split``
    leaves out the plant-stages that lack fitted planes on either side;
    ``tie_share`` and ``masked_hydro`` need none. Reader failures propagate
    unchanged.
    """
    source, baseline_planes = _read_points(
        baseline, {pair.baseline_id: pair.baseline_id for pair in hydro_pairs}
    )
    candidate_points, candidate_planes = _read_points(
        candidate, {pair.candidate_id: pair.baseline_id for pair in hydro_pairs}
    )
    source = _in_stages(source, stages)
    candidate_points = _in_stages(candidate_points, stages)
    if source.is_empty() or candidate_points.is_empty():
        return OperatingPointComparison()
    source = source.with_columns(
        pl.col("novomodelo_id").replace_strict(
            {pair.baseline_id: pair.candidate_id for pair in hydro_pairs},
            return_dtype=pl.Int64,
        )
    )
    source, candidate_points = flag_ties(source, candidate_points)
    names = {pair.candidate_id: pair.name for pair in hydro_pairs}
    return OperatingPointComparison(
        split=split_at_points(
            source, candidate_points, baseline_planes, candidate_planes, names
        ),
        tie_share=tie_share(source, candidate_points),
        masked_hydro=masked_hydro(source, candidate_points, names),
    )
