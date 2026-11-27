"""Paired stage rows of a version comparison.

The baseline occupies the ``newave_*`` fields of ``ResultComparison`` and the
candidate the ``novomodelo_*`` fields, whichever two runs are compared; ``rel_diff``
is relative to the baseline. Values are the readers' native-unit means, passed
through unconverted and never taken from the encoded render slots, so two identical
runs differ by exactly zero. The functions emit no Diagnostic and read no file.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence

import polars as pl

from novomodelo_bridge.comparators.model import ResultComparison, result_diff
from novomodelo_bridge.comparators.version.alignment import AlignedPair
from novomodelo_bridge.comparators.version.labels import SIDES


def result_row(
    entity_type: str,
    name: str,
    baseline_id: int,
    candidate_id: int,
    stage: int,
    variable: str,
    baseline_value: float,
    candidate_value: float,
) -> ResultComparison:
    """One row of a version comparison, filled by the slot rule above."""
    abs_diff, rel_diff = result_diff(baseline_value, candidate_value)
    return ResultComparison(
        entity_type=entity_type,
        entity_name=name,
        newave_code=baseline_id,
        novomodelo_id=candidate_id,
        stage=stage,
        variable=variable,
        newave_value=baseline_value,
        novomodelo_value=candidate_value,
        abs_diff=abs_diff,
        rel_diff=rel_diff,
    )


def paired_stage_rows(
    entity_type: str,
    baseline: pl.DataFrame,
    candidate: pl.DataFrame,
    pairs: Sequence[AlignedPair],
    variables: Sequence[str],
    stages: Collection[int],
) -> list[ResultComparison]:
    """One row per pair, stage and variable that both sides hold a finite value for.

    Each frame holds one row per ``(entity_id, stage_id)`` with a column per
    variable. A pair's ``baseline_id`` indexes *baseline* and its ``candidate_id``
    *candidate*; the frames are never joined on ``entity_id``, because the ids differ
    between runs. A null, NaN or infinite value, or a missing entity or stage row,
    builds no row, so a missing value never reads as a zero. Rows follow *pairs*,
    then ascending stage, then *variables*.

    Raises:
        ValueError: a variable is not a column of both frames, or a frame holds
            more than one row for an ``(entity_id, stage_id)`` pair; the message
            names *entity_type* and the variable or the side. The availability
            rule has already removed such a variable and each reader returns one
            row per pair, so both are internal invariants.
    """
    for variable in variables:
        if variable not in baseline.columns or variable not in candidate.columns:
            raise ValueError(
                f"{entity_type}: variable {variable!r} is not a column of both frames"
            )
    for side, frame in zip(SIDES, (baseline, candidate), strict=True):
        if frame.select("entity_id", "stage_id").is_duplicated().any():
            raise ValueError(
                f"{entity_type}: the {side} frame holds more than one row "
                "for an (entity_id, stage_id) pair"
            )
    columns = ["entity_id", "stage_id", *variables]
    baseline_rows = {
        (row["entity_id"], row["stage_id"]): row
        for row in baseline.select(columns).iter_rows(named=True)
    }
    candidate_rows = {
        (row["entity_id"], row["stage_id"]): row
        for row in candidate.select(columns).iter_rows(named=True)
    }
    ordered_stages = sorted(set(stages))
    rows: list[ResultComparison] = []
    for pair in pairs:
        for stage in ordered_stages:
            baseline_row = baseline_rows.get((pair.baseline_id, stage))
            candidate_row = candidate_rows.get((pair.candidate_id, stage))
            if baseline_row is None or candidate_row is None:
                continue
            for variable in variables:
                baseline_value = baseline_row[variable]
                candidate_value = candidate_row[variable]
                if (
                    baseline_value is None
                    or candidate_value is None
                    or not math.isfinite(baseline_value)
                    or not math.isfinite(candidate_value)
                ):
                    continue
                rows.append(
                    result_row(
                        entity_type,
                        pair.name,
                        pair.baseline_id,
                        pair.candidate_id,
                        stage,
                        variable,
                        baseline_value,
                        candidate_value,
                    )
                )
    return rows
