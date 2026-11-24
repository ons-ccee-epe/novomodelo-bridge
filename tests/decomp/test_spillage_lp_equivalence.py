"""Effective spillage bounds are unchanged by trimming the ``UH`` emission.

Replays the old emission (one base ``UH_vert`` row per declared plant and
stage) and the new one (``convert_spillage_upper_bounds`` plus the ceiling on
the ``hydros.json`` entity) through the real accumulator, then compares both
with a test-local oracle of novomodelo's per-column precedence: block row,
else stage row, else entity value, else unbounded.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pyarrow as pa
import pytest

from novomodelo_bridge.decomp.bounds_accumulator import (
    BoundContribution,
    FloorAboveCeilingError,
    build_bound_tables,
    resolve,
)
from novomodelo_bridge.decomp.converters.bounds import convert_spillage_upper_bounds

_HYDRO = 0
_STAGES = (0, 1)
_BLOCK_COUNTS = {0: 3, 1: 3}


def _qver(
    stage_id: int,
    *,
    block_id: int | None = None,
    lower: float | None = None,
    upper: float | None = None,
) -> BoundContribution:
    return BoundContribution(
        family="hydro",
        entity_id=_HYDRO,
        stage_id=stage_id,
        block_id=block_id,
        axis="spillage",
        lower=lower,
        upper=upper,
        contributor="HQ",
    )


def _old_emission(
    declared: Mapping[int, float], stages: Sequence[int]
) -> list[BoundContribution]:
    return [
        BoundContribution(
            family="hydro",
            entity_id=hydro_id,
            stage_id=stage_id,
            block_id=None,
            axis="spillage",
            lower=None,
            upper=ceiling,
            contributor="UH_vert",
        )
        for hydro_id, ceiling in declared.items()
        for stage_id in stages
    ]


def _effective_spillage(
    table: pa.Table,
    entity_min: Mapping[int, float],
    entity_max: Mapping[int, float],
    hydro_id: int,
    stage_id: int,
    block_id: int,
) -> tuple[float | None, float | None]:
    rows = {
        (row["hydro_id"], row["stage_id"], row["block_id"]): row
        for row in table.to_pylist()
    }
    block_row = rows.get((hydro_id, stage_id, block_id), {})
    stage_row = rows.get((hydro_id, stage_id, None), {})

    def pick(column: str, entity: Mapping[int, float]) -> float | None:
        for row in (block_row, stage_row):
            if row.get(column) is not None:
                return row[column]
        return entity.get(hydro_id)

    return pick("min_spillage_m3s", entity_min), pick("max_spillage_m3s", entity_max)


def _old_table(
    declared: Mapping[int, float], others: Sequence[BoundContribution]
) -> pa.Table:
    rows = resolve([*others, *_old_emission(declared, _STAGES)], _BLOCK_COUNTS)
    return build_bound_tables(rows).hydro


def _new_table(
    declared: Mapping[int, float], others: Sequence[BoundContribution]
) -> pa.Table:
    emitted = convert_spillage_upper_bounds(declared, others)
    return build_bound_tables(resolve([*others, *emitted], _BLOCK_COUNTS)).hydro


_UH_ONLY = {_HYDRO: 100.0}


@pytest.mark.parametrize(
    ("declared", "others", "uh_vert_stages"),
    [
        pytest.param(_UH_ONLY, [], set(), id="a-uh-only"),
        pytest.param(_UH_ONLY, [_qver(0, upper=60.0)], {0}, id="b-stage-qver-below-uh"),
        pytest.param(
            _UH_ONLY, [_qver(0, upper=150.0)], {0}, id="c-stage-qver-above-uh"
        ),
        pytest.param(
            _UH_ONLY, [_qver(0, block_id=1, upper=60.0)], {0}, id="d-block-qver"
        ),
        pytest.param({}, [_qver(0, upper=60.0)], set(), id="e-qver-only"),
        pytest.param(
            {_HYDRO: 0.0},
            [_qver(0, block_id=1, upper=40.0)],
            {0},
            id="f-uh-zero-block-qver",
        ),
        pytest.param(
            _UH_ONLY, [_qver(0, lower=30.0)], {0}, id="g-stage-qver-floor-only"
        ),
    ],
)
def test_trimmed_emission_keeps_effective_spillage_bounds(
    declared: dict[int, float],
    others: list[BoundContribution],
    uh_vert_stages: set[int],
) -> None:
    emitted = convert_spillage_upper_bounds(declared, others)
    assert sorted(c.stage_id for c in emitted) == sorted(uh_vert_stages)

    old = _old_table(declared, others)
    new = _new_table(declared, others)

    for stage_id in _STAGES:
        for block_id in range(_BLOCK_COUNTS[stage_id]):
            assert _effective_spillage(
                new, {}, declared, _HYDRO, stage_id, block_id
            ) == _effective_spillage(old, {}, {}, _HYDRO, stage_id, block_id), (
                stage_id,
                block_id,
            )


def test_uh_only_hydro_has_no_spillage_row() -> None:
    new = _new_table(_UH_ONLY, [])

    assert [
        row
        for row in new.to_pylist()
        if row["min_spillage_m3s"] is not None or row["max_spillage_m3s"] is not None
    ] == []


def test_qver_floor_above_uh_ceiling_still_raises() -> None:
    declared = {_HYDRO: 100.0}
    others = [_qver(0, lower=150.0)]

    with pytest.raises(FloorAboveCeilingError):
        _old_table(declared, others)
    with pytest.raises(FloorAboveCeilingError):
        _new_table(declared, others)
