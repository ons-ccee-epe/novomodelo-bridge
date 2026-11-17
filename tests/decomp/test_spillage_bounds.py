"""Tests ``convert_spillage_upper_bounds`` (``UH`` spill ceiling on shared cells)."""

from __future__ import annotations

from novomodelo_bridge.decomp.bounds_accumulator import BoundContribution
from novomodelo_bridge.decomp.converters.bounds import convert_spillage_upper_bounds


def _contribution(
    entity_id: int,
    stage_id: int,
    *,
    block_id: int | None = None,
    family: str = "hydro",
    axis: str = "spillage",
    upper: float | None = 80.0,
    contributor: str = "RHQ_QVER",
) -> BoundContribution:
    return BoundContribution(
        family=family,
        entity_id=entity_id,
        stage_id=stage_id,
        block_id=block_id,
        axis=axis,
        lower=None,
        upper=upper,
        contributor=contributor,
    )


def test_uh_only_cell_emits_nothing() -> None:
    assert convert_spillage_upper_bounds({3: 50.0}, []) == []


def test_shared_cell_emits_one_base_ceiling() -> None:
    other = [_contribution(3, 1, block_id=0)]

    assert convert_spillage_upper_bounds({3: 50.0}, other) == [
        BoundContribution(
            family="hydro",
            entity_id=3,
            stage_id=1,
            block_id=None,
            axis="spillage",
            lower=None,
            upper=50.0,
            contributor="UH_vert",
        )
    ]


def test_unrelated_contributions_emit_nothing() -> None:
    other = [
        _contribution(4, 1),
        _contribution(4, 2, block_id=0),
        _contribution(3, 1, axis="outflow"),
    ]

    assert convert_spillage_upper_bounds({3: 50.0}, other) == []
