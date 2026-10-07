"""Tier-1 tests for the bounds-assembly accumulator.

No deck (synthetic contributions only). Covers the axis registry (contents,
two-sided/stage-level flags — including the diversion/spillage axes
widened/added, and the ``line``/``hydro_unit_group``/
``contract``/``water_withdrawal`` axes registered — unknown-pair
raise), the per-axis ``intersect`` (tightest pair, unbounded sentinel,
empty-intersection and both upper-only/lower-only raises), ``resolve`` (base
vs per-block materialization + the stage-level guard), and
``build_bound_tables`` (the pyarrow fan-out into the hydro/thermal/pumping
schemas, one row per cell — including a two-sided diversion/spillage group
landing both endpoints in one cell).
"""

from __future__ import annotations

import pytest

from novomodelo_bridge.decomp.bounds_accumulator import (
    AXES,
    AxisSpec,
    BoundContribution,
    ResolvedRow,
    _effective,
    axis_spec,
    build_bound_tables,
    intersect,
    resolve,
)


def test_storage_axis_is_stage_level() -> None:
    spec = axis_spec("hydro", "storage")
    assert spec.block_eligible is False
    assert spec.lower_column == "min_storage_hm3"
    assert spec.upper_column == "max_storage_hm3"


def test_diversion_axis_is_two_sided_block_eligible() -> None:
    """Diversion widened from upper-only to two-sided —
    novomodelo's generic-constraint-authoring landed ``min_diversion_m3s``
    alongside the pre-existing ``max_diversion_m3s``."""
    spec = axis_spec("hydro", "diversion")
    assert spec.lower_column == "min_diversion_m3s"
    assert spec.upper_column == "max_diversion_m3s"
    assert spec.block_eligible is True


def test_spillage_axis_is_two_sided_block_eligible() -> None:
    """A brand-new two-sided, block-eligible ``spillage`` axis (the target
    that spillage lowering routes into)."""
    spec = axis_spec("hydro", "spillage")
    assert spec.lower_column == "min_spillage_m3s"
    assert spec.upper_column == "max_spillage_m3s"
    assert spec.block_eligible is True


def test_thermal_and_pumping_axes_registered() -> None:
    spec = axis_spec("thermal", "generation")
    assert spec.family == "thermal"
    assert spec.block_eligible is True
    assert ("pumping", "flow") in AXES


def test_unknown_axis_raises() -> None:
    with pytest.raises(ValueError, match="hydro.*bogus"):
        axis_spec("hydro", "bogus")


def test_line_direct_and_reverse_axes_are_upper_only_block_eligible() -> None:
    """``decomp.network.convert_lines``/``convert_line_bounds`` emit one
    capacity per flow direction (``direct_mw``/``reverse_mw``), not a
    min/max pair — each direction registers as its own upper-only axis."""
    for axis, column in (("direct", "direct_mw"), ("reverse", "reverse_mw")):
        spec = axis_spec("line", axis)
        assert spec.family == "line"
        assert spec.lower_column is None
        assert spec.upper_column == column
        assert spec.block_eligible is True


def test_hydro_unit_group_axes_are_two_sided_block_eligible() -> None:
    """``group_bounds.convert_hydro_unit_group_bounds`` emits four
    block-eligible bound columns forming two two-sided axes."""
    turbined = axis_spec("hydro_unit_group", "turbined")
    assert turbined.lower_column == "min_turbined_m3s"
    assert turbined.upper_column == "max_turbined_m3s"
    assert turbined.block_eligible is True

    generation = axis_spec("hydro_unit_group", "generation")
    assert generation.lower_column == "min_generation_mw"
    assert generation.upper_column == "max_generation_mw"
    assert generation.block_eligible is True


def test_contract_power_axis_is_two_sided_block_eligible() -> None:
    """``contracts.convert_contract_bounds`` emits ``min_mw``/``max_mw`` as a
    two-sided, block-eligible axis; its ``price_per_mwh`` column is a price,
    not a bound, and is never registered in ``AXES``."""
    spec = axis_spec("contract", "power")
    assert spec.family == "contract"
    assert spec.lower_column == "min_mw"
    assert spec.upper_column == "max_mw"
    assert spec.block_eligible is True
    assert ("contract", "price") not in AXES
    assert ("contract", "price_per_mwh") not in AXES


def test_hydro_water_withdrawal_axis_is_lower_only_stage_level() -> None:
    """``bounds.convert_irrigation_withdrawal`` supplies a per-(hydro, stage)
    compulsory removal with no per-block dimension — registered lower-only
    and stage-level, mirroring ``storage``'s ``block_eligible=False``."""
    spec = axis_spec("hydro", "water_withdrawal")
    assert spec.family == "hydro"
    assert spec.lower_column == "water_withdrawal_m3s"
    assert spec.upper_column is None
    assert spec.block_eligible is False


def test_pre_existing_eight_axes_unchanged() -> None:
    """Every previously registered axis resolves to the exact
    same :class:`AxisSpec` it always did — the six new families/axes are
    additive only."""
    expected = {
        ("hydro", "turbined"): ("min_turbined_m3s", "max_turbined_m3s", True),
        ("hydro", "outflow"): ("min_outflow_m3s", "max_outflow_m3s", True),
        ("hydro", "generation"): ("min_generation_mw", "max_generation_mw", True),
        ("hydro", "storage"): ("min_storage_hm3", "max_storage_hm3", False),
        ("hydro", "diversion"): ("min_diversion_m3s", "max_diversion_m3s", True),
        ("hydro", "spillage"): ("min_spillage_m3s", "max_spillage_m3s", True),
        ("thermal", "generation"): ("min_generation_mw", "max_generation_mw", True),
        ("pumping", "flow"): ("min_m3s", "max_m3s", True),
    }
    assert len(expected) == 8
    for (family, axis), (lower, upper, block_eligible) in expected.items():
        spec = axis_spec(family, axis)
        assert spec.family == family
        assert spec.lower_column == lower
        assert spec.upper_column == upper
        assert spec.block_eligible is block_eligible


def test_hydro_turbined_and_outflow_axes_are_two_sided_block_eligible() -> None:
    for axis in ("turbined", "outflow", "generation", "diversion", "spillage"):
        spec = axis_spec("hydro", axis)
        assert spec.lower_column is not None
        assert spec.upper_column is not None
        assert spec.block_eligible is True


def test_bound_contribution_is_frozen_plain_record() -> None:
    contribution = BoundContribution(
        family="hydro",
        entity_id=3,
        stage_id=1,
        block_id=None,
        axis="turbined",
        lower=0.0,
        upper=120.5,
        contributor="RQ",
    )
    assert contribution.family == "hydro"
    assert contribution.block_id is None
    with pytest.raises(AttributeError):
        contribution.upper = 130.0  # type: ignore[misc]


def test_axis_spec_is_frozen() -> None:
    spec = AxisSpec(
        family="hydro",
        lower_column="min_x",
        upper_column="max_x",
        block_eligible=True,
    )
    with pytest.raises(AttributeError):
        spec.block_eligible = False  # type: ignore[misc]


def _outflow_contribution(
    *, lower: float | None, upper: float | None, contributor: str
) -> BoundContribution:
    return BoundContribution(
        family="hydro",
        entity_id=7,
        stage_id=2,
        block_id=None,
        axis="outflow",
        lower=lower,
        upper=upper,
        contributor=contributor,
    )


def test_intersect_tightest_pair() -> None:
    spec = axis_spec("hydro", "outflow")
    contribs = [
        _outflow_contribution(lower=10.0, upper=100.0, contributor="RQ"),
        _outflow_contribution(lower=20.0, upper=80.0, contributor="RE_12"),
    ]
    assert intersect(contribs, spec) == (20.0, 80.0)


def test_effective_normalizes_unbounded_sentinel_to_none() -> None:
    """``_effective`` delegates to the shared
    :func:`novomodelo_bridge.core.generic_constraint_builder.is_bounded` sentinel check
    (single source) — a magnitude at the sentinel normalizes to
    ``None``, while a genuine bound passes through unchanged."""
    assert _effective(1e21) is None
    assert _effective(5.0) == 5.0


def test_intersect_ignores_unbounded_sentinel() -> None:
    spec = axis_spec("hydro", "outflow")
    contribs = [
        _outflow_contribution(lower=None, upper=1e21, contributor="RQ"),
        _outflow_contribution(lower=None, upper=50.0, contributor="HQ_164"),
    ]
    assert intersect(contribs, spec) == (None, 50.0)


def test_empty_intersection_raises() -> None:
    spec = axis_spec("hydro", "outflow")
    contribs = [
        _outflow_contribution(lower=90.0, upper=None, contributor="RQ"),
        _outflow_contribution(lower=None, upper=50.0, contributor="HQ_164"),
    ]
    with pytest.raises(ValueError, match="RQ.*HQ_164"):
        intersect(contribs, spec)


def test_lower_on_upper_only_axis_raises() -> None:
    """``intersect`` enforces the upper-only guard independently of any
    axis actually registered in :data:`AXES` — built directly, not looked
    up via :func:`axis_spec`, mirroring ``test_axis_spec_is_frozen``'s
    approach (``("line", "direct")`` below covers the live-registry case)."""
    spec = AxisSpec(
        family="hydro",
        lower_column=None,
        upper_column="max_x",
        block_eligible=True,
    )
    contribs = [
        BoundContribution(
            family="hydro",
            entity_id=3,
            stage_id=1,
            block_id=None,
            axis="x",
            lower=5.0,
            upper=None,
            contributor="RD_9",
        )
    ]
    with pytest.raises(ValueError, match="upper-only"):
        intersect(contribs, spec)


def test_lower_on_registered_line_direct_axis_raises() -> None:
    """A genuinely registered upper-only axis (``line``'s ``direct``,
    added here) still loud-fails on a lower contribution."""
    spec = axis_spec("line", "direct")
    contribs = [
        BoundContribution(
            family="line",
            entity_id=2,
            stage_id=0,
            block_id=None,
            axis="direct",
            lower=15.0,
            upper=None,
            contributor="patamar-factor",
        )
    ]
    with pytest.raises(ValueError, match="upper-only"):
        intersect(contribs, spec)


def test_upper_on_registered_water_withdrawal_axis_raises() -> None:
    """The symmetric guard: a genuinely registered lower-only axis
    (``hydro``'s ``water_withdrawal``, added here) loud-fails on
    an upper contribution."""
    spec = axis_spec("hydro", "water_withdrawal")
    contribs = [
        BoundContribution(
            family="hydro",
            entity_id=5,
            stage_id=0,
            block_id=None,
            axis="water_withdrawal",
            lower=None,
            upper=3.0,
            contributor="TI",
        )
    ]
    with pytest.raises(ValueError, match="lower-only"):
        intersect(contribs, spec)


def test_all_base_single_row() -> None:
    contribs = [
        _outflow_contribution(lower=10.0, upper=100.0, contributor="RQ"),
        _outflow_contribution(lower=20.0, upper=80.0, contributor="RE_12"),
    ]
    rows = resolve(contribs, {0: 3})
    assert len(rows) == 1
    row = rows[0]
    assert row.block_id is None
    assert row.family == "hydro"
    assert row.axis == "outflow"
    assert (row.lower, row.upper) == (20.0, 80.0)


def test_per_block_materializes_all_blocks() -> None:
    contribs = [
        BoundContribution(
            family="hydro",
            entity_id=7,
            stage_id=0,
            block_id=None,
            axis="outflow",
            lower=10.0,
            upper=None,
            contributor="RQ",
        ),
        BoundContribution(
            family="hydro",
            entity_id=7,
            stage_id=0,
            block_id=1,
            axis="outflow",
            lower=20.0,
            upper=None,
            contributor="RQ_block1",
        ),
    ]
    rows = resolve(contribs, {0: 3})
    by_block = {row.block_id: row for row in rows}
    assert set(by_block) == {0, 1, 2}
    assert by_block[1].lower == 20.0
    assert by_block[0].lower == 10.0
    assert by_block[2].lower == 10.0
    assert None not in by_block


def test_block_id_on_stage_level_raises() -> None:
    contribs = [
        BoundContribution(
            family="hydro",
            entity_id=4,
            stage_id=0,
            block_id=1,
            axis="storage",
            lower=None,
            upper=500.0,
            contributor="VOLMAX",
        ),
    ]
    with pytest.raises(ValueError, match="storage.*entity_id=4"):
        resolve(contribs, {0: 3})


def test_out_of_range_block_id_raises() -> None:
    # block_id 5 is outside the stage's 3-block range; without a guard it would
    # match no materialization iteration and the bound would be silently lost.
    contribs = [
        BoundContribution(
            family="hydro",
            entity_id=7,
            stage_id=0,
            block_id=5,
            axis="outflow",
            lower=30.0,
            upper=None,
            contributor="HQ_99",
        ),
    ]
    with pytest.raises(ValueError, match=r"out of range.*entity_id=7"):
        resolve(contribs, {0: 3})


def test_empty_group_skipped() -> None:
    contribs = [
        _outflow_contribution(lower=None, upper=1e21, contributor="RQ"),
    ]
    assert resolve(contribs, {2: 3}) == []


def test_single_axis_row() -> None:
    row = ResolvedRow(
        family="hydro",
        entity_id=3,
        stage_id=0,
        block_id=None,
        axis="outflow",
        lower=10.0,
        upper=100.0,
    )
    tables = build_bound_tables([row])

    assert tables.hydro.num_rows == 1
    record = tables.hydro.to_pylist()[0]
    assert record["hydro_id"] == 3
    assert record["stage_id"] == 0
    assert record["block_id"] is None
    assert record["min_outflow_m3s"] == 10.0
    assert record["max_outflow_m3s"] == 100.0
    other_axis_columns = {
        "min_turbined_m3s",
        "max_turbined_m3s",
        "min_generation_mw",
        "max_generation_mw",
        "min_storage_hm3",
        "max_storage_hm3",
        "min_diversion_m3s",
        "max_diversion_m3s",
        "min_spillage_m3s",
        "max_spillage_m3s",
    }
    for column in other_axis_columns:
        assert record[column] is None


def test_two_axes_collapse_to_one_row() -> None:
    rows = [
        ResolvedRow(
            family="hydro",
            entity_id=3,
            stage_id=0,
            block_id=None,
            axis="outflow",
            lower=10.0,
            upper=100.0,
        ),
        ResolvedRow(
            family="hydro",
            entity_id=3,
            stage_id=0,
            block_id=None,
            axis="turbined",
            lower=5.0,
            upper=None,
        ),
    ]
    tables = build_bound_tables(rows)

    assert tables.hydro.num_rows == 1
    record = tables.hydro.to_pylist()[0]
    assert record["min_outflow_m3s"] == 10.0
    assert record["max_outflow_m3s"] == 100.0
    assert record["min_turbined_m3s"] == 5.0
    assert record["max_turbined_m3s"] is None


def test_thermal_table_isolated() -> None:
    row = ResolvedRow(
        family="thermal",
        entity_id=2,
        stage_id=1,
        block_id=None,
        axis="generation",
        lower=50.0,
        upper=500.0,
    )
    tables = build_bound_tables([row])

    assert tables.thermal.num_rows == 1
    record = tables.thermal.to_pylist()[0]
    assert record["thermal_id"] == 2
    assert record["min_generation_mw"] == 50.0
    assert record["max_generation_mw"] == 500.0
    assert tables.hydro.num_rows == 0
    assert tables.pumping.num_rows == 0


def test_diversion_upper_only_column() -> None:
    """An upper-only diversion group still
    emits ``max_diversion_m3s`` alone — ``min_diversion_m3s`` is a real,
    nullable column in the widened schema, but stays ``null`` rather than
    picking up a spurious value."""
    row = ResolvedRow(
        family="hydro",
        entity_id=1,
        stage_id=0,
        block_id=None,
        axis="diversion",
        lower=None,
        upper=40.0,
    )
    tables = build_bound_tables([row])

    assert tables.hydro.num_rows == 1
    record = tables.hydro.to_pylist()[0]
    assert record["max_diversion_m3s"] == 40.0
    assert record["min_diversion_m3s"] is None


def test_diversion_two_sided_group_resolves_and_fans_out_both_columns() -> None:
    """A two-sided diversion group — one
    contributor supplying only the lower, another only the upper — resolves
    through ``resolve`` to one row and fans out through
    ``build_bound_tables`` into ``min_diversion_m3s``/``max_diversion_m3s``
    on the *same* hydro cell."""
    contribs = [
        BoundContribution(
            family="hydro",
            entity_id=3,
            stage_id=0,
            block_id=None,
            axis="diversion",
            lower=10.0,
            upper=None,
            contributor="HQ_1",
        ),
        BoundContribution(
            family="hydro",
            entity_id=3,
            stage_id=0,
            block_id=None,
            axis="diversion",
            lower=None,
            upper=40.0,
            contributor="HQ_2",
        ),
    ]
    rows = resolve(contribs, {0: 1})
    assert len(rows) == 1
    assert (rows[0].lower, rows[0].upper) == (10.0, 40.0)

    tables = build_bound_tables(rows)
    assert tables.hydro.num_rows == 1
    record = tables.hydro.to_pylist()[0]
    assert record["min_diversion_m3s"] == 10.0
    assert record["max_diversion_m3s"] == 40.0


def test_spillage_two_sided_group_resolves_and_fans_out_both_columns() -> None:
    """The resolved spillage group lands
    ``min_spillage_m3s``/``max_spillage_m3s`` in the same hydro cell,
    reusing the registered spillage axis."""
    contribs = [
        BoundContribution(
            family="hydro",
            entity_id=2,
            stage_id=0,
            block_id=None,
            axis="spillage",
            lower=5.0,
            upper=None,
            contributor="HQ_9",
        ),
        BoundContribution(
            family="hydro",
            entity_id=2,
            stage_id=0,
            block_id=None,
            axis="spillage",
            lower=None,
            upper=60.0,
            contributor="HQ_10",
        ),
    ]
    rows = resolve(contribs, {0: 1})
    assert len(rows) == 1
    assert (rows[0].lower, rows[0].upper) == (5.0, 60.0)

    tables = build_bound_tables(rows)
    assert tables.hydro.num_rows == 1
    record = tables.hydro.to_pylist()[0]
    assert record["min_spillage_m3s"] == 5.0
    assert record["max_spillage_m3s"] == 60.0


def test_spillage_upper_only_row_leaves_min_null() -> None:
    row = ResolvedRow(
        family="hydro",
        entity_id=1,
        stage_id=0,
        block_id=None,
        axis="spillage",
        lower=None,
        upper=25.0,
    )
    tables = build_bound_tables([row])

    record = tables.hydro.to_pylist()[0]
    assert record["max_spillage_m3s"] == 25.0
    assert record["min_spillage_m3s"] is None
