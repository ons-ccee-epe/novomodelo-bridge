"""Tier-1 tests for the RE/RHQ single-term bound producers.

Synthetic ``ConstraintCensus``/``ConstraintRecord``/``StageBounds`` only — no
deck, no ``example/`` read, no ``import novomodelo``. Covers the coefficient-sign
side map, the per-block emission loop's block-count clamp, the "both sides
absent" skip, the RHQ ``QDEF``/``QTUR``/``QDES``/``QVER`` axis lowering
(including their coexistence on one plant and, for ``QDES``/``QVER``, the
``lowers_to_bound`` single-unit-term vs multi-term/non-unit-coefficient
classification), the fail-loud dispatcher on an unhandled family (an ``HV``
record is left to its own lowering), and the single-term ``VARM`` ``HV``
lowering to absolute storage contributions and the volume-tipo ``HV`` lowering
(``VDEF``/``VDES``/``VBOM``) to unshifted hm3 contributions.
"""

from __future__ import annotations

import dataclasses
import math
import re
from datetime import date
from pathlib import Path

import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.diagnostics import Severity
from novomodelo_bridge.core.errors import (
    RhvBoundConflictError,
    diagnostic_from_exception,
)
from novomodelo_bridge.decomp.bounds_accumulator import (
    BoundContribution,
    EmptyIntersectionError,
    build_bound_tables,
    resolve,
)
from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.constraint_registers import (
    ConstraintCensus,
    ConstraintRecord,
    ConstraintTerm,
    StageBounds,
    lowers_to_bound,
)
from novomodelo_bridge.decomp.converters.cadastro import (
    EffectiveCadastro,
    storage_envelope,
)
from novomodelo_bridge.decomp.converters.single_term_bounds import (
    HydroCapacities,
    rhv_bound_conflict_error,
    single_term_bound_contributions,
    single_term_hv_contributions,
    split_plant_group_contributions,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.temporal import OperativeStage
from tests.conftest import hv_floor_frame, make_decomp_case


def _stage(index: int, n_blocks: int) -> OperativeStage:
    return OperativeStage(
        index=index,
        start_date=date(2026, 7, 4),
        end_date=date(2026, 7, 11),
        season_id=6,
        block_hours=tuple(168.0 / n_blocks for _ in range(n_blocks)),
    )


def _case(calendar: list[OperativeStage]) -> DecompCase:
    return make_decomp_case(Path("unused"), calendar=calendar)


def _census(*records: ConstraintRecord) -> ConstraintCensus:
    by_family: dict[str, tuple[ConstraintRecord, ...]] = {}
    for record in records:
        by_family[record.family] = (*by_family.get(record.family, ()), record)
    return ConstraintCensus(by_family=by_family, to_bounds=tuple(records))


@pytest.fixture
def id_map() -> DecompIdMap:
    return DecompIdMap(
        bus_codes=(1,), bus_names=("SE",), hydro_codes=(5,), thermal_codes=(5,)
    )


def _capacities(
    hydro_id: int = 0,
    *,
    max_generation_mw: float = 0.0,
    max_turbined_m3s: float = 0.0,
) -> dict[int, HydroCapacities]:
    """One hydro's ``HydroCapacities`` map, keyed by *hydro_id* — the shape
    ``single_term_bound_contributions`` requires. A test sets only the axis
    its scenario actually clamps against; the other axis defaults to ``0.0``
    (unread by that scenario)."""
    return {
        hydro_id: HydroCapacities(
            max_generation_mw=max_generation_mw, max_turbined_m3s=max_turbined_m3s
        )
    }


def _re_record(
    *,
    constraint_id: int,
    coefficient: float,
    bounds: dict[int, StageBounds],
    code: int = 5,
) -> ConstraintRecord:
    return ConstraintRecord(
        family="RE",
        constraint_id=constraint_id,
        stage_start=min(bounds),
        stage_end=max(bounds),
        terms=(
            ConstraintTerm(code=code, coefficient=coefficient, variable="generation"),
        ),
        bounds=bounds,
        per_block=True,
    )


def _hv_record(
    *,
    coefficient: float,
    bounds: dict[int, StageBounds],
    constraint_id: int = 7,
    code: int = 5,
    variable: str = "VARM",
) -> ConstraintRecord:
    return ConstraintRecord(
        family="HV",
        constraint_id=constraint_id,
        stage_start=min(bounds),
        stage_end=max(bounds),
        terms=(ConstraintTerm(code=code, coefficient=coefficient, variable=variable),),
        bounds=bounds,
        per_block=False,
    )


def _hv_effective(
    n_stages: int,
    stage_varying: dict[tuple[int, str], tuple[float, ...]] | None = None,
) -> EffectiveCadastro:
    """Plant 5 with a 100 hm3 floor and a 1100 hm3 ceiling unless *stage_varying* moves them."""
    return EffectiveCadastro(
        base=hv_floor_frame({5: 100.0}),
        n_stages=n_stages,
        stage_varying=stage_varying or {},
    )


_HV_PUMPING_STATION_IDS = {4: 2}
_HV_DIVERSION_HYDRO_IDS = frozenset({0})


def _hv_contributions(
    record: ConstraintRecord,
    id_map: DecompIdMap,
    effective: EffectiveCadastro,
) -> list[BoundContribution]:
    return single_term_hv_contributions(
        _case([_stage(index, 1) for index in range(effective.n_stages)]),
        id_map,
        census=_census(record),
        effective=effective,
        pumping_station_ids=_HV_PUMPING_STATION_IDS,
        diversion_hydro_ids=_HV_DIVERSION_HYDRO_IDS,
    )


def _resolved_storage_rows(
    contributions: list[BoundContribution], n_stages: int
) -> dict[int, dict[str, float | None]]:
    rows = build_bound_tables(
        resolve(contributions, dict.fromkeys(range(n_stages), 1))
    ).hydro.to_pylist()
    assert all(row["block_id"] is None for row in rows)
    return {row["stage_id"]: row for row in rows}


def test_re_single_block_generation_bound(id_map: DecompIdMap) -> None:
    record = _re_record(
        constraint_id=12,
        coefficient=1.0,
        bounds={0: StageBounds(lower=(50.0,), upper=(212.0,))},
    )
    calendar = [_stage(0, 1)]

    contributions = single_term_bound_contributions(
        _case(calendar),
        id_map,
        census=_census(record),
        pumping_station_ids={},
        hydro_capacities=_capacities(max_generation_mw=1_000.0),
    )

    assert len(contributions) == 1
    contribution = contributions[0]
    assert contribution.family == "hydro"
    assert contribution.entity_id == 0
    assert contribution.stage_id == 0
    assert contribution.block_id == 0
    assert contribution.axis == "generation"
    assert contribution.lower == 50.0
    assert contribution.upper == 212.0
    assert contribution.contributor == "RE_12"


def test_re_negative_coefficient_flips_sides(id_map: DecompIdMap) -> None:
    record = _re_record(
        constraint_id=12,
        coefficient=-1.0,
        bounds={0: StageBounds(lower=(50.0,), upper=(212.0,))},
    )
    calendar = [_stage(0, 1)]

    contributions = single_term_bound_contributions(
        _case(calendar),
        id_map,
        census=_census(record),
        pumping_station_ids={},
        hydro_capacities=_capacities(max_generation_mw=1_000.0),
    )

    assert len(contributions) == 1
    assert contributions[0].lower == -212.0
    assert contributions[0].upper == -50.0


def test_re_sign_change_between_stages_flips_that_stage_only(
    id_map: DecompIdMap,
) -> None:
    bounds = {s: StageBounds(lower=(50.0,), upper=(212.0,)) for s in range(2)}
    record = dataclasses.replace(
        _re_record(constraint_id=12, coefficient=1.0, bounds=bounds),
        terms=(
            ConstraintTerm(
                code=5,
                coefficient=1.0,
                variable="generation",
                schedule=((0, 1.0), (1, -1.0)),
            ),
        ),
    )
    calendar = [_stage(0, 1), _stage(1, 1)]

    contributions = single_term_bound_contributions(
        _case(calendar),
        id_map,
        census=_census(record),
        pumping_station_ids={},
        hydro_capacities=_capacities(max_generation_mw=1_000.0),
    )

    assert [(c.stage_id, c.lower, c.upper) for c in contributions] == [
        (0, 50.0, 212.0),
        (1, -212.0, -50.0),
    ]


def test_re_two_block_emits_per_block_contributions(id_map: DecompIdMap) -> None:
    record = _re_record(
        constraint_id=12,
        coefficient=1.0,
        bounds={0: StageBounds(lower=(10.0, 20.0), upper=(100.0, 200.0))},
    )
    calendar = [_stage(0, 2)]

    contributions = single_term_bound_contributions(
        _case(calendar),
        id_map,
        census=_census(record),
        pumping_station_ids={},
        hydro_capacities=_capacities(max_generation_mw=1_000.0),
    )

    by_block = {c.block_id: c for c in contributions}
    assert set(by_block) == {0, 1}
    assert by_block[0].lower == 10.0
    assert by_block[0].upper == 100.0
    assert by_block[1].lower == 20.0
    assert by_block[1].upper == 200.0


def _ft_record(
    *,
    constraint_id: int,
    coefficient: float = 1.0,
    bounds: dict[int, StageBounds],
    code: int = 5,
) -> ConstraintRecord:
    return ConstraintRecord(
        family="RE",
        constraint_id=constraint_id,
        stage_start=min(bounds),
        stage_end=max(bounds),
        terms=(
            ConstraintTerm(
                code=code, coefficient=coefficient, variable="thermal_generation"
            ),
        ),
        bounds=bounds,
        per_block=True,
    )


def test_ft_thermal_single_block_generation_bound(id_map: DecompIdMap) -> None:
    """A single-term ``FT`` RE lowers to a ``family="thermal"`` contribution
    on the novomodelo ``generation`` axis, resolved via ``id_map.thermal_id``."""
    record = _ft_record(
        constraint_id=30,
        bounds={0: StageBounds(lower=(50.0,), upper=(212.0,))},
    )
    calendar = [_stage(0, 1)]

    contributions = single_term_bound_contributions(
        _case(calendar),
        id_map,
        census=_census(record),
        pumping_station_ids={},
        hydro_capacities={},
    )

    assert len(contributions) == 1
    contribution = contributions[0]
    assert contribution.family == "thermal"
    assert contribution.entity_id == 0
    assert contribution.stage_id == 0
    assert contribution.block_id == 0
    assert contribution.axis == "generation"
    assert contribution.lower == 50.0
    assert contribution.upper == 212.0
    assert contribution.contributor == "RE_30"


def test_ft_thermal_two_block_emits_per_block_contributions(
    id_map: DecompIdMap,
) -> None:
    record = _ft_record(
        constraint_id=31,
        bounds={0: StageBounds(lower=(10.0, 20.0), upper=(100.0, 200.0))},
    )
    calendar = [_stage(0, 2)]

    contributions = single_term_bound_contributions(
        _case(calendar),
        id_map,
        census=_census(record),
        pumping_station_ids={},
        hydro_capacities={},
    )

    by_block = {c.block_id: c for c in contributions}
    assert set(by_block) == {0, 1}
    assert by_block[0].family == "thermal"
    assert by_block[0].axis == "generation"
    assert by_block[0].lower == 10.0
    assert by_block[0].upper == 100.0
    assert by_block[1].lower == 20.0
    assert by_block[1].upper == 200.0


def test_re_hydro_regression_still_family_hydro(id_map: DecompIdMap) -> None:
    """Regression: a single-hydro-generation RE (the pre-existing ``FU``
    path) still lowers to a ``family="hydro"`` contribution now that the RE
    branch discriminates on the term variable rather than assuming hydro."""
    record = _re_record(
        constraint_id=32,
        coefficient=1.0,
        bounds={0: StageBounds(lower=(50.0,), upper=(212.0,))},
    )
    calendar = [_stage(0, 1)]

    contributions = single_term_bound_contributions(
        _case(calendar),
        id_map,
        census=_census(record),
        pumping_station_ids={},
        hydro_capacities=_capacities(max_generation_mw=1_000.0),
    )

    assert len(contributions) == 1
    assert contributions[0].family == "hydro"
    assert contributions[0].axis == "generation"


def test_absent_both_sides_emits_no_contribution(id_map: DecompIdMap) -> None:
    record = _re_record(
        constraint_id=12,
        coefficient=1.0,
        bounds={0: StageBounds(lower=(None,), upper=(None,))},
    )
    calendar = [_stage(0, 1)]

    contributions = single_term_bound_contributions(
        _case(calendar),
        id_map,
        census=_census(record),
        pumping_station_ids={},
        hydro_capacities=_capacities(max_generation_mw=1_000.0),
    )

    assert contributions == []


def test_unhandled_family_raises(id_map: DecompIdMap) -> None:
    # "RHA" (HA/LA/CA) is read into the census but never lowered by this
    # module — neither to bounds nor generic — so it stands in for a
    # genuinely unhandled family beside the handled "RE" and "HQ".
    record = ConstraintRecord(
        family="RHA",
        constraint_id=164,
        stage_start=0,
        stage_end=0,
        terms=(ConstraintTerm(code=5, coefficient=1.0, variable="VARM"),),
        bounds={0: StageBounds(lower=(0.0,), upper=(100.0,))},
        per_block=True,
    )
    calendar = [_stage(0, 1)]

    with pytest.raises(ValueError, match="RHA"):
        single_term_bound_contributions(
            _case(calendar),
            id_map,
            census=_census(record),
            pumping_station_ids={},
            hydro_capacities={},
        )


def test_hv_record_in_to_bounds_is_left_to_the_hv_lowering(
    id_map: DecompIdMap,
) -> None:
    record = _hv_record(
        coefficient=1.0, bounds={0: StageBounds(lower=(10.0,), upper=(400.0,))}
    )

    contributions = single_term_bound_contributions(
        _case([_stage(0, 1)]),
        id_map,
        census=_census(record),
        pumping_station_ids={},
        hydro_capacities={},
    )

    assert contributions == []


def test_re_unexpected_bounded_variable_raises(id_map: DecompIdMap) -> None:
    """An RE ``to_bounds`` record whose single term is neither
    ``"generation"`` nor ``"thermal_generation"`` fails loud instead of
    silently dropping the bound — this shape is not produced by the reader
    today (``_BOUNDS_AXIS`` has no such RE entry), but the dispatch must not
    silently accept one."""
    record = ConstraintRecord(
        family="RE",
        constraint_id=33,
        stage_start=0,
        stage_end=0,
        terms=(ConstraintTerm(code=5, coefficient=1.0, variable="QDEF"),),
        bounds={0: StageBounds(lower=(0.0,), upper=(100.0,))},
        per_block=True,
    )
    calendar = [_stage(0, 1)]

    with pytest.raises(ValueError, match="QDEF"):
        single_term_bound_contributions(
            _case(calendar),
            id_map,
            census=_census(record),
            pumping_station_ids={},
            hydro_capacities={},
        )


def test_re_ceiling_above_capacity_clamps_upper(id_map: DecompIdMap) -> None:
    """An RE ceiling above the plant's declared capacity is clamped
    down to that capacity — mirroring the real cross-source mismatch
    (a plant's RE ceiling above its own declared, head-derated
    ``max_generation_mw``) — while the lower bound passes through
    unchanged."""
    record = _re_record(
        constraint_id=12,
        coefficient=1.0,
        bounds={0: StageBounds(lower=(50.0,), upper=(11_000.0,))},
    )
    calendar = [_stage(0, 1)]

    contributions = single_term_bound_contributions(
        _case(calendar),
        id_map,
        census=_census(record),
        pumping_station_ids={},
        hydro_capacities=_capacities(max_generation_mw=9_777.776),
    )

    assert len(contributions) == 1
    assert contributions[0].lower == 50.0
    assert contributions[0].upper == 9_777.776


def test_re_ceiling_at_capacity_is_not_clamped_and_emits_no_diagnostic(
    id_map: DecompIdMap,
) -> None:
    """A ceiling at (or below) capacity passes through unchanged —
    including the boundary case where the ceiling exactly equals the
    capacity — and no clamp diagnostic is emitted."""
    record = _re_record(
        constraint_id=12,
        coefficient=1.0,
        bounds={0: StageBounds(lower=(50.0,), upper=(212.0,))},
    )
    calendar = [_stage(0, 1)]

    with dx.collect() as collected:
        contributions = single_term_bound_contributions(
            _case(calendar),
            id_map,
            census=_census(record),
            pumping_station_ids={},
            hydro_capacities=_capacities(max_generation_mw=212.0),
        )

    assert len(contributions) == 1
    assert contributions[0].upper == 212.0
    assert [d for d in collected if d.code == "decomp-re-generation-clamped"] == []


def test_re_clamp_emits_diagnostic_naming_plant_ceiling_and_capacity(
    id_map: DecompIdMap,
) -> None:
    """A firing clamp is captured, inside ``dx.collect()``, as a
    ``decomp-re-generation-clamped`` Diagnostic naming the plant, the RE
    ceiling, and the declared capacity."""
    record = _re_record(
        constraint_id=12,
        coefficient=1.0,
        bounds={0: StageBounds(lower=(None,), upper=(11_000.0,))},
    )
    calendar = [_stage(0, 1)]

    with dx.collect() as collected:
        single_term_bound_contributions(
            _case(calendar),
            id_map,
            census=_census(record),
            pumping_station_ids={},
            hydro_capacities=_capacities(max_generation_mw=9_777.776),
        )

    clamped = [d for d in collected if d.code == "decomp-re-generation-clamped"]
    assert len(clamped) == 1
    assert clamped[0].severity is Severity.WARNING
    assert "Hydro 0" in clamped[0].summary  # the plant
    assert "11000.0" in clamped[0].summary  # the RE ceiling
    assert "9777.776" in clamped[0].summary  # the declared capacity


def test_re_missing_capacity_entry_raises(id_map: DecompIdMap) -> None:
    """A hydro id absent from ``hydro_capacities`` is a wiring bug — the map
    is required, not defaulted, so the lookup fails loud with ``KeyError``
    rather than silently skipping the clamp."""
    record = _re_record(
        constraint_id=12,
        coefficient=1.0,
        bounds={0: StageBounds(lower=(50.0,), upper=(212.0,))},
    )
    calendar = [_stage(0, 1)]

    with pytest.raises(KeyError):
        single_term_bound_contributions(
            _case(calendar),
            id_map,
            census=_census(record),
            pumping_station_ids={},
            hydro_capacities={},
        )


_ITAIPU_CODE = 66


@pytest.fixture
def itaipu_id_map() -> DecompIdMap:
    return DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(5, _ITAIPU_CODE))


def _frequency_record(
    *, constraint_id: int, frequency: float, lower: float, upper: float, code: int = 66
) -> ConstraintRecord:
    record = _re_record(
        constraint_id=constraint_id,
        coefficient=1.0,
        bounds={0: StageBounds(lower=(lower,), upper=(upper,))},
        code=code,
    )
    term = dataclasses.replace(record.terms[0], frequency=frequency)
    return dataclasses.replace(record, terms=(term,))


def _itaipu_group_capacities(cap: float = 7_000.0) -> dict[tuple[int, int], float]:
    return {(1, 0): cap, (1, 1): cap}


def test_frequency_tagged_re_bounds_its_unit_group_not_the_plant(
    itaipu_id_map: DecompIdMap,
) -> None:
    records = (
        _frequency_record(
            constraint_id=461, frequency=50.0, lower=4267.2, upper=6800.0
        ),
        _frequency_record(
            constraint_id=462, frequency=60.0, lower=2000.0, upper=6900.0
        ),
    )
    calendar = [_stage(0, 1)]
    case = _case(calendar)

    plant = single_term_bound_contributions(
        case,
        itaipu_id_map,
        census=_census(*records),
        pumping_station_ids={},
        hydro_capacities={},
    )
    groups = split_plant_group_contributions(
        case,
        itaipu_id_map,
        census=_census(*records),
        group_capacities=_itaipu_group_capacities(),
    )

    assert plant == []
    assert list(groups) == [1]
    assert [
        (c.family, c.entity_id, c.axis, c.lower, c.upper, c.contributor)
        for c in groups[1]
    ] == [
        ("hydro_unit_group", 0, "generation", 4267.2, 6800.0, "RE_461"),
        ("hydro_unit_group", 1, "generation", 2000.0, 6900.0, "RE_462"),
    ]


def test_frequency_tagged_re_clamps_to_group_capacity_with_one_warning_per_group(
    itaipu_id_map: DecompIdMap,
) -> None:
    records = (
        _frequency_record(
            constraint_id=461, frequency=50.0, lower=4267.2, upper=7086.0
        ),
        _frequency_record(
            constraint_id=561, frequency=50.0, lower=4267.2, upper=7086.0
        ),
    )
    calendar = [_stage(0, 1)]

    with dx.collect() as collected:
        groups = split_plant_group_contributions(
            _case(calendar),
            itaipu_id_map,
            census=_census(*records),
            group_capacities=_itaipu_group_capacities(),
        )

    assert [c.upper for c in groups[1]] == [7_000.0, 7_000.0]
    (clamped,) = [d for d in collected if d.code == "decomp-re-generation-clamped"]
    assert "Unit group 0 of hydro 1" in clamped.summary
    assert "constraints 461, 561" in clamped.summary
    assert "7086.0" in clamped.summary


def test_frequency_tag_on_an_unsplit_plant_raises(
    itaipu_id_map: DecompIdMap,
) -> None:
    record = _frequency_record(
        constraint_id=12, frequency=60.0, lower=0.0, upper=100.0, code=5
    )

    with pytest.raises(ValueError, match="plant 5: a 60 Hz generation term"):
        split_plant_group_contributions(
            _case([_stage(0, 1)]),
            itaipu_id_map,
            census=_census(record),
            group_capacities=_itaipu_group_capacities(),
        )


@pytest.fixture
def hq_id_map() -> DecompIdMap:
    return DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(182,))


def _hq_record(
    *,
    constraint_id: int,
    variable: str,
    bounds: dict[int, StageBounds],
    code: int = 182,
    coefficient: float = 1.0,
) -> ConstraintRecord:
    return ConstraintRecord(
        family="HQ",
        constraint_id=constraint_id,
        stage_start=min(bounds),
        stage_end=max(bounds),
        terms=(ConstraintTerm(code=code, coefficient=coefficient, variable=variable),),
        bounds=bounds,
        per_block=True,
    )


def test_hq_qdef_lowers_to_outflow(hq_id_map: DecompIdMap) -> None:
    record = _hq_record(
        constraint_id=164,
        variable="QDEF",
        bounds={0: StageBounds(lower=(30.0,), upper=(None,))},
    )
    calendar = [_stage(0, 1)]

    contributions = single_term_bound_contributions(
        _case(calendar),
        hq_id_map,
        census=_census(record),
        pumping_station_ids={},
        hydro_capacities={},
    )

    assert len(contributions) == 1
    contribution = contributions[0]
    assert contribution.family == "hydro"
    assert contribution.entity_id == 0
    assert contribution.stage_id == 0
    assert contribution.block_id == 0
    assert contribution.axis == "outflow"
    assert contribution.lower == 30.0
    assert contribution.upper is None
    assert contribution.contributor == "HQ_164"


def test_hq_qtur_lowers_to_turbined(hq_id_map: DecompIdMap) -> None:
    record = _hq_record(
        constraint_id=164,
        variable="QTUR",
        bounds={0: StageBounds(lower=(None,), upper=(150.0,))},
    )
    calendar = [_stage(0, 1)]

    contributions = single_term_bound_contributions(
        _case(calendar),
        hq_id_map,
        census=_census(record),
        pumping_station_ids={},
        hydro_capacities=_capacities(max_turbined_m3s=150.0),
    )

    assert len(contributions) == 1
    contribution = contributions[0]
    assert contribution.axis == "turbined"
    assert contribution.lower is None
    assert contribution.upper == 150.0


def test_hq_qtur_ceiling_above_capacity_clamps_upper(hq_id_map: DecompIdMap) -> None:
    """Boundary-review fix 1: a QTUR ceiling above the plant's declared
    ``max_turbined_m3s`` is clamped down to that capacity — mirroring the RE
    ``generation`` clamp for the identical rule-43 exposure (``QTUR`` ceiling
    exceeding a head-derated ``max_turbined_m3s`` declared elsewhere) —
    while the lower bound passes through unchanged."""
    record = _hq_record(
        constraint_id=164,
        variable="QTUR",
        bounds={0: StageBounds(lower=(30.0,), upper=(973.2,))},
    )
    calendar = [_stage(0, 1)]

    contributions = single_term_bound_contributions(
        _case(calendar),
        hq_id_map,
        census=_census(record),
        pumping_station_ids={},
        hydro_capacities=_capacities(max_turbined_m3s=972.0),
    )

    assert len(contributions) == 1
    assert contributions[0].lower == 30.0
    assert contributions[0].upper == 972.0


def test_hq_qtur_ceiling_at_capacity_is_not_clamped_and_emits_no_diagnostic(
    hq_id_map: DecompIdMap,
) -> None:
    """A QTUR ceiling at (or below) capacity passes through unchanged —
    including the boundary case where the ceiling exactly equals the
    capacity — and no clamp diagnostic is emitted."""
    record = _hq_record(
        constraint_id=164,
        variable="QTUR",
        bounds={0: StageBounds(lower=(None,), upper=(150.0,))},
    )
    calendar = [_stage(0, 1)]

    with dx.collect() as collected:
        contributions = single_term_bound_contributions(
            _case(calendar),
            hq_id_map,
            census=_census(record),
            pumping_station_ids={},
            hydro_capacities=_capacities(max_turbined_m3s=150.0),
        )

    assert len(contributions) == 1
    assert contributions[0].upper == 150.0
    assert [d for d in collected if d.code == "decomp-qtur-turbined-clamped"] == []


def test_hq_qtur_clamp_emits_diagnostic_naming_plant_ceiling_and_capacity(
    hq_id_map: DecompIdMap,
) -> None:
    """A firing QTUR clamp is captured, inside ``dx.collect()``, as a
    ``decomp-qtur-turbined-clamped`` Diagnostic naming the plant, the QTUR
    ceiling, and the declared capacity."""
    record = _hq_record(
        constraint_id=164,
        variable="QTUR",
        bounds={0: StageBounds(lower=(None,), upper=(973.2,))},
    )
    calendar = [_stage(0, 1)]

    with dx.collect() as collected:
        single_term_bound_contributions(
            _case(calendar),
            hq_id_map,
            census=_census(record),
            pumping_station_ids={},
            hydro_capacities=_capacities(max_turbined_m3s=972.0),
        )

    clamped = [d for d in collected if d.code == "decomp-qtur-turbined-clamped"]
    assert len(clamped) == 1
    assert clamped[0].severity is Severity.WARNING
    assert "Hydro 0" in clamped[0].summary  # the plant
    assert "973.2" in clamped[0].summary  # the QTUR ceiling
    assert "972.0" in clamped[0].summary  # the declared capacity


def test_hq_qtur_missing_capacity_entry_raises(hq_id_map: DecompIdMap) -> None:
    """A hydro id absent from ``hydro_capacities`` is a wiring bug — the map
    is required, not defaulted, so the ``QTUR``/``turbined`` clamp lookup
    fails loud with ``KeyError`` rather than silently skipping the clamp."""
    record = _hq_record(
        constraint_id=164,
        variable="QTUR",
        bounds={0: StageBounds(lower=(None,), upper=(150.0,))},
    )
    calendar = [_stage(0, 1)]

    with pytest.raises(KeyError):
        single_term_bound_contributions(
            _case(calendar),
            hq_id_map,
            census=_census(record),
            pumping_station_ids={},
            hydro_capacities={},
        )


def test_hq_qdef_does_not_read_hydro_capacities(hq_id_map: DecompIdMap) -> None:
    """QDEF is not a rule-43 axis, so it never indexes ``hydro_capacities`` —
    an empty map (which would ``KeyError`` on the QTUR path) is fine here."""
    record = _hq_record(
        constraint_id=164,
        variable="QDEF",
        bounds={0: StageBounds(lower=(30.0,), upper=(None,))},
    )
    calendar = [_stage(0, 1)]

    contributions = single_term_bound_contributions(
        _case(calendar),
        hq_id_map,
        census=_census(record),
        pumping_station_ids={},
        hydro_capacities={},
    )

    assert len(contributions) == 1
    assert contributions[0].axis == "outflow"


def test_hq_per_block_outflow_contributions(hq_id_map: DecompIdMap) -> None:
    record = _hq_record(
        constraint_id=164,
        variable="QDEF",
        bounds={0: StageBounds(lower=(10.0, 12.0), upper=(80.0, 90.0))},
    )
    calendar = [_stage(0, 2)]

    contributions = single_term_bound_contributions(
        _case(calendar),
        hq_id_map,
        census=_census(record),
        pumping_station_ids={},
        hydro_capacities={},
    )

    by_block = {c.block_id: c for c in contributions}
    assert set(by_block) == {0, 1}
    assert by_block[0].axis == "outflow"
    assert by_block[0].lower == 10.0
    assert by_block[0].upper == 80.0
    assert by_block[1].axis == "outflow"
    assert by_block[1].lower == 12.0
    assert by_block[1].upper == 90.0


def test_hq_qdef_and_qtur_coexist_on_one_plant(hq_id_map: DecompIdMap) -> None:
    qdef = _hq_record(
        constraint_id=164,
        variable="QDEF",
        bounds={0: StageBounds(lower=(30.0,), upper=(None,))},
    )
    qtur = _hq_record(
        constraint_id=165,
        variable="QTUR",
        bounds={0: StageBounds(lower=(None,), upper=(150.0,))},
    )
    calendar = [_stage(0, 1)]

    contributions = single_term_bound_contributions(
        _case(calendar),
        hq_id_map,
        census=_census(qdef, qtur),
        pumping_station_ids={},
        hydro_capacities=_capacities(max_turbined_m3s=150.0),
    )

    by_axis = {c.axis: c for c in contributions}
    assert set(by_axis) == {"outflow", "turbined"}
    assert by_axis["outflow"].entity_id == 0
    assert by_axis["outflow"].lower == 30.0
    assert by_axis["outflow"].contributor == "HQ_164"
    assert by_axis["turbined"].entity_id == 0
    assert by_axis["turbined"].upper == 150.0
    assert by_axis["turbined"].contributor == "HQ_165"


def test_hq_qdes_lowers_to_diversion(hq_id_map: DecompIdMap) -> None:
    """A single ``±1`` ``QDES`` record with ``L <= QDES <=
    U`` emits a hydro ``diversion`` contribution with the sign-mapped
    ``(lower, upper)`` — here the identity map, since the coefficient is
    ``+1``."""
    record = _hq_record(
        constraint_id=200,
        variable="QDES",
        bounds={0: StageBounds(lower=(10.0,), upper=(40.0,))},
    )
    calendar = [_stage(0, 1)]

    contributions = single_term_bound_contributions(
        _case(calendar),
        hq_id_map,
        census=_census(record),
        pumping_station_ids={},
        hydro_capacities={},
    )

    assert len(contributions) == 1
    contribution = contributions[0]
    assert contribution.family == "hydro"
    assert contribution.entity_id == 0
    assert contribution.stage_id == 0
    assert contribution.block_id == 0
    assert contribution.axis == "diversion"
    assert contribution.lower == 10.0
    assert contribution.upper == 40.0
    assert contribution.contributor == "HQ_200"


def test_hq_qdes_negative_coefficient_flips_sides(hq_id_map: DecompIdMap) -> None:
    """Sign-mapped: ``-1*QDES in [L, U]`` becomes
    ``QDES in [-U, -L]`` on the diversion axis."""
    record = _hq_record(
        constraint_id=200,
        variable="QDES",
        bounds={0: StageBounds(lower=(10.0,), upper=(40.0,))},
        coefficient=-1.0,
    )
    calendar = [_stage(0, 1)]

    contributions = single_term_bound_contributions(
        _case(calendar),
        hq_id_map,
        census=_census(record),
        pumping_station_ids={},
        hydro_capacities={},
    )

    assert len(contributions) == 1
    assert contributions[0].axis == "diversion"
    assert contributions[0].lower == -40.0
    assert contributions[0].upper == -10.0


def test_hq_qver_lowers_to_spillage(hq_id_map: DecompIdMap) -> None:
    """A single ``±1`` ``QVER`` record lowers to a hydro
    ``spillage`` contribution with the sign-mapped ``(lower, upper)``."""
    record = _hq_record(
        constraint_id=201,
        variable="QVER",
        bounds={0: StageBounds(lower=(5.0,), upper=(60.0,))},
    )
    calendar = [_stage(0, 1)]

    contributions = single_term_bound_contributions(
        _case(calendar),
        hq_id_map,
        census=_census(record),
        pumping_station_ids={},
        hydro_capacities={},
    )

    assert len(contributions) == 1
    contribution = contributions[0]
    assert contribution.family == "hydro"
    assert contribution.entity_id == 0
    assert contribution.stage_id == 0
    assert contribution.block_id == 0
    assert contribution.axis == "spillage"
    assert contribution.lower == 5.0
    assert contribution.upper == 60.0
    assert contribution.contributor == "HQ_201"


def test_hq_qver_negative_coefficient_flips_sides(hq_id_map: DecompIdMap) -> None:
    """Sign-mapped, mirroring the QDES case above."""
    record = _hq_record(
        constraint_id=201,
        variable="QVER",
        bounds={0: StageBounds(lower=(5.0,), upper=(60.0,))},
        coefficient=-1.0,
    )
    calendar = [_stage(0, 1)]

    contributions = single_term_bound_contributions(
        _case(calendar),
        hq_id_map,
        census=_census(record),
        pumping_station_ids={},
        hydro_capacities={},
    )

    assert len(contributions) == 1
    assert contributions[0].axis == "spillage"
    assert contributions[0].lower == -60.0
    assert contributions[0].upper == -5.0


def test_hq_qdef_qdes_qver_all_coexist_on_one_plant(hq_id_map: DecompIdMap) -> None:
    """The four ``HQ`` flow tipos routed through the same
    ``_hq_flow_contributions``/``_HQ_AXIS_BY_VARIABLE`` lookup all land on
    distinct axes for the same plant and all survive."""
    qdef = _hq_record(
        constraint_id=164,
        variable="QDEF",
        bounds={0: StageBounds(lower=(30.0,), upper=(None,))},
    )
    qdes = _hq_record(
        constraint_id=200,
        variable="QDES",
        bounds={0: StageBounds(lower=(10.0,), upper=(40.0,))},
    )
    qver = _hq_record(
        constraint_id=201,
        variable="QVER",
        bounds={0: StageBounds(lower=(5.0,), upper=(60.0,))},
    )
    calendar = [_stage(0, 1)]

    contributions = single_term_bound_contributions(
        _case(calendar),
        hq_id_map,
        census=_census(qdef, qdes, qver),
        pumping_station_ids={},
        hydro_capacities={},
    )

    by_axis = {c.axis: c for c in contributions}
    assert set(by_axis) == {"outflow", "diversion", "spillage"}
    assert by_axis["diversion"].lower == 10.0
    assert by_axis["diversion"].upper == 40.0
    assert by_axis["spillage"].lower == 5.0
    assert by_axis["spillage"].upper == 60.0


def test_qdes_single_unit_term_lowers_to_bound() -> None:
    """``lowers_to_bound`` returns ``True`` for a single
    ``±1`` ``QDES`` term — the classifier ``single_term_bound_contributions``
    is downstream of."""
    record = ConstraintRecord(
        family="HQ",
        constraint_id=200,
        stage_start=0,
        stage_end=0,
        terms=(ConstraintTerm(code=182, coefficient=1.0, variable="QDES"),),
        bounds={0: StageBounds(lower=(10.0,), upper=(40.0,))},
        per_block=True,
    )
    assert lowers_to_bound(record)


def test_qdes_multi_term_does_not_lower_to_bound() -> None:
    """A multi-term ``QDES`` constraint stays generic."""
    record = ConstraintRecord(
        family="HQ",
        constraint_id=200,
        stage_start=0,
        stage_end=0,
        terms=(
            ConstraintTerm(code=182, coefficient=1.0, variable="QDES"),
            ConstraintTerm(code=183, coefficient=1.0, variable="QDES"),
        ),
        bounds={0: StageBounds(lower=(10.0,), upper=(40.0,))},
        per_block=True,
    )
    assert not lowers_to_bound(record)


def test_qdes_non_unit_coefficient_does_not_lower_to_bound() -> None:
    """A non-unit-coefficient single ``QDES`` term stays
    generic."""
    record = ConstraintRecord(
        family="HQ",
        constraint_id=200,
        stage_start=0,
        stage_end=0,
        terms=(ConstraintTerm(code=182, coefficient=0.5, variable="QDES"),),
        bounds={0: StageBounds(lower=(10.0,), upper=(40.0,))},
        per_block=True,
    )
    assert not lowers_to_bound(record)


def test_qver_single_unit_term_lowers_to_bound() -> None:
    """``lowers_to_bound`` returns ``True`` for a single
    ``±1`` ``QVER`` term."""
    record = ConstraintRecord(
        family="HQ",
        constraint_id=201,
        stage_start=0,
        stage_end=0,
        terms=(ConstraintTerm(code=182, coefficient=-1.0, variable="QVER"),),
        bounds={0: StageBounds(lower=(5.0,), upper=(60.0,))},
        per_block=True,
    )
    assert lowers_to_bound(record)


def test_qver_multi_term_does_not_lower_to_bound() -> None:
    """A multi-term ``QVER`` constraint stays generic."""
    record = ConstraintRecord(
        family="HQ",
        constraint_id=201,
        stage_start=0,
        stage_end=0,
        terms=(
            ConstraintTerm(code=182, coefficient=1.0, variable="QVER"),
            ConstraintTerm(code=183, coefficient=1.0, variable="QVER"),
        ),
        bounds={0: StageBounds(lower=(5.0,), upper=(60.0,))},
        per_block=True,
    )
    assert not lowers_to_bound(record)


@pytest.fixture
def qbom_id_map() -> DecompIdMap:
    """The QBOM cases resolve their entity id through ``pumping_station_ids``,
    never ``id_map.hydro_id``, so this fixture carries no hydro code."""
    return DecompIdMap(bus_codes=(1,), bus_names=("SE",))


def _qbom_record(
    *,
    constraint_id: int,
    bounds: dict[int, StageBounds],
    code: int = 30,
) -> ConstraintRecord:
    return ConstraintRecord(
        family="HQ",
        constraint_id=constraint_id,
        stage_start=min(bounds),
        stage_end=max(bounds),
        terms=(ConstraintTerm(code=code, coefficient=1.0, variable="QBOM"),),
        bounds=bounds,
        per_block=True,
    )


def test_hq_qbom_lowers_to_pumping_flow_bound(qbom_id_map: DecompIdMap) -> None:
    """A single-term ``QBOM`` RHQ lowers to a pumping ``flow`` bound resolved
    through ``pumping_station_ids``."""
    record = _qbom_record(
        constraint_id=166,
        bounds={0: StageBounds(lower=(5.0,), upper=(80.0,))},
    )
    calendar = [_stage(0, 1)]

    contributions = single_term_bound_contributions(
        _case(calendar),
        qbom_id_map,
        census=_census(record),
        pumping_station_ids={30: 0},
        hydro_capacities={},
    )

    assert len(contributions) == 1
    contribution = contributions[0]
    assert contribution.family == "pumping"
    assert contribution.entity_id == 0
    assert contribution.stage_id == 0
    assert contribution.block_id == 0
    assert contribution.axis == "flow"
    assert contribution.lower == 5.0
    assert contribution.upper == 80.0
    assert contribution.contributor == "HQ_166"


def test_hq_qbom_no_station_warns_and_skips(qbom_id_map: DecompIdMap) -> None:
    """A QBOM term whose code has no matching pumping station is dropped —
    skip-not-partial, never a raw ``KeyError`` and never a fall-back to a
    generic constraint."""
    record = _qbom_record(
        constraint_id=166,
        bounds={0: StageBounds(lower=(5.0,), upper=(80.0,))},
    )
    calendar = [_stage(0, 1)]

    with dx.collect() as collected:
        contributions = single_term_bound_contributions(
            _case(calendar),
            qbom_id_map,
            census=_census(record),
            pumping_station_ids={},
            hydro_capacities={},
        )

    assert contributions == []
    warnings = [d for d in collected if d.severity is Severity.WARNING]
    assert len(warnings) == 1
    assert warnings[0].code == "decomp-rhq-qbom-no-station"


@pytest.mark.parametrize(
    ("coefficient", "lv"),
    [(1.0, (10.0, 400.0)), (-1.0, (-400.0, -10.0))],
    ids=["positive", "negative"],
)
def test_varm_hv_lowers_a_relative_limit_to_absolute_storage(
    id_map: DecompIdMap, coefficient: float, lv: tuple[float, float]
) -> None:
    effective = _hv_effective(2, {(5, "volume_minimo"): (100.0, 150.0)})
    record = _hv_record(
        coefficient=coefficient,
        bounds={stage: StageBounds(lower=(lv[0],), upper=(lv[1],)) for stage in (0, 1)},
    )

    rows = _resolved_storage_rows(_hv_contributions(record, id_map, effective), 2)

    static_floor = storage_envelope(effective, 5)[0]
    for stage, floor in ((0, 100.0), (1, 150.0)):
        generic = sorted(
            (side + coefficient * (floor - static_floor)) / coefficient + static_floor
            for side in lv
        )
        assert (rows[stage]["min_storage_hm3"], rows[stage]["max_storage_hm3"]) == (
            pytest.approx(generic[0], rel=1e-12),
            pytest.approx(generic[1], rel=1e-12),
        )
    assert rows[0]["min_storage_hm3"] == pytest.approx(110.0, rel=1e-12)
    assert rows[1]["max_storage_hm3"] == pytest.approx(550.0, rel=1e-12)


def test_varm_hv_emits_the_limit_and_the_storage_range_per_stage(
    id_map: DecompIdMap,
) -> None:
    effective = _hv_effective(1)
    record = _hv_record(
        coefficient=1.0, bounds={0: StageBounds(lower=(10.0,), upper=(400.0,))}
    )

    contributions = _hv_contributions(record, id_map, effective)

    assert [
        (c.family, c.entity_id, c.stage_id, c.block_id, c.axis, c.contributor)
        for c in contributions
    ] == [
        ("hydro", 0, 0, None, "storage", "HV_7"),
        ("hydro", 0, 0, None, "storage", "storage-range"),
    ]
    assert [(c.lower, c.upper) for c in contributions] == [
        (110.0, 500.0),
        (100.0, 1100.0),
    ]


@pytest.mark.parametrize(
    "stage_1_lv", [(None, None), (-1e21, 1e21)], ids=["none", "sentinel"]
)
def test_varm_hv_skips_a_stage_with_no_effective_side_and_never_shifts_a_sentinel(
    id_map: DecompIdMap, stage_1_lv: tuple[float | None, float | None]
) -> None:
    record = _hv_record(
        coefficient=1.0,
        bounds={
            0: StageBounds(lower=(10.0,), upper=(1e21,)),
            1: StageBounds(lower=(stage_1_lv[0],), upper=(stage_1_lv[1],)),
        },
    )

    contributions = _hv_contributions(record, id_map, _hv_effective(2))

    assert [c for c in contributions if c.stage_id == 1] == []
    rows = _resolved_storage_rows(contributions, 2)
    assert set(rows) == {0}
    assert rows[0]["min_storage_hm3"] == 110.0
    assert rows[0]["max_storage_hm3"] == 1100.0


def test_varm_hv_clamps_a_looser_side_to_the_storage_range_without_a_conflict(
    id_map: DecompIdMap,
) -> None:
    record = _hv_record(
        coefficient=1.0, bounds={0: StageBounds(lower=(-50.0,), upper=(2000.0,))}
    )

    with dx.collect() as collected:
        rows = _resolved_storage_rows(
            _hv_contributions(record, id_map, _hv_effective(1)), 1
        )

    assert (rows[0]["min_storage_hm3"], rows[0]["max_storage_hm3"]) == (100.0, 1100.0)
    assert list(collected) == []


def test_varm_hv_side_above_the_storage_ceiling_raises_on_resolve(
    id_map: DecompIdMap,
) -> None:
    record = _hv_record(
        coefficient=1.0, bounds={0: StageBounds(lower=(1500.0,), upper=(None,))}
    )
    contributions = _hv_contributions(record, id_map, _hv_effective(1))

    with pytest.raises(ValueError, match="empty intersection on hydro axis"):
        resolve(contributions, {0: 1})


def test_varm_hv_on_an_uncadastred_plant_warns_and_is_skipped(
    id_map: DecompIdMap,
) -> None:
    record = _hv_record(
        coefficient=1.0,
        bounds={0: StageBounds(lower=(10.0,), upper=(400.0,))},
        code=9,
    )

    with dx.collect() as collected:
        contributions = _hv_contributions(record, id_map, _hv_effective(1))

    assert contributions == []
    assert [(d.severity, d.code) for d in collected] == [
        (Severity.WARNING, "decomp-rhv-varm-plant-uncadastred")
    ]


@pytest.mark.parametrize(
    (
        "variable",
        "code",
        "lv",
        "table",
        "id_column",
        "entity_id",
        "columns",
        "expected",
    ),
    [
        (
            "VDEF",
            5,
            (50.0, 400.0),
            "hydro",
            "hydro_id",
            0,
            ("min_outflow_hm3", "max_outflow_hm3"),
            (50.0, 400.0),
        ),
        (
            "VDES",
            5,
            (0.0, 50.0),
            "hydro",
            "hydro_id",
            0,
            ("min_diversion_hm3", "max_diversion_hm3"),
            (0.0, 50.0),
        ),
        (
            "VBOM",
            4,
            (None, 20.0),
            "pumping",
            "pumping_station_id",
            2,
            ("min_hm3", "max_hm3"),
            (None, 20.0),
        ),
    ],
)
def test_volume_tipo_hv_lands_in_its_column(
    id_map: DecompIdMap,
    variable: str,
    code: int,
    lv: tuple[float | None, float | None],
    table: str,
    id_column: str,
    entity_id: int,
    columns: tuple[str, str],
    expected: tuple[float | None, float | None],
) -> None:
    record = _hv_record(
        coefficient=1.0,
        bounds={0: StageBounds(lower=(lv[0],), upper=(lv[1],))},
        code=code,
        variable=variable,
    )

    contributions = _hv_contributions(record, id_map, _hv_effective(1))
    (row,) = getattr(
        build_bound_tables(resolve(contributions, {0: 1})), table
    ).to_pylist()

    assert row[id_column] == entity_id
    assert row["block_id"] is None
    assert (row[columns[0]], row[columns[1]]) == expected
    untouched = set(row) - {id_column, "stage_id", "block_id", *columns}
    assert {name: row[name] for name in untouched} == dict.fromkeys(untouched)


def test_vdef_hv_is_unshifted_and_a_negative_coefficient_flips_the_sides(
    id_map: DecompIdMap,
) -> None:
    effective = _hv_effective(1, {(5, "volume_minimo"): (150.0,)})
    record = _hv_record(
        coefficient=-1.0,
        bounds={0: StageBounds(lower=(-400.0,), upper=(-50.0,))},
        variable="VDEF",
    )

    (contribution,) = _hv_contributions(record, id_map, effective)

    assert (contribution.axis, contribution.lower, contribution.upper) == (
        "outflow_volume",
        50.0,
        400.0,
    )


@pytest.mark.parametrize(
    ("lower", "expected"),
    [(-5.0, None), (-0.0, 0.0), (0.0, 0.0)],
    ids=["negative-dropped", "negative-zero-normalized", "zero-literal"],
)
def test_volume_tipo_hv_lower_follows_the_non_negative_rule(
    id_map: DecompIdMap, lower: float, expected: float | None
) -> None:
    record = _hv_record(
        coefficient=1.0,
        bounds={0: StageBounds(lower=(lower,), upper=(100.0,))},
        variable="VDEF",
    )

    (contribution,) = _hv_contributions(record, id_map, _hv_effective(1))

    assert contribution.lower == expected
    if contribution.lower is not None:
        assert math.copysign(1.0, contribution.lower) == 1.0
    assert contribution.upper == 100.0


def test_volume_tipo_hv_negative_coefficient_zero_upper_is_written_as_zero(
    id_map: DecompIdMap,
) -> None:
    record = _hv_record(
        coefficient=-1.0,
        bounds={0: StageBounds(lower=(0.0,), upper=(400.0,))},
        variable="VDEF",
    )

    (contribution,) = _hv_contributions(record, id_map, _hv_effective(1))

    assert contribution.contributor == "HV_7"
    assert contribution.lower is None
    assert contribution.upper == 0.0
    assert contribution.upper is not None
    assert math.copysign(1.0, contribution.upper) == 1.0


@pytest.mark.parametrize(
    ("lv", "resolved"),
    [((-1e21, 100.0), (None, 100.0)), ((50.0, -1e21), (50.0, None))],
    ids=["lower-sentinel", "upper-sentinel"],
)
def test_volume_tipo_hv_sentinel_side_is_no_bound_not_a_negative_one(
    id_map: DecompIdMap,
    lv: tuple[float, float],
    resolved: tuple[float | None, float | None],
) -> None:
    record = _hv_record(
        coefficient=1.0,
        bounds={0: StageBounds(lower=(lv[0],), upper=(lv[1],))},
        variable="VDEF",
    )

    contributions = _hv_contributions(record, id_map, _hv_effective(1))

    assert [c.contributor for c in contributions] == ["HV_7"]
    (row,) = build_bound_tables(resolve(contributions, {0: 1})).hydro.to_pylist()
    assert (row["min_outflow_hm3"], row["max_outflow_hm3"]) == resolved


def test_volume_tipo_hv_negative_upper_adds_the_non_negative_contribution_and_conflicts(
    id_map: DecompIdMap,
) -> None:
    record = _hv_record(
        coefficient=1.0,
        bounds={0: StageBounds(lower=(None,), upper=(-3.0,))},
        code=4,
        variable="VBOM",
    )

    contributions = _hv_contributions(record, id_map, _hv_effective(1))

    assert [(c.contributor, c.lower, c.upper) for c in contributions] == [
        ("HV_7", None, -3.0),
        ("non-negative volume", 0.0, None),
    ]
    with pytest.raises(ValueError, match="empty intersection on pumping axis"):
        resolve(contributions, {0: 1})


def test_hv_with_an_unmapped_variable_raises_naming_it(id_map: DecompIdMap) -> None:
    record = _hv_record(
        coefficient=1.0,
        bounds={0: StageBounds(lower=(10.0,), upper=(400.0,))},
        variable="VTIPO",
    )

    with pytest.raises(ValueError, match="unhandled HV variable 'VTIPO'"):
        _hv_contributions(record, id_map, _hv_effective(1))


def test_hv_lowering_ignores_other_families(id_map: DecompIdMap) -> None:
    record = _re_record(
        constraint_id=12,
        coefficient=1.0,
        bounds={0: StageBounds(lower=(50.0,), upper=(212.0,))},
    )

    assert _hv_contributions(record, id_map, _hv_effective(1)) == []


_CONFLICT_ID_MAP = DecompIdMap(
    bus_codes=(1,), bus_names=("SE",), hydro_codes=(43,), thermal_codes=()
)


def _empty_intersection(
    lowers: tuple[tuple[str, float], ...],
    uppers: tuple[tuple[str, float], ...],
    *,
    family: str = "hydro",
    entity_id: int = 0,
    stage_id: int = 1,
    axis: str = "storage",
) -> EmptyIntersectionError:
    return EmptyIntersectionError(
        "raw accumulator message",
        family=family,
        entity_id=entity_id,
        stage_id=stage_id,
        block_id=None,
        axis=axis,
        lowers=lowers,
        uppers=uppers,
    )


def _translate(error: EmptyIntersectionError) -> RhvBoundConflictError | None:
    return rhv_bound_conflict_error(
        error,
        id_map=_CONFLICT_ID_MAP,
        hydro_names={0: "FURNAS"},
        pumping_station_ids={4: 2},
    )


def _conflict_message(error: EmptyIntersectionError) -> str | None:
    translated = _translate(error)
    return None if translated is None else str(translated)


def test_hv_against_the_storage_range_is_worded_with_plant_stage_and_values() -> None:
    error = _empty_intersection(
        (("HV_20", 1600.0), ("storage-range", 100.0)),
        (("storage-range", 1100.0),),
    )

    assert _conflict_message(error) == (
        "HV constraint(s) 20 cannot all hold on plant code 43 (FURNAS) at stage 2: "
        "the stored volume must be at least 1600.0 hm³ (HV constraint 20) and at "
        "most 1100.0 hm³ (the plant's storage range)."
    )


def test_two_hvs_are_listed_ascending_and_only_the_binding_upper_is_named() -> None:
    error = _empty_intersection((("HV_20", 900.0),), (("HV_115", 500.0), ("VE", 800.0)))

    assert _conflict_message(error) == (
        "HV constraint(s) 20, 115 cannot all hold on plant code 43 (FURNAS) at "
        "stage 2: the stored volume must be at least 900.0 hm³ (HV constraint 20) "
        "and at most 500.0 hm³ (HV constraint 115)."
    )


def test_negative_pumping_volume_upper_names_the_station_and_the_zero_floor() -> None:
    error = _empty_intersection(
        (("non-negative volume", 0.0),),
        (("HV_903", -3.0),),
        family="pumping",
        entity_id=2,
        stage_id=0,
        axis="volume",
    )

    assert _conflict_message(error) == (
        "HV constraint(s) 903 cannot all hold on pumping station code 4 at stage 1: "
        "the stage pumped volume must be at least 0.0 hm³ (a volume cannot be "
        "negative) and at most -3.0 hm³ (HV constraint 903)."
    )


@pytest.mark.parametrize(
    ("axis", "quantity"),
    [
        ("outflow_volume", "stage outflow volume"),
        ("diversion_volume", "stage diverted volume"),
    ],
)
def test_hydro_volume_axes_are_worded_as_stage_volumes(
    axis: str, quantity: str
) -> None:
    error = _empty_intersection((("HV_20", 900.0),), (("HV_21", 500.0),), axis=axis)

    message = _conflict_message(error)

    assert message is not None
    assert f"the {quantity} must be at least" in message


@pytest.mark.parametrize(
    ("label", "phrase"),
    [
        ("storage-envelope", "the plant's storage range"),
        ("storage-range", "the plant's storage range"),
        ("VE", "the VE flood-control volume"),
        ("non-negative volume", "a volume cannot be negative"),
        ("RQ", "another bound on the plant"),
    ],
)
def test_non_hv_source_is_phrased_by_label(label: str, phrase: str) -> None:
    error = _empty_intersection((("HV_20", 900.0),), ((label, 500.0),))

    message = _conflict_message(error)

    assert message is not None
    assert message.endswith(f"at most 500.0 hm³ ({phrase}).")


def test_sources_tied_at_the_binding_value_are_listed_once_per_phrase() -> None:
    error = _empty_intersection(
        (("HV_20", 900.0), ("HV_21", 900.0), ("storage-range", 100.0)),
        (("storage-envelope", 500.0), ("storage-range", 500.0)),
    )

    message = _conflict_message(error)

    assert message is not None
    assert "(HV constraint 20, HV constraint 21)" in message
    assert message.endswith("at most 500.0 hm³ (the plant's storage range).")


def test_one_hv_on_both_sides_is_listed_once() -> None:
    error = _empty_intersection((("HV_20", 900.0),), (("HV_20", 500.0),))

    message = _conflict_message(error)

    assert message is not None
    assert message.startswith("HV constraint(s) 20 cannot all hold")


def test_conflict_without_an_hv_contributor_is_not_translated() -> None:
    error = _empty_intersection((("storage-envelope", 900.0),), (("VE", 500.0),))

    assert _conflict_message(error) is None


def test_conflict_on_a_plant_missing_from_the_names_raises_key_error() -> None:
    error = _empty_intersection((("HV_20", 900.0),), (("HV_21", 500.0),))

    with pytest.raises(KeyError):
        rhv_bound_conflict_error(
            error,
            id_map=_CONFLICT_ID_MAP,
            hydro_names={},
            pumping_station_ids={},
        )


def test_rhv_bound_conflict_diagnostics_have_no_repo_internal_references() -> None:
    errors = [
        _empty_intersection(
            (("HV_20", 1600.0), ("storage-range", 100.0)),
            (("storage-range", 1100.0),),
        ),
        _empty_intersection((("HV_20", 900.0),), (("HV_115", 500.0), ("VE", 800.0))),
        _empty_intersection(
            (("non-negative volume", 0.0),),
            (("HV_903", -3.0),),
            family="pumping",
            entity_id=2,
            axis="volume",
        ),
    ]
    leaks = (
        "storage-range",
        "storage-envelope",
        "HV_",
        ".py",
        "plans/",
        "ticket",
        "epic",
    )

    for error in errors:
        translated = _translate(error)
        assert translated is not None
        diagnostic = diagnostic_from_exception(translated, context="Conversion")
        for text in (diagnostic.summary, diagnostic.remediation or ""):
            assert text
            assert re.search(r"_[a-z]", text) is None, text
            for leak in leaks:
                assert leak not in text, f"leaks {leak!r}: {text!r}"
