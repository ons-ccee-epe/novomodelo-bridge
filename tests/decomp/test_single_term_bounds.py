"""Tier-1 tests for the RE/RHQ/RHV single-term bound producers.

Synthetic ``ConstraintCensus``/``ConstraintRecord``/``StageBounds`` only — no
deck, no ``example/`` read, no ``import novomodelo``. Covers the coefficient-sign
side map, the per-block emission loop's block-count clamp, the "both sides
absent" skip, the RHQ ``QDEF``/``QTUR``/``QDES``/``QVER`` axis lowering
(including their coexistence on one plant and, for ``QDES``/``QVER``, the
``lowers_to_bound`` single-unit-term vs multi-term/non-unit-coefficient
classification), the RHV ``VARM`` additive floor
conversion (including the per-stage effective floor and the no-cadastro
warning-skip), and the fail-loud dispatcher on an unhandled family.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.diagnostics import Severity
from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.constraint_registers import (
    ConstraintCensus,
    ConstraintRecord,
    ConstraintTerm,
    StageBounds,
    lowers_to_bound,
)
from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro
from novomodelo_bridge.decomp.converters.single_term_bounds import (
    HydroCapacities,
    single_term_bound_contributions,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.temporal import OperativeStage
from tests.conftest import make_decomp_case


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


@pytest.fixture
def effective() -> EffectiveCadastro:
    """Minimal placeholder — the RE path never reads it."""
    df = pd.DataFrame({5: {}}).T
    df.index.name = "codigo_usina"
    return EffectiveCadastro(base=df, n_stages=1, stage_varying={})


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


def test_re_single_block_generation_bound(
    id_map: DecompIdMap, effective: EffectiveCadastro
) -> None:
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
        effective=effective,
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


def test_re_negative_coefficient_flips_sides(
    id_map: DecompIdMap, effective: EffectiveCadastro
) -> None:
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
        effective=effective,
        hydro_capacities=_capacities(max_generation_mw=1_000.0),
    )

    assert len(contributions) == 1
    assert contributions[0].lower == -212.0
    assert contributions[0].upper == -50.0


def test_re_two_block_emits_per_block_contributions(
    id_map: DecompIdMap, effective: EffectiveCadastro
) -> None:
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
        effective=effective,
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


def test_ft_thermal_single_block_generation_bound(
    id_map: DecompIdMap, effective: EffectiveCadastro
) -> None:
    """A single-term ``FT`` RE lowers to a ``family="thermal"`` contribution
    on the novomodelo ``generation`` axis, resolved via ``id_map.thermal_id``
    (M1)."""
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
        effective=effective,
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
    id_map: DecompIdMap, effective: EffectiveCadastro
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
        effective=effective,
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


def test_re_hydro_regression_still_family_hydro(
    id_map: DecompIdMap, effective: EffectiveCadastro
) -> None:
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
        effective=effective,
        hydro_capacities=_capacities(max_generation_mw=1_000.0),
    )

    assert len(contributions) == 1
    assert contributions[0].family == "hydro"
    assert contributions[0].axis == "generation"


def test_absent_both_sides_emits_no_contribution(
    id_map: DecompIdMap, effective: EffectiveCadastro
) -> None:
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
        effective=effective,
        hydro_capacities=_capacities(max_generation_mw=1_000.0),
    )

    assert contributions == []


def test_unhandled_family_raises(
    id_map: DecompIdMap, effective: EffectiveCadastro
) -> None:
    # "RHA" (HA/LA/CA) is read into the census but never lowered by this
    # module — neither to bounds nor generic — so it stands in for a
    # genuinely unhandled family now that "RE"/"HQ"/"HV" are all wired.
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
            effective=effective,
            hydro_capacities={},
        )


def test_re_unexpected_bounded_variable_raises(
    id_map: DecompIdMap, effective: EffectiveCadastro
) -> None:
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
            effective=effective,
            hydro_capacities={},
        )


def test_re_ceiling_above_capacity_clamps_upper(
    id_map: DecompIdMap, effective: EffectiveCadastro
) -> None:
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
        effective=effective,
        hydro_capacities=_capacities(max_generation_mw=9_777.776),
    )

    assert len(contributions) == 1
    assert contributions[0].lower == 50.0
    assert contributions[0].upper == 9_777.776


def test_re_ceiling_at_capacity_is_not_clamped_and_emits_no_diagnostic(
    id_map: DecompIdMap, effective: EffectiveCadastro
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
            effective=effective,
            hydro_capacities=_capacities(max_generation_mw=212.0),
        )

    assert len(contributions) == 1
    assert contributions[0].upper == 212.0
    assert [d for d in collected if d.code == "decomp-re-generation-clamped"] == []


def test_re_clamp_emits_diagnostic_naming_plant_ceiling_and_capacity(
    id_map: DecompIdMap, effective: EffectiveCadastro
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
            effective=effective,
            hydro_capacities=_capacities(max_generation_mw=9_777.776),
        )

    clamped = [d for d in collected if d.code == "decomp-re-generation-clamped"]
    assert len(clamped) == 1
    assert clamped[0].severity is Severity.WARNING
    assert "Hydro 0" in clamped[0].summary  # the plant
    assert "11000.0" in clamped[0].summary  # the RE ceiling
    assert "9777.776" in clamped[0].summary  # the declared capacity


def test_re_missing_capacity_entry_raises(
    id_map: DecompIdMap, effective: EffectiveCadastro
) -> None:
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
            effective=effective,
            hydro_capacities={},
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


def test_hq_qdef_lowers_to_outflow(
    hq_id_map: DecompIdMap, effective: EffectiveCadastro
) -> None:
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
        effective=effective,
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


def test_hq_qtur_lowers_to_turbined(
    hq_id_map: DecompIdMap, effective: EffectiveCadastro
) -> None:
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
        effective=effective,
        hydro_capacities=_capacities(max_turbined_m3s=150.0),
    )

    assert len(contributions) == 1
    contribution = contributions[0]
    assert contribution.axis == "turbined"
    assert contribution.lower is None
    assert contribution.upper == 150.0


def test_hq_qtur_ceiling_above_capacity_clamps_upper(
    hq_id_map: DecompIdMap, effective: EffectiveCadastro
) -> None:
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
        effective=effective,
        hydro_capacities=_capacities(max_turbined_m3s=972.0),
    )

    assert len(contributions) == 1
    assert contributions[0].lower == 30.0
    assert contributions[0].upper == 972.0


def test_hq_qtur_ceiling_at_capacity_is_not_clamped_and_emits_no_diagnostic(
    hq_id_map: DecompIdMap, effective: EffectiveCadastro
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
            effective=effective,
            hydro_capacities=_capacities(max_turbined_m3s=150.0),
        )

    assert len(contributions) == 1
    assert contributions[0].upper == 150.0
    assert [d for d in collected if d.code == "decomp-qtur-turbined-clamped"] == []


def test_hq_qtur_clamp_emits_diagnostic_naming_plant_ceiling_and_capacity(
    hq_id_map: DecompIdMap, effective: EffectiveCadastro
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
            effective=effective,
            hydro_capacities=_capacities(max_turbined_m3s=972.0),
        )

    clamped = [d for d in collected if d.code == "decomp-qtur-turbined-clamped"]
    assert len(clamped) == 1
    assert clamped[0].severity is Severity.WARNING
    assert "Hydro 0" in clamped[0].summary  # the plant
    assert "973.2" in clamped[0].summary  # the QTUR ceiling
    assert "972.0" in clamped[0].summary  # the declared capacity


def test_hq_qtur_missing_capacity_entry_raises(
    hq_id_map: DecompIdMap, effective: EffectiveCadastro
) -> None:
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
            effective=effective,
            hydro_capacities={},
        )


def test_hq_qdef_does_not_read_hydro_capacities(
    hq_id_map: DecompIdMap, effective: EffectiveCadastro
) -> None:
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
        effective=effective,
        hydro_capacities={},
    )

    assert len(contributions) == 1
    assert contributions[0].axis == "outflow"


def test_hq_per_block_outflow_contributions(
    hq_id_map: DecompIdMap, effective: EffectiveCadastro
) -> None:
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
        effective=effective,
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


def test_hq_qdef_and_qtur_coexist_on_one_plant(
    hq_id_map: DecompIdMap, effective: EffectiveCadastro
) -> None:
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
        effective=effective,
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


def test_hq_qdes_lowers_to_diversion(
    hq_id_map: DecompIdMap, effective: EffectiveCadastro
) -> None:
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
        effective=effective,
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


def test_hq_qdes_negative_coefficient_flips_sides(
    hq_id_map: DecompIdMap, effective: EffectiveCadastro
) -> None:
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
        effective=effective,
        hydro_capacities={},
    )

    assert len(contributions) == 1
    assert contributions[0].axis == "diversion"
    assert contributions[0].lower == -40.0
    assert contributions[0].upper == -10.0


def test_hq_qver_lowers_to_spillage(
    hq_id_map: DecompIdMap, effective: EffectiveCadastro
) -> None:
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
        effective=effective,
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


def test_hq_qver_negative_coefficient_flips_sides(
    hq_id_map: DecompIdMap, effective: EffectiveCadastro
) -> None:
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
        effective=effective,
        hydro_capacities={},
    )

    assert len(contributions) == 1
    assert contributions[0].axis == "spillage"
    assert contributions[0].lower == -60.0
    assert contributions[0].upper == -5.0


def test_hq_qdef_qdes_qver_all_coexist_on_one_plant(
    hq_id_map: DecompIdMap, effective: EffectiveCadastro
) -> None:
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
        effective=effective,
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


def test_hq_qbom_lowers_to_pumping_flow_bound(
    qbom_id_map: DecompIdMap, effective: EffectiveCadastro
) -> None:
    """A single-term ``QBOM`` RHQ lowers to a pumping ``flow`` bound resolved
    through ``pumping_station_ids`` (M2)."""
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
        effective=effective,
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


def test_hq_qbom_no_station_warns_and_skips(
    qbom_id_map: DecompIdMap, effective: EffectiveCadastro
) -> None:
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
            effective=effective,
            hydro_capacities={},
        )

    assert contributions == []
    warnings = [d for d in collected if d.severity is Severity.WARNING]
    assert len(warnings) == 1
    assert warnings[0].code == "decomp-rhq-qbom-no-station"


def _hv_hidr_frame(
    *, code: int = 1, volume_minimo: float = 20.0, volume_maximo: float = 100.0
) -> pd.DataFrame:
    df = pd.DataFrame(
        {code: {"volume_minimo": volume_minimo, "volume_maximo": volume_maximo}}
    ).T
    df.index.name = "codigo_usina"
    return df


@pytest.fixture
def hv_id_map() -> DecompIdMap:
    return DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))


def _hv_record(
    *,
    constraint_id: int,
    coefficient: float,
    bounds: dict[int, StageBounds],
    code: int = 1,
) -> ConstraintRecord:
    return ConstraintRecord(
        family="HV",
        constraint_id=constraint_id,
        stage_start=min(bounds),
        stage_end=max(bounds),
        terms=(ConstraintTerm(code=code, coefficient=coefficient, variable="VARM"),),
        bounds=bounds,
        per_block=False,
    )


def test_hv_varm_additive_floor_plus_lv(hv_id_map: DecompIdMap) -> None:
    effective = EffectiveCadastro(base=_hv_hidr_frame(), n_stages=1, stage_varying={})
    record = _hv_record(
        constraint_id=7,
        coefficient=1.0,
        bounds={0: StageBounds(lower=(5.0,), upper=(30.0,))},
    )
    calendar = [_stage(0, 1)]

    contributions = single_term_bound_contributions(
        _case(calendar),
        hv_id_map,
        census=_census(record),
        pumping_station_ids={},
        effective=effective,
        hydro_capacities={},
    )

    assert len(contributions) == 1
    contribution = contributions[0]
    assert contribution.family == "hydro"
    assert contribution.entity_id == 0
    assert contribution.stage_id == 0
    assert contribution.block_id is None
    assert contribution.axis == "storage"
    assert contribution.lower == 25.0
    assert contribution.upper == 50.0
    assert contribution.contributor == "HV_7"


def test_hv_varm_uses_per_stage_effective_floor(hv_id_map: DecompIdMap) -> None:
    effective = EffectiveCadastro(
        base=_hv_hidr_frame(),
        n_stages=2,
        stage_varying={(1, "volume_minimo"): (20.0, 40.0)},
    )
    record = _hv_record(
        constraint_id=7,
        coefficient=1.0,
        bounds={
            0: StageBounds(lower=(5.0,), upper=(None,)),
            1: StageBounds(lower=(5.0,), upper=(None,)),
        },
    )
    calendar = [_stage(0, 1), _stage(1, 1)]

    contributions = single_term_bound_contributions(
        _case(calendar),
        hv_id_map,
        census=_census(record),
        pumping_station_ids={},
        effective=effective,
        hydro_capacities={},
    )

    by_stage = {c.stage_id: c for c in contributions}
    assert set(by_stage) == {0, 1}
    assert by_stage[0].lower == 25.0
    assert by_stage[1].lower == 45.0
    for contribution in contributions:
        assert contribution.block_id is None
        assert contribution.upper is None


def test_hv_varm_negative_coefficient_flips_then_adds_floor(
    hv_id_map: DecompIdMap,
) -> None:
    effective = EffectiveCadastro(base=_hv_hidr_frame(), n_stages=1, stage_varying={})
    record = _hv_record(
        constraint_id=7,
        coefficient=-1.0,
        bounds={0: StageBounds(lower=(5.0,), upper=(30.0,))},
    )
    calendar = [_stage(0, 1)]

    contributions = single_term_bound_contributions(
        _case(calendar),
        hv_id_map,
        census=_census(record),
        pumping_station_ids={},
        effective=effective,
        hydro_capacities={},
    )

    assert len(contributions) == 1
    assert contributions[0].lower == -10.0
    assert contributions[0].upper == 15.0


def test_hv_varm_uncadastred_plant_warns_and_skips(hv_id_map: DecompIdMap) -> None:
    effective = EffectiveCadastro(base=_hv_hidr_frame(), n_stages=1, stage_varying={})
    record = _hv_record(
        constraint_id=7,
        coefficient=1.0,
        bounds={0: StageBounds(lower=(5.0,), upper=(30.0,))},
        code=99,
    )
    calendar = [_stage(0, 1)]

    with dx.collect() as collected:
        contributions = single_term_bound_contributions(
            _case(calendar),
            hv_id_map,
            census=_census(record),
            pumping_station_ids={},
            effective=effective,
            hydro_capacities={},
        )

    assert contributions == []
    warnings = [d for d in collected if d.severity is Severity.WARNING]
    assert len(warnings) == 1
    assert warnings[0].code == "decomp-rhv-varm-plant-uncadastred"
