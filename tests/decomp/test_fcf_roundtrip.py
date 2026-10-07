"""Cut-level round-trip test for the boundary FCF importer.

Proves the authored checkpoint's coefficients are value-faithful to the
source cuts: this module independently parses a hand-authored
``BoundaryCuts``/``TerminalManifest``/``DecompIdMap``, writes and reloads it
through ``synthetic_roundtrip`` (no deck, no novomodelo binary), and cross-
evaluates its cut coefficients against a *direct, independent* evaluation —
never through :func:`novomodelo_bridge.decomp.fcf.mapper.map_boundary_cuts`, the
mapper's own coefficient placement, which would make this a circular,
Python-vs-Python check. It instead builds its own physical<->slot join from
the *reloaded* ``entity_manifest`` — the ground truth for where the writer
actually placed each mapped coefficient — via this module's own
``_resolve``/``_theta_source``/``_theta_novomodelo`` oracle.

Agreement to f64 tolerance (``policy.fbs`` stores ``coefficients`` as
``float64``, so authored values round-trip through novomodelo-io exactly) across
both a direct coefficient comparison and a deterministic evaluation sweep
catches a sign flip, a unit error, a lag off-by-one, or a dropped
``cost_scale_factor`` marker (which makes novomodelo silently scale every value
by 10⁶) in one test.

Gated on ``@requires_novomodelo_python`` (novomodelo is import-able) stacked with
``@requires_writer_binding`` (the installed wheel actually exposes the
``write_policy_checkpoint`` binding this path calls — an older, importable
wheel lacking it would otherwise fail at runtime with ``AttributeError``
instead of skipping).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from novomodelo_bridge.core.units import C_M3S2HM3, MONTH_HOURS
from novomodelo_bridge.decomp.fcf.bootstrap import TerminalManifest
from novomodelo_bridge.decomp.fcf.cortes import StageCutRecord
from novomodelo_bridge.decomp.id_map import DecompIdMap
from tests._fcf_fixtures import (
    make_boundary_cuts,
    make_cut_record,
    make_id_map,
    make_manifest,
    make_slot,
    synthetic_roundtrip,
)
from tests.conftest import requires_novomodelo_python, requires_writer_binding

#: novomodelo `policy.fbs` entity_type codes (mirrors `fcf/mapper.py`'s private
#: constants of the same name/value — re-declared here, never imported, so
#: this oracle's slot lookup never depends on the mapper under test).
_HYDRO_STORAGE = 0
_HYDRO_INFLOW_LAG = 1
_HYDRO_TRANSIT_BUCKET = 3
_STORAGE_SUBINDEX = 0

#: Random states in the AC 4 sweep beyond the all-zero and per-plant
#: unit-storage states — 2 + 8 = 10, comfortably above the required >= 8
#: floor. Shared by the tier-2 synthetic sweep test below.
_N_RANDOM_STATES = 8

#: The tier-2 synthetic case's two resolved plant codes (joining hydro ids 0
#: and 1) and its one source-only, D3-dropped plant code.
_RESOLVED_PLANT_CODES = (10, 20)
_DROPPED_PLANT_CODE = 30


@dataclass(frozen=True)
class _PhysicalState:
    """One synthetic terminal-state probe, keyed by source plant index.

    ``storage[plant_index]`` and ``inflow[(plant_index, depth)]`` are plain
    physical values — never a dense state-dimension vector until a caller
    projects them onto the manifest's slot positions (``_theta_novomodelo``'s
    ``x_novomodelo`` argument) or reads them off directly (``_theta_source``).
    A key absent from either mapping is implicitly 0.0.
    """

    storage: dict[int, float]
    inflow: dict[tuple[int, int], float]


def _slot_index(
    manifest: Sequence[Mapping[str, Any]],
    entity_type: int,
    entity_id: int,
    subindex: int,
) -> int | None:
    """Position of the manifest slot keyed ``(entity_type, entity_id, subindex)``.

    A linear scan over the *reloaded* ``entity_manifest`` — the ground truth
    for where the checkpoint writer actually placed each coefficient — never
    `fcf.mapper._index_manifest`, which is the mapper's own bookkeeping and
    would make this oracle circular.
    """
    for position, slot in enumerate(manifest):
        if (
            slot["entity_type"] == entity_type
            and slot["entity_id"] == entity_id
            and slot["subindex"] == subindex
        ):
            return position
    return None


def _resolve(
    plant_codes: Sequence[int],
    id_map: DecompIdMap,
    manifest: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, tuple[int, int, dict[int, int]]], frozenset[int]]:
    """Resolve each source plant index to its target manifest slots.

    Returns ``(resolved, dropped)``. ``resolved[plant_index] = (hydro_id,
    storage_position, {lag_depth: lag_position})`` for a plant that both
    resolves via ``id_map.hydro_id`` and has a ``HydroStorage`` slot in
    ``manifest``; ``dropped`` is the set of plant indices excluded for
    either reason (D3 — a source-only plant absent from ``id_map``, or
    missing its ``HydroStorage`` slot in ``manifest``).
    ``lag_positions`` only carries depths 1..12 whose
    ``(HydroInflowLag, hydro_id, depth-1)`` slot is actually present — a
    hydro-specific lag depth shorter than 12 is a legitimate case shape,
    not a read bug.
    """
    resolved: dict[int, tuple[int, int, dict[int, int]]] = {}
    dropped: set[int] = set()
    for plant_index, code in enumerate(plant_codes):
        try:
            hydro_id = id_map.hydro_id(code)
        except KeyError:
            dropped.add(plant_index)
            continue
        storage_position = _slot_index(
            manifest, _HYDRO_STORAGE, hydro_id, _STORAGE_SUBINDEX
        )
        if storage_position is None:
            dropped.add(plant_index)
            continue
        lag_positions: dict[int, int] = {}
        for depth in range(1, 13):
            position = _slot_index(manifest, _HYDRO_INFLOW_LAG, hydro_id, depth - 1)
            if position is not None:
                lag_positions[depth] = position
        resolved[plant_index] = (hydro_id, storage_position, lag_positions)
    return resolved, frozenset(dropped)


def _theta_source(
    records: Sequence[StageCutRecord],
    resolved: Mapping[int, tuple[int, int, dict[int, int]]],
    state: _PhysicalState,
    cost_unit_hours: float,
) -> float:
    """``max_k(rhs_k + pi_varm_k . storage + pi_qafl_k . inflow)`` over active
    source records, restricted to resolved plants and present lag slots —
    the independent oracle AC 4 cross-evaluates against ``_theta_novomodelo``.

    Terms are scaled per family to match the mapper's cost-unit conversion
    (``fcf.mapper``), since the authored (reloaded) cut ``_theta_novomodelo`` reads
    is in novomodelo cost units: the intercept and storage by ``cost_unit_hours``
    (the ``($·mês)/h -> $`` integration over the coupling stage's hours), and
    the inflow-lag additionally by ``C_M3S2HM3`` (the Hm³<-m³/s factor for
    novomodelo's m³/s lag state). ``cost_unit_hours`` is the same value the mapper
    was given (MONTH_HOURS for the synthetic fixtures below).
    """
    lag_factor = cost_unit_hours * C_M3S2HM3
    best = float("-inf")
    for record in records:
        if not record.is_active:
            continue
        value = record.rhs * cost_unit_hours
        for plant_index, (
            _hydro_id,
            _storage_position,
            lag_positions,
        ) in resolved.items():
            value += (
                record.pi_varm[plant_index]
                * state.storage.get(plant_index, 0.0)
                * cost_unit_hours
            )
            plant_lags = record.pi_qafl[plant_index]
            for depth in lag_positions:
                value += (
                    plant_lags[depth - 1]
                    * state.inflow.get((plant_index, depth), 0.0)
                    * lag_factor
                )
        if value > best:
            best = value
    return best


def _theta_novomodelo(
    cuts: Sequence[Mapping[str, Any]], x_novomodelo: Sequence[float]
) -> float:
    """``max_k(intercept_k + coefficients_k . x_novomodelo)`` over active reloaded
    cuts — novomodelo's own ``Policy.evaluate`` semantics, computed directly from
    the checkpoint the writer authored.
    """
    best = float("-inf")
    for cut in cuts:
        if not cut["is_active"]:
            continue
        value = float(cut["intercept"]) + sum(
            c * x for c, x in zip(cut["coefficients"], x_novomodelo, strict=True)
        )
        if value > best:
            best = value
    return best


# ---------------------------------------------------------------------------
# Tier 2 — in-wheel synthetic round trip (no deck, no novomodelo binary).
# ---------------------------------------------------------------------------


def _synthetic_two_plant_case() -> tuple[
    tuple[int, ...], DecompIdMap, TerminalManifest
]:
    """A synthetic two-resolved-plant, one-dropped-plant case.

    Plant codes 10 and 20 resolve to hydro ids 0 and 1 (each with a full
    12-slot ``HydroInflowLag`` family); plant code 30 is source-only —
    absent from ``id_map``, so ``map_boundary_cuts`` drops it (D3). One
    trailing ``HydroTransitBucket`` slot is never targeted by the mapper,
    giving the coefficient-identity test below an unmapped slot to check
    against 0.0.
    """
    plant_codes = (*_RESOLVED_PLANT_CODES, _DROPPED_PLANT_CODE)
    id_map = make_id_map(_RESOLVED_PLANT_CODES)
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),  # position 0: plant 10
            make_slot(_HYDRO_STORAGE, 1, 0),  # position 1: plant 20
            *[
                make_slot(_HYDRO_INFLOW_LAG, 0, lag)  # positions 2..13
                for lag in range(12)
            ],
            *[
                make_slot(_HYDRO_INFLOW_LAG, 1, lag)  # positions 14..25
                for lag in range(12)
            ],
            make_slot(_HYDRO_TRANSIT_BUCKET, 0, 0),  # position 26: unmapped
        ]
    )
    return plant_codes, id_map, manifest


@requires_novomodelo_python
@requires_writer_binding
def test_synthetic_roundtrip_coefficient_identity(tmp_path: Path) -> None:
    """AC 1/2 — coefficient identity, no deck and no novomodelo binary.

    Reuses this module's own ``_resolve`` oracle (never
    ``fcf.mapper.map_boundary_cuts``'s bookkeeping) against a hand-authored
    multi-cut ``BoundaryCuts`` with one active and one inactive record: for
    every active reloaded cut, each resolved plant's ``HydroStorage``
    coefficient equals the source ``pi_varm``, each present
    ``HydroInflowLag`` coefficient equals the source ``pi_qafl`` at that
    depth (nonzero at depths 1-2, proving no off-by-one), and every unmapped
    slot is exactly 0.0. Also checks the intercept and the
    ``cost_scale_factor == 1.0`` marker survive the round trip.
    """
    plant_codes, id_map, manifest = _synthetic_two_plant_case()
    active_record = make_cut_record(
        pi_varm=(3.0, 5.0, 7.0),
        pi_qafl=((1.5, 2.5) + (0.0,) * 10, (0.0,) * 12, (0.0,) * 12),
        rhs=100.0,
        cut_id=1,
        iteration=1,
        forward_pass_index=1,
        is_active=True,
    )
    inactive_record = make_cut_record(
        pi_varm=(30.0, 50.0, 70.0),
        rhs=999.0,
        cut_id=2,
        iteration=2,
        forward_pass_index=1,
        is_active=False,
    )
    cuts = make_boundary_cuts(plant_codes, (active_record, inactive_record))

    reloaded = synthetic_roundtrip(tmp_path / "boundary", cuts, manifest, id_map)
    assert reloaded["metadata"]["producer"]["cost_scale_factor"] == 1.0

    entry = reloaded["stage_cuts"][0]
    resolved, dropped = _resolve(plant_codes, id_map, entry["entity_manifest"])
    assert dropped == frozenset({2})
    assert len(resolved) == 2

    mapped_positions: set[int] = set()
    for _hydro_id, storage_position, lag_positions in resolved.values():
        mapped_positions.add(storage_position)
        mapped_positions.update(lag_positions.values())

    active_cuts = [cut for cut in entry["cuts"] if cut["is_active"]]
    assert len(active_cuts) == 1
    cut = active_cuts[0]
    coefficients = cut["coefficients"]
    # Authored values are the source terms scaled to novomodelo cost units by
    # MONTH_HOURS (fcf.mapper's ($·mês)/h -> $ conversion).
    assert math.isclose(
        cut["intercept"], active_record.rhs * MONTH_HOURS, rel_tol=1e-9, abs_tol=1e-6
    )

    for plant_index, (_hydro_id, storage_position, lag_positions) in resolved.items():
        source_storage = active_record.pi_varm[plant_index] * MONTH_HOURS
        assert math.isclose(
            coefficients[storage_position], source_storage, rel_tol=1e-9, abs_tol=1e-6
        ), f"plant_index={plant_index} storage slot {storage_position}"
        plant_lags = active_record.pi_qafl[plant_index]
        for depth, lag_position in lag_positions.items():
            assert math.isclose(
                coefficients[lag_position],
                plant_lags[depth - 1] * MONTH_HOURS * C_M3S2HM3,
                rel_tol=1e-9,
                abs_tol=1e-6,
            ), f"plant_index={plant_index} lag depth={depth} slot {lag_position}"

    for position, value in enumerate(coefficients):
        if position not in mapped_positions:
            assert value == 0.0, f"unmapped slot {position} = {value!r}"


@requires_novomodelo_python
@requires_writer_binding
def test_synthetic_roundtrip_theta_sweep(tmp_path: Path) -> None:
    """AC 3 — ``theta_novomodelo(x) == theta_source(x)`` over >= 8 states, no deck
    and no novomodelo binary.

    Reuses this module's own ``_theta_source``/``_theta_novomodelo`` oracle
    against the same synthetic two-plant case, over the all-zero,
    per-plant-unit-storage, and ``rng(0)``-random states.
    """
    plant_codes, id_map, manifest = _synthetic_two_plant_case()
    records = (
        make_cut_record(
            pi_varm=(3.0, 5.0, 7.0),
            pi_qafl=((1.5, 2.5) + (0.0,) * 10, (0.4,) + (0.0,) * 11, (0.0,) * 12),
            rhs=100.0,
            cut_id=1,
            iteration=1,
            forward_pass_index=1,
            is_active=True,
        ),
        make_cut_record(
            pi_varm=(1.0, -2.0, 4.0),
            pi_qafl=((0.2,) * 12, (0.0,) * 12, (0.0,) * 12),
            rhs=-50.0,
            cut_id=2,
            iteration=2,
            forward_pass_index=1,
            is_active=True,
        ),
        make_cut_record(
            pi_varm=(30.0, 50.0, 70.0),
            rhs=999.0,
            cut_id=3,
            iteration=3,
            forward_pass_index=1,
            is_active=False,
        ),
    )
    cuts = make_boundary_cuts(plant_codes, records)

    reloaded = synthetic_roundtrip(tmp_path / "boundary", cuts, manifest, id_map)
    entry = reloaded["stage_cuts"][0]
    cuts_list = entry["cuts"]
    state_dimension = entry["state_dimension"]

    resolved, dropped = _resolve(plant_codes, id_map, entry["entity_manifest"])
    assert dropped == frozenset({2})

    rng = np.random.default_rng(0)
    states: list[_PhysicalState] = [
        _PhysicalState(storage={}, inflow={}),
        _PhysicalState(storage={i: 1.0 for i in resolved}, inflow={}),
    ]
    for _ in range(_N_RANDOM_STATES):
        storage = {i: float(rng.normal(scale=1000.0)) for i in resolved}
        inflow = {
            (i, depth): float(rng.normal(scale=1000.0))
            for i, (_hydro_id, _storage_position, lag_positions) in resolved.items()
            for depth in lag_positions
        }
        states.append(_PhysicalState(storage=storage, inflow=inflow))
    assert len(states) >= 8

    for index, state in enumerate(states):
        x_novomodelo = [0.0] * state_dimension
        for plant_index, (
            _hydro_id,
            storage_position,
            lag_positions,
        ) in resolved.items():
            x_novomodelo[storage_position] = state.storage.get(plant_index, 0.0)
            for depth, lag_position in lag_positions.items():
                x_novomodelo[lag_position] = state.inflow.get((plant_index, depth), 0.0)

        theta_novomodelo = _theta_novomodelo(cuts_list, x_novomodelo)
        # Synthetic fixture used synthetic_roundtrip's default cost_unit_hours.
        theta_source = _theta_source(
            records, resolved, state, cost_unit_hours=MONTH_HOURS
        )
        assert math.isclose(
            theta_novomodelo, theta_source, rel_tol=1e-9, abs_tol=1e-6
        ), (
            f"state {index}: theta_novomodelo={theta_novomodelo!r} != "
            f"theta_source={theta_source!r}"
        )


@requires_novomodelo_python
@requires_writer_binding
def test_synthetic_roundtrip_carries_slot_dates_and_priced_date(
    tmp_path: Path,
) -> None:
    """D5 — the write->load round trip carries the dated self-describing
    schema, no deck and no novomodelo binary.

    Authors a one-plant, one-cut synthetic checkpoint via
    `synthetic_roundtrip` and asserts the reloaded terminal
    `entity_manifest[0]` dict carries the per-slot `interval_start` date
    field, the reloaded pool carries the manifest's own `priced_state_date`
    (the date the boundary loader selects a source against), and that
    `metadata["producer"]["cost_scale_factor"] == 1.0` survives the round
    trip (guards the legacy 10**6-scale marker from silently reappearing).
    """
    plant_codes = (10,)
    id_map = make_id_map(plant_codes)
    manifest = make_manifest([make_slot(_HYDRO_STORAGE, 0, 0)])
    record = make_cut_record(pi_varm=(3.0,), rhs=100.0, cut_id=1, iteration=1)
    cuts = make_boundary_cuts(plant_codes, (record,))

    reloaded = synthetic_roundtrip(tmp_path / "boundary", cuts, manifest, id_map)

    entry = reloaded["stage_cuts"][0]
    assert "interval_start" in entry["entity_manifest"][0]
    assert entry["priced_state_date"] == manifest.priced_state_date
    assert reloaded["metadata"]["producer"]["cost_scale_factor"] == 1.0
