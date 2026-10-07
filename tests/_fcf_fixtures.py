"""Shared synthetic builders for the boundary-cut import tests.

Every deck-independent FCF test needs the same synthetic building blocks: a
:class:`~novomodelo_bridge.decomp.fcf.cortes.CortesHeader`,
:class:`~novomodelo_bridge.decomp.fcf.cortes.StageCutRecord`,
:class:`~novomodelo_bridge.decomp.fcf.cortes.BoundaryCuts`, a
:class:`~novomodelo_bridge.decomp.fcf.bootstrap.TerminalManifest` (with
correctly-shaped slot dicts), a
:class:`~novomodelo_bridge.decomp.fcf.mapper.MappedCut`, and a minimal
:class:`~novomodelo_bridge.decomp.id_map.DecompIdMap` — plus, for the tier-2
round-trip, :func:`synthetic_roundtrip`, which runs
``map_boundary_cuts -> build_stage_cuts_payload -> build_metadata ->
write_boundary_checkpoint -> novomodelo.results.load_policy`` end to end.

These are plain functions, not pytest fixtures — mirrors
``tests/conftest.py``'s "plain builder functions imported by name"
convention (``make_case``, ``make_nw_files``, ``hydro_with_group``). Import
them with ``from tests._fcf_fixtures import ...``.

The module imports cleanly with novomodelo absent: every top-level import here
touches only novomodelo-free symbols (``fcf/bootstrap.py``, ``fcf/cortes.py``,
``fcf/mapper.py``, ``fcf/writer.py``, and ``id_map.py`` all import novomodelo only
inside function bodies, never at module scope). :func:`synthetic_roundtrip`
is the only function in this module that touches novomodelo, and it does so via a
call-site ``import novomodelo`` inside the function — it must only be called from
a ``@requires_novomodelo_python``-guarded test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from novomodelo_bridge.core.units import MONTH_HOURS
from novomodelo_bridge.decomp.fcf.bootstrap import TerminalManifest
from novomodelo_bridge.decomp.fcf.cortes import (
    BoundaryCuts,
    CortesHeader,
    StageCutRecord,
)
from novomodelo_bridge.decomp.fcf.mapper import (
    GnlRingPlan,
    MappedCut,
    map_boundary_cuts,
)
from novomodelo_bridge.decomp.fcf.writer import (
    build_metadata,
    build_stage_cuts_payload,
    write_boundary_checkpoint,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

#: Fixed timestamp handed to `build_metadata`'s `created_at` — this module
#: never calls `datetime.now()` (matches `fcf/writer.py`'s own convention of
#: taking `created_at` as a caller-supplied parameter).
_CREATED_AT = "2026-08-03T00:00:00Z"


def make_cortes_header(
    plant_codes: tuple[int, ...],
    *,
    lag_maximo_gnl: int = 0,
    n_patamares: int = 3,
    submercado_codes: tuple[int, ...] = (1,),
    record_size: int = 112,
) -> CortesHeader:
    """A synthetic source header carrying only `plant_codes` meaningfully."""
    return CortesHeader(
        plant_codes=plant_codes,
        submercado_codes=submercado_codes,
        n_patamares=n_patamares,
        lag_maximo_gnl=lag_maximo_gnl,
        n_plants=len(plant_codes),
        individualized=True,
        record_size=record_size,
        last_cut_record_by_stage=(1,),
    )


def make_cut_record(
    *,
    pi_varm: tuple[float, ...],
    pi_qafl: tuple[tuple[float, ...], ...] | None = None,
    pi_gnl: tuple[float, ...] = (),
    rhs: float = 0.0,
    cut_id: int = 1,
    iteration: int = 1,
    forward_pass_index: int = 1,
    is_active: bool = True,
) -> StageCutRecord:
    """A synthetic `StageCutRecord`; `pi_qafl` defaults to all-zero lags."""
    n_plants = len(pi_varm)
    return StageCutRecord(
        cut_id=cut_id,
        iteration=iteration,
        forward_pass_index=forward_pass_index,
        is_active=is_active,
        rhs=rhs,
        pi_varm=pi_varm,
        pi_qafl=(
            pi_qafl
            if pi_qafl is not None
            else tuple((0.0,) * 12 for _ in range(n_plants))
        ),
        pi_gnl=pi_gnl,
    )


def make_boundary_cuts(
    plant_codes: tuple[int, ...],
    records: tuple[StageCutRecord, ...],
    *,
    boundary_stage: int = 10,
) -> BoundaryCuts:
    """A synthetic `BoundaryCuts` over `plant_codes`."""
    return BoundaryCuts(
        header=make_cortes_header(plant_codes),
        boundary_stage=boundary_stage,
        records=records,
    )


_SLOT_DATE_SENTINEL = -2147483648


def _next_month_anchor(yyyymmdd: int) -> int:
    """The `YYYYMM01` of the month after `yyyymmdd`'s — a valid interval end
    strictly after any `interval_start` inside that month.

    Mirrors novomodelo's own `[stage.start_date, stage.end_date)` interval dating,
    whose exclusive `end_date` for a monthly delivery stage is the next
    month's first day.
    """
    year, month = divmod(yyyymmdd // 100, 100)
    return (
        (year + 1) * 10_000 + 101
        if month == 12
        else year * 10_000 + (month + 1) * 100 + 1
    )


def make_slot(
    entity_type: int,
    entity_id: int,
    subindex: int,
    *,
    was_active: bool = True,
    interval_start: int = _SLOT_DATE_SENTINEL,
    interval_end: int | None = None,
) -> dict[str, object]:
    """One hand-authored terminal-manifest slot dict.

    Carries `was_active`, `interval_start`, and `interval_end` alongside the
    positional key (`entity_type`, `entity_id`, `subindex`). `interval_start`
    is the slot's delivery-stage start day, the per-slot date field the GNL
    ring placement (`fcf/mapper.py`) keys covered/non-covered lanes off.
    `interval_end` defaults to the next-month anchor when `interval_start` is
    live and left unset — novomodelo's checkpoint writer rejects a live
    `interval_start` with no paired `interval_end` — and to the sentinel
    otherwise. The default `-2147483648` is `i32::MIN`, novomodelo's
    `ENTITY_SLOT_DATE_SENTINEL` for "no date" — the same value
    `write_policy_checkpoint` itself defaults to when a slot omits the key.
    """
    if interval_end is None:
        interval_end = (
            _next_month_anchor(interval_start)
            if interval_start != _SLOT_DATE_SENTINEL
            else _SLOT_DATE_SENTINEL
        )
    return {
        "entity_type": entity_type,
        "entity_id": entity_id,
        "subindex": subindex,
        "was_active": was_active,
        "interval_start": interval_start,
        "interval_end": interval_end,
    }


#: The absent season descriptor (novomodelo's `SEASON_CYCLE_CODE_ABSENT` == 255,
#: no seasons, no hydros) — the seasonless-study default for a synthetic
#: manifest; a seasonless loading study never fires the season gate.
_ABSENT_SEASON_MANIFEST: dict[str, object] = {
    "cycle_code": 255,
    "n_seasons": 0,
    "hydro_orders": [],
}


def make_manifest(
    slots: Sequence[dict[str, object]],
    *,
    node_id: int = 0,
    graph_stage_id: int = 10,
    priced_state_date: int = 20_261_101,
    season_manifest: dict[str, object] | None = None,
) -> TerminalManifest:
    """A synthetic terminal manifest; `state_dimension` == slot count.

    `node_id`/`graph_stage_id` default to a real single node and this
    module's own terminal-stage convention (`make_boundary_cuts`'s
    `boundary_stage` and `synthetic_roundtrip`'s `stage_id` both default to
    `10`), never the `-1` shared-pool sentinel `TerminalManifest` forbids.
    `priced_state_date` defaults to an arbitrary real `YYYYMMDD` (never the
    undated sentinel `bootstrap_terminal_manifest` rejects). `season_manifest`
    defaults to the absent descriptor (a seasonless synthetic study).
    """
    return TerminalManifest(
        entity_manifest=tuple(slots),
        state_dimension=len(slots),
        node_id=node_id,
        graph_stage_id=graph_stage_id,
        priced_state_date=priced_state_date,
        season_manifest=(
            _ABSENT_SEASON_MANIFEST if season_manifest is None else season_manifest
        ),
    )


def make_mapped_cut(
    *,
    coefficients: tuple[float, ...],
    intercept: float = 1.0,
    cut_id: int = 1,
    iteration: int = 1,
    forward_pass_index: int = 0,
    is_active: bool = True,
    inflow_lag_coefficients: Mapping[int, tuple[float, ...]] | None = None,
) -> MappedCut:
    """One hand-authored `MappedCut`."""
    return MappedCut(
        intercept=intercept,
        coefficients=coefficients,
        cut_id=cut_id,
        iteration=iteration,
        forward_pass_index=forward_pass_index,
        is_active=is_active,
        inflow_lag_coefficients=dict(inflow_lag_coefficients or {}),
    )


def make_id_map(
    hydro_codes: tuple[int, ...],
    *,
    bus_codes: tuple[int, ...] = (1,),
    bus_names: tuple[str, ...] = ("SE",),
) -> DecompIdMap:
    """A minimal `DecompIdMap` carrying only the hydro-code join surface."""
    return DecompIdMap(
        bus_codes=bus_codes, bus_names=bus_names, hydro_codes=hydro_codes
    )


def synthetic_roundtrip(
    boundary_dir: Path,
    cuts: BoundaryCuts,
    manifest: TerminalManifest,
    id_map: DecompIdMap,
    *,
    stage_id: int = 10,
    cost_scale_factor: float = 1.0,
    cost_unit_hours: float = MONTH_HOURS,
    gnl_plan: GnlRingPlan | None = None,
    coupling_block_hours: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Map, write, and reload a synthetic boundary checkpoint; no deck, no
    novomodelo binary.

    Runs `map_boundary_cuts -> build_stage_cuts_payload -> build_metadata ->
    write_boundary_checkpoint -> novomodelo.results.load_policy` against
    `boundary_dir`, then returns the reloaded policy dict verbatim. `novomodelo`
    is imported lazily, inside this function body, so the rest of the
    module stays importable without the novomodelo-python wheel — only
    call this from a `@requires_novomodelo_python`-guarded test. `gnl_plan`
    defaults to `None`, forwarded verbatim to `map_boundary_cuts`, so every
    existing caller keeps leaving the GNL ring at `0.0` unchanged; pass it to
    exercise a populated ring. `coupling_block_hours` defaults to `None`; when
    `None` and `gnl_plan` is given, a uniform split
    `[cost_unit_hours / n_patamares] * n_patamares` is derived so a
    GNL-exercising caller need not hand-compute the per-block vector for the
    common uniform case — a storage-only caller (`gnl_plan=None`) is
    untouched, since the mapper never requires `coupling_block_hours` when no
    GNL coefficient is placed.

    Raises
    ------
    ValueError, RuntimeError
        Propagated verbatim from `map_boundary_cuts` / `build_stage_cuts_payload`
        / `build_metadata` / `write_boundary_checkpoint`.
    """
    import novomodelo

    if coupling_block_hours is None and gnl_plan is not None:
        n_patamares = cuts.header.n_patamares
        coupling_block_hours = (cost_unit_hours / n_patamares,) * n_patamares

    mapping = map_boundary_cuts(
        cuts,
        manifest,
        id_map,
        cost_unit_hours=cost_unit_hours,
        gnl_plan=gnl_plan,
        coupling_block_hours=coupling_block_hours,
    )
    stage_cuts_payload = build_stage_cuts_payload(
        mapping,
        manifest,
        stage_id=stage_id,
        cost_scale_factor=cost_scale_factor,
        node_id=manifest.node_id,
        graph_stage_id=manifest.graph_stage_id,
        priced_state_date=manifest.priced_state_date,
    )
    completed_iterations = max((cut.iteration for cut in mapping.cuts), default=0)
    metadata = build_metadata(
        num_stages=1,
        cost_scale_factor=cost_scale_factor,
        completed_iterations=completed_iterations,
        final_lower_bound=0.0,
        max_iterations=completed_iterations,
        forward_passes=0,
        warm_start_cuts=0,
        rng_seed=0,
        created_at=_CREATED_AT,
        season_manifest=manifest.season_manifest,
    )
    write_boundary_checkpoint(boundary_dir, stage_cuts_payload, metadata)

    return novomodelo.results.load_policy(
        boundary_dir.parent, policy_subdir=boundary_dir.name
    )
