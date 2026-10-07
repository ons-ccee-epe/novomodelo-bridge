"""Manifest-to-manifest mapper for the source model's boundary cuts.

The bootstrap stage (``fcf/bootstrap.py``) reads back cobre's
terminal ``entity_manifest`` — the target case's per-slot state-vector
layout. This module maps each of the source model's boundary cuts
(``fcf/cortes.py``'s :class:`~cobre_bridge.decomp.fcf.cortes.BoundaryCuts`)
onto that layout: storage terms join by plant code, inflow-lag
terms join 1:1 by calendar-month lag slot, and — when the caller supplies a
:class:`GnlRingPlan` — GNL-anticipated-ring terms join each target's
*covered* dated ring slot(s) via a chain-rule patamar sum over ``pi_gnl``
(narrowed to the class-3 signaled, month-anchored lanes cobre's excised ring
actually carries). A source plant with no match in the target manifest is
dropped (D3), never folded into a neighbour, and recorded in
:class:`MappingResult.dropped` for the diagnostics layer to render; a GNL
source/target term with no live counterpart — including a dated ring slot
whose delivery falls before :attr:`GnlRingPlan.post_horizon_start` (an
in-study delivery, priced by the in-study committed window rather than the
ring) — is dropped the same way into :class:`MappingResult.gnl_dropped`.
``HydroTransitBucket`` slots, and any ``AnticipatedThermalState`` ring slot
with no resolved target (including the undated sentinel slot and every
non-covered dated slot), are left at an explicit coefficient ``0.0``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cobre_bridge.core.units import C_M3S2HM3

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from cobre_bridge.decomp.fcf.bootstrap import TerminalManifest
    from cobre_bridge.decomp.fcf.cortes import BoundaryCuts
    from cobre_bridge.decomp.id_map import DecompIdMap

# --- FCF cost-unit conversion -------------------------------------------------
# The source's individualized cut coefficients carry three unit conventions (see
# ``nwlistcf.rel``'s ``UNIDADES DE MEDIDA`` header): ``RHS [($·mês)/h]`` and
# ``PIVARM``/``PIAFL [($·mês)/(Hm³·h)]`` are a *per-hour* rate with an implicit
# monthly normalization, and ``PIGTAD`` (the GNL / anticipated-thermal term,
# ``pi_gnl``) is an energy price ``[$/MWh]`` — none of them a plain cost. cobre's
# terminal future cost enters an objective already in ``$`` (the same base as the
# converted immediate costs), so every FCF term must be brought to ``$`` over the
# coupling stage's hours. :func:`map_boundary_cuts` takes that hour count as
# ``cost_unit_hours`` — the terminal/coupling stage's actual duration, which is
# how the source integrates its per-hour future-cost rate at the coupling
# (empirically it reproduces the source's ``E(CF)`` to ~2-3%, where a fixed 730-h
# month overshoots by ~15% on a short coupling period). The GNL term needs those
# same hours split *per coupling block* (``coupling_block_hours``, patamar order)
# rather than summed — see the next paragraph. Without any factor the loaded FCF
# is ~700× too small, cobre under-values stored water by three orders of
# magnitude, drains the reservoirs, and the boundary policy is numerically inert.
#
# The intercept and storage (``PIVARM``) terms take ``× cost_unit_hours`` alone:
# ``× cost_unit_hours`` integrates the per-hour rate and the storage state is
# already Hm³. The GNL (``PIGTAD``) term takes a *per-block, hours-weighted*
# collapse instead of that flat ``× cost_unit_hours`` — ``pi_gnl`` is an energy
# price ``$/MWh`` pricing cobre's anticipated-thermal ring state, a flat power
# dispatch ``G`` [MW] (``cobre-core`` ``generic_constraint.rs``: "anticipated
# thermal unit (MW)"). The energy delivered in coupling block ``p`` is ``G · h_p``,
# so the chain rule gives ``∂E(CF)/∂G = Σ_p pi_gnl[p] · h_p`` — the coupling
# stage's *per-block* hours in patamar order (``coupling_block_hours``), never its
# total (``cost_unit_hours`` alone would over-count the GNL term by
# ``n_patamares``, since ``pi_gnl`` already carries one coefficient per patamar).
# No ``C_M3S2HM3`` here either way — ``pi_gnl``'s energy pricing needs no
# Hm³-to-m³/s conversion. The inflow-lag (``PIAFL``) term
# takes an *additional* ``× C_M3S2HM3``: ``PIAFL`` is per-Hm³, so
# ``× cost_unit_hours`` yields ``R$/Hm³`` — correct against a storage state in
# Hm³, but cobre's *inflow-lag* state variable is a raw flow rate in **m³/s**
# (the same physical quantity as the ``z_inflow`` column, stored unscaled in the
# state vector). Converting ``R$/Hm³ → R$/(m³/s)`` multiplies by the fixed
# Hm³-per-(m³/s) month factor :data:`~cobre_bridge.core.units.C_M3S2HM3`
# (``= 2.628``, the source's monthly inflow-volume convention, per the SDDP
# review): a 1 m³/s recent inflow represents 2.628 Hm³ of monthly volume, so it
# carries 2.628× the R$/Hm³ water value. Storage must NOT take this factor and
# the lag must not omit it — either asymmetry mis-prices the terminal water value
# and biases the cost-to-go. (For this deck the binding terminal cuts carry no
# nonzero ``PIAFL`` at the wet plants, so the extra 2.628 is inert at θ — it is
# applied for dimensional correctness at other states and in the interior cuts.)
#
# --- Inflow-lag mean fold (deviation-vs-raw reconciliation) -------------------
# The source's PAR(p) inflow-lag term prices the inflow *deviation from the
# seasonal mean* — its state is the increment ``Q_ℓ - μ_ℓ`` about the long-term
# mean ``μ_ℓ`` (the MLT), not the absolute inflow (reference manual §5.1.9.2:
# energies are computed on incremental inflows). But cobre evaluates the loaded
# cut at its *raw* inflow-lag state ``Q_ℓ`` (the PAR lag coefficients are stored
# "in original units"; the standardized ``σ·η`` form lives only in the forward
# inflow model, not the loaded cut). Feeding raw ``Q_ℓ`` against a coefficient
# built for the deviation over-subtracts ``Σ_ℓ PIAFL_ℓ·μ_ℓ`` and over-drains the
# reservoirs (observed: converged thermal −19.6→−35.5 %, spot −35.4→−55.9 %).
# The fix folds the mean into the intercept instead of touching cobre's state:
# with ``μ_ℓ`` supplied per plant per lag (``map_boundary_cuts``'s
# ``inflow_lag_means``, built by ``decomp/inflow_mlt.py``), each cut's RHS is
# reduced by ``Σ_ℓ (PIAFL_scaled_ℓ)·μ_ℓ`` so the loaded cut reads
# ``RHS - Σ PIAFL·μ + Σ PIAFL·Q = RHS + Σ PIAFL·(Q - μ)`` — deviation-correct at
# the raw state, per scenario and per lag. ``σ`` is not needed: ``PIAFL`` is
# per-Hm³ (``1/σ`` is already inside it), so only the means matter. The fold uses
# the *scaled* coefficient (``× cost_unit_hours × C_M3S2HM3``) and ``μ_ℓ`` in
# m³/s (cobre's raw lag-state units), and folds only the lag terms actually
# placed — a dropped plant/lag contributes nothing, so the RHS can never carry a
# mean cobre will not offset with a matching ``Σ PIAFL·Q`` term.

#: cobre `policy.fbs` entity_type codes (confirmed against `policy_export.rs`).
_HYDRO_STORAGE = 0
_HYDRO_INFLOW_LAG = 1
_ANTICIPATED_THERMAL_STATE = 2
_HYDRO_TRANSIT_BUCKET = 3

#: `HydroStorage`'s `subindex` is always 0 (policy.fbs: one slot per plant).
_STORAGE_SUBINDEX = 0

#: cobre's `ENTITY_SLOT_DATE_SENTINEL` (`i32::MIN`), the "no date" value for
#: every per-slot date field — here an `AnticipatedThermalState` slot whose
#: `interval_start` is unset (the undated in-study anticipation, already priced
#: by the converter's `past_anticipated_commitments`), never a GNL target.
_SLOT_DATE_SENTINEL = -2147483648


@dataclass(frozen=True)
class MappedCut:
    """One source cut mapped onto the target's state-vector layout.

    ``coefficients`` is a full-length vector aligned to
    ``TerminalManifest.state_dimension`` — every target slot has an
    explicit coefficient, never merely unset. ``intercept`` is the source
    record's ``rhs`` (the ``alpha - beta'xhat`` form; never re-derived, per
    §2.1), scaled to cobre's cost units and — when :func:`map_boundary_cuts`
    is given ``inflow_lag_means`` — reduced by the seasonal-mean fold
    ``Σ placed_lag_coef · mu`` (see the module header). The intercept and every
    coefficient are scaled to cobre's cost units by :func:`map_boundary_cuts` —
    intercept and storage by ``× cost_unit_hours``, inflow-lag by an additional
    ``× C_M3S2HM3``, and GNL by the per-block ``coupling_block_hours``
    hours-weighted collapse (see the module header). ``cut_id``, ``iteration``,
    ``forward_pass_index``, and ``is_active`` are the source
    ``StageCutRecord``'s provenance fields, carried verbatim so the
    checkpoint writer has every field it needs without
    re-reading the source.
    """

    intercept: float
    coefficients: tuple[float, ...]
    cut_id: int
    iteration: int
    forward_pass_index: int
    is_active: bool
    #: Inflow-lag gradient terms keyed by cobre hydro id (``{hydro_id:
    #: (coef_depth1, …, coef_depthN)}``), separate from the storage-aligned
    #: ``coefficients``. Populated only when the boundary needs lag slots the
    #: target manifest does not yet carry (``map_boundary_cuts``'s
    #: ``inflow_lag_depth``); cobre's ``write_policy_checkpoint`` reserves the
    #: canonical ``HydroInflowLag`` slots and places these values. Empty (the
    #: default) when the manifest already carries the lag slots (placed into
    #: ``coefficients``) or the boundary prices no inflow-lag state.
    inflow_lag_coefficients: Mapping[int, tuple[float, ...]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class DroppedTerm:
    """A source-only plant's storage term, dropped rather than folded (D3).

    ``beta`` is that plant's ``pi_varm`` coefficient in the source's first
    cut record — a representative value for the diagnostics layer
    to report; drop status itself is a per-plant, cut-invariant property
    (target availability does not vary cut to cut), so exactly one
    :class:`DroppedTerm` is recorded per unresolvable plant, not one per cut.
    """

    plant_code: int
    beta: float


@dataclass(frozen=True)
class GnlThermalTarget:
    """One GNL thermal's ring membership: which `pi_gnl` axes feed it.

    ``thermal_id`` is the cobre ring `entity_id`; ``submercado`` is the
    1-based source submercado index matching the `pi_gnl` sbm-major column
    layout (`cortes.py::_gnl_columns`); ``nl_lag`` is the plant's 1-based
    dispatch-anticipation lag — the `pi_gnl` lag-axis index whose coefficient
    lands on this thermal's ring slot(s) [ASSUMPTION B].
    """

    thermal_id: int
    submercado: int
    nl_lag: int


@dataclass(frozen=True)
class GnlRingPlan:
    """The resolved submercado -> GNL-thermal membership for one deck.

    Built by the importer from the deck's own GNL declarations;
    this module never derives it — see the module docstring's deck-free
    contract. ``post_horizon_start`` is the ``YYYYMM01`` month-anchor of the
    earliest post-study stage, computed by the importer from
    ``post_study_stages.json`` — this module stays deck-free and never reads
    that file itself, only the threaded-in int. Under cobre's excised
    anticipated ring, a dated ring slot is *covered* (receives the `pi_gnl`
    coefficient) when ``post_horizon_start is None`` (no filter — the
    default, so every existing construction keeps placing on all dated
    slots) or its ``interval_start >= post_horizon_start`` — i.e. it is one
    of the class-3 signaled lanes; otherwise it is a non-covered in-study
    delivery and is dropped (see :func:`_resolve_gnl_targets`).
    """

    targets: tuple[GnlThermalTarget, ...]
    post_horizon_start: int | None = None


@dataclass(frozen=True)
class GnlDroppedTerm:
    """A GNL source/target term that reached no live dated ring slot.

    ``thermal_id`` is ``None`` for a source-submercado drop (no
    :class:`GnlThermalTarget` claims that submercado at all) and the
    resolved thermal id for a target-side drop — out-of-range
    lag/submercado, a thermal with no dated ring slot at all, or
    a dated slot whose ``interval_start`` falls before the
    post-study horizon (a non-covered lane). ``coefficient`` is a
    representative value for the diagnostics layer to report:
    the summed source coefficient for a source-submercado drop; for most
    target-side drops it is ``0.0`` (no source coefficient is attributable
    to a target that never resolves at all) — EXCEPT the non-covered
    post-horizon-lane drop, whose column set *does* resolve, so it carries
    the representative summed ``pi_gnl`` coefficient that WOULD have been
    placed had the slot been covered (see ``_first_active_gnl_sum``).
    Recorded once per unresolvable (submercado, lag) or target, never
    folded into a neighbour (D3-like).
    """

    thermal_id: int | None
    submercado: int
    nl_lag: int
    coefficient: float
    reason: str


@dataclass(frozen=True)
class MappingResult:
    """The mapped cuts plus every D3-dropped source-only term.

    ``dropped`` carries storage/lag source-only plants; ``gnl_dropped``
    (defaulted, so existing constructions keep working) carries GNL
    source-submercado and target terms with no live dated ring slot.
    """

    cuts: tuple[MappedCut, ...]
    dropped: tuple[DroppedTerm, ...]
    gnl_dropped: tuple[GnlDroppedTerm, ...] = ()


def _default_lag_slot_of(depth: int) -> int:
    """Identity calendar-month lag mapping: source lag depth d -> subindex d-1."""
    return depth - 1


def _slot_int(slot: Mapping[str, object], field: str) -> int:
    """Extract `field` from a manifest slot dict, validating it is an `int`.

    cobre's own `load_policy` hands back each slot as an untyped
    `dict[str, object]`; this narrows the three positional-key fields
    (`entity_type`, `entity_id`, `subindex`) with an explicit runtime check
    rather than a bare `cast`, so a malformed slot fails loudly here instead
    of silently corrupting a downstream index lookup.
    """
    value = slot[field]
    if not isinstance(value, int):
        raise TypeError(
            f"manifest slot field {field!r} is {type(value).__name__}, not "
            f"int: {value!r}"
        )
    return value


def _index_manifest(manifest: TerminalManifest) -> dict[tuple[int, int, int], int]:
    """Index the target manifest by (entity_type, entity_id, subindex) -> position."""
    return {
        (
            _slot_int(slot, "entity_type"),
            _slot_int(slot, "entity_id"),
            _slot_int(slot, "subindex"),
        ): position
        for position, slot in enumerate(manifest.entity_manifest)
    }


def _index_gnl_ring(
    manifest: TerminalManifest,
) -> dict[int, tuple[tuple[int, int, int], ...]]:
    """Index the target manifest's `AnticipatedThermalState` ring by thermal id.

    Unlike `_index_manifest` (which discards the slot's dates and would
    silently collapse a thermal's sentinel and dated slots onto the same
    key), this keeps every `(subindex, interval_start, position)` triple per
    `entity_id` so the GNL placement can tell a dated slot from the undated
    sentinel. `interval_start` is the slot's day-accurate delivery-stage start
    (`YYYYMMDD`), the anchor the covered/non-covered split below compares
    against the post-study horizon. A separate index from `_index_manifest`'s
    `(entity_type, entity_id, subindex) -> position` contract, which the
    storage/lag path still depends on unchanged.
    """
    by_thermal: dict[int, list[tuple[int, int, int]]] = {}
    for position, slot in enumerate(manifest.entity_manifest):
        if _slot_int(slot, "entity_type") != _ANTICIPATED_THERMAL_STATE:
            continue
        thermal_id = _slot_int(slot, "entity_id")
        subindex = _slot_int(slot, "subindex")
        interval_start = _slot_int(slot, "interval_start")
        by_thermal.setdefault(thermal_id, []).append(
            (subindex, interval_start, position)
        )
    return {thermal_id: tuple(slots) for thermal_id, slots in by_thermal.items()}


def _lag_subindex_bound(slot_positions: Mapping[tuple[int, int, int], int]) -> int:
    """One past the max `HydroInflowLag` subindex present, or 0 if absent.

    ``0`` signals the family is entirely absent from the manifest — a
    legitimate case shape (a storage-only converted case),
    not a read bug — and disables the `lag_slot_of` bounds check entirely.
    """
    lag_subindices = [
        subindex
        for entity_type, _entity_id, subindex in slot_positions
        if entity_type == _HYDRO_INFLOW_LAG
    ]
    return max(lag_subindices) + 1 if lag_subindices else 0


def _validated_lag_subindices(
    lag_slot_of: Callable[[int], int], lag_bound: int
) -> tuple[int, ...]:
    """Resolve `lag_slot_of(d)` for calendar-month depth `d` in 1..12.

    Returns a 12-tuple of target subindices (index 0 == lag depth 1),
    bound-checked against `lag_bound` up front — before any cut is
    processed — so a mis-injected `lag_slot_of` fails fast with a
    `ValueError` naming the offending depth, never an `IndexError` from a
    downstream indexing operation. The check is skipped when `lag_bound`
    is 0 (no `HydroInflowLag` family in the manifest at all).
    """
    subindices = tuple(lag_slot_of(depth) for depth in range(1, 13))
    if lag_bound == 0:
        return subindices
    for depth, subindex in enumerate(subindices, start=1):
        if not 0 <= subindex < lag_bound:
            raise ValueError(
                f"lag_slot_of({depth}) = {subindex} is out of range for the "
                f"manifest's {lag_bound} HydroInflowLag slot(s)"
            )
    return subindices


def _component_codes(
    plant_code: int,
    id_map: DecompIdMap,
    complexo_components: Mapping[int, Sequence[int]] | None,
) -> tuple[int, ...]:
    """The DECOMP plant code(s) a source header code resolves onto.

    An operated plant is its own single component. A NEWAVE *complexo* code —
    absent from the DECOMP model but listed in `complexo_components` (the `CX`
    register: a complexo aggregates several DECOMP plants that share its future
    cost) — resolves onto its component plants, so the complexo's cut
    coefficients replicate onto each (see :func:`_resolve_storage_targets`). An
    unknown code that is neither operated nor a complexo resolves onto nothing
    (dropped).
    """
    try:
        id_map.hydro_id(plant_code)
    except KeyError:
        if complexo_components is not None:
            return tuple(complexo_components.get(plant_code, ()))
        return ()
    return (plant_code,)


def _resolve_storage_targets(
    cuts: BoundaryCuts,
    id_map: DecompIdMap,
    slot_positions: Mapping[tuple[int, int, int], int],
    complexo_components: Mapping[int, Sequence[int]] | None = None,
) -> tuple[dict[int, tuple[tuple[int, int], ...]], tuple[DroppedTerm, ...]]:
    """Resolve each source plant's target `(hydro_id, storage position)` slot(s).

    Normally a source plant maps 1:1 to its own cobre id, so its value is a
    one-tuple `((hydro_id, position),)`. A NEWAVE *complexo* header code
    (absent from `id_map` but present in `complexo_components` — the `CX`
    register) maps **1→many** onto its DECOMP component plants: the complexo's
    coefficients are replicated onto EACH component's slot, so the sum over the
    components reconstructs the complexo's aggregate cut term (the complexo's
    aggregate storage/inflow state is the sum of its components' — see the
    module header). Each component gets the *full* coefficient (a
    sum-decomposition, not a split).

    A source plant/complexo is dropped (D3) only when it resolves onto no live
    target at all — an unknown code that is neither operated nor a complexo, or
    a complexo whose every component is unknown to `id_map` or lacks a
    `HydroStorage` slot. Its storage *and* inflow-lag terms are then omitted
    from every mapped cut, never folded into a neighbour. A component that
    individually lacks a `HydroStorage` slot is skipped without dropping the
    whole complexo.
    """
    representative_varm = cuts.records[0].pi_varm if cuts.records else ()
    resolved: dict[int, tuple[tuple[int, int], ...]] = {}
    dropped: list[DroppedTerm] = []
    for plant_index, plant_code in enumerate(cuts.header.plant_codes):
        targets: list[tuple[int, int]] = []
        for component_code in _component_codes(plant_code, id_map, complexo_components):
            try:
                hydro_id = id_map.hydro_id(component_code)
            except KeyError:
                continue
            position = slot_positions.get((_HYDRO_STORAGE, hydro_id, _STORAGE_SUBINDEX))
            if position is not None:
                targets.append((hydro_id, position))
        if targets:
            resolved[plant_index] = tuple(targets)
        else:
            beta = representative_varm[plant_index] if representative_varm else 0.0
            dropped.append(DroppedTerm(plant_code=plant_code, beta=beta))
    return resolved, tuple(dropped)


def _first_active_gnl_sum(cuts: BoundaryCuts, cols: tuple[int, ...]) -> float:
    """The chain-rule `Σ_p pi_gnl[cols]` from `cuts`' first active record.

    A representative value for a non-covered dated slot's `GnlDroppedTerm`
    — the sum that WOULD have been placed had the slot been
    covered — mirroring `_resolve_storage_targets`'s representative-`beta`
    convention. `0.0` when `cuts` carries no active record at all (never
    raised on; the diagnostics layer reports a representative figure, not a
    load-bearing one).
    """
    for record in cuts.records:
        if record.is_active:
            return math.fsum(record.pi_gnl[column] for column in cols)
    return 0.0


def _resolve_gnl_targets(
    cuts: BoundaryCuts,
    ring_index: Mapping[int, tuple[tuple[int, int, int], ...]],
    gnl_plan: GnlRingPlan | None,
) -> tuple[dict[int, tuple[int, ...]], tuple[GnlDroppedTerm, ...]]:
    """Resolve each GNL target's ring position(s) and `pi_gnl` columns once.

    Mirrors `_resolve_storage_targets`'s "resolve once, write per record"
    split: cut-invariant, computed exactly once regardless of how many
    records `cuts` carries. Returns `resolved: {ring position -> tuple of
    pi_gnl flat-column indices to sum}` plus every GNL drop.

    `ring_index` (built from the reloaded terminal manifest) never carries
    the já-comandada (class-4) window at all — cobre excises it from the
    ring entirely — so a target's dated slot(s) are exactly the in-study and
    class-3 signaled deliveries, keyed by each slot's `interval_start` (its
    delivery-stage start day). They split into *covered*
    (`interval_start >= gnl_plan.post_horizon_start`, or
    `post_horizon_start is None` — no filter) — the class-3 signaled lanes —
    and *non-covered* (`interval_start < post_horizon_start`) — an in-study
    delivery, priced by the in-study committed window rather than the ring:
    only covered slots land in `resolved`; each non-covered slot is dropped
    into the returned `GnlDroppedTerm`s (reason names the in-study committed
    window), staying at coefficient `0.0` exactly like the existing GNL
    drops — never folded onto a neighbour. A target with dated slots that
    are *all* non-covered still gets the non-covered drop path, never the
    "no dated ring slot for thermal" reason (that one is reserved for a
    target with zero dated slots at all).

    Returns `({}, ())` — no GNL mapping at all — when `gnl_plan` is `None`,
    `cuts` has no records, the source deck carries no GNL lag axis
    (`lag_maximo_gnl == 0`), or the boundary's `pi_gnl` is empty (a
    non-GNL deck).

    Raises
    ------
    ValueError
        If `len(cuts.records[0].pi_gnl)` is not a multiple of
        `n_patamares * lag_maximo_gnl` — a reader/plan layout
        inconsistency, named explicitly rather than silently truncated or
        padded.
    """
    n_patamares = cuts.header.n_patamares
    lag_maximo_gnl = cuts.header.lag_maximo_gnl
    if gnl_plan is None or not cuts.records or lag_maximo_gnl == 0:
        return {}, ()
    width = len(cuts.records[0].pi_gnl)
    if width == 0:
        return {}, ()

    block = n_patamares * lag_maximo_gnl
    if width % block != 0:
        raise ValueError(
            f"pi_gnl width {width} is not a multiple of n_patamares "
            f"({n_patamares}) * lag_maximo_gnl ({lag_maximo_gnl}) = {block}"
        )
    n_submercados = width // block

    def col(submercado: int, patamar: int, lag: int) -> int:
        """Flat pi_gnl column for (submercado, patamar, lag), 1-based axes."""
        return ((submercado - 1) * n_patamares + (patamar - 1)) * lag_maximo_gnl + (
            lag - 1
        )

    resolved: dict[int, tuple[int, ...]] = {}
    dropped: list[GnlDroppedTerm] = []
    targeted_submercados: set[int] = set()

    for target in gnl_plan.targets:
        if not (1 <= target.submercado <= n_submercados) or not (
            1 <= target.nl_lag <= lag_maximo_gnl
        ):
            dropped.append(
                GnlDroppedTerm(
                    thermal_id=target.thermal_id,
                    submercado=target.submercado,
                    nl_lag=target.nl_lag,
                    coefficient=0.0,
                    reason="lag/submercado out of pi_gnl range",
                )
            )
            continue
        targeted_submercados.add(target.submercado)
        dated = tuple(
            (interval_start, position)
            for _subindex, interval_start, position in ring_index.get(
                target.thermal_id, ()
            )
            if interval_start != _SLOT_DATE_SENTINEL
        )
        if not dated:
            dropped.append(
                GnlDroppedTerm(
                    thermal_id=target.thermal_id,
                    submercado=target.submercado,
                    nl_lag=target.nl_lag,
                    coefficient=0.0,
                    reason="no dated ring slot for thermal",
                )
            )
            continue

        post_horizon_start = gnl_plan.post_horizon_start
        covered = tuple(
            position
            for interval_start, position in dated
            if post_horizon_start is None or interval_start >= post_horizon_start
        )
        has_uncovered = any(
            post_horizon_start is not None and interval_start < post_horizon_start
            for interval_start, _position in dated
        )

        cols = tuple(
            col(target.submercado, patamar, target.nl_lag)
            for patamar in range(1, n_patamares + 1)
        )
        for position in covered:
            resolved[position] = cols

        if has_uncovered:
            dropped.append(
                GnlDroppedTerm(
                    thermal_id=target.thermal_id,
                    submercado=target.submercado,
                    nl_lag=target.nl_lag,
                    coefficient=_first_active_gnl_sum(cuts, cols),
                    reason=(
                        "delivery before the post-study horizon (in-study, "
                        "priced by the in-study committed window, not the "
                        "anticipated ring)"
                    ),
                )
            )

    # Source-submercado drops (D3-like): a submercado no target claims, but
    # whose pi_gnl carries a nonzero coefficient on some active record, has
    # a source term with no live GNL thermal to receive it.
    active_records = tuple(record for record in cuts.records if record.is_active)
    for submercado in range(1, n_submercados + 1):
        if submercado in targeted_submercados:
            continue
        for lag in range(1, lag_maximo_gnl + 1):
            lag_cols = tuple(
                col(submercado, patamar, lag) for patamar in range(1, n_patamares + 1)
            )
            first_nonzero = next(
                (
                    record
                    for record in active_records
                    if any(record.pi_gnl[c] != 0.0 for c in lag_cols)
                ),
                None,
            )
            if first_nonzero is None:
                continue
            dropped.append(
                GnlDroppedTerm(
                    thermal_id=None,
                    submercado=submercado,
                    nl_lag=lag,
                    coefficient=math.fsum(first_nonzero.pi_gnl[c] for c in lag_cols),
                    reason="no GNL thermal in submercado",
                )
            )

    return resolved, tuple(dropped)


def map_boundary_cuts(
    cuts: BoundaryCuts,
    manifest: TerminalManifest,
    id_map: DecompIdMap,
    *,
    cost_unit_hours: float,
    lag_slot_of: Callable[[int], int] = _default_lag_slot_of,
    gnl_plan: GnlRingPlan | None = None,
    coupling_block_hours: Sequence[float] | None = None,
    inflow_lag_means: Mapping[int, Sequence[float]] | None = None,
    complexo_components: Mapping[int, Sequence[int]] | None = None,
    inflow_lag_depth: int = 0,
) -> MappingResult:
    """Map every cut in `cuts.records` onto `manifest`'s state-vector layout.

    `cost_unit_hours` is the coupling (terminal) stage's duration in hours: the
    source model's FCF coefficients are a per-hour cost rate (``($·mês)/h`` etc.,
    see the module header), so every mapped term is scaled to cobre's plain-$
    objective units by integrating over these hours. The intercept and storage
    terms take ``× cost_unit_hours``; the inflow-lag term takes an additional
    ``× C_M3S2HM3`` because cobre's inflow-lag state is a m³/s flow rate, not the
    Hm³ volume ``PIAFL`` is defined against (see the module header). The GNL term
    instead takes the *per-block* hours-weighted collapse
    `Σ_p pi_gnl[p] · coupling_block_hours[p]` — `pi_gnl` is a `$/MWh` energy
    price on a flat-power ring state, so the coupling stage's total hours would
    over-count it by `n_patamares` (see the module header's derivation).

    Storage terms join by plant code (`HydroStorage`, D3-drop on a
    source-only plant); inflow-lag terms join 1:1 by `lag_slot_of` onto
    `HydroInflowLag`. When `gnl_plan` is given, each `AnticipatedThermalState`
    ring slot named by one of its targets' *covered* dated slot(s) — i.e.
    `interval_start >= gnl_plan.post_horizon_start`, or every dated slot when
    `post_horizon_start is None` — carries the hours-weighted patamar sum
    `Σ_p pi_gnl[col(s,p,nl_lag)] · coupling_block_hours[p]` (`math.fsum`,
    order-independent); a target with no dated ring slot, a dated slot whose
    delivery falls before the post-study horizon (an in-study delivery,
    priced by the in-study committed window rather than the ring), or a
    source submercado with no matching target, is dropped and recorded in
    `MappingResult.gnl_dropped`, never folded onto a neighbour.
    `HydroTransitBucket` slots, the sentinel (undated) `AnticipatedThermalState`
    slot, and every ring slot with no resolved (covered) target are left at
    an explicit `0.0` regardless of `gnl_plan`; `gnl_plan=None` (the default)
    leaves the entire ring at `0.0`, byte-for-byte matching this function's
    pre-GNL behaviour, and never requires `coupling_block_hours` (see the
    guard below). `intercept` is the source record's `rhs` (never re-derived
    from alpha/x-hat), scaled to cobre's cost units and then reduced by the
    `inflow_lag_means` fold below; every
    coefficient is likewise scaled to cobre's cost units per family —
    intercept/storage by ``× cost_unit_hours``, inflow-lag by an additional
    ``× C_M3S2HM3`` for cobre's m³/s inflow-lag state, and GNL by the
    per-block `coupling_block_hours` collapse above (see the module header).
    Produces one `MappedCut` per source record, active or not —
    active-frontier selection is the checkpoint writer's concern, not the
    mapper's.

    `coupling_block_hours` is the coupling stage's per-block hours in patamar
    order — required whenever `gnl_plan` resolves at least one live ring
    target (`resolved_gnl` non-empty); `None` is only valid when no GNL
    coefficient is placed (no `gnl_plan`, or every target dropped).

    `inflow_lag_means` folds the seasonal-mean inflow into the cut RHS (see the
    module header): `{hydro_id: (mu_depth1, …, mu_depth12)}` in m³/s, one
    12-vector per plant aligned to the boundary cut's lag-depth axis (built by
    `decomp/inflow_mlt.py::coupling_lag_means`). The source prices the inflow
    *deviation* `Q - mu`, but cobre evaluates the loaded cut at its raw lag
    state `Q`, so for every lag coefficient actually placed the intercept is
    reduced by `placed_coef · mu[depth]`, making the loaded cut
    `RHS_scaled - Σ coef·mu + Σ coef·Q = RHS_scaled + Σ coef·(Q - mu)`. Summed
    over exactly the placed (plant, lag) terms, so a dropped plant or lag never
    contributes to the fold. `None` (the default) folds nothing — the intercept
    is the plain scaled `rhs` — byte-for-byte the pre-fold behaviour.

    `complexo_components` maps a NEWAVE *complexo* header code to its DECOMP
    component plant codes (the `CX` register — a complexo aggregates several
    DECOMP plants that share one future cost). A complexo code (absent from
    `id_map`) would otherwise be dropped; given this map, its cut coefficients
    are **replicated onto each component** (storage on each component's storage
    slot, inflow-lag on each component's lag slots, both scaled and folded
    exactly as an ordinary plant's), so the sum over the components reconstructs
    the complexo's aggregate term (aggregate state = Σ component states — a
    sum-decomposition, each gets the full coefficient, not `coef/N`). `None`
    (the default) leaves complexo codes dropped, byte-for-byte the pre-CX
    behaviour.

    Raises
    ------
    ValueError
        If the target manifest has no `HydroStorage` slots at all (a
        terminal-manifest read bug — never raised for a merely-absent
        `HydroInflowLag`/`HydroTransitBucket`/`AnticipatedThermalState`
        family, a legitimate case shape); if `lag_slot_of` returns a
        subindex out of range for the manifest's `HydroInflowLag` slots; if
        `gnl_plan` is given and `cuts`' `pi_gnl` width is not a multiple of
        `n_patamares * lag_maximo_gnl`; if a GNL coefficient would be placed
        and `coupling_block_hours` is `None` or its length disagrees with
        `cuts.header.n_patamares`; or if a mapped coefficient vector's
        length disagrees with `manifest.state_dimension`.
    """
    slot_positions = _index_manifest(manifest)
    if not any(entity_type == _HYDRO_STORAGE for entity_type, _, _ in slot_positions):
        raise ValueError(
            "target manifest has no HydroStorage slots at all; a real "
            "terminal-manifest read must carry at least the storage family"
        )

    resolved_storage, dropped = _resolve_storage_targets(
        cuts, id_map, slot_positions, complexo_components
    )
    lag_bound = _lag_subindex_bound(slot_positions)
    lag_subindices = _validated_lag_subindices(lag_slot_of, lag_bound)

    gnl_ring_index = _index_gnl_ring(manifest)
    resolved_gnl, gnl_dropped = _resolve_gnl_targets(cuts, gnl_ring_index, gnl_plan)

    # The GNL branch needs the coupling stage's per-block hours (patamar
    # order), never merely its total, to weight each pi_gnl patamar column
    # independently (see the module header). Validated once here, next to
    # `resolved_gnl` itself, rather than inside the per-record loop below —
    # mirrors `_resolve_storage_targets`/`_validated_lag_subindices`'s
    # "resolve/validate once, write per record" split. A `gnl_plan` that
    # resolves no live target (`resolved_gnl` empty) needs no per-block hours
    # at all, so `coupling_block_hours=None` is never an error in that case.
    if resolved_gnl:
        if coupling_block_hours is None:
            raise ValueError(
                "GNL ring placement requires coupling_block_hours (the "
                "coupling stage's per-block hours in patamar order), but "
                "none was given"
            )
        if len(coupling_block_hours) != cuts.header.n_patamares:
            raise ValueError(
                f"coupling_block_hours has {len(coupling_block_hours)} "
                f"block(s), but cuts.header.n_patamares expects "
                f"{cuts.header.n_patamares}"
            )

    # Cost-unit factors (see the module header): the intercept/storage terms
    # integrate the per-hour source rate over the coupling stage's hours; the
    # inflow-lag term additionally converts cobre's m³/s lag state to the Hm³
    # `PIAFL` is defined against. The GNL term is scaled separately, per
    # coupling block, inside the per-record loop below.
    cost_unit_factor = cost_unit_hours
    inflow_lag_factor = cost_unit_hours * C_M3S2HM3

    mapped_cuts: list[MappedCut] = []
    for record in cuts.records:
        coefficients = [0.0] * manifest.state_dimension
        # Mean-fold accumulator (see the module header + `inflow_lag_means`): the
        # source prices the inflow *deviation* Q - mu, but cobre evaluates the
        # loaded cut at its raw lag state Q, so the seasonal mean is folded into
        # the intercept. Summed per record over exactly the lag coefficients
        # actually placed, so a dropped plant/lag contributes nothing to the
        # fold either — the fold can never reference a term cobre won't apply.
        inflow_lag_coefficients: dict[int, tuple[float, ...]] = {}
        rhs_fold = 0.0
        for plant_index, targets in resolved_storage.items():
            # `targets` is one slot for an ordinary plant, or several for a
            # complexo (CX): the same coefficient replicates onto every
            # component, so `Σ` over them reconstructs the complexo's aggregate
            # term (aggregate state = Σ component states — module header).
            storage_coefficient = record.pi_varm[plant_index] * cost_unit_factor
            plant_lags = record.pi_qafl[plant_index]
            for hydro_id, storage_position in targets:
                coefficients[storage_position] = storage_coefficient
                plant_means = (
                    inflow_lag_means.get(hydro_id)
                    if inflow_lag_means is not None
                    else None
                )
                if lag_bound > 0:
                    # The manifest already carries this hydro's HydroInflowLag
                    # slots — place each depth's coefficient into the aligned
                    # vector (join 1:1 by `lag_slot_of`).
                    for depth_index, subindex in enumerate(lag_subindices):
                        lag_position = slot_positions.get(
                            (_HYDRO_INFLOW_LAG, hydro_id, subindex)
                        )
                        if lag_position is not None:
                            lag_coefficient = (
                                plant_lags[depth_index] * inflow_lag_factor
                            )
                            coefficients[lag_position] = lag_coefficient
                            if plant_means is not None:
                                rhs_fold += lag_coefficient * plant_means[depth_index]
                elif inflow_lag_depth > 0:
                    # The manifest carries no lag slots (a DECOMP case has no
                    # PAR(p) model for cobre to size them from), so emit the lag
                    # coefficients keyed by hydro (depth 1..N); cobre's
                    # write_policy_checkpoint reserves the canonical
                    # HydroInflowLag slots and places them. Same per-depth
                    # scaling and mean-fold as the aligned path above.
                    lag_coeffs = tuple(
                        plant_lags[depth_index] * inflow_lag_factor
                        for depth_index in range(inflow_lag_depth)
                    )
                    inflow_lag_coefficients[hydro_id] = lag_coeffs
                    if plant_means is not None:
                        for depth_index in range(inflow_lag_depth):
                            rhs_fold += (
                                lag_coeffs[depth_index] * plant_means[depth_index]
                            )

        for gnl_position, gnl_cols in resolved_gnl.items():
            # Hours-weighted collapse: `pi_gnl` prices an energy
            # state, so each patamar column is weighted by that patamar's own
            # coupling-block hours, not by the coupling stage's total hours.
            # `coupling_block_hours` is guaranteed non-None here — validated
            # above whenever `resolved_gnl` is non-empty.
            coefficients[gnl_position] = math.fsum(
                record.pi_gnl[column] * hours
                for column, hours in zip(
                    gnl_cols,
                    coupling_block_hours,  # type: ignore[arg-type]
                    strict=True,
                )
            )

        if len(coefficients) != manifest.state_dimension:
            raise ValueError(
                f"mapped coefficient vector length {len(coefficients)} != "
                f"state_dimension {manifest.state_dimension}"
            )

        mapped_cuts.append(
            MappedCut(
                intercept=record.rhs * cost_unit_factor - rhs_fold,
                coefficients=tuple(coefficients),
                cut_id=record.cut_id,
                iteration=record.iteration,
                forward_pass_index=record.forward_pass_index,
                is_active=record.is_active,
                inflow_lag_coefficients=inflow_lag_coefficients,
            )
        )

    return MappingResult(
        cuts=tuple(mapped_cuts), dropped=dropped, gnl_dropped=gnl_dropped
    )
