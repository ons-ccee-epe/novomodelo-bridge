"""Assemble and write the source model's boundary-cut checkpoint.

The mapper (``fcf/mapper.py``) produces a :class:`MappingResult`
of :class:`~novomodelo_bridge.decomp.fcf.mapper.MappedCut`, each already aligned
to the terminal manifest's state-vector layout (``fcf/bootstrap.py``).
This module assembles that pair into the plain-dict payload and
metadata shapes ``novomodelo.write_policy_checkpoint`` expects, then calls it to
produce ``boundary/{manifest.bin, cuts/<pool>.bin, basis/}`` — a raw,
one-pool checkpoint the target case's ``config.json -> policy.boundary``
loads.

Every pool payload carries ``cost_scale_factor``, the units of its cut values:
novomodelo refuses a pool without one. :func:`build_stage_cuts_payload` and
:func:`build_metadata` (which records it in the ``producer`` block as
provenance) refuse an unset factor before any novomodelo call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from novomodelo_bridge.decomp.fcf.bootstrap import ensure_writer_binding

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from novomodelo_bridge.decomp.fcf.bootstrap import TerminalManifest
    from novomodelo_bridge.decomp.fcf.mapper import MappingResult


def build_stage_cuts_payload(
    mapping: MappingResult,
    manifest: TerminalManifest,
    *,
    stage_id: int,
    cost_scale_factor: float | None,
    node_id: int,
    graph_stage_id: int,
    priced_state_date: int,
) -> dict[str, Any]:
    """Assemble one ``stage_cuts`` entry from `mapping` over `manifest`.

    Each ``cuts[]`` entry carries the source ``MappedCut``'s real provenance
    verbatim — ``iteration``, ``forward_pass_index``, and ``is_active`` are
    never re-synthesized — with ``slot_index`` set to the cut's dense 0-based
    position in the pool (its `enumerate` index here, not the source
    ``cut_id``, which the checkpoint format does not hold).
    ``entity_manifest`` is `manifest.entity_manifest` copied verbatim
    (never re-derived) so a future novomodelo layout change breaks loudly at
    load, not silently, and so each slot's own date fields
    (``reference_date``/``interval_start``/``interval_end``) round-trip.
    A cut's activity rides on its own ``is_active``; ``populated_count`` is
    `len(cuts)`. ``cost_scale_factor``,
    ``node_id``, ``graph_stage_id``, and ``priced_state_date`` are copied
    verbatim into the payload — the boundary loader reads them per pool to
    select the source pool by calendar date (matching ``priced_state_date``
    against the loading study's own boundary date), to require a single-node
    terminal source, and to reject a checkpoint that lacks them.

    Raises
    ------
    ValueError
        If `node_id == -1` (the shared-pool sentinel is not a real single
        node), or if any `MappedCut.coefficients` length disagrees with
        `manifest.state_dimension` — the sentinel check runs before the cut
        loop, the length check inside it, both before any novomodelo call.
    RuntimeError
        If `cost_scale_factor` is `None` — novomodelo refuses a pool without
        its factor.
    """
    if cost_scale_factor is None:
        raise RuntimeError(
            "cost_scale_factor must not be None: novomodelo rejects a cut "
            "pool without the factor its values are in"
        )
    if node_id == -1:
        raise ValueError(
            "node_id must not be -1: that value is the shared-pool "
            "sentinel, not a real single node"
        )

    cuts: list[dict[str, Any]] = []
    for slot_index, mapped in enumerate(mapping.cuts):
        if len(mapped.coefficients) != manifest.state_dimension:
            raise ValueError(
                f"cut {mapped.cut_id}: coefficients has "
                f"{len(mapped.coefficients)} entries, expected "
                f"state_dimension={manifest.state_dimension}"
            )
        cuts.append(
            {
                "slot_index": slot_index,
                "iteration": mapped.iteration,
                "forward_pass_index": mapped.forward_pass_index,
                "intercept": mapped.intercept,
                "coefficients": list(mapped.coefficients),
                "is_active": mapped.is_active,
                # Inflow-lag gradient terms keyed by hydro (depth 1..N), placed by
                # novomodelo's write_policy_checkpoint into the canonical HydroInflowLag
                # slots it reserves. Empty when the boundary prices no lag.
                "inflow_lag_coefficients": {
                    hydro_id: list(coeffs)
                    for hydro_id, coeffs in mapped.inflow_lag_coefficients.items()
                },
            }
        )

    return {
        "stage_id": stage_id,
        "state_dimension": manifest.state_dimension,
        "capacity": len(cuts),
        "cuts": cuts,
        "populated_count": len(cuts),
        "entity_manifest": list(manifest.entity_manifest),
        "cost_scale_factor": cost_scale_factor,
        "node_id": node_id,
        "graph_stage_id": graph_stage_id,
        "priced_state_date": priced_state_date,
    }


def build_metadata(
    *,
    num_stages: int,
    cost_scale_factor: float | None,
    completed_iterations: int,
    final_lower_bound: float,
    max_iterations: int,
    forward_passes: int,
    warm_start_cuts: int,
    rng_seed: int,
    created_at: str,
    season_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the checkpoint `metadata` dict, refusing an unset `cost_scale_factor`.

    Checkpoint metadata is a small core (`created_at`, `num_stages`) plus a
    namespaced `producer` block carrying the algorithm-specific provenance.
    The software identity is not a caller field: novomodelo stamps the writing
    build's own `software` and `software_version`, and only that build loads
    the checkpoint. `state_dimension`
    is not a metadata field: it is per-pool, on each `stage_cuts` payload
    (:func:`build_stage_cuts_payload`). `created_at` is accepted as a parameter
    rather than derived internally (this module never calls `datetime.now()`) —
    the caller supplies an ISO 8601 timestamp.

    `season_manifest` (the study-global `cycle_code`/`n_seasons`/`hydro_orders`
    descriptor) is copied in when given, so novomodelo's season-compatibility gate
    accepts the boundary. Omitted (`None`), the key is left off entirely and
    novomodelo defaults it to the absent descriptor — correct only for a seasonless
    loading study; the boundary-FCF importer always supplies the study's own.

    Raises
    ------
    RuntimeError
        If `cost_scale_factor` is `None`.
    """
    if cost_scale_factor is None:
        raise RuntimeError(
            "cost_scale_factor must not be None: it records the factor the "
            "checkpoint's cut pools are in, which novomodelo requires of "
            "every pool"
        )
    metadata: dict[str, Any] = {
        "created_at": created_at,
        "num_stages": num_stages,
        "producer": {
            "completed_iterations": completed_iterations,
            "final_lower_bound": final_lower_bound,
            "max_iterations": max_iterations,
            "forward_passes": forward_passes,
            "warm_start_cuts": warm_start_cuts,
            "rng_seed": rng_seed,
            "cost_scale_factor": cost_scale_factor,
        },
    }
    if season_manifest is not None:
        metadata["season_manifest"] = dict(season_manifest)
    return metadata


def write_boundary_checkpoint(
    boundary_dir: Path,
    stage_cuts_payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    inflow_lag_depth: int = 0,
) -> None:
    """Write `stage_cuts_payload` + `metadata` to `boundary_dir` via novomodelo.

    Calls :func:`ensure_writer_binding` first (the environment gate), then
    `novomodelo.write_policy_checkpoint(boundary_dir, [stage_cuts_payload], metadata,
    inflow_lag_depth=inflow_lag_depth)` with `stage_bases`/`stage_states` left at
    their defaults — a raw-authored checkpoint carries neither, so
    `boundary_dir/basis/` is written empty and no `states/` directory is created.

    ``inflow_lag_depth`` (when ``>= 1``) has novomodelo reserve that many canonical
    ``HydroInflowLag`` slots and place the cuts' ``inflow_lag_coefficients`` — the
    boundary then self-describes its lag depth. ``0`` (the default, and the
    storage-only case) reserves no slots, leaving the checkpoint byte-identical to
    a no-lag boundary. The reservation is a novomodelo-side feature the pin and
    :data:`~novomodelo_bridge.novomodelo.compat.MIN_NOVOMODELO_VERSION` floor guarantee is present.

    Raises
    ------
    RuntimeError
        Via `ensure_writer_binding`, if the installed `novomodelo` wheel lacks
        `write_policy_checkpoint`.
    """
    ensure_writer_binding()
    import novomodelo

    novomodelo.write_policy_checkpoint(
        boundary_dir,
        [stage_cuts_payload],
        metadata,
        inflow_lag_depth=inflow_lag_depth,
    )
