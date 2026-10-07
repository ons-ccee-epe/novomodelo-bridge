"""Bootstrap the terminal cut-slot manifest from a 1-iteration cobre run.

The importer must author boundary cuts whose per-slot layout
(``entity_type``, ``entity_id``, ``subindex``) matches cobre's own terminal
pool exactly — cobre validates that identity on load. Rather than
re-implement cobre's slot-layout rules in Python (fragile, and it drifts
silently the moment cobre's own layout changes), this module runs cobre
itself for exactly one training iteration on the converted case, with
checkpointing on, and reads the emitted checkpoint's terminal
``entity_manifest`` + ``state_dimension`` back via
:func:`cobre.results.load_policy`. The mapper and the
writer then author cuts against that manifest verbatim, so any
future cobre layout change breaks loudly — a manifest mismatch on load — not
silently.

The run is driven **in-process** through :func:`cobre.run.run` (the same
cobre-python wheel the writer and reader already require), not a subprocess
``cobre`` binary: one dependency, no binary/wheel version skew, and no
``--cobre-bin`` path to resolve. ``cobre.run.run``'s ``config_overrides``
deep-merges an override into ``config.json`` **in memory** (it never mutates
the case); the only on-disk copy is a disposable flattened variant of the
case under ``work_dir`` (:func:`_flatten_terminal_fan`, which collapses the
terminal fan so the boundary pool has a single owning node), never a mutation
of ``case_dir``. Its ``on_iteration`` callback stops the run cooperatively
after the first iteration boundary — the terminal state-vector layout is
structural, present from iteration one.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def ensure_writer_binding() -> None:
    """Raise if the installed ``cobre`` wheel lacks ``write_policy_checkpoint``.

    The importer's writer needs
    ``cobre.write_policy_checkpoint`` to author a boundary-cut checkpoint;
    this is the environment gate, callable standalone before any
    bootstrap/mapper/writer work runs.

    Raises
    ------
    RuntimeError
        With a self-contained, end-user-facing remediation (reinstall/upgrade
        the required ``cobre-python`` dependency via ``pip``, or re-run with
        ``--no-fcf``). Raised both when ``cobre`` is not installed at all
        (caught as ``ModuleNotFoundError``) and when it is installed but
        predates ``write_policy_checkpoint``.
    """
    try:
        import cobre
    except ModuleNotFoundError:
        raise RuntimeError(
            "cobre is not installed. cobre-python is a required dependency of "
            "cobre-bridge — install it (for example: pip install cobre-python), "
            "or reinstall cobre-bridge. To convert without the boundary "
            "cost-to-go function, re-run with --no-fcf."
        ) from None
    if not hasattr(cobre, "write_policy_checkpoint"):
        raise RuntimeError(
            "The installed cobre package is too old for the boundary cost-to-go "
            "import (it does not provide the policy checkpoint writer). Upgrade "
            "it (for example: pip install --upgrade cobre-python), or reinstall "
            "cobre-bridge. To convert without the boundary cost-to-go function, "
            "re-run with --no-fcf."
        )


#: cobre's ``STAGE_CUTS_PRICED_STATE_DATE_SENTINEL`` (``i32::MIN``) — a pool
#: whose ``priced_state_date`` reads this was written before priced dates were
#: recorded; the date-driven boundary loader rejects such a checkpoint.
_PRICED_STATE_DATE_SENTINEL = -(2**31)


@dataclass(frozen=True)
class TerminalManifest:
    """The terminal stage's per-slot entity identity and state dimension.

    ``entity_manifest`` is the tuple of per-slot dicts
    :func:`cobre.results.load_policy` returns verbatim (each carrying at
    least ``entity_type``, ``entity_id``, ``subindex``); ``state_dimension``
    is the terminal stage's own slot count, cross-checked against the
    checkpoint metadata's ``state_dimension`` in
    :func:`bootstrap_terminal_manifest`. ``node_id`` and ``graph_stage_id``
    are read back from that same terminal pool — the authoritative real
    single-node/graph-stage identity the self-describing checkpoint payload
    carries. ``priced_state_date`` is that pool's own ``YYYYMMDD`` priced
    instant (the terminal stage's exclusive ``end_date``); the boundary
    loader selects the source pool by matching it against the loading study's
    own boundary date, so the authored checkpoint must carry it verbatim.
    ``season_manifest`` is the study-global season/PAR-order descriptor
    (``cycle_code``/``n_seasons``/``hydro_orders``) read from the bootstrap
    checkpoint's metadata; the boundary loader's season-compatibility gate
    rejects a source whose descriptor is absent or disagrees with the loading
    study's, so the authored checkpoint carries the study's own verbatim.
    """

    entity_manifest: tuple[dict[str, object], ...]
    state_dimension: int
    node_id: int
    graph_stage_id: int
    priced_state_date: int
    season_manifest: dict[str, object]


def _flatten_terminal_fan(case_dir: Path, variant_dir: Path) -> None:
    """Materialise a copy of ``case_dir`` under ``variant_dir`` whose
    ``stages.json`` ``policy_graph`` carries empty ``nodes``/``transitions``.

    cobre's chain builder turns an empty ``nodes``/``transitions`` pair into
    one node per stage, so the terminal stage gets a single owning node
    instead of the real case's terminal-fan leaves — see
    :func:`bootstrap_terminal_manifest`'s docstring for why a single-node
    terminal is what the bootstrap run needs. Every other file (``config``,
    ``system``, ``scenarios``, ``constraints``, ``post_study_stages.json``,
    ``initial_conditions.json``) is copied verbatim, so the terminal state
    layout read back from this variant is the real case's own. ``case_dir``
    itself is only ever read, never written.

    Every stage additionally gets ``num_openings: 1``: cobre's schema
    requires an explicit per-stage opening count once ``nodes[]`` is empty
    (the "chain dialect"; a real case's node-native ``stages.json`` never
    sets it, since the fan width there comes from ``nodes``/``transitions``
    instead), and ``1`` is exactly the width that turns the chain into the
    single-node-per-stage graph this variant needs — confirmed empirically
    against the installed cobre wheel: the flattened terminal's
    ``entity_manifest``/``state_dimension``/``graph_stage_id`` come back
    identical to the real fanned case's own, differing only in ``node_id``
    (the real fanned case's ``-1`` shared-pool sentinel becomes a real id).
    """
    shutil.copytree(case_dir, variant_dir, dirs_exist_ok=True)
    stages_path = variant_dir / "stages.json"
    with stages_path.open(encoding="utf-8") as handle:
        stages_doc = json.load(handle)
    stages_doc["policy_graph"]["nodes"] = []
    stages_doc["policy_graph"]["transitions"] = []
    for stage in stages_doc["stages"]:
        stage["num_openings"] = 1
    with stages_path.open("w", encoding="utf-8") as handle:
        json.dump(stages_doc, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def bootstrap_terminal_manifest(case_dir: Path, *, work_dir: Path) -> TerminalManifest:
    """Run cobre for 1 iteration on a single-terminal-leaf variant of
    ``case_dir`` and read its terminal manifest.

    ``case_dir``'s own ``stages.json`` emits a trunk-plus-terminal-fan policy
    graph (one leaf node per scenario at the terminal stage); cobre stamps a
    pool's ``node_id`` from its *sole* owning graph node, so a ≥2-leaf
    terminal fan gives the terminal pool no single owner — the ``-1``
    shared-pool sentinel this function raises on below. cobre's boundary
    loader accepts a single-node *source* checkpoint injected into a fanned
    *target*, so rather than train against ``case_dir`` itself, this trains
    against a :func:`_flatten_terminal_fan` variant under ``work_dir`` whose
    policy graph is a plain per-stage chain — giving the terminal pool a
    real single owner — while every other file, and so the terminal state
    layout read back, is ``case_dir``'s own. ``case_dir`` is never mutated.

    The checkpoint is written under ``work_dir`` (the single-iteration cap is
    applied through ``config_overrides``' in-memory deep-merge, and the
    ``on_iteration`` callback requests a cooperative stop at the first
    iteration boundary). Simulation is skipped.

    The bootstrap reserves no inflow-lag depth explicitly: cobre sizes the
    ``HydroInflowLag`` state block from the case's own PAR(p) model order
    (``resolve_state_layout``), the same autoregressive structure the DECOMP
    boundary cuts' ``pi_qafl`` terms derive from, so the terminal manifest
    already carries the slots the mapper places those terms onto.
    It then reads the emitted checkpoint back via
    :func:`cobre.results.load_policy` and returns the terminal stage's (the
    entry whose ``stage_id`` is max) ``entity_manifest``, ``state_dimension``,
    ``node_id``, ``graph_stage_id``, and ``priced_state_date``, plus the
    study-global ``season_manifest`` from the checkpoint metadata. Because this
    trains ``case_dir``'s own single-terminal-leaf variant, that pool is
    priced at exactly the instant the real case's boundary loader selects a
    source against — reading it back is the authoritative, drift-free source
    for the authored checkpoint's own ``priced_state_date``.

    Depends only on ``cobre.run.run`` and ``cobre.results.load_policy`` —
    neither needs the ``write_policy_checkpoint`` binding, so this function
    does not call :func:`ensure_writer_binding`. The mapper/writer stages
    that consume the returned :class:`TerminalManifest` do need it; a caller
    composing the full bootstrap -> mapper -> writer pipeline calls
    :func:`ensure_writer_binding` once, up front.

    Raises
    ------
    RuntimeError
        If the loaded checkpoint has no stage cuts, an empty terminal
        ``entity_manifest``, a terminal pool missing ``node_id``,
        ``graph_stage_id``, or ``priced_state_date`` (or one whose
        ``priced_state_date`` is the undated sentinel), or a terminal pool
        ``node_id`` that is the ``-1`` shared-pool sentinel — on the flattened
        variant this last case means the flattening itself did not take (a
        real error, since the whole point of training the variant is to never
        see the sentinel here). A failure inside the run itself propagates as
        ``cobre.run.run``'s own exception.
    """
    import cobre

    variant_dir = work_dir / "bootstrap_variant"
    _flatten_terminal_fan(case_dir, variant_dir)

    output_dir = work_dir / "output"
    # `simulation.enabled=False` skips the post-training simulation (the
    # bootstrap needs only the policy checkpoint's terminal manifest); cobre's
    # `run` gates simulation on the config, not a `skip_simulation` argument.
    config_overrides: dict[str, object] = {
        "training.stopping_rules": [{"type": "iteration_limit", "limit": 1}],
        "simulation.enabled": False,
    }
    cobre.run.run(
        str(variant_dir),
        output_dir=str(output_dir),
        config_overrides=config_overrides,
        on_iteration=lambda _info: True,
    )

    policy = cobre.results.load_policy(output_dir, policy_subdir="policy")
    stage_cuts = policy["stage_cuts"]
    if not stage_cuts:
        raise RuntimeError(f"checkpoint at {output_dir} has no stage cuts")

    # cobre 0.14 carries state_dimension per pool (on each stage_cuts entry),
    # not on the checkpoint metadata, so the terminal pool's own value is the
    # authoritative state-vector width — there is no separate global metadata
    # value left to cross-check it against.
    terminal = max(stage_cuts, key=lambda stage: stage["stage_id"])
    terminal_state_dimension = int(terminal["state_dimension"])

    entity_manifest = tuple(terminal["entity_manifest"])
    if not entity_manifest:
        raise RuntimeError(
            f"checkpoint at {output_dir} has an empty terminal entity_manifest"
        )

    if "node_id" not in terminal:
        raise RuntimeError(f"checkpoint at {output_dir} terminal pool has no node_id")
    node_id = int(terminal["node_id"])
    if node_id == -1:
        raise RuntimeError(
            f"checkpoint at {output_dir} terminal pool node_id is -1, the "
            "shared-pool sentinel (no real single node)"
        )

    if "graph_stage_id" not in terminal:
        raise RuntimeError(
            f"checkpoint at {output_dir} terminal pool has no graph_stage_id"
        )
    graph_stage_id = int(terminal["graph_stage_id"])

    if "priced_state_date" not in terminal:
        raise RuntimeError(
            f"checkpoint at {output_dir} terminal pool has no priced_state_date"
        )
    priced_state_date = int(terminal["priced_state_date"])
    if priced_state_date == _PRICED_STATE_DATE_SENTINEL:
        raise RuntimeError(
            f"checkpoint at {output_dir} terminal pool priced_state_date is the "
            "undated sentinel (no priced instant); the boundary loader selects a "
            "source pool by date, so an undated pool cannot be authored"
        )

    # Study-global season/PAR-order descriptor, carried into the authored
    # boundary so cobre's season-compatibility gate accepts it. A missing key
    # (never an absent-but-present descriptor) means the installed cobre
    # predates the season-manifest round-trip — the capability probe rejects
    # that up front, so this is a defensive contract check.
    metadata = policy["metadata"]
    if "season_manifest" not in metadata:
        raise RuntimeError(
            f"checkpoint at {output_dir} metadata carries no season_manifest "
            "(the installed cobre predates the season-manifest round-trip)"
        )
    season_manifest = dict(metadata["season_manifest"])

    return TerminalManifest(
        entity_manifest=entity_manifest,
        state_dimension=terminal_state_dimension,
        node_id=node_id,
        graph_stage_id=graph_stage_id,
        priced_state_date=priced_state_date,
        season_manifest=season_manifest,
    )
