"""Capability probe gating boundary-FCF import on a real checkpoint round trip.

``convert decomp`` imports the deck's boundary FCF by default, which needs a
``novomodelo-python`` that can write and reload the dated self-describing policy
checkpoint format (``novomodelo-python`` is a required bridge dependency). This
module gates the ``--boundary-fcf`` path on a real write -> load round trip
rather than a version-string check: a round trip also catches a broken,
partial, or ABI-mismatched wheel that reports a satisfying version yet cannot
actually read back what it wrote. It authors a minimal synthetic checkpoint
via ``novomodelo.write_policy_checkpoint``, reloads it via
``novomodelo.results.load_policy``, and asserts the reloaded terminal pool carries
the self-describing ``cost_scale_factor``/``node_id``/``graph_stage_id`` fields
plus its ``priced_state_date`` (the date the boundary loader selects a source
against), that its ``entity_manifest`` slot carries the per-slot
``interval_start`` date field, and that the reloaded metadata carries the
``season_manifest`` descriptor (the study-global season/PAR-order gate).

Mirrors ``fcf/bootstrap.py``'s ``ensure_writer_binding`` convention of a
lazy, function-body-only ``import novomodelo`` so this module stays importable
in a novomodelo-free (tier-1) environment.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from novomodelo_bridge.decomp.fcf.bootstrap import TerminalManifest
from novomodelo_bridge.decomp.fcf.mapper import MappedCut, MappingResult
from novomodelo_bridge.decomp.fcf.writer import build_metadata, build_stage_cuts_payload

#: The probe's single synthetic terminal-manifest slot. The per-slot date
#: fields (`reference_date`/`interval_start`/`interval_end`) are deliberately
#: omitted — novomodelo's `write_policy_checkpoint` treats them as optional
#: (defaulting to the "not applicable" sentinel) — this probe only cares
#: whether the *reloaded* slot carries the key at all, never what it writes.
_PROBE_SLOT: dict[str, object] = {
    "entity_type": 0,
    "entity_id": 0,
    "subindex": 0,
    "was_active": True,
}
_PROBE_STAGE_ID = 0
#: An arbitrary real `YYYYMMDD` priced date for the synthetic pool; the probe
#: only round-trips it, never date-matches against a study.
_PROBE_PRICED_STATE_DATE = 20_260_101
_PROBE_MANIFEST = TerminalManifest(
    entity_manifest=(_PROBE_SLOT,),
    state_dimension=1,
    node_id=0,
    graph_stage_id=_PROBE_STAGE_ID,
    priced_state_date=_PROBE_PRICED_STATE_DATE,
    # Unused by the writer (season data rides on the metadata, not the manifest);
    # the absent descriptor only satisfies the required dataclass field.
    season_manifest={"cycle_code": 255, "n_seasons": 0, "hydro_orders": []},
)
_PROBE_MAPPING = MappingResult(
    cuts=(
        MappedCut(
            intercept=0.0,
            coefficients=(0.0,),
            cut_id=0,
            iteration=0,
            forward_pass_index=0,
            is_active=True,
        ),
    ),
    dropped=(),
)
_PROBE_CREATED_AT = "1970-01-01T00:00:00Z"

#: Remediation text raised on any probe failure. Kept as a module-level
#: constant so tests assert against it directly.
REMEDIATION = (
    "The boundary cost-to-go function could not be imported: the installed "
    "novomodelo package cannot write and read back the policy checkpoint format it "
    "requires. novomodelo-python is a required dependency of novomodelo-bridge — reinstall "
    "or upgrade it (for example: pip install --upgrade novomodelo-python), or "
    "reinstall novomodelo-bridge, then try again. To convert without the boundary "
    "cost-to-go function, re-run with --no-fcf."
)

#: Every exception type the CBVF round trip can fail with: `ModuleNotFoundError`
#: (novomodelo absent), `AttributeError` (missing `write_policy_checkpoint`/`results`
#: binding), `ValueError`/`OSError`/`RuntimeError` (every `novomodelo.errors.NovomodeloError`
#: leaf subclasses one of those three builtins),
#: and `TypeError`/`KeyError` from this probe's own access into a malformed
#: reloaded policy dict. Never a bare `except:`.
_PROBE_FAILURE_TYPES: tuple[type[Exception], ...] = (
    ModuleNotFoundError,
    AttributeError,
    ValueError,
    OSError,
    RuntimeError,
    TypeError,
    KeyError,
)


def ensure_boundary_fcf_capability() -> None:
    """Raise unless the installed novomodelo wheel writes+loads the checkpoint format.

    Writes a minimal one-slot synthetic checkpoint into a
    :class:`tempfile.TemporaryDirectory`, reloads it via
    ``novomodelo.results.load_policy``, and requires the reloaded terminal pool to
    carry a non-``None`` ``cost_scale_factor`` and the ``node_id``,
    ``graph_stage_id`` and ``priced_state_date`` keys, the reloaded terminal
    ``entity_manifest`` slot to carry an ``interval_start`` key, and the
    reloaded metadata to carry a ``season_manifest`` — the dated
    self-describing schema an older wheel lacks. Leaves no artifacts on disk.

    Raises
    ------
    RuntimeError
        Carrying :data:`REMEDIATION` — a self-contained, end-user-facing
        message (the novomodelo-python install/upgrade fix plus the ``--no-fcf``
        escape hatch, with no repo-internal paths) — on any failure: novomodelo
        absent, no writer binding, the write/load call itself raising, or a
        reloaded pool/slot/metadata lacking any of ``cost_scale_factor``,
        ``node_id``, ``graph_stage_id``, ``priced_state_date``,
        ``interval_start``, or ``season_manifest``.
    """
    try:
        _probe_cbvf_roundtrip()
    except _PROBE_FAILURE_TYPES as err:
        raise RuntimeError(REMEDIATION) from err


def _probe_cbvf_roundtrip() -> None:
    """Write, reload, and format-check the synthetic checkpoint.

    Raises
    ------
    RuntimeError
        If the reloaded terminal pool lacks a non-``None``
        ``cost_scale_factor`` or the ``node_id``/``graph_stage_id``/
        ``priced_state_date`` keys, if its ``entity_manifest`` slot lacks
        ``interval_start``, or if the reloaded metadata lacks
        ``season_manifest`` — caught and re-wrapped by
        :func:`ensure_boundary_fcf_capability`.
    """
    import novomodelo

    payload = build_stage_cuts_payload(
        _PROBE_MAPPING,
        _PROBE_MANIFEST,
        stage_id=_PROBE_STAGE_ID,
        cost_scale_factor=1.0,
        node_id=0,
        graph_stage_id=_PROBE_STAGE_ID,
        priced_state_date=_PROBE_PRICED_STATE_DATE,
    )
    metadata = build_metadata(
        num_stages=1,
        cost_scale_factor=1.0,
        completed_iterations=0,
        final_lower_bound=0.0,
        max_iterations=0,
        forward_passes=0,
        warm_start_cuts=0,
        rng_seed=0,
        created_at=_PROBE_CREATED_AT,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        boundary_dir = Path(tmp_dir) / "boundary"
        novomodelo.write_policy_checkpoint(boundary_dir, [payload], metadata)
        policy = novomodelo.results.load_policy(
            boundary_dir.parent, policy_subdir=boundary_dir.name
        )
        terminal = max(policy["stage_cuts"], key=lambda stage: stage["stage_id"])
        if terminal.get("cost_scale_factor") is None:
            raise RuntimeError(
                "reloaded terminal pool lacks a non-None cost_scale_factor"
            )
        if "node_id" not in terminal:
            raise RuntimeError("reloaded terminal pool lacks node_id")
        if "graph_stage_id" not in terminal:
            raise RuntimeError("reloaded terminal pool lacks graph_stage_id")
        if "priced_state_date" not in terminal:
            raise RuntimeError("reloaded terminal pool lacks priced_state_date")

        entity_manifest = terminal["entity_manifest"]
        if not entity_manifest or "interval_start" not in entity_manifest[0]:
            raise RuntimeError(
                "reloaded terminal entity_manifest slot lacks the "
                "interval_start date field"
            )

        if "season_manifest" not in policy.get("metadata", {}):
            raise RuntimeError(
                "reloaded checkpoint metadata lacks the season_manifest descriptor"
            )
