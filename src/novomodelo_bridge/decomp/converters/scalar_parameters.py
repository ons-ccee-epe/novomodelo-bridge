"""DECOMP writer for ``constraints/generic_parameters.json``.

RHE (``HE``/``CM``) constraints are stored-energy generics of the VminOP
shape ``Σ_{p∈REE} @rho_acum_h{id} * hydro_storage(id) >= limite``.
Each ``@rho_acum_h{id}`` token is a scalar-parameter sigil that
cobre must resolve at solve time; cobre defaults it to its own *point*
accumulated productivity, but the source model evaluates stored energy with
the **integrated** accumulated productivity (the volume-integral EARM
convention), which differs by up to ~10% on plants with head swing. So the
LHS coefficient at ``@rho_acum_h{id}`` must be overridden with the same
integrated per-stage ρ_acum the RHE generic emitter uses to build the RHS, or
the LP's LHS silently drifts from the RHS.

This module owns building and writing that declaration file for the DECOMP
case. It delegates entry-building to the model-agnostic
:func:`cobre_bridge.cobre.scalar_parameters.build_scalar_parameters` —
the same reuse posture as ``decomp/productivity.py``.

Override contract (produced by the RHE generic emitter, consumed here):
``{cobre_hydro_id: [ρ_acum per stage in MWmês/hm³]}`` — one float per
operative stage, already divided by the DECOMP stage-hours energy factor
(the RHE generic emitter owns that division). A hydro absent from the
override map keeps
cobre's ``computed`` default; a hydro present switches to a ``per_stage``
entry.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from cobre_bridge.cobre.case_writer import CaseWriter
from cobre_bridge.cobre.scalar_parameters import (
    build_scalar_parameters,
    rho_acum_name,  # re-exported for the RHE emitter's @-sigil expression
)

__all__ = [
    "build_decomp_scalar_parameters",
    "write_scalar_parameters",
    "rho_acum_name",
]

# Live path on cobre's feat/generic-constraint-authoring branch
# (load_scalar_parameters_json -> case_dir/constraints/generic_parameters.json).
# The rename from the old system/scalar_parameters.json landed
# there (a clean break — the old path is now rejected); the JSON
# shape is unchanged. Pending its tagged release (see MIN_COBRE_VERSION).
_SCALAR_PARAMETERS_RELPATH = Path("constraints") / "generic_parameters.json"


def build_decomp_scalar_parameters(
    hydro_ids: Iterable[int],
    rho_acum_per_stage_overrides: Mapping[int, Sequence[float]] | None = None,
) -> dict:
    """Return the ``generic_parameters.json`` dict for a DECOMP case.

    Thin delegation to
    :func:`cobre_bridge.cobre.scalar_parameters.build_scalar_parameters`
    — *hydro_ids* and *rho_acum_per_stage_overrides* pass straight through,
    the entry-building logic is not reimplemented here.

    *hydro_ids* are cobre 0-based hydro ids (the declared set); the override
    keys are the same 0-based ids, mapping each hydro to its per-stage
    integrated ρ_acum in MWmês/hm³ (see module docstring for the full
    contract). A hydro absent from the override map keeps cobre's
    ``computed`` default.
    """
    return build_scalar_parameters(hydro_ids, rho_acum_per_stage_overrides)


def write_scalar_parameters(writer: CaseWriter, params: Mapping[str, object]) -> None:
    """Write *params* to ``constraints/generic_parameters.json`` via *writer*.

    Delegates the byte format and the ``dry_run``/would-write bookkeeping to
    the shared :class:`~cobre_bridge.cobre.case_writer.CaseWriter` — this module owns
    only the relpath the DECOMP case writes it under.
    """
    writer.write_json(_SCALAR_PARAMETERS_RELPATH.as_posix(), params)
