"""DECOMP writer for ``constraints/generic_parameters.json``.

RHE (``HE``/``CM``) constraints are stored-energy generics of the shape
``Σ_{p∈REE} @rho_acum_scaled_h{id} * hydro_storage_final(id) >= limite``
(MWh). Every ``@name`` token must be declared in the parameters file or
novomodelo fails to load the case; novomodelo resolves the scaled integrated
accumulated productivity itself at solve time.

This module owns building and writing that declaration file for the DECOMP
case. It delegates entry-building to the model-agnostic
:func:`novomodelo_bridge.novomodelo.scalar_parameters.build_scalar_parameters` —
the same reuse posture as ``decomp/productivity.py``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from novomodelo_bridge.novomodelo.case_writer import CaseWriter
from novomodelo_bridge.novomodelo.scalar_parameters import (
    build_scalar_parameters,
    earm_max_name,  # re-exported for the RHE emitter's @-sigil expression
    rho_acum_scaled_name,  # re-exported for the RHE emitter's @-sigil expression
)

__all__ = [
    "build_decomp_scalar_parameters",
    "earm_max_name",
    "write_scalar_parameters",
    "rho_acum_scaled_name",
]

# Live path on novomodelo's feat/generic-constraint-authoring branch
# (load_scalar_parameters_json -> case_dir/constraints/generic_parameters.json).
# The rename from the old system/scalar_parameters.json landed
# there (a clean break — the old path is now rejected); the JSON
# shape is unchanged. Pending its tagged release (see MIN_NOVOMODELO_VERSION).
_SCALAR_PARAMETERS_RELPATH = Path("constraints") / "generic_parameters.json"


def build_decomp_scalar_parameters(hydro_ids: Iterable[int]) -> dict:
    """Return the ``generic_parameters.json`` dict for a DECOMP case.

    Thin delegation to
    :func:`novomodelo_bridge.novomodelo.scalar_parameters.build_scalar_parameters`
    over *hydro_ids* (novomodelo 0-based hydro ids, the declared set) — the
    entry-building logic is not reimplemented here.
    """
    return build_scalar_parameters(hydro_ids)


def write_scalar_parameters(writer: CaseWriter, params: Mapping[str, object]) -> None:
    """Write *params* to ``constraints/generic_parameters.json`` via *writer*.

    Delegates the byte format and the ``dry_run``/would-write bookkeeping to
    the shared :class:`~novomodelo_bridge.novomodelo.case_writer.CaseWriter` — this module owns
    only the relpath the DECOMP case writes it under.
    """
    writer.write_json(_SCALAR_PARAMETERS_RELPATH.as_posix(), params)
