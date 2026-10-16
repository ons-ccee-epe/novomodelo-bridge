"""Scalar parameter declarations for ``constraints/generic_parameters.json``.

Novomodelo resolves every ``@name`` token of a ``generic_constraints.json``
expression against this parallel parameters file; an undeclared name is a
load-time error. novomodelo-bridge therefore always declares the per-hydro
``computed`` parameters a generated or handwritten constraint may reference,
for every hydro in the converted case:

- ``rho_eq_h{id}`` — point equivalent productivity, MW/(m³/s)
- ``rho_acum_h{id}`` — point accumulated (cascade) productivity, MW/(m³/s)
- ``rho_eq_scaled_h{id}`` — integrated equivalent productivity, MWh/hm³
- ``rho_acum_scaled_h{id}`` — integrated accumulated productivity, MWh/hm³
- ``earm_max_h{id}`` — maximum useful stored energy, MWh

The values are derived by novomodelo at solve time; the bridge only declares the
names. The scaled variants multiply the integrated productivity by novomodelo's
fixed ``10⁶/3600`` hm³ → (m³/s)·h factor, so ``@rho_acum_scaled_h{id} ×
volume[hm³]`` is stored energy in MWh with no dependence on stage length.
Stored-energy constraints (VminOP, RHE) use the scaled variants.

A converter may also declare ``per_stage`` parameters it computes itself (the
VminOP security-curve fraction); see :func:`build_scalar_parameters`.

See ``schemas/generic_parameters.schema.json`` in the novomodelo repo for the
on-disk shape and the valid ``computed_spec.tag`` variants.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

from novomodelo_bridge.novomodelo import schemas as novomodelo_schemas


def rho_eq_name(hydro_id: int) -> str:
    """Canonical ``@name`` for the point equivalent productivity of one hydro."""
    return f"rho_eq_h{hydro_id}"


def rho_acum_name(hydro_id: int) -> str:
    """Canonical ``@name`` for the point accumulated productivity of one hydro."""
    return f"rho_acum_h{hydro_id}"


def rho_eq_scaled_name(hydro_id: int) -> str:
    """Canonical ``@name`` for the scaled integrated equivalent productivity."""
    return f"rho_eq_scaled_h{hydro_id}"


def rho_acum_scaled_name(hydro_id: int) -> str:
    """Canonical ``@name`` for the scaled integrated accumulated productivity."""
    return f"rho_acum_scaled_h{hydro_id}"


def earm_max_name(hydro_id: int) -> str:
    """Canonical ``@name`` for the maximum useful stored energy of one hydro."""
    return f"earm_max_h{hydro_id}"


#: ``(name builder, computed_spec.tag)`` for every per-hydro parameter, in
#: declaration order.
_PER_HYDRO_COMPUTED: tuple[tuple[Callable[[int], str], str], ...] = (
    (rho_eq_name, "equivalent_productivity"),
    (rho_acum_name, "accumulated_productivity"),
    (rho_eq_scaled_name, "scaled_integrated_equivalent_productivity"),
    (rho_acum_scaled_name, "scaled_integrated_accumulated_productivity"),
    (earm_max_name, "scaled_max_stored_energy"),
)


def build_scalar_parameters(
    hydro_ids: Iterable[int],
    per_stage: Mapping[str, Sequence[float]] | None = None,
) -> dict:
    """Return a ``generic_parameters.json`` dict.

    Declares every per-hydro ``computed`` parameter for each id in
    *hydro_ids* (deduplicated, ascending), then one ``per_stage`` entry per
    *per_stage* item, in mapping order. Each *per_stage* value list holds one
    float per study stage, indexed by stage id from 0 — novomodelo rejects a list
    whose length differs from the stage count.
    """
    entries: list[dict] = []
    for hydro_id in sorted({int(h) for h in hydro_ids}):
        for name_of, tag in _PER_HYDRO_COMPUTED:
            entries.append(
                {
                    "id": len(entries),
                    "name": name_of(hydro_id),
                    "kind": "computed",
                    "computed_spec": {"tag": tag, "hydro_id": hydro_id},
                }
            )
    for name, values in (per_stage or {}).items():
        entries.append(
            {
                "id": len(entries),
                "name": name,
                "kind": "per_stage",
                "values": [
                    [stage_id, float(value)] for stage_id, value in enumerate(values)
                ],
            }
        )

    return {
        "$schema": novomodelo_schemas.schema_url_for(
            "constraints/generic_parameters.json"
        ),
        "scalar_parameters": entries,
    }
