"""Scalar parameter declarations for ``constraints/generic_parameters.json``.

Cobre added the ``@name`` sigil for `generic_constraints.json`
expressions and a parallel parameters file that declares
every parameter referenced via ``@name``. When the file is absent any
``@name`` token in a constraint expression causes a load-time error, so
cobre-bridge always emits it with the per-hydro computed parameters that
are most likely to appear in handwritten or generated constraints.

For every non-fictitious hydro in the converted case we declare two
``computed`` entries:

- ``rho_eq_h{id}`` — :math:`\\rho_{eq}` (equivalent productivity)
- ``rho_acum_h{id}`` — :math:`\\rho_{acum}` (accumulated cascade productivity)

These two are the canonical the source-model-style productivities used by every
hydro-storage / hydro-generation energy constraint we know of (VminOP, minimum cascade
energy, EARM-based bounds, ...). The values themselves are derived by cobre from VHA
geometry and ρ_esp at solve time — we only declare the names.

See ``schemas/generic_parameters.schema.json`` in the cobre repo for the
on-disk shape and the seven valid ``computed_spec.tag`` variants.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from cobre_bridge.cobre import schemas as cobre_schemas


def rho_acum_name(hydro_id: int) -> str:
    """Canonical ``@name`` for the accumulated productivity of one hydro."""
    return f"rho_acum_h{hydro_id}"


def rho_eq_name(hydro_id: int) -> str:
    """Canonical ``@name`` for the equivalent productivity of one hydro."""
    return f"rho_eq_h{hydro_id}"


def build_scalar_parameters(
    hydro_ids: Iterable[int],
    rho_acum_per_stage_overrides: Mapping[int, Sequence[float]] | None = None,
) -> dict:
    """Return a ``generic_parameters.json`` dict declaring per-hydro parameters.

    ``rho_eq_h{id}`` is always emitted as ``computed`` (the LP coefficient
    is the gen = ρ·Q point productivity that cobre derives from the VHA
    geometry).  ``rho_acum_h{id}`` is emitted as ``computed`` by default —
    but when *rho_acum_per_stage_overrides* maps the hydro to a list of
    per-stage values, the entry switches to ``kind: "per_stage"`` so that
    every constraint referencing ``@rho_acum_h{id}`` uses the supplied
    numeric values instead of cobre's cascade-summed point productivity.

    Used by the VminOP pathway to inject the source model's stored-energy (EARM)
    productivity convention — the cascade-summed integrated productivity ``ρ_esp ·
    (1/useful) · ∫_vmin^vmax h(V) dV`` — which differs from the point productivity by up
    to ~10% on plants with non-trivial head swing and is what the source model itself
    uses to evaluate VminOP constraints.
    """
    unique_ids = sorted({int(h) for h in hydro_ids})
    overrides = (
        {int(h): list(vs) for h, vs in rho_acum_per_stage_overrides.items()}
        if rho_acum_per_stage_overrides is not None
        else {}
    )

    entries: list[dict] = []
    next_id = 0
    for hydro_id in unique_ids:
        entries.append(
            {
                "id": next_id,
                "name": rho_eq_name(hydro_id),
                "kind": "computed",
                "computed_spec": {
                    "tag": "equivalent_productivity",
                    "hydro_id": hydro_id,
                },
            }
        )
        next_id += 1

        override_values = overrides.get(hydro_id)
        if override_values is not None:
            entries.append(
                {
                    "id": next_id,
                    "name": rho_acum_name(hydro_id),
                    "kind": "per_stage",
                    "values": [
                        [stage_id, float(value)]
                        for stage_id, value in enumerate(override_values)
                    ],
                }
            )
        else:
            entries.append(
                {
                    "id": next_id,
                    "name": rho_acum_name(hydro_id),
                    "kind": "computed",
                    "computed_spec": {
                        "tag": "accumulated_productivity",
                        "hydro_id": hydro_id,
                    },
                }
            )
        next_id += 1

    return {
        "$schema": cobre_schemas.schema_url_for("constraints/generic_parameters.json"),
        "scalar_parameters": entries,
    }
