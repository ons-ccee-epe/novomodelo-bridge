"""Public registry mapping each Cobre case output file to its ``$schema`` URL.

Every ``$schema`` value emitted by either conversion track reads
:func:`schema_url_for`, keyed by the case-relative output path, so both
tracks share one canonical source instead of duplicating per-module
constants. ``tests/cobre/test_cobre_schemas.py`` pins the registry's shape and its
16-entry coverage.
"""

from __future__ import annotations

_SCHEMA_BASE = (
    "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/schemas"
)


def _url(schema_name: str) -> str:
    return f"{_SCHEMA_BASE}/{schema_name}.schema.json"


SCHEMA_URLS: dict[str, str] = {
    "config.json": _url("config"),
    "stages.json": _url("stages"),
    "penalties.json": _url("penalties"),
    "initial_conditions.json": _url("initial_conditions"),
    "system/buses.json": _url("buses"),
    "system/lines.json": _url("lines"),
    "system/thermals.json": _url("thermals"),
    "system/hydros.json": _url("hydros"),
    "system/hydro_production_models.json": _url("production_models"),
    "system/non_controllable_sources.json": _url("non_controllable_sources"),
    "system/pumping_stations.json": _url("pumping_stations"),
    "system/energy_contracts.json": _url("energy_contracts"),
    "scenarios/load_factors.json": _url("load_factors"),
    "scenarios/non_controllable_factors.json": _url("non_controllable_factors"),
    "constraints/generic_constraints.json": _url("generic_constraints"),
    "constraints/generic_parameters.json": _url("generic_parameters"),
}


def schema_url_for(relpath: str) -> str:
    """Return the ``$schema`` URL registered for the case-relative `relpath`.

    Raises:
        KeyError: `relpath` has no registered schema URL.
    """
    return SCHEMA_URLS[relpath]
