"""Schema probe of a Novomodelo output: which partitions and columns it holds.

Each probe reads parquet footers, never rows, and reports facts only; the caller
decides what drift means and whether to emit a diagnostic. A probe returns
``None`` for an absent subject (no such directory or file, or a simulation
directory without a parquet file) and the frozenset of its column names
otherwise. A present file or partition whose schema cannot be read raises
:class:`~novomodelo_bridge.novomodelo.readers.NovomodeloReadError`, so a missing
column is drift while an unreadable file is a failure.

A simulation entity is a hive dataset: its columns are those of its first file
plus the partition column ``scenario_id``, so a later file with another schema
surfaces only when a reader runs.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from novomodelo_bridge.novomodelo.case_io import case_dir_for
from novomodelo_bridge.novomodelo.readers import (
    NovomodeloReadError,
    scan_simulation_entity,
)

SIMULATION_ENTITIES: tuple[str, ...] = (
    "hydros",
    "thermals",
    "buses",
    "exchanges",
    "costs",
    "non_controllables",
    "hydro_bus_generation",
)

#: Each key is a path relative to the output directory, or to the case directory
#: when its value is ``True``.
PARQUET_FILES: dict[str, bool] = {
    "training/convergence.parquet": False,
    "training/dictionaries/bounds.parquet": False,
    "simulation/scenario_summary.parquet": False,
    "hydro_models/fpha_hyperplanes.parquet": False,
    "constraints/hydro_bounds.parquet": True,
    "constraints/line_bounds.parquet": True,
    "constraints/thermal_bounds.parquet": True,
    "scenarios/load_seasonal_stats.parquet": True,
    "generic_constraints/resolved_echo.parquet": False,
}


def probe_simulation_entity(output_dir: Path, entity: str) -> frozenset[str] | None:
    """Columns of ``simulation/<entity>`` under *output_dir*, ``scenario_id`` included.

    An *entity* outside :data:`SIMULATION_ENTITIES` is probed all the same.
    """
    # Checked before the scan: a directory without a parquet file makes the scan
    # raise, and it is absent, not unreadable.
    if next((output_dir / "simulation" / entity).rglob("*.parquet"), None) is None:
        return None
    lf = scan_simulation_entity(output_dir, entity)
    return None if lf is None else frozenset(lf.collect_schema().names())


def probe_parquet_file(path: Path) -> frozenset[str] | None:
    """Columns of the parquet file at *path*."""
    if not path.is_file():
        return None
    try:
        return frozenset(pl.read_parquet_schema(path))
    except Exception as exc:
        raise NovomodeloReadError(f"Failed to read the schema of {path}") from exc


def probe_output(output_dir: Path) -> dict[str, frozenset[str] | None]:
    """One probe per entry of :data:`SIMULATION_ENTITIES`, keyed
    ``simulation/<entity>``, then per key of :data:`PARQUET_FILES`.

    Propagates the first :class:`NovomodeloReadError`.
    """
    case_dir = case_dir_for(output_dir)
    return {
        **{
            f"simulation/{entity}": probe_simulation_entity(output_dir, entity)
            for entity in SIMULATION_ENTITIES
        },
        **{
            relative: probe_parquet_file(
                (case_dir if case_relative else output_dir) / relative
            )
            for relative, case_relative in PARQUET_FILES.items()
        },
    }


def is_novomodelo_output(output_dir: Path) -> bool:
    """Whether *output_dir* holds a ``training`` or ``simulation`` directory."""
    return (output_dir / "training").is_dir() or (output_dir / "simulation").is_dir()
