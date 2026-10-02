"""ECO CSVs: the Novomodelo cuts exactly as read, before any transformation.

This is the pre-transformation audit artifact. No division by 1000, no sign
flip, no axis remapping, no reordering: one row per affine piece, one column
per coefficient in `entity_manifest` order. `is_active` and
`in_active_cut_indices` are recorded as columns and never used as filters (P10).
"""

from __future__ import annotations

import csv
import gzip
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from novomodelo_bridge.dessem.exporters.fcf.reader import (
        EntitySlotRecord,
        StageCutPool,
    )

_LOG = logging.getLogger(__name__)

FIXED_COLUMNS = (
    "pool_stage_id",
    "node_id",
    "piece_id",
    "slot_index",
    "iteration",
    "forward_pass_index",
    "is_active",
    "in_active_cut_indices",
    "intercept",
)

# Entity type codes for label generation
ENTITY_TYPE_HYDRO_STORAGE = 0
ENTITY_TYPE_HYDRO_INFLOW_LAG = 1
ENTITY_TYPE_ANTICIPATED_THERMAL = 2
ENTITY_TYPE_HYDRO_TRANSIT_BUCKET = 3

_LABEL_FORMATS = {
    ENTITY_TYPE_HYDRO_STORAGE: "storage_h{entity_id}",
    ENTITY_TYPE_HYDRO_INFLOW_LAG: "inflow_lag_h{entity_id}_l{subindex}",
    ENTITY_TYPE_ANTICIPATED_THERMAL: "anticipated_thermal_t{entity_id}_s{subindex}",
    ENTITY_TYPE_HYDRO_TRANSIT_BUCKET: "transit_bucket_h{entity_id}_d{subindex}",
}


def slot_label(slot: EntitySlotRecord) -> str:
    """Return the canonical column name for a slot."""
    fmt = _LABEL_FORMATS.get(slot.entity_type)
    if fmt is None:
        return f"unknown_{slot.entity_type}_{slot.entity_id}_{slot.subindex}"
    return fmt.format(entity_id=slot.entity_id, subindex=slot.subindex)


def eco_csv_path(output_dir: Path, eco_subdir: str, pool_id: int) -> Path:
    """Destination of a pool's plain-CSV ECO artifact."""
    return output_dir / eco_subdir / f"eco_cuts_pool_{pool_id:03d}.csv"


def eco_csv_gzip_path(output_dir: Path, eco_subdir: str, pool_id: int) -> Path:
    """Destination of a pool's gzipped ECO artifact."""
    plain = eco_csv_path(output_dir, eco_subdir, pool_id)
    return plain.with_name(plain.name + ".gz")


def _format_float(value: float) -> str:
    """Format a float with full precision for round-trip."""
    return repr(float(value))


def _write_rows(pool: StageCutPool, handle: TextIO) -> int:
    """Write CSV rows for a pool."""
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow([*FIXED_COLUMNS, *(slot_label(slot) for slot in pool.slots)])

    active = set(pool.active_cut_indices)
    slot_count = len(pool.slots)
    for index, piece in enumerate(pool.pieces):
        coefficient_count = len(piece.coefficients)
        if coefficient_count != slot_count:
            raise ValueError(
                f"piece index {index} of pool stage_id={pool.stage_id} "
                f"carries {coefficient_count} coefficients but pool has "
                f"{slot_count} slots"
            )
        writer.writerow(
            [
                pool.stage_id,
                pool.node_id,
                piece.piece_id,
                piece.slot_index,
                piece.iteration,
                piece.forward_pass_index,
                piece.is_active,
                index in active,
                _format_float(piece.intercept),
                *(_format_float(float(v)) for v in piece.coefficients),
            ]
        )
    return len(pool.pieces)


def _log_written(pool: StageCutPool, path: Path, rows: int) -> None:
    """Log that an ECO CSV was written."""
    _LOG.info(
        "wrote ECO CSV %s: stage_id=%d node_id=%d rows=%d columns=%d",
        path,
        pool.stage_id,
        pool.node_id,
        rows,
        len(FIXED_COLUMNS) + len(pool.slots),
    )


def write_eco_csv(pool: StageCutPool, path: Path) -> int:
    """Write the pool verbatim as plain CSV, returning the number of data rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        rows = _write_rows(pool, handle)
    _log_written(pool, path, rows)
    return rows


def write_eco_csv_gzip(pool: StageCutPool, path: Path) -> int:
    """Write the pool verbatim as gzipped CSV, returning the number of data rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        rows = _write_rows(pool, handle)
    _log_written(pool, path, rows)
    return rows


def select_eco_pools(
    pool_ids: Iterable[int], terminal_pool_id: int, include_terminal: bool
) -> tuple[int, ...]:
    """The pool ids to emit, ascending, with the terminal pool excluded by default."""
    ordered = tuple(sorted(set(pool_ids)))
    if not ordered:
        raise ValueError("pool_ids must not be empty")
    if include_terminal:
        return ordered
    return tuple(p for p in ordered if p != terminal_pool_id)


def emit_eco_csvs(
    pools: Mapping[int, StageCutPool],
    declared_pool_ids: Iterable[int],
    output_dir: Path,
    eco_subdir: str,
    terminal_pool_id: int,
    include_terminal: bool,
) -> dict[int, Path]:
    """Write the selected pools' ECO artifacts, gzip for the terminal pool only.

    Selection comes from `declared_pool_ids` (the case's own pool count), not
    from the keys of `pools`.
    """
    selected = select_eco_pools(declared_pool_ids, terminal_pool_id, include_terminal)
    _LOG.info("emitting ECO CSVs for pools %s", list(selected))
    if not include_terminal:
        _LOG.info("skipping terminal pool %d for ECO CSV", terminal_pool_id)

    written: dict[int, Path] = {}
    for pool_id in selected:
        pool = pools[pool_id]
        if pool_id == terminal_pool_id:
            _LOG.warning(
                "writing terminal pool ECO CSV for pool %d: %d pieces x %d "
                "coefficients, gzipped",
                pool_id,
                len(pool.pieces),
                len(pool.slots),
            )
            path = eco_csv_gzip_path(output_dir, eco_subdir, pool_id)
            write_eco_csv_gzip(pool, path)
        else:
            path = eco_csv_path(output_dir, eco_subdir, pool_id)
            write_eco_csv(pool, path)
        written[pool_id] = path
    return written
