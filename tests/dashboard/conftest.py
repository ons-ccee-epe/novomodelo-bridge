"""Case builders used only by tests/dashboard/."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def write_hydro_productivity(case_dir: Path, productivity: Mapping[int, float]) -> None:
    """Write ``system/hydro_energy_productivity.parquet`` with one stage-independent
    row per hydro id, mapped to its productivity in MW/(m³/s)."""
    path = case_dir / "system" / "hydro_energy_productivity.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "hydro_id": pa.array(list(productivity), pa.int32()),
                "stage_id": pa.array([None] * len(productivity), pa.int32()),
                "equivalent_productivity_mw_per_m3s": pa.array(
                    list(productivity.values()), pa.float64()
                ),
            }
        ),
        path,
    )
