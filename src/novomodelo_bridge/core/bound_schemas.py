"""Shared PyArrow schemas for bounds tables emitted by both tracks.

Schemas that appear in both NEWAVE and DECOMP output live here so neither track
imports from the other.
"""

import pyarrow as pa

PUMPING_BOUNDS_SCHEMA = pa.schema(
    [
        pa.field("pumping_station_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("block_id", pa.int32(), nullable=True),
        pa.field("min_m3s", pa.float64(), nullable=True),
        pa.field("max_m3s", pa.float64(), nullable=True),
    ]
)
