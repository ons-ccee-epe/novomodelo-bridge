"""Unit tests for the NEWAVE results comparators, on hand-built frames."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from novomodelo_bridge.comparators.newave.results import (
    _compare_thermals,
    _line_bounds_and_meta,
)


class TestLineBoundsAndMeta:
    def test_unparseable_lines_json_keeps_the_bounds_and_empties_meta(
        self, tmp_path: Path
    ) -> None:
        case_dir = tmp_path / "case"
        (case_dir / "system").mkdir(parents=True)
        (case_dir / "system" / "lines.json").write_text("{not json", encoding="utf-8")
        (case_dir / "constraints").mkdir()
        pq.write_table(
            pa.table(
                {
                    "line_id": pa.array([0], type=pa.int32()),
                    "stage_id": pa.array([0], type=pa.int32()),
                    "block_id": pa.array([None], type=pa.int32()),
                    "direct_mw": pa.array([1200.0]),
                    "reverse_mw": pa.array([800.0]),
                }
            ),
            case_dir / "constraints" / "line_bounds.parquet",
        )
        output_dir = case_dir / "output"
        output_dir.mkdir()

        line_bounds, line_meta = _line_bounds_and_meta(output_dir)

        assert line_meta == []
        assert line_bounds.row(0, named=True)["direct_mw"] == 1200.0


class TestCompareThermals:
    def test_compares_the_total_generation_whatever_the_row_order(self) -> None:
        nw_thermal = pl.DataFrame(
            {
                "newave_code": [1, 1, 1],
                "stage": [9, 9, 9],
                "variable": ["GTERMTOT", "GTERM", "GTMIN"],
                "value": [626.3, 116.48, 509.82],
            }
        )
        novomodelo_thermal = pl.DataFrame(
            {"entity_id": [0], "stage_id": [0], "generation_mw": [600.0]}
        )

        results = _compare_thermals(
            nw_thermal, novomodelo_thermal, {1: "ANGRA 1"}, {0: {"name": "ANGRA 1"}}
        )

        assert [(r.stage, r.newave_value) for r in results] == [
            (0, pytest.approx(626.3))
        ]
