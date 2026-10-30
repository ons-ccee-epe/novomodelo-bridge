"""Unit tests for the NEWAVE results comparators, on hand-built frames."""

from __future__ import annotations

import polars as pl
import pytest

from novomodelo_bridge.comparators.newave.results import _compare_thermals


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
