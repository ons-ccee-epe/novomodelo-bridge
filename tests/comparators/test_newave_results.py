"""Unit tests for the NEWAVE results comparators, on hand-built frames."""

from __future__ import annotations

import polars as pl
import pytest

from novomodelo_bridge.comparators.newave.results import (
    _compare_buses,
    _compare_thermals,
    _nwlistop_net_import,
)


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


def _intercambio(rows: list[tuple[int, int, int, float]]) -> pl.DataFrame:
    """NWLISTOP pair flows ``(from, to, stage, MW)``, signed from -> to."""
    frm, to, stage, value = zip(*rows, strict=True)
    return pl.DataFrame(
        {
            "from_submarket_code": list(frm),
            "to_submarket_code": list(to),
            "stage": list(stage),
            "value": list(value),
        }
    )


class TestNwlistopNetImport:
    def test_a_subsystem_imports_its_inflows_less_its_outflows(self) -> None:
        # SE (1) sends 300 MW to NE (3) and receives 100 MW from S (2); NE
        # sends 50 MW back to FC (11).
        pairs = _intercambio([(1, 3, 3, 300.0), (1, 2, 3, -100.0), (3, 11, 3, 50.0)])

        net_import = _nwlistop_net_import(pairs)

        assert net_import.rows() == [
            (1, 3, -200.0),
            (2, 3, -100.0),
            (3, 3, 250.0),
            (11, 3, 50.0),
        ]


class TestCompareBusesNetImport:
    def test_rows_take_the_medias_stage_offset_and_skip_buses_without_load(
        self,
    ) -> None:
        nw_system = pl.DataFrame(
            {
                "newave_code": [1, 1, 11, 11],
                "stage": [3, 4, 3, 4],
                "variable": ["CMO"] * 4,
                "value": [100.0, 110.0, 0.0, 0.0],
            }
        )
        nw_net_import = pl.DataFrame(
            {
                "newave_code": [1, 1, 11, 11],
                "stage": [3, 4, 3, 4],
                "value": [-200.0, -150.0, 50.0, 40.0],
            }
        )
        novomodelo_bus = pl.DataFrame(
            {
                "entity_id": [0, 0, 1, 1],
                "stage_id": [0, 1, 0, 1],
                "spot_price": [101.0, 111.0, 0.0, 0.0],
                "deficit_mw": [0.0, 0.0, 0.0, 0.0],
                "net_import_mw": [-190.0, -150.0, None, None],
            }
        )

        results = _compare_buses(
            nw_system,
            novomodelo_bus,
            {1: "SUDESTE", 11: "NOFICT1"},
            {0: {"name": "SUDESTE"}, 1: {"name": "NOFICT1"}},
            nw_net_import,
        )

        assert [
            (r.entity_name, r.stage, r.newave_value, r.novomodelo_value)
            for r in results
            if r.variable == "net_import_mw"
        ] == [("SUDESTE", 0, -200.0, -190.0), ("SUDESTE", 1, -150.0, -150.0)]
