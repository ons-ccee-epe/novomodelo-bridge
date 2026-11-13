"""Tier-1 tests for the DECOMP ``penalties.json`` construction."""

from __future__ import annotations

import copy
from datetime import date
from pathlib import Path

import pandas as pd

from novomodelo_bridge.core.penalties import FlowMicroCosts
from novomodelo_bridge.decomp.converters.config import convert_penalties
from novomodelo_bridge.decomp.converters.network import (
    append_iv_se_line,
    convert_lines,
)
from novomodelo_bridge.decomp.converters.penalties import (
    flow_micro_costs,
    with_fictitious_exchange_cost,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.temporal import build_operative_calendar
from tests.conftest import make_decomp_case

_MICRO_KEYS = ("spillage_cost", "turbined_cost", "diversion_cost")


def test_default_flow_micro_costs_are_decomp_defaults() -> None:
    assert flow_micro_costs() == FlowMicroCosts(
        spillage_cost=0.00036, turbined_cost=0.000396, diversion_cost=0.00036
    )


def test_micro_tier_ignores_productivity() -> None:
    low = convert_penalties(8291.25, [0.5, 1.0])
    high = convert_penalties(8291.25, [2.0, 3.0])

    assert {k: low["hydro"][k] for k in _MICRO_KEYS} == {
        k: high["hydro"][k] for k in _MICRO_KEYS
    }
    assert (
        low["hydro"]["outflow_violation_below_cost"]
        != high["hydro"]["outflow_violation_below_cost"]
    )
    assert low["line"] == high["line"] == {"exchange_cost": 0.0001}


def _line(line_id: int, source: int, target: int) -> dict:
    return {
        "id": line_id,
        "name": f"{source}-{target}",
        "operational_start_date": "2026-07-18",
        "source_bus_id": source,
        "target_bus_id": target,
        "capacity": {"direct_mw": 100.0, "reverse_mw": 100.0},
    }


class _LinelessDadger:
    def ia(self, df: bool = True) -> pd.DataFrame | None:
        return None


def test_post_pass_halves_only_fictitious_lines() -> None:
    lines_doc = {
        "$schema": "lines.schema.json",
        "lines": [_line(0, 0, 4), _line(1, 0, 2), _line(2, 5, 1)],
    }
    before = copy.deepcopy(lines_doc)

    result = with_fictitious_exchange_cost(lines_doc, {4, 5})

    assert lines_doc == before
    assert result["$schema"] == "lines.schema.json"
    assert [line["id"] for line in result["lines"]] == [0, 1, 2]
    assert [list(line)[-1] for line in result["lines"]] == [
        "exchange_cost",
        "capacity",
        "exchange_cost",
    ]
    assert [line.get("exchange_cost") for line in result["lines"]] == [
        5e-05,
        None,
        5e-05,
    ]


def test_post_pass_covers_the_synthesized_iv_se_line() -> None:
    id_map = DecompIdMap(bus_codes=(1, 2), bus_names=("SE", "S"))
    calendar = build_operative_calendar(date(2026, 7, 18), [[15.0, 64.0, 89.0]] * 2)
    case = make_decomp_case(Path("unused"), dadger=_LinelessDadger(), calendar=calendar)
    lines_doc, line_bounds = convert_lines(case, id_map)

    lines_doc, _ = append_iv_se_line(
        case,
        lines_doc=lines_doc,
        line_bounds=line_bounds,
        source_bus_id=id_map.transhipment_bus_id,
        target_bus_id=id_map.bus_id(1),
        capacity_mw=3700.0,
    )
    result = with_fictitious_exchange_cost(lines_doc, {id_map.transhipment_bus_id})

    assert [line["name"] for line in result["lines"]] == ["IV-SE"]
    assert result["lines"][0]["exchange_cost"] == 5e-05


def test_both_endpoints_fictitious_is_still_half() -> None:
    result = with_fictitious_exchange_cost({"lines": [_line(0, 4, 5)]}, {4, 5})

    assert result["lines"][0]["exchange_cost"] == 5e-05
