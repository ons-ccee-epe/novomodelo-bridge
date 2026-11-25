"""Tests for the auxiliary coupling plants' case entries (``fcf/auxiliary.py``).

Tier 1, synthetic: hand-built auxiliaries and minimal case tables.
"""

from __future__ import annotations

import pyarrow as pa

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.decomp.fcf.auxiliary import (
    auxiliary_initial_storage,
    auxiliary_observation_windows,
    auxiliary_production_models,
    coupling_auxiliary_diagnostic,
    with_auxiliary_external_inflows,
    with_auxiliary_hydros,
    with_auxiliary_inflow_stats,
    with_auxiliary_productivity,
)
from novomodelo_bridge.decomp.fcf.coupling import CouplingAuxiliary
from novomodelo_bridge.decomp.id_map import DecompIdMap

#: Plant 1 is converted; 901 and 902 are cut plants the case lacks.
_PLANT_1 = CouplingAuxiliary(
    plant_code=1,
    hydro_id=1001,
    name="ACOPL-1 A",
    plant_index=0,
    coupling_posto=11,
    operation_posto=1,
    shift_m3s=20.0,
    terminal_inflows_m3s=(25.0, 0.0),
    max_gap_m3s=20.0,
)
_CUT_901 = CouplingAuxiliary(
    plant_code=901,
    hydro_id=1901,
    name="ACOPL-901 B",
    plant_index=2,
    coupling_posto=901,
    operation_posto=None,
    shift_m3s=7.0,
    terminal_inflows_m3s=(10.0, 0.0),
    max_gap_m3s=7.0,
)
_CUT_902 = CouplingAuxiliary(
    plant_code=902,
    hydro_id=1902,
    name="ACOPL-902",
    plant_index=3,
    coupling_posto=902,
    operation_posto=None,
    shift_m3s=0.0,
    terminal_inflows_m3s=(1.0, 2.0),
    max_gap_m3s=2.0,
)

_ID_MAP = DecompIdMap(bus_codes=(1, 2), bus_names=("SE", "S"), hydro_codes=(1,))


def _hydros_doc() -> dict:
    return {"hydros": [{"id": 0, "name": "A", "unit_groups": [{"bus_id": 1}]}]}


def test_with_auxiliary_hydros_appends_storage_free_plants_on_their_buses() -> None:
    doc = _hydros_doc()

    result = with_auxiliary_hydros(
        doc,
        (_PLANT_1, _CUT_901, _CUT_902),
        id_map=_ID_MAP,
        submercados={901: 2, 902: 9},
        operational_start_date="2026-01-01",
    )

    assert doc == _hydros_doc()
    added = result["hydros"][1:]
    assert [(h["id"], h["name"]) for h in added] == [
        (1001, "ACOPL-1 A"),
        (1901, "ACOPL-901 B"),
        (1902, "ACOPL-902"),
    ]
    # Plant 1's bus; subsystem 2's bus; subsystem 9 has no bus, so the first.
    assert [h["unit_groups"][0]["bus_id"] for h in added] == [
        1,
        _ID_MAP.bus_id(2),
        0,
    ]
    for hydro in added:
        assert hydro["reservoir"] == {"min_storage_hm3": 0.0, "max_storage_hm3": 0.0}
        assert hydro["generation"]["max_turbined_m3s"] == 0.0
        assert hydro["downstream_id"] is None
        assert hydro["penalties"] == {"spillage_cost": 0.0}


def test_with_auxiliary_hydros_without_auxiliaries_is_the_same_document() -> None:
    doc = _hydros_doc()

    assert (
        with_auxiliary_hydros(
            doc, (), id_map=_ID_MAP, submercados={}, operational_start_date="x"
        )
        is doc
    )


def test_with_auxiliary_external_inflows_carry_the_planned_inflow_at_the_last_stage() -> (
    None
):
    table = pa.table(
        {
            "stage_id": pa.array([0, 1, 1], pa.int32()),
            "scenario_id": pa.array([0, 0, 1], pa.int32()),
            "hydro_id": pa.array([0, 0, 0], pa.int32()),
            "value_m3s": [5.0, 6.0, 7.0],
        }
    )

    result = with_auxiliary_external_inflows(table, (_PLANT_1,), n_stages=2)

    assert result.schema == table.schema
    assert result.to_pylist() == [
        {"stage_id": 0, "scenario_id": 0, "hydro_id": 0, "value_m3s": 5.0},
        {"stage_id": 0, "scenario_id": 0, "hydro_id": 1001, "value_m3s": 0.0},
        {"stage_id": 1, "scenario_id": 0, "hydro_id": 0, "value_m3s": 6.0},
        {"stage_id": 1, "scenario_id": 0, "hydro_id": 1001, "value_m3s": 25.0},
        {"stage_id": 1, "scenario_id": 1, "hydro_id": 0, "value_m3s": 7.0},
        {"stage_id": 1, "scenario_id": 1, "hydro_id": 1001, "value_m3s": 0.0},
    ]


def test_with_auxiliary_inflow_stats_add_an_identity_row_per_stage() -> None:
    table = pa.table(
        {
            "hydro_id": pa.array([0, 0], pa.int32()),
            "stage_id": pa.array([0, 1], pa.int32()),
            "mean_m3s": [5.0, 6.0],
            "std_m3s": [1.5, 2.0],
        }
    )

    result = with_auxiliary_inflow_stats(table, (_PLANT_1,))

    assert result.schema == table.schema
    assert result.to_pylist()[2:] == [
        {"hydro_id": 1001, "stage_id": 0, "mean_m3s": 0.0, "std_m3s": 1.0},
        {"hydro_id": 1001, "stage_id": 1, "mean_m3s": 0.0, "std_m3s": 1.0},
    ]


def test_with_auxiliary_productivity_adds_a_zero_productivity_row() -> None:
    table = pa.table(
        {
            "hydro_id": pa.array([0], pa.int32()),
            "stage_id": pa.array([None], pa.int32()),
            "equivalent_productivity_mw_per_m3s": [0.9],
            "reference_outflow_m3s": pa.array([None], pa.float64()),
            "specific_productivity_mw_per_m3s_per_m": pa.array([None], pa.float64()),
            "integrated_equivalent_productivity_mw_per_m3s": [0.9],
        }
    )

    result = with_auxiliary_productivity(table, (_PLANT_1,))

    assert result.schema == table.schema
    added = result.to_pylist()[1]
    assert added["hydro_id"] == 1001
    assert added["equivalent_productivity_mw_per_m3s"] == 0.0
    assert added["integrated_equivalent_productivity_mw_per_m3s"] == 0.0


def test_auxiliary_production_models_hold_constant_productivity_throughout() -> None:
    assert auxiliary_production_models((_PLANT_1,))[0]["stage_ranges"] == [
        {"start_stage_id": 0, "end_stage_id": None, "model": "constant_productivity"}
    ]


def test_auxiliary_initial_entries_start_empty_with_zero_observations() -> None:
    windows = [
        {"hydro_id": 0, "start_date": "2025-12-01", "end_date": "2026-01-01"},
        {"hydro_id": 1, "start_date": "2025-12-01", "end_date": "2026-01-01"},
        {"hydro_id": 0, "start_date": "2025-11-01", "end_date": "2025-12-01"},
    ]

    assert auxiliary_initial_storage((_PLANT_1,)) == [
        {"hydro_id": 1001, "value_hm3": 0.0}
    ]
    assert auxiliary_observation_windows(windows, (_PLANT_1,)) == [
        {
            "hydro_id": 1001,
            "start_date": "2025-11-01",
            "end_date": "2025-12-01",
            "value_m3s": 0.0,
        },
        {
            "hydro_id": 1001,
            "start_date": "2025-12-01",
            "end_date": "2026-01-01",
            "value_m3s": 0.0,
        },
    ]


def test_coupling_auxiliary_diagnostic_lists_each_plant() -> None:
    diagnostic = coupling_auxiliary_diagnostic(
        (_PLANT_1, _CUT_901), names={1: "A", 901: "B"}
    )

    assert diagnostic.code == "boundary-fcf-coupling-auxiliary-plants"
    assert diagnostic.severity is dx.Severity.INFO
    assert diagnostic.title.endswith("(2)")
    assert diagnostic.table is not None
    assert diagnostic.table.rows == [
        [1, "A", 11, 1, 1001, 20.0],
        [901, "B", 901, "-", 1901, 7.0],
    ]
