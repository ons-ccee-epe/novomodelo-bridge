"""Tests for the tables that translate the entity codes the LIBs electrical
cards cite and give their data terms values (tier 1, no ``novomodelo`` import)."""

from __future__ import annotations

import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.decomp.converters.contracts import Contract
from novomodelo_bridge.decomp.converters.electrical_constraints import (
    DataAliases,
    build_code_map,
    data_term_aliases,
    machine_set_table,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap


def _id_map() -> DecompIdMap:
    return DecompIdMap(
        bus_codes=(1, 2),
        bus_names=("SE", "S"),
        hydro_codes=(66, 285),
        thermal_codes=(7,),
    )


def _contract(contract_id: int, kind: str, numero: int) -> Contract:
    return Contract(
        id=contract_id,
        kind=kind,
        numero=numero,
        name=f"C{numero}",
        bus_code=1,
        stages=[],
    )


def test_code_map_lists_every_entity_of_each_table() -> None:
    code_map = build_code_map(_id_map(), {}, {11: 3}, [])

    assert code_map == {
        "hydros": {66: 0, 285: 1},
        "thermals": {7: 0},
        "ncs_pee": {11: 3},
        "submarkets": {1: 0, 2: 1, 6: 2},
        "contracts": {},
    }


def test_code_map_lists_gnl_plants_among_the_thermals() -> None:
    code_map = build_code_map(_id_map(), {86: 5, 3: 4}, {}, [])

    assert code_map["thermals"] == {3: 4, 7: 0, 86: 5}


def test_code_map_rejects_a_gnl_code_that_is_also_a_ct_code() -> None:
    with pytest.raises(ValueError, match=r"\[7\]"):
        build_code_map(_id_map(), {7: 5}, {}, [])


def test_code_map_maps_code_6_to_the_transhipment_bus() -> None:
    id_map = _id_map()

    code_map = build_code_map(id_map, {}, {}, [])

    assert code_map["submarkets"][6] == id_map.transhipment_bus_id


def test_a_declared_subsystem_6_keeps_its_code() -> None:
    id_map = DecompIdMap(bus_codes=(1, 6), bus_names=("SE", "X"))

    code_map = build_code_map(id_map, {}, {}, [])

    assert code_map["submarkets"] == {1: 0, 6: 1}


def test_code_map_contracts_with_distinct_numbers_emit_no_diagnostic() -> None:
    contracts = [_contract(0, "import", 4), _contract(1, "export", 5)]

    with dx.collect() as sink:
        code_map = build_code_map(_id_map(), {}, {}, contracts)

    assert code_map["contracts"] == {4: 0, 5: 1}
    assert sink == []


def test_code_map_omits_a_contract_number_shared_by_import_and_export() -> None:
    contracts = [
        _contract(0, "import", 4),
        _contract(1, "export", 4),
        _contract(2, "export", 5),
    ]

    with dx.collect() as sink:
        code_map = build_code_map(_id_map(), {}, {}, contracts)

    assert code_map["contracts"] == {5: 2}
    [diagnostic] = sink
    assert diagnostic.code == "decomp-libs-electrical-contract-code-ambiguous"
    assert diagnostic.severity is dx.Severity.INFO
    assert diagnostic.category == "Special constraints"
    assert diagnostic.table is not None
    assert diagnostic.table.rows == [[4]]
    assert diagnostic.remediation is not None
    text = f"{diagnostic.summary} {diagnostic.remediation}"
    for leak in ("docs/", "plans/", "~/git", "ticket-", "epic-", "_"):
        assert leak not in text


_HYDROS = [
    {"id": 0, "unit_groups": [{"id": 0}, {"id": 1}]},
    {"id": 1, "unit_groups": [{"id": 0}]},
]


def test_code_map_has_no_machine_sets_table() -> None:
    assert "machine_sets" not in build_code_map(_id_map(), {}, {}, [])


def test_machine_set_table_numbers_each_plants_unit_groups_from_one() -> None:
    assert machine_set_table(_HYDROS, _id_map()) == {66: {1: 0, 2: 1}, 285: {1: 0}}


def test_machine_set_table_is_in_plant_code_order_whatever_the_hydros_order() -> None:
    assert list(machine_set_table(_HYDROS[::-1], _id_map())) == [66, 285]


def test_machine_set_table_omits_a_plant_without_unit_groups() -> None:
    hydros = [{"id": 0, "unit_groups": []}, {"id": 1, "unit_groups": [{"id": 0}]}]

    assert machine_set_table(hydros, _id_map()) == {285: {1: 0}}


def test_machine_set_table_numbers_groups_by_id_whatever_their_order() -> None:
    hydros = [{"id": 0, "unit_groups": [{"id": 4}, {"id": 2}]}]

    assert machine_set_table(hydros, _id_map()) == {66: {1: 2, 2: 4}}


_DATA_ID_MAP = DecompIdMap(
    bus_codes=(1, 4), bus_names=("SE", "N"), hydro_codes=(), thermal_codes=()
)
_ANDE = (1, {0: [7000.0, 6500.0], 1: [7100.0, 6600.0]})
_PQ = {
    (4, "N_PCHgd"): ((26.0, 27.0), (28.0, 29.0)),
    (1, "SECO_PCHgd"): ((1.0, 2.0), (3.0, 4.0)),
}
_CARGA_ANDE = {(0, 0): 7000.0, (0, 1): 6500.0, (1, 0): 7100.0, (1, 1): 6600.0}
_PEQ_N_PCH = {(0, 0): 26.0, (0, 1): 27.0, (1, 0): 28.0, (1, 1): 29.0}


def _data_aliases() -> DataAliases:
    return data_term_aliases(_ANDE, _PQ, _DATA_ID_MAP)


def test_data_term_aliases_name_pq_series_by_submarket_name() -> None:
    aliases = _data_aliases()

    assert set(aliases) == {"carga_ande", "peq_N_PCHgd_N", "peq_SECO_PCHgd_SE"}
    assert aliases["carga_ande"] == _CARGA_ANDE
    assert aliases["peq_N_PCHgd_N"] == _PEQ_N_PCH
    assert aliases["peq_SECO_PCHgd_SE"] == {
        (0, 0): 1.0,
        (0, 1): 2.0,
        (1, 0): 3.0,
        (1, 1): 4.0,
    }


def test_data_term_aliases_carga_ande_is_named_without_the_submarket() -> None:
    assert set(data_term_aliases((4, {0: [1.0]}), {}, _DATA_ID_MAP)) == {"carga_ande"}


def test_data_term_aliases_without_ande_or_pq_are_empty() -> None:
    assert data_term_aliases(None, {}, _DATA_ID_MAP) == {}


def test_data_term_aliases_without_pq_keep_the_ande_load() -> None:
    assert data_term_aliases(_ANDE, {}, _DATA_ID_MAP) == {"carga_ande": _CARGA_ANDE}


def test_data_term_aliases_without_ande_keep_the_pq_series() -> None:
    assert set(data_term_aliases(None, _PQ, _DATA_ID_MAP)) == {
        "peq_N_PCHgd_N",
        "peq_SECO_PCHgd_SE",
    }


def test_data_term_aliases_two_pq_series_with_the_same_alias_raise() -> None:
    id_map = DecompIdMap(
        bus_codes=(2, 3), bus_names=("X", "X"), hydro_codes=(), thermal_codes=()
    )
    pq = {(2, "A_PCHgd"): ((1.0,),), (3, "A_PCHgd"): ((2.0,),)}

    with pytest.raises(ValueError, match="peq_A_PCHgd_X") as error:
        data_term_aliases(None, pq, id_map)

    assert "(2, 'A_PCHgd')" in str(error.value)
    assert "(3, 'A_PCHgd')" in str(error.value)


def test_data_term_aliases_pq_in_an_undeclared_submarket_raise() -> None:
    with pytest.raises(KeyError, match="9"):
        data_term_aliases(None, {(9, "A_PCHgd"): ((1.0,),)}, _DATA_ID_MAP)
