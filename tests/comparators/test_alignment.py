"""Branch coverage for ``comparators.newave.alignment``.

``read_reference_names`` and ``build_entity_alignment`` are exercised
directly against a synthetic ``NewaveCase`` (built via the conftest
``make_case`` helper, no file I/O) and a hand-built ``NewaveIdMap`` /
``lines_json`` list -- no new fixture files are needed for this module.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from novomodelo_bridge.comparators.newave.alignment import (
    build_entity_alignment,
    read_reference_names,
)
from novomodelo_bridge.newave.id_map import NewaveIdMap
from tests.conftest import make_case


def _confhd_usinas() -> pd.DataFrame:
    return pd.DataFrame(
        {"codigo_usina": [10, 20], "nome_usina": ["HYDRO A", "HYDRO B "]}
    )


def _conft_usinas() -> pd.DataFrame:
    return pd.DataFrame({"codigo_usina": [100], "nome_usina": ["THERM A"]})


def _custo_deficit(rows: list[tuple[int, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"codigo_submercado": c, "nome_submercado": n} for c, n in rows]
    )


class _FakeConfhd:
    def __init__(self, usinas: pd.DataFrame) -> None:
        self.usinas = usinas


class _FakeConft:
    def __init__(self, usinas: pd.DataFrame) -> None:
        self.usinas = usinas


class _FakeSistema:
    def __init__(self, custo_deficit: pd.DataFrame | None) -> None:
        self.custo_deficit = custo_deficit


class _KeyErrorIdMap:
    """Duck-typed ``NewaveIdMap`` stand-in whose enumeration and lookup
    disagree, to hit ``build_entity_alignment``'s defensive KeyError-skip.

    The real ``NewaveIdMap.all_hydro_codes``/``all_thermal_codes`` are drawn
    from the exact dict ``hydro_id``/``thermal_id`` look up, so that lookup
    can never raise for a code the real class itself enumerated;
    ``build_entity_alignment`` only calls these four documented members, so a
    duck-typed double can exercise the skip branch directly.
    """

    all_hydro_codes = [10, 999]
    all_thermal_codes = [100, 888]
    all_bus_ids: list[int] = []

    def hydro_id(self, code: int) -> int:
        if code == 999:
            raise KeyError(code)
        return 0

    def thermal_id(self, code: int) -> int:
        if code == 888:
            raise KeyError(code)
        return 0

    def bus_id(self, code: int) -> int:
        raise KeyError(code)


class TestReadReferenceNames:
    def test_reads_hydro_thermal_and_deduped_subsystem_names(
        self, tmp_path: Path
    ) -> None:
        case = make_case(
            tmp_path,
            confhd=_FakeConfhd(_confhd_usinas()),
            conft=_FakeConft(_conft_usinas()),
            sistema=_FakeSistema(
                _custo_deficit([(1, "SUDESTE"), (1, "SUDESTE"), (2, "SUL")])
            ),
        )
        hydro_names, thermal_names, subsystem_names = read_reference_names(case)
        assert hydro_names == {10: "HYDRO A", 20: "HYDRO B"}
        assert thermal_names == {100: "THERM A"}
        # Two custo_deficit rows for submercado 1 (one per patamar) dedupe to
        # one subsystem-name entry.
        assert subsystem_names == {1: "SUDESTE", 2: "SUL"}

    def test_custo_deficit_none_yields_empty_subsystem_names(
        self, tmp_path: Path
    ) -> None:
        case = make_case(
            tmp_path,
            confhd=_FakeConfhd(_confhd_usinas()),
            conft=_FakeConft(_conft_usinas()),
            sistema=_FakeSistema(None),
        )
        _, _, subsystem_names = read_reference_names(case)
        assert subsystem_names == {}


class TestBuildEntityAlignment:
    def test_aligns_hydros_thermals_and_lines(self, tmp_path: Path) -> None:
        id_map = NewaveIdMap(
            subsystem_ids=[1, 2], hydro_codes=[10, 20], thermal_codes=[100]
        )
        case = make_case(
            tmp_path,
            confhd=_FakeConfhd(_confhd_usinas()),
            conft=_FakeConft(_conft_usinas()),
            sistema=_FakeSistema(_custo_deficit([(1, "SUDESTE"), (2, "SUL")])),
        )
        lines_json = [{"id": 0, "source_bus_id": 0, "target_bus_id": 1, "name": "L1"}]

        alignment = build_entity_alignment(id_map, case, lines_json)

        assert {(h.newave_code, h.novomodelo_id, h.name) for h in alignment.hydros} == {
            (10, 0, "HYDRO A"),
            (20, 1, "HYDRO B"),
        }
        assert {
            (t.newave_code, t.novomodelo_id, t.name) for t in alignment.thermals
        } == {(100, 0, "THERM A")}
        assert len(alignment.lines) == 1
        line = alignment.lines[0]
        assert line.novomodelo_line_id == 0
        assert line.newave_de == 1
        assert line.newave_para == 2

    def test_unnamed_entity_falls_back_to_code_label(self, tmp_path: Path) -> None:
        """A hydro/thermal code absent from the source model names dict still
        gets an alignment entry, labelled ``code_<n>`` rather than skipped."""
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[30], thermal_codes=[200])
        case = make_case(
            tmp_path,
            confhd=_FakeConfhd(pd.DataFrame({"codigo_usina": [], "nome_usina": []})),
            conft=_FakeConft(pd.DataFrame({"codigo_usina": [], "nome_usina": []})),
            sistema=_FakeSistema(None),
        )

        alignment = build_entity_alignment(id_map, case, lines_json=[])

        assert alignment.hydros[0].name == "code_30"
        assert alignment.thermals[0].name == "code_200"

    def test_line_with_unmapped_bus_is_skipped_and_warns(
        self, tmp_path: Path, caplog
    ) -> None:
        id_map = NewaveIdMap(subsystem_ids=[1, 2], hydro_codes=[10], thermal_codes=[])
        case = make_case(
            tmp_path,
            confhd=_FakeConfhd(_confhd_usinas()),
            conft=_FakeConft(pd.DataFrame({"codigo_usina": [], "nome_usina": []})),
            sistema=_FakeSistema(None),
        )
        lines_json = [
            {"id": 0, "source_bus_id": 0, "target_bus_id": 1, "name": "MAPPED"},
            # bus 99 is not registered in id_map -> unmapped, skipped.
            {"id": 1, "source_bus_id": 0, "target_bus_id": 99, "name": "UNMAPPED"},
        ]

        with caplog.at_level(logging.WARNING):
            alignment = build_entity_alignment(id_map, case, lines_json)

        assert [line.name for line in alignment.lines] == ["MAPPED"]
        assert "unmapped buses" in caplog.text
        assert "UNMAPPED" in caplog.text

    def test_hydro_and_thermal_keyerror_skip_branches(self, tmp_path: Path) -> None:
        id_map = _KeyErrorIdMap()
        case = make_case(
            tmp_path,
            confhd=_FakeConfhd(_confhd_usinas()),
            conft=_FakeConft(_conft_usinas()),
            sistema=_FakeSistema(None),
        )

        alignment = build_entity_alignment(id_map, case, lines_json=[])

        assert [h.newave_code for h in alignment.hydros] == [10]
        assert [t.newave_code for t in alignment.thermals] == [100]
