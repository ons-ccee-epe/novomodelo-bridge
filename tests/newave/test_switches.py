"""The ``dger.dat`` switch table (``novomodelo_bridge.newave.switches``)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from novomodelo_bridge.core.diagnostics import Severity
from novomodelo_bridge.newave.switches import (
    SWITCH_OFF_CODE,
    DgerSwitches,
    switch_off_diagnostic,
    switched_off_inputs,
)
from tests.conftest import make_nw_files

_PLAIN = {
    "agrint": "agrupamento_livre",
    "c_adic": "considera_carga_adicional",
    "ghmin": "considera_ghmin",
    "re_dat": "restricoes_eletricas",
    "restricao_eletrica": "restricoes_eletricas_especiais",
}


def _dger(**values: object) -> MagicMock:
    """A dger double whose unset switch lines read as absent (``None``)."""
    dger = MagicMock()
    for field in (
        *_PLAIN.values(),
        "desconsidera_vazao_minima",
        "restricao_turbinamento",
    ):
        setattr(dger, field, None)
    for field, value in values.items():
        setattr(dger, field, value)
    return dger


class TestFromDger:
    def test_absent_lines_count_as_on(self) -> None:
        switches = DgerSwitches.from_dger(_dger())
        for name in (*_PLAIN, "min_outflow", "turbined_max", "turbined_min"):
            switch = getattr(switches, name)
            assert switch.on, name
            assert switch.value is None

    def test_a_test_double_attribute_counts_as_absent(self) -> None:
        """A MagicMock attribute is not a number, so it reads as an absent line."""
        assert DgerSwitches.from_dger(MagicMock()).ghmin.on

    @pytest.mark.parametrize("name", sorted(_PLAIN))
    def test_plain_switch_off_at_zero_on_otherwise(self, name: str) -> None:
        field = _PLAIN[name]
        assert not getattr(DgerSwitches.from_dger(_dger(**{field: 0})), name).on
        assert getattr(DgerSwitches.from_dger(_dger(**{field: 1})), name).on
        assert getattr(DgerSwitches.from_dger(_dger(**{field: 2})), name).on

    def test_min_outflow_switch_is_inverted(self) -> None:
        assert DgerSwitches.from_dger(_dger(desconsidera_vazao_minima=0)).min_outflow.on
        off = DgerSwitches.from_dger(_dger(desconsidera_vazao_minima=1)).min_outflow
        assert not off.on
        assert off.label == "DESCONSIDERA VAZMIN"
        assert off.value == 1

    @pytest.mark.parametrize(
        ("value", "max_on", "min_on"),
        [(0, False, False), (1, True, True), (2, True, False), (3, False, True)],
    )
    def test_turbining_switch_selects_directions(
        self, value: int, max_on: bool, min_on: bool
    ) -> None:
        switches = DgerSwitches.from_dger(_dger(restricao_turbinamento=value))
        assert switches.turbined_max.on is max_on
        assert switches.turbined_min.on is min_on
        assert switches.turbined_max.ignored != switches.turbined_min.ignored


class TestSwitchOffDiagnostic:
    def test_is_info_and_names_the_switch_and_the_input(self) -> None:
        switch = DgerSwitches.from_dger(_dger(considera_ghmin=0)).ghmin
        diag = switch_off_diagnostic(switch)
        assert diag.code == SWITCH_OFF_CODE
        assert diag.severity is Severity.INFO
        assert "ghmin.dat" in diag.title
        assert "CONSIDERA GHMIN = 0" in diag.title
        assert diag.notes == ["switch: considera_ghmin"]
        assert diag.remediation is not None and "dger.dat" in diag.remediation


class TestSwitchedOffInputs:
    def test_reports_only_present_inputs_whose_switch_is_off(
        self, tmp_path: Path
    ) -> None:
        files = make_nw_files(
            tmp_path, ghmin=tmp_path / "ghmin.dat", modif=tmp_path / "modif.dat"
        )
        switches = DgerSwitches.from_dger(
            _dger(
                considera_ghmin=0,
                agrupamento_livre=0,  # agrint.dat absent: not reported
                restricao_turbinamento=3,  # modif present: TURBMAXT reported
                restricoes_eletricas_especiais=0,  # csv absent: not reported
            )
        )
        off = switched_off_inputs(files, switches)
        assert [s.ignored for s in off] == [
            "ghmin.dat",
            "TURBMAXT records of modif.dat",
        ]

    def test_min_outflow_needs_no_file_and_csv_presence_comes_from_the_index(
        self, tmp_path: Path
    ) -> None:
        files = make_nw_files(
            tmp_path, restricao_eletrica=tmp_path / "restricao-eletrica.csv"
        )
        switches = DgerSwitches.from_dger(
            _dger(desconsidera_vazao_minima=1, restricoes_eletricas_especiais=0)
        )
        off = switched_off_inputs(files, switches)
        assert [s.ignored for s in off] == [
            "restricao-eletrica.csv",
            "minimum outflow (VAZMIN, VAZMINT)",
        ]


# ---------------------------------------------------------------------------
# Pumping station (bombeamento) switch
# ---------------------------------------------------------------------------


class TestBombeamentoSwitch:
    def test_bombeamento_switch_uses_plain_pattern(self) -> None:
        """bombeamento switch follows plain() pattern: off at 0, on otherwise."""
        switches_on = DgerSwitches.from_dger(_dger(estacoes_bombeamento=1))
        switches_off = DgerSwitches.from_dger(_dger(estacoes_bombeamento=0))
        switches_absent = DgerSwitches.from_dger(_dger())

        assert switches_on.bombeamento.on is True
        assert switches_off.bombeamento.on is False
        assert switches_absent.bombeamento.on is True  # absent = on

    def test_bombeamento_switch_label_and_ignored(self) -> None:
        """bombeamento switch has correct label and ignored description."""
        switches = DgerSwitches.from_dger(_dger(estacoes_bombeamento=1))

        assert switches.bombeamento.label == "ESTACOES BOMBEAMENTO"
        assert switches.bombeamento.ignored == "pumping stations"

    def test_bombeamento_in_switched_off_inputs(self, tmp_path: Path) -> None:
        """bombeamento appears in switched_off_inputs when file present and switch off."""
        files = make_nw_files(
            tmp_path, estacao_bombeamento=tmp_path / "estacao-bombeamento.csv"
        )
        switches = DgerSwitches.from_dger(_dger(estacoes_bombeamento=0))

        off = switched_off_inputs(files, switches)

        ignored_names = [s.ignored for s in off]
        assert "pumping stations" in ignored_names

    def test_bombeamento_not_in_switched_off_when_file_absent(
        self, tmp_path: Path
    ) -> None:
        """bombeamento not reported when file is absent, even if switch is off."""
        files = make_nw_files(tmp_path)  # No pumping file
        switches = DgerSwitches.from_dger(_dger(estacoes_bombeamento=0))

        off = switched_off_inputs(files, switches)

        ignored_names = [s.ignored for s in off]
        assert "pumping stations" not in ignored_names
