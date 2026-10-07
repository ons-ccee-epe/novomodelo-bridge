"""Tests for the DECOMP reader-context object (``decomp/case.py``).

Tier 1 — imports no ``novomodelo``; a synthetic ``_FakeDadger`` stub stands in for
a parsed deck, and no test reads a deck under ``example/``.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

import novomodelo_bridge.decomp.case as case_module
import novomodelo_bridge.decomp.converters.fpha as fpha_module
import novomodelo_bridge.decomp.temporal as temporal_module
from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.id_map import DecompIdMap
from tests.conftest import _FakeDadger, make_decomp_case, make_decomp_files

_CASE_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "novomodelo_bridge"
    / "decomp"
    / "case.py"
)
#: Substrings that would reintroduce the pipeline<->case import cycle
#: (or a converter-module coupling) if they appeared in a module-scope
#: import of ``case.py``. TYPE_CHECKING-guarded and in-function imports are
#: exempt -- see the AST walk below, which only inspects ``tree.body``.
_BANNED_MODULE_SCOPE_SUBSTRINGS = (
    "pipeline",
    "decomp.converters.hydro",
    "decomp.temporal",
    "decomp.id_map",
    "decomp.fpha",
)


class TestRequiredReaders:
    def test_dadger_parses_once_and_caches(self, tmp_path: Path) -> None:
        sentinel = object()
        mock_read = MagicMock(return_value=sentinel)
        original_read = case_module.Dadger.read
        case_module.Dadger.read = mock_read  # type: ignore[method-assign]
        try:
            case = DecompCase(files=make_decomp_files(tmp_path))

            first = case.dadger
            second = case.dadger
        finally:
            case_module.Dadger.read = original_read  # type: ignore[method-assign]

        assert first is sentinel
        assert second is sentinel
        mock_read.assert_called_once()

    def test_id_map_and_calendar_derive_from_the_cached_dadger(
        self, tmp_path: Path
    ) -> None:
        sb = pd.DataFrame({"codigo_submercado": [1, 2], "nome_submercado": ["SE", "S"]})
        uh = pd.DataFrame({"codigo_usina": [10, 20], "volume_inicial": [50.0, None]})
        ct = pd.DataFrame({"codigo_usina": [100]})
        fake_dadger = _FakeDadger(sb=sb, uh=uh, ct=ct)

        canned_calendar = [object()]
        calendar_spy = MagicMock(return_value=canned_calendar)
        original_calendar_fn = temporal_module.operative_calendar_from_dadger
        temporal_module.operative_calendar_from_dadger = calendar_spy  # type: ignore[assignment]
        try:
            case = make_decomp_case(tmp_path, dadger=fake_dadger)

            id_map = case.id_map
            calendar = case.calendar
        finally:
            temporal_module.operative_calendar_from_dadger = original_calendar_fn  # type: ignore[assignment]

        assert isinstance(id_map, DecompIdMap)
        assert id_map.hydro_codes == (10,)
        assert id_map.thermal_codes == (100,)
        assert calendar is canned_calendar
        calendar_spy.assert_called_once_with(fake_dadger)


class TestOptionalReaders:
    def test_optional_readers_return_none_and_attempt_no_parse(
        self, tmp_path: Path
    ) -> None:
        dadgnl_read = MagicMock()
        renovaveis_read = MagicMock()
        restricoes_read = MagicMock()
        polinjus_read = MagicMock()
        originals = (
            case_module.Dadgnl.read,
            case_module.Renovaveis.read,
            case_module.Restricoes.read,
            fpha_module.read_polinjus,
        )
        case_module.Dadgnl.read = dadgnl_read  # type: ignore[method-assign]
        case_module.Renovaveis.read = renovaveis_read  # type: ignore[method-assign]
        case_module.Restricoes.read = restricoes_read  # type: ignore[method-assign]
        fpha_module.read_polinjus = polinjus_read  # type: ignore[assignment]
        try:
            # make_decomp_files defaults every optional path to None.
            case = make_decomp_case(tmp_path)

            assert case.dadgnl is None
            assert case.renovaveis is None
            assert case.polinjus is None
            assert case.libs_restricao_eletrica is None
        finally:
            (
                case_module.Dadgnl.read,
                case_module.Renovaveis.read,
                case_module.Restricoes.read,
                fpha_module.read_polinjus,
            ) = originals

        dadgnl_read.assert_not_called()
        renovaveis_read.assert_not_called()
        restricoes_read.assert_not_called()
        polinjus_read.assert_not_called()


class TestDerivedState:
    def test_start_date_returns_the_first_stage_start_date(
        self, tmp_path: Path
    ) -> None:
        class _StubStage:
            start_date = date(2026, 1, 3)

        case = make_decomp_case(tmp_path, calendar=[_StubStage()])

        assert case.start_date == date(2026, 1, 3)

    def test_vazoes_is_reachable_only_via_case_files(self, tmp_path: Path) -> None:
        case = make_decomp_case(tmp_path)

        assert not hasattr(DecompCase, "vazoes")
        assert case.files.vazoes == tmp_path / "vazoes"


class TestFromDirectory:
    def test_from_directory_wires_files_without_parsing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        known_files = make_decomp_files(tmp_path)
        monkeypatch.setattr(
            "novomodelo_bridge.decomp.case.discover_decomp_files",
            MagicMock(return_value=known_files),
        )

        case = DecompCase.from_directory(tmp_path)

        assert case.files is known_files
        assert "dadger" not in case.__dict__


class TestImportCycleGuard:
    def test_case_and_pipeline_import_together_without_cycle(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import novomodelo_bridge.decomp.case; "
                "import novomodelo_bridge.decomp.pipeline; "
                "print('ok')",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ok"

    def test_case_module_carries_no_module_scope_pipeline_or_converter_import(
        self,
    ) -> None:
        tree = ast.parse(_CASE_MODULE_PATH.read_text(encoding="utf-8"))

        module_scope_names: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                module_scope_names.append(node.module)
            elif isinstance(node, ast.Import):
                module_scope_names.extend(alias.name for alias in node.names)

        offenders = [
            name
            for name in module_scope_names
            if any(bad in name for bad in _BANNED_MODULE_SCOPE_SUBSTRINGS)
        ]
        assert offenders == []
