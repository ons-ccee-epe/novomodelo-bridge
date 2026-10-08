"""Tests for the ``compare newave`` command."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestCompareDatasetWiring:
    """Compare handlers sourced from the canonical dataset.

    Patch the heavy readers (``NewaveCase``, alignment, ``compare_*``) so the real
    dataset build + ``write_artifacts`` + dataset-driven printers run without the source
    model/Novomodelo I/O.
    """

    def _invoke_main(
        self,
        argv: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[int, str, str]:
        import io

        from novomodelo_bridge import cli

        monkeypatch.setattr(sys, "argv", ["novomodelo-bridge", *argv])

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        exit_code = 0

        with patch("sys.stdout", stdout_buf), patch("sys.stderr", stderr_buf):
            try:
                cli.main()
            except SystemExit as exc:
                exit_code = int(exc.code) if exc.code is not None else 0

        return exit_code, stdout_buf.getvalue(), stderr_buf.getvalue()

    @staticmethod
    def _results() -> object:
        from novomodelo_bridge.comparators.model import ResultComparison

        return [
            ResultComparison(
                entity_type="hydro",
                entity_name="ITAIPU",
                newave_code=10,
                novomodelo_id=0,
                stage=0,
                variable="generation_mw",
                newave_value=100.0,
                novomodelo_value=110.0,
                abs_diff=10.0,
                rel_diff=0.1,
            ),
        ]

    def _patch_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from novomodelo_bridge.comparators.analyze import build_results_dataset
        from novomodelo_bridge.comparators.model import PercentileData
        from tests.conftest import make_nw_files

        # ``.files`` must be a real ``NewaveFiles`` dataclass (not a further
        # MagicMock attribute) — ``hash_input_files`` reflects over it via
        # ``dataclasses.fields``, which raises on a non-dataclass. The paths
        # need not exist: a missing file degrades to a ``None`` hash/size.
        monkeypatch.setattr(
            "novomodelo_bridge.newave.case.NewaveCase.from_directory",
            classmethod(
                lambda cls, _dir: MagicMock(
                    id_map=MagicMock(), files=make_nw_files(Path("nw"))
                )
            ),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.newave.alignment.build_entity_alignment",
            lambda *a, **k: MagicMock(),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.novomodelo.readers.read_novomodelo_lines",
            lambda _dir: [],
        )
        # ``compare_results`` now returns the canonical ``ComparisonDataset``;
        # build it from the same fixture rows so the CLI path is exercised.
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.newave.results.compare_results",
            lambda **k: build_results_dataset(self._results(), PercentileData(), 1e-2),
        )

    def test_compare_results_emits_artifacts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_results(monkeypatch)
        novomodelo_dir = tmp_path / "novomodelo"
        novomodelo_dir.mkdir()

        code, stdout, _ = self._invoke_main(
            ["compare", "newave", str(tmp_path / "nw"), str(novomodelo_dir)],
            monkeypatch,
        )

        assert code == 0
        manifest_path = novomodelo_dir / "comparison_artifacts" / "comparison.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["command"] == "compare newave"
        assert "Artifacts written to" in stdout

    def test_compare_results_artifacts_without_output_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`compare results` WITHOUT ``-o`` still writes artifacts and exits 0.

        Guards the serialization contract: the render inputs (``results`` and
        the drained ``PercentileData`` frames) live in ``dataset.render`` and
        round-trip through ``to_dir`` (invoked by ``write_artifacts``) via the
        frame-wrapping serializer, which handles their non-JSON-native values.
        """
        self._patch_results(monkeypatch)
        novomodelo_dir = tmp_path / "novomodelo"
        novomodelo_dir.mkdir()

        code, stdout, _ = self._invoke_main(
            ["compare", "newave", str(tmp_path / "nw"), str(novomodelo_dir)],
            monkeypatch,
        )

        assert code == 0
        artifacts_dir = novomodelo_dir / "comparison_artifacts"
        assert (artifacts_dir / "comparison.json").exists()
        # to_dir round-trip artifacts prove metadata serialized cleanly.
        assert (artifacts_dir / "comparison.parquet").exists()
        assert (artifacts_dir / "metadata.json").exists()
        assert "Artifacts written to" in stdout

    def test_load_compare_context_missing_source_model_exits_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing source-model case directory exits 1 with a clean stderr message.

        The shared ``_load_compare_context`` helper turns a ``FileNotFoundError``
        from the case reader into a clean exit-1 message instead of surfacing an
        uncaught traceback.
        """

        def _raise_missing(cls: object, _dir: Path) -> object:
            raise FileNotFoundError("caso.dat not found")

        monkeypatch.setattr(
            "novomodelo_bridge.newave.case.NewaveCase.from_directory",
            classmethod(_raise_missing),
        )
        novomodelo_dir = tmp_path / "novomodelo"
        novomodelo_dir.mkdir()

        code, stdout, stderr = self._invoke_main(
            ["compare", "newave", str(tmp_path / "nw"), str(novomodelo_dir)],
            monkeypatch,
        )

        assert code == 1
        assert stdout == ""
        assert "caso.dat not found" in stderr

    def test_compare_results_html_tabs_intact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from novomodelo_bridge.comparators.html_report import COMPARISON_TABS

        self._patch_results(monkeypatch)
        novomodelo_dir = tmp_path / "novomodelo"
        novomodelo_dir.mkdir()

        code, _, _ = self._invoke_main(
            [
                "compare",
                "newave",
                str(tmp_path / "nw"),
                str(novomodelo_dir),
                "--format",
                "html",
            ],
            monkeypatch,
        )

        assert code == 0
        report_path = novomodelo_dir / "comparison_artifacts" / "report.html"
        assert report_path.exists()
        html = report_path.read_text(encoding="utf-8")
        for tab_id, _label in COMPARISON_TABS:
            assert tab_id in html

    def test_compare_results_default_writes_parquet_no_html(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No ``--format``: default writes queryable artifacts, no HTML."""
        self._patch_results(monkeypatch)
        novomodelo_dir = tmp_path / "novomodelo"
        novomodelo_dir.mkdir()

        code, _, _ = self._invoke_main(
            ["compare", "newave", str(tmp_path / "nw"), str(novomodelo_dir)],
            monkeypatch,
        )

        assert code == 0
        artifacts_dir = novomodelo_dir / "comparison_artifacts"
        assert (artifacts_dir / "comparison.json").exists()
        assert (artifacts_dir / "comparison.parquet").exists()
        assert (artifacts_dir / "summary.json").exists()
        assert not (artifacts_dir / "report.html").exists()

    def test_compare_results_format_console_only_no_data(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--format console``: only the manifest is written (opt out of data)."""
        self._patch_results(monkeypatch)
        novomodelo_dir = tmp_path / "novomodelo"
        novomodelo_dir.mkdir()

        code, _, _ = self._invoke_main(
            [
                "compare",
                "newave",
                str(tmp_path / "nw"),
                str(novomodelo_dir),
                "--format",
                "console",
            ],
            monkeypatch,
        )

        assert code == 0
        artifacts_dir = novomodelo_dir / "comparison_artifacts"
        assert (artifacts_dir / "comparison.json").exists()
        assert not (artifacts_dir / "comparison.parquet").exists()

    def test_compare_results_format_parquet_json_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--format parquet,json``: queryable artifacts present, exit 0."""
        self._patch_results(monkeypatch)
        novomodelo_dir = tmp_path / "novomodelo"
        novomodelo_dir.mkdir()

        code, _, _ = self._invoke_main(
            [
                "compare",
                "newave",
                str(tmp_path / "nw"),
                str(novomodelo_dir),
                "--format",
                "parquet,json",
            ],
            monkeypatch,
        )

        assert code == 0
        artifacts_dir = novomodelo_dir / "comparison_artifacts"
        assert (artifacts_dir / "comparison.parquet").exists()
        assert (artifacts_dir / "summary.json").exists()

    def test_compare_unknown_format_exits_2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--format bogus``: stderr names the bad token, exit 2."""
        self._patch_results(monkeypatch)
        novomodelo_dir = tmp_path / "novomodelo"
        novomodelo_dir.mkdir()

        code, _, stderr = self._invoke_main(
            [
                "compare",
                "newave",
                str(tmp_path / "nw"),
                str(novomodelo_dir),
                "--format",
                "bogus",
            ],
            monkeypatch,
        )

        assert code == 2
        assert "bogus" in stderr

    def test_compare_results_help_omits_output_flag(
        self, monkeypatch: pytest.MonkeyPatch, dumb_terminal: None
    ) -> None:
        """``compare results --help`` lists --format/--out-dir, not --output/-o."""
        code, stdout, _ = self._invoke_main(
            ["compare", "newave", "--help"],
            monkeypatch,
        )

        assert code == 0
        assert "--format" in stdout
        assert "--out-dir" in stdout
        assert "--output" not in stdout
        assert "-o " not in stdout

    def test_compare_results_artifact_oserror_does_not_change_exit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_results(monkeypatch)
        novomodelo_dir = tmp_path / "novomodelo"
        novomodelo_dir.mkdir()

        def _boom(*_a: object, **_k: object) -> object:
            raise OSError("disk full")

        monkeypatch.setattr(
            "novomodelo_bridge.comparators.export.write_artifacts", _boom
        )

        code, _, stderr = self._invoke_main(
            ["compare", "newave", str(tmp_path / "nw"), str(novomodelo_dir)],
            monkeypatch,
        )

        assert code == 0
        assert "failed to write artifacts" in stderr


class TestCompareJson:
    """``compare bounds``/``compare results`` ``--json`` verdict.

    Patches the heavy readers (``NewaveCase``, alignment, ``compare_*``) so the
    real dataset build + verdict derivation run without source-model/Novomodelo I/O,
    then asserts the unified envelope on stdout, the exit-code contract, and the
    no-Rich-on-stdout property.
    """

    #: Rich glyphs / box-drawing that must NEVER appear on a ``--json`` stdout.
    _RICH_GLYPHS = ("✓", "⚠", "─", "━", "│", "┃")

    def _invoke_main(
        self,
        argv: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[int, str, str]:
        import io

        from novomodelo_bridge import cli

        monkeypatch.setattr(sys, "argv", ["novomodelo-bridge", *argv])

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        exit_code = 0

        with patch("sys.stdout", stdout_buf), patch("sys.stderr", stderr_buf):
            try:
                cli.main()
            except SystemExit as exc:
                exit_code = int(exc.code) if exc.code is not None else 0

        return exit_code, stdout_buf.getvalue(), stderr_buf.getvalue()

    @staticmethod
    def _results(*, within_tol: bool) -> object:
        """Build result rows that are fully within tol (matched) or divergent.

        With ``within_tol`` the novomodelo value equals the newave value, so the
        derived ``within_tol_rate`` is ``1.0`` and ``all_within_tol`` is True.
        Otherwise the novomodelo value diverges past the default ``1e-2`` tolerance.
        """
        from novomodelo_bridge.comparators.model import ResultComparison

        novomodelo_value = 100.0 if within_tol else 110.0
        abs_diff = 0.0 if within_tol else 10.0
        rel_diff = 0.0 if within_tol else 0.1
        return [
            ResultComparison(
                entity_type="hydro",
                entity_name="ITAIPU",
                newave_code=10,
                novomodelo_id=0,
                stage=0,
                variable="generation_mw",
                newave_value=100.0,
                novomodelo_value=novomodelo_value,
                abs_diff=abs_diff,
                rel_diff=rel_diff,
            ),
        ]

    def _patch_common(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Patch the shared context loaders (case, alignment, lines.json)."""
        from tests.conftest import make_nw_files

        # ``.files`` must be a real ``NewaveFiles`` dataclass (not a further
        # MagicMock attribute) — ``hash_input_files`` reflects over it via
        # ``dataclasses.fields``, which raises on a non-dataclass. The paths
        # need not exist: a missing file degrades to a ``None`` hash/size.
        monkeypatch.setattr(
            "novomodelo_bridge.newave.case.NewaveCase.from_directory",
            classmethod(
                lambda cls, _dir: MagicMock(
                    id_map=MagicMock(), files=make_nw_files(Path("nw"))
                )
            ),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.newave.alignment.build_entity_alignment",
            lambda *a, **k: MagicMock(),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.novomodelo.readers.read_novomodelo_lines",
            lambda _dir: [],
        )

    def _patch_results(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        within_tol: bool,
    ) -> None:
        from novomodelo_bridge.comparators.analyze import build_results_dataset
        from novomodelo_bridge.comparators.model import PercentileData

        self._patch_common(monkeypatch)
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.newave.results.compare_results",
            lambda **k: build_results_dataset(
                self._results(within_tol=within_tol), PercentileData(), 1e-2
            ),
        )

    def _assert_no_rich_stdout(self, stdout: str) -> None:
        for glyph in self._RICH_GLYPHS:
            assert glyph not in stdout
        assert "Artifacts written to" not in stdout
        assert "HTML report written to" not in stdout

    def test_compare_results_json_divergent_status_mismatch_exit_0(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Divergent results → ``status="mismatch"`` but always exit 0."""
        self._patch_results(monkeypatch, within_tol=False)
        novomodelo_dir = tmp_path / "novomodelo"
        novomodelo_dir.mkdir()

        code, stdout, _ = self._invoke_main(
            ["compare", "newave", str(tmp_path / "nw"), str(novomodelo_dir), "--json"],
            monkeypatch,
        )

        assert code == 0
        doc = json.loads(stdout)
        assert doc["command"] == "compare newave"
        assert doc["status"] == "mismatch"
        assert doc["summary"]["all_within_tol"] is False
        assert doc["summary"]["within_tol"] < doc["summary"]["total"]
        self._assert_no_rich_stdout(stdout)

    def test_compare_results_json_within_tol_status_ok_exit_0(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fully-within-tol results → ``status="ok"`` and exit 0."""
        self._patch_results(monkeypatch, within_tol=True)
        novomodelo_dir = tmp_path / "novomodelo"
        novomodelo_dir.mkdir()

        code, stdout, _ = self._invoke_main(
            ["compare", "newave", str(tmp_path / "nw"), str(novomodelo_dir), "--json"],
            monkeypatch,
        )

        assert code == 0
        doc = json.loads(stdout)
        assert doc["status"] == "ok"
        assert doc["summary"]["all_within_tol"] is True
        self._assert_no_rich_stdout(stdout)

    def test_compare_results_json_novomodelo_read_error_exit_2_no_stdout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A NovomodeloReadError exits 2 with stderr only — no stdout JSON."""
        from novomodelo_bridge.novomodelo.readers import NovomodeloReadError

        self._patch_common(monkeypatch)

        def _raise(**_k: object) -> object:
            raise NovomodeloReadError("bad parquet")

        monkeypatch.setattr(
            "novomodelo_bridge.comparators.newave.results.compare_results",
            _raise,
        )
        novomodelo_dir = tmp_path / "novomodelo"
        novomodelo_dir.mkdir()

        code, stdout, _ = self._invoke_main(
            ["compare", "newave", str(tmp_path / "nw"), str(novomodelo_dir), "--json"],
            monkeypatch,
        )

        assert code == 2
        doc = json.loads(stdout)
        assert doc["command"] == "compare newave"
        assert doc["status"] == "error"
        assert len(doc["diagnostics"]) == 1
        assert "bad parquet" in doc["diagnostics"][0]["summary"]

    def test_compare_results_partition_missing_exit_2_no_stdout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: NovomodeloPartitionMissingError (raised by
        read_novomodelo_bus_aggregates against a pre-0.13 / 0.13-incomplete
        output dir lacking simulation/hydro_bus_generation/) extends
        BridgeError -- a hierarchy disjoint from NovomodeloReadError
        (RuntimeError). The compare newave CLI handler must catch it too,
        rendering a clean ERROR line + exit 2, not an unhandled traceback.
        This drives the REAL reader against a genuinely 0.13-incomplete
        output dir (simulation/hydros/ present, simulation/
        hydro_bus_generation/ absent), so the exception message is
        production-generated, not hand-typed."""
        from novomodelo_bridge.novomodelo.readers import read_novomodelo_bus_aggregates

        self._patch_common(monkeypatch)

        novomodelo_dir = tmp_path / "novomodelo"
        (novomodelo_dir / "simulation" / "hydros").mkdir(parents=True)

        def _raise(**_k: object) -> object:
            return read_novomodelo_bus_aggregates(novomodelo_dir)

        monkeypatch.setattr(
            "novomodelo_bridge.comparators.newave.results.compare_results",
            _raise,
        )

        code, stdout, _ = self._invoke_main(
            ["compare", "newave", str(tmp_path / "nw"), str(novomodelo_dir), "--json"],
            monkeypatch,
        )

        assert code == 2
        doc = json.loads(stdout)
        assert doc["command"] == "compare newave"
        assert doc["status"] == "error"
        summary = doc["diagnostics"][0]["summary"]
        assert str(novomodelo_dir / "simulation" / "hydro_bus_generation") in summary
        assert "0.13.0" in summary


class TestCompareConfigEnvPrecedence:
    """Integration precedence tests for config/env wiring.

    Runs ``compare results`` in-process via ``cli.main``, with the heavy readers
    stubbed and ``compare_results`` / ``write_artifacts`` patched with recording
    wrappers so the resolved ``tolerance`` / ``out_dir`` / ``formats`` reaching
    them can be asserted. The cwd is an isolated tmp subdir and
    ``XDG_CONFIG_HOME`` / ``HOME`` point at empty tmp subdirs, so only the test's
    own ``novomodelo-bridge.toml`` (when written) is seen.
    """

    def _invoke_main(
        self,
        argv: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[int, str, str]:
        import io

        from novomodelo_bridge import cli

        monkeypatch.setattr(sys, "argv", ["novomodelo-bridge", *argv])

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        exit_code = 0

        with patch("sys.stdout", stdout_buf), patch("sys.stderr", stderr_buf):
            try:
                cli.main()
            except SystemExit as exc:
                exit_code = int(exc.code) if exc.code is not None else 0

        return exit_code, stdout_buf.getvalue(), stderr_buf.getvalue()

    @staticmethod
    def _isolate_config_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Chdir to an empty workdir and point config discovery at empty dirs.

        Returns the workdir (where a ``novomodelo-bridge.toml`` may be written). The
        XDG/HOME fallbacks are redirected to empty tmp subdirs so no real user
        config leaks into the resolution.
        """
        workdir = tmp_path / "work"
        workdir.mkdir()
        xdg = tmp_path / "xdg"
        xdg.mkdir()
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.chdir(workdir)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
        monkeypatch.setenv("HOME", str(home))
        return workdir

    def _stub_readers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stub the heavy source-model / Novomodelo readers shared by both commands."""
        from tests.conftest import make_nw_files

        # ``.files`` must be a real ``NewaveFiles`` dataclass (not a further
        # MagicMock attribute) — ``hash_input_files`` reflects over it via
        # ``dataclasses.fields``, which raises on a non-dataclass. The paths
        # need not exist: a missing file degrades to a ``None`` hash/size.
        monkeypatch.setattr(
            "novomodelo_bridge.newave.case.NewaveCase.from_directory",
            classmethod(
                lambda cls, _dir: MagicMock(
                    id_map=MagicMock(), files=make_nw_files(Path("nw"))
                )
            ),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.newave.alignment.build_entity_alignment",
            lambda *a, **k: MagicMock(),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.novomodelo.readers.read_novomodelo_lines",
            lambda _dir: [],
        )

    def _capture_results_tolerance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> dict[str, object]:
        """Patch ``compare_results`` with a recorder; return the captured kwargs."""
        from novomodelo_bridge.comparators.analyze import build_results_dataset
        from novomodelo_bridge.comparators.model import PercentileData

        captured: dict[str, object] = {}

        def _recorder(**kwargs: object) -> object:
            captured.update(kwargs)
            return build_results_dataset([], PercentileData(), 1e-2)

        monkeypatch.setattr(
            "novomodelo_bridge.comparators.newave.results.compare_results", _recorder
        )
        return captured

    def _capture_out_dir(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
        """Patch ``write_artifacts`` with a recorder; return the captured kwargs."""
        captured: dict[str, object] = {}

        def _recorder(*_a: object, **kwargs: object) -> None:
            captured.update(kwargs)

        monkeypatch.setattr(
            "novomodelo_bridge.comparators.export.write_artifacts", _recorder
        )
        return captured

    # -- results tolerance -------------------------------------------------

    def test_results_config_fills_tolerance(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``[compare.results]`` tolerance reaches compare_results."""
        workdir = self._isolate_config_env(tmp_path, monkeypatch)
        (workdir / "novomodelo-bridge.toml").write_text(
            "[compare.results]\ntolerance = 4e-2\n", encoding="utf-8"
        )
        self._stub_readers(monkeypatch)
        captured = self._capture_results_tolerance(monkeypatch)
        novomodelo_dir = tmp_path / "novomodelo"
        novomodelo_dir.mkdir()

        code, _, _ = self._invoke_main(
            ["compare", "newave", str(tmp_path / "nw"), str(novomodelo_dir)],
            monkeypatch,
        )

        assert code == 0
        assert captured["tolerance"] == 4e-2

    # -- out-dir from config -----------------------------------------------

    def test_out_dir_from_config_reaches_write_artifacts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``[compare] out_dir`` flows to ``write_artifacts`` as a ``Path``."""
        workdir = self._isolate_config_env(tmp_path, monkeypatch)
        (workdir / "novomodelo-bridge.toml").write_text(
            '[compare]\nout_dir = "art"\n', encoding="utf-8"
        )
        self._stub_readers(monkeypatch)
        self._capture_results_tolerance(monkeypatch)
        captured = self._capture_out_dir(monkeypatch)
        novomodelo_dir = tmp_path / "novomodelo"
        novomodelo_dir.mkdir()

        code, _, _ = self._invoke_main(
            ["compare", "newave", str(tmp_path / "nw"), str(novomodelo_dir)],
            monkeypatch,
        )

        assert code == 0
        assert captured["out_dir"] == Path("art")

    def test_out_dir_env_beats_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``NOVOMODELO_BRIDGE_OUT_DIR`` overrides a ``[compare] out_dir`` config value."""
        workdir = self._isolate_config_env(tmp_path, monkeypatch)
        (workdir / "novomodelo-bridge.toml").write_text(
            '[compare]\nout_dir = "art_cfg"\n', encoding="utf-8"
        )
        monkeypatch.setenv("NOVOMODELO_BRIDGE_OUT_DIR", "art_env")
        self._stub_readers(monkeypatch)
        self._capture_results_tolerance(monkeypatch)
        captured = self._capture_out_dir(monkeypatch)
        novomodelo_dir = tmp_path / "novomodelo"
        novomodelo_dir.mkdir()

        code, _, _ = self._invoke_main(
            ["compare", "newave", str(tmp_path / "nw"), str(novomodelo_dir)],
            monkeypatch,
        )

        assert code == 0
        assert captured["out_dir"] == Path("art_env")

    def test_format_env_beats_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``NOVOMODELO_BRIDGE_FORMAT`` overrides a ``[compare] format`` config value.

        The config asks for ``console`` only (which writes no file artifacts);
        the env asks for ``parquet``. If the env wins, ``write_artifacts`` is
        called with ``formats=["parquet"]``; if the config wrongly won, it would
        not be called at all.
        """
        workdir = self._isolate_config_env(tmp_path, monkeypatch)
        (workdir / "novomodelo-bridge.toml").write_text(
            '[compare]\nformat = ["console"]\n', encoding="utf-8"
        )
        monkeypatch.setenv("NOVOMODELO_BRIDGE_FORMAT", "parquet")
        self._stub_readers(monkeypatch)
        self._capture_results_tolerance(monkeypatch)
        captured = self._capture_out_dir(monkeypatch)
        novomodelo_dir = tmp_path / "novomodelo"
        novomodelo_dir.mkdir()

        code, _, _ = self._invoke_main(
            ["compare", "newave", str(tmp_path / "nw"), str(novomodelo_dir)],
            monkeypatch,
        )

        assert code == 0
        assert captured["formats"] == ["parquet"]

    # -- malformed config: warn-but-run -------------------------------------

    def test_malformed_config_warns_on_stderr_and_still_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A malformed config warns on stderr only and keeps the built-in default."""
        workdir = self._isolate_config_env(tmp_path, monkeypatch)
        (workdir / "novomodelo-bridge.toml").write_text(
            "this is = not valid = toml\n", encoding="utf-8"
        )
        self._stub_readers(monkeypatch)
        captured = self._capture_results_tolerance(monkeypatch)
        novomodelo_dir = tmp_path / "novomodelo"
        novomodelo_dir.mkdir()

        code, stdout, stderr = self._invoke_main(
            ["compare", "newave", str(tmp_path / "nw"), str(novomodelo_dir)],
            monkeypatch,
        )

        # Not the config-error exit; the comparison still runs at the default.
        assert code == 0
        assert captured["tolerance"] == 1e-2
        assert "novomodelo-bridge.toml" in stderr
        assert "novomodelo-bridge.toml" not in stdout
