"""``compare`` early-exit fail-paths routed through ``_fail``.

Covers the three compare failures ``_load_compare_context`` (source-model case
missing, exit 1), ``_export_compare_artifacts`` (bad ``--format``, exit 2), and
the two compare handlers' read-error ``except`` blocks (exit 2) now go
through — each emits a valid ``--json`` error envelope on stdout instead of
staying empty/stderr-only, and a plain run still renders the diagnostic to
stderr. The missing-case and read-error conditions fail before any primary
result is printed, so their plain-mode stdout is empty; the bad-``--format``
condition fails only after the compare itself already ran, so its plain-mode
stdout still carries the (untouched) results summary. Tier-1: no ``novomodelo``
import at module scope; all inputs are synthetic/mocked, never a gitignored
``example/`` deck.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    import pytest


def _invoke_main(
    argv: list[str], monkeypatch: pytest.MonkeyPatch
) -> tuple[int, str, str]:
    """Run the CLI in-process with ``sys.argv``/stdout/stderr captured."""
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


def _patch_compare_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the shared NEWAVE case/alignment/lines.json loaders."""
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
        "novomodelo_bridge.novomodelo.readers.read_novomodelo_lines", lambda _dir: []
    )


def _fake_results_dataset() -> object:
    """A one-row within-tol dataset built through the shared assembly kernel."""
    from novomodelo_bridge.comparators.analyze import build_results_dataset
    from novomodelo_bridge.comparators.model import PercentileData, ResultComparison

    results = [
        ResultComparison(
            entity_type="hydro",
            entity_name="ITAIPU",
            newave_code=10,
            novomodelo_id=0,
            stage=0,
            variable="generation_mw",
            newave_value=100.0,
            novomodelo_value=100.0,
            abs_diff=0.0,
            rel_diff=0.0,
        ),
    ]
    return build_results_dataset(results, PercentileData(), 1e-2)


class TestCompareNewaveMissingSourceCase:
    """``_load_compare_context``'s ``FileNotFoundError`` -> ``_fail(..., 1)``."""

    @staticmethod
    def _raise_missing(cls: object, _dir: Path) -> object:
        raise FileNotFoundError("caso.dat not found")

    def test_json_emits_error_envelope_exit_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "novomodelo_bridge.newave.case.NewaveCase.from_directory",
            classmethod(self._raise_missing),
        )
        novomodelo_dir = tmp_path / "novomodelo"
        novomodelo_dir.mkdir()

        code, stdout, stderr = _invoke_main(
            ["compare", "newave", str(tmp_path / "nw"), str(novomodelo_dir), "--json"],
            monkeypatch,
        )

        assert code == 1
        document = json.loads(stdout)
        assert document["command"] == "compare newave"
        assert document["status"] == "error"
        assert document["summary"] == {}
        assert "caso.dat not found" in document["diagnostics"][0]["summary"]
        assert stderr == ""

    def test_plain_empty_stdout_stderr_diagnostic_exit_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "novomodelo_bridge.newave.case.NewaveCase.from_directory",
            classmethod(self._raise_missing),
        )
        novomodelo_dir = tmp_path / "novomodelo"
        novomodelo_dir.mkdir()

        code, stdout, stderr = _invoke_main(
            ["compare", "newave", str(tmp_path / "nw"), str(novomodelo_dir)],
            monkeypatch,
        )

        assert code == 1
        assert stdout == ""
        assert "caso.dat not found" in stderr


class TestCompareNewaveBadFormat:
    """``_export_compare_artifacts``'s ``ValueError`` -> ``_fail(..., 2)``."""

    def test_json_emits_error_envelope_exit_2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_compare_context(monkeypatch)
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.newave.results.compare_results",
            lambda **k: _fake_results_dataset(),
        )
        novomodelo_dir = tmp_path / "novomodelo"
        novomodelo_dir.mkdir()

        code, stdout, stderr = _invoke_main(
            [
                "compare",
                "newave",
                str(tmp_path / "nw"),
                str(novomodelo_dir),
                "--format",
                "bogus",
                "--json",
            ],
            monkeypatch,
        )

        assert code == 2
        document = json.loads(stdout)
        assert document["command"] == "compare newave"
        assert document["status"] == "error"
        assert "bogus" in document["diagnostics"][0]["summary"]
        assert stderr == ""

    def test_plain_stderr_diagnostic_exit_2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unlike the other two conditions, ``--format`` fails AFTER the compare
        already ran, so the results summary (the untouched primary-render path)
        still reaches stdout; only the diagnostic — not the whole result — moves
        to stderr. Mirrors the pre-existing ``bogus`` stderr/exit-2 assertions."""
        _patch_compare_context(monkeypatch)
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.newave.results.compare_results",
            lambda **k: _fake_results_dataset(),
        )
        novomodelo_dir = tmp_path / "novomodelo"
        novomodelo_dir.mkdir()

        code, stdout, stderr = _invoke_main(
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
        assert "variables within tol" in stdout
        assert "bogus" in stderr


class TestCompareDecompReadError:
    """``compare decomp``'s read-error ``except`` -> ``_fail(..., 2)``."""

    @staticmethod
    def _raise_read_error(*_args: object, **_kwargs: object) -> object:
        from novomodelo_bridge.novomodelo.readers import NovomodeloReadError

        raise NovomodeloReadError("bad parquet partition")

    def test_json_emits_error_envelope_exit_2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.build_decomp_dataset",
            self._raise_read_error,
        )

        code, stdout, stderr = _invoke_main(
            ["compare", "decomp", str(tmp_path), str(tmp_path), "--json"],
            monkeypatch,
        )

        assert code == 2
        document = json.loads(stdout)
        assert document["command"] == "compare decomp"
        assert document["status"] == "error"
        assert "bad parquet partition" in document["diagnostics"][0]["summary"]
        assert stderr == ""

    def test_plain_empty_stdout_stderr_diagnostic_exit_2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.build_decomp_dataset",
            self._raise_read_error,
        )

        code, stdout, stderr = _invoke_main(
            ["compare", "decomp", str(tmp_path), str(tmp_path)],
            monkeypatch,
        )

        assert code == 2
        assert stdout == ""
        assert "bad parquet partition" in stderr


class TestCompareNewaveSuccessStillExitsZero:
    """A successful ``compare newave`` still exits 0 through the routed helpers."""

    def test_success_exit_0(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_compare_context(monkeypatch)
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.newave.results.compare_results",
            lambda **k: _fake_results_dataset(),
        )
        novomodelo_dir = tmp_path / "novomodelo"
        novomodelo_dir.mkdir()

        code, _stdout, _stderr = _invoke_main(
            ["compare", "newave", str(tmp_path / "nw"), str(novomodelo_dir)],
            monkeypatch,
        )

        assert code == 0
