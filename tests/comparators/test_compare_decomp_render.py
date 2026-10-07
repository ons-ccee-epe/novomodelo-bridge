"""``compare decomp`` renders the full shared per-variable table.

Point ``compare decomp`` at the same ``print_results_summary_from_dataset``
renderer ``compare newave`` uses (``reference_label="DECOMP"``), console-
threaded and ``--quiet``-gated. Tier 1 -- pure Python, no ``novomodelo`` import, no
``example/`` deck: the dataset build is stubbed via the same
``novomodelo_bridge.comparators.decomp.results.build_decomp_dataset`` monkeypatch
seam ``TestCompareDecompCommand`` below and ``TestCompareDiagnosticsWiring``
(``tests/comparators/test_compare.py``) already use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from novomodelo_bridge.core.errors import FieldParseError
from tests.conftest import _empty_fake_dataset, _fake_dataset

if TYPE_CHECKING:
    from novomodelo_bridge.comparators.dataset import ComparisonDataset


def _two_variable_dataset() -> ComparisonDataset:
    """A hermetic dataset with two variables, built through the shared
    ``analyze.build_results_dataset`` assembly kernel (as
    ``build_decomp_dataset`` does), so ``footer_counts`` metadata and the
    per-variable ``summary`` rows the renderer reads are both populated."""
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
            novomodelo_value=110.0,
            abs_diff=10.0,
            rel_diff=0.1,
        ),
        ResultComparison(
            entity_type="hydro",
            entity_name="ITAIPU",
            newave_code=10,
            novomodelo_id=0,
            stage=0,
            variable="turbined_m3s",
            newave_value=200.0,
            novomodelo_value=180.0,
            abs_diff=20.0,
            rel_diff=0.1,
        ),
    ]
    dataset = build_results_dataset(results, PercentileData(), 1e-2)
    dataset.metadata["unmapped"] = {"hydro": [], "thermal": [], "bus": []}
    return dataset


def _invoke(argv_tail: list[str], monkeypatch: pytest.MonkeyPatch, deck: Path) -> Any:
    """Invoke ``compare decomp`` through the real Typer app via ``CliRunner``,
    with the dataset build stubbed at its public entry point."""
    from novomodelo_bridge.cli import app
    from tests.conftest import make_decomp_case

    # ``DecompCase.from_directory`` is re-invoked (for manifest hashing) after
    # ``build_decomp_dataset`` is mocked away; give it a real ``DecompFiles``
    # dataclass instead of trying to discover a deck under the fake ``deck`` dir.
    monkeypatch.setattr(
        "novomodelo_bridge.decomp.case.DecompCase.from_directory",
        classmethod(lambda cls, _dir: make_decomp_case(Path("decomp"))),
    )
    monkeypatch.setattr(
        "novomodelo_bridge.comparators.decomp.results.build_decomp_dataset",
        lambda *_args, **_kwargs: _two_variable_dataset(),
    )
    argv = ["compare", "decomp", str(deck), str(deck), *argv_tail]
    return CliRunner().invoke(app, argv)


def test_compare_decomp_renders_the_full_shared_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stdout carries the ``DECOMP``-labelled header and a table row per
    compared variable -- not just the one-line headline."""
    result = _invoke([], monkeypatch, tmp_path)

    assert result.exit_code == 0
    assert "Novomodelo vs DECOMP Results Comparison" in result.stdout
    assert "generation_mw" in result.stdout
    assert "turbined_m3s" in result.stdout


def test_compare_decomp_quiet_suppresses_header_and_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--quiet`` suppresses both the header and the per-variable table."""
    result = _invoke(["--quiet"], monkeypatch, tmp_path)

    assert result.exit_code == 0
    assert "Novomodelo vs DECOMP Results Comparison" not in result.stdout
    assert "generation_mw" not in result.stdout
    assert "turbined_m3s" not in result.stdout


def test_compare_decomp_json_emits_one_verdict_and_suppresses_the_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--json`` emits exactly one JSON verdict on stdout (the full
    table is suppressed) and the process exits 0."""
    result = _invoke(["--json"], monkeypatch, tmp_path)

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "compare decomp"
    assert "Novomodelo vs DECOMP Results Comparison" not in result.stdout


class TestCompareDecompCommand:
    """The ``compare decomp`` subcommand, with the dataset build stubbed."""

    @staticmethod
    def _invoke(
        argv: list[str],
        monkeypatch: pytest.MonkeyPatch,
        dataset: ComparisonDataset | None = None,
    ) -> Any:
        from typer.testing import CliRunner

        from novomodelo_bridge.cli import app
        from tests.conftest import make_decomp_case

        # ``DecompCase.from_directory`` is re-invoked (for manifest hashing)
        # after ``build_decomp_dataset`` is mocked away; give it a real
        # ``DecompFiles`` dataclass instead of trying to discover a deck under
        # the fake ``tmp_path``.
        monkeypatch.setattr(
            "novomodelo_bridge.decomp.case.DecompCase.from_directory",
            classmethod(lambda cls, _dir: make_decomp_case(Path("decomp"))),
        )
        resolved_dataset = dataset if dataset is not None else _fake_dataset()
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.build_decomp_dataset",
            lambda *_args, **_kwargs: resolved_dataset,
        )
        return CliRunner().invoke(app, argv)

    def test_renders_headline_and_exits_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The legacy per-variable table and bounds table renderer
        is retired; the shared ``build_compare_verdict`` headline is the sole
        terminal summary."""
        result = self._invoke(
            ["compare", "decomp", str(tmp_path), str(tmp_path)], monkeypatch
        )
        assert result.exit_code == 0
        assert "Operation comparison" not in result.stdout
        assert "Final bounds" not in result.stdout

    def test_headline_leads_stdout_on_a_diverging_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The shared ``build_compare_verdict`` headline, from ``_fake_dataset``'s
        mismatch (1/2 within tol, worst ``turbined_m3s`` at 12% sMAPE), is the
        first line of stdout."""
        result = self._invoke(
            ["compare", "decomp", str(tmp_path), str(tmp_path)], monkeypatch
        )
        assert result.exit_code == 0
        lines = result.stdout.splitlines()
        assert lines[0] == "⚠ 1/2 variables within tol — worst: turbined_m3s sMAPE 12%"

    def test_headline_leads_stdout_when_all_within_tol(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        result = self._invoke(
            ["compare", "decomp", str(tmp_path), str(tmp_path)],
            monkeypatch,
            dataset=_fake_dataset(all_within_tol=True),
        )
        assert result.exit_code == 0
        lines = result.stdout.splitlines()
        assert lines[0] == "✓ 2/2 variables within tol"

    def test_json_carries_the_summary_and_unmapped_codes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``_fake_dataset``'s ``turbined_m3s`` row (``within_tol_rate=0.0``)
        diverges while ``generation_mw`` (``within_tol_rate=1.0``) does not, so
        the default fixture reports a mismatch."""
        result = self._invoke(
            ["compare", "decomp", str(tmp_path), str(tmp_path), "--json"], monkeypatch
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert list(payload.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert payload["command"] == "compare decomp"
        assert payload["status"] == "mismatch"
        summary = payload["summary"]
        assert list(summary.keys()) == [
            "within_tol",
            "total",
            "worst_variable",
            "worst_smape",
            "all_within_tol",
            "stages",
            "variables",
            "unmapped",
        ]
        assert summary["within_tol"] == 1
        assert summary["total"] == 2
        assert summary["worst_variable"] == "turbined_m3s"
        assert summary["worst_smape"] == pytest.approx(0.12)
        assert summary["all_within_tol"] is False
        assert summary["stages"] == 2
        assert {row["variable"] for row in summary["variables"]} == {
            "generation_mw",
            "turbined_m3s",
        }
        assert summary["unmapped"]["thermal"] == [86, 224]

    def test_json_status_is_ok_when_dataset_reports_all_within_tol(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When the dataset's ``within_tol_rate`` is 1.0 for every variable
        (the tolerance was already applied upstream when the caller built the
        dataset), the CLI reports ``ok``."""
        result = self._invoke(
            ["compare", "decomp", str(tmp_path), str(tmp_path), "--json"],
            monkeypatch,
            dataset=_fake_dataset(all_within_tol=True),
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["status"] == "ok"
        assert payload["summary"]["all_within_tol"] is True

    def test_json_reports_no_comparable_rows_when_comparison_is_empty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        result = self._invoke(
            ["compare", "decomp", str(tmp_path), str(tmp_path), "--json"],
            monkeypatch,
            dataset=_empty_fake_dataset(),
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["status"] == "no-comparable-rows"
        assert payload["summary"]["stages"] == 0
        assert payload["summary"]["variables"] == []
        assert payload["summary"]["within_tol"] == 0
        assert payload["summary"]["total"] == 0
        assert payload["summary"]["all_within_tol"] is False

    def test_unreadable_output_exits_two(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def _boom(*_args: object, **_kwargs: object) -> ComparisonDataset:
            raise FileNotFoundError("dec_oper_sist.csv not found")

        from typer.testing import CliRunner

        from novomodelo_bridge.cli import app

        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.build_decomp_dataset", _boom
        )
        result = CliRunner().invoke(
            app, ["compare", "decomp", str(tmp_path), str(tmp_path)]
        )
        assert result.exit_code == 2

    def test_field_parse_error_from_dataset_build_exits_two(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The mandatory id-map step inside ``build_decomp_dataset`` can now
        raise the typed ``FieldParseError`` (a malformed deck with no ``SB``
        register) -- the handler must still route it to the clean exit-2
        diagnostic instead of an uncaught traceback."""

        def _boom(*_args: object, **_kwargs: object) -> ComparisonDataset:
            raise FieldParseError(
                "the deck has no SB records; cannot build the id map",
                field="SB register",
            )

        from typer.testing import CliRunner

        from novomodelo_bridge.cli import app

        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.build_decomp_dataset", _boom
        )
        result = CliRunner().invoke(
            app, ["compare", "decomp", str(tmp_path), str(tmp_path)]
        )
        assert result.exit_code == 2

    def test_writes_artifacts_to_the_default_out_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        novomodelo_output_dir = tmp_path / "novomodelo"
        result = self._invoke(
            ["compare", "decomp", str(tmp_path), str(novomodelo_output_dir)],
            monkeypatch,
        )
        assert result.exit_code == 0
        artifacts = novomodelo_output_dir / "comparison_artifacts"
        assert (artifacts / "comparison.parquet").exists()
        assert (artifacts / "comparison.json").exists()

    def test_format_and_out_dir_flags_with_json_keep_stdout_pure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        novomodelo_output_dir = tmp_path / "novomodelo"
        other = tmp_path / "other"
        result = self._invoke(
            [
                "compare",
                "decomp",
                str(tmp_path),
                str(novomodelo_output_dir),
                "--format",
                "json",
                "--out-dir",
                str(other),
                "--json",
            ],
            monkeypatch,
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["command"] == "compare decomp"
        assert (other / "summary.json").exists()
        assert "Artifacts written to" not in result.stdout

    def test_format_html_writes_the_shared_multi_tab_report_labelled_decomp(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The HTML block now renders the SAME shared multi-tab
        report as ``compare newave`` (``build_comparison_report``), not the
        legacy single-page renderer, with the reference series labelled
        "DECOMP" (never "NEWAVE")."""
        out_dir = tmp_path / "artifacts"
        result = self._invoke(
            [
                "compare",
                "decomp",
                str(tmp_path),
                str(tmp_path),
                "--format",
                "html",
                "--out-dir",
                str(out_dir),
            ],
            monkeypatch,
        )
        assert result.exit_code == 0
        report_path = out_dir / "report.html"
        assert report_path.exists()
        report = report_path.read_text(encoding="utf-8")
        assert "tab-system" in report
        assert "tab-overview" in report
        assert "Storage by Bus" in report
        assert "DECOMP" in report
        assert "NEWAVE" not in report

    def test_format_html_advisory_routes_to_stderr_under_json(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "artifacts"
        result = self._invoke(
            [
                "compare",
                "decomp",
                str(tmp_path),
                str(tmp_path),
                "--format",
                "html",
                "--out-dir",
                str(out_dir),
                "--json",
            ],
            monkeypatch,
        )
        assert result.exit_code == 0
        assert (out_dir / "report.html").exists()
        json.loads(result.stdout)  # stdout carries only the JSON verdict
        assert "HTML report written to" not in result.stdout

    def test_partition_missing_output_exits_two(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Regression: NovomodeloPartitionMissingError extends
        BridgeError, a hierarchy disjoint from NovomodeloReadError (RuntimeError)
        and FileNotFoundError/ValueError. The compare decomp CLI handler
        must catch it too -- a clean ERROR line + exit 2, not an unhandled
        traceback -- mirroring the compare newave fix and this class's own
        NovomodeloReadError-analogue test above."""
        from novomodelo_bridge.core.errors import NovomodeloPartitionMissingError

        sim_dir = tmp_path / "novomodelo" / "simulation" / "hydro_bus_generation"

        def _boom(*_args: object, **_kwargs: object) -> ComparisonDataset:
            raise NovomodeloPartitionMissingError(
                f"Novomodelo output partition not found: {sim_dir}. The "
                "hydro_bus_generation partition is produced by novomodelo "
                ">= 0.13.0; this output directory may predate that novomodelo "
                "version.",
                path=str(sim_dir),
            )

        from typer.testing import CliRunner

        from novomodelo_bridge.cli import app

        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.build_decomp_dataset", _boom
        )
        result = CliRunner().invoke(
            app, ["compare", "decomp", str(tmp_path), str(tmp_path)]
        )
        # exit_code == 2 (not 1) proves this is the clean typer.Exit(code=2)
        # path, not an unhandled exception caught by CliRunner's default
        # catch_exceptions=True (which would report exit_code == 1).
        assert result.exit_code == 2
        assert "Novomodelo output partition not found" in result.stderr
        assert "hydro_bus_generation" in result.stderr
        assert "0.13.0" in result.stderr
