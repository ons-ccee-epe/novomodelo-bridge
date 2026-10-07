"""Unit tests for the ``_fail`` CLI failure helper (``novomodelo_bridge.cli.failure``).

Tier 1 — pure Python, imports no novomodelo. These tests call ``_fail`` directly to
isolate the helper's own envelope/rendering/exit-code contract from the CLI
wiring around each early-exit call site; that wiring is covered separately by
the fail-path subprocess tests (``test_cli_convert_failpaths.py``,
``test_cli_dashboard_ux.py``, ``test_cli_compare_failpaths.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from novomodelo_bridge.cli.args import CompareArgs, ConvertArgs, DashboardArgs
from novomodelo_bridge.cli.failure import _fail
from novomodelo_bridge.cli.verdict import convert_summary
from novomodelo_bridge.core.errors import SourceFileError


def _convert_args(*, json_output: bool) -> ConvertArgs:
    return ConvertArgs(
        json_output=json_output,
        verbose=0,
        log_file=None,
        no_color=True,
        quiet=False,
        src=Path("src"),
        dst=Path("dst"),
        validate=False,
        force=False,
        diagnostics_json=None,
        dry_run=False,
    )


def _dashboard_args(*, json_output: bool) -> DashboardArgs:
    return DashboardArgs(
        json_output=json_output,
        verbose=0,
        log_file=None,
        no_color=True,
        quiet=False,
        case_dir=Path("case"),
        output=None,
        open_browser=False,
    )


def _compare_args(*, json_output: bool) -> CompareArgs:
    return CompareArgs(
        json_output=json_output,
        verbose=0,
        log_file=None,
        no_color=True,
        quiet=False,
        source_dir=Path("source"),
        novomodelo_output_dir=Path("novomodelo"),
        tolerance=None,
        format=None,
        out_dir=None,
    )


def test_fail_json_emits_one_error_envelope_and_exits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--json``: exactly one envelope on stdout, status "error", exit code passed through."""
    args = _convert_args(json_output=True)
    summary = convert_summary(0, 0, 0, 0, 0)

    with pytest.raises(typer.Exit) as excinfo:
        _fail("convert newave", args, SourceFileError("x"), 1, summary=summary)

    assert excinfo.value.exit_code == 1

    captured = capsys.readouterr()
    # ``json.loads`` raises on trailing/extra data, so a clean parse of the
    # whole stdout blob already proves exactly one JSON object was written.
    document = json.loads(captured.out)
    assert document["schema_version"] == 1
    assert document["command"] == "convert newave"
    assert document["status"] == "error"
    assert len(document["diagnostics"]) == 1


def test_fail_non_json_renders_diagnostic_on_stderr_and_leaves_stdout_empty(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without ``--json``: stdout stays empty, stderr carries the diagnostic panel."""
    args = _dashboard_args(json_output=False)

    with pytest.raises(typer.Exit) as excinfo:
        _fail("dashboard", args, ValueError("no output"), 1)

    assert excinfo.value.exit_code == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() != ""


def test_fail_json_defaults_summary_to_empty_dict_with_five_envelope_keys(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With no ``summary`` passed, the envelope's ``summary`` is ``{}`` and only
    the five envelope keys are present."""
    args = _compare_args(json_output=True)

    with pytest.raises(typer.Exit) as excinfo:
        _fail("compare newave", args, ValueError("boom"), 2)

    assert excinfo.value.exit_code == 2

    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert document["summary"] == {}
    assert set(document.keys()) == {
        "schema_version",
        "command",
        "status",
        "summary",
        "diagnostics",
    }
