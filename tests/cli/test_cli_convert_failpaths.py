"""Subprocess tests for the ``convert newave`` early-exit failure paths.

Tier 1 — pure Python, imports no novomodelo. Locks in the migration of
``_run_newave_conversion``/``_run_decomp_conversion`` from ``SimpleNamespace``
to the typed ``ConvertArgs``, and the routing of the two NEWAVE early-exit
paths (source-missing, dest-not-empty) through ``_fail``: a
``--json`` run on either condition must now emit a valid error envelope
instead of empty stdout.
"""

from __future__ import annotations

import inspect
import json
import typing
from pathlib import Path

import pytest

from novomodelo_bridge.cli.args import ConvertArgs
from novomodelo_bridge.cli.convert import _run_decomp_conversion, _run_newave_conversion
from tests.cli.conftest import _run_cli_subprocess
from tests.conftest import _make_fake_newave_dir

_ZEROED_CONVERT_SUMMARY = {
    "hydros": 0,
    "thermals": 0,
    "buses": 0,
    "lines": 0,
    "stages": 0,
}


class TestConvertNewaveSourceMissingFailpath:
    def test_json_emits_one_error_envelope_with_zeroed_summary(
        self, tmp_path: Path
    ) -> None:
        dst = tmp_path / "dst"

        result = _run_cli_subprocess(
            "convert",
            "newave",
            str(tmp_path / "nonexistent"),
            str(dst),
            "--json",
        )

        assert result.returncode == 1
        document = json.loads(result.stdout)
        assert document["command"] == "convert newave"
        assert document["status"] == "error"
        assert document["summary"] == _ZEROED_CONVERT_SUMMARY
        assert len(document["diagnostics"]) == 1

    def test_without_json_renders_diagnostic_on_stderr_with_empty_stdout(
        self, tmp_path: Path
    ) -> None:
        dst = tmp_path / "dst"

        result = _run_cli_subprocess(
            "convert", "newave", str(tmp_path / "nonexistent"), str(dst)
        )

        assert result.returncode == 1
        assert result.stdout == ""
        assert "does not exist" in result.stderr


class TestConvertNewaveDestNotEmptyFailpath:
    def test_json_emits_one_error_envelope_with_use_force_substring(
        self, tmp_path: Path
    ) -> None:
        src = tmp_path / "src"
        src.mkdir()
        dst = tmp_path / "dst"
        dst.mkdir()
        (dst / "existing.txt").write_text("hello")

        result = _run_cli_subprocess("convert", "newave", str(src), str(dst), "--json")

        assert result.returncode == 1
        document = json.loads(result.stdout)
        assert document["command"] == "convert newave"
        assert document["status"] == "error"
        assert len(document["diagnostics"]) == 1
        assert "Use --force" in document["diagnostics"][0]["summary"]
        # The refusal fires before anything touches the destination.
        assert [p.name for p in dst.iterdir()] == ["existing.txt"]

    def test_without_json_renders_diagnostic_on_stderr_with_empty_stdout(
        self, tmp_path: Path
    ) -> None:
        src = tmp_path / "src"
        src.mkdir()
        dst = tmp_path / "dst"
        dst.mkdir()
        (dst / "existing.txt").write_text("hello")

        result = _run_cli_subprocess("convert", "newave", str(src), str(dst))

        assert result.returncode == 1
        assert result.stdout == ""
        assert "Use --force" in result.stderr


class TestConvertHandlerSignatures:
    """Both handlers are typed on ``ConvertArgs``, not ``SimpleNamespace``."""

    @pytest.mark.parametrize("func", [_run_newave_conversion, _run_decomp_conversion])
    def test_args_parameter_is_annotated_convert_args(
        self, func: typing.Callable[..., None]
    ) -> None:
        hints = typing.get_type_hints(func)
        assert hints["args"] is ConvertArgs

    @pytest.mark.parametrize("func", [_run_newave_conversion, _run_decomp_conversion])
    def test_does_not_reference_simple_namespace(
        self, func: typing.Callable[..., None]
    ) -> None:
        assert "SimpleNamespace" not in inspect.getsource(func)


class TestCliExitCodes:
    """Subprocess-based tests for error paths that don't require inewave I/O."""

    def test_exit_code_1_when_src_missing(self, tmp_path: Path) -> None:
        dst = tmp_path / "dst"
        result = _run_cli_subprocess(
            "convert",
            "newave",
            str(tmp_path / "nonexistent"),
            str(dst),
        )
        assert result.returncode == 1
        assert "does not exist" in result.stderr

    def test_exit_code_1_when_dst_nonempty_no_force(self, tmp_path: Path) -> None:
        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        dst.mkdir()
        (dst / "existing.txt").write_text("hello")

        result = _run_cli_subprocess("convert", "newave", str(src), str(dst))
        assert result.returncode == 1
        assert "not empty" in result.stderr

    def test_exit_code_1_when_required_file_missing(self, tmp_path: Path) -> None:
        src = _make_fake_newave_dir(tmp_path)
        (src / "hidr.dat").unlink()
        dst = tmp_path / "dst"

        result = _run_cli_subprocess("convert", "newave", str(src), str(dst))
        assert result.returncode == 1
        assert "hidr.dat" in result.stderr

    def test_convert_missing_file_renders_error_diagnostic_subprocess(
        self, tmp_path: Path
    ) -> None:
        """A real discovery failure renders an ERROR diagnostic (✖) naming the file."""
        src = _make_fake_newave_dir(tmp_path)
        (src / "hidr.dat").unlink()
        dst = tmp_path / "dst"

        result = _run_cli_subprocess("convert", "newave", str(src), str(dst))
        assert result.returncode == 1
        assert "✖" in result.stderr
        assert "hidr.dat" in result.stderr

    def test_convert_json_missing_source_emits_error_json_subprocess(
        self, tmp_path: Path
    ) -> None:
        """``--json`` on a source missing a required file emits error JSON to stdout."""
        src = _make_fake_newave_dir(tmp_path)
        (src / "hidr.dat").unlink()
        dst = tmp_path / "dst"

        result = _run_cli_subprocess("convert", "newave", str(src), str(dst), "--json")

        assert result.returncode == 1
        doc = json.loads(result.stdout)
        assert doc["schema_version"] == 1
        assert doc["command"] == "convert newave"
        assert doc["status"] == "error"
        assert doc["summary"]["hydros"] == 0
        assert doc["diagnostics"]
        # The Rich diagnostic block must not also render on stderr.
        assert "✖" not in result.stderr

    def test_convert_without_source_exits_nonzero(self) -> None:
        """``convert`` with no SOURCE must error (exit 2), not silently succeed."""
        result = _run_cli_subprocess("convert")
        assert result.returncode == 2

    def test_compare_without_source_exits_nonzero(self) -> None:
        """``compare`` with no SOURCE must error (exit 2), not silently succeed."""
        result = _run_cli_subprocess("compare")
        assert result.returncode == 2
