"""Tests for the ``convert`` command family (newave and decomp)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import _make_fake_newave_dir

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_decomp_dir(tmp_path: Path) -> Path:
    """Create a minimal discoverable deck directory: ``caso.dat`` naming ``rv0``,
    plus ``dadger``/``vazoes``/``hidr`` stubs. ``discover_decomp_files`` only
    stats/globs these, never parses them."""
    decomp_dir = tmp_path / "decomp_case"
    decomp_dir.mkdir()
    (decomp_dir / "caso.dat").write_text("rv0\n")
    (decomp_dir / "dadger.rv0").write_text("stub")
    (decomp_dir / "vazoes.rv0").write_text("stub")
    (decomp_dir / "hidr.dat").write_text("stub")
    return decomp_dir


def _make_fake_decomp_dir_with_cuts(tmp_path: Path) -> Path:
    """Like :func:`_make_fake_decomp_dir`, plus stub ``cortesh``/``cortes``
    files so the real (unmocked) ``discover_decomp_files`` resolves both via
    its glob fallback — the boundary-FCF gating path only globs/stats these,
    never parses their contents."""
    decomp_dir = _make_fake_decomp_dir(tmp_path)
    (decomp_dir / "cortesh.rv0").write_text("stub")
    (decomp_dir / "cortes.rv0").write_text("stub")
    return decomp_dir


class TestCliInProcess:
    """In-process CLI tests that patch the pipeline to avoid inewave I/O."""

    def _invoke_main(
        self,
        argv: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[int, str, str]:
        """Run cli.main() in-process, capturing stdout/stderr and exit code."""
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

    def test_exit_code_0_with_force_on_nonempty_dst(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        dst.mkdir()
        (dst / "existing.txt").write_text("hello")

        fake_report = ConversionReport(
            hydro_count=1,
            thermal_count=1,
            bus_count=1,
            line_count=0,
            stage_count=12,
        )

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, stdout, _ = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--force"],
                monkeypatch,
            )

        assert code == 0
        assert "1 hydros" in stdout

    def test_stdout_contains_converted_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=10,
            thermal_count=5,
            bus_count=4,
            line_count=3,
            stage_count=60,
        )

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, stdout, _ = self._invoke_main(
                ["convert", "newave", str(src), str(dst)],
                monkeypatch,
            )

        assert code == 0
        assert "10 hydros" in stdout
        assert "5 thermals" in stdout
        assert "60 stages" in stdout

    def test_convert_failure_renders_error_diagnostic(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pipeline exception renders an ERROR diagnostic on stderr; exit 1."""
        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            side_effect=ValueError("boom"),
        ):
            code, stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst)],
                monkeypatch,
            )

        assert code == 1
        # The failure is an ERROR-styled diagnostic on stderr, not stdout.
        assert "✖" in stderr
        assert "Conversion failed" in stderr
        assert "boom" in stderr
        assert "boom" not in stdout

    def test_convert_verdict_shape(self) -> None:
        """The convert ``summary``+``status`` helpers feed the unified envelope."""
        from novomodelo_bridge.cli.verdict import (
            _convert_status,
            _convert_verdict_summary,
            build_verdict,
        )
        from novomodelo_bridge.core.conversion import ConversionReport
        from novomodelo_bridge.core.diagnostics import Diagnostic, Severity

        report = ConversionReport(
            hydro_count=10,
            thermal_count=5,
            bus_count=4,
            line_count=3,
            stage_count=60,
        )
        info = Diagnostic(
            code="some-info",
            severity=Severity.INFO,
            category="Conversion",
            title="An info",
            summary="just so",
        )

        summary = _convert_verdict_summary(report)
        status = _convert_status([info], success="ok")
        doc = build_verdict("convert newave", status, summary, [info])

        assert list(doc.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert doc["schema_version"] == 1
        assert doc["command"] == "convert newave"
        # No ERROR diagnostic → status is "ok".
        assert doc["status"] == "ok"
        assert doc["summary"] == {
            "hydros": 10,
            "thermals": 5,
            "buses": 4,
            "lines": 3,
            "stages": 60,
        }
        assert doc["diagnostics"] == [info.to_dict()]

    def test_convert_verdict_error_status_on_error_diagnostic(self) -> None:
        """Any ERROR-severity diagnostic flips ``status`` to ``"error"``."""
        from novomodelo_bridge.cli.verdict import (
            _convert_status,
            _convert_verdict_summary,
            build_verdict,
        )
        from novomodelo_bridge.core.diagnostics import Diagnostic, Severity

        error = Diagnostic(
            code="boom-code",
            severity=Severity.ERROR,
            category="Conversion failure",
            title="Conversion failed",
            summary="boom",
        )

        summary = _convert_verdict_summary(None)
        status = _convert_status([error], success="ok")
        doc = build_verdict("convert newave", status, summary, [error])

        assert doc["schema_version"] == 1
        assert doc["status"] == "error"
        # Failure path → zero counts.
        assert doc["summary"] == {
            "hydros": 0,
            "thermals": 0,
            "buses": 0,
            "lines": 0,
            "stages": 0,
        }
        assert doc["diagnostics"] == [error.to_dict()]

    def test_convert_json_success_emits_stdout_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--json`` on a successful conversion emits one JSON verdict to stdout."""
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=10,
            thermal_count=5,
            bus_count=4,
            line_count=3,
            stage_count=60,
        )

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--json"],
                monkeypatch,
            )

        assert code == 0
        doc = json.loads(stdout)
        assert list(doc.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert doc["schema_version"] == 1
        assert doc["command"] == "convert newave"
        assert doc["status"] == "ok"
        assert doc["summary"]["hydros"] == 10
        # --validate not requested → no validation sub-object.
        assert "validation" not in doc["summary"]
        # No Rich human summary leaked onto stdout.
        assert "Converted" not in stdout
        # The only thing on stderr is the always-on conversion-manifest note,
        # which is intentionally routed to stderr to keep stdout byte-clean.
        assert "Conversion manifest written" in stderr
        assert "Converted" not in stderr

    def test_convert_json_failure_emits_error_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--json`` on a pipeline failure emits ``status == "error"``; exit 1."""
        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            side_effect=ValueError("boom"),
        ):
            code, stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--json"],
                monkeypatch,
            )

        assert code == 1
        doc = json.loads(stdout)
        assert doc["schema_version"] == 1
        assert list(doc.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert doc["status"] == "error"
        assert doc["summary"]["hydros"] == 0
        # Failure path never requested validation.
        assert "validation" not in doc["summary"]
        assert len(doc["diagnostics"]) == 1
        assert doc["diagnostics"][0]["summary"] == "boom"
        # The Rich diagnostic block must not also render.
        assert "✖" not in stderr

    def test_convert_json_error_status_without_exception_exits_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An ERROR diagnostic reaching a non-raising report exits 1 under --json."""
        from novomodelo_bridge.core.conversion import ConversionReport
        from novomodelo_bridge.core.diagnostics import Diagnostic, Severity

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=10,
            thermal_count=5,
            bus_count=4,
            line_count=3,
            stage_count=60,
            diagnostics=[
                Diagnostic(
                    code="newave-some-error",
                    severity=Severity.ERROR,
                    category="Conversion",
                    title="An error",
                    summary="boom",
                )
            ],
        )

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, stdout, _stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--json"],
                monkeypatch,
            )

        assert code == 1
        doc = json.loads(stdout)
        assert doc["status"] == "error"

    def test_convert_json_coexists_with_diagnostics_json_sidecar(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--json`` and ``--diagnostics-json PATH`` both produce output."""
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        sidecar = tmp_path / "out.json"

        fake_report = ConversionReport(
            hydro_count=2,
            thermal_count=1,
            bus_count=1,
            line_count=0,
            stage_count=12,
        )

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, stdout, _ = self._invoke_main(
                [
                    "convert",
                    "newave",
                    str(src),
                    str(dst),
                    "--json",
                    "--diagnostics-json",
                    str(sidecar),
                ],
                monkeypatch,
            )

        assert code == 0
        # stdout is the JSON verdict (carries schema_version/command/status).
        verdict = json.loads(stdout)
        assert verdict["schema_version"] == 1
        assert verdict["command"] == "convert newave"
        assert verdict["status"] == "ok"
        # The sidecar was also written (its payload has no command/status keys).
        assert sidecar.exists()
        sidecar_doc = json.loads(sidecar.read_text(encoding="utf-8"))
        assert sidecar_doc["summary"]["hydros"] == 2
        assert "command" not in sidecar_doc

    def test_convert_without_json_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without ``--json`` the human ``✓ Converted ...`` summary still prints."""
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=10,
            thermal_count=5,
            bus_count=4,
            line_count=3,
            stage_count=60,
        )

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, stdout, _ = self._invoke_main(
                ["convert", "newave", str(src), str(dst)],
                monkeypatch,
            )

        assert code == 0
        assert "Converted" in stdout
        # The default path emits no JSON document.
        with pytest.raises(json.JSONDecodeError):
            json.loads(stdout)

    def test_conversion_manifest_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful conversion leaves a valid provenance manifest in dst."""
        from novomodelo_bridge.cli.conversion_manifest import ConversionManifest
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=10,
            thermal_count=5,
            bus_count=4,
            line_count=3,
            stage_count=60,
        )

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, _stdout, _stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst)],
                monkeypatch,
            )

        assert code == 0
        manifest_path = dst / "conversion_manifest.json"
        assert manifest_path.exists()
        manifest = ConversionManifest.from_json(manifest_path)
        assert manifest.entity_counts == {
            "hydros": 10,
            "thermals": 5,
            "buses": 4,
            "lines": 3,
            "stages": 60,
        }
        assert manifest.command == "convert newave"
        # The fake source dir's stub files were discovered and hashed.
        assert manifest.input_files

    def test_conversion_manifest_records_min_novomodelo_version(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The manifest's ``min_novomodelo_version`` tracks the CLI constant, pinned.

        A manifest written after the bump must record the real
        floor (``"0.18.0"``), not a stale value — the manifest is provenance,
        and a wrong floor there is false provenance. Pinning the literal (not
        just equality with the constant) catches an accidental revert of the
        constant itself.
        """
        from novomodelo_bridge.cli import MIN_NOVOMODELO_VERSION
        from novomodelo_bridge.cli.conversion_manifest import ConversionManifest
        from novomodelo_bridge.core.conversion import ConversionReport

        assert MIN_NOVOMODELO_VERSION == "0.18.0"

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=12
        )

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, _stdout, _stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst)],
                monkeypatch,
            )

        assert code == 0
        manifest = ConversionManifest.from_json(dst / "conversion_manifest.json")
        assert manifest.min_novomodelo_version == "0.18.0"
        assert manifest.min_novomodelo_version == MIN_NOVOMODELO_VERSION

    def test_manifest_not_in_json_verdict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--json`` stdout is the convert verdict; the manifest is a side file."""
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=10,
            thermal_count=5,
            bus_count=4,
            line_count=3,
            stage_count=60,
        )

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, stdout, _stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--json"],
                monkeypatch,
            )

        assert code == 0
        doc = json.loads(stdout)
        # Only the deterministic envelope keys — no manifest timestamp leaks in.
        assert list(doc.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert "timestamp" not in doc
        # The manifest still exists separately on disk.
        assert (dst / "conversion_manifest.json").exists()

    def test_convert_json_validate_folds_into_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--json --validate`` folds the validation outcome under ``summary``.

        The machine outcome lands in ``summary["validation"]`` (status stays
        diagnostics-derived), stdout is pure JSON, and a failed validation still
        exits 2.
        """
        import types

        from novomodelo_bridge.cli import MIN_NOVOMODELO_VERSION
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=10,
            thermal_count=5,
            bus_count=4,
            line_count=3,
            stage_count=60,
        )

        # Fix the installed-version gate at exactly MIN_NOVOMODELO_VERSION so this
        # test exercises the injected fake ``novomodelo.io.validate`` below
        # regardless of whichever novomodelo-python happens to be installed in the
        # dev/CI venv (which may itself now be older than MIN_NOVOMODELO_VERSION).
        monkeypatch.setattr(
            "novomodelo_bridge.novomodelo.compat._installed_novomodelo_python_version",
            lambda: MIN_NOVOMODELO_VERSION,
        )
        # Inject a fake ``novomodelo.io`` whose ``validate`` reports a failure with one
        # warning and two errors; the real ``novomodelo`` package is not installed.
        novomodelo_pkg = types.ModuleType("novomodelo")
        novomodelo_io = types.ModuleType("novomodelo.io")
        novomodelo_io.validate = lambda _dst: {  # type: ignore[attr-defined]
            "valid": False,
            "warnings": ["w"],
            "errors": ["e1", "e2"],
        }
        novomodelo_pkg.io = novomodelo_io  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "novomodelo", novomodelo_pkg)
        monkeypatch.setitem(sys.modules, "novomodelo.io", novomodelo_io)

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--json", "--validate"],
                monkeypatch,
            )

        # Validation failure flips the exit code, never the verdict status.
        assert code == 2
        doc = json.loads(stdout)
        assert list(doc.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        # status is still diagnostics-derived ("ok"), NOT flipped by validation.
        assert doc["status"] == "ok"
        assert doc["summary"]["validation"] == {
            "ran": True,
            "valid": False,
            "warnings": 1,
            "errors": 2,
        }
        # No validation text leaked onto stdout; the human messages stay on stderr.
        assert "Validation" not in stdout
        assert "Validation failed." in stderr

    def test_convert_validate_failure_precedence_over_ok_status_exits_2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``--validate`` failure exits 2 even though ``status`` stays "ok".

        Exercises the ``_gate_convert_exit`` precedence rule (validation-failure
        over error-status) on the branch where there is no error status to
        compete with — the report carries no ERROR diagnostic.
        """
        import types

        from novomodelo_bridge.cli import MIN_NOVOMODELO_VERSION
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=10,
            thermal_count=5,
            bus_count=4,
            line_count=3,
            stage_count=60,
        )

        monkeypatch.setattr(
            "novomodelo_bridge.novomodelo.compat._installed_novomodelo_python_version",
            lambda: MIN_NOVOMODELO_VERSION,
        )
        novomodelo_pkg = types.ModuleType("novomodelo")
        novomodelo_io = types.ModuleType("novomodelo.io")
        novomodelo_io.validate = lambda _dst: {  # type: ignore[attr-defined]
            "valid": False,
            "warnings": [],
            "errors": ["e1"],
        }
        novomodelo_pkg.io = novomodelo_io  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "novomodelo", novomodelo_pkg)
        monkeypatch.setitem(sys.modules, "novomodelo.io", novomodelo_io)

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, _stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--validate"],
                monkeypatch,
            )

        assert code == 2
        assert "Validation failed." in stderr

    def test_convert_json_validate_raising_still_emits_one_verdict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A raising ``novomodelo.io.validate`` under ``--json`` still emits one verdict.

        The conversion succeeded, so stdout must still carry exactly one JSON
        object (the --json contract) even though validation crashed and the
        command exits 2; ``summary.validation`` records the failure.
        """
        import types

        from novomodelo_bridge.cli import MIN_NOVOMODELO_VERSION
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=10,
            thermal_count=5,
            bus_count=4,
            line_count=3,
            stage_count=60,
        )

        def _raise(_dst: str) -> dict[str, object]:
            raise RuntimeError("validator blew up")

        # Fix the installed-version gate at exactly MIN_NOVOMODELO_VERSION so this
        # test reaches (and exercises) the injected raising ``validate`` below
        # regardless of whichever novomodelo-python happens to be installed in the
        # dev/CI venv (which may itself now be older than MIN_NOVOMODELO_VERSION).
        monkeypatch.setattr(
            "novomodelo_bridge.novomodelo.compat._installed_novomodelo_python_version",
            lambda: MIN_NOVOMODELO_VERSION,
        )
        novomodelo_pkg = types.ModuleType("novomodelo")
        novomodelo_io = types.ModuleType("novomodelo.io")
        novomodelo_io.validate = _raise  # type: ignore[attr-defined]
        novomodelo_pkg.io = novomodelo_io  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "novomodelo", novomodelo_pkg)
        monkeypatch.setitem(sys.modules, "novomodelo.io", novomodelo_io)

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--json", "--validate"],
                monkeypatch,
            )

        assert code == 2
        doc = json.loads(stdout)
        assert list(doc.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert doc["status"] == "ok"
        assert doc["summary"]["validation"] == {
            "ran": False,
            "valid": None,
            "warnings": 0,
            "errors": 1,
        }
        assert "validator blew up" in stderr
        assert "validator blew up" not in stdout

    def test_convert_json_without_validate_has_no_validation_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without ``--validate`` the convert ``summary`` carries no validation key."""
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=10,
            thermal_count=5,
            bus_count=4,
            line_count=3,
            stage_count=60,
        )

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, stdout, _ = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--json"],
                monkeypatch,
            )

        assert code == 0
        doc = json.loads(stdout)
        assert "validation" not in doc["summary"]

    def test_manifest_records_diagnostics(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The manifest carries the conversion's diagnostics + their summary."""
        from novomodelo_bridge.cli.conversion_manifest import ConversionManifest
        from novomodelo_bridge.core.conversion import ConversionReport
        from novomodelo_bridge.core.diagnostics import Diagnostic, Severity

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"

        warning = Diagnostic(
            code="some-warning",
            severity=Severity.WARNING,
            category="Conversion",
            title="A warning",
            summary="heads up",
        )
        fake_report = ConversionReport(
            hydro_count=1,
            thermal_count=1,
            bus_count=1,
            line_count=0,
            stage_count=12,
            diagnostics=[warning],
        )

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, _stdout, _stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst)],
                monkeypatch,
            )

        assert code == 0
        manifest = ConversionManifest.from_json(dst / "conversion_manifest.json")
        assert manifest.diagnostics_summary == {"warning": 1}
        assert len(manifest.diagnostics) == 1

    def test_clear_dst_removes_manifest(self, tmp_path: Path) -> None:
        """``clear_dst_contents`` removes a stale top-level manifest on --force."""
        from novomodelo_bridge.core.conversion import clear_dst_contents
        from novomodelo_bridge.newave.pipeline import NEWAVE_CLEARED_ARTIFACTS

        dst = tmp_path / "dst"
        dst.mkdir()
        manifest_path = dst / "conversion_manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")

        clear_dst_contents(dst, NEWAVE_CLEARED_ARTIFACTS)

        assert not manifest_path.exists()

    def test_manifest_write_failure_does_not_fail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A manifest write OSError is warned-and-swallowed; exit stays 0."""
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=1,
            thermal_count=1,
            bus_count=1,
            line_count=0,
            stage_count=12,
        )

        def _raise(self: object, path: Path) -> None:
            raise OSError("disk full")

        with (
            patch(
                "novomodelo_bridge.newave.pipeline.convert_newave_case",
                return_value=fake_report,
            ),
            patch(
                "novomodelo_bridge.cli.conversion_manifest.ConversionManifest.to_json",
                _raise,
            ),
        ):
            code, _stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst)],
                monkeypatch,
            )

        assert code == 0
        assert "failed to write conversion manifest" in stderr

    def test_dry_run_writes_nothing_to_dst(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--dry-run`` into an empty dst writes nothing and exits 0."""
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=10,
            thermal_count=5,
            bus_count=4,
            line_count=3,
            stage_count=60,
            would_write_paths=[str(dst / "config.json")],
        )

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, stdout, _stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--dry-run"],
                monkeypatch,
            )

        assert code == 0
        # No destination directory is created and nothing is written.
        assert not dst.exists() or list(dst.iterdir()) == []
        assert "Dry run — no files written" in stdout
        assert "config.json" in stdout

    def test_dry_run_converter_error_exits_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A converter error under ``--dry-run`` exits 1 and names the failure."""
        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            side_effect=ValueError("boom"),
        ):
            code, stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--dry-run"],
                monkeypatch,
            )

        assert code == 1
        assert "Conversion failed" in stderr
        assert "boom" in stderr
        assert "boom" not in stdout

    def test_dry_run_json_document_is_deterministic(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--dry-run --json`` emits a sorted, dst-relative would-write document."""
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"

        # Deliberately unsorted absolute paths under dst.
        fake_report = ConversionReport(
            hydro_count=10,
            thermal_count=5,
            bus_count=4,
            line_count=3,
            stage_count=60,
            would_write_paths=[
                str(dst / "system" / "hydros.json"),
                str(dst / "config.json"),
            ],
        )

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, stdout, _stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--dry-run", "--json"],
                monkeypatch,
            )

        assert code == 0
        doc = json.loads(stdout)
        # would_write moves UNDER summary; only the five envelope keys at the top.
        assert list(doc.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert "would_write" not in doc
        assert doc["schema_version"] == 1
        assert doc["command"] == "convert newave"
        assert doc["status"] == "dry-run"
        assert doc["summary"] == {
            "hydros": 10,
            "thermals": 5,
            "buses": 4,
            "lines": 3,
            "stages": 60,
            # dst-relative, forward-slash, sorted.
            "would_write": ["config.json", "system/hydros.json"],
        }
        assert "timestamp" not in doc

    def test_dry_run_nonempty_dst_without_force_exits_one_and_preserves_dst(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-empty dst without ``--force`` is refused even under ``--dry-run``."""
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        dst.mkdir()
        existing = dst / "existing.txt"
        existing.write_text("keep me", encoding="utf-8")

        fake_report = ConversionReport(
            hydro_count=1,
            thermal_count=1,
            bus_count=1,
            line_count=0,
            stage_count=12,
            would_write_paths=[str(dst / "config.json")],
        )

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=fake_report,
        ) as convert:
            code, _stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--dry-run"],
                monkeypatch,
            )

        assert code == 1
        assert "Use --force" in stderr
        # The destination is untouched: the guard fired before the pipeline ran.
        assert convert.call_count == 0
        assert existing.read_text(encoding="utf-8") == "keep me"
        assert list(dst.iterdir()) == [existing]

    def test_dry_run_with_validate_emits_note_and_skips_validation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--dry-run --validate`` skips validation and notes it on stderr."""
        import builtins

        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=1,
            thermal_count=1,
            bus_count=1,
            line_count=0,
            stage_count=12,
            would_write_paths=[str(dst / "config.json")],
        )

        real_import = builtins.__import__

        def _guard_import(name: str, *args: object, **kwargs: object) -> object:
            # No novomodelo validation import may be attempted under --dry-run.
            if name.startswith("novomodelo.io"):
                raise AssertionError(
                    "validation must not import novomodelo under --dry-run"
                )
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        with (
            patch(
                "novomodelo_bridge.newave.pipeline.convert_newave_case",
                return_value=fake_report,
            ),
            patch.object(builtins, "__import__", _guard_import),
        ):
            code, _stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--dry-run", "--validate"],
                monkeypatch,
            )

        assert code == 0
        assert "--validate is ignored under --dry-run" in stderr
        # No validation output of any kind reached stderr.
        assert "Validation" not in stderr

    @staticmethod
    def _inject_novomodelo_io(
        monkeypatch: pytest.MonkeyPatch, validate: MagicMock
    ) -> MagicMock:
        """Inject a fake ``novomodelo`` / ``novomodelo.io`` whose ``validate`` is *validate*.

        Mirrors the established pattern in the other ``--validate`` tests so the
        real (unreleased-schema) novomodelo package is never touched. Returns the
        ``validate`` mock so the test can assert on its call count.
        """
        import types

        novomodelo_pkg = types.ModuleType("novomodelo")
        novomodelo_io = types.ModuleType("novomodelo.io")
        novomodelo_io.validate = validate  # type: ignore[attr-defined]
        novomodelo_pkg.io = novomodelo_io  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "novomodelo", novomodelo_pkg)
        monkeypatch.setitem(sys.modules, "novomodelo.io", novomodelo_io)
        return validate

    def test_validate_skipped_when_novomodelo_python_old(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A too-old novomodelo-python skips ``validate`` and exits 0."""
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=12
        )

        monkeypatch.setattr(
            "novomodelo_bridge.novomodelo.compat._installed_novomodelo_python_version",
            lambda: "0.9.0",
        )
        validate = self._inject_novomodelo_io(monkeypatch, MagicMock())

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, _stdout, _stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--validate"],
                monkeypatch,
            )

        assert code == 0
        validate.assert_not_called()

    def test_validate_skip_json_payload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The skip records the explicit reason under ``summary.validation``."""
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=12
        )

        monkeypatch.setattr(
            "novomodelo_bridge.novomodelo.compat._installed_novomodelo_python_version",
            lambda: "0.9.0",
        )
        validate = self._inject_novomodelo_io(monkeypatch, MagicMock())

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, stdout, _stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--json", "--validate"],
                monkeypatch,
            )

        assert code == 0
        validate.assert_not_called()
        doc = json.loads(stdout)
        # status stays diagnostics-derived; the skip lands only under summary.
        assert doc["status"] == "ok"
        assert doc["summary"]["validation"] == {
            "ran": False,
            "valid": None,
            "warnings": 0,
            "errors": 0,
            "skipped_reason": "novomodelo-python-too-old",
        }
        assert doc["summary"]["validation"]["ran"] is False
        assert (
            doc["summary"]["validation"]["skipped_reason"]
            == "novomodelo-python-too-old"
        )

    def test_validate_skip_human_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The skip note lands on stderr with the version + skip phrasing."""
        from novomodelo_bridge.cli import MIN_NOVOMODELO_VERSION
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=12
        )

        monkeypatch.setattr(
            "novomodelo_bridge.novomodelo.compat._installed_novomodelo_python_version",
            lambda: "0.9.0",
        )
        validate = self._inject_novomodelo_io(monkeypatch, MagicMock())

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, _stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--validate"],
                monkeypatch,
            )

        assert code == 0
        validate.assert_not_called()
        assert "skipping novomodelo-python validation" in stderr
        assert MIN_NOVOMODELO_VERSION in stderr

    def test_validate_skipped_for_installed_0_12_below_new_min(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An install below the floor gets an honest skip, not a false reject.

        A novomodelo-python older than ``MIN_NOVOMODELO_VERSION`` cannot read the emitted
        output, so the gate must skip validation rather than run it and reject:
        no ``validate`` call, exit 0, and a warning naming *both* the installed
        and required versions so the skip is diagnosable rather than a silent
        no-op.
        """
        from novomodelo_bridge.cli import MIN_NOVOMODELO_VERSION
        from novomodelo_bridge.core.conversion import ConversionReport

        assert MIN_NOVOMODELO_VERSION == "0.18.0"

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=12
        )

        monkeypatch.setattr(
            "novomodelo_bridge.novomodelo.compat._installed_novomodelo_python_version",
            lambda: "0.12.0",
        )
        validate = self._inject_novomodelo_io(monkeypatch, MagicMock())

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--json", "--validate"],
                monkeypatch,
            )

        assert code == 0
        validate.assert_not_called()
        # Does not fail, and does not silently pretend it validated: the
        # warning names both versions, and the JSON summary is explicit that
        # validation did not run (not that it ran and passed).
        assert "skipping novomodelo-python validation" in stderr
        assert "0.12.0" in stderr
        assert "0.18.0" in stderr
        assert MIN_NOVOMODELO_VERSION in stderr
        doc = json.loads(stdout)
        assert doc["status"] == "ok"
        assert doc["summary"]["validation"] == {
            "ran": False,
            "valid": None,
            "warnings": 0,
            "errors": 0,
            "skipped_reason": "novomodelo-python-too-old",
        }

    def test_validate_runs_when_novomodelo_python_supports_schema(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The case validates when the installed novomodelo-python knows the schema."""
        from novomodelo_bridge.cli import MIN_NOVOMODELO_VERSION
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=12
        )

        monkeypatch.setattr(
            "novomodelo_bridge.novomodelo.compat._installed_novomodelo_python_version",
            lambda: MIN_NOVOMODELO_VERSION,
        )
        validate = self._inject_novomodelo_io(
            monkeypatch,
            MagicMock(return_value={"valid": True, "warnings": [], "errors": []}),
        )

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, _stdout, _stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--validate"],
                monkeypatch,
            )

        assert code == 0
        validate.assert_called_once()

    def test_validate_falls_through_when_novomodelo_python_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No installed novomodelo-python → the version gate defers to the generic skip."""
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=12
        )

        monkeypatch.setattr(
            "novomodelo_bridge.novomodelo.compat._installed_novomodelo_python_version",
            lambda: None,
        )
        # Force ``import novomodelo.io`` to raise regardless of whether the real
        # novomodelo-python is installed in this environment (a ``None`` entry in
        # sys.modules makes the import fail), so the generic "not installed"
        # branch runs rather than the version gate.
        monkeypatch.setitem(sys.modules, "novomodelo", None)
        monkeypatch.setitem(sys.modules, "novomodelo.io", None)

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, _stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--validate"],
                monkeypatch,
            )

        assert code == 0
        assert "novomodelo package not installed" in stderr

    def test_validate_runs_when_novomodelo_python_metadata_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No version metadata + an importable novomodelo.io → validation runs."""
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=12
        )

        monkeypatch.setattr(
            "novomodelo_bridge.novomodelo.compat._installed_novomodelo_python_version",
            lambda: None,
        )
        validate = self._inject_novomodelo_io(
            monkeypatch,
            MagicMock(return_value={"valid": True, "warnings": [], "errors": []}),
        )

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, _stdout, _stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--validate"],
                monkeypatch,
            )

        assert code == 0
        validate.assert_called_once()

    def test_convert_newave_validate_unchanged_by_helper(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``convert newave --validate`` still renders the interop warning.

        ``convert newave`` passes an empty whitelist to the shared
        ``_run_novomodelo_validation`` helper, so a warning DECOMP whitelists (the
        novomodelo external-solver-interop note) must still render here — the
        byte-identical-behavior guarantee the helper extraction
        must not break. Contrast with
        ``test_convert_decomp_validate_whitelists_interop`` below.
        """
        from novomodelo_bridge.cli import MIN_NOVOMODELO_VERSION
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=12
        )

        monkeypatch.setattr(
            "novomodelo_bridge.novomodelo.compat._installed_novomodelo_python_version",
            lambda: MIN_NOVOMODELO_VERSION,
        )
        interop_warning = (
            "inflow lags are disabled on all study stages. This is a valid "
            "configuration for external-solver interoperability; otherwise "
            "it is likely a misconfiguration."
        )
        validate = self._inject_novomodelo_io(
            monkeypatch,
            MagicMock(
                return_value={
                    "valid": True,
                    "warnings": [interop_warning],
                    "errors": [],
                }
            ),
        )

        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, _stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--validate"],
                monkeypatch,
            )

        assert code == 0
        validate.assert_called_once()
        assert "external-solver interoperability" in stderr
        assert "Validation warning:" in stderr

    def test_convert_decomp_success_shows_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful ``convert decomp`` prints the ``✓ Converted ...`` summary."""
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=3, thermal_count=2, bus_count=1, line_count=0, stage_count=4
        )

        with patch(
            "novomodelo_bridge.decomp.pipeline.convert_decomp_case",
            return_value=fake_report,
        ):
            code, stdout, _stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst)],
                monkeypatch,
            )

        assert code == 0
        assert "✓ Converted 3 hydros" in stdout

    def test_convert_decomp_warning_diagnostic_renders_rollup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``WARNING`` diagnostic on the report renders the notes roll-up + title."""
        from novomodelo_bridge.core.conversion import ConversionReport
        from novomodelo_bridge.core.diagnostics import Diagnostic, Severity

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"

        warning = Diagnostic(
            code="decomp-some-warning",
            severity=Severity.WARNING,
            category="Conversion",
            title="A DECOMP warning",
            summary="heads up",
        )
        fake_report = ConversionReport(
            hydro_count=1,
            thermal_count=1,
            bus_count=1,
            line_count=0,
            stage_count=4,
            diagnostics=[warning],
        )

        with patch(
            "novomodelo_bridge.decomp.pipeline.convert_decomp_case",
            return_value=fake_report,
        ):
            code, _stdout, stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst)],
                monkeypatch,
            )

        assert code == 0
        assert "Conversion notes:" in stderr
        assert "A DECOMP warning" in stderr

    def test_convert_decomp_json_success_emits_verdict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``convert decomp --json`` emits the unified verdict envelope."""
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=3, thermal_count=2, bus_count=1, line_count=0, stage_count=4
        )

        with patch(
            "novomodelo_bridge.decomp.pipeline.convert_decomp_case",
            return_value=fake_report,
        ):
            code, stdout, _stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst), "--json"],
                monkeypatch,
            )

        assert code == 0
        doc = json.loads(stdout)
        assert list(doc.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert doc["command"] == "convert decomp"
        assert doc["status"] == "ok"
        assert set(doc["summary"]) >= {"hydros", "thermals", "buses", "lines", "stages"}

    def test_convert_decomp_json_failure_emits_error_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A conversion failure under ``--json`` emits ``status == "error"``; exit 1."""
        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"

        with patch(
            "novomodelo_bridge.decomp.pipeline.convert_decomp_case",
            side_effect=ValueError("bad"),
        ):
            code, stdout, _stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst), "--json"],
                monkeypatch,
            )

        assert code == 1
        doc = json.loads(stdout)
        assert doc["status"] == "error"

    def test_convert_decomp_error_status_without_exception_exits_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An ERROR diagnostic reaching a non-raising report exits 1 without --json."""
        from novomodelo_bridge.core.conversion import ConversionReport
        from novomodelo_bridge.core.diagnostics import Diagnostic, Severity

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=3,
            thermal_count=2,
            bus_count=1,
            line_count=0,
            stage_count=4,
            diagnostics=[
                Diagnostic(
                    code="decomp-some-error",
                    severity=Severity.ERROR,
                    category="Conversion",
                    title="An error",
                    summary="boom",
                )
            ],
        )

        with patch(
            "novomodelo_bridge.decomp.pipeline.convert_decomp_case",
            return_value=fake_report,
        ):
            code, _stdout, _stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst), "--no-fcf"],
                monkeypatch,
            )

        assert code == 1

    def test_convert_decomp_diagnostics_json_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--diagnostics-json`` writes the report-shaped sidecar (summary + findings)."""
        from novomodelo_bridge.core.conversion import ConversionReport
        from novomodelo_bridge.core.diagnostics import Diagnostic, Severity

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"
        json_path = tmp_path / "diag.json"

        fake_report = ConversionReport(
            hydro_count=3,
            thermal_count=2,
            bus_count=1,
            line_count=0,
            stage_count=4,
            diagnostics=[
                Diagnostic(
                    code="decomp-some-warning",
                    severity=Severity.WARNING,
                    category="Conversion",
                    title="A DECOMP warning",
                    summary="heads up",
                )
            ],
        )

        with patch(
            "novomodelo_bridge.decomp.pipeline.convert_decomp_case",
            return_value=fake_report,
        ):
            code, _stdout, _stderr = self._invoke_main(
                [
                    "convert",
                    "decomp",
                    str(src),
                    str(dst),
                    "--diagnostics-json",
                    str(json_path),
                ],
                monkeypatch,
            )

        assert code == 0
        assert json_path.exists()
        payload = json.loads(json_path.read_text())
        assert set(payload) == {"summary", "diagnostics"}
        assert payload["summary"]["hydros"] == 3
        assert [d["code"] for d in payload["diagnostics"]] == ["decomp-some-warning"]

    def test_convert_decomp_diagnostics_json_coexists_with_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sidecar and ``--json`` are independent: stdout verdict AND file written."""
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"
        json_path = tmp_path / "diag.json"

        fake_report = ConversionReport(
            hydro_count=3, thermal_count=2, bus_count=1, line_count=0, stage_count=4
        )

        with patch(
            "novomodelo_bridge.decomp.pipeline.convert_decomp_case",
            return_value=fake_report,
        ):
            code, stdout, _stderr = self._invoke_main(
                [
                    "convert",
                    "decomp",
                    str(src),
                    str(dst),
                    "--diagnostics-json",
                    str(json_path),
                    "--json",
                ],
                monkeypatch,
            )

        assert code == 0
        # stdout is exactly the unified verdict object …
        doc = json.loads(stdout)
        assert doc["command"] == "convert decomp"
        # … and the sidecar was written all the same.
        assert json_path.exists()
        assert set(json.loads(json_path.read_text())) == {"summary", "diagnostics"}

    def test_convert_decomp_no_diagnostics_json_writes_no_sidecar(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Absent the flag, no sidecar file is created."""
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"
        json_path = tmp_path / "diag.json"

        fake_report = ConversionReport(
            hydro_count=3, thermal_count=2, bus_count=1, line_count=0, stage_count=4
        )

        with patch(
            "novomodelo_bridge.decomp.pipeline.convert_decomp_case",
            return_value=fake_report,
        ):
            code, _stdout, _stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst)],
                monkeypatch,
            )

        assert code == 0
        assert not json_path.exists()

    def test_convert_decomp_validate_whitelists_interop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``convert decomp --validate`` whitelists the interop inflow-lags warning.

        The P3 lag-blind stage shape (``inflow_lags=false`` on every stage,
        locked in ``test_decomp_temporal.py``) trips novomodelo's non-fatal
        external-solver-interop warning on purpose; DECOMP's whitelist keeps
        it off stderr while newave's does not (see the sibling
        ``test_convert_newave_validate_unchanged_by_helper`` above).
        """
        from novomodelo_bridge.cli import MIN_NOVOMODELO_VERSION
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"

        monkeypatch.setattr(
            "novomodelo_bridge.novomodelo.compat._installed_novomodelo_python_version",
            lambda: MIN_NOVOMODELO_VERSION,
        )
        interop_warning = (
            "inflow lags are disabled on all study stages. This is a valid "
            "configuration for external-solver interoperability; otherwise "
            "it is likely a misconfiguration."
        )
        validate = self._inject_novomodelo_io(
            monkeypatch,
            MagicMock(
                return_value={
                    "valid": True,
                    "warnings": [interop_warning],
                    "errors": [],
                }
            ),
        )

        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=4
        )
        with patch(
            "novomodelo_bridge.decomp.pipeline.convert_decomp_case",
            return_value=fake_report,
        ):
            code, _stdout, stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst), "--validate"],
                monkeypatch,
            )

        assert code == 0
        validate.assert_called_once()
        assert "external-solver interoperability" not in stderr
        assert "Validation warning:" not in stderr

    def test_convert_decomp_dry_run_writes_nothing_to_dst(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--dry-run`` into an empty dst writes nothing and exits 0."""
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=3,
            thermal_count=2,
            bus_count=1,
            line_count=0,
            stage_count=4,
            would_write_paths=[str(dst / "config.json")],
        )

        with patch(
            "novomodelo_bridge.decomp.pipeline.convert_decomp_case",
            return_value=fake_report,
        ):
            code, stdout, _stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst), "--dry-run"],
                monkeypatch,
            )

        assert code == 0
        # No destination directory is created and nothing is written.
        assert not dst.exists() or list(dst.iterdir()) == []
        assert "Dry run — no files written" in stdout
        assert "config.json" in stdout

    def test_convert_decomp_dry_run_json_document_is_deterministic(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--dry-run --json`` emits a sorted, dst-relative would-write document."""
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"

        # Deliberately unsorted absolute paths under dst.
        fake_report = ConversionReport(
            hydro_count=3,
            thermal_count=2,
            bus_count=1,
            line_count=0,
            stage_count=4,
            would_write_paths=[
                str(dst / "system" / "hydros.json"),
                str(dst / "config.json"),
            ],
        )

        with patch(
            "novomodelo_bridge.decomp.pipeline.convert_decomp_case",
            return_value=fake_report,
        ):
            code, stdout, _stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst), "--dry-run", "--json"],
                monkeypatch,
            )

        assert code == 0
        doc = json.loads(stdout)
        # would_write moves UNDER summary; only the five envelope keys at the top.
        assert list(doc.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert "would_write" not in doc
        assert doc["command"] == "convert decomp"
        assert doc["status"] == "dry-run"
        assert doc["summary"] == {
            "hydros": 3,
            "thermals": 2,
            "buses": 1,
            "lines": 0,
            "stages": 4,
            # dst-relative, forward-slash, sorted.
            "would_write": ["config.json", "system/hydros.json"],
        }

    def test_convert_decomp_dry_run_json_failure_has_would_write_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A conversion failure under ``--dry-run --json`` still carries the
        ``would_write`` key (parity with ``convert newave``): a consumer that
        reads ``summary["would_write"]`` on any dry-run verdict must not
        KeyError just because the conversion failed.
        """
        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"

        with patch(
            "novomodelo_bridge.decomp.pipeline.convert_decomp_case",
            side_effect=ValueError("bad"),
        ):
            code, stdout, _stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst), "--dry-run", "--json"],
                monkeypatch,
            )

        assert code == 1
        doc = json.loads(stdout)
        assert list(doc.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert doc["command"] == "convert decomp"
        assert doc["summary"]["would_write"] == []

    def test_convert_decomp_dry_run_with_validate_emits_note_and_skips_validation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--dry-run --validate`` skips validation and notes it on stderr."""
        import builtins

        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=1,
            thermal_count=1,
            bus_count=1,
            line_count=0,
            stage_count=4,
            would_write_paths=[str(dst / "config.json")],
        )

        real_import = builtins.__import__

        def _guard_import(name: str, *args: object, **kwargs: object) -> object:
            # No novomodelo validation import may be attempted under --dry-run.
            if name.startswith("novomodelo.io"):
                raise AssertionError(
                    "validation must not import novomodelo under --dry-run"
                )
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        with (
            patch(
                "novomodelo_bridge.decomp.pipeline.convert_decomp_case",
                return_value=fake_report,
            ),
            patch.object(builtins, "__import__", _guard_import),
        ):
            code, _stdout, stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst), "--dry-run", "--validate"],
                monkeypatch,
            )

        assert code == 0
        assert "--validate is ignored under --dry-run" in stderr
        # No validation output of any kind reached stderr.
        assert "Validation" not in stderr

    def test_convert_decomp_dry_run_writes_no_diagnostics_json_sidecar(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--dry-run --diagnostics-json`` writes no sidecar: the dry-run branch
        returns before the sidecar block runs."""
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"
        json_path = tmp_path / "diag.json"

        fake_report = ConversionReport(
            hydro_count=1,
            thermal_count=1,
            bus_count=1,
            line_count=0,
            stage_count=4,
            would_write_paths=[str(dst / "config.json")],
        )

        with patch(
            "novomodelo_bridge.decomp.pipeline.convert_decomp_case",
            return_value=fake_report,
        ):
            code, _stdout, _stderr = self._invoke_main(
                [
                    "convert",
                    "decomp",
                    str(src),
                    str(dst),
                    "--dry-run",
                    "--diagnostics-json",
                    str(json_path),
                ],
                monkeypatch,
            )

        assert code == 0
        assert not json_path.exists()

    def test_convert_decomp_manifest_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful ``convert decomp`` leaves a valid provenance manifest."""
        from novomodelo_bridge.cli.conversion_manifest import ConversionManifest
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=3, thermal_count=2, bus_count=1, line_count=0, stage_count=4
        )

        with patch(
            "novomodelo_bridge.decomp.pipeline.convert_decomp_case",
            return_value=fake_report,
        ):
            code, _stdout, _stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst)],
                monkeypatch,
            )

        assert code == 0
        manifest_path = dst / "conversion_manifest.json"
        assert manifest_path.exists()
        manifest = ConversionManifest.from_json(manifest_path)
        assert manifest.command == "convert decomp"
        assert manifest.entity_counts == {
            "hydros": 3,
            "thermals": 2,
            "buses": 1,
            "lines": 0,
            "stages": 4,
        }
        # The stub deck's dadger/vazoes/hidr files were discovered and hashed.
        assert manifest.input_files

    def test_convert_decomp_manifest_records_min_novomodelo_version(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The DECOMP manifest's ``min_novomodelo_version`` tracks the CLI constant."""
        from novomodelo_bridge.cli import MIN_NOVOMODELO_VERSION
        from novomodelo_bridge.cli.conversion_manifest import ConversionManifest
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"
        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=12
        )

        with patch(
            "novomodelo_bridge.decomp.pipeline.convert_decomp_case",
            return_value=fake_report,
        ):
            code, _stdout, _stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst)],
                monkeypatch,
            )

        assert code == 0
        manifest = ConversionManifest.from_json(dst / "conversion_manifest.json")
        assert manifest.min_novomodelo_version == MIN_NOVOMODELO_VERSION

    def test_convert_decomp_dry_run_writes_no_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--dry-run`` writes no provenance manifest: the dry-run branch
        returns before the manifest-write block runs."""
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=1,
            thermal_count=1,
            bus_count=1,
            line_count=0,
            stage_count=4,
            would_write_paths=[str(dst / "config.json")],
        )

        with patch(
            "novomodelo_bridge.decomp.pipeline.convert_decomp_case",
            return_value=fake_report,
        ):
            code, _stdout, _stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst), "--dry-run"],
                monkeypatch,
            )

        assert code == 0
        assert not (dst / "conversion_manifest.json").exists()

    # ------------------------------------------------------------------
    # Boundary FCF (default on; --no-fcf opts out; in-process, no --novomodelo-bin)
    # ------------------------------------------------------------------

    def test_convert_decomp_no_fcf_skips_import(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--no-fcf`` skips the importer even when the deck declares cut
        files: no ``boundary/`` directory, and the deck is never re-discovered
        for the FCF gate."""
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_decomp_dir_with_cuts(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=4
        )

        with (
            patch(
                "novomodelo_bridge.decomp.pipeline.convert_decomp_case",
                return_value=fake_report,
            ),
            patch(
                "novomodelo_bridge.decomp.fcf.importer.import_boundary_fcf"
            ) as mock_import,
        ):
            code, _stdout, _stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst), "--no-fcf"],
                monkeypatch,
            )

        assert code == 0
        mock_import.assert_not_called()
        assert not (dst / "boundary").exists()

    def test_convert_decomp_boundary_fcf_happy_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With cut files present (the default), the CLI builds exactly one
        ``DecompCase`` for the FCF step and passes it positionally to the
        importer, which runs with ``cost_scale_factor=1.0``, exits 0,
        confirms the boundary FCF on stderr, and whose ``--json`` verdict
        carries ``summary["boundary_fcf"]``."""
        from novomodelo_bridge.core.conversion import ConversionReport
        from novomodelo_bridge.decomp.case import DecompCase

        src = _make_fake_decomp_dir_with_cuts(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=4
        )

        with (
            patch(
                "novomodelo_bridge.decomp.pipeline.convert_decomp_case",
                return_value=fake_report,
            ),
            patch(
                "novomodelo_bridge.decomp.fcf.capability.ensure_boundary_fcf_capability"
            ) as mock_capability,
            patch(
                "novomodelo_bridge.decomp.fcf.importer.import_boundary_fcf",
                return_value=dst / "boundary",
            ) as mock_import,
            patch(
                "novomodelo_bridge.decomp.case.DecompCase.from_directory",
                wraps=DecompCase.from_directory,
            ) as mock_from_directory,
        ):
            code, stdout, stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst), "--json"],
                monkeypatch,
            )

        assert code == 0
        mock_from_directory.assert_called_once()
        mock_capability.assert_called_once()
        mock_import.assert_called_once()
        assert mock_import.call_args.kwargs["cost_scale_factor"] == 1.0
        assert "novomodelo_bin" not in mock_import.call_args.kwargs
        assert mock_import.call_args.args[0] == dst
        assert isinstance(mock_import.call_args.args[1], DecompCase)
        # The boundary-FCF confirmation surfaces on stderr regardless of --json;
        # the boundary loads on a plain `novomodelo run <case>` (no --output recipe).
        assert f"novomodelo run {dst}" in stderr
        assert "--output" not in stderr
        doc = json.loads(stdout)
        assert doc["summary"]["boundary_fcf"] == {
            "imported": True,
            "path": "boundary",
        }

    def test_convert_decomp_missing_cortes_skips_fcf(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deck that declares no cortes files converts with exit 0 and an
        INFO note (not an error): the importer never runs and no ``boundary/``
        directory is written."""
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_decomp_dir(tmp_path)  # no cortesh/cortes files
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=4
        )

        with (
            patch(
                "novomodelo_bridge.decomp.pipeline.convert_decomp_case",
                return_value=fake_report,
            ),
            patch(
                "novomodelo_bridge.decomp.fcf.importer.import_boundary_fcf"
            ) as mock_import,
        ):
            code, _stdout, stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst)],
                monkeypatch,
            )

        assert code == 0
        assert "no cortes/cortesh files" in stderr
        mock_import.assert_not_called()
        assert not (dst / "boundary").exists()

    def test_convert_decomp_boundary_fcf_capability_guard_failure_exits_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failing capability probe exits 1 with the install remediation."""
        from novomodelo_bridge.core.conversion import ConversionReport
        from novomodelo_bridge.decomp.fcf.capability import REMEDIATION

        src = _make_fake_decomp_dir_with_cuts(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=4
        )

        with (
            patch(
                "novomodelo_bridge.decomp.pipeline.convert_decomp_case",
                return_value=fake_report,
            ),
            patch(
                "novomodelo_bridge.decomp.fcf.capability.ensure_boundary_fcf_capability",
                side_effect=RuntimeError(REMEDIATION),
            ),
            patch(
                "novomodelo_bridge.decomp.fcf.importer.import_boundary_fcf"
            ) as mock_import,
        ):
            code, _stdout, stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst)],
                monkeypatch,
            )

        assert code == 1
        assert "novomodelo-python" in stderr
        mock_import.assert_not_called()

    def test_convert_decomp_boundary_fcf_runs_before_validate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The boundary-FCF import runs BEFORE ``--validate``; a validation
        failure still exits 2 once the import already succeeded."""
        from novomodelo_bridge.cli import MIN_NOVOMODELO_VERSION
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_decomp_dir_with_cuts(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=4
        )

        call_order: list[str] = []

        def _fake_import(*args: object, **kwargs: object) -> Path:
            call_order.append("import_boundary_fcf")
            return dst / "boundary"

        def _fake_validate(*args: object, **kwargs: object) -> dict[str, object]:
            call_order.append("validate")
            return {"valid": False, "warnings": [], "errors": ["boom"]}

        monkeypatch.setattr(
            "novomodelo_bridge.novomodelo.compat._installed_novomodelo_python_version",
            lambda: MIN_NOVOMODELO_VERSION,
        )
        self._inject_novomodelo_io(monkeypatch, MagicMock(side_effect=_fake_validate))

        with (
            patch(
                "novomodelo_bridge.decomp.pipeline.convert_decomp_case",
                return_value=fake_report,
            ),
            patch(
                "novomodelo_bridge.decomp.fcf.capability.ensure_boundary_fcf_capability"
            ),
            patch(
                "novomodelo_bridge.decomp.fcf.importer.import_boundary_fcf",
                side_effect=_fake_import,
            ),
        ):
            code, _stdout, _stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst), "--validate"],
                monkeypatch,
            )

        assert code == 2
        assert call_order == ["import_boundary_fcf", "validate"]

    def test_convert_decomp_fcf_skipped_under_dry_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--dry-run`` skips the boundary FCF import and notes it on stderr,
        even when the deck declares cut files."""
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_decomp_dir_with_cuts(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=1,
            thermal_count=1,
            bus_count=1,
            line_count=0,
            stage_count=4,
            would_write_paths=[str(dst / "config.json")],
        )

        with (
            patch(
                "novomodelo_bridge.decomp.pipeline.convert_decomp_case",
                return_value=fake_report,
            ),
            patch(
                "novomodelo_bridge.decomp.fcf.importer.import_boundary_fcf"
            ) as mock_import,
        ):
            code, _stdout, stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst), "--dry-run"],
                monkeypatch,
            )

        assert code == 0
        assert "boundary FCF import is skipped under --dry-run" in stderr
        mock_import.assert_not_called()
        assert not dst.exists() or list(dst.iterdir()) == []

    def test_convert_decomp_boundary_fcf_importer_diagnostics_reach_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The importer runs inside a ``dx.collect()`` sink,
        so a ``Diagnostic`` it emits — here the GNL anticipated-ring
        deviation — reaches the ``--json`` verdict's ``diagnostics`` array.
        This test fails against the pre-sink CLI (no ``dx.collect()``
        wrapping ``import_boundary_fcf``), proving the previously deferred gap
        is closed."""
        from novomodelo_bridge.core import diagnostics as dx
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_decomp_dir_with_cuts(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=4
        )

        def _fake_import(*args: object, **kwargs: object) -> Path:
            dx.emit(
                dx.Diagnostic(
                    code="boundary-fcf-gnl-anticipated-deviation",
                    severity=dx.Severity.INFO,
                    category="Boundary FCF",
                    title="GNL anticipated ring carries a per-patamar sum",
                    summary="synthetic deviation diagnostic for the sink test",
                )
            )
            return dst / "boundary"

        with (
            patch(
                "novomodelo_bridge.decomp.pipeline.convert_decomp_case",
                return_value=fake_report,
            ),
            patch(
                "novomodelo_bridge.decomp.fcf.capability.ensure_boundary_fcf_capability"
            ),
            patch(
                "novomodelo_bridge.decomp.fcf.importer.import_boundary_fcf",
                side_effect=_fake_import,
            ),
        ):
            code, stdout, _stderr = self._invoke_main(
                [
                    "convert",
                    "decomp",
                    str(src),
                    str(dst),
                    "--json",
                ],
                monkeypatch,
            )

        assert code == 0
        doc = json.loads(stdout)
        codes = {d["code"] for d in doc["diagnostics"]}
        assert "boundary-fcf-gnl-anticipated-deviation" in codes
        # ``status`` stays "ok": the importer's diagnostic is INFO-severity,
        # so surfacing it never flips the verdict outcome.
        assert doc["status"] == "ok"

    def test_convert_decomp_boundary_fcf_importer_diagnostics_render_panel(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same emitting mock, without ``--json`` — the
        diagnostic's title renders on stderr (the Rich panel), and the
        existing happy-path boundary-FCF confirmation still holds (no
        double-render, no exit-code change)."""
        from novomodelo_bridge.core import diagnostics as dx
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_decomp_dir_with_cuts(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=4
        )

        def _fake_import(*args: object, **kwargs: object) -> Path:
            dx.emit(
                dx.Diagnostic(
                    code="boundary-fcf-gnl-anticipated-deviation",
                    severity=dx.Severity.INFO,
                    category="Boundary FCF",
                    title="GNL anticipated ring carries a per-patamar sum",
                    summary="synthetic deviation diagnostic for the sink test",
                )
            )
            return dst / "boundary"

        with (
            patch(
                "novomodelo_bridge.decomp.pipeline.convert_decomp_case",
                return_value=fake_report,
            ),
            patch(
                "novomodelo_bridge.decomp.fcf.capability.ensure_boundary_fcf_capability"
            ),
            patch(
                "novomodelo_bridge.decomp.fcf.importer.import_boundary_fcf",
                side_effect=_fake_import,
            ),
        ):
            code, _stdout, stderr = self._invoke_main(
                [
                    "convert",
                    "decomp",
                    str(src),
                    str(dst),
                ],
                monkeypatch,
            )

        assert code == 0
        assert "GNL anticipated ring carries a per-patamar sum" in stderr
        # The boundary-FCF confirmation still surfaces (happy-path intact).
        assert f"novomodelo run {dst}" in stderr
        assert "--output" not in stderr

    def test_convert_decomp_boundary_fcf_importer_diagnostics_reach_sidecar(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same emitting mock as
        ``..._reach_json``, but without ``--json`` and with
        ``--diagnostics-json`` — the importer's ``Diagnostic`` (captured by
        the ``dx.collect()`` sink) must reach the sidecar file too, not just
        the ``--json`` stdout verdict. This test fails against the pre-fix
        CLI, where the sidecar is written BEFORE the boundary-FCF block runs
        and therefore only ever contains ``report.diagnostics``."""
        from novomodelo_bridge.core import diagnostics as dx
        from novomodelo_bridge.core.conversion import ConversionReport

        src = _make_fake_decomp_dir_with_cuts(tmp_path)
        dst = tmp_path / "dst"
        json_path = tmp_path / "diag.json"

        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=4
        )

        def _fake_import(*args: object, **kwargs: object) -> Path:
            dx.emit(
                dx.Diagnostic(
                    code="boundary-fcf-gnl-anticipated-deviation",
                    severity=dx.Severity.INFO,
                    category="Boundary FCF",
                    title="GNL anticipated ring carries a per-patamar sum",
                    summary="synthetic deviation diagnostic for the sink test",
                )
            )
            return dst / "boundary"

        with (
            patch(
                "novomodelo_bridge.decomp.pipeline.convert_decomp_case",
                return_value=fake_report,
            ),
            patch(
                "novomodelo_bridge.decomp.fcf.capability.ensure_boundary_fcf_capability"
            ),
            patch(
                "novomodelo_bridge.decomp.fcf.importer.import_boundary_fcf",
                side_effect=_fake_import,
            ),
        ):
            code, _stdout, _stderr = self._invoke_main(
                [
                    "convert",
                    "decomp",
                    str(src),
                    str(dst),
                    "--diagnostics-json",
                    str(json_path),
                ],
                monkeypatch,
            )

        assert code == 0
        assert json_path.exists()
        payload = json.loads(json_path.read_text())
        codes = {d["code"] for d in payload["diagnostics"]}
        assert "boundary-fcf-gnl-anticipated-deviation" in codes

    def test_convert_decomp_diagnostics_json_unchanged_without_fcf(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Guard: a ``convert decomp --diagnostics-json --no-fcf`` sidecar is
        exactly ``report.diagnostics`` — deferring the sidecar write past the
        (here, skipped) boundary-FCF block must not regress the contract."""
        from novomodelo_bridge.core.conversion import ConversionReport
        from novomodelo_bridge.core.diagnostics import Diagnostic, Severity

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"
        json_path = tmp_path / "diag.json"

        fake_report = ConversionReport(
            hydro_count=3,
            thermal_count=2,
            bus_count=1,
            line_count=0,
            stage_count=4,
            diagnostics=[
                Diagnostic(
                    code="decomp-some-warning",
                    severity=Severity.WARNING,
                    category="Conversion",
                    title="A DECOMP warning",
                    summary="heads up",
                )
            ],
        )

        with patch(
            "novomodelo_bridge.decomp.pipeline.convert_decomp_case",
            return_value=fake_report,
        ):
            code, _stdout, _stderr = self._invoke_main(
                [
                    "convert",
                    "decomp",
                    str(src),
                    str(dst),
                    "--diagnostics-json",
                    str(json_path),
                    "--no-fcf",
                ],
                monkeypatch,
            )

        assert code == 0
        payload = json.loads(json_path.read_text())
        assert set(payload) == {"summary", "diagnostics"}
        assert [d["code"] for d in payload["diagnostics"]] == ["decomp-some-warning"]


class TestConversionDiagnosticsRendering:
    """The convert subcommand renders structured diagnostics (names/stages/values)."""

    def _invoke_main(
        self, argv: list[str], monkeypatch: pytest.MonkeyPatch
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

    def _report_with_gtmin(self):
        from novomodelo_bridge.core.conversion import ConversionReport
        from novomodelo_bridge.core.diagnostics import (
            Diagnostic,
            DiagnosticTable,
            Severity,
        )

        diag = Diagnostic(
            code="thermal-gtmin-above-capacity",
            severity=Severity.WARNING,
            category="Thermal bounds",
            title="GTMIN exceeds available capacity (1 plant(s))",
            summary="one plant affected",
            table=DiagnosticTable(
                columns=["Plant", "Code", "Stages", "GTMIN MW", "Cap MW"],
                rows=[["ANGRA 2", 13, "2-3", 481.3, 423.4]],
            ),
            remediation="Check EXPT FCMAX/GTMIN and MANUTT for these plants.",
        )
        return ConversionReport(hydro_count=1, diagnostics=[diag])

    def test_structured_diagnostic_shows_name_stages_values(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=self._report_with_gtmin(),
        ):
            code, stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst)], monkeypatch
            )
        assert code == 0
        assert "1 hydros" in stdout  # summary on stdout
        # The named pain: plant name, the stages, and the values are all surfaced.
        assert "ANGRA 2" in stderr
        assert "2-3" in stderr
        assert "481.3" in stderr
        assert "423.4" in stderr

    def test_quiet_suppresses_summary_but_keeps_warnings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=self._report_with_gtmin(),
        ):
            code, stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--quiet"], monkeypatch
            )
        assert code == 0
        assert "Converted" not in stdout  # summary suppressed
        assert "ANGRA 2" in stderr  # warnings still shown

    def test_diagnostics_json_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        json_path = tmp_path / "diag.json"
        with patch(
            "novomodelo_bridge.newave.pipeline.convert_newave_case",
            return_value=self._report_with_gtmin(),
        ):
            code, _stdout, _stderr = self._invoke_main(
                [
                    "convert",
                    "newave",
                    str(src),
                    str(dst),
                    "--diagnostics-json",
                    str(json_path),
                ],
                monkeypatch,
            )
        assert code == 0
        assert json_path.exists()
        payload = json.loads(json_path.read_text())
        assert payload["summary"]["hydros"] == 1
        codes = [d["code"] for d in payload["diagnostics"]]
        assert "thermal-gtmin-above-capacity" in codes
