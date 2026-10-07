"""Tests for pipeline.py."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest

from tests.conftest import (
    _FAKE_LOAD_FACTORS,
    _FAKE_NCS_FACTORS,
    _all_converter_patches,
    _make_fake_newave_dir,
)
from tests.newave.conftest import _run_with_all_mocks


class TestConversionReport:
    def test_str_format(self) -> None:
        from novomodelo_bridge.core.conversion import ConversionReport

        report = ConversionReport(
            hydro_count=3,
            thermal_count=5,
            bus_count=4,
            line_count=2,
            stage_count=60,
        )
        s = str(report)
        assert "3 hydros" in s
        assert "5 thermals" in s
        assert "4 buses" in s
        assert "2 lines" in s
        assert "60 stages" in s

    def test_default_zeros(self) -> None:
        from novomodelo_bridge.core.conversion import ConversionReport

        report = ConversionReport()
        assert report.hydro_count == 0
        assert report.thermal_count == 0
        assert report.bus_count == 0
        assert report.line_count == 0
        assert report.stage_count == 0
        assert report.warnings == []


# ---------------------------------------------------------------------------
# Pipeline unit tests (all converters mocked)
# ---------------------------------------------------------------------------


class TestConvertNewaweCasePipeline:
    """Unit tests for pipeline.convert_newave_case with all converters mocked."""

    def test_all_output_files_written(self, tmp_path: Path) -> None:
        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "novomodelo_case"

        _run_with_all_mocks(src, dst)

        expected = [
            dst / "config.json",
            dst / "stages.json",
            dst / "penalties.json",
            dst / "initial_conditions.json",
            dst / "system" / "hydros.json",
            dst / "system" / "thermals.json",
            dst / "system" / "buses.json",
            dst / "system" / "lines.json",
            dst / "scenarios" / "inflow_seasonal_stats.parquet",
            dst / "scenarios" / "load_seasonal_stats.parquet",
            dst / "scenarios" / "inflow_history.parquet",
            dst / "system" / "hydro_geometry.parquet",
            dst / "scenarios" / "load_factors.json",
            dst / "constraints" / "line_bounds.parquet",
            dst / "system" / "non_controllable_sources.json",
            dst / "scenarios" / "non_controllable_factors.json",
            dst / "scenarios" / "non_controllable_stats.parquet",
        ]
        for f in expected:
            assert f.exists(), f"Expected output file not found: {f}"

    def test_exchange_factors_json_is_not_written(self, tmp_path: Path) -> None:
        """The per-block exchange factors are folded into line_bounds.parquet
        (novomodelo decision 10); the pipeline must not write the deleted file."""
        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "novomodelo_case"

        _run_with_all_mocks(src, dst)

        assert not (dst / "constraints" / "exchange_factors.json").exists()

    def test_load_factors_and_ncs_factors_still_byte_identical(
        self, tmp_path: Path
    ) -> None:
        """``load_factors.json`` and ``non_controllable_factors.json`` are
        untouched by the exchange-factors migration (which draws the line at
        authored-vs-sampled data; deleting these by analogy would be wrong)."""
        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "novomodelo_case"

        _run_with_all_mocks(src, dst)

        load_factors_path = dst / "scenarios" / "load_factors.json"
        with load_factors_path.open(encoding="utf-8") as f:
            assert json.load(f) == _FAKE_LOAD_FACTORS

        ncs_factors_path = dst / "scenarios" / "non_controllable_factors.json"
        with ncs_factors_path.open(encoding="utf-8") as f:
            assert json.load(f) == _FAKE_NCS_FACTORS

    def test_json_files_are_valid_json(self, tmp_path: Path) -> None:
        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "novomodelo_case"

        _run_with_all_mocks(src, dst)

        for json_path in [
            dst / "config.json",
            dst / "stages.json",
            dst / "system" / "hydros.json",
            dst / "system" / "thermals.json",
            dst / "system" / "buses.json",
            dst / "system" / "lines.json",
        ]:
            with json_path.open(encoding="utf-8") as f:
                data = json.load(f)
            assert data is not None, f"Invalid JSON: {json_path}"

    def test_parquet_files_are_readable(self, tmp_path: Path) -> None:
        import pyarrow.parquet as pq

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "novomodelo_case"

        _run_with_all_mocks(src, dst)

        inflow = pq.read_table(dst / "scenarios" / "inflow_seasonal_stats.parquet")
        assert inflow.num_columns == 4
        load = pq.read_table(dst / "scenarios" / "load_seasonal_stats.parquet")
        assert load.num_columns == 4

    def test_report_counts_from_converter_output(self, tmp_path: Path) -> None:
        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "novomodelo_case"

        report = _run_with_all_mocks(src, dst)

        assert report.hydro_count == 2  # type: ignore[union-attr]
        assert report.thermal_count == 1
        assert report.bus_count == 3
        assert report.line_count == 1
        assert report.stage_count == 12

    def test_production_models_written_when_converter_returns_data(
        self, tmp_path: Path
    ) -> None:
        """When convert_production_models returns data, the file is written."""
        from novomodelo_bridge.newave.pipeline import convert_newave_case

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "novomodelo_case"

        _FAKE_PROD_MODELS = {
            "production_models": [
                {
                    "hydro_id": 0,
                    "selection_mode": "stage_ranges",
                    "stage_ranges": [
                        {
                            "start_stage_id": 0,
                            "end_stage_id": None,
                            "model": "constant_productivity",
                            "productivity_mw_per_m3s": 1.23,
                        }
                    ],
                }
            ]
        }

        fake_id_map = MagicMock()
        # Use ExitStack for correct LIFO teardown to avoid mock leakage.
        import contextlib

        patches = _all_converter_patches(fake_id_map)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            # Override the production_models patch (entered last -> exits first).
            stack.enter_context(
                patch(
                    "novomodelo_bridge.newave.pipeline.hydro_conv.convert_production_models",
                    return_value=_FAKE_PROD_MODELS,
                )
            )
            convert_newave_case(src, dst)

        pm_path = dst / "system" / "hydro_production_models.json"
        assert pm_path.exists(), "hydro_production_models.json not written"
        with pm_path.open(encoding="utf-8") as f:
            data = json.load(f)
        assert data["production_models"][0]["hydro_id"] == 0

    def test_production_models_always_written(self, tmp_path: Path) -> None:
        """Novomodelo HEAD requires hydro_production_models.json — pipeline always writes it.

        Productivity moved out of `hydros.json:generation`, so the production
        models file is now mandatory for the converted case to load in novomodelo.
        """
        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "novomodelo_case"

        _run_with_all_mocks(src, dst)

        assert (dst / "system" / "hydro_production_models.json").exists()

    def test_missing_required_file_raises(self, tmp_path: Path) -> None:
        from novomodelo_bridge.newave.pipeline import convert_newave_case

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "novomodelo_case"

        with (
            patch(
                "novomodelo_bridge.newave.pipeline.NewaveCase.from_directory",
                side_effect=FileNotFoundError(
                    f"Required NEWAVE file not found in {src}: hidr.dat"
                ),
            ),
            pytest.raises(FileNotFoundError) as exc_info,
        ):
            convert_newave_case(src, dst)
        assert "hidr.dat" in str(exc_info.value)

    def test_dry_run_does_not_call_write_table(self, tmp_path: Path) -> None:
        """``dry_run=True`` writes nothing yet records the would-write paths."""
        import contextlib

        from novomodelo_bridge.core.conversion import ConversionReport
        from novomodelo_bridge.newave.pipeline import convert_newave_case

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "novomodelo_case"

        fake_id_map = MagicMock()
        with (
            patch(
                "novomodelo_bridge.novomodelo.case_writer.pq.write_table"
            ) as write_table,
            contextlib.ExitStack() as stack,
        ):
            for p in _all_converter_patches(fake_id_map):
                stack.enter_context(p)
            report = convert_newave_case(src, dst, dry_run=True)

        assert isinstance(report, ConversionReport)
        # No Parquet table is written and no destination directory is created.
        assert write_table.call_count == 0
        assert not dst.exists() or list(dst.iterdir()) == []
        # The would-write listing is still populated (covers JSON and Parquet).
        assert report.would_write_paths
        assert str(dst / "config.json") in report.would_write_paths
        assert str(dst / "system" / "hydros.json") in report.would_write_paths


class TestEmissionCheckWiring:
    """The post-emission self-checks run inside the real
    pipeline body, before the writes, and their findings flip the convert
    verdict through ``_convert_status`` — not merely by inspecting the
    diagnostic."""

    def test_no_hydro_bounds_reports_rule_43_not_applicable(
        self, tmp_path: Path
    ) -> None:
        """The fully-mocked fixture never builds a hydro_bounds table, so rule
        43 is explicitly "not applicable" (INFO), not silently absent, and the
        convert verdict stays clean."""
        from novomodelo_bridge.cli.verdict import _convert_status
        from novomodelo_bridge.core.diagnostics import Severity

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "novomodelo_case"
        report = _run_with_all_mocks(src, dst)

        not_applicable = [
            d
            for d in report.diagnostics  # type: ignore[union-attr]
            if d.code == "hydro-bounds-raising-not-applicable"
        ]
        assert len(not_applicable) == 1
        assert not_applicable[0].severity is Severity.INFO
        assert _convert_status(report.diagnostics, success="ok") == "ok"  # type: ignore[union-attr]

    def test_over_declared_bound_is_clamped_and_warned_not_errored(
        self, tmp_path: Path
    ) -> None:
        """A hydro_bounds row above the plant's declared max_turbined_m3s is
        clamped back to the declaration and reported as a WARNING, so the
        convert verdict stays "ok" instead of erroring on novomodelo rule 43."""
        import contextlib

        from novomodelo_bridge.cli.verdict import _convert_status
        from novomodelo_bridge.core.diagnostics import Severity
        from novomodelo_bridge.newave.pipeline import convert_newave_case

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "novomodelo_case"

        hydros = {
            "$schema": "http://example",
            "hydros": [
                {
                    "id": 0,
                    "name": "Test Plant",
                    "generation": {
                        "max_turbined_m3s": 100.0,
                        "max_generation_mw": 50.0,
                    },
                    "unit_groups": [
                        {
                            "id": 0,
                            "max_turbined_m3s": 100.0,
                            "max_generation_mw": 50.0,
                        }
                    ],
                }
            ],
        }
        over_declared_bounds = pa.table(
            {
                "hydro_id": pa.array([0], type=pa.int32()),
                "stage_id": pa.array([2], type=pa.int32()),
                "max_turbined_m3s": pa.array([150.0], type=pa.float64()),
            }
        )

        fake_id_map = MagicMock()
        with contextlib.ExitStack() as stack:
            for p in _all_converter_patches(fake_id_map):
                stack.enter_context(p)
            stack.enter_context(
                patch(
                    "novomodelo_bridge.newave.pipeline.hydro_conv.convert_hydros",
                    return_value=hydros,
                )
            )
            stack.enter_context(
                patch(
                    "novomodelo_bridge.newave.pipeline.hydro_conv.convert_storage_bounds",
                    return_value=over_declared_bounds,
                )
            )
            report = convert_newave_case(src, dst)

        # The clamp resolves it: a warning, not a rule-43 error.
        assert not any(
            d.code == "hydro-bounds-raises-declared-capacity"
            for d in report.diagnostics
        )
        clamps = [
            d
            for d in report.diagnostics
            if d.code == "hydro-bounds-clamped-to-declared-capacity"
        ]
        assert len(clamps) == 1
        assert clamps[0].severity is Severity.WARNING

        # Load-bearing: a warning does not flip the verdict.
        assert _convert_status(report.diagnostics, success="ok") == "ok"


# ---------------------------------------------------------------------------
# Pipeline integration tests for inflow_history.parquet
# ---------------------------------------------------------------------------


class TestPipelineInflowHistory:
    """Tests verifying that convert_newave_case always writes inflow_history.parquet."""

    def test_inflow_history_always_written(self, tmp_path: Path) -> None:
        """inflow_history.parquet is always written (from vazoes.dat)."""
        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "novomodelo_case"

        _run_with_all_mocks(src, dst)

        history_path = dst / "scenarios" / "inflow_history.parquet"
        assert history_path.exists(), "inflow_history.parquet was not written"


class TestConversionWarningCapture:
    """``convert_newave_case`` surfaces converter warnings via ConversionReport."""

    def test_captures_and_dedupes_package_warnings(self, tmp_path: Path) -> None:
        from novomodelo_bridge.core.conversion import ConversionReport
        from novomodelo_bridge.newave import pipeline
        from novomodelo_bridge.newave.pipeline import convert_newave_case

        log = logging.getLogger("novomodelo_bridge.newave.converters.fake")

        def fake_impl(
            src: Path,
            dst: Path,
            on_phase: object = None,
            *,
            dry_run: bool = False,
        ) -> ConversionReport:
            log.warning("vazpast.dat unreadable; using empty tendency")
            log.warning("vazpast.dat unreadable; using empty tendency")  # duplicate
            log.info("informational, not a degradation")  # below WARNING → ignored
            log.warning("REE.DAT has no entries")
            return ConversionReport(hydro_count=3)

        with patch.object(pipeline, "_convert_newave_case_impl", side_effect=fake_impl):
            report = convert_newave_case(tmp_path, tmp_path)

        assert report.hydro_count == 3
        assert report.warnings == [
            "vazpast.dat unreadable; using empty tendency",
            "REE.DAT has no entries",
        ]

    def test_no_warnings_when_clean(self, tmp_path: Path) -> None:
        from novomodelo_bridge.core.conversion import ConversionReport
        from novomodelo_bridge.newave import pipeline
        from novomodelo_bridge.newave.pipeline import convert_newave_case

        with patch.object(
            pipeline,
            "_convert_newave_case_impl",
            return_value=ConversionReport(hydro_count=1),
        ):
            report = convert_newave_case(tmp_path, tmp_path)

        assert report.warnings == []

    def test_collector_detached_even_on_exception(self, tmp_path: Path) -> None:
        from novomodelo_bridge.newave import pipeline
        from novomodelo_bridge.newave.pipeline import convert_newave_case

        pkg_logger = logging.getLogger("novomodelo_bridge")
        handlers_before = list(pkg_logger.handlers)
        with (
            patch.object(
                pipeline,
                "_convert_newave_case_impl",
                side_effect=RuntimeError("boom"),
            ),
            pytest.raises(RuntimeError, match="boom"),
        ):
            convert_newave_case(tmp_path, tmp_path)

        # The capture handler must be removed in the finally block, leaving the
        # package logger's handler list exactly as it was.
        assert pkg_logger.handlers == handlers_before

    def test_partial_outputs_cleared_on_failure(self, tmp_path: Path) -> None:
        """A failure partway through the write phase must not leave a partial,
        valid-looking case behind: the known pipeline outputs are removed so a
        plain (no --force) re-run is not refused as non-empty."""
        from novomodelo_bridge.newave import pipeline
        from novomodelo_bridge.newave.pipeline import convert_newave_case

        dst = tmp_path / "dst"

        def fake_impl(
            src: Path,
            d: Path,
            on_phase: object = None,
            *,
            dry_run: bool = False,
        ) -> object:
            # Simulate a write phase that got partway: a top-level JSON and a
            # system/ subdir were written before the failure.
            (d / "system").mkdir(parents=True, exist_ok=True)
            (d / "config.json").write_text("{}")
            (d / "system" / "hydros.json").write_text("{}")
            raise RuntimeError("disk full mid-write")

        with (
            patch.object(pipeline, "_convert_newave_case_impl", side_effect=fake_impl),
            pytest.raises(RuntimeError, match="disk full"),
        ):
            convert_newave_case(tmp_path, dst)

        # No pipeline outputs survive — dst holds no half-written case.
        assert not (dst / "config.json").exists()
        assert not (dst / "system").exists()
        # dst itself may remain but must be empty, so a no-force re-run proceeds.
        assert not any(dst.iterdir())

    def test_dry_run_failure_preserves_pre_existing_dst_contents(
        self, tmp_path: Path
    ) -> None:
        """A dry-run failure must never clear ``dst``: it wrote nothing, and
        ``dst`` may be a pre-existing populated directory the user never
        asked to clear."""
        from novomodelo_bridge.newave import pipeline
        from novomodelo_bridge.newave.pipeline import convert_newave_case

        dst = tmp_path / "dst"
        dst.mkdir()
        (dst / "config.json").write_text("{}")
        (dst / "system").mkdir()
        (dst / "system" / "hydros.json").write_text("{}")

        def fake_impl(
            src: Path,
            d: Path,
            on_phase: object = None,
            *,
            dry_run: bool = False,
        ) -> object:
            raise RuntimeError("boom")

        with (
            patch.object(pipeline, "_convert_newave_case_impl", side_effect=fake_impl),
            pytest.raises(RuntimeError, match="boom"),
        ):
            convert_newave_case(tmp_path, dst, dry_run=True)

        assert (dst / "config.json").exists()
        assert (dst / "system").exists()


def test_convert_newave_case_threads_on_phase(tmp_path: Path) -> None:
    """``convert_newave_case`` forwards its ``on_phase`` callback to the impl."""
    from novomodelo_bridge.core.conversion import ConversionReport
    from novomodelo_bridge.newave import pipeline
    from novomodelo_bridge.newave.pipeline import convert_newave_case

    received: list[str] = []

    def fake_impl(
        src: Path,
        dst: Path,
        on_phase: object = None,
        *,
        dry_run: bool = False,
    ) -> ConversionReport:
        if on_phase is not None:
            on_phase("Discovering files")  # type: ignore[operator]
        return ConversionReport()

    with patch.object(pipeline, "_convert_newave_case_impl", side_effect=fake_impl):
        convert_newave_case(tmp_path, tmp_path, on_phase=received.append)

    assert received == ["Discovering files"]
