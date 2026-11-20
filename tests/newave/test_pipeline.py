"""Tests for pipeline.py."""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable
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

_LOG = logging.getLogger(__name__)


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
            dst / "system" / "pumping_stations.json",
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
        """The per-block exchange factors are folded into line_bounds.parquet;
        the pipeline writes no exchange_factors.json."""
        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "novomodelo_case"

        _run_with_all_mocks(src, dst)

        assert not (dst / "constraints" / "exchange_factors.json").exists()

    def test_inflow_seasonal_stats_are_left_to_novomodelo(self, tmp_path: Path) -> None:
        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "novomodelo_case"

        _run_with_all_mocks(src, dst)

        assert not (dst / "scenarios" / "inflow_seasonal_stats.parquet").exists()
        assert (dst / "scenarios" / "inflow_history.parquet").exists()

    def test_load_factors_and_ncs_factors_written_verbatim(
        self, tmp_path: Path
    ) -> None:
        """``load_factors.json`` and ``non_controllable_factors.json`` are
        written verbatim; only the exchange factors fold into
        ``line_bounds.parquet``."""
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

        load = pq.read_table(dst / "scenarios" / "load_seasonal_stats.parquet")
        assert load.num_columns == 4

    def test_report_counts_from_converter_output(self, tmp_path: Path) -> None:
        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "novomodelo_case"

        report = _run_with_all_mocks(src, dst)

        assert report.hydro_count == 2
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


class TestOptionalLinesAndThermals:
    @staticmethod
    def _run(tmp_path: Path, *, thermals: list, lines: list) -> Path:
        import contextlib

        from novomodelo_bridge.newave.pipeline import convert_newave_case

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "novomodelo_case"
        with contextlib.ExitStack() as stack:
            for p in _all_converter_patches(MagicMock()):
                stack.enter_context(p)
            stack.enter_context(
                patch(
                    "novomodelo_bridge.newave.pipeline.thermal_conv.convert_thermals",
                    return_value={"thermals": thermals},
                )
            )
            stack.enter_context(
                patch(
                    "novomodelo_bridge.newave.pipeline.network_conv.convert_lines",
                    return_value={"lines": lines},
                )
            )
            convert_newave_case(src, dst)
        return dst

    def test_no_lines_and_no_thermals_omits_both_files(self, tmp_path: Path) -> None:
        dst = self._run(tmp_path, thermals=[], lines=[])

        assert (dst / "system" / "buses.json").is_file()
        assert not (dst / "system" / "lines.json").exists()
        assert not (dst / "system" / "thermals.json").exists()

    def test_no_lines_keeps_thermals_file(self, tmp_path: Path) -> None:
        dst = self._run(tmp_path, thermals=[{"id": 0}], lines=[])

        assert not (dst / "system" / "lines.json").exists()
        assert (dst / "system" / "thermals.json").is_file()


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
            for d in report.diagnostics
            if d.code == "hydro-bounds-raising-not-applicable"
        ]
        assert len(not_applicable) == 1
        assert not_applicable[0].severity is Severity.INFO
        assert _convert_status(report.diagnostics, success="ok") == "ok"

    def test_over_declared_bound_is_clamped_with_a_note_not_errored(
        self, tmp_path: Path
    ) -> None:
        """A hydro_bounds row above the plant's declared max_turbined_m3s is
        clamped back to the declaration and reported as a note, so the
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
        fake_id_map.all_hydro_codes = [6]
        fake_id_map.hydro_id.side_effect = {6: 0}.__getitem__
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

        # The clamp resolves it: a note, not a rule-43 error.
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
        assert clamps[0].severity is Severity.INFO

        # Load-bearing: the clamp does not flip the verdict.
        assert _convert_status(report.diagnostics, success="ok") == "ok"

    def test_storage_floor_above_the_reservoir_stops_the_conversion_naming_modif(
        self, tmp_path: Path
    ) -> None:
        """novomodelo holds a non-filling plant's storage floor as a bound, so a
        VMINT above the reservoir would leave the stage with no solution: the
        conversion stops before writing, naming modif.dat."""
        import contextlib

        from novomodelo_bridge.core.errors import FieldParseError
        from novomodelo_bridge.newave.pipeline import convert_newave_case

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "novomodelo_case"
        modif = src / "modif.dat"

        hydros = {
            "$schema": "http://example",
            "hydros": [
                {
                    "id": 0,
                    "name": "Test Plant",
                    "reservoir": {"min_storage_hm3": 10.0, "max_storage_hm3": 500.0},
                }
            ],
        }
        floor_above_the_reservoir = pa.table(
            {
                "hydro_id": pa.array([0], type=pa.int32()),
                "stage_id": pa.array([2], type=pa.int32()),
                "min_storage_hm3": pa.array([600.0], type=pa.float64()),
            }
        )

        fake_id_map = MagicMock()
        fake_id_map.all_hydro_codes = [6]
        fake_id_map.hydro_id.side_effect = {6: 0}.__getitem__
        fake_case = MagicMock()
        fake_case.id_map = fake_id_map
        fake_case.files.modif = modif
        with contextlib.ExitStack() as stack:
            for p in _all_converter_patches(fake_id_map):
                stack.enter_context(p)
            stack.enter_context(
                patch(
                    "novomodelo_bridge.newave.pipeline.NewaveCase.from_directory",
                    return_value=fake_case,
                )
            )
            stack.enter_context(
                patch(
                    "novomodelo_bridge.newave.pipeline.hydro_conv.convert_hydros",
                    return_value=hydros,
                )
            )
            stack.enter_context(
                patch(
                    "novomodelo_bridge.newave.pipeline.hydro_conv.convert_storage_bounds",
                    return_value=floor_above_the_reservoir,
                )
            )
            with pytest.raises(
                FieldParseError, match="above the plant's maximum"
            ) as err:
                convert_newave_case(src, dst)

        assert err.value.path == str(modif)
        assert not (dst / "constraints").exists()


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
    """``convert_newave_case`` surfaces converter findings via ConversionReport."""

    def test_reports_each_warning_once_and_no_log_line(self, tmp_path: Path) -> None:
        from novomodelo_bridge.core import diagnostics as dx
        from novomodelo_bridge.core.conversion import ConversionReport
        from novomodelo_bridge.newave import pipeline
        from novomodelo_bridge.newave.pipeline import convert_newave_case

        def _warning(code: str, summary: str) -> dx.Diagnostic:
            return dx.Diagnostic(
                code=code,
                severity=dx.Severity.WARNING,
                category="Test",
                title=code,
                summary=summary,
            )

        def fake_impl(
            src: Path,
            dst: Path,
            on_phase: object = None,
            *,
            dry_run: bool = False,
            float_digits: int | None = None,
        ) -> ConversionReport:
            dx.emit(_warning("a", "vazpast.dat could not be read"), logger=_LOG)
            dx.emit(_warning("a", "vazpast.dat could not be read"), logger=_LOG)
            logging.getLogger("novomodelo_bridge.newave.fake").warning("log text")
            dx.emit(_warning("b", "ree.dat has no entries"), logger=_LOG)
            return ConversionReport(hydro_count=3)

        with patch.object(pipeline, "_convert_newave_case_impl", side_effect=fake_impl):
            report = convert_newave_case(tmp_path, tmp_path)

        assert report.hydro_count == 3
        assert report.warnings == [
            "vazpast.dat could not be read",
            "ree.dat has no entries",
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

    def test_partial_outputs_cleared_on_failure(self, tmp_path: Path) -> None:
        """A failure partway through the write phase must not leave a partial,
        valid-looking case behind: the known pipeline outputs, and the
        directory the run created for them, are removed."""
        from novomodelo_bridge.newave import pipeline
        from novomodelo_bridge.newave.pipeline import convert_newave_case

        dst = tmp_path / "dst"

        def fake_impl(
            src: Path,
            d: Path,
            on_phase: object = None,
            *,
            dry_run: bool = False,
            float_digits: int | None = None,
        ) -> object:
            # Simulate a write phase that got partway: a top-level JSON and a
            # system/ subdir were written before the failure.
            (d / "system").mkdir(parents=True, exist_ok=True)
            (d / "config.json").write_text("{}", encoding="utf-8")
            (d / "system" / "hydros.json").write_text("{}", encoding="utf-8")
            raise RuntimeError("disk full mid-write")

        with (
            patch.object(pipeline, "_convert_newave_case_impl", side_effect=fake_impl),
            pytest.raises(RuntimeError, match="disk full"),
        ):
            convert_newave_case(tmp_path, dst)

        assert not dst.exists()

    def test_failure_hands_earlier_diagnostics_to_the_callers_sink(
        self,
        newave_mini_deck: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from novomodelo_bridge.core import diagnostics as dx
        from novomodelo_bridge.newave.converters import hydro
        from novomodelo_bridge.newave.pipeline import convert_newave_case

        src = tmp_path / "deck"
        shutil.copytree(newave_mini_deck, src)
        clast = src / "clast.dat"
        clast.write_bytes(b"\t" + clast.read_bytes())

        def fail(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("hydro conversion failed")

        monkeypatch.setattr(hydro, "convert_hydros", fail)
        with (
            dx.collect() as outer,
            pytest.raises(RuntimeError, match="hydro conversion failed"),
        ):
            convert_newave_case(src, tmp_path / "out")

        tab = [d for d in outer if d.code == "source-tab-character"]
        assert len(tab) == 1 and tab[0].table is not None
        assert tab[0].table.rows == [["clast.dat", "1"]]

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
        (dst / "config.json").write_text("{}", encoding="utf-8")
        (dst / "system").mkdir()
        (dst / "system" / "hydros.json").write_text("{}", encoding="utf-8")

        def fake_impl(
            src: Path,
            d: Path,
            on_phase: object = None,
            *,
            dry_run: bool = False,
            float_digits: int | None = None,
        ) -> object:
            raise RuntimeError("boom")

        with (
            patch.object(pipeline, "_convert_newave_case_impl", side_effect=fake_impl),
            pytest.raises(RuntimeError, match="boom"),
        ):
            convert_newave_case(tmp_path, dst, dry_run=True, force=True)

        assert (dst / "config.json").exists()
        assert (dst / "system").exists()

    def test_real_run_against_populated_dst_refuses_without_clearing(
        self, tmp_path: Path
    ) -> None:
        """Without ``force`` a populated ``dst`` is refused before the
        conversion's failure rollback could delete the case it holds."""
        from novomodelo_bridge.core.errors import OutputExistsError
        from novomodelo_bridge.newave.pipeline import convert_newave_case

        dst = tmp_path / "case"
        dst.mkdir()
        existing = dst / "config.json"
        existing.write_bytes(b"keep me")

        with pytest.raises(OutputExistsError, match="already holds files"):
            convert_newave_case(tmp_path / "src", dst)

        assert existing.read_bytes() == b"keep me"


def test_convert_newave_case_threads_on_phase(tmp_path: Path) -> None:
    """``convert_newave_case`` forwards its ``on_phase`` callback to the impl."""
    from novomodelo_bridge.core.conversion import ConversionReport
    from novomodelo_bridge.newave import pipeline
    from novomodelo_bridge.newave.pipeline import convert_newave_case

    received: list[str] = []

    def fake_impl(
        src: Path,
        dst: Path,
        on_phase: Callable[[str], None] | None = None,
        *,
        dry_run: bool = False,
        float_digits: int | None = None,
    ) -> ConversionReport:
        if on_phase is not None:
            on_phase("Discovering files")
        return ConversionReport()

    with patch.object(pipeline, "_convert_newave_case_impl", side_effect=fake_impl):
        convert_newave_case(tmp_path, tmp_path, on_phase=received.append)

    assert received == ["Discovering files"]
