"""Tests of ``comparators.version.production``: the fitted production surface of two cases.

``surface_frames`` runs on literal plane frames and range maps; the end-to-end cases
read ``build_solved_case`` trees through ``read_side``, rewriting ``hydros.json``
and the plane file with the standard library and polars where a case needs
different limits or planes. The reader wrapping patches the module's public
reader entry points.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl
import pytest

from novomodelo_bridge.comparators.fpha import (
    FPHA_METRICS_SCHEMA,
    FPHA_SPILL_SCHEMA,
    FPHA_SURFACE_SCHEMA,
)
from novomodelo_bridge.comparators.version import production
from novomodelo_bridge.comparators.version.alignment import AlignedPair, align_by_name
from novomodelo_bridge.comparators.version.production import (
    PlantRange,
    fpha_surface_comparison,
    surface_frames,
)
from novomodelo_bridge.comparators.version.sides import SideInputs, read_side
from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.novomodelo.readers import NovomodeloReadError
from tests.novomodelo_output_builders import (
    SolvedCaseSpec,
    build_solved_case,
    corrupt_file,
)

_LOG = logging.getLogger(__name__)

_RANGE = PlantRange(100.0, 5000.0, 12000.0)
_FURNAS = AlignedPair("Furnas", 3, 7)
_ITAIPU = AlignedPair("Itaipu", 4, 8)
_PLANE_SCHEMA = {
    "hydro_id": pl.Int32,
    "stage_id": pl.Int32,
    "plane_id": pl.Int32,
    "gamma_0": pl.Float64,
    "gamma_v": pl.Float64,
    "gamma_q": pl.Float64,
    "gamma_s": pl.Float64,
    "kappa": pl.Float64,
}
_SCHEMAS = (FPHA_METRICS_SCHEMA, FPHA_SURFACE_SCHEMA, FPHA_SPILL_SCHEMA)
_FPHA_FILE = "hydro_models/fpha_hyperplanes.parquet"

_Frames = tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]


def _planes(
    hydro_id: int,
    stages: Sequence[int],
    *,
    gamma_0: float | Mapping[int, float] = 10.0,
    kappa: float = 1.0,
    gamma_v: float = 0.001,
) -> pl.DataFrame:
    """One plane per stage; *gamma_0* is a value for every stage or a value by stage."""
    intercepts = [
        gamma_0[stage] if isinstance(gamma_0, Mapping) else gamma_0 for stage in stages
    ]
    count = len(stages)
    return pl.DataFrame(
        {
            "hydro_id": [hydro_id] * count,
            "stage_id": list(stages),
            "plane_id": [0] * count,
            "gamma_0": intercepts,
            "gamma_v": [gamma_v] * count,
            "gamma_q": [0.08] * count,
            "gamma_s": [-0.01] * count,
            "kappa": [kappa] * count,
        },
        schema=_PLANE_SCHEMA,
    )


def _frames(
    baseline: pl.DataFrame | None,
    candidate: pl.DataFrame | None,
    *,
    baseline_ranges: Mapping[int, PlantRange] | None = None,
    candidate_ranges: Mapping[int, PlantRange] | None = None,
    pairs: Sequence[AlignedPair] = (_FURNAS,),
    stages: Sequence[int] = (0,),
) -> _Frames:
    return surface_frames(
        baseline,
        candidate,
        {3: _RANGE, 4: _RANGE} if baseline_ranges is None else baseline_ranges,
        {7: _RANGE, 8: _RANGE} if candidate_ranges is None else candidate_ranges,
        pairs,
        stages,
    )


def _stages_of(frame: pl.DataFrame) -> list[int]:
    return sorted(frame["stage"].unique().to_list())


def _assert_empty_with_schemas(frames: _Frames) -> None:
    for frame, schema in zip(frames, _SCHEMAS, strict=True):
        assert frame.is_empty()
        assert frame.columns == list(schema)


class TestReencoding:
    def test_kappa_folds_into_the_intercept(self) -> None:
        baseline = _planes(3, [0], gamma_0=20.0, kappa=0.5)
        candidate = _planes(7, [0], gamma_0=10.0, kappa=1.0)

        metrics, _, _ = _frames(baseline, candidate)

        assert metrics["max_abs_dev"].to_list() == [0.0]
        assert metrics["nmae"].to_list() == [0.0]
        assert metrics["bias"].to_list() == [0.0]
        assert metrics["gh_max_ratio"].to_list() == [1.0]

    def test_a_higher_candidate_intercept_shows_as_a_positive_deviation(self) -> None:
        baseline = _planes(3, [0], gamma_0=20.0, kappa=0.5)
        candidate = _planes(7, [0], gamma_0=15.0, kappa=1.0)

        metrics, _, _ = _frames(baseline, candidate)

        assert metrics["max_abs_dev"].item() == pytest.approx(5.0)
        assert metrics["bias"].item() > 0.0

    def test_a_kappa_on_the_candidate_scales_its_intercept(self) -> None:
        baseline = _planes(3, [0], gamma_0=10.0)
        candidate = _planes(7, [0], gamma_0=10.0, kappa=1.5)

        metrics, _, _ = _frames(baseline, candidate)

        assert metrics["max_abs_dev"].item() == pytest.approx(5.0)

    def test_the_volume_is_taken_absolute_on_both_sides(self) -> None:
        planes = _planes(3, [0], gamma_v=0.001)

        metrics, _, _ = _frames(
            planes,
            planes.with_columns(pl.lit(7, dtype=pl.Int32).alias("hydro_id")),
        )

        assert metrics["max_abs_dev"].to_list() == [0.0]

    def test_the_planes_of_each_side_are_counted(self) -> None:
        baseline = _planes(3, [0, 0, 0])
        candidate = _planes(7, [0, 0])

        metrics, _, _ = _frames(baseline, candidate)

        assert metrics["n_planes_newave"].to_list() == [3]
        assert metrics["n_planes_novomodelo"].to_list() == [2]


class TestOutputFrames:
    def test_the_ids_names_and_sources_are_the_candidate_id_the_baseline_name_and_the_chart_labels(
        self,
    ) -> None:
        metrics, surface, spill = _frames(_planes(3, [0]), _planes(7, [0]))

        assert metrics["novomodelo_id"].to_list() == [7]
        assert metrics["plant_name"].to_list() == ["Furnas"]
        for frame in (surface, spill):
            assert frame["novomodelo_id"].unique().to_list() == [7]
            assert frame["plant_name"].unique().to_list() == ["Furnas"]
            assert sorted(frame["source"].unique().to_list()) == [
                "newave",
                "novomodelo",
            ]

    def test_the_frames_keep_the_shared_schemas(self) -> None:
        frames = _frames(_planes(3, [0]), _planes(7, [0]))

        for frame, schema in zip(frames, _SCHEMAS, strict=True):
            assert dict(frame.schema) == schema

    def test_the_surface_draws_five_volume_nodes_by_five_flow_nodes(self) -> None:
        metrics, surface, _ = _frames(_planes(3, [0]), _planes(7, [0]))

        assert metrics["n_v"].to_list() == [5]
        assert surface.height == 2 * 25
        assert surface["v_hm3"].n_unique() == 5
        assert surface["q_m3s"].n_unique() == 5

    def test_two_plants_are_ordered_by_candidate_id_then_stage(self) -> None:
        baseline = pl.concat([_planes(4, [0, 1]), _planes(3, [0, 1])])
        candidate = pl.concat([_planes(8, [0, 1]), _planes(7, [0, 1])])

        metrics, surface, spill = _frames(
            baseline, candidate, pairs=(_ITAIPU, _FURNAS), stages=(0, 1)
        )

        assert metrics.select("novomodelo_id", "stage").rows() == [
            (7, 0),
            (7, 1),
            (8, 0),
            (8, 1),
        ]
        for frame in (surface, spill):
            keys = frame.select("novomodelo_id", "stage").rows()
            assert keys == sorted(keys)


class TestStages:
    def test_each_frame_holds_the_real_stage_ids(self) -> None:
        baseline = _planes(3, [2, 3])
        candidate = _planes(7, [2, 3])

        frames = _frames(baseline, candidate, stages=(2, 3))

        for frame in frames:
            assert _stages_of(frame) == [2, 3]

    def test_a_subset_of_the_stages_leaves_the_others_out(self) -> None:
        baseline = _planes(3, [2, 3])
        candidate = _planes(7, [2, 3])

        frames = _frames(baseline, candidate, stages=(3,))

        for frame in frames:
            assert _stages_of(frame) == [3]

    def test_equal_stage_ids_are_paired_when_the_stages_are_not_contiguous(
        self,
    ) -> None:
        intercepts = {0: 10.0, 1: 20.0, 2: 30.0, 4: 50.0}
        baseline = _planes(3, [0, 1, 2, 4], gamma_0=intercepts)
        candidate = _planes(7, [0, 1, 2, 4], gamma_0={**intercepts, 4: 52.0})

        metrics, _, _ = _frames(baseline, candidate, stages=(1, 4))

        assert metrics["stage"].to_list() == [1, 4]
        assert metrics["max_abs_dev"][0] == 0.0
        assert metrics["max_abs_dev"][1] == pytest.approx(2.0)

    def test_a_stage_held_on_one_side_only_is_not_compared(self) -> None:
        baseline = _planes(3, [0, 1])
        candidate = _planes(7, [1, 2])

        frames = _frames(baseline, candidate, stages=(0, 1, 2))

        for frame in frames:
            assert _stages_of(frame) == [1]

    def test_a_stage_outside_the_given_stages_is_not_compared(self) -> None:
        planes = _planes(3, [0, 5])

        frames = _frames(planes, planes.with_columns(pl.lit(7).alias("hydro_id")))

        for frame in frames:
            assert _stages_of(frame) == [0]

    def test_a_stage_without_planes_leaves_the_ranks_of_the_others_intact(
        self,
    ) -> None:
        baseline = _planes(3, [0, 2], gamma_0={0: 10.0, 2: 30.0})
        candidate = _planes(7, [0, 2], gamma_0={0: 10.0, 2: 30.0})

        metrics, _, _ = _frames(baseline, candidate, stages=(0, 1, 2))

        assert metrics["stage"].to_list() == [0, 2]
        assert metrics["max_abs_dev"].to_list() == [0.0, 0.0]

    def test_empty_stages_give_empty_frames(self) -> None:
        planes = _planes(3, [0])

        _assert_empty_with_schemas(_frames(planes, _planes(7, [0]), stages=()))


class TestRanges:
    def test_the_grid_spans_the_intersection_of_both_ranges(self) -> None:
        _, surface, _ = _frames(
            _planes(3, [0]),
            _planes(7, [0]),
            baseline_ranges={3: PlantRange(100.0, 5000.0, 12000.0)},
            candidate_ranges={7: PlantRange(200.0, 4000.0, 9000.0)},
        )

        assert surface["v_hm3"].min() == 200.0
        assert surface["v_hm3"].max() == 4000.0
        assert surface["q_m3s"].min() == 0.0
        assert surface["q_m3s"].max() == 9000.0

    def test_the_intersection_is_symmetric_in_the_two_cases(self) -> None:
        _, surface, _ = _frames(
            _planes(3, [0]),
            _planes(7, [0]),
            baseline_ranges={3: PlantRange(200.0, 4000.0, 9000.0)},
            candidate_ranges={7: PlantRange(100.0, 5000.0, 12000.0)},
        )

        assert surface["v_hm3"].min() == 200.0
        assert surface["v_hm3"].max() == 4000.0
        assert surface["q_m3s"].max() == 9000.0

    def test_storage_intervals_that_do_not_overlap_leave_the_plant_out(self) -> None:
        frames = _frames(
            _planes(3, [0]),
            _planes(7, [0]),
            baseline_ranges={3: PlantRange(100.0, 200.0, 12000.0)},
            candidate_ranges={7: PlantRange(300.0, 400.0, 12000.0)},
        )

        _assert_empty_with_schemas(frames)

    def test_a_zero_turbined_limit_leaves_the_plant_out(self) -> None:
        frames = _frames(
            _planes(3, [0]),
            _planes(7, [0]),
            baseline_ranges={3: PlantRange(100.0, 5000.0, 0.0)},
            candidate_ranges={7: _RANGE},
        )

        _assert_empty_with_schemas(frames)

    def test_a_run_of_river_plant_has_one_volume_node(self) -> None:
        metrics, surface, _ = _frames(
            _planes(3, [0]),
            _planes(7, [0]),
            baseline_ranges={3: PlantRange(0.0, 0.0, 12000.0)},
            candidate_ranges={7: PlantRange(0.0, 0.0, 12000.0)},
        )

        assert metrics["n_v"].to_list() == [1]
        assert surface.height == 2 * 5

    def test_storage_intervals_that_touch_give_one_volume_node(self) -> None:
        metrics, surface, _ = _frames(
            _planes(3, [0]),
            _planes(7, [0]),
            baseline_ranges={3: PlantRange(100.0, 200.0, 12000.0)},
            candidate_ranges={7: PlantRange(200.0, 400.0, 12000.0)},
        )

        assert metrics["n_v"].to_list() == [1]
        assert surface["v_hm3"].unique().to_list() == [200.0]

    @pytest.mark.parametrize("side", ["baseline", "candidate"])
    def test_a_paired_plant_missing_from_a_range_map_is_left_out(
        self, side: str
    ) -> None:
        baseline = pl.concat([_planes(3, [0]), _planes(4, [0])])
        candidate = pl.concat([_planes(7, [0]), _planes(8, [0])])
        ranges = {
            "baseline": ({4: _RANGE}, {7: _RANGE, 8: _RANGE}),
            "candidate": ({3: _RANGE, 4: _RANGE}, {8: _RANGE}),
        }[side]

        metrics, _, _ = _frames(
            baseline,
            candidate,
            baseline_ranges=ranges[0],
            candidate_ranges=ranges[1],
            pairs=(_FURNAS, _ITAIPU),
        )

        assert metrics["novomodelo_id"].to_list() == [8]

    def test_ranges_are_read_by_the_id_of_their_own_side(self) -> None:
        metrics, surface, _ = _frames(
            _planes(3, [0]),
            _planes(7, [0]),
            baseline_ranges={7: PlantRange(0.0, 1.0, 1.0), 3: _RANGE},
            candidate_ranges={3: PlantRange(0.0, 1.0, 1.0), 7: _RANGE},
        )

        assert metrics["n_v"].to_list() == [5]
        assert surface["v_hm3"].max() == 5000.0


class TestWhatIsCompared:
    def test_a_plant_fitted_on_one_side_only_has_no_row(self) -> None:
        baseline = pl.concat([_planes(3, [0]), _planes(4, [0])])
        candidate = _planes(7, [0])

        frames = _frames(baseline, candidate, pairs=(_FURNAS, _ITAIPU))

        for frame in frames:
            assert frame["novomodelo_id"].unique().to_list() == [7]

    def test_a_plant_fitted_on_the_candidate_only_has_no_row(self) -> None:
        baseline = _planes(3, [0])
        candidate = pl.concat([_planes(7, [0]), _planes(8, [0])])

        frames = _frames(baseline, candidate, pairs=(_FURNAS, _ITAIPU))

        for frame in frames:
            assert frame["novomodelo_id"].unique().to_list() == [7]

    def test_a_plant_without_a_pair_has_no_row(self) -> None:
        baseline = pl.concat([_planes(3, [0]), _planes(9, [0])])
        candidate = pl.concat([_planes(7, [0]), _planes(9, [0])])

        frames = _frames(baseline, candidate, baseline_ranges={3: _RANGE, 9: _RANGE})

        for frame in frames:
            assert frame["novomodelo_id"].unique().to_list() == [7]

    def test_a_pair_that_nothing_is_fitted_for_gives_empty_frames(self) -> None:
        frames = _frames(_planes(9, [0]), _planes(9, [0]))

        _assert_empty_with_schemas(frames)

    def test_nothing_is_emitted_for_a_plant_that_is_not_compared(self) -> None:
        baseline = pl.concat([_planes(3, [0]), _planes(4, [0]), _planes(9, [0])])
        candidate = pl.concat([_planes(7, [0]), _planes(9, [0])])

        with dx.collect() as sink:
            _frames(
                baseline,
                candidate,
                baseline_ranges={3: _RANGE, 4: _RANGE, 9: _RANGE},
                pairs=(_FURNAS, _ITAIPU),
            )

        assert sink == []


class TestEmptyInputs:
    @pytest.mark.parametrize("missing", ["baseline", "candidate", "both"])
    def test_a_side_without_planes_gives_empty_frames(self, missing: str) -> None:
        baseline = None if missing in ("baseline", "both") else _planes(3, [0])
        candidate = None if missing in ("candidate", "both") else _planes(7, [0])

        _assert_empty_with_schemas(_frames(baseline, candidate))

    @pytest.mark.parametrize("missing", ["baseline", "candidate", "both"])
    def test_a_side_with_an_empty_plane_frame_gives_empty_frames(
        self, missing: str
    ) -> None:
        empty = pl.DataFrame(schema=_PLANE_SCHEMA)
        baseline = empty if missing in ("baseline", "both") else _planes(3, [0])
        candidate = empty if missing in ("candidate", "both") else _planes(7, [0])

        _assert_empty_with_schemas(_frames(baseline, candidate))

    def test_no_pairs_give_empty_frames(self) -> None:
        _assert_empty_with_schemas(_frames(_planes(3, [0]), _planes(7, [0]), pairs=()))

    def test_empty_range_maps_give_empty_frames(self) -> None:
        frames = _frames(
            _planes(3, [0]),
            _planes(7, [0]),
            baseline_ranges={},
            candidate_ranges={},
        )

        _assert_empty_with_schemas(frames)

    def test_empty_inputs_raise_nothing_and_emit_nothing(self) -> None:
        with dx.collect() as sink:
            surface_frames(None, None, {}, {}, [], [])

        assert sink == []


def _pairs(baseline: SideInputs, candidate: SideInputs) -> list[AlignedPair]:
    return list(align_by_name("hydro", baseline.hydros, candidate.hydros).pairs)


def _case_sides(
    tmp_path: Path,
    baseline_spec: SolvedCaseSpec = SolvedCaseSpec(),
    candidate_spec: SolvedCaseSpec = SolvedCaseSpec(id_offset=100),
) -> tuple[SideInputs, SideInputs]:
    baseline = read_side(
        build_solved_case(tmp_path / "baseline", baseline_spec), "baseline"
    )
    candidate = read_side(
        build_solved_case(tmp_path / "candidate", candidate_spec), "candidate"
    )
    return baseline, candidate


def _compare(baseline: SideInputs, candidate: SideInputs) -> _Frames:
    return fpha_surface_comparison(
        baseline, candidate, _pairs(baseline, candidate), [0, 1]
    )


def _set_limits(
    side: SideInputs,
    position: int,
    *,
    max_storage_hm3: float | None = None,
    max_turbined_m3s: float | None = None,
) -> None:
    path = side.output_dir.parent / "system" / "hydros.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    hydro = document["hydros"][position]
    if max_storage_hm3 is not None:
        hydro["reservoir"]["max_storage_hm3"] = max_storage_hm3
    if max_turbined_m3s is not None:
        hydro["generation"]["max_turbined_m3s"] = max_turbined_m3s
    path.write_text(json.dumps(document), encoding="utf-8")


class TestSolvedCases:
    def test_identical_cases_with_different_ids_agree_everywhere(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _case_sides(tmp_path)

        metrics, surface, spill = _compare(baseline, candidate)

        assert metrics.height == 4
        assert set(metrics["novomodelo_id"].to_list()) == {100, 101}
        assert metrics["n_planes_newave"].to_list() == [2] * 4
        assert metrics["n_planes_novomodelo"].to_list() == [2] * 4
        assert metrics["n_v"].to_list() == [5] * 4
        assert metrics["max_abs_dev"].to_list() == [0.0] * 4
        assert metrics["nmae"].to_list() == [0.0] * 4
        assert metrics["bias"].to_list() == [0.0] * 4
        assert metrics["gh_max_ratio"].to_list() == [1.0] * 4
        assert spill["stage"].unique().sort().to_list() == [0, 1]
        counts = surface.group_by("novomodelo_id", "stage", "source").len()
        assert counts.height == 8
        assert counts["len"].to_list() == [25] * 8
        assert sorted(metrics["plant_name"].unique().to_list()) == [
            "FURNAS",
            "ITAIPU",
        ]

    def test_a_clean_comparison_emits_nothing(self, tmp_path: Path) -> None:
        baseline, candidate = _case_sides(tmp_path)

        with dx.collect() as sink:
            _compare(baseline, candidate)

        assert sink == []

    def test_the_ranges_are_the_static_limits_of_each_sides_own_plant(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _case_sides(tmp_path)
        _set_limits(candidate, 0, max_storage_hm3=4000.0)
        _set_limits(baseline, 1, max_turbined_m3s=9000.0)

        _, surface, _ = _compare(baseline, candidate)

        furnas = surface.filter(pl.col("plant_name") == "FURNAS")
        itaipu = surface.filter(pl.col("plant_name") == "ITAIPU")
        assert furnas["v_hm3"].min() == 100.0
        assert furnas["v_hm3"].max() == 4000.0
        assert furnas["q_m3s"].max() == 12000.0
        assert itaipu["v_hm3"].max() == 5000.0
        assert itaipu["q_m3s"].max() == 9000.0

    def test_a_baseline_without_planes_gives_empty_frames(self, tmp_path: Path) -> None:
        baseline, candidate = _case_sides(
            tmp_path, SolvedCaseSpec(fpha_planes=False), SolvedCaseSpec(id_offset=100)
        )

        _assert_empty_with_schemas(_compare(baseline, candidate))

    def test_a_candidate_without_planes_gives_empty_frames(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _case_sides(
            tmp_path, SolvedCaseSpec(), SolvedCaseSpec(id_offset=100, fpha_planes=False)
        )

        _assert_empty_with_schemas(_compare(baseline, candidate))

    def test_cases_without_hydros_give_empty_frames(self, tmp_path: Path) -> None:
        spec = SolvedCaseSpec(hydros=())
        baseline, candidate = _case_sides(
            tmp_path, spec, SolvedCaseSpec(hydros=(), id_offset=100)
        )

        with dx.collect() as sink:
            frames = _compare(baseline, candidate)

        _assert_empty_with_schemas(frames)
        assert sink == []

    def test_a_plant_of_the_candidate_only_has_no_row(self, tmp_path: Path) -> None:
        baseline, candidate = _case_sides(
            tmp_path,
            SolvedCaseSpec(hydros=("FURNAS",)),
            SolvedCaseSpec(hydros=("FURNAS", "ITAIPU"), id_offset=100),
        )

        metrics, _, _ = _compare(baseline, candidate)

        assert metrics["plant_name"].unique().to_list() == ["FURNAS"]
        assert metrics["novomodelo_id"].unique().to_list() == [100]

    def test_a_paired_plant_with_planes_on_the_candidate_only_has_no_row(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _case_sides(tmp_path)
        path = baseline.output_dir / _FPHA_FILE
        pl.read_parquet(path).filter(pl.col("hydro_id") == 0).write_parquet(path)

        metrics, surface, spill = _compare(baseline, candidate)

        assert metrics["plant_name"].unique().to_list() == ["FURNAS"]
        assert surface["plant_name"].unique().to_list() == ["FURNAS"]
        assert spill["plant_name"].unique().to_list() == ["FURNAS"]

    def test_a_stage_the_cases_do_not_share_is_not_compared(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _case_sides(
            tmp_path, SolvedCaseSpec(stages=1), SolvedCaseSpec(stages=2, id_offset=100)
        )

        frames = _compare(baseline, candidate)

        for frame in frames:
            assert _stages_of(frame) == [0]

    @pytest.mark.parametrize("damaged", ["baseline", "candidate"])
    def test_an_unreadable_plane_file_propagates(
        self, tmp_path: Path, damaged: str
    ) -> None:
        baseline, candidate = _case_sides(tmp_path)
        corrupt_file(
            (baseline if damaged == "baseline" else candidate).output_dir, _FPHA_FILE
        )

        with pytest.raises(NovomodeloReadError):
            _compare(baseline, candidate)


class TestReaderDiagnostics:
    @staticmethod
    def _diagnostic() -> dx.Diagnostic:
        return dx.Diagnostic(
            code="report-file-unreadable",
            severity=dx.Severity.WARNING,
            category="Case and result files",
            title="x.json could not be read",
            summary="The file could not be read.",
            table=dx.DiagnosticTable(columns=["File"], rows=[["x.json"]]),
            notes=["Path: x.json"],
            remediation="Check x.json.",
        )

    def test_each_sides_reader_diagnostics_are_tagged_with_the_side(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        baseline, candidate = _case_sides(tmp_path)
        diagnostic = self._diagnostic()

        def emitting(output_dir: Path) -> dict[int, dict]:
            dx.emit(diagnostic, logger=_LOG)
            return {}

        monkeypatch.setattr(production, "read_novomodelo_hydro_metadata", emitting)

        with dx.collect() as sink:
            _compare(baseline, candidate)

        assert [d.summary for d in sink] == [
            "Baseline output: The file could not be read.",
            "Candidate output: The file could not be read.",
        ]

    def test_a_reader_failure_still_reaches_the_sink_before_it_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        baseline, candidate = _case_sides(tmp_path)
        diagnostic = self._diagnostic()

        def failing(output_dir: Path) -> dict[int, dict]:
            dx.emit(diagnostic, logger=_LOG)
            raise NovomodeloReadError("unreadable")

        monkeypatch.setattr(production, "read_novomodelo_hydro_metadata", failing)

        with dx.collect() as sink, pytest.raises(NovomodeloReadError):
            _compare(baseline, candidate)

        assert [d.summary for d in sink] == [
            "Baseline output: The file could not be read."
        ]
