"""Tests of ``comparators.version.operating_points``: the operating points of two cases.

``split_at_points`` runs on literal point frames and plane mappings; the end-to-end
cases read ``build_solved_case`` trees through ``read_side``, rewriting the plane
file with polars where a case needs a plant without planes. The reader wrapping
patches the module's public reader entry point.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

import polars as pl
import pytest

from novomodelo_bridge.comparators.analyze import build_results_dataset
from novomodelo_bridge.comparators.fpha import FPHA_SPLIT_SCHEMA
from novomodelo_bridge.comparators.model import PercentileData
from novomodelo_bridge.comparators.operating_points import (
    FPHA_FIT_INPUTS_SCHEMA,
    FPHA_SLICE_SCHEMA,
    MASKED_HYDRO_SCHEMA,
    TIE_SHARE_SCHEMA,
    OperatingPointComparison,
    operating_point_fields,
)
from novomodelo_bridge.comparators.report_builder import build_comparison_tabs
from novomodelo_bridge.comparators.version import operating_points
from novomodelo_bridge.comparators.version.alignment import AlignedPair, align_by_name
from novomodelo_bridge.comparators.version.operating_points import (
    operating_point_comparison,
    split_at_points,
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

_POINT_SCHEMA = {
    "code": pl.Int64,
    "novomodelo_id": pl.Int64,
    "stage": pl.Int64,
    "scenario": pl.Int64,
    "block": pl.Int64,
    "weight": pl.Float64,
    "v_hm3": pl.Float64,
    "q_m3s": pl.Float64,
    "s_m3s": pl.Float64,
    "outflow_m3s": pl.Float64,
    "gh_mw": pl.Float64,
    "worthless": pl.Boolean,
    "tie": pl.Boolean,
}
_PLANE_SCHEMA = {
    "gamma_0": pl.Float64,
    "gamma_v": pl.Float64,
    "gamma_q": pl.Float64,
    "gamma_s": pl.Float64,
    "kappa": pl.Float64,
}
_NULL_COLUMNS = (
    "rho_exact_at_source",
    "rho_exact_at_novomodelo",
    "fit_error_source",
    "fit_error_novomodelo",
)
_FPHA_FILE = "hydro_models/fpha_hyperplanes.parquet"
_SPLIT_TITLE = "Why the realized productivities differ"

_Planes = Mapping[tuple[int, int], pl.DataFrame]


def _points(
    code: int,
    novomodelo_id: int,
    q: Sequence[float],
    gh: Sequence[float],
    *,
    stage: int = 0,
    tied: Collection[int] = (),
    v_hm3: float = 0.0,
    s_m3s: float = 0.0,
) -> pl.DataFrame:
    """One point per entry of *q*, weight 1.0, block order; *tied* holds the
    positions of the points that are ties."""
    count = len(q)
    return pl.DataFrame(
        {
            "code": [code] * count,
            "novomodelo_id": [novomodelo_id] * count,
            "stage": [stage] * count,
            "scenario": [0] * count,
            "block": list(range(count)),
            "weight": [1.0] * count,
            "v_hm3": [v_hm3] * count,
            "q_m3s": list(q),
            "s_m3s": [s_m3s] * count,
            "outflow_m3s": list(q),
            "gh_mw": list(gh),
            "worthless": [False] * count,
            "tie": [position in tied for position in range(count)],
        },
        schema=_POINT_SCHEMA,
    )


def _planes(*planes: Mapping[str, float], kappa: float = 1.0) -> pl.DataFrame:
    """One row per plane; a coefficient a plane leaves out is zero."""
    rows = [
        {
            "gamma_0": 0.0,
            "gamma_v": 0.0,
            "gamma_q": 0.0,
            "gamma_s": 0.0,
            **plane,
            "kappa": kappa,
        }
        for plane in planes
    ]
    return pl.DataFrame(rows, schema=_PLANE_SCHEMA)


_BASELINE_PLANES: _Planes = {(1, 0): _planes({"gamma_q": 0.08})}
_CANDIDATE_PLANES: _Planes = {(101, 0): _planes({"gamma_q": 0.084})}
_NAMES = {101: "Furnas"}


def _split(
    source: pl.DataFrame | None = None,
    candidate: pl.DataFrame | None = None,
    baseline_planes: _Planes = _BASELINE_PLANES,
    candidate_planes: _Planes = _CANDIDATE_PLANES,
    names: Mapping[int, str] = _NAMES,
) -> pl.DataFrame:
    """Defaults: plant 101 (baseline code 1) at stage 0, two points at 100 and
    200 m3/s, the baseline generating 0.08 MW per m3/s and the candidate 0.084."""
    return split_at_points(
        _points(1, 101, [100.0, 200.0], [8.0, 16.0]) if source is None else source,
        _points(1, 101, [100.0, 200.0], [8.4, 16.8])
        if candidate is None
        else candidate,
        baseline_planes,
        candidate_planes,
        names,
    )


class TestSplitAtPoints:
    def test_the_gap_splits_into_the_model_and_the_operating_point(self) -> None:
        row = _split().row(0, named=True)

        assert row["rho_source"] == pytest.approx(0.08)
        assert row["rho_novomodelo"] == pytest.approx(0.084)
        assert row["rho_source_planes_at_source"] == pytest.approx(0.08)
        assert row["rho_novomodelo_planes_at_source"] == pytest.approx(0.084)
        assert row["d_model"] == pytest.approx(0.05)
        assert row["d_operating_point"] == pytest.approx(0.0)
        assert row["d_total"] == pytest.approx(0.05)

    def test_one_row_per_plant_and_stage_in_the_schema_order(self) -> None:
        split = _split()

        assert split.height == 1
        assert split.columns == list(FPHA_SPLIT_SCHEMA)
        assert split.schema == pl.Schema(FPHA_SPLIT_SCHEMA)

    def test_the_name_comes_from_the_candidate_id(self) -> None:
        row = _split().row(0, named=True)

        assert row["novomodelo_id"] == 101
        assert row["plant_name"] == _NAMES[101]

    def test_the_columns_that_need_the_exact_function_are_null(self) -> None:
        row = _split().row(0, named=True)

        for column in _NULL_COLUMNS:
            assert row[column] is None, column

    def test_the_total_is_the_product_of_the_model_and_operating_point_parts(
        self,
    ) -> None:
        candidate_planes = {(101, 0): _planes({"gamma_0": 10.0, "gamma_q": 0.05})}
        candidate = _points(1, 101, [200.0, 200.0], [20.0, 20.0])

        row = _split(candidate=candidate, candidate_planes=candidate_planes).row(
            0, named=True
        )

        assert row["rho_novomodelo_planes_at_source"] == pytest.approx(35.0 / 300.0)
        assert row["rho_novomodelo"] == pytest.approx(0.1)
        assert row["d_operating_point"] == pytest.approx(0.1 / (35.0 / 300.0) - 1.0)
        assert (1.0 + row["d_model"]) * (1.0 + row["d_operating_point"]) - 1.0 == (
            pytest.approx(row["d_total"])
        )

    def test_a_tie_leaves_the_productivities_and_shows_in_the_tie_share(self) -> None:
        source = _points(1, 101, [100.0, 200.0], [8.0, 16.0], tied={0})
        candidate = _points(1, 101, [100.0, 200.0], [8.4, 16.8], tied={0})

        row = _split(source, candidate).row(0, named=True)

        assert row["tie_share"] == pytest.approx(0.5)
        assert row["q_source_m3s"] == 200.0
        assert row["gh_source_mw"] == 16.0
        assert row["rho_source"] == pytest.approx(0.08)
        assert row["rho_novomodelo_planes_at_source"] == pytest.approx(0.084)

    def test_the_envelope_is_the_minimum_over_the_planes_at_each_point(self) -> None:
        baseline_planes = {
            (1, 0): _planes({"gamma_q": 0.1}, {"gamma_0": 4.0, "gamma_q": 0.05})
        }
        candidate_planes = {
            (101, 0): _planes(
                {"gamma_q": 0.1}, {"gamma_0": 4.0, "gamma_q": 0.05}, kappa=2.0
            )
        }
        source = _points(1, 101, [20.0, 200.0], [2.0, 14.0])

        row = _split(
            source,
            baseline_planes=baseline_planes,
            candidate_planes=candidate_planes,
        ).row(0, named=True)

        assert row["rho_source_planes_at_source"] == pytest.approx(16.0 / 220.0)
        assert row["rho_novomodelo_planes_at_source"] == pytest.approx(20.0 / 220.0)

    def test_the_planes_are_read_at_the_storage_turbined_and_spilled_flows(
        self,
    ) -> None:
        plane = {"gamma_v": 0.002, "gamma_q": 0.08, "gamma_s": -0.01}
        source = _points(1, 101, [100.0], [7.0], v_hm3=500.0, s_m3s=50.0)

        row = _split(
            source,
            baseline_planes={(1, 0): _planes(plane)},
            candidate_planes={(101, 0): _planes(plane)},
        ).row(0, named=True)

        assert row["rho_source"] == pytest.approx(0.07)
        assert row["rho_source_planes_at_source"] == pytest.approx(0.085)
        assert row["rho_novomodelo_planes_at_source"] == pytest.approx(0.085)

    @pytest.mark.parametrize("missing", ["baseline", "candidate"])
    def test_a_plant_without_planes_on_one_side_has_no_row(self, missing: str) -> None:
        source = pl.concat(
            [
                _points(1, 101, [100.0, 200.0], [8.0, 16.0]),
                _points(2, 102, [100.0, 200.0], [8.0, 16.0]),
            ]
        )
        candidate = pl.concat(
            [
                _points(1, 101, [100.0, 200.0], [8.4, 16.8]),
                _points(2, 102, [100.0, 200.0], [8.4, 16.8]),
            ]
        )
        baseline_planes = {**_BASELINE_PLANES, (2, 0): _planes({"gamma_q": 0.08})}
        candidate_planes = {**_CANDIDATE_PLANES, (102, 0): _planes({"gamma_q": 0.084})}
        if missing == "baseline":
            del baseline_planes[(2, 0)]
        else:
            del candidate_planes[(102, 0)]

        split = _split(
            source,
            candidate,
            baseline_planes=baseline_planes,
            candidate_planes=candidate_planes,
            names={101: "Furnas", 102: "Itaipu"},
        )

        assert split["novomodelo_id"].to_list() == [101]

    def test_a_stage_without_planes_leaves_only_that_stage_out(self) -> None:
        source = pl.concat(
            [
                _points(1, 101, [100.0], [8.0], stage=0),
                _points(1, 101, [100.0], [8.0], stage=1),
            ]
        )
        candidate = pl.concat(
            [
                _points(1, 101, [100.0], [8.4], stage=0),
                _points(1, 101, [100.0], [8.4], stage=1),
            ]
        )

        split = _split(source, candidate)

        assert split["stage"].to_list() == [0]

    @pytest.mark.parametrize("missing", ["baseline", "candidate", "both"])
    def test_no_plant_with_planes_gives_the_typed_empty_frame(
        self, missing: str
    ) -> None:
        split = _split(
            baseline_planes={} if missing in ("baseline", "both") else _BASELINE_PLANES,
            candidate_planes={}
            if missing in ("candidate", "both")
            else _CANDIDATE_PLANES,
        )

        assert split.is_empty()
        assert split.schema == pl.Schema(FPHA_SPLIT_SCHEMA)

    def test_points_that_turbine_nothing_give_null_productivities(self) -> None:
        source = _points(1, 101, [0.0, 0.0], [0.0, 0.0])
        candidate = _points(1, 101, [0.0, 0.0], [0.0, 0.0])
        planes = {"gamma_0": 1.0}

        row = _split(
            source,
            candidate,
            baseline_planes={(1, 0): _planes(planes)},
            candidate_planes={(101, 0): _planes(planes)},
        ).row(0, named=True)

        for column in (
            "rho_source",
            "rho_novomodelo",
            "rho_source_planes_at_source",
            "rho_novomodelo_planes_at_source",
            "d_total",
            "d_model",
            "d_operating_point",
        ):
            assert row[column] is None, column
        assert row["tie_share"] == 0.0
        assert row["q_source_m3s"] == 0.0

    def test_every_point_a_tie_gives_null_productivities_and_a_full_tie_share(
        self,
    ) -> None:
        source = _points(1, 101, [100.0, 200.0], [8.0, 16.0], tied={0, 1})
        candidate = _points(1, 101, [100.0, 200.0], [8.4, 16.8], tied={0, 1})

        row = _split(source, candidate).row(0, named=True)

        assert row["tie_share"] == 1.0
        assert row["rho_source"] is None
        assert row["rho_novomodelo"] is None
        assert row["d_total"] is None

    def test_rows_follow_the_candidate_id_then_the_stage(self) -> None:
        source = pl.concat(
            [
                _points(2, 102, [100.0], [8.0], stage=1),
                _points(1, 101, [100.0], [8.0], stage=1),
                _points(1, 101, [100.0], [8.0], stage=0),
            ]
        )
        candidate = source.with_columns(pl.col("gh_mw") * 1.05)
        plane = _planes({"gamma_q": 0.08})

        split = _split(
            source,
            candidate,
            baseline_planes={(1, 0): plane, (1, 1): plane, (2, 1): plane},
            candidate_planes={(101, 0): plane, (101, 1): plane, (102, 1): plane},
            names={101: "Furnas", 102: "Itaipu"},
        )

        assert split.select("novomodelo_id", "stage").rows() == [
            (101, 0),
            (101, 1),
            (102, 1),
        ]

    def test_the_input_frames_are_not_changed(self) -> None:
        source = _points(1, 101, [100.0, 200.0], [8.0, 16.0])
        candidate = _points(1, 101, [100.0, 200.0], [8.4, 16.8])

        _split(source, candidate)

        assert source.columns == list(_POINT_SCHEMA)
        assert candidate.columns == list(_POINT_SCHEMA)


_FURNAS = AlignedPair("FURNAS", 0, 100)
_ITAIPU = AlignedPair("ITAIPU", 1, 101)


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


def _compare(
    baseline: SideInputs,
    candidate: SideInputs,
    stages: Collection[int] = (0, 1),
    pairs: Sequence[AlignedPair] | None = None,
) -> OperatingPointComparison:
    return operating_point_comparison(
        baseline,
        candidate,
        _pairs(baseline, candidate) if pairs is None else pairs,
        stages,
    )


def _assert_default(comparison: OperatingPointComparison) -> None:
    default = OperatingPointComparison()
    for field in dataclasses.fields(OperatingPointComparison):
        frame = getattr(comparison, field.name)
        assert frame.is_empty(), field.name
        assert frame.schema == getattr(default, field.name).schema, field.name


def _drop_planes(side: SideInputs, hydro_id: int) -> None:
    path = side.output_dir / _FPHA_FILE
    pl.read_parquet(path).filter(pl.col("hydro_id") != hydro_id).write_parquet(path)


class TestIdenticalCases:
    def test_the_split_has_a_row_per_plant_and_stage_and_no_gap(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _case_sides(tmp_path)

        split = _compare(baseline, candidate).split

        assert split.height == 4
        assert set(split["novomodelo_id"].to_list()) == {100, 101}
        assert sorted(split["plant_name"].unique().to_list()) == ["FURNAS", "ITAIPU"]
        assert split["d_total"].to_list() == [0.0] * 4
        assert (
            split["rho_novomodelo_planes_at_source"].to_list()
            == split["rho_source_planes_at_source"].to_list()
        )
        assert split["rho_source"].null_count() == 0
        for column in _NULL_COLUMNS:
            assert split[column].null_count() == 4, column

    def test_no_point_is_a_tie(self, tmp_path: Path) -> None:
        baseline, candidate = _case_sides(tmp_path)

        tie_share = _compare(baseline, candidate).tie_share

        assert tie_share.columns == list(TIE_SHARE_SCHEMA)
        assert tie_share["stage"].to_list() == [0, 1]
        assert tie_share["source_tie_share"].to_list() == [0.0, 0.0]
        assert tie_share["novomodelo_tie_share"].to_list() == [0.0, 0.0]

    def test_the_masked_rows_carry_the_baseline_id_and_the_candidate_id(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _case_sides(tmp_path)

        masked = _compare(baseline, candidate).masked_hydro

        assert masked.columns == list(MASKED_HYDRO_SCHEMA)
        assert masked.height == 20
        assert masked["abs_diff"].to_list() == [0.0] * 20
        assert set(masked["newave_code"].to_list()) == {0, 1}
        assert set(masked["novomodelo_id"].to_list()) == {100, 101}
        assert (masked["novomodelo_id"] - masked["newave_code"]).to_list() == [100] * 20
        assert sorted(masked["entity_name"].unique().to_list()) == ["FURNAS", "ITAIPU"]

    def test_the_exact_function_frames_stay_typed_and_empty(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _case_sides(tmp_path)

        comparison = _compare(baseline, candidate)

        assert comparison.fit_inputs.is_empty()
        assert comparison.fit_inputs.columns == list(FPHA_FIT_INPUTS_SCHEMA)
        assert comparison.slices.is_empty()
        assert comparison.slices.columns == list(FPHA_SLICE_SCHEMA)

    def test_the_surface_frames_are_left_to_the_caller(self, tmp_path: Path) -> None:
        baseline, candidate = _case_sides(tmp_path)

        comparison = _compare(baseline, candidate)

        default = OperatingPointComparison()
        for name in ("metrics", "surface", "spill"):
            assert getattr(comparison, name).is_empty(), name
            assert getattr(comparison, name).schema == getattr(default, name).schema

    def test_the_comparison_emits_nothing(self, tmp_path: Path) -> None:
        baseline, candidate = _case_sides(tmp_path)

        with dx.collect() as sink:
            _compare(baseline, candidate)

        assert sink == []


class TestPairing:
    def test_plants_pair_by_name_and_not_by_position(self, tmp_path: Path) -> None:
        baseline, candidate = _case_sides(
            tmp_path,
            SolvedCaseSpec(),
            SolvedCaseSpec(hydros=("ITAIPU", "FURNAS"), id_offset=100),
        )

        comparison = _compare(baseline, candidate)

        names = dict(comparison.split.select("novomodelo_id", "plant_name").iter_rows())
        assert names == {100: "ITAIPU", 101: "FURNAS"}
        furnas = comparison.masked_hydro.filter(
            (pl.col("entity_name") == "FURNAS")
            & (pl.col("variable") == "generation_mw")
            & (pl.col("stage") == 0)
        ).row(0, named=True)
        assert furnas["newave_code"] == 0
        assert furnas["novomodelo_id"] == 101
        assert furnas["newave_value"] == pytest.approx(410.0)
        assert furnas["novomodelo_value"] == pytest.approx(820.0)

    def test_each_plant_is_set_against_the_planes_of_its_own_id(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _case_sides(
            tmp_path,
            SolvedCaseSpec(),
            SolvedCaseSpec(hydros=("ITAIPU", "FURNAS"), id_offset=100),
        )

        split = _compare(baseline, candidate).split

        assert split.height == 4
        assert (
            split["rho_novomodelo_planes_at_source"]
            != split["rho_source_planes_at_source"]
        ).all()

    def test_a_plant_of_the_baseline_only_has_no_row_anywhere(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _case_sides(
            tmp_path,
            SolvedCaseSpec(),
            SolvedCaseSpec(hydros=("FURNAS",), id_offset=100),
        )

        comparison = _compare(baseline, candidate)

        assert comparison.split["novomodelo_id"].unique().to_list() == [100]
        assert comparison.masked_hydro["novomodelo_id"].unique().to_list() == [100]

    def test_no_pairs_give_the_default_comparison(self, tmp_path: Path) -> None:
        baseline, candidate = _case_sides(tmp_path)

        _assert_default(_compare(baseline, candidate, pairs=()))


class TestStages:
    def test_a_single_stage_leaves_the_others_out_of_every_frame(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _case_sides(tmp_path)

        comparison = _compare(baseline, candidate, stages=[1])

        assert comparison.split["stage"].unique().to_list() == [1]
        assert comparison.tie_share["stage"].to_list() == [1]
        assert comparison.masked_hydro["stage"].unique().to_list() == [1]

    @pytest.mark.parametrize("stages", [[], [5]])
    def test_stages_without_points_give_the_default_comparison(
        self, tmp_path: Path, stages: list[int]
    ) -> None:
        baseline, candidate = _case_sides(tmp_path)

        _assert_default(_compare(baseline, candidate, stages=stages))

    def test_differing_scenario_counts_are_logged_by_the_shared_alignment_only(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        baseline, candidate = _case_sides(
            tmp_path,
            SolvedCaseSpec(scenarios=2),
            SolvedCaseSpec(scenarios=3, id_offset=100),
        )

        with caplog.at_level(logging.INFO), dx.collect() as sink:
            comparison = _compare(baseline, candidate)

        assert "2 source-model scenarios against 3" in caplog.text
        assert sink == []
        assert comparison.split.height == 4


class TestTies:
    def test_prices_below_the_zero_threshold_tie_every_point(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _case_sides(
            tmp_path,
            SolvedCaseSpec(scale=1.0e-6),
            SolvedCaseSpec(scale=1.0e-6, id_offset=100),
        )

        comparison = _compare(baseline, candidate)

        assert comparison.tie_share["source_tie_share"].to_list() == [1.0, 1.0]
        assert comparison.tie_share["novomodelo_tie_share"].to_list() == [1.0, 1.0]
        assert comparison.masked_hydro.is_empty()
        assert comparison.masked_hydro.columns == list(MASKED_HYDRO_SCHEMA)
        assert comparison.split.height == 4
        assert comparison.split["tie_share"].to_list() == [1.0] * 4
        assert comparison.split["rho_source"].to_list() == [None] * 4


class TestMissingPlanes:
    @pytest.mark.parametrize("missing", ["baseline", "candidate"])
    def test_a_side_without_planes_leaves_the_split_empty_and_the_rest_filled(
        self, tmp_path: Path, missing: str
    ) -> None:
        baseline, candidate = _case_sides(
            tmp_path,
            SolvedCaseSpec(fpha_planes=missing != "baseline"),
            SolvedCaseSpec(fpha_planes=missing != "candidate", id_offset=100),
        )

        comparison = _compare(baseline, candidate)

        assert comparison.split.is_empty()
        assert comparison.split.schema == pl.Schema(FPHA_SPLIT_SCHEMA)
        assert comparison.tie_share.height == 2
        assert comparison.masked_hydro.height == 20

    @pytest.mark.parametrize("missing", ["baseline", "candidate"])
    def test_a_plant_without_planes_leaves_the_split_only(
        self, tmp_path: Path, missing: str
    ) -> None:
        baseline, candidate = _case_sides(tmp_path)
        if missing == "baseline":
            _drop_planes(baseline, 0)
        else:
            _drop_planes(candidate, 100)

        comparison = _compare(baseline, candidate)

        assert comparison.split["plant_name"].unique().to_list() == ["ITAIPU"]
        assert sorted(comparison.masked_hydro["entity_name"].unique().to_list()) == [
            "FURNAS",
            "ITAIPU",
        ]
        assert comparison.tie_share.height == 2


class TestCasesWithoutPoints:
    def test_a_baseline_without_hydros_gives_the_default_comparison(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _case_sides(
            tmp_path, SolvedCaseSpec(hydros=()), SolvedCaseSpec(id_offset=100)
        )

        with dx.collect() as sink:
            comparison = _compare(baseline, candidate, pairs=[_FURNAS, _ITAIPU])

        _assert_default(comparison)
        assert sink == []

    def test_a_candidate_without_hydros_gives_the_default_comparison(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _case_sides(
            tmp_path, SolvedCaseSpec(), SolvedCaseSpec(hydros=(), id_offset=100)
        )

        with dx.collect() as sink:
            comparison = _compare(baseline, candidate, pairs=[_FURNAS, _ITAIPU])

        _assert_default(comparison)
        assert sink == []

    def test_cases_without_hydros_give_the_default_comparison(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _case_sides(
            tmp_path,
            SolvedCaseSpec(hydros=()),
            SolvedCaseSpec(hydros=(), id_offset=100),
        )

        _assert_default(_compare(baseline, candidate))

    @pytest.mark.parametrize("damaged", ["baseline", "candidate"])
    @pytest.mark.parametrize("target", [_FPHA_FILE, "simulation/hydros"])
    def test_an_unreadable_file_propagates(
        self, tmp_path: Path, damaged: str, target: str
    ) -> None:
        baseline, candidate = _case_sides(tmp_path)
        corrupt_file(
            (baseline if damaged == "baseline" else candidate).output_dir, target
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

        def emitting(output_dir: Path) -> pl.DataFrame | None:
            dx.emit(diagnostic, logger=_LOG)
            return None

        monkeypatch.setattr(operating_points, "read_novomodelo_fpha_planes", emitting)

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

        def failing(output_dir: Path) -> pl.DataFrame | None:
            dx.emit(diagnostic, logger=_LOG)
            raise NovomodeloReadError("unreadable")

        monkeypatch.setattr(operating_points, "read_novomodelo_fpha_planes", failing)

        with dx.collect() as sink, pytest.raises(NovomodeloReadError):
            _compare(baseline, candidate)

        assert [d.summary for d in sink] == [
            "Baseline output: The file could not be read."
        ]


class TestReport:
    def test_the_frames_render_on_the_productivity_tab(self, tmp_path: Path) -> None:
        baseline, candidate = _case_sides(tmp_path)
        comparison = _compare(baseline, candidate)

        dataset = build_results_dataset(
            [], PercentileData(**operating_point_fields(comparison)), 0.05
        )
        tabs = build_comparison_tabs(dataset)

        assert _SPLIT_TITLE in tabs["tab-productivity"]

    def test_the_default_comparison_renders_without_the_split(
        self, tmp_path: Path
    ) -> None:
        baseline, candidate = _case_sides(tmp_path)
        comparison = _compare(baseline, candidate, stages=[])

        dataset = build_results_dataset(
            [], PercentileData(**operating_point_fields(comparison)), 0.05
        )
        tabs = build_comparison_tabs(dataset)

        assert _SPLIT_TITLE not in tabs["tab-productivity"]
