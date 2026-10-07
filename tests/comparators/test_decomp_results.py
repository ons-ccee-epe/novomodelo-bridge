"""Core kernels + dataset-build tests for ``comparators.decomp.results``.

First carve out of the legacy ``test_decomp_results_compare.py`` mega file:
the pure-function kernels (``_stage_rows``/``_scenario_mean``/
``_map_entities``/``_result_comparisons``/``_bus_side``) plus
``build_decomp_dataset``'s base dataset-assembly, shared-case-build, and
single-parse tests. The remaining concern bands (network, energy balance,
costs, performance, productivity, FPHA, REE, evaporation, constraints, CLI)
stay in the mega file pending their own carve.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import polars as pl
import pytest

from novomodelo_bridge.comparators.decomp.results import (
    _BUS_VARIABLES,
    _CANONICAL_VARIABLE,
    _HYDRO_VARIABLES,
    _THERMAL_VARIABLES,
    _AlignedDecompFrames,
    _bus_side,
    _map_entities,
    _result_comparisons,
    _scenario_mean,
    _stage_rows,
    _weighted_group_mean,
    build_decomp_dataset,
)
from novomodelo_bridge.core.errors import FieldParseError, SourceFileError
from novomodelo_bridge.decomp.files import DecompFiles
from novomodelo_bridge.decomp.id_map import DecompIdMap
from tests.comparators.conftest import (
    _aligned_fixture,
    _no_dec_oper,
    _patch_aligned_frames,
    _patch_ree_sources,
    _patch_shared_case,
    _ree_aligned_fixture,
    _ree_id_map,
    _usih_frame,
    _write_generic_constraints_case,
)
from tests.conftest import _FakeDadger as _ConstraintFakeDadger


def _source_frame() -> pl.DataFrame:
    """Two stages of one plant: per-block rows plus the stage-aggregate row."""
    return pl.DataFrame(
        {
            "estagio": [1, 1, 1, 2, 2, 2],
            "no": [1, 1, 1, 2, 2, 2],
            "patamar": [1.0, 2.0, None, 1.0, 2.0, None],
            "duracao": [24.0, 144.0, None, 24.0, 144.0, None],
            "codigo_usina": [10, 10, 10, 10, 10, 10],
            "geracao_MW": [120.0, 60.0, 68.57, 100.0, 50.0, 57.14],
        }
    )


class TestStageRows:
    def test_prefers_the_aggregate_row(self) -> None:
        rows = _stage_rows(_source_frame())
        assert len(rows) == 2
        assert "patamar" not in rows.columns
        assert rows["geracao_MW"].to_list() == [68.57, 57.14]

    def test_falls_back_to_block_rows_when_absent(self) -> None:
        frame = _source_frame().filter(pl.col("patamar").is_not_null())
        rows = _stage_rows(frame)
        assert len(rows) == 4


class TestWeightedGroupMean:
    """The shared kernel `_scenario_mean` and
    `_probability_weighted_stage_cost` both delegate to."""

    def test_weighted_mean_matches_the_probability_weighted_expectation(self) -> None:
        frame = pl.DataFrame(
            {
                "estagio": [3, 3],
                "probabilidade": [0.6, 0.4],
                "geracao_MW": [100.0, 20.0],
            }
        )
        out = _weighted_group_mean(frame, ["estagio"], ["geracao_MW"])
        assert out["geracao_MW"].to_list() == [pytest.approx(0.6 * 100.0 + 0.4 * 20.0)]

    def test_zero_weight_sum_falls_back_to_the_plain_mean(self) -> None:
        """The ``1e-12`` guard: a group whose weights sum to ~0 must use the
        plain mean instead of dividing by zero."""
        frame = pl.DataFrame(
            {
                "estagio": [3, 3],
                "probabilidade": [0.0, 0.0],
                "geracao_MW": [10.0, 30.0],
            }
        )
        out = _weighted_group_mean(frame, ["estagio"], ["geracao_MW"])
        assert out["geracao_MW"].to_list() == [pytest.approx(20.0)]

    def test_agrees_with_scenario_means_weighted_branch(self) -> None:
        """`_scenario_mean`'s weighted branch is exactly this kernel applied
        to its own joined frame -- calling it directly on the same shape must
        reproduce the identical per-group value."""
        joined = pl.DataFrame(
            {
                "estagio": [3, 3],
                "codigo_usina": [10, 10],
                "cenario": [1, 2],
                "geracao_MW": [100.0, 20.0],
                "probabilidade": [0.6, 0.4],
            }
        )
        via_kernel = _weighted_group_mean(
            joined, ["estagio", "codigo_usina"], ["geracao_MW"]
        )

        frame = joined.drop("probabilidade")
        probabilities = joined.select("estagio", "cenario", "probabilidade")
        via_scenario_mean = _scenario_mean(
            frame,
            "estagio",
            ["geracao_MW"],
            entity_column="codigo_usina",
            probabilities=probabilities,
        )

        assert via_kernel["geracao_MW"].to_list() == pytest.approx(
            via_scenario_mean["geracao_MW"].to_list()
        )


class TestScenarioMean:
    def test_averages_over_nodes(self) -> None:
        frame = pl.DataFrame(
            {
                "estagio": [3, 3, 3],
                "codigo_usina": [10, 10, 10],
                "geracao_MW": [10.0, 20.0, 60.0],
            }
        )
        out = _scenario_mean(
            frame, "estagio", ["geracao_MW"], entity_column="codigo_usina"
        )
        assert out["geracao_MW"].to_list() == [30.0]

    def test_probability_weighted_fan_stage_differs_from_unweighted(self) -> None:
        """A fan stage (3) with unequal (0.6/0.4) opening probabilities must
        be weighted, not plainly averaged; a single-scenario stage (1) must
        be unaffected -- a weighted mean over one row is that row's value."""
        frame = pl.DataFrame(
            {
                "estagio": [1, 3, 3],
                "cenario": [1, 1, 2],
                "codigo_usina": [10, 10, 10],
                "geracao_MW": [50.0, 100.0, 20.0],
            }
        )
        probabilities = pl.DataFrame(
            {
                "estagio": [1, 3, 3],
                "cenario": [1, 1, 2],
                "probabilidade": [1.0, 0.6, 0.4],
            }
        )

        weighted = _scenario_mean(
            frame,
            "estagio",
            ["geracao_MW"],
            entity_column="codigo_usina",
            probabilities=probabilities,
        )
        unweighted = _scenario_mean(
            frame, "estagio", ["geracao_MW"], entity_column="codigo_usina"
        )

        weighted_by_stage = dict(
            zip(weighted["estagio"].to_list(), weighted["geracao_MW"].to_list())
        )
        unweighted_by_stage = dict(
            zip(unweighted["estagio"].to_list(), unweighted["geracao_MW"].to_list())
        )
        # Fan stage: weighted mean = 0.6*100 + 0.4*20 = 68.0, NOT the
        # unweighted mean(100, 20) = 60.0.
        assert weighted_by_stage[3] == pytest.approx(0.6 * 100.0 + 0.4 * 20.0)
        assert weighted_by_stage[3] != pytest.approx(unweighted_by_stage[3])
        # Deterministic single-scenario stage: unaffected by weighting.
        assert weighted_by_stage[1] == pytest.approx(unweighted_by_stage[1])
        assert weighted_by_stage[1] == pytest.approx(50.0)

    def test_none_probabilities_keeps_the_unweighted_default(self) -> None:
        frame = pl.DataFrame(
            {
                "estagio": [3, 3, 3],
                "cenario": [1, 2, 3],
                "codigo_usina": [10, 10, 10],
                "geracao_MW": [10.0, 20.0, 60.0],
            }
        )
        out = _scenario_mean(
            frame,
            "estagio",
            ["geracao_MW"],
            entity_column="codigo_usina",
            probabilities=None,
        )
        assert out["geracao_MW"].to_list() == [30.0]

    def test_empty_probabilities_frame_keeps_the_unweighted_default(self) -> None:
        frame = pl.DataFrame(
            {
                "estagio": [3, 3, 3],
                "cenario": [1, 2, 3],
                "codigo_usina": [10, 10, 10],
                "geracao_MW": [10.0, 20.0, 60.0],
            }
        )
        out = _scenario_mean(
            frame,
            "estagio",
            ["geracao_MW"],
            entity_column="codigo_usina",
            probabilities=pl.DataFrame(),
        )
        assert out["geracao_MW"].to_list() == [30.0]

    def test_frame_without_cenario_column_keeps_the_unweighted_default(self) -> None:
        """A frame with no ``cenario`` column of its own (e.g. a table
        `_stage_rows` already collapsed past the scenario axis) must ignore
        *probabilities* entirely rather than fail trying to join on a column
        that does not exist."""
        frame = pl.DataFrame(
            {
                "estagio": [3, 3, 3],
                "codigo_usina": [10, 10, 10],
                "geracao_MW": [10.0, 20.0, 60.0],
            }
        )
        probabilities = pl.DataFrame(
            {"estagio": [3], "cenario": [1], "probabilidade": [0.9]}
        )
        out = _scenario_mean(
            frame,
            "estagio",
            ["geracao_MW"],
            entity_column="codigo_usina",
            probabilities=probabilities,
        )
        assert out["geracao_MW"].to_list() == [30.0]


class TestScenarioMeanCompositeKey:
    """Grouping by a composite entity key -- an interchange
    corridor's ``(de, para)`` code pair, not a single entity code."""

    def test_averages_over_nodes_per_corridor(self) -> None:
        frame = pl.DataFrame(
            {
                "estagio": [1, 1, 1, 1],
                "codigo_submercado_de": [1, 1, 2, 2],
                "codigo_submercado_para": [2, 2, 3, 3],
                "intercambio_origem_MW": [100.0, 120.0, 10.0, 30.0],
            }
        )
        out = _scenario_mean(
            frame,
            "estagio",
            ["intercambio_origem_MW"],
            entity_column=["codigo_submercado_de", "codigo_submercado_para"],
        )
        by_pair = {
            (row["codigo_submercado_de"], row["codigo_submercado_para"]): row[
                "intercambio_origem_MW"
            ]
            for row in out.iter_rows(named=True)
        }
        assert by_pair == {(1, 2): 110.0, (2, 3): 20.0}


class TestMapEntities:
    def test_maps_codes_and_rebases_stages(self) -> None:
        frame = pl.DataFrame({"estagio": [1, 2], "codigo_usina": [10, 10]})
        mapped, unmapped = _map_entities(frame, "codigo_usina", {10: 4})
        assert mapped["entity_id"].to_list() == [4, 4]
        assert mapped["stage_id"].to_list() == [0, 1]
        assert unmapped == []

    def test_reports_unmapped_codes_instead_of_dropping_silently(self) -> None:
        frame = pl.DataFrame({"estagio": [1, 1], "codigo_usina": [10, 99]})
        mapped, unmapped = _map_entities(frame, "codigo_usina", {10: 4})
        assert mapped["entity_id"].to_list() == [4]
        assert unmapped == [99]

    def test_keeps_the_original_code_as_newave_code(self) -> None:
        """``build_decomp_dataset`` needs the reference code back to fill
        ``ResultComparison.newave_code`` -- ``_map_entities`` must not drop it
        once it has been used to derive ``entity_id``."""
        frame = pl.DataFrame({"estagio": [1, 2], "codigo_usina": [10, 10]})
        mapped, _unmapped = _map_entities(frame, "codigo_usina", {10: 4})
        assert mapped["newave_code"].to_list() == [10, 10]


class TestResultComparisons:
    """``_result_comparisons`` joins one level's two frames into the
    ``ResultComparison`` rows that feed ``build_decomp_dataset``."""

    def _pair(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        source = pl.DataFrame(
            {
                "entity_id": [0, 1],
                "newave_code": [10, 11],
                "stage_id": [0, 0],
                "geracao_MW": [100.0, 50.0],
            }
        )
        novomodelo = pl.DataFrame(
            {
                "entity_id": [0, 1],
                "stage_id": [0, 0],
                "generation_mw": [90.0, 50.0],
            }
        )
        return source, novomodelo

    def test_emits_one_result_comparison_per_row_with_the_canonical_variable(
        self,
    ) -> None:
        source, novomodelo = self._pair()
        results = _result_comparisons(
            source, novomodelo, _HYDRO_VARIABLES, names={0: "A", 1: "B"}
        )
        assert {r.variable for r in results} == {"generation_mw"}
        by_id = {r.novomodelo_id: r for r in results}
        assert by_id[0].newave_code == 10
        assert by_id[0].entity_name == "A"
        assert by_id[0].entity_type == "hydro"
        assert by_id[0].stage == 0
        assert by_id[0].newave_value == 100.0
        assert by_id[0].novomodelo_value == 90.0
        assert by_id[0].abs_diff == pytest.approx(10.0)
        assert by_id[0].rel_diff == pytest.approx(0.1)
        assert by_id[1].newave_code == 11
        assert by_id[1].abs_diff == pytest.approx(0.0)

    def test_variables_missing_on_either_side_are_skipped(self) -> None:
        source, novomodelo = self._pair()
        results = _result_comparisons(source, novomodelo, _HYDRO_VARIABLES, names={})
        # Only generation's columns are present in both frames.
        assert {r.variable for r in results} == {"generation_mw"}

    def test_empty_join_yields_no_results(self) -> None:
        source = pl.DataFrame(
            {
                "entity_id": [0],
                "newave_code": [10],
                "stage_id": [0],
                "geracao_MW": [1.0],
            }
        )
        novomodelo = pl.DataFrame(
            {"entity_id": [9], "stage_id": [9], "generation_mw": [1.0]}
        )
        assert _result_comparisons(source, novomodelo, _HYDRO_VARIABLES, names={}) == []

    def test_canonical_variable_covers_all_eight_today_variables(self) -> None:
        """D-SOURCE-TOKEN-adjacent guard: every ``_Variable`` spec this module
        ships must resolve to a canonical chart name -- a spec with no entry
        would raise a ``KeyError`` deep inside ``_result_comparisons``."""
        all_vars = _HYDRO_VARIABLES + _THERMAL_VARIABLES + _BUS_VARIABLES
        assert len(all_vars) == 8
        canonical_names = {_CANONICAL_VARIABLE[(v.level, v.name)] for v in all_vars}
        assert canonical_names == {
            "generation_mw",
            "turbined_m3s",
            "spillage_m3s",
            "outflow_m3s",
            "storage_final_hm3",
            "deficit_mw",
            "spot_price",
        }


class TestBuildDecompDataset:
    """``build_decomp_dataset`` assembles the canonical dataset for the
    current 8 DECOMP variables via the shared ``_read_aligned_frames`` seam."""

    def test_dataset_validates_with_the_canonical_variables(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        dataset.validate()
        assert set(dataset.summary["variable"].to_list()) == {
            "generation_mw",
            "turbined_m3s",
            "spillage_m3s",
            "outflow_m3s",
            "storage_final_hm3",
            "deficit_mw",
            "spot_price",
            # derived realized hydro productivity (both fixture
            # plants turbine well above the zero-guard on both sides).
            "productivity_mw_per_m3s",
        }

    def test_tidy_sources_are_newave_and_novomodelo_only(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert set(dataset.tidy["source"].unique().to_list()) == {
            "newave",
            "novomodelo",
        }

    def test_hydro_storage_rows_compare_useful_volume(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The novomodelo-side value must be ``useful_storage_hm3`` (already
        ``storage_final_hm3 - min_storage_hm3`` upstream in ``_novomodelo_hydro``),
        not the raw absolute ``storage_final_hm3``."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        novomodelo_storage = (
            dataset.tidy.filter(
                (pl.col("variable") == "storage_final_hm3")
                & (pl.col("source") == "novomodelo")
            )
            .sort("entity_id")["value"]
            .to_list()
        )
        assert novomodelo_storage == [480.0, 300.0]

    def test_unmapped_codes_surface_in_metadata_and_are_excluded_from_tidy(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert dataset.metadata["unmapped"] == {
            "hydro": [],
            "thermal": [86, 224],
            "bus": [],
            "line": [],
            "ree": [],
            "evaporation": [],
        }
        thermal_codes = {
            r.newave_code for r in dataset.render.results if r.entity_type == "thermal"
        }
        assert 86 not in thermal_codes
        assert 224 not in thermal_codes

    def test_calls_the_shared_stat_kernel_not_a_local_reimplementation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``top_divergences``/``footer_counts`` are only populated by
        ``analyze.build_results_dataset``, so their presence is proof that
        function ran instead of a locally re-derived summary."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert "top_divergences" in dataset.metadata
        assert "footer_counts" in dataset.metadata

    def test_empty_comparison_returns_a_schema_valid_empty_dataset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fixture = _aligned_fixture()
        empty = _AlignedDecompFrames(
            source_hydro=fixture.source_hydro.clear(),
            source_thermal=fixture.source_thermal.clear(),
            source_bus=fixture.source_bus.clear(),
            novomodelo_hydro=fixture.novomodelo_hydro.clear(),
            novomodelo_thermal=fixture.novomodelo_thermal.clear(),
            novomodelo_bus=fixture.novomodelo_bus.clear(),
            hydro_names={},
            thermal_names={},
            bus_names={},
            unmapped={"hydro": [], "thermal": [], "bus": []},
        )
        _patch_aligned_frames(monkeypatch, empty)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        dataset.validate()
        assert dataset.tidy.is_empty()
        assert dataset.summary.is_empty()
        assert dataset.metadata["unmapped"] == {
            "hydro": [],
            "thermal": [],
            "bus": [],
            "line": [],
            "ree": [],
            "evaporation": [],
        }


class TestBusSideExcludesTranshipment:
    """The converter-created transhipment bus (``DecompIdMap.transhipment_bus_id``)
    has no source-model subsystem code -- it is referenced only by name in
    ``IA`` records, never emitted as an ``SB`` row -- so it can never appear
    among the ``codigo_submercado`` values ``_bus_side`` reads from the
    source model's system results table. The code -> id mapping ``_bus_side``
    builds its rows from (``{code: id_map.bus_id(code) for code in
    id_map.bus_codes}``) only ever holds values in ``range(len(bus_codes))``,
    one short of ``transhipment_bus_id`` -- so this is a pre-existing
    structural guarantee, not a new filter. This test pins that guarantee
    down as a regression guard rather than changing behaviour."""

    def test_bus_side_never_emits_the_transhipment_bus_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        id_map = DecompIdMap(bus_codes=(1, 2), bus_names=("SUDESTE", "SUL"))
        source_frame = pl.DataFrame(
            {
                "estagio": [1, 1],
                "no": [1, 1],
                "patamar": [None, None],
                "codigo_submercado": [1, 2],
                "deficit_MW": [0.0, 0.0],
                "cmo": [40.0, 45.0],
            }
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_sist",
            lambda *_args, **_kwargs: source_frame,
        )
        bus_codes = {code: id_map.bus_id(code) for code in id_map.bus_codes}

        mapped, unmapped = _bus_side(tmp_path, bus_codes)

        assert unmapped == []
        assert id_map.transhipment_bus_id not in mapped["entity_id"].to_list()


def _decomp_files_stub(tmp_path: Path) -> DecompFiles:
    return DecompFiles(
        revision="rv0",
        dadger=tmp_path / "dadger.rv0",
        vazoes=tmp_path / "vazoes.rv0",
        hidr=tmp_path / "hidr.dat",
        dadgnl=None,
        renovaveis=None,
        polinjus=None,
        libs_restricao_eletrica=None,
        cortesh=None,
        cortes=None,
    )


def _patch_discoverable_deck_with_no_sb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Make deck discovery and ``Dadger.read`` succeed but hand back a deck
    with no ``SB`` register, so ``DecompIdMap.from_dadger`` reaches its real
    ``FieldParseError`` parse-boundary raise (rather than the discovery
    failure the bare-``tmp_path`` tests exercise)."""
    monkeypatch.setattr(
        "novomodelo_bridge.decomp.case.discover_decomp_files",
        lambda _src: _decomp_files_stub(tmp_path),
    )
    monkeypatch.setattr(
        "idecomp.decomp.Dadger.read", lambda _path: _ConstraintFakeDadger()
    )


class TestBuildDecompDatasetSharedCaseBuild:
    """The deck is now parsed exactly once, via the shared
    ``DecompCase`` built at the top of ``build_decomp_dataset`` --
    retargeted replacement for the old per-helper ``_build_line_id_map(...)
    is None``/``_decomp_constraint_context(...) is None`` graceful-degrade
    unit tests. A bad/deckless deck now raises at that shared build (the
    same typed error ``_read_aligned_frames`` already raised first)
    rather than silently degrading each of the three sites
    independently."""

    def test_deckless_directory_raises_the_typed_discovery_error(
        self, tmp_path: Path
    ) -> None:
        """A bare directory (no ``caso.dat``) raises ``SourceFileError`` at
        the shared case build, not a per-section ``None`` degrade."""
        with pytest.raises(SourceFileError):
            build_decomp_dataset(tmp_path, tmp_path)

    def test_discoverable_deck_with_no_sb_records_raises_field_parse_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A discoverable, readable deck whose ``SB`` register is absent
        raises the typed parse-boundary ``FieldParseError`` from
        ``DecompIdMap.from_dadger`` (via the shared case's ``id_map``) at the
        same first-touch position ``_read_aligned_frames`` reaches it."""
        _patch_discoverable_deck_with_no_sb(monkeypatch, tmp_path)

        with pytest.raises(FieldParseError):
            build_decomp_dataset(tmp_path, tmp_path)


def _minimal_sist_frame() -> pl.DataFrame:
    """One stage, one bus -- the minimal ``dec_oper_sist``-shaped row both
    ``_bus_side`` and ``_energy_balance_frames`` read (the single-parse spy
    exercises the real, unmocked ``_read_aligned_frames``,
    so its own readers need a real-enough frame instead of the
    ``_read_aligned_frames``-level stub every other fixture in this module
    uses)."""
    return pl.DataFrame(
        {
            "estagio": [1],
            "no": [1],
            "cenario": [1],
            "patamar": [None],
            "codigo_submercado": [1],
            "deficit_MW": [0.0],
            "cmo": [40.0],
        }
    )


class TestBuildDecompDatasetSingleParse:
    """`build_decomp_dataset` parses the deck exactly
    once via the shared `DecompCase`, no matter how many of the three
    historical parse sites (read/align, the Network/Productivity/REE/
    evaporation id map, and the Constraints-tab census) a given run
    exercises. Unlike every other fixture in this module, these tests do
    NOT patch ``_read_aligned_frames`` away -- that seam is itself one of
    the three sites under test, so it must run for real."""

    def test_dadger_read_invoked_exactly_once_for_the_whole_build(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A spy on the public ``idecomp.decomp.Dadger.read`` entry point --
        the seam every one of the three historical parse sites called
        through -- sees exactly one invocation for a whole
        ``build_decomp_dataset`` run that exercises all three, down from
        three separate ``discover_decomp_files -> Dadger.read ->
        DecompIdMap.from_dadger`` parses."""
        decomp_dir = tmp_path / "deck"
        case_dir = tmp_path / "case"
        constraints = [
            {
                "id": 0,
                "name": "VminOP_1",
                "description": "unrecognized family/id -- exercises the "
                "census build without needing a matching register record",
                "expression": "",
                "slack": {"enabled": True, "penalty": 1000.0},
            }
        ]
        output_dir = _write_generic_constraints_case(case_dir, constraints, [])

        spy = MagicMock(
            return_value=_ConstraintFakeDadger(
                sb=pd.DataFrame({"codigo_submercado": [1], "nome_submercado": ["SE"]})
            )
        )
        monkeypatch.setattr("idecomp.decomp.Dadger.read", spy)
        monkeypatch.setattr(
            "novomodelo_bridge.decomp.case.discover_decomp_files",
            lambda _src: _decomp_files_stub(decomp_dir),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_usih",
            lambda *_a, **_k: _usih_frame([{"codigo_usina": 999, "estagio": 1}]),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_usit",
            lambda *_a, **_k: _usih_frame([{"codigo_usina": 998, "estagio": 1}]),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_sist",
            lambda *_a, **_k: _minimal_sist_frame(),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results._cost_frames",
            lambda *_a, **_k: ({}, pl.DataFrame()),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.novomodelo_readers."
            "read_novomodelo_bus_aggregates",
            lambda *_a, **_k: pl.DataFrame(),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.novomodelo_readers."
            "read_novomodelo_hydro_bus_labels",
            lambda *_a, **_k: {},
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.constraints.evaluate_lhs_novomodelo",
            lambda *_a, **_k: pl.DataFrame(),
        )

        dataset = build_decomp_dataset(decomp_dir, output_dir)

        assert spy.call_count == 1
        # Sanity: the Constraints-tab census build (the third historical
        # parse site) actually ran as part of this build.
        assert dataset.render.gc_constraints == constraints

    def test_dataset_equal_across_base_ree_and_constraints_sections(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """ComparisonDataset-equality regression: one ``build_decomp_dataset``
        run exercising the hydro/thermal/bus base variables, the REE rollup,
        and the Constraints-tab DECOMP-side LHS together -- all three reuse
        the SAME shared case's ``id_map``/``dadger`` -- reproduces exactly
        the values each section's own dedicated test (``TestBuildDecompDataset``,
        ``TestBuildDecompDatasetRee``, ``TestBuildDecompDatasetConstraints``)
        independently verifies in isolation. A behaviour-preserving
        consolidation of the three parse sites cannot change any of these."""
        _patch_aligned_frames(monkeypatch, _ree_aligned_fixture())
        _patch_shared_case(monkeypatch, id_map=_ree_id_map())
        _patch_ree_sources(monkeypatch)
        case_dir = tmp_path / "case"
        constraints = [
            {
                "id": 0,
                "name": "RHE_115",
                "description": "RHE stored-energy constraint 115",
                "expression": "@rho_acum_h0 * hydro_storage(0)",
                "slack": {"enabled": True, "penalty": 1000.0},
            }
        ]
        bound_rows = [
            {
                "constraint_id": 0,
                "stage_id": 0,
                "block_id": None,
                "bound_lower": 3097.40,
                "bound_upper": None,
            }
        ]
        output_dir = _write_generic_constraints_case(case_dir, constraints, bound_rows)
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_usih",
            _no_dec_oper,
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_usit",
            _no_dec_oper,
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_rhesoft",
            lambda *_a, **_k: pl.DataFrame(
                {
                    "estagio": [1],
                    "no": [1],
                    "cenario": [1],
                    "codigo_restricao": [115],
                    "valor_MW": [2951.58],
                    "violacao_absoluta_MW": [145.83],
                }
            ),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.constraints.evaluate_lhs_novomodelo",
            lambda *_a, **_k: pl.DataFrame(
                {"constraint_id": [0], "stage_id": [0], "lhs_value": [3000.0]}
            ),
        )

        dataset = build_decomp_dataset(case_dir, output_dir)
        dataset.validate()

        # Base hydro/thermal/bus/productivity variables (TestBuildDecompDataset).
        assert set(dataset.summary["variable"].to_list()) >= {
            "generation_mw",
            "turbined_m3s",
            "spillage_m3s",
            "outflow_m3s",
            "storage_final_hm3",
            "deficit_mw",
            "spot_price",
            "productivity_mw_per_m3s",
        }
        # metadata["unmapped"] per level (TestBuildDecompDataset).
        assert dataset.metadata["unmapped"] == {
            "hydro": [],
            "thermal": [86, 224],
            "bus": [],
            "line": [],
            "ree": [],
            "evaporation": [],
        }

        # REE rollup (TestBuildDecompDatasetRee).
        ree_rows = dataset.tidy.filter(pl.col("entity_type") == "ree")
        assert set(ree_rows["variable"].unique().to_list()) == {
            "ena_mwmes",
            "earm_final_mwmes",
        }
        assert set(ree_rows["source"].unique().to_list()) == {"newave", "novomodelo"}

        # Constraints tab DECOMP-side LHS, derived via the shared case's
        # dadger/id_map (TestBuildDecompDatasetConstraints).
        assert dataset.render.gc_constraints == constraints
        nw_row = dataset.render.gc_lhs_newave.row(0, named=True)
        assert nw_row["constraint_id"] == 0
        assert nw_row["stage_id"] == 0
        assert nw_row["lhs_value"] == pytest.approx(2951.58)
        cb_row = dataset.render.gc_lhs_novomodelo.row(0, named=True)
        assert cb_row == {"constraint_id": 0, "stage_id": 0, "lhs_value": 3000.0}
