"""Network-tab tests for ``comparators.decomp.results``.

Second carve out of the legacy ``test_decomp_results_compare.py`` mega file:
corridor/line alignment, the DECOMP interchange side, line entity
names and result comparisons, line bounds/metadata, and the Network tab's
``build_decomp_dataset`` rows. The remaining concern bands (energy balance,
costs, performance, hydro/thermal detail, productivity, FPHA, REE,
evaporation, constraints, CLI) stay in the mega file pending their own carve.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import pytest

from novomodelo_bridge.comparators.decomp.results import (
    _corridor_line_alignment,
    _interc_side,
    _line_bounds_and_meta,
    _line_entity_names,
    _line_result_comparisons,
    _read_novomodelo_lines_index,
    build_decomp_dataset,
)
from novomodelo_bridge.comparators.report_builder import build_comparison_report
from novomodelo_bridge.decomp.id_map import DecompIdMap
from tests.comparators.conftest import (
    _aligned_fixture,
    _patch_aligned_frames,
    _patch_shared_case,
)


def _decomp_id_map_three_subsystems() -> DecompIdMap:
    """SE (code 1) -> bus 0, S (code 2) -> bus 1, NE (code 3) -> bus 2, plus
    the converter-created transhipment bus at id 3."""
    return DecompIdMap(bus_codes=(1, 2, 3), bus_names=("SE", "S", "NE"))


def _line_entry(
    line_id: int,
    source_bus_id: int,
    target_bus_id: int,
    *,
    name: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": line_id,
        "source_bus_id": source_bus_id,
        "target_bus_id": target_bus_id,
        "capacity": {"direct_mw": 1000.0, "reverse_mw": 1000.0},
    }
    if name is not None:
        entry["name"] = name
    return entry


def _write_lines_json(case_dir: Path, lines: list[dict[str, Any]]) -> Path:
    """Write ``system/lines.json`` under *case_dir* and return the Novomodelo
    output dir (``case_dir/output``) that ``case_dir_for`` resolves back to
    *case_dir* from."""
    system_dir = case_dir / "system"
    system_dir.mkdir(parents=True, exist_ok=True)
    (system_dir / "lines.json").write_text(json.dumps({"lines": lines}))
    output_dir = case_dir / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def _write_line_bounds_parquet(case_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Write ``constraints/line_bounds.parquet`` under *case_dir* -- the
    Network tab's per-stage capacity source (see
    ``decomp/network.py::convert_lines``'s own ``_LINE_BOUNDS_SCHEMA``)."""
    constraints_dir = case_dir / "constraints"
    constraints_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(constraints_dir / "line_bounds.parquet")


class TestCorridorLineAlignment:
    """``_corridor_line_alignment`` -- corridor (bus_de, bus_para)
    -> ordered novomodelo (line_id, sign) legs."""

    def test_direct_line_corridor_maps_with_positive_sign(self, tmp_path: Path) -> None:
        id_map = _decomp_id_map_three_subsystems()
        output_dir = _write_lines_json(tmp_path, [_line_entry(0, 0, 1)])

        alignment = _corridor_line_alignment(output_dir, id_map)

        assert alignment[(0, 1)] == [(0, 1)]

    def test_reverse_declared_line_gets_negative_sign(self, tmp_path: Path) -> None:
        """A novomodelo line declared S -> SE realizes the SE -> S corridor with
        sign -1, orienting the line's own net_flow_mw onto the requested
        direction."""
        id_map = _decomp_id_map_three_subsystems()
        output_dir = _write_lines_json(tmp_path, [_line_entry(0, 1, 0)])

        alignment = _corridor_line_alignment(output_dir, id_map)

        assert alignment[(0, 1)] == [(0, -1)]

    def test_star_topology_two_leg_corridor(self, tmp_path: Path) -> None:
        """No direct SE-S line; SE<->IV + IV<->S realizes the SE -> S
        corridor as two consistently-oriented legs."""
        id_map = _decomp_id_map_three_subsystems()
        transhipment = id_map.transhipment_bus_id
        output_dir = _write_lines_json(
            tmp_path,
            [
                _line_entry(0, 0, transhipment),
                _line_entry(1, transhipment, 1),
            ],
        )

        alignment = _corridor_line_alignment(output_dir, id_map)

        assert alignment[(0, 1)] == [(0, 1), (1, 1)]

    def test_corridor_with_no_realizing_path_is_absent(self, tmp_path: Path) -> None:
        id_map = _decomp_id_map_three_subsystems()
        # Only SE-S is wired; NE (bus id 2) has no line at all, direct or
        # via the transhipment bus.
        output_dir = _write_lines_json(tmp_path, [_line_entry(0, 0, 1)])

        alignment = _corridor_line_alignment(output_dir, id_map)

        assert (0, 2) not in alignment
        assert (2, 0) not in alignment

    def test_missing_lines_json_yields_an_empty_alignment(self, tmp_path: Path) -> None:
        id_map = _decomp_id_map_three_subsystems()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        assert _corridor_line_alignment(output_dir, id_map) == {}

    def test_ambiguous_duplicate_line_pair_is_excluded_not_guessed(
        self, tmp_path: Path
    ) -> None:
        """Two lines declared for the same ordered bus pair make that leg
        ambiguous under the star-topology assumption -- both are dropped
        rather than one picked arbitrarily."""
        id_map = _decomp_id_map_three_subsystems()
        output_dir = _write_lines_json(
            tmp_path, [_line_entry(0, 0, 1), _line_entry(1, 0, 1)]
        )

        assert (0, 1) not in _read_novomodelo_lines_index(output_dir)
        assert (0, 1) not in _corridor_line_alignment(output_dir, id_map)


def _dec_oper_interc_frame(
    *, de: int, para: int, origem_mw: float, perdas_mw: float = 5.0
) -> pl.DataFrame:
    """One corridor, one stage, one node -- the patamar-null aggregate row
    ``_interc_side`` reads via ``_stage_rows``."""
    return pl.DataFrame(
        {
            "estagio": [1],
            "no": [1],
            "cenario": [1],
            "patamar": [None],
            "codigo_submercado_de": [de],
            "codigo_submercado_para": [para],
            "intercambio_origem_MW": [origem_mw],
            "intercambio_destino_MW": [origem_mw - perdas_mw],
            "perdas_MW": [perdas_mw],
        }
    )


class TestIntercSide:
    """``_interc_side`` -- the aligned per-(novomodelo line_id, stage)
    DECOMP net-flow frame, plus the unresolved-corridor report."""

    def _patch_interc(
        self, monkeypatch: pytest.MonkeyPatch, frame: pl.DataFrame
    ) -> None:
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_interc",
            lambda *_args, **_kwargs: frame,
        )

    def test_direct_line_corridor_reproduces_a_positive_flow(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Sign orientation: a known de -> para positive flow, on a novomodelo
        line declared in that same de -> para direction, yields a positive
        aligned novomodelo net-flow."""
        id_map = _decomp_id_map_three_subsystems()
        output_dir = _write_lines_json(tmp_path, [_line_entry(0, 0, 1)])
        self._patch_interc(
            monkeypatch, _dec_oper_interc_frame(de=1, para=2, origem_mw=250.0)
        )

        result, unresolved = _interc_side(tmp_path, output_dir, id_map)

        assert unresolved == []
        assert result.to_dicts() == [
            {"entity_id": 0, "stage_id": 0, "net_flow_mw": 250.0}
        ]

    def test_star_topology_corridor_reproduces_the_flow_on_both_legs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        id_map = _decomp_id_map_three_subsystems()
        transhipment = id_map.transhipment_bus_id
        output_dir = _write_lines_json(
            tmp_path,
            [_line_entry(0, 0, transhipment), _line_entry(1, transhipment, 1)],
        )
        self._patch_interc(
            monkeypatch, _dec_oper_interc_frame(de=1, para=2, origem_mw=180.0)
        )

        result, unresolved = _interc_side(tmp_path, output_dir, id_map)

        assert unresolved == []
        by_line = {
            row["entity_id"]: row["net_flow_mw"] for row in result.iter_rows(named=True)
        }
        assert by_line == {0: 180.0, 1: 180.0}

    def test_unresolved_corridor_is_reported_and_excluded_from_the_frame(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        id_map = _decomp_id_map_three_subsystems()
        # NE (code 3, bus id 2) has no realizing line at all.
        output_dir = _write_lines_json(tmp_path, [_line_entry(0, 0, 1)])
        self._patch_interc(
            monkeypatch, _dec_oper_interc_frame(de=1, para=3, origem_mw=90.0)
        )

        result, unresolved = _interc_side(tmp_path, output_dir, id_map)

        assert result.is_empty()
        assert unresolved == [(1, 3)]

    def test_missing_lines_json_leaves_every_corridor_unresolved(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        id_map = _decomp_id_map_three_subsystems()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        self._patch_interc(
            monkeypatch, _dec_oper_interc_frame(de=1, para=2, origem_mw=100.0)
        )

        result, unresolved = _interc_side(tmp_path, output_dir, id_map)

        assert result.is_empty()
        assert unresolved == [(1, 2)]

    def test_reverse_declared_line_flips_the_aligned_sign(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The novomodelo line runs S -> SE; the DECOMP corridor is declared
        SE -> S with a positive flow, so the aligned novomodelo net-flow must be
        negative to stay oriented to the line's own
        source_bus_id -> target_bus_id convention."""
        id_map = _decomp_id_map_three_subsystems()
        output_dir = _write_lines_json(tmp_path, [_line_entry(0, 1, 0)])
        self._patch_interc(
            monkeypatch, _dec_oper_interc_frame(de=1, para=2, origem_mw=100.0)
        )

        result, unresolved = _interc_side(tmp_path, output_dir, id_map)

        assert unresolved == []
        assert result.to_dicts() == [
            {"entity_id": 0, "stage_id": 0, "net_flow_mw": -100.0}
        ]

    def test_losses_are_read_but_not_subtracted_from_the_net_flow(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The aligned net-flow is the origin-side reading, matching a
        lossless novomodelo line -- not ``intercambio_origem_MW - perdas_MW``."""
        id_map = _decomp_id_map_three_subsystems()
        output_dir = _write_lines_json(tmp_path, [_line_entry(0, 0, 1)])
        self._patch_interc(
            monkeypatch,
            _dec_oper_interc_frame(de=1, para=2, origem_mw=200.0, perdas_mw=12.0),
        )

        result, _unresolved = _interc_side(tmp_path, output_dir, id_map)

        assert result["net_flow_mw"].to_list() == [200.0]

    def test_a_corridor_reported_by_name_only_is_unresolved_not_a_crash(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Regression guard: the source model reports some corridors (the
        Itaipu 60 Hz ``IV`` node, in particular) with a null
        ``codigo_submercado_de``/``_para`` -- a name-only endpoint outside
        the ``SB`` register. This must surface as unresolved (``-1``
        sentinel), never raise, since it can never resolve through
        ``_corridor_line_alignment`` regardless (the transhipment bus is
        never an outer-loop endpoint there)."""
        id_map = _decomp_id_map_three_subsystems()
        output_dir = _write_lines_json(tmp_path, [_line_entry(0, 0, 1)])
        frame = pl.DataFrame(
            {
                "estagio": [1],
                "no": [1],
                "cenario": [1],
                "patamar": [None],
                "codigo_submercado_de": [None],
                "codigo_submercado_para": [2],
                "intercambio_origem_MW": [50.0],
                "intercambio_destino_MW": [50.0],
                "perdas_MW": [0.0],
            },
            schema_overrides={"codigo_submercado_de": pl.Float64},
        )
        self._patch_interc(monkeypatch, frame)

        result, unresolved = _interc_side(tmp_path, output_dir, id_map)

        assert result.is_empty()
        assert unresolved == [(-1, 2)]

    def test_both_directions_of_one_interface_net_to_a_single_row(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Regression: the source model reports every physical interface as
        two corridor rows, one per direction (both ``SE -> S`` and
        ``S -> SE``). Both align onto the same direct novomodelo line leg with
        opposite sign -- the returned frame must collapse them to a single
        ``(entity_id, stage_id)`` row (not the duplicate keys that used to
        fan out ``_line_result_comparisons``'s join into a spurious extra
        row), and that row's net flow must be the signed sum of the two
        directions."""
        id_map = _decomp_id_map_three_subsystems()
        output_dir = _write_lines_json(tmp_path, [_line_entry(0, 0, 1)])
        frame = pl.concat(
            [
                _dec_oper_interc_frame(de=1, para=2, origem_mw=250.0),
                _dec_oper_interc_frame(de=2, para=1, origem_mw=90.0),
            ]
        )
        self._patch_interc(monkeypatch, frame)

        result, unresolved = _interc_side(tmp_path, output_dir, id_map)

        assert unresolved == []
        assert result.to_dicts() == [
            {"entity_id": 0, "stage_id": 0, "net_flow_mw": 160.0}
        ]

    def test_both_directions_of_one_interface_net_correctly_on_a_shared_star_leg(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The same duplicate-direction netting must hold on each leg of a
        two-leg star corridor: ``SE -> S`` and ``S -> SE`` both realize
        through ``SE <-> transhipment <-> S``, so each leg must collapse to
        one row equal to ``flow(SE -> S) - flow(S -> SE)``, not two rows
        colliding on the same ``(entity_id, stage_id)`` key."""
        id_map = _decomp_id_map_three_subsystems()
        transhipment = id_map.transhipment_bus_id
        output_dir = _write_lines_json(
            tmp_path,
            [_line_entry(0, 0, transhipment), _line_entry(1, transhipment, 1)],
        )
        frame = pl.concat(
            [
                _dec_oper_interc_frame(de=1, para=2, origem_mw=180.0),
                _dec_oper_interc_frame(de=2, para=1, origem_mw=70.0),
            ]
        )
        self._patch_interc(monkeypatch, frame)

        result, unresolved = _interc_side(tmp_path, output_dir, id_map)

        assert unresolved == []
        by_line = {
            row["entity_id"]: row["net_flow_mw"] for row in result.iter_rows(named=True)
        }
        assert by_line == {0: 110.0, 1: 110.0}


class TestLineEntityNames:
    """Display name per novomodelo line id for the Network tab."""

    def test_uses_the_name_from_line_meta_when_present(self) -> None:
        id_map = _decomp_id_map_three_subsystems()
        line_meta = [_line_entry(0, 0, 1, name="SE-S")]

        names = _line_entity_names(line_meta, id_map)

        assert names == {0: "SE-S"}

    def test_falls_back_to_a_bus_name_pair_label_when_name_is_absent(self) -> None:
        id_map = _decomp_id_map_three_subsystems()
        line_meta = [_line_entry(0, 0, 1)]  # no "name" key

        names = _line_entity_names(line_meta, id_map)

        assert names == {0: "SE<->S"}

    def test_falls_back_to_a_line_id_label_when_the_bus_ids_are_unresolvable(
        self,
    ) -> None:
        id_map = _decomp_id_map_three_subsystems()
        line_meta = [_line_entry(0, 0, 99)]  # bus id 99 is not declared

        names = _line_entity_names(line_meta, id_map)

        assert names == {0: "line_0"}


class TestLineResultComparisons:
    """Corridor-aligned line ``ResultComparison`` rows, joining
    the DECOMP net-flow onto Novomodelo's per-line simulation means."""

    def test_id_map_none_returns_no_rows_and_no_unresolved(
        self, tmp_path: Path
    ) -> None:
        results, unresolved = _line_result_comparisons(tmp_path, tmp_path, None, [])

        assert results == []
        assert unresolved == []

    def test_missing_source_interchange_table_degrades_to_empty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        id_map = _decomp_id_map_three_subsystems()

        def _boom(*_args: object, **_kwargs: object) -> pl.DataFrame:
            raise FileNotFoundError("dec_oper_interc.csv not found")

        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_interc", _boom
        )

        results, unresolved = _line_result_comparisons(tmp_path, tmp_path, id_map, [])

        assert results == []
        assert unresolved == []

    def test_joins_source_and_novomodelo_into_result_comparison_rows(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        id_map = _decomp_id_map_three_subsystems()
        line_meta = [_line_entry(0, 0, 1, name="SE-S")]
        output_dir = _write_lines_json(tmp_path, line_meta)
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_interc",
            lambda *_args, **_kwargs: _dec_oper_interc_frame(
                de=1, para=2, origem_mw=250.0
            ),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.novomodelo_readers."
            "read_novomodelo_line_means",
            lambda *_args, **_kwargs: pl.DataFrame(
                {"entity_id": [0], "stage_id": [0], "net_flow_mw": [240.0]}
            ),
        )

        results, unresolved = _line_result_comparisons(
            tmp_path, output_dir, id_map, line_meta
        )

        assert unresolved == []
        assert len(results) == 1
        row = results[0]
        assert row.entity_type == "line"
        assert row.entity_name == "SE-S"
        assert row.novomodelo_id == 0
        assert row.stage == 0
        assert row.variable == "net_flow_mw"
        assert row.newave_value == 250.0
        assert row.novomodelo_value == 240.0
        # D-SOURCE-TOKEN-adjacent: no single source code covers a line that
        # may be a shared leg of more than one corridor -- see the module
        # docstring.
        assert row.newave_code == 0

    def test_unresolved_corridors_surface_even_with_no_novomodelo_line_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """NE (code 3) has no realizing line -- reported unresolved even
        though there is nothing to join against on the Novomodelo side."""
        id_map = _decomp_id_map_three_subsystems()
        output_dir = _write_lines_json(tmp_path, [_line_entry(0, 0, 1)])
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_interc",
            lambda *_args, **_kwargs: _dec_oper_interc_frame(
                de=1, para=3, origem_mw=90.0
            ),
        )

        results, unresolved = _line_result_comparisons(tmp_path, output_dir, id_map, [])

        assert results == []
        assert unresolved == [[1, 3]]


class TestLineBoundsAndMeta:
    """Novomodelo-side line capacity bounds + metadata, read straight
    from the converted case (mirrors ``results.compare_results``)."""

    def test_reads_line_bounds_parquet_and_lines_json(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "case"
        line_meta_in = [_line_entry(0, 0, 1, name="SE-S")]
        output_dir = _write_lines_json(case_dir, line_meta_in)
        _write_line_bounds_parquet(
            case_dir,
            [
                {
                    "line_id": 0,
                    "stage_id": 0,
                    "block_id": None,
                    "direct_mw": 1200.0,
                    "reverse_mw": 800.0,
                }
            ],
        )

        line_bounds, line_meta = _line_bounds_and_meta(output_dir)

        assert isinstance(line_bounds, pl.DataFrame)
        assert {"line_id", "stage_id", "direct_mw", "reverse_mw"}.issubset(
            set(line_bounds.columns)
        )
        row = line_bounds.row(0, named=True)
        assert row["direct_mw"] == 1200.0
        assert row["reverse_mw"] == 800.0
        assert line_meta == line_meta_in
        assert line_meta[0]["capacity"]["direct_mw"] == 1000.0

    def test_missing_files_degrade_to_empty(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        line_bounds, line_meta = _line_bounds_and_meta(output_dir)

        assert isinstance(line_bounds, pl.DataFrame)
        assert line_bounds.is_empty()
        assert line_meta == []


def _patch_network(
    monkeypatch: pytest.MonkeyPatch,
    *,
    id_map: DecompIdMap,
    interc_frame: pl.DataFrame,
    novomodelo_line_means: pl.DataFrame,
    novomodelo_line_pct: pl.DataFrame | None = None,
) -> None:
    """Stub the line seam: the deck's id map, its interchange
    table, and Novomodelo's own per-line simulation output -- mirroring
    ``_patch_aligned_frames``'s "patch at the seam" convention."""
    _patch_shared_case(monkeypatch, id_map=id_map)
    monkeypatch.setattr(
        "novomodelo_bridge.comparators.decomp.results.read_dec_oper_interc",
        lambda *_args, **_kwargs: interc_frame,
    )
    monkeypatch.setattr(
        "novomodelo_bridge.comparators.decomp.results.novomodelo_readers.read_novomodelo_line_means",
        lambda *_args, **_kwargs: novomodelo_line_means,
    )
    if novomodelo_line_pct is not None:
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.novomodelo_readers."
            "read_novomodelo_line_percentiles",
            lambda *_args, **_kwargs: novomodelo_line_pct,
        )


class TestBuildDecompDatasetNetwork:
    """Line rows + Network tab metadata in
    ``build_decomp_dataset``, on top of the ``_aligned_fixture``."""

    def _case_dirs(self, tmp_path: Path) -> tuple[Path, Path]:
        """A deck dir and a converted-case ``output/`` dir, isolated under
        *tmp_path* -- unlike ``build_decomp_dataset(tmp_path, tmp_path)``,
        ``case_dir_for(novomodelo_output_dir)`` must resolve to a real directory
        this test controls, since ``_line_bounds_and_meta`` reads
        ``system/``/``constraints/`` from it."""
        case_dir = tmp_path / "case"
        output_dir = _write_lines_json(case_dir, [_line_entry(0, 0, 1, name="SE-S")])
        _write_line_bounds_parquet(
            case_dir,
            [
                {
                    "line_id": 0,
                    "stage_id": 0,
                    "block_id": None,
                    "direct_mw": 1200.0,
                    "reverse_mw": 800.0,
                }
            ],
        )
        return tmp_path / "deck", output_dir

    def test_line_rows_and_summary_present_with_expected_sources(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        decomp_dir, novomodelo_output_dir = self._case_dirs(tmp_path)
        _patch_network(
            monkeypatch,
            id_map=_decomp_id_map_three_subsystems(),
            interc_frame=_dec_oper_interc_frame(de=1, para=2, origem_mw=250.0),
            novomodelo_line_means=pl.DataFrame(
                {"entity_id": [0], "stage_id": [0], "net_flow_mw": [240.0]}
            ),
        )

        dataset = build_decomp_dataset(decomp_dir, novomodelo_output_dir)

        line_rows = dataset.tidy.filter(pl.col("entity_type") == "line")
        assert not line_rows.is_empty()
        assert set(line_rows["variable"].unique().to_list()) == {"net_flow_mw"}
        assert set(line_rows["source"].unique().to_list()) <= {"newave", "novomodelo"}
        assert "net_flow_mw" in dataset.summary["variable"].to_list()

    def test_line_percentiles_populate_metadata_when_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        decomp_dir, novomodelo_output_dir = self._case_dirs(tmp_path)
        _patch_network(
            monkeypatch,
            id_map=_decomp_id_map_three_subsystems(),
            interc_frame=_dec_oper_interc_frame(de=1, para=2, origem_mw=250.0),
            novomodelo_line_means=pl.DataFrame(
                {"entity_id": [0], "stage_id": [0], "net_flow_mw": [240.0]}
            ),
            novomodelo_line_pct=pl.DataFrame(
                {
                    "entity_id": [0],
                    "stage_id": [0],
                    "net_flow_mw_p10": [200.0],
                    "net_flow_mw_p50": [240.0],
                    "net_flow_mw_p90": [280.0],
                }
            ),
        )

        dataset = build_decomp_dataset(decomp_dir, novomodelo_output_dir)

        line_pct = dataset.render.line
        assert isinstance(line_pct, pl.DataFrame)
        assert not line_pct.is_empty()
        assert {"net_flow_mw_p10", "net_flow_mw_p50", "net_flow_mw_p90"}.issubset(
            set(line_pct.columns)
        )

    def test_line_percentiles_stay_empty_when_novomodelo_output_lacks_them(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No percentile mock: no ``simulation/exchanges`` partition under
        the case's output dir, so the reader degrades to its own empty
        default and no band is fabricated (deterministic-tree caveat)."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        decomp_dir, novomodelo_output_dir = self._case_dirs(tmp_path)
        _patch_network(
            monkeypatch,
            id_map=_decomp_id_map_three_subsystems(),
            interc_frame=_dec_oper_interc_frame(de=1, para=2, origem_mw=250.0),
            novomodelo_line_means=pl.DataFrame(
                {"entity_id": [0], "stage_id": [0], "net_flow_mw": [240.0]}
            ),
        )

        dataset = build_decomp_dataset(decomp_dir, novomodelo_output_dir)

        line_pct = dataset.render.line
        assert isinstance(line_pct, pl.DataFrame)
        assert line_pct.is_empty()

    def test_line_bounds_is_polars_with_the_expected_columns(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        decomp_dir, novomodelo_output_dir = self._case_dirs(tmp_path)
        _patch_network(
            monkeypatch,
            id_map=_decomp_id_map_three_subsystems(),
            interc_frame=_dec_oper_interc_frame(de=1, para=2, origem_mw=250.0),
            novomodelo_line_means=pl.DataFrame(
                {"entity_id": [0], "stage_id": [0], "net_flow_mw": [240.0]}
            ),
        )

        dataset = build_decomp_dataset(decomp_dir, novomodelo_output_dir)

        line_bounds = dataset.render.line_bounds
        assert isinstance(line_bounds, pl.DataFrame)
        assert {"line_id", "stage_id", "direct_mw", "reverse_mw"}.issubset(
            set(line_bounds.columns)
        )
        assert line_bounds.row(0, named=True)["direct_mw"] == 1200.0

    def test_line_meta_has_the_nested_capacity_shape(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        decomp_dir, novomodelo_output_dir = self._case_dirs(tmp_path)
        _patch_network(
            monkeypatch,
            id_map=_decomp_id_map_three_subsystems(),
            interc_frame=_dec_oper_interc_frame(de=1, para=2, origem_mw=250.0),
            novomodelo_line_means=pl.DataFrame(
                {"entity_id": [0], "stage_id": [0], "net_flow_mw": [240.0]}
            ),
        )

        dataset = build_decomp_dataset(decomp_dir, novomodelo_output_dir)

        line_meta = dataset.render.line_meta
        assert isinstance(line_meta, list)
        assert line_meta[0]["id"] == 0
        assert line_meta[0]["capacity"]["direct_mw"] == 1000.0
        assert line_meta[0]["capacity"]["reverse_mw"] == 1000.0
        assert line_meta[0]["name"] == "SE-S"

    def test_network_tab_renders_in_the_html_report(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        decomp_dir, novomodelo_output_dir = self._case_dirs(tmp_path)
        _patch_network(
            monkeypatch,
            id_map=_decomp_id_map_three_subsystems(),
            interc_frame=_dec_oper_interc_frame(de=1, para=2, origem_mw=250.0),
            novomodelo_line_means=pl.DataFrame(
                {"entity_id": [0], "stage_id": [0], "net_flow_mw": [240.0]}
            ),
        )

        dataset = build_decomp_dataset(decomp_dir, novomodelo_output_dir)
        html = build_comparison_report(dataset)

        assert "Line Net Flow" in html
        assert "No line interchange data available." not in html
        assert "Plotly.newPlot" in html

    def test_unresolved_corridors_surface_in_unmapped_line(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        decomp_dir, novomodelo_output_dir = self._case_dirs(tmp_path)
        # NE (code 3) has no realizing line in this case's lines.json.
        _patch_network(
            monkeypatch,
            id_map=_decomp_id_map_three_subsystems(),
            interc_frame=_dec_oper_interc_frame(de=1, para=3, origem_mw=90.0),
            novomodelo_line_means=pl.DataFrame(
                {"entity_id": [0], "stage_id": [0], "net_flow_mw": [240.0]}
            ),
        )

        dataset = build_decomp_dataset(decomp_dir, novomodelo_output_dir)

        assert dataset.metadata["unmapped"]["line"] == [[1, 3]]
        line_rows = dataset.tidy.filter(pl.col("entity_type") == "line")
        assert line_rows.is_empty()
