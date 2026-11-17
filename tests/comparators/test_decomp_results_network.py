"""Network-tab tests for ``comparators.decomp.results``.

Second carve out of the legacy ``test_decomp_results_compare.py`` mega file:
corridor/line alignment, the DECOMP interchange side, line entity
names and result comparisons, and the Network tab's
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
    _line_entity_names,
    _line_result_comparisons,
    _read_novomodelo_lines_index,
    build_decomp_dataset,
)
from novomodelo_bridge.comparators.report_builder import build_comparison_report
from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.decomp.id_map import DecompIdMap
from tests.comparators.conftest import (
    _aligned_fixture,
    _patch_aligned_frames,
    _patch_shared_case,
    _unmapped_codes,
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
    (system_dir / "lines.json").write_text(
        json.dumps({"lines": lines}), encoding="utf-8"
    )
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
    -> the novomodelo (line_id, sign) that carries it."""

    def test_direct_line_corridor_maps_with_positive_sign(self, tmp_path: Path) -> None:
        output_dir = _write_lines_json(tmp_path, [_line_entry(0, 0, 1)])

        alignment = _corridor_line_alignment(output_dir)

        assert alignment[(0, 1)] == (0, 1)

    def test_reverse_declared_line_gets_negative_sign(self, tmp_path: Path) -> None:
        """A novomodelo line declared S -> SE realizes the SE -> S corridor with
        sign -1, orienting the line's own net_flow_mw onto the requested
        direction."""
        output_dir = _write_lines_json(tmp_path, [_line_entry(0, 1, 0)])

        alignment = _corridor_line_alignment(output_dir)

        assert alignment[(0, 1)] == (0, -1)

    def test_a_transhipment_line_carries_only_its_own_corridor(
        self, tmp_path: Path
    ) -> None:
        """SE<->IV and IV<->S each carry their own corridor; with no direct
        SE-S line, SE -> S is not built out of the two."""
        id_map = _decomp_id_map_three_subsystems()
        transhipment = id_map.transhipment_bus_id
        output_dir = _write_lines_json(
            tmp_path,
            [
                _line_entry(0, 0, transhipment),
                _line_entry(1, transhipment, 1),
            ],
        )

        alignment = _corridor_line_alignment(output_dir)

        assert alignment[(0, transhipment)] == (0, 1)
        assert alignment[(transhipment, 0)] == (0, -1)
        assert alignment[(transhipment, 1)] == (1, 1)
        assert (0, 1) not in alignment

    def test_a_line_declared_in_the_corridor_direction_wins(
        self, tmp_path: Path
    ) -> None:
        output_dir = _write_lines_json(
            tmp_path, [_line_entry(0, 1, 0), _line_entry(1, 0, 1)]
        )

        alignment = _corridor_line_alignment(output_dir)

        assert alignment[(0, 1)] == (1, 1)
        assert alignment[(1, 0)] == (0, 1)

    def test_corridor_with_no_realizing_path_is_absent(self, tmp_path: Path) -> None:
        # Only SE-S is wired; NE (bus id 2) has no line at all.
        output_dir = _write_lines_json(tmp_path, [_line_entry(0, 0, 1)])

        alignment = _corridor_line_alignment(output_dir)

        assert (0, 2) not in alignment
        assert (2, 0) not in alignment

    def test_missing_lines_json_yields_an_empty_alignment(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        assert _corridor_line_alignment(output_dir) == {}

    def test_ambiguous_duplicate_line_pair_is_excluded_not_guessed(
        self, tmp_path: Path
    ) -> None:
        """Two lines declared for the same ordered bus pair make that leg
        ambiguous -- both are dropped rather than one picked arbitrarily."""
        output_dir = _write_lines_json(
            tmp_path, [_line_entry(0, 0, 1), _line_entry(1, 0, 1)]
        )

        with dx.collect() as collected:
            assert (0, 1) not in _read_novomodelo_lines_index(output_dir)
        assert (0, 1) not in _corridor_line_alignment(output_dir)

        [diag] = collected
        assert diag.code == "line-bus-pair-ambiguous"
        assert diag.severity is dx.Severity.WARNING
        assert diag.title == "Bus pairs joined by several lines (1)"
        assert diag.table is not None
        assert diag.table.rows == [[0, 1]]


#: The names ``dec_oper_interc`` prints beside each code of
#: :func:`_decomp_id_map_three_subsystems`, and the code-less transhipment node.
_INTERC_NAMES = {1: "SE", 2: "S", 3: "NE", None: "IV"}


def _dec_oper_interc_frame(
    *, de: int | None, para: int | None, origem_mw: float, perdas_mw: float = 5.0
) -> pl.DataFrame:
    """One corridor, one stage, one node -- the patamar-null aggregate row
    ``_interc_side`` reads via ``_stage_rows``. A ``None`` endpoint is the
    transhipment node, which ``dec_oper_interc`` names without a code."""
    return pl.DataFrame(
        {
            "estagio": [1],
            "no": [1],
            "cenario": [1],
            "patamar": [None],
            "codigo_submercado_de": [de],
            "nome_submercado_de": [_INTERC_NAMES[de]],
            "codigo_submercado_para": [para],
            "nome_submercado_para": [_INTERC_NAMES[para]],
            "intercambio_origem_MW": [origem_mw],
            "intercambio_destino_MW": [origem_mw - perdas_mw],
            "perdas_MW": [perdas_mw],
        },
        schema_overrides={
            "codigo_submercado_de": pl.Int64,
            "codigo_submercado_para": pl.Int64,
        },
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

    def test_corridors_to_the_code_less_transhipment_node_resolve_by_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """DECOMP reports the corridors through the Itaipu 60 Hz ``IV`` node
        with a null code; each lands on its own transhipment line."""
        id_map = _decomp_id_map_three_subsystems()
        transhipment = id_map.transhipment_bus_id
        output_dir = _write_lines_json(
            tmp_path,
            [_line_entry(0, 0, transhipment), _line_entry(1, transhipment, 1)],
        )
        self._patch_interc(
            monkeypatch,
            pl.concat(
                [
                    _dec_oper_interc_frame(de=1, para=None, origem_mw=180.0),
                    _dec_oper_interc_frame(de=None, para=2, origem_mw=150.0),
                ]
            ),
        )

        result, unresolved = _interc_side(tmp_path, output_dir, id_map)

        assert unresolved == []
        by_line = {
            row["entity_id"]: row["net_flow_mw"] for row in result.iter_rows(named=True)
        }
        assert by_line == {0: 180.0, 1: 150.0}

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

    def test_a_code_less_endpoint_with_an_unknown_name_is_unresolved(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A null code whose name is no bus's is reported with the ``-1``
        stand-in, never raised."""
        id_map = _decomp_id_map_three_subsystems()
        output_dir = _write_lines_json(tmp_path, [_line_entry(0, 0, 1)])
        frame = _dec_oper_interc_frame(de=None, para=2, origem_mw=50.0).with_columns(
            pl.lit("XX").alias("nome_submercado_de")
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

    def test_both_directions_of_a_transhipment_corridor_net_to_one_row(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``SE -> IV`` and ``IV -> SE`` net onto the one SE-IV line."""
        id_map = _decomp_id_map_three_subsystems()
        transhipment = id_map.transhipment_bus_id
        output_dir = _write_lines_json(tmp_path, [_line_entry(0, 0, transhipment)])
        frame = pl.concat(
            [
                _dec_oper_interc_frame(de=1, para=None, origem_mw=180.0),
                _dec_oper_interc_frame(de=None, para=1, origem_mw=70.0),
            ]
        )
        self._patch_interc(monkeypatch, frame)

        result, unresolved = _interc_side(tmp_path, output_dir, id_map)

        assert unresolved == []
        assert result.to_dicts() == [
            {"entity_id": 0, "stage_id": 0, "net_flow_mw": 110.0}
        ]


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

    def test_missing_source_interchange_table_is_noted_and_degrades_to_empty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        id_map = _decomp_id_map_three_subsystems()

        def _boom(*_args: object, **_kwargs: object) -> pl.DataFrame:
            raise FileNotFoundError("dec_oper_interc.csv not found")

        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_interc", _boom
        )

        with dx.collect() as collected:
            results, unresolved = _line_result_comparisons(
                tmp_path, tmp_path, id_map, []
            )

        assert results == []
        assert unresolved == []
        assert [(d.code, d.title) for d in collected] == [
            ("report-file-missing", "dec_oper_interc.csv not found")
        ]

    def test_unreadable_source_interchange_table_is_reported_and_degrades_to_empty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        id_map = _decomp_id_map_three_subsystems()

        def _boom(*_args: object, **_kwargs: object) -> pl.DataFrame:
            raise ValueError("dec_oper_interc.csv parsed empty")

        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_interc", _boom
        )

        with dx.collect() as collected:
            results, unresolved = _line_result_comparisons(
                tmp_path, tmp_path, id_map, []
            )

        assert results == []
        assert unresolved == []
        assert [(d.code, d.title) for d in collected] == [
            ("report-file-unreadable", "dec_oper_interc.csv could not be read")
        ]

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
        this test controls, since ``read_novomodelo_network_lines`` reads
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

        assert _unmapped_codes(dataset, "line") == [[1, 3]]
        line_rows = dataset.tidy.filter(pl.col("entity_type") == "line")
        assert line_rows.is_empty()
