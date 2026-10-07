"""Constraints tab tests for ``comparators.decomp.results``.

Covers the register-term lookup helpers (stage-frame, storage, term
dispatch, RHE achieved LHS), the DECOMP-side generic-constraint LHS
derivation (``_generic_constraint_lhs_decomp``), and the Constraints tab's
``gc_*`` metadata in ``build_decomp_dataset``.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from novomodelo_bridge.comparators.decomp.results import (
    _CONSTRAINT_NAME_RE,
    _GC_LHS_SCHEMA,
    _UNSUPPORTED_TERM_VARIABLES,
    _DecompConstraintLookups,
    _generic_constraint_lhs_decomp,
    _rhe_lhs_lookup,
    _stage_frame_to_lookup,
    _storage_lookup,
    _term_lookup_value,
    build_decomp_dataset,
)
from novomodelo_bridge.comparators.report_builder import build_comparison_report
from novomodelo_bridge.decomp.constraint_registers import (
    ConstraintCensus,
    ConstraintRecord,
    ConstraintTerm,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap
from tests.comparators.conftest import (
    _aligned_fixture,
    _extract_tab_content,
    _no_dec_oper,
    _patch_aligned_frames,
    _usih_frame,
    _write_generic_constraints_case,
)
from tests.conftest import _FakeDadger as _ConstraintFakeDadger
from tests.conftest import make_decomp_case


def _re_record(
    constraint_id: int,
    terms: tuple[ConstraintTerm, ...],
    *,
    stage_start: int = 0,
    stage_end: int = 1,
    family: str = "RE",
) -> ConstraintRecord:
    """A minimal `ConstraintRecord` fixture carrying only the fields
    `_generic_constraint_lhs_decomp` reads (`family`, `constraint_id`,
    `stage_start`/`stage_end`, `terms`); `bounds` is never read by the
    LHS derivation."""
    return ConstraintRecord(
        family=family,
        constraint_id=constraint_id,
        stage_start=stage_start,
        stage_end=stage_end,
        terms=terms,
        bounds={},
        per_block=family in ("RE", "HQ"),
    )


def _gc(cid: int, name: str, expression: str = "") -> dict[str, object]:
    """A minimal ``generic_constraints.json`` entry: only ``id``/``name``
    are read by `_generic_constraint_lhs_decomp` itself."""
    return {"id": cid, "name": name, "expression": expression}


class TestStageFrameToLookup:
    """`_stage_frame_to_lookup`: one `dec_oper_*` column -> {(code, stage):
    value}, 1-based `estagio` converted to 0-based `stage_id`."""

    def test_builds_zero_based_stage_lookup(self) -> None:
        frame = pl.DataFrame(
            {"codigo_usina": [10, 10], "estagio": [1, 2], "geracao_MW": [100.0, 90.0]}
        )
        assert _stage_frame_to_lookup(frame, "geracao_MW") == {
            (10, 0): 100.0,
            (10, 1): 90.0,
        }

    def test_empty_frame_returns_empty_dict(self) -> None:
        assert _stage_frame_to_lookup(pl.DataFrame(), "geracao_MW") == {}

    def test_missing_column_returns_empty_dict(self) -> None:
        frame = pl.DataFrame({"codigo_usina": [10], "estagio": [1]})
        assert _stage_frame_to_lookup(frame, "geracao_MW") == {}

    def test_null_values_are_skipped(self) -> None:
        frame = pl.DataFrame(
            {"codigo_usina": [10, 11], "estagio": [1, 1], "geracao_MW": [100.0, None]}
        )
        assert _stage_frame_to_lookup(frame, "geracao_MW") == {(10, 0): 100.0}


class TestStorageLookup:
    """`_storage_lookup`: absolute storage = useful volume + Vmin, Vmin
    resolved via the id map onto the novomodelo-side `min_storage_hm3` registry."""

    def test_adds_min_storage_floor_via_id_map(self) -> None:
        hydro_frame = pl.DataFrame(
            {"codigo_usina": [10], "estagio": [1], "volume_util_final_hm3": [120.0]}
        )
        id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(10,))
        assert _storage_lookup(hydro_frame, id_map, {0: 30.0}) == {(10, 0): 150.0}

    def test_unmapped_code_is_excluded(self) -> None:
        hydro_frame = pl.DataFrame(
            {"codigo_usina": [99], "estagio": [1], "volume_util_final_hm3": [120.0]}
        )
        id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(10,))
        assert _storage_lookup(hydro_frame, id_map, {0: 30.0}) == {}

    def test_missing_min_storage_entry_is_excluded(self) -> None:
        hydro_frame = pl.DataFrame(
            {"codigo_usina": [10], "estagio": [1], "volume_util_final_hm3": [120.0]}
        )
        id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(10,))
        assert _storage_lookup(hydro_frame, id_map, {}) == {}


class TestTermLookupValue:
    """`_term_lookup_value`: dispatches one register term to its lookup."""

    def _lookups(self) -> _DecompConstraintLookups:
        return _DecompConstraintLookups(
            hydro_generation={(10, 0): 100.0},
            thermal_generation={(5, 0): 30.0},
            flow={"QDEF": {(10, 0): 80.0}},
            storage={(10, 0): 500.0},
        )

    def test_generation_term(self) -> None:
        term = ConstraintTerm(code=10, coefficient=1.0, variable="generation")
        assert _term_lookup_value(term, self._lookups(), 0) == 100.0

    def test_thermal_generation_term(self) -> None:
        term = ConstraintTerm(code=5, coefficient=1.0, variable="thermal_generation")
        assert _term_lookup_value(term, self._lookups(), 0) == 30.0

    def test_varm_term(self) -> None:
        term = ConstraintTerm(code=10, coefficient=1.0, variable="VARM")
        assert _term_lookup_value(term, self._lookups(), 0) == 500.0

    def test_flow_term(self) -> None:
        term = ConstraintTerm(code=10, coefficient=1.0, variable="QDEF")
        assert _term_lookup_value(term, self._lookups(), 0) == 80.0

    def test_unresolvable_stage_returns_none(self) -> None:
        term = ConstraintTerm(code=10, coefficient=1.0, variable="generation")
        assert _term_lookup_value(term, self._lookups(), 5) is None

    def test_unsupported_variable_returns_none(self) -> None:
        term = ConstraintTerm(code=0, coefficient=1.0, variable="interchange")
        assert _term_lookup_value(term, self._lookups(), 0) is None


class TestUnsupportedTermVariables:
    def test_contains_interchange_and_qbom(self) -> None:
        assert _UNSUPPORTED_TERM_VARIABLES == frozenset({"interchange", "QBOM"})


class TestConstraintNameRe:
    """`_CONSTRAINT_NAME_RE`: recovers a novomodelo generic constraint's source
    family + register id from the emitter-authored ``name`` field."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("RE_401", ("RE", "401")),
            ("HQ_12", ("HQ", "12")),
            ("HV_5", ("HV", "5")),
            ("RHE_101", ("RHE", "101")),
        ],
    )
    def test_matches_known_prefixes(self, name: str, expected: tuple[str, str]) -> None:
        match = _CONSTRAINT_NAME_RE.match(name)
        assert match is not None
        assert match.groups() == expected

    @pytest.mark.parametrize("name", ["VminOP_1", "AGRINT_3", "RE401", ""])
    def test_rejects_unknown_names(self, name: str) -> None:
        assert _CONSTRAINT_NAME_RE.match(name) is None


class TestRheLhsLookup:
    """`_rhe_lhs_lookup`: RHE achieved LHS = valor_MW (the achieved value)."""

    def test_uses_valor_as_the_achieved_lhs(self) -> None:
        frame = pl.DataFrame(
            {
                "estagio": [4],
                "no": [4],
                "cenario": [1],
                "codigo_restricao": [101],
                "valor_MW": [34550.64],
                "violacao_absoluta_MW": [0.0],
            }
        )
        assert _rhe_lhs_lookup(frame) == {101: {3: 34550.64}}

    def test_uses_valor_not_the_bound_during_violation(self) -> None:
        # valor_MW is the achieved LHS; valor_MW + violacao_absoluta_MW would be
        # the target/bound (Meta), which must NOT be used as the LHS.
        frame = pl.DataFrame(
            {
                "estagio": [4],
                "no": [4],
                "cenario": [1],
                "codigo_restricao": [115],
                "valor_MW": [2951.58],
                "violacao_absoluta_MW": [145.83],
            }
        )
        lookup = _rhe_lhs_lookup(frame)
        assert lookup[115][3] == pytest.approx(2951.58)

    def test_empty_frame_returns_empty_dict(self) -> None:
        assert _rhe_lhs_lookup(pl.DataFrame()) == {}


class TestGenericConstraintLhsDecomp:
    """`_generic_constraint_lhs_decomp`: the DECOMP-side LHS derivation.
    The census/id map now come from the shared `DecompCase`
    (`case.dadger`/`case.id_map`)
    instead of a per-call `_decomp_constraint_context` re-parse -- these
    tests build that `case` via `make_decomp_case` and patch the public
    `read_constraints` seam rather than the now-removed private helper."""

    def test_no_constraints_returns_empty_schema(self, tmp_path: Path) -> None:
        case = make_decomp_case(tmp_path)

        result = _generic_constraint_lhs_decomp(case, tmp_path, [])

        assert result.schema == _GC_LHS_SCHEMA
        assert result.is_empty()

    def test_re_hydro_generation_terms_sum_by_coefficient(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        record = _re_record(
            401,
            (
                ConstraintTerm(code=155, coefficient=1.0, variable="generation"),
                ConstraintTerm(code=157, coefficient=1.0, variable="generation"),
            ),
        )
        census = ConstraintCensus(
            by_family={"RE": (record,), "HQ": (), "HV": (), "HE": ()}
        )
        id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(155, 157))
        case = make_decomp_case(tmp_path, dadger=_ConstraintFakeDadger(), id_map=id_map)
        monkeypatch.setattr(
            "novomodelo_bridge.decomp.constraint_registers.read_constraints",
            lambda *_a, **_k: census,
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_usih",
            lambda *_a, **_k: _usih_frame(
                [
                    {"codigo_usina": 155, "estagio": 1, "geracao_MW": 3000.0},
                    {"codigo_usina": 157, "estagio": 1, "geracao_MW": 3077.78},
                ]
            ),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_usit",
            _no_dec_oper,
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_rhesoft",
            lambda *_a, **_k: pl.DataFrame(),
        )

        result = _generic_constraint_lhs_decomp(case, tmp_path, [_gc(0, "RE_401")])

        row = result.row(0, named=True)
        assert row["constraint_id"] == 0
        assert row["stage_id"] == 0
        assert row["lhs_value"] == pytest.approx(6077.78)

    def test_hv_varm_term_uses_absolute_storage_floor(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        record = _re_record(
            9001,
            (ConstraintTerm(code=10, coefficient=1.0, variable="VARM"),),
            family="HV",
            stage_start=0,
            stage_end=0,
        )
        census = ConstraintCensus(
            by_family={"RE": (), "HQ": (), "HV": (record,), "HE": ()}
        )
        id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(10,))
        case = make_decomp_case(tmp_path, dadger=_ConstraintFakeDadger(), id_map=id_map)
        monkeypatch.setattr(
            "novomodelo_bridge.decomp.constraint_registers.read_constraints",
            lambda *_a, **_k: census,
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_usih",
            lambda *_a, **_k: _usih_frame(
                [{"codigo_usina": 10, "estagio": 1, "volume_util_final_hm3": 120.0}]
            ),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_usit",
            _no_dec_oper,
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_rhesoft",
            lambda *_a, **_k: pl.DataFrame(),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.novomodelo_readers."
            "read_novomodelo_hydro_metadata",
            lambda *_a, **_k: {0: {"min_storage_hm3": 30.0}},
        )

        result = _generic_constraint_lhs_decomp(case, tmp_path, [_gc(3, "HV_9001")])

        assert result.to_dicts() == [
            {"constraint_id": 3, "stage_id": 0, "lhs_value": 150.0}
        ]

    def test_rhe_soft_constraint_lhs_is_the_achieved_valor(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A soft (RHE) constraint's per-stage ``lhs_value`` is the
        operation's achieved value (``valor_MW``) -- NOT valor + the shortfall
        (which would be the target/bound ``Meta``). Verified on a constructed
        fixture, independent of any census (RHE reads `DecOperRheSoft`
        directly); the shared case's ``dadger`` is a bare, register-less
        fake -- ``read_constraints`` degrades it to an empty census, which
        the RHE branch never consults."""
        case = make_decomp_case(
            tmp_path,
            dadger=_ConstraintFakeDadger(),
            id_map=DecompIdMap(bus_codes=(), bus_names=()),
        )
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
                    "estagio": [4],
                    "no": [4],
                    "cenario": [1],
                    "codigo_restricao": [115],
                    "limite_MW": [3097.40],
                    "valor_MW": [2951.58],
                    "violacao_absoluta_MW": [145.83],
                    "violacao_percentual": [1.412423],
                }
            ),
        )

        result = _generic_constraint_lhs_decomp(case, tmp_path, [_gc(7, "RHE_115")])

        rows = result.to_dicts()
        assert len(rows) == 1
        assert rows[0]["constraint_id"] == 7
        assert rows[0]["stage_id"] == 3
        assert rows[0]["lhs_value"] == pytest.approx(2951.58)

    def test_interchange_term_skips_whole_constraint_novomodelo_only(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An RE record mixing a resolvable ``generation`` term with an
        unresolvable ``interchange`` term must not fabricate a partial LHS
        from the resolvable term alone -- the whole constraint renders
        novomodelo-only."""
        record = _re_record(
            405,
            (
                ConstraintTerm(code=141, coefficient=1.0, variable="generation"),
                ConstraintTerm(
                    code=0,
                    coefficient=1.0,
                    variable="interchange",
                    submarket_de="SE",
                    submarket_para="S",
                ),
            ),
        )
        census = ConstraintCensus(
            by_family={"RE": (record,), "HQ": (), "HV": (), "HE": ()}
        )
        id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(141,))
        case = make_decomp_case(tmp_path, dadger=_ConstraintFakeDadger(), id_map=id_map)
        monkeypatch.setattr(
            "novomodelo_bridge.decomp.constraint_registers.read_constraints",
            lambda *_a, **_k: census,
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_usih",
            lambda *_a, **_k: _usih_frame(
                [{"codigo_usina": 141, "estagio": 1, "geracao_MW": 1000.0}]
            ),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_usit",
            _no_dec_oper,
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_rhesoft",
            lambda *_a, **_k: pl.DataFrame(),
        )

        result = _generic_constraint_lhs_decomp(case, tmp_path, [_gc(2, "RE_405")])

        assert result.is_empty()

    def test_unrecognized_name_skips_constraint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        case = make_decomp_case(
            tmp_path,
            dadger=_ConstraintFakeDadger(),
            id_map=DecompIdMap(bus_codes=(), bus_names=()),
        )
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
            lambda *_a, **_k: pl.DataFrame(),
        )

        result = _generic_constraint_lhs_decomp(case, tmp_path, [_gc(9, "VminOP_1")])

        assert result.is_empty()

    def test_empty_census_yields_no_re_hq_hv_rows(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A shared case whose census carries no matching RE/HQ/HV record
        (a bare, register-less fake ``dadger`` -> ``read_constraints``
        degrades to an empty census) must degrade RE/HQ/HV to novomodelo-only,
        never raise, even when the matching ``dec_oper_usih`` generation
        value is otherwise available."""
        case = make_decomp_case(
            tmp_path,
            dadger=_ConstraintFakeDadger(),
            id_map=DecompIdMap(bus_codes=(), bus_names=()),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_usih",
            lambda *_a, **_k: _usih_frame(
                [{"codigo_usina": 155, "estagio": 1, "geracao_MW": 3000.0}]
            ),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_usit",
            _no_dec_oper,
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_rhesoft",
            lambda *_a, **_k: pl.DataFrame(),
        )

        result = _generic_constraint_lhs_decomp(case, tmp_path, [_gc(0, "RE_401")])

        assert result.is_empty()


class TestBuildDecompDatasetConstraints:
    """Fills ``gc_constraints``/``gc_bounds``/``gc_lhs_newave``/
    ``gc_lhs_novomodelo`` -- the novomodelo-side pieces reused verbatim from
    `constraints`, the DECOMP-side LHS newly derived."""

    def test_no_generic_constraints_case_renders_empty_no_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        case_dir = tmp_path / "case"
        output_dir = _write_generic_constraints_case(case_dir, [], [])

        dataset = build_decomp_dataset(case_dir, output_dir)

        assert dataset.render.gc_constraints == []
        assert dataset.render.gc_lhs_newave.is_empty()
        assert dataset.render.gc_lhs_novomodelo.is_empty()
        html = build_comparison_report(dataset)  # must not raise
        constraints_tab = _extract_tab_content(html, "tab-constraints")
        assert "Generic Constraints — LHS vs Bound" in constraints_tab

    def test_populated_case_wires_gc_metadata(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
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

        assert dataset.render.gc_constraints == constraints
        assert dataset.render.gc_bounds.height == 1
        nw_row = dataset.render.gc_lhs_newave.row(0, named=True)
        assert nw_row["constraint_id"] == 0
        assert nw_row["stage_id"] == 0
        assert nw_row["lhs_value"] == pytest.approx(2951.58)
        cb_row = dataset.render.gc_lhs_novomodelo.row(0, named=True)
        assert cb_row == {"constraint_id": 0, "stage_id": 0, "lhs_value": 3000.0}

        html = build_comparison_report(dataset)
        constraints_tab = _extract_tab_content(html, "tab-constraints")
        assert "Generic Constraints — LHS vs Bound" in constraints_tab
        assert "Plotly.newPlot" in constraints_tab

    def test_rhe_lhs_forwards_overrides_and_bound_stays_raw_nonnegative(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Regression guard: the RHE novomodelo-side LHS now resolves
        ``@rho_acum_h{id}`` via the LP's per-stage override (same mechanism as
        VminOP), forwarded through ``evaluate_lhs_novomodelo``'s third positional
        arg -- and, unlike the source-model VminOP path, the RHE bound is
        NEVER useful-energy-shifted (dead volume can exceed the bound for
        DECOMP RHE), so it must reach the dataset unchanged and non-negative.
        """
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
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
        raw_bound_lower = 3097.40
        bound_rows = [
            {
                "constraint_id": 0,
                "stage_id": 0,
                "block_id": None,
                "bound_lower": raw_bound_lower,
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
        calls: list[tuple[object, ...]] = []

        def _capturing_evaluate_lhs_novomodelo(
            *args: object, **_kw: object
        ) -> pl.DataFrame:
            calls.append(args)
            return pl.DataFrame(
                {"constraint_id": [0], "stage_id": [0], "lhs_value": [3000.0]}
            )

        monkeypatch.setattr(
            "novomodelo_bridge.comparators.constraints.evaluate_lhs_novomodelo",
            _capturing_evaluate_lhs_novomodelo,
        )

        dataset = build_decomp_dataset(case_dir, output_dir)

        # evaluate_lhs_novomodelo received a third positional arg -- the
        # rho_acum_overrides mapping ({} here, since no
        # generic_parameters.json was written by this fixture).
        assert len(calls) == 1
        assert calls[0][2] == {}

        bound_row = dataset.render.gc_bounds.row(0, named=True)
        assert bound_row["bound_lower"] == pytest.approx(raw_bound_lower)
        assert bound_row["bound_lower"] >= 0.0
