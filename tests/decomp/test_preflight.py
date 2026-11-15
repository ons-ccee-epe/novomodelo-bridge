"""Tests for ``check decomp`` — deck validation without conversion."""

from __future__ import annotations

import inspect
import shutil
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from idecomp.decomp.modelos.dadger import (
    ACALTEFE,
    ACCOTVAZ,
    ACNCHAVE,
    ACNPOSNW,
    ACTIPERH,
    ACVERTJU,
    ACVOLMIN,
)

from novomodelo_bridge.core.diagnostics import Severity, collect
from novomodelo_bridge.core.errors import FieldParseError, UnsupportedSourceError
from novomodelo_bridge.core.libs_index import UNREAD_LINES_CODE
from novomodelo_bridge.core.preflight import (
    CheckItem,
    PreflightResult,
    PreflightVerdict,
    optional_input_advisory,
)
from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.constraint_registers import (
    detect_libs_electrical,
    detect_unconverted_rha,
    drop_unknown_entities,
    read_constraints,
)
from novomodelo_bridge.decomp.converters.cadastro import (
    _SCALAR_AC_SPECS,
    ALL_AC_CLASSES,
    APPLIED_AC_CLASSES,
    DEFERRED_AC_CLASSES,
    CadastroResolutionReport,
    EffectiveCadastro,
    OutOfHorizon,
    ac_overrides_deferred_diagnostic,
    topology_varies_diagnostic,
)
from novomodelo_bridge.decomp.files import DecompFiles
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.pipeline import (
    _read_constraint_census,
    convert_decomp_case,
)
from novomodelo_bridge.decomp.preflight import (
    _ac_coverage,
    _load_factor_check,
    _special_constraint_coverage,
    run_decomp_preflight,
)
from tests.conftest import branching_vazoes, make_decomp_case
from tests.decomp.test_cadastro import _FakeDadger
from tests.decomp.test_constraint_registers import (
    _cm,
    _coeff,
    _decl,
    _he,
    _lv,
    _qdef_ceiling_and_floor,
)
from tests.decomp.test_constraint_registers import _FakeDadger as _ConstraintFakeDadger
from tests.decomp.test_libs_electrical import (
    _re_formula,
    _re_inequacao,
    _re_inequacao_per_pat,
    _StubRestricoes,
)
from tests.decomp.test_network_load import (
    _ID_MAP,
    _calendar_rv3,
    _dp_frame,
    _StubDadger,
)


class TestDiscoveryFailure:
    def test_a_directory_that_is_not_a_deck_blocks(self, tmp_path: Path) -> None:
        result = run_decomp_preflight(tmp_path)
        assert result.verdict is PreflightVerdict.WILL_NOT_CONVERT
        assert result.checks[0].passed is False
        assert result.diagnostics, "a blocking verdict must carry a diagnostic"

    def test_nothing_is_written(self, tmp_path: Path) -> None:
        run_decomp_preflight(tmp_path)
        assert list(tmp_path.iterdir()) == []


def _decomp_files(tmp_path: Path) -> DecompFiles:
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


class TestIdMapReconcile:
    """A typed refusal from ``from_dadger`` (a parse failure or an SB record the
    case cannot carry) and the id map's own invariant checks all degrade to a
    diagnosed ``WILL_NOT_CONVERT`` instead of crashing.
    """

    def _run_with_id_map_error(self, tmp_path: Path, error: Exception):
        files = _decomp_files(tmp_path)
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "novomodelo_bridge.decomp.preflight.discover_decomp_files",
                    return_value=files,
                )
            )
            stack.enter_context(
                patch(
                    "idecomp.decomp.Dadger.read",
                    return_value=object(),
                )
            )
            stack.enter_context(
                patch(
                    "novomodelo_bridge.decomp.id_map.DecompIdMap.from_dadger",
                    side_effect=error,
                )
            )
            return run_decomp_preflight(tmp_path)

    def test_field_parse_error_reconciles_to_will_not_convert(
        self, tmp_path: Path
    ) -> None:
        error = FieldParseError(
            "the deck has no SB records; cannot build the id map",
            field="SB register",
        )
        result = self._run_with_id_map_error(tmp_path, error)

        assert result.verdict is PreflightVerdict.WILL_NOT_CONVERT
        id_map_check = next(c for c in result.checks if c.label == "Entity id map")
        assert id_map_check.passed is False
        assert any(d.code == "source-field-parse" for d in result.diagnostics)

    def test_unsupported_source_error_reconciles_to_will_not_convert(
        self, tmp_path: Path
    ) -> None:
        error = UnsupportedSourceError(
            "dadger.rvN declares a subsystem named IV in SB.",
            path="dadger.rvN",
            field="SB nome_submercado",
        )
        result = self._run_with_id_map_error(tmp_path, error)

        assert result.verdict is PreflightVerdict.WILL_NOT_CONVERT
        id_map_check = next(c for c in result.checks if c.label == "Entity id map")
        assert id_map_check.passed is False
        diagnostic = next(
            d for d in result.diagnostics if d.code == "source-unsupported"
        )
        assert list(diagnostic.notes) == [
            "file: dadger.rvN",
            "field: SB nome_submercado",
        ]

    def test_post_init_value_error_is_still_caught(self, tmp_path: Path) -> None:
        error = ValueError("bus codes must be strictly ascending; got (2, 1)")
        result = self._run_with_id_map_error(tmp_path, error)

        assert result.verdict is PreflightVerdict.WILL_NOT_CONVERT
        id_map_check = next(c for c in result.checks if c.label == "Entity id map")
        assert id_map_check.passed is False


class TestTreeShapeCheck:
    def test_a_trunk_that_branches_fails_as_convert_decomp_refuses_it(
        self, decomp_mini_deck: Path, tmp_path: Path
    ) -> None:
        with branching_vazoes():
            with pytest.raises(UnsupportedSourceError) as raised:
                convert_decomp_case(decomp_mini_deck, tmp_path / "out")
            result = run_decomp_preflight(decomp_mini_deck)

        assert result.verdict is PreflightVerdict.WILL_NOT_CONVERT
        shape = next(c for c in result.checks if c.label.startswith("Tree shape"))
        assert shape.passed is False
        [diagnostic] = [d for d in result.diagnostics if d.code == "source-unsupported"]
        assert diagnostic.summary == str(raised.value)
        assert list(diagnostic.notes) == [
            "file: vazoes.rvN",
            "field: probabilidades cenario",
        ]

    def test_a_terminal_fan_passes_with_its_width(self, decomp_mini_deck: Path) -> None:
        result = run_decomp_preflight(decomp_mini_deck)

        shape = next(c for c in result.checks if c.label.startswith("Tree shape"))
        assert (shape.passed, shape.detail) == (True, "a terminal fan of 2 scenarios")


class TestTabCharacterAdvisory:
    def test_tab_in_a_register_line_warns_and_one_in_a_comment_does_not(
        self, decomp_mini_deck: Path, tmp_path: Path
    ) -> None:
        deck = tmp_path / "deck"
        shutil.copytree(decomp_mini_deck, deck)
        dadger = deck / "dadger.rv2"
        lines = dadger.read_bytes().split(b"\n")
        assert lines[0].startswith(b"&") and lines[6].startswith(b"TE")
        lines[0] = b"&\t" + lines[0][1:]
        lines[6] = lines[6].rstrip(b"\r") + b"\t"
        dadger.write_bytes(b"\n".join(lines))

        result = run_decomp_preflight(deck)

        tab = [d for d in result.diagnostics if d.code == "source-tab-character"]
        assert len(tab) == 1 and tab[0].severity is Severity.WARNING
        assert tab[0].table is not None
        assert tab[0].table.rows == [["dadger.rv2", "7"]]
        assert result.verdict is PreflightVerdict.WARNINGS

    def test_mini_deck_has_no_tab_warning(self, decomp_mini_deck: Path) -> None:
        result = run_decomp_preflight(decomp_mini_deck)

        assert "source-tab-character" not in {d.code for d in result.diagnostics}
        assert any(
            check.detail == "none found" and check.passed for check in result.checks
        )


class TestEmptyHydroRegistry:
    def test_an_empty_hidr_fails_the_cadastro_check(
        self, decomp_mini_deck: Path, tmp_path: Path
    ) -> None:
        deck = tmp_path / "deck"
        shutil.copytree(decomp_mini_deck, deck)
        (deck / "hidr.dat").write_bytes(b"")

        with pytest.warns(UserWarning):
            result = run_decomp_preflight(deck)

        [check] = [
            c for c in result.checks if c.label == "AC cadastro overrides resolvable"
        ]
        assert check.passed is False
        assert check.detail is not None
        assert check.detail.startswith("hidr.dat holds no hydro plant records")
        assert result.verdict is PreflightVerdict.WILL_NOT_CONVERT


class TestOptionalInputAdvisory:
    """DECOMP adopts the shared
    :func:`novomodelo_bridge.core.preflight.optional_input_advisory` helper for all six
    optional ``DecompFiles`` fields, in place of the old two-field hard-coded
    loop.
    """

    @staticmethod
    def _files(tmp_path: Path, **overrides: Path | None) -> DecompFiles:
        return DecompFiles(
            revision="rv0",
            dadger=tmp_path / "dadger.rv0",
            vazoes=tmp_path / "vazoes.rv0",
            hidr=tmp_path / "hidr.dat",
            dadgnl=overrides.get("dadgnl"),
            renovaveis=overrides.get("renovaveis"),
            polinjus=overrides.get("polinjus"),
            libs_restricao_eletrica=overrides.get("libs_restricao_eletrica"),
            cortesh=overrides.get("cortesh"),
            cortes=overrides.get("cortes"),
            libs_index=overrides.get("libs_index"),
        )

    def test_reports_every_absent_decomp_optional(self, tmp_path: Path) -> None:
        files = self._files(tmp_path)

        checks, diagnostics = optional_input_advisory(files)

        assert len(checks) == 8
        assert all(check.passed for check in checks)
        [advisory] = diagnostics
        assert advisory.code == "optional-file-absent"
        assert advisory.severity is Severity.INFO
        assert advisory.table is not None

        reported = {row[0] for row in advisory.table.rows}
        assert reported == {
            "dadgnl",
            "renovaveis",
            "polinjus",
            "libs_restricao_eletrica",
            "cortesh",
            "cortes",
            "libs_index",
            "mlt",
        }

    def test_skips_present_decomp_optional(self, tmp_path: Path) -> None:
        files = self._files(tmp_path, dadgnl=tmp_path / "dadgnl.rv0")

        _checks, diagnostics = optional_input_advisory(files)

        [advisory] = diagnostics
        assert advisory.table is not None
        assert len(advisory.table.rows) == 7
        assert ["dadgnl"] not in advisory.table.rows

    def test_run_decomp_preflight_uses_the_shared_advisory_helper(self) -> None:
        source = inspect.getsource(run_decomp_preflight)
        assert "optional_input_advisory" in source
        assert '("dadgnl", "renovaveis")' not in source


class TestAcCoverageRegistry:
    """The resolver's ``APPLIED_AC_CLASSES`` registry and the reflected idecomp
    ``AC`` universe (``ALL_AC_CLASSES``) stay consistent with each other, and the
    derived deferred bucket is populated by enumerate-and-diff rather than a
    hand-maintained list — proven by three families
    (``NCHAVE``/``TIPERH``/``VERTJU``) the old blanket warning never named.
    """

    def test_coverage_registry_applied_classes_count_is_19(self) -> None:
        assert len(APPLIED_AC_CLASSES) == 19
        assert ACNPOSNW in APPLIED_AC_CLASSES

    def test_coverage_registry_applies_altefe(self) -> None:
        assert ACALTEFE in APPLIED_AC_CLASSES

    def test_coverage_registry_scalar_specs_subset_of_applied(self) -> None:
        scalar_classes = {spec.ac_class for spec in _SCALAR_AC_SPECS}
        assert scalar_classes <= APPLIED_AC_CLASSES

    def test_coverage_registry_applied_subset_of_all_ac_classes(self) -> None:
        assert APPLIED_AC_CLASSES <= ALL_AC_CLASSES

    def test_coverage_registry_all_ac_classes_has_at_least_25_members(self) -> None:
        # idecomp 1.13.0 exposes exactly 25; ``>=`` stays robust to a future add.
        assert len(ALL_AC_CLASSES) >= 25

    def test_coverage_registry_deferred_bucket_contains_uncovered_families(
        self,
    ) -> None:
        assert {ACNCHAVE, ACTIPERH, ACVERTJU} <= DEFERRED_AC_CLASSES
        assert DEFERRED_AC_CLASSES == ALL_AC_CLASSES - APPLIED_AC_CLASSES


_NO_LINKS = EffectiveCadastro(base=pd.DataFrame(), n_stages=1, stage_varying={})


class TestAcCoverage:
    """``_ac_coverage``'s two-bucket classification and its
    always-passing summary ``CheckItem`` plus per-bucket diagnostics.
    """

    def test_ac_coverage_mixed_presence_emits_all_three_diagnostics(self) -> None:
        present = pd.DataFrame({"codigo_usina": [1]})
        dadger = _FakeDadger({ACVOLMIN: present, ACCOTVAZ: present, ACALTEFE: present})
        report = CadastroResolutionReport(
            applied={"volume_minimo": 1},
            out_of_horizon=(
                OutOfHorizon(code=7, param="volume_minimo", mes=12, ano=2030),
            ),
        )

        checks, diagnostics = _ac_coverage(dadger, _NO_LINKS, report)

        assert len(checks) == 1
        assert checks[0].label == "AC cadastro-override coverage"
        assert checks[0].passed is True

        by_code = {d.code: d for d in diagnostics}
        assert set(by_code) == {
            "decomp-ac-overrides-applied",
            "decomp-ac-overrides-deferred",
            "decomp-ac-out-of-horizon",
        }
        assert by_code["decomp-ac-overrides-applied"].severity is Severity.INFO
        assert by_code["decomp-ac-overrides-deferred"].severity is Severity.WARNING
        assert by_code["decomp-ac-out-of-horizon"].severity is Severity.INFO

        def registers(code: str) -> list[object]:
            table = by_code[code].table
            assert table is not None
            return [row[0] for row in table.rows]

        assert "COTVAZ" in registers("decomp-ac-overrides-deferred")
        assert "ALTEFE" not in registers("decomp-ac-overrides-deferred")
        assert "ALTEFE" in registers("decomp-ac-overrides-applied")
        assert by_code[
            "decomp-ac-overrides-deferred"
        ] == ac_overrides_deferred_diagnostic(dadger)

        table = by_code["decomp-ac-out-of-horizon"].table
        assert table is not None
        assert table.columns == ["Code", "AC register", "Month", "Year"]
        assert table.rows == [[7, "VOLMIN", 12, 2030]]

    def test_ac_coverage_applied_only_emits_only_info_diagnostic(self) -> None:
        present = pd.DataFrame({"codigo_usina": [1]})
        dadger = _FakeDadger({ACVOLMIN: present})
        report = CadastroResolutionReport(
            applied={"volume_minimo": 1}, out_of_horizon=()
        )

        checks, diagnostics = _ac_coverage(dadger, _NO_LINKS, report)

        assert len(checks) == 1
        assert checks[0].passed is True
        assert len(diagnostics) == 1
        assert diagnostics[0].code == "decomp-ac-overrides-applied"
        assert diagnostics[0].severity is Severity.INFO

    def test_ac_coverage_warns_about_a_cascade_register_that_varies(self) -> None:
        """``check decomp`` raises the WARNING ``convert decomp`` raises for a
        temporal ``AC NUMJUS``, from the same builder."""
        base = pd.DataFrame({1: {"nome_usina": "FURNAS  "}}).T
        effective = EffectiveCadastro(
            base=base,
            n_stages=3,
            stage_varying={},
            downstream_links={1: (5, 5, 7)},
        )
        dadger = _FakeDadger({})
        report = CadastroResolutionReport(applied={}, out_of_horizon=())

        _checks, diagnostics = _ac_coverage(dadger, effective, report)

        expected = topology_varies_diagnostic(effective)
        assert expected is not None
        assert expected.code == "decomp-ac-topology-varies"
        assert expected.severity is Severity.WARNING
        assert diagnostics == [expected]

    def test_ac_coverage_empty_deck_emits_only_the_passing_summary(self) -> None:
        dadger = _FakeDadger({})
        report = CadastroResolutionReport(applied={}, out_of_horizon=())

        checks, diagnostics = _ac_coverage(dadger, _NO_LINKS, report)

        assert len(checks) == 1
        assert checks[0].passed is True
        assert diagnostics == []


#: The plants the synthetic special-constraint deck's FU, CQ and CV terms name.
_SC_ID_MAP = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(21, 31, 40))


class TestSpecialConstraintCoverage:
    """``_special_constraint_coverage``'s converted (RE/HQ/HV/HE,
    bounds-lowered vs generic-emitted) vs deferred (RHA/LIBs-electrical)
    classification, and its always-passing summary ``CheckItem``.
    """

    @staticmethod
    def _synthetic_dadger(**extra: pd.DataFrame) -> _ConstraintFakeDadger:
        """One record per family: RE (FU), HV (VARM), and HQ (QDES, a
        diversion bound) lower to bounds; HE
        (always generic) is the only one that emits as a generic
        constraint — mirrors the register fixtures in
        ``test_decomp_constraint_registers.py``."""
        return _ConstraintFakeDadger(
            re=_decl((10, 1, 2)),
            fu=_coeff((10, 21, 1.0, 1, float("nan")), freq=True),
            hq=_decl((6, 1, 1)),
            cq=_coeff((6, 31, 1.0, 1, "QDES"), tipo=True),
            hv=_decl((7, 1, 2)),
            cv=_coeff((7, 40, 1.0, 1, "VARM"), tipo=True),
            lv=_lv((7, 1, 100.0, 900.0)),
            he=_he((8, 1, 20.0, 2, 0, 1, "PRODRHE.DAT", 3370.0, 1)),
            cm=_cm((8, 3, 1.0)),
            **extra,
        )

    @staticmethod
    def _rha_dadger() -> _ConstraintFakeDadger:
        """The synthetic deck plus one ``HA`` inflow constraint."""
        return TestSpecialConstraintCoverage._synthetic_dadger(
            ha=pd.DataFrame(
                [(1, 1, 2)],
                columns=["codigo_restricao", "estagio_inicial", "estagio_final"],
            )
        )

    def test_mixed_deck_emits_summary_plus_converted_and_deferred_diagnostics(
        self,
    ) -> None:
        dadger = self._rha_dadger()
        census = read_constraints(dadger)

        checks, diagnostics = _special_constraint_coverage(dadger, None, _SC_ID_MAP, {})

        assert len(checks) == 1
        assert checks[0].label == "Special-constraint coverage"
        assert checks[0].passed is True

        by_code = {d.code: d for d in diagnostics}
        assert set(by_code) == {
            "decomp-special-constraints-converted",
            "decomp-rha-family-unconverted",
        }
        assert by_code["decomp-special-constraints-converted"].severity is (
            Severity.INFO
        )
        assert by_code["decomp-rha-family-unconverted"].severity is (Severity.WARNING)
        present_families = [fam for fam, records in census.by_family.items() if records]
        assert checks[0].detail == (
            f"{len(census.to_bounds)} records lowered to bounds, "
            f"{len(census.to_generic)} emitted as generic constraints, "
            f"{len(present_families)} families present in the deck"
        )

    def test_converted_counts_match_the_census_probe(self) -> None:
        dadger = self._synthetic_dadger()
        census = read_constraints(dadger)

        _checks, diagnostics = _special_constraint_coverage(
            dadger, None, _SC_ID_MAP, {}
        )

        converted = next(
            d for d in diagnostics if d.code == "decomp-special-constraints-converted"
        )
        # Probe — asserted against the census's own counts, never
        # a hard-coded oracle.
        assert converted.table is not None
        assert sum(int(str(row[1])) for row in converted.table.rows) == len(
            census.to_bounds
        )
        assert sum(int(str(row[2])) for row in converted.table.rows) == len(
            census.to_generic
        )
        assert len(census.to_bounds) == 3  # RE FU + HV VARM + HQ QDES
        assert len(census.to_generic) == 1  # HE (always generic)

    def test_a_constraint_on_a_missing_entity_is_left_out_as_convert_does(
        self,
    ) -> None:
        """``check decomp`` counts the screened census and raises the screen's
        WARNING, as ``convert decomp`` does."""
        dadger = self._synthetic_dadger()
        _, expected = drop_unknown_entities(
            read_constraints(dadger), hydro_codes=(), thermal_codes=(), pumping_codes=()
        )
        no_plants = DecompIdMap(bus_codes=(1,), bus_names=("SE",))

        checks, diagnostics = _special_constraint_coverage(dadger, None, no_plants, {})

        converted, *unknown = diagnostics
        assert unknown == expected
        assert [d.code for d in unknown] == [
            "decomp-re-plant-not-operated",
            "decomp-rhq-plant-not-operated",
            "decomp-rhv-varm-plant-uncadastred",
        ]
        assert converted.title == "Special constraints converted (1)"
        assert converted.table is not None
        assert [row[0] for row in converted.table.rows] == ["HE"]
        assert checks[0].detail == (
            "0 records lowered to bounds, 1 emitted as generic constraints, "
            "4 families present in the deck"
        )

    def test_conflicting_flow_restrictions_are_reported_as_convert_does(
        self,
    ) -> None:
        dadger = _qdef_ceiling_and_floor()
        id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(118,))
        hidr = pd.DataFrame({118: {"nome_usina": "BILLINGS    "}}).T
        hidr.index.name = "codigo_usina"
        effective = EffectiveCadastro(base=hidr, n_stages=2, stage_varying={})
        with collect() as converted:
            _read_constraint_census(dadger, effective, id_map, {})

        _checks, diagnostics = _special_constraint_coverage(
            dadger, None, id_map, {}, plant_names={118: "BILLINGS"}
        )

        [expected] = converted
        assert expected.code == "decomp-conflicting-flow-restrictions"
        assert [d for d in diagnostics if d.code == expected.code] == [expected]

    def test_a_conflict_on_a_missing_pumping_station_is_not_reported(self) -> None:
        id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(118,))

        _checks, diagnostics = _special_constraint_coverage(
            _qdef_ceiling_and_floor(variable="QBOM"), None, id_map, {}
        )

        assert [d.code for d in diagnostics] == ["decomp-rhq-qbom-no-station"]

    def test_deferred_rha_diagnostic_is_sourced_from_the_helper_verbatim(
        self,
    ) -> None:
        dadger = self._rha_dadger()
        expected = detect_unconverted_rha(dadger)
        assert expected is not None

        _checks, diagnostics = _special_constraint_coverage(
            dadger, None, _SC_ID_MAP, {}
        )

        rha_diagnostic = next(
            d for d in diagnostics if d.code == "decomp-rha-family-unconverted"
        )
        assert rha_diagnostic == expected

    def test_deferred_libs_electrical_diagnostic_is_sourced_from_the_e1_helper(
        self, tmp_path: Path
    ) -> None:
        dadger = _ConstraintFakeDadger()
        libs_file = tmp_path / "lib_restricao-eletrica-especial.csv"
        expected = detect_libs_electrical(libs_file)
        assert expected is not None  # sanity: the fixture trips the LIBs token

        _checks, diagnostics = _special_constraint_coverage(
            dadger, libs_file, _SC_ID_MAP, {}
        )

        assert diagnostics == [expected]

    def test_a_libs_file_with_a_long_form_card_raises_no_libs_warning(
        self, tmp_path: Path
    ) -> None:
        _checks, diagnostics = _special_constraint_coverage(
            _ConstraintFakeDadger(),
            tmp_path / "lib_restricao-eletrica-especial.csv",
            _SC_ID_MAP,
            {},
            libs_long_form=True,
        )

        assert "decomp-libs-electrical-present" not in {d.code for d in diagnostics}

    def test_no_special_constraints_deck_emits_only_the_passing_summary(
        self,
    ) -> None:
        dadger = _ConstraintFakeDadger()

        checks, diagnostics = _special_constraint_coverage(dadger, None, _SC_ID_MAP, {})

        assert len(checks) == 1
        assert checks[0].passed is True
        assert checks[0].detail == (
            "0 records lowered to bounds, 0 emitted as generic constraints, "
            "0 families present in the deck"
        )
        assert diagnostics == []


class TestLibsElectricalWarning:
    """``check decomp`` warns that the LIBS electrical constraints are not
    converted only when ``convert decomp`` reads no long-form card from the
    file, as ``convert decomp`` does."""

    _LIBS_FILE = "lib_restricao-eletrica-especial.csv"

    def _run(
        self,
        decomp_mini_deck: Path,
        tmp_path: Path,
        cards: object | None,
        reader: MagicMock | None = None,
        *,
        text: str = "",
    ) -> PreflightResult:
        """Check the mini deck with a LIBS file holding *text*, whose cards are
        *cards* when given, read by *reader* in place of the real reader when
        given."""
        deck = tmp_path / "deck"
        shutil.copytree(decomp_mini_deck, deck)
        with (deck / "indices.csv").open("a", encoding="utf-8") as index:
            index.write(f" RESTRICAO-ELETRICA-ESPECIAL ;eletricas ;{self._LIBS_FILE}\n")
        (deck / self._LIBS_FILE).write_text(text, encoding="utf-8")
        with ExitStack() as stack:
            if cards is not None:
                stack.enter_context(
                    patch.object(DecompCase, "libs_restricao_eletrica", new=cards)
                )
            if reader is not None:
                stack.enter_context(
                    patch(
                        "novomodelo_bridge.decomp.converters.libs_electrical"
                        ".read_libs_electrical",
                        new=reader,
                    )
                )
            return run_decomp_preflight(deck)

    def test_a_file_with_a_long_form_card_raises_no_libs_warning(
        self, decomp_mini_deck: Path, tmp_path: Path
    ) -> None:
        result = self._run(
            decomp_mini_deck,
            tmp_path,
            _StubRestricoes(),
            MagicMock(return_value=object()),
        )

        codes = {d.code for d in result.diagnostics}
        assert "decomp-libs-electrical-present" not in codes

    def test_a_file_without_a_long_form_card_warns(
        self, decomp_mini_deck: Path, tmp_path: Path
    ) -> None:
        result = self._run(
            decomp_mini_deck, tmp_path, _StubRestricoes(), MagicMock(return_value=None)
        )

        codes = {d.code for d in result.diagnostics}
        assert "decomp-libs-electrical-present" in codes

    def test_a_period_row_that_does_not_fit_its_restriction_warns_once(
        self, decomp_mini_deck: Path, tmp_path: Path
    ) -> None:
        cards = _StubRestricoes(
            restricao_eletrica_formula=_re_formula((401, "ger_usih(1)")),
            restricao_eletrica_inequacao_periodo_patamar=_re_inequacao_per_pat(
                (401, 1, 1, None, "ger_usih(1)", "<=", "10"),
            ),
        )

        result = self._run(decomp_mini_deck, tmp_path, cards)

        [unused] = [
            d
            for d in result.diagnostics
            if d.code == "decomp-electrical-period-rows-unused"
        ]
        assert unused.severity is Severity.WARNING
        assert unused.table is not None
        assert unused.table.rows == [
            ["RESTRICAO-ELETRICA-INEQUACAO-PERIODO-PATAMAR", 401]
        ]
        assert "decomp-libs-electrical-present" not in {
            d.code for d in result.diagnostics
        }
        assert result.verdict is PreflightVerdict.WARNINGS

    def test_lines_the_reader_keeps_as_text_are_warned_as_convert_does(
        self, decomp_mini_deck: Path, tmp_path: Path
    ) -> None:
        result = self._run(
            decomp_mini_deck,
            tmp_path,
            None,
            text="& comentario\n\nrestricao-eletrica-formula; 1; ger_usih(1)\n",
        )

        [unread] = [d for d in result.diagnostics if d.code == UNREAD_LINES_CODE]
        assert unread.severity is Severity.WARNING
        assert unread.table is not None
        assert unread.table.rows == [["restricao-eletrica-formula; 1; ger_usih(1)"]]
        assert result.verdict is PreflightVerdict.WARNINGS

    def test_a_file_whose_lines_are_all_read_raises_no_unread_warning(
        self, decomp_mini_deck: Path, tmp_path: Path
    ) -> None:
        result = self._run(decomp_mini_deck, tmp_path, None, text="& comentario\n")

        assert UNREAD_LINES_CODE not in {d.code for d in result.diagnostics}

    def test_an_unreadable_file_fails_the_check(
        self, decomp_mini_deck: Path, tmp_path: Path
    ) -> None:
        result = self._run(
            decomp_mini_deck,
            tmp_path,
            _StubRestricoes(),
            MagicMock(side_effect=ValueError("restriction 7 is malformed")),
        )

        [readable] = [
            c
            for c in result.checks
            if c.label == "LIBS electrical restrictions readable"
        ]
        assert readable.passed is False
        assert readable.detail == "restriction 7 is malformed"
        assert result.verdict is PreflightVerdict.WILL_NOT_CONVERT
        assert "decomp-libs-electrical-present" not in {
            d.code for d in result.diagnostics
        }

    def test_an_inequality_operator_convert_cannot_read_fails_the_check(
        self, decomp_mini_deck: Path, tmp_path: Path
    ) -> None:
        cards = _StubRestricoes(
            restricao_eletrica_inequacao=_re_inequacao((7, "ger_usih(1)", "==", "0")),
        )

        result = self._run(decomp_mini_deck, tmp_path, cards)

        [readable] = [
            c
            for c in result.checks
            if c.label == "LIBS electrical restrictions readable"
        ]
        assert readable.passed is False
        [failure] = [d for d in result.diagnostics if d.code == "source-field-parse"]
        assert failure.notes == [
            f"file: {tmp_path / 'deck' / self._LIBS_FILE}",
            "field: operador",
        ]
        assert "RESTRICAO-ELETRICA-INEQUACAO row" in failure.summary
        assert result.verdict is PreflightVerdict.WILL_NOT_CONVERT


class TestLoadFactorCheck:
    """``_load_factor_check`` must call ``convert_load_factors(case,
    id_map)`` -- the pre-refactor ``(dadger, id_map, calendar)`` call shape
    raises ``TypeError`` on every deck that reaches it, uncaught by the
    ``except (ValueError, KeyError, AttributeError)`` clause below.
    """

    def test_runs_to_completion_and_returns_a_check_item(self) -> None:
        case = make_decomp_case(
            Path("unused"),
            dadger=_StubDadger(dp=_dp_frame()),
            calendar=_calendar_rv3(),
        )

        item = _load_factor_check(case, _ID_MAP)

        assert isinstance(item, CheckItem)
        assert item.label == "Load block factors reproduce the stage span"
        assert item.passed is True
