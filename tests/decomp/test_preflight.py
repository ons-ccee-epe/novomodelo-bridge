"""Tests for ``check decomp`` — deck validation without conversion."""

from __future__ import annotations

import inspect
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from idecomp.decomp.modelos.dadger import (
    ACALTEFE,
    ACCOTVAZ,
    ACNCHAVE,
    ACTIPERH,
    ACVERTJU,
    ACVOLMIN,
)

from novomodelo_bridge.core.diagnostics import Severity
from novomodelo_bridge.core.errors import FieldParseError
from novomodelo_bridge.core.preflight import (
    CheckItem,
    PreflightVerdict,
    optional_input_advisory,
)
from novomodelo_bridge.decomp.constraint_registers import (
    detect_libs_electrical,
    detect_unreadable_electrical,
    read_constraints,
)
from novomodelo_bridge.decomp.converters.cadastro import (
    _SCALAR_AC_SPECS,
    APPLIED_AC_CLASSES,
    UNINGESTABLE_AC_CLASSES,
    CadastroResolutionReport,
    OutOfHorizon,
)
from novomodelo_bridge.decomp.files import DecompFiles
from novomodelo_bridge.decomp.preflight import (
    _ALL_AC_CLASSES,
    _ac_coverage,
    _load_factor_check,
    _special_constraint_coverage,
    run_decomp_preflight,
)
from tests.conftest import make_decomp_case
from tests.decomp.test_cadastro import _FakeDadger
from tests.decomp.test_constraint_registers import (
    _cm,
    _coeff,
    _decl,
    _he,
    _lv,
)
from tests.decomp.test_constraint_registers import _FakeDadger as _ConstraintFakeDadger
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
    """The id-map ``except`` widens from ``ValueError`` alone to
    ``(FieldParseError, ValueError)`` -- the typed ``from_dadger`` parse
    boundary and the id map's own ``__post_init__`` invariant checks both
    still degrade to a diagnosed ``WILL_NOT_CONVERT`` instead of crashing.
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

    def test_post_init_value_error_is_still_caught(self, tmp_path: Path) -> None:
        error = ValueError("bus codes must be strictly ascending; got (2, 1)")
        result = self._run_with_id_map_error(tmp_path, error)

        assert result.verdict is PreflightVerdict.WILL_NOT_CONVERT
        id_map_check = next(c for c in result.checks if c.label == "Entity id map")
        assert id_map_check.passed is False


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
        )

    def test_reports_all_six_decomp_optionals(self, tmp_path: Path) -> None:
        files = self._files(tmp_path)

        checks, diagnostics = optional_input_advisory(files)

        assert len(checks) == 6
        assert all(check.passed for check in checks)
        assert len(diagnostics) == 6
        assert all(d.code == "optional-file-absent" for d in diagnostics)
        assert all(d.severity is Severity.INFO for d in diagnostics)

        reported = {
            note.removeprefix("field: ")
            for d in diagnostics
            for note in d.notes
            if note.startswith("field: ")
        }
        assert reported == {
            "dadgnl",
            "renovaveis",
            "polinjus",
            "libs_restricao_eletrica",
            "cortesh",
            "cortes",
        }

    def test_skips_present_decomp_optional(self, tmp_path: Path) -> None:
        files = self._files(tmp_path, dadgnl=tmp_path / "dadgnl.rv0")

        _checks, diagnostics = optional_input_advisory(files)

        assert len(diagnostics) == 5
        assert not any("field: dadgnl" in d.notes for d in diagnostics)

    def test_run_decomp_preflight_uses_the_shared_advisory_helper(self) -> None:
        source = inspect.getsource(run_decomp_preflight)
        assert "optional_input_advisory" in source
        assert '("dadgnl", "renovaveis")' not in source


class TestAcCoverageRegistry:
    """The resolver's ``APPLIED_AC_CLASSES``/
    ``UNINGESTABLE_AC_CLASSES`` registries and the reflected idecomp ``AC``
    universe (``_ALL_AC_CLASSES``) stay consistent with each other, and the
    derived deferred bucket is populated by enumerate-and-diff rather than a
    hand-maintained list — proven by three families
    (``NCHAVE``/``TIPERH``/``VERTJU``) the old blanket warning never named.
    """

    def test_coverage_registry_applied_classes_count_is_16(self) -> None:
        assert len(APPLIED_AC_CLASSES) == 16

    def test_coverage_registry_uningestable_is_altefe_only(self) -> None:
        assert UNINGESTABLE_AC_CLASSES == frozenset({ACALTEFE})

    def test_coverage_registry_applied_and_uningestable_disjoint(self) -> None:
        assert not (APPLIED_AC_CLASSES & UNINGESTABLE_AC_CLASSES)

    def test_coverage_registry_scalar_specs_subset_of_applied(self) -> None:
        scalar_classes = {spec.ac_class for spec in _SCALAR_AC_SPECS}
        assert scalar_classes <= APPLIED_AC_CLASSES

    def test_coverage_registry_applied_subset_of_all_ac_classes(self) -> None:
        assert APPLIED_AC_CLASSES <= _ALL_AC_CLASSES

    def test_coverage_registry_uningestable_subset_of_all_ac_classes(self) -> None:
        assert UNINGESTABLE_AC_CLASSES <= _ALL_AC_CLASSES

    def test_coverage_registry_all_ac_classes_has_at_least_25_members(self) -> None:
        # idecomp 1.13.0 exposes exactly 25; ``>=`` stays robust to a future add.
        assert len(_ALL_AC_CLASSES) >= 25

    def test_coverage_registry_deferred_bucket_contains_uncovered_families(
        self,
    ) -> None:
        deferred = _ALL_AC_CLASSES - APPLIED_AC_CLASSES - UNINGESTABLE_AC_CLASSES
        assert {ACNCHAVE, ACTIPERH, ACVERTJU} <= deferred


class TestAcCoverage:
    """``_ac_coverage``'s three-bucket classification and its
    always-passing summary ``CheckItem`` plus per-bucket diagnostics.
    """

    def test_ac_coverage_mixed_presence_emits_all_four_diagnostics(self) -> None:
        present = pd.DataFrame({"codigo_usina": [1]})
        dadger = _FakeDadger({ACVOLMIN: present, ACCOTVAZ: present, ACALTEFE: present})
        report = CadastroResolutionReport(
            applied={"volume_minimo": 1},
            out_of_horizon=(
                OutOfHorizon(code=7, param="volume_minimo", mes=12, ano=2030),
            ),
        )

        checks, diagnostics = _ac_coverage(dadger, report)

        assert len(checks) == 1
        assert checks[0].label == "AC cadastro-override coverage"
        assert checks[0].passed is True

        by_code = {d.code: d for d in diagnostics}
        assert set(by_code) == {
            "decomp-ac-overrides-applied",
            "decomp-ac-overrides-deferred",
            "decomp-ac-altefe-uningestable",
            "decomp-ac-out-of-horizon",
        }
        assert by_code["decomp-ac-overrides-applied"].severity is Severity.INFO
        assert by_code["decomp-ac-overrides-deferred"].severity is Severity.WARNING
        assert by_code["decomp-ac-altefe-uningestable"].severity is Severity.WARNING
        assert by_code["decomp-ac-out-of-horizon"].severity is Severity.WARNING

        deferred_summary = by_code["decomp-ac-overrides-deferred"].summary
        assert "COTVAZ" in deferred_summary
        assert "ALTEFE" not in deferred_summary

        table = by_code["decomp-ac-out-of-horizon"].table
        assert table is not None
        assert table.columns == ["plant", "param", "month", "year"]
        assert table.rows == [[7, "volume_minimo", 12, 2030]]

    def test_ac_coverage_applied_only_emits_only_info_diagnostic(self) -> None:
        present = pd.DataFrame({"codigo_usina": [1]})
        dadger = _FakeDadger({ACVOLMIN: present})
        report = CadastroResolutionReport(
            applied={"volume_minimo": 1}, out_of_horizon=()
        )

        checks, diagnostics = _ac_coverage(dadger, report)

        assert len(checks) == 1
        assert checks[0].passed is True
        assert len(diagnostics) == 1
        assert diagnostics[0].code == "decomp-ac-overrides-applied"
        assert diagnostics[0].severity is Severity.INFO

    def test_ac_coverage_empty_deck_emits_only_the_passing_summary(self) -> None:
        dadger = _FakeDadger({})
        report = CadastroResolutionReport(applied={}, out_of_horizon=())

        checks, diagnostics = _ac_coverage(dadger, report)

        assert len(checks) == 1
        assert checks[0].passed is True
        assert diagnostics == []


class TestSpecialConstraintCoverage:
    """``_special_constraint_coverage``'s converted (RE/HQ/HV/HE,
    bounds-lowered vs generic-emitted) vs deferred (FE/RHA/LIBs-electrical)
    classification, and its always-passing summary ``CheckItem``.
    """

    @staticmethod
    def _synthetic_dadger() -> _ConstraintFakeDadger:
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
        )

    @staticmethod
    def _files(
        tmp_path: Path, dadger_text: str, *, indices_csv: str | None = None
    ) -> SimpleNamespace:
        """A minimal ``files``-shaped stub — only ``.dadger`` is read by the
        coverage function, so a duck-typed object avoids coupling this test
        to ``DecompFiles``'s full field list."""
        dadger_path = tmp_path / "dadger.rv0"
        dadger_path.write_text(dadger_text, encoding="latin-1")
        if indices_csv is not None:
            (tmp_path / "indices.csv").write_text(indices_csv, encoding="latin-1")
        return SimpleNamespace(dadger=dadger_path)

    def test_mixed_deck_emits_summary_plus_converted_and_deferred_diagnostics(
        self, tmp_path: Path
    ) -> None:
        dadger = self._synthetic_dadger()
        census = read_constraints(dadger)
        files = self._files(tmp_path, "RE    10    1    2\nFE    10   21  1.0\n")

        checks, diagnostics = _special_constraint_coverage(dadger, files)

        assert len(checks) == 1
        assert checks[0].label == "Special-constraint coverage"
        assert checks[0].passed is True

        by_code = {d.code: d for d in diagnostics}
        assert set(by_code) == {
            "decomp-special-constraints-converted",
            "decomp-fe-participation-unreadable",
        }
        assert by_code["decomp-special-constraints-converted"].severity is (
            Severity.INFO
        )
        assert by_code["decomp-fe-participation-unreadable"].severity is (
            Severity.WARNING
        )
        present_families = [fam for fam, records in census.by_family.items() if records]
        assert checks[0].detail == (
            f"{len(census.to_bounds)} record(s) lowered to bounds, "
            f"{len(census.to_generic)} emitted as generic constraints, "
            f"{len(present_families)} family(ies) present in the deck"
        )

    def test_converted_counts_match_the_census_probe(self, tmp_path: Path) -> None:
        dadger = self._synthetic_dadger()
        census = read_constraints(dadger)
        files = self._files(tmp_path, "RE    10    1    2\n")

        _checks, diagnostics = _special_constraint_coverage(dadger, files)

        converted = next(
            d for d in diagnostics if d.code == "decomp-special-constraints-converted"
        )
        # Probe — asserted against the census's own counts, never
        # a hard-coded oracle.
        assert str(len(census.to_bounds)) in converted.summary
        assert str(len(census.to_generic)) in converted.summary
        assert len(census.to_bounds) == 3  # RE FU + HV VARM + HQ QDES
        assert len(census.to_generic) == 1  # HE (always generic)

    def test_deferred_fe_diagnostic_is_sourced_from_the_e1_helper_verbatim(
        self, tmp_path: Path
    ) -> None:
        dadger = self._synthetic_dadger()
        dadger_text = "RE    10    1    2\nFE    10   21  1.0\n"
        files = self._files(tmp_path, dadger_text)
        expected = detect_unreadable_electrical(files.dadger)
        assert len(expected) == 1  # sanity: the fixture trips FE, not RHA

        _checks, diagnostics = _special_constraint_coverage(dadger, files)

        fe_diagnostic = next(
            d for d in diagnostics if d.code == "decomp-fe-participation-unreadable"
        )
        assert fe_diagnostic == expected[0]

    def test_deferred_libs_electrical_diagnostic_is_sourced_from_the_e1_helper(
        self, tmp_path: Path
    ) -> None:
        dadger = _ConstraintFakeDadger()
        files = self._files(
            tmp_path,
            "VAZOES record only\n",
            indices_csv=(
                "RESTRICAO-ELETRICA-ESPECIAL;Descricao;"
                "lib_restricao-eletrica-especial.csv\n"
            ),
        )
        expected = detect_libs_electrical(files.dadger.parent)
        assert expected is not None  # sanity: the fixture trips the LIBs token

        _checks, diagnostics = _special_constraint_coverage(dadger, files)

        assert diagnostics == [expected]

    def test_no_special_constraints_deck_emits_only_the_passing_summary(
        self, tmp_path: Path
    ) -> None:
        dadger = _ConstraintFakeDadger()
        files = self._files(tmp_path, "VAZOES record only\n")

        checks, diagnostics = _special_constraint_coverage(dadger, files)

        assert len(checks) == 1
        assert checks[0].passed is True
        assert checks[0].detail == (
            "0 record(s) lowered to bounds, 0 emitted as generic constraints, "
            "0 family(ies) present in the deck"
        )
        assert diagnostics == []


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
