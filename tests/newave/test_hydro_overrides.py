"""Unit tests for the source model hydro overrides converter."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.diagnostics import Severity, finalize_diagnostics
from tests.conftest import make_case, make_nw_files, modif_mock
from tests.newave.conftest import _make_hidr_cadastro, _read_modif

# The title/summary/remediation/notes strings overrides.py emits reach a
# pip-installed user with no repo checkout — none may leak a repo-internal
# reference (mirrors test_constraints.py's own marker scan).
_REPO_INTERNAL_LEAKS = (
    "docs/",
    "plans/",
    "~/git",
    "feat/",
    "ticket-",
    "epic-",
    "src/",
    ".py",
)


def _assert_no_repo_internal_leaks(collected: list[dx.Diagnostic]) -> None:
    for diag in collected:
        strings = [diag.title, diag.summary, *diag.notes]
        if diag.remediation is not None:
            strings.append(diag.remediation)
        for s in strings:
            for leak in _REPO_INTERNAL_LEAKS:
                assert leak not in s, f"diagnostic {diag.code!r} leaks {leak!r}: {s!r}"


# ---------------------------------------------------------------------------
# _apply_permanent_overrides unit tests
# ---------------------------------------------------------------------------


class TestApplyPermanentOverrides:
    """Unit tests for ``_apply_permanent_overrides``."""

    def _base_cadastro(self) -> pd.DataFrame:
        return _make_hidr_cadastro()

    def _modif_case(self, tmp_path, mock_modif):
        """Build a case whose MODIF reader is *mock_modif* (path set)."""
        return make_case(
            make_nw_files(tmp_path, modif=tmp_path / "modif.dat"),
            modif=mock_modif,
        )

    def test_missing_modif_returns_unchanged(self, tmp_path) -> None:
        """No MODIF.DAT -> cadastro returned unchanged."""
        from novomodelo_bridge.newave.converters.hydro import _apply_permanent_overrides

        cadastro = self._base_cadastro()
        result = _apply_permanent_overrides(cadastro, make_case(tmp_path, modif=None))
        pd.testing.assert_frame_equal(result, cadastro)

    def test_volmax_override(self, tmp_path) -> None:
        """VOLMAX record updates volume_maximo for the target plant."""
        from novomodelo_bridge.newave.converters.hydro import _apply_permanent_overrides

        # Build MODIF mock: plant 1 gets VOLMAX=2000.
        volmax_rec = MagicMock()
        volmax_rec.__class__.__name__ = "VOLMAX"
        type(volmax_rec).__name__ = "VOLMAX"
        volmax_rec.volume = 2000.0
        volmax_rec.unidade = "h"

        mock_modif = modif_mock((1, [volmax_rec]))

        result = _apply_permanent_overrides(
            self._base_cadastro(), self._modif_case(tmp_path, mock_modif)
        )

        assert float(result.loc[1, "volume_maximo"]) == pytest.approx(2000.0)
        # Plant 2 must be unchanged.
        assert float(result.loc[2, "volume_maximo"]) == pytest.approx(500.0)

    @staticmethod
    def _volume_rec(type_name: str, volume: float, unidade: str) -> MagicMock:
        rec = MagicMock()
        type(rec).__name__ = type_name
        rec.volume = volume
        rec.unidade = unidade
        return rec

    def _apply(self, tmp_path, code: int, records: list) -> pd.DataFrame:
        from novomodelo_bridge.newave.converters.hydro import _apply_permanent_overrides

        mock_modif = modif_mock((code, records))
        return _apply_permanent_overrides(
            self._base_cadastro(), self._modif_case(tmp_path, mock_modif)
        )

    def test_volmax_percent_is_of_the_useful_volume(self, tmp_path) -> None:
        """``VOLMAX 55 '%'`` on [100, 1000] is 100 + 0.55 * 900, not 55 hm³."""
        result = self._apply(tmp_path, 1, [self._volume_rec("VOLMAX", 55.0, "%")])
        assert float(result.loc[1, "volume_maximo"]) == pytest.approx(595.0)

    def test_volmin_percent_is_of_the_useful_volume(self, tmp_path) -> None:
        """``VOLMIN 20 '%'`` on [50, 500] is 50 + 0.20 * 450."""
        result = self._apply(tmp_path, 2, [self._volume_rec("VOLMIN", 20.0, "%")])
        assert float(result.loc[2, "volume_minimo"]) == pytest.approx(140.0)

    def test_percent_resolves_against_the_registry_regardless_of_order(
        self, tmp_path
    ) -> None:
        """A ``VOLMIN`` in hm³ before a ``VOLMAX`` in percent does not move the
        base the percentage is taken from: hidr.dat's [100, 1000] stays the
        reference, so 50 % is 550, not 200 + 0.5 * 800."""
        result = self._apply(
            tmp_path,
            1,
            [
                self._volume_rec("VOLMIN", 200.0, "h"),
                self._volume_rec("VOLMAX", 50.0, "%"),
            ],
        )
        assert float(result.loc[1, "volume_minimo"]) == pytest.approx(200.0)
        assert float(result.loc[1, "volume_maximo"]) == pytest.approx(550.0)

    def test_unit_is_read_case_and_space_insensitively(self, tmp_path) -> None:
        result = self._apply(tmp_path, 1, [self._volume_rec("VOLMAX", 800.0, " H ")])
        assert float(result.loc[1, "volume_maximo"]) == pytest.approx(800.0)

    def test_the_quoted_unit_the_file_writes_reads_as_percent(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.hydro import _apply_permanent_overrides

        modif = _read_modif(tmp_path, " USINA      1   USINA_A", " VOLMAX 55 '%'")
        result = _apply_permanent_overrides(
            self._base_cadastro(), self._modif_case(tmp_path, modif)
        )
        assert float(result.loc[1, "volume_maximo"]) == pytest.approx(595.0)

    def test_vazmin_override(self, tmp_path) -> None:
        """VAZMIN record updates vazao_minima_historica for the target plant."""
        from novomodelo_bridge.newave.converters.hydro import _apply_permanent_overrides

        vazmin_rec = MagicMock()
        type(vazmin_rec).__name__ = "VAZMIN"
        vazmin_rec.vazao = 75.5

        mock_modif = modif_mock((2, [vazmin_rec]))

        result = _apply_permanent_overrides(
            self._base_cadastro(), self._modif_case(tmp_path, mock_modif)
        )

        assert float(result.loc[2, "vazao_minima_historica"]) == pytest.approx(75.5)
        # Plant 1 must be unchanged (was 0).
        assert float(result.loc[1, "vazao_minima_historica"]) == pytest.approx(0.0)

    def test_numcnj_nummaq_override(self, tmp_path) -> None:
        """NUMCNJ + NUMMAQ records update machine set counts."""
        from novomodelo_bridge.newave.converters.hydro import _apply_permanent_overrides

        numcnj_rec = MagicMock()
        type(numcnj_rec).__name__ = "NUMCNJ"
        numcnj_rec.numero = 2

        nummaq_rec = MagicMock()
        type(nummaq_rec).__name__ = "NUMMAQ"
        nummaq_rec.conjunto = 2
        nummaq_rec.numero_maquinas = 3

        mock_modif = modif_mock((1, [numcnj_rec, nummaq_rec]))

        result = _apply_permanent_overrides(
            self._base_cadastro(), self._modif_case(tmp_path, mock_modif)
        )

        assert int(result.loc[1, "numero_conjuntos_maquinas"]) == 2
        assert int(result.loc[1, "maquinas_conjunto_2"]) == 3

    def test_potefe_override(self, tmp_path) -> None:
        """POTEFE replaces the conjunto's nominal power, leaving the others."""
        from novomodelo_bridge.newave.converters.hydro import _apply_permanent_overrides

        potefe_rec = MagicMock()
        type(potefe_rec).__name__ = "POTEFE"
        potefe_rec.potencia = 500.0
        potefe_rec.conjunto = 2

        mock_modif = modif_mock((2, [potefe_rec]))

        with dx.collect() as collected:
            result = _apply_permanent_overrides(
                self._base_cadastro(), self._modif_case(tmp_path, mock_modif)
            )

        assert result.loc[2, "potencia_nominal_conjunto_2"] == 500.0
        assert result.loc[2, "potencia_nominal_conjunto_1"] == 150.0
        assert collected == []

    @pytest.mark.parametrize(
        ("type_name", "attr", "prefix", "raw", "expected"),
        [
            (
                "VOLCOTA",
                "polinomio_volume_cota",
                "volume_cota",
                [400.0, 0.2, 0.0, 0.0, 0.0],
                [400.0, 0.2, 0.0, 0.0, 0.0],
            ),
            (
                "COTAREA",
                "polinomio_cota_area",
                "cota_area",
                [-1.0e7, 9.0e4, -2.5e2, 3.0e-1, 0.0],
                [-1.0e7, 9.0e4, -2.5e2, 3.0e-1, 0.0],
            ),
            # A record that truncates its tail leaves the high-order terms at zero.
            (
                "VOLCOTA",
                "polinomio_volume_cota",
                "volume_cota",
                [400.0, None, None, None, None],
                [400.0, 0.0, 0.0, 0.0, 0.0],
            ),
        ],
    )
    def test_polynomial_override_replaces_the_whole_polynomial(
        self, tmp_path, type_name, attr, prefix, raw, expected
    ) -> None:
        from novomodelo_bridge.newave.converters.hydro import _apply_permanent_overrides

        poly_rec = MagicMock()
        type(poly_rec).__name__ = type_name
        setattr(poly_rec, attr, raw)

        mock_modif = modif_mock((1, [poly_rec]))

        base = self._base_cadastro()
        columns = [f"a{i}_{prefix}" for i in range(5)]
        with dx.collect() as collected:
            result = _apply_permanent_overrides(
                base, self._modif_case(tmp_path, mock_modif)
            )

        assert result.loc[1, columns].tolist() == expected
        assert result.loc[2, columns].tolist() == base.loc[2, columns].tolist()
        assert collected == []

    def test_unknown_plant_code_skipped(self, tmp_path) -> None:
        """Plant code not in cadastro: diagnostic emitted, no crash."""
        from novomodelo_bridge.newave.converters.hydro import _apply_permanent_overrides

        mock_modif = modif_mock((999, []))  # 999 is not in the cadastro

        with dx.collect() as collected:
            result = _apply_permanent_overrides(
                self._base_cadastro(), self._modif_case(tmp_path, mock_modif)
            )

        pd.testing.assert_frame_equal(result, self._base_cadastro(), check_dtype=False)
        assert len(collected) == 1
        diag = collected[0]
        assert diag.code == "modif-override-plant-uncadastred"
        assert diag.severity is Severity.WARNING
        assert diag.category == "Cadastro overrides"
        assert diag.table is not None
        assert diag.table.columns == ["Code"]
        assert diag.table.rows == [[999]]

        _assert_no_repo_internal_leaks(collected)

    def test_temporal_records_skipped_in_permanent_pass(self, tmp_path) -> None:
        """Temporal override types are ignored in _apply_permanent_overrides."""
        import datetime

        from novomodelo_bridge.newave.converters.hydro import _apply_permanent_overrides

        vazmint_rec = MagicMock()
        type(vazmint_rec).__name__ = "VAZMINT"
        vazmint_rec.data_inicio = datetime.datetime(2025, 1, 1)
        vazmint_rec.vazao = 999.0  # large value that should NOT be applied

        mock_modif = modif_mock((1, [vazmint_rec]))

        result = _apply_permanent_overrides(
            self._base_cadastro(), self._modif_case(tmp_path, mock_modif)
        )

        # vazao_minima_historica must stay at the base value (0).
        assert float(result.loc[1, "vazao_minima_historica"]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _extract_temporal_overrides unit tests
# ---------------------------------------------------------------------------


class TestExtractTemporalOverrides:
    """Unit tests for ``_extract_temporal_overrides``."""

    def _modif_case(self, tmp_path, mock_modif):
        """Build a case whose MODIF reader is *mock_modif* (path set)."""
        return make_case(
            make_nw_files(tmp_path, modif=tmp_path / "modif.dat"),
            modif=mock_modif,
        )

    def test_missing_modif_returns_empty(self, tmp_path) -> None:
        """No MODIF.DAT -> empty dict returned, no error."""
        from novomodelo_bridge.newave.converters.hydro import (
            _extract_temporal_overrides,
        )

        result = _extract_temporal_overrides(make_case(tmp_path, modif=None), [1, 2])
        assert result == {}

    def test_extracts_vazmint_records(self, tmp_path) -> None:
        """VAZMINT record is extracted with correct month, year, value."""
        import datetime

        from novomodelo_bridge.newave.converters.hydro import (
            _extract_temporal_overrides,
        )

        vazmint_rec = MagicMock()
        type(vazmint_rec).__name__ = "VAZMINT"
        vazmint_rec.data_inicio = datetime.datetime(2025, 1, 1)
        vazmint_rec.periodo = None
        vazmint_rec.mes = 1
        vazmint_rec.vazao = 50.0

        mock_modif = modif_mock((1, [vazmint_rec]))

        result = _extract_temporal_overrides(
            self._modif_case(tmp_path, mock_modif), [1, 2]
        )

        assert 1 in result
        assert result[1] == [
            {
                "type": "VAZMINT",
                "month": 1,
                "year": 2025,
                "period": None,
                "value": 50.0,
            }
        ]

    def test_vmaxt_and_vmint_carry_their_unit(self, tmp_path) -> None:
        """Dated volume records keep the unit column so the bounds converter
        can tell hm³ from percent of the useful volume."""
        import datetime

        from novomodelo_bridge.newave.converters.hydro import (
            _extract_temporal_overrides,
        )

        vmaxt_rec = MagicMock()
        type(vmaxt_rec).__name__ = "VMAXT"
        vmaxt_rec.data_inicio = datetime.datetime(2025, 1, 1)
        vmaxt_rec.volume = 73.2
        vmaxt_rec.unidade = "%"
        vmint_rec = MagicMock()
        type(vmint_rec).__name__ = "VMINT"
        vmint_rec.data_inicio = datetime.datetime(2025, 2, 1)
        vmint_rec.volume = 1500.0
        vmint_rec.unidade = "h"

        mock_modif = modif_mock((1, [vmaxt_rec, vmint_rec]))

        with dx.collect() as collected:
            result = _extract_temporal_overrides(
                self._modif_case(tmp_path, mock_modif), [1, 2]
            )

        assert collected == []
        assert result[1] == [
            {
                "type": "VMAXT",
                "month": 1,
                "year": 2025,
                "period": None,
                "value": 73.2,
                "unit": "%",
            },
            {
                "type": "VMINT",
                "month": 2,
                "year": 2025,
                "period": None,
                "value": 1500.0,
                "unit": "h",
            },
        ]

    def test_dated_volume_without_unit_is_percent_and_reported(self, tmp_path) -> None:
        import datetime

        from novomodelo_bridge.newave.converters.hydro import (
            _extract_temporal_overrides,
        )

        vmaxt_rec = MagicMock()
        type(vmaxt_rec).__name__ = "VMAXT"
        vmaxt_rec.data_inicio = datetime.datetime(2025, 1, 1)
        vmaxt_rec.volume = 73.2
        vmaxt_rec.unidade = None

        mock_modif = modif_mock((1, [vmaxt_rec]))

        with dx.collect() as collected:
            result = _extract_temporal_overrides(
                self._modif_case(tmp_path, mock_modif), [1, 2]
            )

        assert result[1][0]["unit"] == "%"
        assert [d.code for d in collected] == ["modif-temporal-volume-unit-unknown"]
        assert collected[0].severity is Severity.WARNING
        assert collected[0].table is not None
        assert collected[0].table.rows == [[1, "VMAXT", ""]]

    def test_filters_by_confhd_codes(self, tmp_path) -> None:
        """Plants not in confhd_codes are excluded from the result."""
        import datetime

        from novomodelo_bridge.newave.converters.hydro import (
            _extract_temporal_overrides,
        )

        vazmint_rec = MagicMock()
        type(vazmint_rec).__name__ = "VAZMINT"
        vazmint_rec.data_inicio = datetime.datetime(2025, 3, 1)
        vazmint_rec.periodo = None
        vazmint_rec.mes = 3
        vazmint_rec.vazao = 40.0

        # Plant 99 is NOT in confhd_codes [1, 2].

        mock_modif = modif_mock((99, [vazmint_rec]))

        result = _extract_temporal_overrides(
            self._modif_case(tmp_path, mock_modif), [1, 2]
        )

        assert result == {}

    def test_preserves_file_order(self, tmp_path) -> None:
        """Multiple records for the same plant are returned in file order."""
        import datetime

        from novomodelo_bridge.newave.converters.hydro import (
            _extract_temporal_overrides,
        )

        def _vazmint(month: int, vazao: float) -> MagicMock:
            r = MagicMock()
            type(r).__name__ = "VAZMINT"
            r.data_inicio = datetime.datetime(2025, month, 1)
            r.periodo = None
            r.mes = month
            r.vazao = vazao
            return r

        recs = [_vazmint(1, 50.0), _vazmint(6, 60.0), _vazmint(3, 55.0)]

        mock_modif = modif_mock((1, recs))

        result = _extract_temporal_overrides(
            self._modif_case(tmp_path, mock_modif), [1]
        )

        assert len(result[1]) == 3
        assert result[1][0]["value"] == pytest.approx(50.0)
        assert result[1][1]["value"] == pytest.approx(60.0)
        assert result[1][2]["value"] == pytest.approx(55.0)

    def test_extracts_cfuga_records(self, tmp_path) -> None:
        """CFUGA record extracted with correct level value."""
        import datetime

        from novomodelo_bridge.newave.converters.hydro import (
            _extract_temporal_overrides,
        )

        cfuga_rec = MagicMock()
        type(cfuga_rec).__name__ = "CFUGA"
        cfuga_rec.data_inicio = datetime.datetime(2025, 6, 1)
        cfuga_rec.nivel = 75.4

        mock_modif = modif_mock((2, [cfuga_rec]))

        result = _extract_temporal_overrides(
            self._modif_case(tmp_path, mock_modif), [2]
        )

        assert result[2] == [
            {
                "type": "CFUGA",
                "month": 6,
                "year": 2025,
                "period": None,
                "value": pytest.approx(75.4),
            }
        ]

    def test_extracts_turbmint_turbmaxt_records(self, tmp_path) -> None:
        """TURBMINT and TURBMAXT records use turbinamento field."""
        import datetime

        from novomodelo_bridge.newave.converters.hydro import (
            _extract_temporal_overrides,
        )

        turbmint_rec = MagicMock()
        type(turbmint_rec).__name__ = "TURBMINT"
        turbmint_rec.data_inicio = datetime.datetime(2025, 11, 1)
        turbmint_rec.turbinamento = 330.0

        turbmaxt_rec = MagicMock()
        type(turbmaxt_rec).__name__ = "TURBMAXT"
        turbmaxt_rec.data_inicio = datetime.datetime(2025, 3, 1)
        turbmaxt_rec.turbinamento = 322.0

        mock_modif = modif_mock((1, [turbmint_rec, turbmaxt_rec]))

        result = _extract_temporal_overrides(
            self._modif_case(tmp_path, mock_modif), [1]
        )

        assert result[1][0] == {
            "type": "TURBMINT",
            "month": 11,
            "year": 2025,
            "period": None,
            "value": pytest.approx(330.0),
        }
        assert result[1][1] == {
            "type": "TURBMAXT",
            "month": 3,
            "year": 2025,
            "period": None,
            "value": pytest.approx(322.0),
        }

    def test_extracts_pre_pos_markers(self, tmp_path) -> None:
        """PRE/POS year markers are carried through with ``year=None`` and the
        ``period`` field set, so the step-function builder can map them onto the
        horizon entry (PRE) and the post-study tail (POS)."""
        import datetime

        from novomodelo_bridge.newave.converters.hydro import (
            _extract_temporal_overrides,
        )

        pre_rec = MagicMock()
        type(pre_rec).__name__ = "VAZMINT"
        pre_rec.data_inicio = None
        pre_rec.periodo = "PRE"
        pre_rec.mes = 1
        pre_rec.vazao = 100.0

        study_rec = MagicMock()
        type(study_rec).__name__ = "VAZMINT"
        study_rec.data_inicio = datetime.datetime(2030, 6, 1)
        study_rec.periodo = None
        study_rec.mes = 6
        study_rec.vazao = 300.0

        pos_rec = MagicMock()
        type(pos_rec).__name__ = "VAZMINT"
        pos_rec.data_inicio = None
        pos_rec.periodo = "POS"
        pos_rec.mes = 6
        pos_rec.vazao = 300.0

        mock_modif = modif_mock((1, [pre_rec, study_rec, pos_rec]))

        result = _extract_temporal_overrides(
            self._modif_case(tmp_path, mock_modif), [1]
        )

        assert result[1] == [
            {
                "type": "VAZMINT",
                "month": 1,
                "year": None,
                "period": "PRE",
                "value": 100.0,
            },
            {
                "type": "VAZMINT",
                "month": 6,
                "year": 2030,
                "period": None,
                "value": 300.0,
            },
            {
                "type": "VAZMINT",
                "month": 6,
                "year": None,
                "period": "POS",
                "value": 300.0,
            },
        ]


# ---------------------------------------------------------------------------
# _read_ghmin_per_stage unit tests
# ---------------------------------------------------------------------------


class TestReadGhminPerStage:
    """Unit tests for ``_read_ghmin_per_stage``.

    GHMIN values are time-varying and now live in
    ``hydro_bounds.parquet:min_generation_mw`` rather than the static
    ``hydros.json:generation.min_generation_mw``.  This helper expands
    each (plant, month, year) record into a per-(plant, stage_0based)
    mapping with step-function semantics and seasonal post-study
    repetition.
    """

    def _ghmin_case(self, tmp_path, mock_ghmin):
        """Build a case whose GHMIN reader is *mock_ghmin* (path set)."""
        return make_case(
            make_nw_files(tmp_path, ghmin=tmp_path / "ghmin.dat"),
            ghmin=mock_ghmin,
        )

    def test_missing_ghmin_returns_empty(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.hydro import _read_ghmin_per_stage

        result = _read_ghmin_per_stage(
            make_case(tmp_path, ghmin=None),
            start_year=2024,
            start_month=9,
            study_months=12,
            total_stages=24,
        )
        assert result == {}

    def test_step_function_persists_until_next_entry(self, tmp_path) -> None:
        """Sparse entries persist the last applied value forward."""
        import datetime

        from novomodelo_bridge.newave.converters.hydro import _read_ghmin_per_stage

        # Plant 1 at Sep 2024 = 100 MW, Dec 2024 = 80 MW.
        # Stages 0 (Sep) and 1 (Oct) and 2 (Nov) should all be 100.
        # Stage 3 (Dec) and onwards should be 80 within the study.
        ghmin_df = pd.DataFrame(
            {
                "codigo_usina": [1, 1],
                "data": [
                    datetime.datetime(2024, 9, 1),
                    datetime.datetime(2024, 12, 1),
                ],
                "patamar": [0, 0],
                "geracao": [100.0, 80.0],
            }
        )
        mock_ghmin = MagicMock()
        mock_ghmin.geracoes = ghmin_df

        result = _read_ghmin_per_stage(
            self._ghmin_case(tmp_path, mock_ghmin),
            start_year=2024,
            start_month=9,
            study_months=12,
            total_stages=12,
        )

        per_stage = result[1]
        assert per_stage[0] == pytest.approx(100.0)
        assert per_stage[1] == pytest.approx(100.0)
        assert per_stage[2] == pytest.approx(100.0)
        assert per_stage[3] == pytest.approx(80.0)
        assert per_stage[4] == pytest.approx(80.0)

    def test_post_study_uses_pos_seasonal_pattern(self, tmp_path) -> None:
        """POS year=9999 entries supply per-calendar-month values."""
        import datetime

        from novomodelo_bridge.newave.converters.hydro import _read_ghmin_per_stage

        ghmin_df = pd.DataFrame(
            {
                "codigo_usina": [1, 1, 1],
                "data": [
                    datetime.datetime(2024, 9, 1),  # study Sep 2024
                    datetime.datetime(9999, 9, 1),  # POS Sep
                    datetime.datetime(9999, 12, 1),  # POS Dec
                ],
                "patamar": [0, 0, 0],
                "geracao": [100.0, 150.0, 200.0],
            }
        )
        mock_ghmin = MagicMock()
        mock_ghmin.geracoes = ghmin_df

        result = _read_ghmin_per_stage(
            self._ghmin_case(tmp_path, mock_ghmin),
            start_year=2024,
            start_month=9,
            study_months=12,  # study ends Aug 2025
            total_stages=24,  # post-study: Sep 2025 – Aug 2026
        )

        per_stage = result[1]
        # Stage 12 = Sep 2025 → POS Sep = 150.
        assert per_stage[12] == pytest.approx(150.0)
        # Stage 15 = Dec 2025 → POS Dec = 200.
        assert per_stage[15] == pytest.approx(200.0)

    def test_patamar_nonzero_excluded(self, tmp_path) -> None:
        """Rows with patamar != 0 are excluded — only the all-blocks mean
        is meaningful at hydro_bounds' stage granularity."""
        import datetime

        from novomodelo_bridge.newave.converters.hydro import _read_ghmin_per_stage

        ghmin_df = pd.DataFrame(
            {
                "codigo_usina": [1, 1],
                "data": [
                    datetime.datetime(2024, 9, 1),
                    datetime.datetime(2024, 9, 1),
                ],
                "patamar": [1, 2],
                "geracao": [50.0, 60.0],
            }
        )
        mock_ghmin = MagicMock()
        mock_ghmin.geracoes = ghmin_df

        result = _read_ghmin_per_stage(
            self._ghmin_case(tmp_path, mock_ghmin),
            start_year=2024,
            start_month=9,
            study_months=12,
            total_stages=12,
        )

        assert result == {}


# ---------------------------------------------------------------------------
# Structured-diagnostic coverage for the MODIF.DAT override readers
# ---------------------------------------------------------------------------


class TestApplyPermanentOverridesDiagnostics:
    """Emission-shape coverage for the two ``_apply_permanent_overrides``
    diagnostics: uncadastred plants and unsupported/unknown permanent types."""

    def _base_cadastro(self) -> pd.DataFrame:
        return _make_hidr_cadastro()

    def _modif_case(self, tmp_path, mock_modif):
        return make_case(
            make_nw_files(tmp_path, modif=tmp_path / "modif.dat"),
            modif=mock_modif,
        )

    def test_unsupported_types_fold_into_the_same_code(self, tmp_path) -> None:
        """A modelled-but-unconsumed type and a genuinely unknown one both land
        in ``modif-permanent-override-unsupported`` — the Type column is what
        distinguishes them, not the code."""
        from novomodelo_bridge.newave.converters.hydro import _apply_permanent_overrides

        volcota_rec = MagicMock()
        type(volcota_rec).__name__ = "VMINP"
        unknown_rec = MagicMock()
        type(unknown_rec).__name__ = "SOME_FUTURE_TYPE"

        mock_modif = modif_mock((1, [volcota_rec]), (2, [unknown_rec]))

        with dx.collect() as collected:
            _apply_permanent_overrides(
                self._base_cadastro(), self._modif_case(tmp_path, mock_modif)
            )

        assert len(collected) == 1
        diag = collected[0]
        assert diag.code == "modif-permanent-override-unsupported"
        assert diag.severity is Severity.WARNING
        assert diag.category == "Cadastro overrides"
        assert diag.table is not None
        assert diag.table.columns == ["Code", "Type"]
        assert diag.table.rows == [[1, "VMINP"], [2, "SOME_FUTURE_TYPE"]]

        _assert_no_repo_internal_leaks(collected)

    def test_defaultregister_stays_debug_and_emits_no_diagnostic(
        self, tmp_path, caplog
    ) -> None:
        """DefaultRegister (the inewave unmodeled-record sentinel) is a
        deliberate keep-as-log exception: DEBUG only, never a Diagnostic."""
        import logging

        from novomodelo_bridge.newave.converters.hydro import _apply_permanent_overrides

        default_rec = MagicMock()
        type(default_rec).__name__ = "DefaultRegister"

        mock_modif = modif_mock((1, [default_rec]))

        with (
            dx.collect() as collected,
            caplog.at_level(
                logging.DEBUG,
                logger="novomodelo_bridge.newave.converters.hydro.overrides",
            ),
        ):
            _apply_permanent_overrides(
                self._base_cadastro(), self._modif_case(tmp_path, mock_modif)
            )

        assert collected == []
        assert any(
            r.levelno == logging.DEBUG and "DefaultRegister" in r.message
            for r in caplog.records
        )

    def test_no_sink_fallback_logs_one_warning(self, tmp_path, caplog) -> None:
        """With no active collect() sink, emit() degrades to a single logging
        record — the pre-migration caplog contract keeps working."""
        import logging

        from novomodelo_bridge.newave.converters.hydro import _apply_permanent_overrides

        mock_modif = modif_mock((999, []))  # 999 is not in the cadastro

        with caplog.at_level(logging.WARNING):
            _apply_permanent_overrides(
                self._base_cadastro(), self._modif_case(tmp_path, mock_modif)
            )

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1

    def test_permanent_volume_without_unit_is_hm3_and_reported(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.hydro import _apply_permanent_overrides

        volmax_rec = MagicMock()
        type(volmax_rec).__name__ = "VOLMAX"
        volmax_rec.volume = 2000.0
        volmax_rec.unidade = "??"

        mock_modif = modif_mock((1, [volmax_rec]))

        with dx.collect() as collected:
            result = _apply_permanent_overrides(
                self._base_cadastro(), self._modif_case(tmp_path, mock_modif)
            )

        assert float(result.loc[1, "volume_maximo"]) == pytest.approx(2000.0)
        assert [d.code for d in collected] == ["modif-permanent-volume-unit-unknown"]
        assert collected[0].table is not None
        assert collected[0].table.rows == [[1, "VOLMAX", "??"]]

    def test_no_findings_emits_nothing(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.hydro import _apply_permanent_overrides

        volmax_rec = MagicMock()
        type(volmax_rec).__name__ = "VOLMAX"
        volmax_rec.volume = 2000.0
        volmax_rec.unidade = "h"

        mock_modif = modif_mock((1, [volmax_rec]))

        with dx.collect() as collected:
            _apply_permanent_overrides(
                self._base_cadastro(), self._modif_case(tmp_path, mock_modif)
            )

        assert collected == []


class TestExtractTemporalOverridesDiagnostics:
    """A VAZMINT record with no month, or with neither a year nor a PRE/POS
    marker, is skipped and reported once."""

    def _modif_case(self, tmp_path, records):
        mock_modif = modif_mock((1, records))
        return make_case(
            make_nw_files(tmp_path, modif=tmp_path / "modif.dat"),
            modif=mock_modif,
        )

    @staticmethod
    def _vazmint(mes, periodo, data_inicio) -> MagicMock:
        rec = MagicMock()
        type(rec).__name__ = "VAZMINT"
        rec.mes = mes
        rec.periodo = periodo
        rec.data_inicio = data_inicio
        rec.vazao = 50.0
        return rec

    def test_undated_records_are_skipped_and_reported(self, tmp_path) -> None:
        import datetime

        from novomodelo_bridge.newave.converters.hydro import (
            _extract_temporal_overrides,
        )

        records = [
            self._vazmint(None, "POS", None),
            self._vazmint(3, None, None),
            self._vazmint(1, None, datetime.datetime(2025, 1, 1)),
        ]
        with dx.collect() as collected:
            result = _extract_temporal_overrides(
                self._modif_case(tmp_path, records), [1]
            )

        assert [o["year"] for o in result[1]] == [2025]
        [diag] = collected
        assert diag.code == "modif-temporal-override-undated"
        assert diag.severity is Severity.WARNING
        assert diag.table is not None
        assert diag.table.rows == [[1, "VAZMINT"], [1, "VAZMINT"]]
        _assert_no_repo_internal_leaks(collected)

    def test_no_sink_fallback_logs_one_warning(self, tmp_path, caplog) -> None:
        import logging

        from novomodelo_bridge.newave.converters.hydro import (
            _extract_temporal_overrides,
        )

        case = self._modif_case(tmp_path, [self._vazmint(None, "PRE", None)])
        with caplog.at_level(logging.WARNING):
            _extract_temporal_overrides(case, [1])

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1


class TestOverridesResidualLegacyWarning:
    """OQ3: none of the four codes this module now emits leaks through the
    generic ``legacy-warning`` bridge, and the residual bridge itself (shared
    by every not-yet-migrated module) still works for an unrelated string."""

    def test_hydro_finding_carries_no_legacy_warning(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.hydro import _apply_permanent_overrides

        mock_modif = modif_mock((999, []))  # 999 is not in the cadastro

        with dx.collect() as collected:
            _apply_permanent_overrides(
                _make_hidr_cadastro(),
                make_case(
                    make_nw_files(tmp_path, modif=tmp_path / "modif.dat"),
                    modif=mock_modif,
                ),
            )

        assert not any(d.code == "legacy-warning" for d in collected)

    def test_finalize_diagnostics_still_wraps_an_unrelated_legacy_string(self) -> None:
        result = finalize_diagnostics([], ["some other warning"])

        assert len(result) == 1
        assert result[0].code == "legacy-warning"
        assert result[0].summary == "some other warning"


class TestReadDiversionRecords:
    """``CDESVIO`` records of ``modif.dat``, read through inewave's own parser."""

    def _records(self, tmp_path, *lines: str):
        from novomodelo_bridge.newave.converters.hydro.overrides import (
            read_diversion_records,
        )

        return read_diversion_records(
            make_case(tmp_path, modif=_read_modif(tmp_path, *lines))
        )

    def test_each_record_belongs_to_the_usina_above_it(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.hydro.overrides import DiversionRecord

        records = self._records(
            tmp_path,
            " USINA    118                               BILLINGS",
            " VAZMIN       34",
            " CDESVIO 119 100. ",
            " USINA    119                               HENRY BORDEN",
            " VAZMIN       10",
            " USINA    314                               PIMENTAL",
            " CDESVIO 288 9999999. ",
        )
        assert records == {
            118: DiversionRecord(target=119, max_flow_m3s=100.0),
            314: DiversionRecord(target=288, max_flow_m3s=9999999.0),
        }

    def test_the_flow_is_optional(self, tmp_path) -> None:
        records = self._records(
            tmp_path,
            " USINA    118                               BILLINGS",
            " CDESVIO 119",
        )
        assert records[118].target == 119
        assert records[118].max_flow_m3s is None

    def test_an_unparseable_flow_reads_as_no_flow(self, tmp_path) -> None:
        records = self._records(
            tmp_path, " USINA    118    BILLINGS", " CDESVIO 119 abc"
        )
        assert records[118].max_flow_m3s is None

    def test_a_lowercase_keyword_is_not_a_record(self, tmp_path) -> None:
        assert (
            self._records(tmp_path, " USINA    118    BILLINGS", " cdesvio 119") == {}
        )

    def test_the_last_record_of_a_plant_wins(self, tmp_path) -> None:
        records = self._records(
            tmp_path,
            " USINA    118                               BILLINGS",
            " CDESVIO 119 100.",
            " CDESVIO 120 50.",
        )
        assert (records[118].target, records[118].max_flow_m3s) == (120, 50.0)

    def test_no_modif_and_no_record_read_as_empty(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.hydro.overrides import (
            read_diversion_records,
        )

        assert read_diversion_records(make_case(tmp_path, modif=None)) == {}
        assert (
            self._records(
                tmp_path, " USINA    118                               BILLINGS"
            )
            == {}
        )

    @pytest.mark.parametrize(
        ("lines", "reason"),
        [
            ((" CDESVIO 119 100.",), "before any USINA"),
            ((" USINA    118    BILLINGS", " CDESVIO"), "no integer target"),
            ((" USINA    118    BILLINGS", " CDESVIO X 100."), "no integer target"),
            ((" USINA    118    BILLINGS", " CDESVIO 118 100."), "diverts to itself"),
            ((" USINA    118    BILLINGS", " CDESVIO 119 -1."), "maximum flow -1;"),
        ],
    )
    def test_a_malformed_record_fails_naming_the_problem(
        self, tmp_path, lines: tuple[str, ...], reason: str
    ) -> None:
        from novomodelo_bridge.core.errors import FieldParseError

        with pytest.raises(FieldParseError, match=reason) as excinfo:
            self._records(tmp_path, *lines)
        assert excinfo.value.path == "modif.dat"
        assert excinfo.value.field == "CDESVIO"


class TestModifFileOrder:
    """Records come from every ``USINA`` block of a real ``modif.dat``, in file
    order, including the file's last record."""

    _USINA_A = " USINA      1                               USINA_A"
    _USINA_B = " USINA      2                               USINA_B"

    @staticmethod
    def _case(tmp_path, *lines: str):
        return make_case(
            make_nw_files(tmp_path, modif=tmp_path / "modif.dat"),
            modif=_read_modif(tmp_path, *lines),
        )

    def test_a_file_with_a_single_block_applies_its_records(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.hydro import _apply_permanent_overrides

        case = self._case(tmp_path, self._USINA_A, " VAZMIN       71")
        result = _apply_permanent_overrides(_make_hidr_cadastro(), case)
        assert float(result.loc[1, "vazao_minima_historica"]) == 71.0

    def test_a_cdesvio_record_is_not_an_unsupported_override(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.hydro import _apply_permanent_overrides

        case = self._case(tmp_path, self._USINA_A, " CDESVIO   2 100.")
        with dx.collect() as collected:
            _apply_permanent_overrides(_make_hidr_cadastro(), case)
        assert [d.code for d in collected] == []

    def test_the_last_record_of_the_file_is_read(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.hydro import (
            _extract_temporal_overrides,
        )

        case = self._case(
            tmp_path,
            self._USINA_A,
            " CMONT     1 2026   96.90",
            " CMONT    12 2026   96.78",
        )
        overrides = _extract_temporal_overrides(case, [1])
        assert [(o["month"], o["value"]) for o in overrides[1]] == [
            (1, pytest.approx(96.90)),
            (12, pytest.approx(96.78)),
        ]

    def test_every_block_of_a_plant_is_read_in_file_order(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.hydro import (
            _apply_permanent_overrides,
            _extract_temporal_overrides,
        )

        case = self._case(
            tmp_path,
            self._USINA_A,
            " VAZMIN       71",
            self._USINA_B,
            " VAZMIN        5",
            self._USINA_A,
            " VAZMINT   1 2026 1100.00",
            " VAZMIN       80",
        )
        cadastro = _apply_permanent_overrides(_make_hidr_cadastro(), case)
        assert float(cadastro.loc[1, "vazao_minima_historica"]) == 80.0
        assert float(cadastro.loc[2, "vazao_minima_historica"]) == 5.0
        temporal = _extract_temporal_overrides(case, [1, 2])
        assert [(o["type"], o["month"], o["value"]) for o in temporal[1]] == [
            ("VAZMINT", 1, 1100.0)
        ]
