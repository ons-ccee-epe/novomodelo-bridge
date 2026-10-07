"""Unit tests for the source model hydro productivity converter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.diagnostics import Severity
from novomodelo_bridge.newave.id_map import NewaveIdMap
from tests.conftest import make_case, make_nw_files
from tests.newave.conftest import (
    _make_cfuga_rec,
    _make_confhd_df,
    _make_hidr_cadastro,
    _make_prod_model_dger_mock,
)

# The title/summary/remediation/notes strings productivity.py emits reach a
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
# _per_stage_productivities: sazonaliza_cfuga_cmont behaviour
# ---------------------------------------------------------------------------


class TestPerStageProductivitiesSazonalCfugaCmont:
    """Verify that CFUGA/CMONT step-function carries vs. seasonal cycling.

    The source model's Dger ``sazonaliza_cfuga_cmont`` flag changes how CFUGA/CMONT
    overrides are extended beyond their last explicit
    entry: when ``0`` the last applied value carries forward
    indefinitely (pure step function); when ``1`` each calendar
    month picks up the value from the latest year that defined it
    and that seasonal pattern repeats month-by-month thereafter.
    """

    def _hreg(self) -> pd.Series:
        # Linear cota polynomial so we can read head off the coefficients.
        return _make_hreg(
            {
                "tipo_regulacao": "M",
                "volume_minimo": 0.0,
                "volume_maximo": 1000.0,
                "a0_volume_cota": 0.0,
                "a1_volume_cota": 1.0,
                "a2_volume_cota": 0.0,
                "a3_volume_cota": 0.0,
                "a4_volume_cota": 0.0,
                "canal_fuga_medio": 0.0,
                "tipo_perda": 1,
                "perdas": 0.0,
                "produtibilidade_especifica": 1.0,
                "volume_referencia": 500.0,
            }
        )

    def _dger_case(self, tmp_path, sazonaliza: int, num_anos_estudo: int = 3):
        """Build a case whose ``dger`` carries a controllable
        ``sazonaliza_cfuga_cmont``.

        ``num_anos_estudo`` defaults to 3 → study_months = 4 + 2*12 = 28 (start
        month 9), placing the seasonal-cycle assertions inside the study period.
        Lower it to push the post-study freeze boundary earlier.
        """
        mock_dger = MagicMock()
        mock_dger.ano_inicio_estudo = 2024
        mock_dger.mes_inicio_estudo = 9
        mock_dger.num_anos_estudo = num_anos_estudo
        mock_dger.sazonaliza_cfuga_cmont = sazonaliza
        return make_case(tmp_path, dger=mock_dger)

    def test_step_function_carries_when_sazonaliza_zero(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.hydro import _per_stage_productivities

        overrides = [
            {"type": "CFUGA", "year": 2024, "month": 9, "value": 5.0},
            {"type": "CFUGA", "year": 2024, "month": 10, "value": 10.0},
        ]
        vals = _per_stage_productivities(
            self._hreg(),
            base_productivity=0.0,
            drop_overrides=overrides,
            case=self._dger_case(tmp_path, sazonaliza=0),
            total_stages=24,
        )

        # tipo_regulacao = "M": v_65 = vmin + 0.65*(vmax-vmin) = 650.
        # cota(650) = 0 + 1*650 = 650. perdas=0 → prod = 1 * (650 - cfuga).
        # Stage 0 = Sep 2024 → CFUGA 5.0 → 645.
        # Stage 1 = Oct 2024 → CFUGA 10.0 → 640.
        # Stage 12 = Sep 2025 → no event → step-function carries 10.0 →
        # head = 640 (step function, NOT seasonal).
        assert vals[0] == pytest.approx(645.0)
        assert vals[1] == pytest.approx(640.0)
        assert vals[12] == pytest.approx(640.0)

    def test_seasonal_cycle_after_last_event_when_sazonaliza_one(
        self, tmp_path
    ) -> None:
        from novomodelo_bridge.newave.converters.hydro import _per_stage_productivities

        overrides = [
            {"type": "CFUGA", "year": 2024, "month": 9, "value": 5.0},
            {"type": "CFUGA", "year": 2024, "month": 10, "value": 10.0},
        ]
        vals = _per_stage_productivities(
            self._hreg(),
            base_productivity=0.0,
            drop_overrides=overrides,
            case=self._dger_case(tmp_path, sazonaliza=1),
            total_stages=24,
        )

        # See test_step_function_carries_when_sazonaliza_zero for the
        # head computation: prod = 650 - cfuga at every stage.
        # Stage 0 = Sep 2024 = explicit 5.0 → 645.
        # Stage 1 = Oct 2024 = explicit 10.0 → 640.
        # Stage 12 = Sep 2025 → AFTER last_event_stage (1, Oct 2024) →
        # seasonal cfuga[9] = 5.0 → 645.
        # Stage 13 = Oct 2025 → seasonal cfuga[10] = 10.0 → 640.
        assert vals[0] == pytest.approx(645.0)
        assert vals[1] == pytest.approx(640.0)
        assert vals[12] == pytest.approx(645.0)
        assert vals[13] == pytest.approx(640.0)

    def test_seasonal_picks_latest_year_value_per_month(self, tmp_path) -> None:
        """When the same calendar month appears in multiple years, the
        latest year's value becomes the seasonal value."""
        from novomodelo_bridge.newave.converters.hydro import _per_stage_productivities

        overrides = [
            {"type": "CFUGA", "year": 2024, "month": 9, "value": 5.0},
            {"type": "CFUGA", "year": 2025, "month": 9, "value": 7.0},  # newer
            {"type": "CFUGA", "year": 2025, "month": 10, "value": 10.0},
        ]
        vals = _per_stage_productivities(
            self._hreg(),
            base_productivity=0.0,
            drop_overrides=overrides,
            case=self._dger_case(tmp_path, sazonaliza=1),
            total_stages=36,
        )

        # Stage 24 = Sep 2026 → AFTER last_event_stage → seasonal cfuga[9]
        # = 7.0 (Sep 2025 won over Sep 2024) → prod = 650 - 7 = 643.
        assert vals[24] == pytest.approx(643.0)

    def test_no_overrides_returns_base(self, tmp_path) -> None:
        """Without any CFUGA/CMONT overrides the base value is returned
        unchanged at every stage."""
        from novomodelo_bridge.newave.converters.hydro import _per_stage_productivities

        vals = _per_stage_productivities(
            self._hreg(),
            base_productivity=42.0,
            drop_overrides=[],
            case=make_case(tmp_path),
            total_stages=12,
        )
        assert vals == [42.0] * 12

    def test_post_study_continues_seasonal_cycle(self, tmp_path) -> None:
        """Post-study continues the seasonal CFUGA/CMONT cycle (no freeze).

        VOLREF_SAZ / CFUGA-CMONT seasonal patterns are re-applied every year,
        including post-study, when ``sazonaliza_cfuga_cmont == 1`` — only the
        no-flag bounds (outflow / turbined) freeze. With study_months = 4,
        stages 4+ are post-study and must keep cycling Sep=645 / Oct=640.
        """
        from novomodelo_bridge.newave.converters.hydro import _per_stage_productivities

        overrides = [
            {"type": "CFUGA", "year": 2024, "month": 9, "value": 5.0},
            {"type": "CFUGA", "year": 2024, "month": 10, "value": 10.0},
        ]
        vals = _per_stage_productivities(
            self._hreg(),
            base_productivity=0.0,
            drop_overrides=overrides,
            case=self._dger_case(tmp_path, sazonaliza=1, num_anos_estudo=1),
            total_stages=24,
        )

        # Post-study Sep (stage 12) keeps the seasonal Sep value (645), and
        # post-study Oct (stage 13) keeps Oct (640) — NOT frozen at Dec's 640.
        assert vals[12] == pytest.approx(645.0)
        assert vals[13] == pytest.approx(640.0)


def _make_hreg(overrides: dict) -> pd.Series:
    """Build a minimal plant cadastro row (pd.Series) for unit tests.

    Provides sensible defaults for all columns consumed by
    ``_compute_productivity``.  Pass ``overrides`` to customise
    individual fields.
    """
    defaults: dict = {
        "nome_usina": "TEST",
        "produtibilidade_especifica": 0.009,
        "volume_minimo": 100.0,
        "volume_maximo": 1000.0,
        "volume_referencia": 500.0,
        "canal_fuga_medio": 250.0,
        "tipo_regulacao": "M",
        "tipo_perda": 1,
        "perdas": 5.0,  # percentage — divided by 100 in _compute_productivity
        "a0_volume_cota": 300.0,
        "a1_volume_cota": 0.1,
        "a2_volume_cota": 0.0,
        "a3_volume_cota": 0.0,
        "a4_volume_cota": 0.0,
    }
    defaults.update(overrides)
    return pd.Series(defaults)


# ---------------------------------------------------------------------------
# _compute_productivity unit tests
# ---------------------------------------------------------------------------


class TestComputeProductivity:
    """Unit tests for the ``_compute_productivity`` helper."""

    def test_monthly_regulated_linear_polynomial(self) -> None:
        """tipo_regulacao='M': poly evaluated at 65% useful storage (the source model
        ``produtibilidade_altura_65`` convention)."""
        from novomodelo_bridge.newave.converters.hydro import _compute_productivity

        hreg = _make_hreg(
            {
                "tipo_regulacao": "M",
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "a2_volume_cota": 0.0,
                "a3_volume_cota": 0.0,
                "a4_volume_cota": 0.0,
                "volume_minimo": 100.0,
                "volume_maximo": 1000.0,
                "canal_fuga_medio": 250.0,
                "tipo_perda": 1,
                "perdas": 5.0,  # 5%
                "produtibilidade_especifica": 0.009,
            }
        )
        # 65% of useful storage: v_65 = 100 + 0.65 * (1000 - 100) = 685.0
        # poly(685) = 300 + 0.1 * 685 = 368.5
        # net_drop = 368.5 - 250.0 = 118.5
        # adjusted_drop = 118.5 * (1 - 5.0/100) = 112.575
        # result = 0.009 * 112.575 = 1.013175
        v_65_height = 300.0 + 0.1 * (100.0 + 0.65 * (1000.0 - 100.0))
        expected = 0.009 * (1.0 - 5.0 / 100.0) * (v_65_height - 250.0)
        result = _compute_productivity(hreg)
        assert result == pytest.approx(expected)

    def test_run_of_river_point_evaluation(self) -> None:
        """tipo_regulacao='D': evaluates poly at volume_referencia."""
        from novomodelo_bridge.newave.converters.hydro import _compute_productivity

        hreg = _make_hreg(
            {
                "tipo_regulacao": "D",
                "volume_referencia": 500.0,
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "a2_volume_cota": 0.0,
                "a3_volume_cota": 0.0,
                "a4_volume_cota": 0.0,
                "canal_fuga_medio": 250.0,
                "tipo_perda": 1,
                "perdas": 5.0,  # 5%
                "produtibilidade_especifica": 0.009,
            }
        )
        # poly(500) = 300 + 0.1*500 = 350.0
        # net_drop = 350.0 - 250.0 = 100.0
        # adjusted_drop = 100.0 * (1 - 5.0/100) = 95.0
        # result = 0.009 * 95.0 = 0.855
        expected = 0.009 * (1.0 - 5.0 / 100.0) * (350.0 - 250.0)
        result = _compute_productivity(hreg)
        assert result == pytest.approx(expected)

    def test_multiplicative_loss(self) -> None:
        """tipo_perda=1: adjusted_drop = net_drop * (1 - perdas/100)."""
        from novomodelo_bridge.newave.converters.hydro import _compute_productivity

        hreg = _make_hreg(
            {
                "tipo_regulacao": "D",
                "volume_referencia": 500.0,
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "a2_volume_cota": 0.0,
                "a3_volume_cota": 0.0,
                "a4_volume_cota": 0.0,
                "canal_fuga_medio": 250.0,
                "tipo_perda": 1,
                "perdas": 10.0,  # 10%
                "produtibilidade_especifica": 0.009,
            }
        )
        # net_drop = (300 + 50) - 250 = 100.0
        # adjusted = 100.0 * (1 - 10.0/100) = 90.0
        expected = 0.009 * 90.0
        result = _compute_productivity(hreg)
        assert result == pytest.approx(expected)

    def test_additive_loss(self) -> None:
        """tipo_perda=2: adjusted_drop = net_drop - perdas."""
        from novomodelo_bridge.newave.converters.hydro import _compute_productivity

        hreg = _make_hreg(
            {
                "tipo_regulacao": "D",
                "volume_referencia": 500.0,
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "a2_volume_cota": 0.0,
                "a3_volume_cota": 0.0,
                "a4_volume_cota": 0.0,
                "canal_fuga_medio": 250.0,
                "tipo_perda": 2,
                "perdas": 3.5,
                "produtibilidade_especifica": 0.009,
            }
        )
        # net_drop = 350.0 - 250.0 = 100.0
        # adjusted = 100.0 - 3.5 = 96.5
        expected = 0.009 * 96.5
        result = _compute_productivity(hreg)
        assert result == pytest.approx(expected)

    def test_no_loss(self) -> None:
        """tipo_perda=0 (or unknown): no loss applied, adjusted_drop = net_drop."""
        from novomodelo_bridge.newave.converters.hydro import _compute_productivity

        hreg = _make_hreg(
            {
                "tipo_regulacao": "D",
                "volume_referencia": 500.0,
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "a2_volume_cota": 0.0,
                "a3_volume_cota": 0.0,
                "a4_volume_cota": 0.0,
                "canal_fuga_medio": 250.0,
                "tipo_perda": 0,
                "perdas": 99.0,
                "produtibilidade_especifica": 0.009,
            }
        )
        # tipo_perda=0 -> no loss applied, perdas value ignored
        # net_drop = 350.0 - 250.0 = 100.0
        expected = 0.009 * 100.0
        result = _compute_productivity(hreg)
        assert result == pytest.approx(expected)

    def test_equal_volumes_fallback(self) -> None:
        """tipo_regulacao='M' with vmin == vmax: v_65 collapses to that point."""
        from novomodelo_bridge.newave.converters.hydro import _compute_productivity

        hreg = _make_hreg(
            {
                "tipo_regulacao": "M",
                "volume_minimo": 500.0,
                "volume_maximo": 500.0,
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "a2_volume_cota": 0.0,
                "a3_volume_cota": 0.0,
                "a4_volume_cota": 0.0,
                "canal_fuga_medio": 250.0,
                "tipo_perda": 1,
                "perdas": 0.0,
                "produtibilidade_especifica": 0.009,
            }
        )
        # vmin == vmax: v_65 = 500.0; poly(500) = 350.0; net_drop = 100.0
        expected = 0.009 * 100.0
        result = _compute_productivity(hreg)
        assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _compute_productivity with override parameters
# ---------------------------------------------------------------------------


class TestComputeProductivityOverrides:
    """Unit tests for ``_compute_productivity`` with canal_fuga/cmont overrides."""

    def test_canal_fuga_override_replaces_base(self) -> None:
        """canal_fuga_override replaces canal_fuga_medio in the net drop calc."""
        from novomodelo_bridge.newave.converters.hydro import _compute_productivity

        hreg = _make_hreg(
            {
                "tipo_regulacao": "D",
                "volume_referencia": 500.0,
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "canal_fuga_medio": 250.0,
                "tipo_perda": 0,
                "perdas": 0.0,
                "produtibilidade_especifica": 0.009,
            }
        )
        # poly(500) = 300 + 50 = 350
        # With override canal_fuga=260: net_drop = 350 - 260 = 90
        base = _compute_productivity(hreg)  # uses 250 -> drop 100
        overridden = _compute_productivity(hreg, canal_fuga_override=260.0)
        assert base == pytest.approx(0.009 * 100.0)
        assert overridden == pytest.approx(0.009 * 90.0)

    def test_cmont_override_replaces_polynomial_height(self) -> None:
        """cmont_override bypasses the polynomial and uses the supplied height."""
        from novomodelo_bridge.newave.converters.hydro import _compute_productivity

        hreg = _make_hreg(
            {
                "tipo_regulacao": "D",
                "volume_referencia": 500.0,
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "canal_fuga_medio": 250.0,
                "tipo_perda": 0,
                "perdas": 0.0,
                "produtibilidade_especifica": 0.009,
            }
        )
        # cmont=380 overrides polynomial; net_drop = 380 - 250 = 130
        result = _compute_productivity(hreg, cmont_override=380.0)
        assert result == pytest.approx(0.009 * 130.0)

    def test_both_overrides_together(self) -> None:
        """canal_fuga_override and cmont_override can both be active."""
        from novomodelo_bridge.newave.converters.hydro import _compute_productivity

        hreg = _make_hreg(
            {
                "tipo_regulacao": "D",
                "volume_referencia": 500.0,
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "canal_fuga_medio": 250.0,
                "tipo_perda": 0,
                "perdas": 0.0,
                "produtibilidade_especifica": 0.009,
            }
        )
        # cmont=400, canal_fuga=260 -> net_drop = 400 - 260 = 140
        result = _compute_productivity(
            hreg, canal_fuga_override=260.0, cmont_override=400.0
        )
        assert result == pytest.approx(0.009 * 140.0)

    def test_no_overrides_matches_original_behaviour(self) -> None:
        """With no overrides, M-plant ρ comes from poly evaluated at 65% storage."""
        from novomodelo_bridge.newave.converters.hydro import _compute_productivity

        hreg = _make_hreg(
            {
                "tipo_regulacao": "M",
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "volume_minimo": 100.0,
                "volume_maximo": 1000.0,
                "canal_fuga_medio": 250.0,
                "tipo_perda": 1,
                "perdas": 5.0,  # 5%
                "produtibilidade_especifica": 0.009,
            }
        )
        # v_65 = 100 + 0.65 * 900 = 685; poly(685) = 368.5; net_drop = 118.5
        v_65_height = 300.0 + 0.1 * (100.0 + 0.65 * (1000.0 - 100.0))
        expected = 0.009 * (1.0 - 5.0 / 100.0) * (v_65_height - 250.0)
        assert _compute_productivity(hreg) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# convert_production_models unit tests
# ---------------------------------------------------------------------------


def _make_cmont_rec(month: int, year: int, nivel: float) -> MagicMock:
    import datetime

    r = MagicMock()
    type(r).__name__ = "CMONT"
    r.data_inicio = datetime.datetime(year, month, 1)
    r.nivel = nivel
    return r


class TestEquivalentProductivity:
    """Unit tests for ``_equivalent_productivity`` (the source model PRODT)."""

    def test_linear_polynomial_uses_mean_head(self) -> None:
        """h(v)=a0+a1·v → mean head over [vmin,vmax] = a0 + a1·(vmin+vmax)/2."""
        from novomodelo_bridge.newave.converters.hydro import _equivalent_productivity

        hreg = _make_hreg({})  # a0=300, a1=0.1, vmin=100, vmax=1000, cfuga=250
        # mean head = 300 + 0.1·550 = 355 ; net = 105 ; ·0.95 loss ; ·0.009 pesp
        assert _equivalent_productivity(hreg) == pytest.approx(
            0.009 * (355.0 - 250.0) * 0.95
        )

    def test_differs_from_point_reference(self) -> None:
        """PRODT (mean over range) ≠ the 65%-volume point productivity."""
        from novomodelo_bridge.newave.converters.hydro import (
            _compute_productivity,
            _equivalent_productivity,
        )

        # Non-linear forebay curve so the average over [vmin,vmax] differs from
        # the value at the 65% reference point.
        hreg = _make_hreg({"a2_volume_cota": 1e-4})
        assert _equivalent_productivity(hreg) != pytest.approx(
            _compute_productivity(hreg)
        )

    def test_run_of_river_uses_point_head(self) -> None:
        """Vmax == Vmin → head evaluated at Vmin (no integral)."""
        from novomodelo_bridge.newave.converters.hydro import _equivalent_productivity

        hreg = _make_hreg({"volume_minimo": 500.0, "volume_maximo": 500.0})
        # h(500) = 300 + 0.1·500 = 350 ; net 100 ; ·0.95 ; ·0.009
        assert _equivalent_productivity(hreg) == pytest.approx(
            0.009 * (350.0 - 250.0) * 0.95
        )

    def test_canal_fuga_override(self) -> None:
        from novomodelo_bridge.newave.converters.hydro import _equivalent_productivity

        hreg = _make_hreg({})
        assert _equivalent_productivity(
            hreg, canal_fuga_override=300.0
        ) == pytest.approx(0.009 * (355.0 - 300.0) * 0.95)

    def test_cmont_override_pins_forebay(self) -> None:
        """CMONT pins the upstream level → head = cmont − cfuga (no integral)."""
        from novomodelo_bridge.newave.converters.hydro import _equivalent_productivity

        hreg = _make_hreg({})
        assert _equivalent_productivity(hreg, cmont_override=400.0) == pytest.approx(
            0.009 * (400.0 - 250.0) * 0.95
        )

    def test_zero_polynomial_returns_zero(self) -> None:
        from novomodelo_bridge.newave.converters.hydro import _equivalent_productivity

        hreg = _make_hreg({f"a{i}_volume_cota": 0.0 for i in range(5)})
        assert _equivalent_productivity(hreg) == 0.0


class TestProductivitySinMeans:
    """SIN-mean productivity aggregation over synthetic plant sets.

    These exercise the EX / FICT / out-of-cadastro filtering and the averaging
    wiring of the ``PROD_MEDIA_SIN`` helpers without depending on the
    (git-ignored) example case. Every expected value is *derived* from the same
    synthetic cadastro the function reads — never a hard-coded example snapshot.
    The per-plant productivity math itself is covered by
    :class:`TestEquivalentProductivity` / :class:`TestComputeProductivity`.
    """

    @staticmethod
    def _cadastro(rows: dict[int, dict]) -> pd.DataFrame:
        """Build a ``Hidr.cadastro``-shaped frame indexed by plant code."""
        return pd.DataFrame({code: _make_hreg(ov) for code, ov in rows.items()}).T

    @staticmethod
    def _confhd(rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def _case(self, tmp_path, cadastro: pd.DataFrame, confhd: pd.DataFrame):
        """Build a case with hidr/confhd mocks. The ``_apply_permanent_overrides``
        identity patch is returned separately so callers can ``with`` on it."""
        hidr_obj = MagicMock()
        hidr_obj.cadastro = cadastro
        confhd_obj = MagicMock()
        confhd_obj.usinas = confhd
        case = make_case(tmp_path, hidr=hidr_obj, confhd=confhd_obj)
        overrides = patch(
            "novomodelo_bridge.newave.converters.hydro.productivity._apply_permanent_overrides",
            new=lambda cadastro, case: cadastro,
        )
        return case, overrides

    def test_prodt_sin_mean_averages_existing_nonfict_in_cadastro(
        self, tmp_path: Path
    ) -> None:
        """Mean over EX, non-FICT plants present in cadastro; others excluded."""
        from novomodelo_bridge.newave.converters.hydro import (
            _equivalent_productivity,
            compute_prodt_sin_mean,
        )

        cadastro = self._cadastro(
            {
                1: {"produtibilidade_especifica": 0.009, "canal_fuga_medio": 250.0},
                2: {"produtibilidade_especifica": 0.010, "canal_fuga_medio": 300.0},
                # code 3: ρ=0 sharing plant 1's posto → fictitious → excluded
                3: {"produtibilidade_especifica": 0.0},
                # code 4 is EX/non-fict but absent from cadastro → skipped
            }
        )
        confhd = self._confhd(
            [
                {
                    "codigo_usina": 1,
                    "nome_usina": "PLANT A",
                    "posto": 1,
                    "usina_existente": "EX",
                },
                {
                    "codigo_usina": 2,
                    "nome_usina": "PLANT B",
                    "posto": 2,
                    "usina_existente": "EX",
                },
                {
                    "codigo_usina": 3,
                    "nome_usina": "FICT TWIN",
                    "posto": 1,
                    "usina_existente": "EX",
                },
                {
                    "codigo_usina": 4,
                    "nome_usina": "PLANT D",
                    "posto": 4,
                    "usina_existente": "EX",
                },
                {
                    "codigo_usina": 5,
                    "nome_usina": "PLANT E",
                    "posto": 5,
                    "usina_existente": "NE",
                },
            ]
        )
        case, overrides = self._case(tmp_path, cadastro, confhd)
        with overrides:
            result = compute_prodt_sin_mean(case)

        expected = (
            _equivalent_productivity(cadastro.loc[1])
            + _equivalent_productivity(cadastro.loc[2])
        ) / 2
        assert result == pytest.approx(expected)

    def test_prodt_sin_mean_no_eligible_plants_returns_unit_fallback(
        self, tmp_path: Path
    ) -> None:
        """No EX/non-FICT plant in cadastro → fall back to 1.0, not divide-by-zero."""
        from novomodelo_bridge.newave.converters.hydro import compute_prodt_sin_mean

        cadastro = self._cadastro({1: {}})
        confhd = self._confhd(
            [
                {
                    "codigo_usina": 9,
                    "nome_usina": "FICT. ONLY",
                    "usina_existente": "EX",
                },
                {"codigo_usina": 8, "nome_usina": "GONE", "usina_existente": "NE"},
            ]
        )
        case, overrides = self._case(tmp_path, cadastro, confhd)
        with overrides:
            assert compute_prodt_sin_mean(case) == 1.0

    def test_per_stage_prodt_flat_without_temporal_overrides(
        self, tmp_path: Path
    ) -> None:
        """No CFUGA/CMONT override → every stage equals the constant SIN mean."""
        from novomodelo_bridge.newave.converters.hydro import (
            _equivalent_productivity,
            compute_per_stage_prodt_sin_mean,
        )

        cadastro = self._cadastro(
            {
                1: {"produtibilidade_especifica": 0.009},
                2: {"produtibilidade_especifica": 0.011, "canal_fuga_medio": 280.0},
            }
        )
        confhd = self._confhd(
            [
                {"codigo_usina": 1, "nome_usina": "PLANT A", "usina_existente": "EX"},
                {"codigo_usina": 2, "nome_usina": "PLANT B", "usina_existente": "EX"},
            ]
        )
        case, overrides = self._case(tmp_path, cadastro, confhd)
        with (
            overrides,
            patch(
                "novomodelo_bridge.newave.converters.hydro.productivity._total_study_stages",
                return_value=4,
            ),
            patch(
                "novomodelo_bridge.newave.converters.hydro.productivity._extract_temporal_overrides",
                return_value={},
            ),
        ):
            per_stage = compute_per_stage_prodt_sin_mean(case)

        base = (
            _equivalent_productivity(cadastro.loc[1])
            + _equivalent_productivity(cadastro.loc[2])
        ) / 2
        assert len(per_stage) == 4
        assert all(v == pytest.approx(base) for v in per_stage)

    def test_per_stage_prodt_routes_overrides_and_averages_per_stage(
        self, tmp_path: Path
    ) -> None:
        """A plant carrying a CFUGA override drifts; the SIN mean tracks it per
        stage."""
        from novomodelo_bridge.newave.converters.hydro import (
            _equivalent_productivity,
            compute_per_stage_prodt_sin_mean,
        )

        cadastro = self._cadastro({1: {}, 2: {"produtibilidade_especifica": 0.011}})
        confhd = self._confhd(
            [
                {"codigo_usina": 1, "nome_usina": "PLANT A", "usina_existente": "EX"},
                {"codigo_usina": 2, "nome_usina": "PLANT B", "usina_existente": "EX"},
            ]
        )

        def fake_series(hreg, base, drops, case, total_stages):
            # Plants with a routed CFUGA/CMONT override drift per stage; others flat.
            if drops:
                return [base, base * 1.02, base * 0.98][:total_stages]
            return [base] * total_stages

        case, overrides = self._case(tmp_path, cadastro, confhd)
        with (
            overrides,
            patch(
                "novomodelo_bridge.newave.converters.hydro.productivity._total_study_stages",
                return_value=3,
            ),
            patch(
                "novomodelo_bridge.newave.converters.hydro.productivity._extract_temporal_overrides",
                return_value={1: [{"type": "CFUGA"}]},
            ),
            patch(
                "novomodelo_bridge.newave.converters.hydro.productivity._per_stage_equivalent_productivities",
                side_effect=fake_series,
            ),
        ):
            per_stage = compute_per_stage_prodt_sin_mean(case)

        b1 = _equivalent_productivity(cadastro.loc[1])
        b2 = _equivalent_productivity(cadastro.loc[2])
        expected = [
            (b1 + b2) / 2,
            (b1 * 1.02 + b2) / 2,
            (b1 * 0.98 + b2) / 2,
        ]
        assert per_stage == pytest.approx(expected)

    def test_max_prodtacum_sin_picks_cascade_max(self, tmp_path: Path) -> None:
        """Accumulated productivity peaks at the head of the longest cascade."""
        from novomodelo_bridge.newave.converters.constraints import (
            compute_max_prodtacum_sin,
        )
        from novomodelo_bridge.newave.converters.hydro import _compute_productivity

        # Cascade A(1) → B(2) → terminal; C(3) standalone.
        cadastro = self._cadastro({1: {}, 2: {}, 3: {}})
        confhd = self._confhd(
            [
                {
                    "codigo_usina": 1,
                    "nome_usina": "A",
                    "usina_existente": "EX",
                    "codigo_usina_jusante": 2,
                },
                {
                    "codigo_usina": 2,
                    "nome_usina": "B",
                    "usina_existente": "EX",
                    "codigo_usina_jusante": 0,
                },
                {
                    "codigo_usina": 3,
                    "nome_usina": "C",
                    "usina_existente": "EX",
                    "codigo_usina_jusante": 0,
                },
            ]
        )
        hidr_obj = MagicMock()
        hidr_obj.cadastro = cadastro
        confhd_obj = MagicMock()
        confhd_obj.usinas = confhd
        case = make_case(tmp_path, hidr=hidr_obj, confhd=confhd_obj)
        with patch(
            "novomodelo_bridge.newave.converters.constraints._apply_permanent_overrides",
            new=lambda cadastro, case: cadastro,
        ):
            result = compute_max_prodtacum_sin(case)

        def own(code: int) -> float:
            hreg = cadastro.loc[code]
            useful = float(hreg["volume_maximo"]) - float(hreg["volume_minimo"])
            return _compute_productivity(hreg, useful_volume_override=useful)

        acc_a = own(1) + own(2)  # A accumulates B downstream
        assert result == pytest.approx(max(acc_a, own(2), own(3)))

    def test_max_prodtacum_sin_returns_none_on_read_error(self, tmp_path: Path) -> None:
        """Unreadable the source model inputs → None (soft fallback for mocked
        pipelines)."""
        from novomodelo_bridge.newave.converters.constraints import (
            compute_max_prodtacum_sin,
        )

        case = make_case(tmp_path)
        # Drop the conftest default hidr so accessing ``case.hidr`` triggers the
        # patched ``Hidr.read`` (this test exercises the read-error fallback).
        case.__dict__.pop("hidr", None)
        with patch("novomodelo_bridge.newave.case.Hidr") as mh:
            mh.read.side_effect = OSError("no file")
            assert compute_max_prodtacum_sin(case) is None


class TestConvertProductionModels:
    """Unit tests for ``convert_production_models``."""

    def _make_id_map(self) -> NewaveIdMap:
        return NewaveIdMap(
            subsystem_ids=[1],
            hydro_codes=[1, 2],
            thermal_codes=[],
        )

    def _base_case(
        self,
        tmp_path: Path,
        *,
        modif=None,
        ano_inicio: int = 2025,
        mes_inicio: int = 1,
        num_anos: int = 5,
    ):
        """Build a case with hidr/confhd/dger mocks (and optional modif).

        Pass *modif* as a mock reader object; the matching path is set so the
        ``case.files.modif`` guard sees a present file.
        """
        mock_hidr = MagicMock()
        mock_hidr.cadastro = _make_hidr_cadastro()

        mock_confhd = MagicMock()
        mock_confhd.usinas = _make_confhd_df()

        mock_dger = _make_prod_model_dger_mock(
            ano_inicio=ano_inicio,
            mes_inicio=mes_inicio,
            num_anos=num_anos,
            num_anos_pos=0,
        )

        parsed: dict = {"hidr": mock_hidr, "confhd": mock_confhd, "dger": mock_dger}
        if modif is not None:
            parsed["modif"] = modif
            files = make_nw_files(tmp_path, modif=tmp_path / "modif.dat")
        else:
            files = make_nw_files(tmp_path)
        return make_case(files, **parsed)

    def test_returns_all_hydros_when_no_modif(self, tmp_path: Path) -> None:
        """No MODIF.DAT: every hydro still gets a single-range entry.

        Novomodelo HEAD requires the productivity for every hydro to live in
        ``hydro_production_models.json`` (it was removed from hydros.json
        generation block), so we always emit an entry per plant.
        """
        case = self._base_case(tmp_path)

        from novomodelo_bridge.newave.converters.hydro import convert_production_models

        result = convert_production_models(case, self._make_id_map())
        assert result is not None
        models = result["production_models"]
        assert len(models) == 2  # USINA_A and USINA_B
        for model in models:
            assert model["selection_mode"] == "stage_ranges"
            ranges = model["stage_ranges"]
            assert len(ranges) == 1
            assert ranges[0]["start_stage_id"] == 0
            assert ranges[0]["end_stage_id"] is None
            # Productivity now lives in hydro_energy_productivity.parquet,
            # not in the JSON stage_range entries.
            assert "productivity_mw_per_m3s" not in ranges[0]

    def test_returns_all_hydros_when_no_cfuga_cmont(self, tmp_path: Path) -> None:
        """MODIF.DAT present but only VAZMINT overrides -> per-hydro entries with single
        range."""
        import datetime

        vazmint_rec = MagicMock()
        type(vazmint_rec).__name__ = "VAZMINT"
        vazmint_rec.data_inicio = datetime.datetime(2025, 3, 1)
        vazmint_rec.vazao = 50.0

        usina_rec = MagicMock()
        usina_rec.codigo = 1

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [vazmint_rec]

        case = self._base_case(tmp_path, modif=mock_modif)

        from novomodelo_bridge.newave.converters.hydro import convert_production_models

        result = convert_production_models(case, self._make_id_map())
        assert result is not None
        models = result["production_models"]
        # Both hydros still get an entry, each with one stage range
        # covering the whole horizon (no CFUGA/CMONT temporal overrides).
        assert len(models) == 2
        for model in models:
            ranges = model["stage_ranges"]
            assert len(ranges) == 1
            assert ranges[0]["end_stage_id"] is None

    def test_single_cfuga_override_two_ranges(self, tmp_path: Path) -> None:
        """One CFUGA override at stage 3 -> two stage_ranges (base then overridden)."""
        cfuga_rec = _make_cfuga_rec(month=4, year=2025, nivel=60.0)
        usina_rec = MagicMock()
        usina_rec.codigo = 1

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [cfuga_rec]

        # Study: start Jan 2025, 5 years -> 60 stages total.
        case = self._base_case(
            tmp_path, modif=mock_modif, ano_inicio=2025, mes_inicio=1, num_anos=5
        )

        from novomodelo_bridge.newave.converters.hydro import convert_production_models

        result = convert_production_models(case, self._make_id_map())

        assert result is not None
        assert "production_models" in result
        models = result["production_models"]
        # JSON now carries only model selection — productivity moved to
        # hydro_energy_productivity.parquet. Both USINA_A and USINA_B emit
        # one model-only stage_range entry; per-stage variation for USINA_A's
        # CFUGA override is asserted via TestConvertHydroEnergyProductivity.
        assert len(models) == 2

        model_a = next(m for m in models if m["hydro_id"] == 0)
        assert model_a["selection_mode"] == "stage_ranges"
        ranges = model_a["stage_ranges"]
        assert len(ranges) == 1
        assert ranges[0]["start_stage_id"] == 0
        assert ranges[0]["end_stage_id"] is None
        assert ranges[0]["model"] == "constant_productivity"
        assert "productivity_mw_per_m3s" not in ranges[0]

    def test_cmont_override_bypasses_polynomial(self, tmp_path: Path) -> None:
        """CMONT override at stage 0 -> single stage_range using cmont as height."""
        cmont_rec = _make_cmont_rec(month=1, year=2025, nivel=400.0)
        usina_rec = MagicMock()
        usina_rec.codigo = 1

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [cmont_rec]

        case = self._base_case(
            tmp_path, modif=mock_modif, ano_inicio=2025, mes_inicio=1, num_anos=5
        )

        from novomodelo_bridge.newave.converters.hydro import convert_production_models

        result = convert_production_models(case, self._make_id_map())

        assert result is not None
        models = result["production_models"]
        # JSON has model-only entries — productivity moved to the parquet.
        assert len(models) == 2
        model_a = next(m for m in models if m["hydro_id"] == 0)
        ranges = model_a["stage_ranges"]
        assert len(ranges) == 1
        assert ranges[0]["start_stage_id"] == 0
        assert ranges[0]["end_stage_id"] is None
        assert "productivity_mw_per_m3s" not in ranges[0]

    def test_multiple_overrides_three_ranges(self, tmp_path: Path) -> None:
        """Two CFUGA overrides -> three stage_ranges."""
        recs = [
            _make_cfuga_rec(month=6, year=2025, nivel=55.0),  # stage 5
            _make_cfuga_rec(month=1, year=2026, nivel=65.0),  # stage 12
        ]
        usina_rec = MagicMock()
        usina_rec.codigo = 1

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = recs

        case = self._base_case(
            tmp_path, modif=mock_modif, ano_inicio=2025, mes_inicio=1, num_anos=5
        )

        from novomodelo_bridge.newave.converters.hydro import convert_production_models

        result = convert_production_models(case, self._make_id_map())

        assert result is not None
        # JSON now has a single model-only stage_range per hydro; the multiple
        # CFUGA breakpoints surface in hydro_energy_productivity.parquet.
        ranges = result["production_models"][0]["stage_ranges"]
        assert len(ranges) == 1
        assert ranges[0]["start_stage_id"] == 0
        assert ranges[0]["end_stage_id"] is None
        assert "productivity_mw_per_m3s" not in ranges[0]

    def test_output_sorted_by_hydro_id(self, tmp_path: Path) -> None:
        """production_models list is sorted ascending by hydro_id."""
        # Both plants have CFUGA overrides; plant codes 1 and 2 -> ids 0 and 1.
        usina_rec1 = MagicMock()
        usina_rec1.codigo = 1
        usina_rec2 = MagicMock()
        usina_rec2.codigo = 2

        def _mods(code: int) -> list:
            if code == 1:
                return [_make_cfuga_rec(month=3, year=2025, nivel=55.0)]
            return [_make_cfuga_rec(month=6, year=2025, nivel=55.0)]

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec2, usina_rec1]  # reversed order
        mock_modif.modificacoes_usina.side_effect = _mods

        case = self._base_case(
            tmp_path, modif=mock_modif, ano_inicio=2025, mes_inicio=1, num_anos=5
        )

        from novomodelo_bridge.newave.converters.hydro import convert_production_models

        result = convert_production_models(case, self._make_id_map())

        assert result is not None
        ids = [m["hydro_id"] for m in result["production_models"]]
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# Hydro energy productivity (parquet) conversion
# ---------------------------------------------------------------------------


class TestConvertHydroEnergyProductivity:
    """Per-(hydro, stage) ρ_eq override parquet for the novomodelo productivity-resolution
    contract."""

    def _make_id_map(self) -> NewaveIdMap:
        return NewaveIdMap(
            subsystem_ids=[1],
            hydro_codes=[1, 2],
            thermal_codes=[],
        )

    def _base_case(
        self,
        tmp_path: Path,
        *,
        modif=None,
        volref_volumes: pd.DataFrame | None = None,
        ano_inicio: int = 2025,
        mes_inicio: int = 1,
        num_anos: int = 5,
    ):
        """Build a case with hidr/confhd/dger mocks plus optional modif/volref.

        *modif* is a mock reader object; *volref_volumes* is a DataFrame for the
        VolrefSaz reader. Matching file paths are set so the ``case.files.X``
        guards see present files.
        """
        mock_hidr = MagicMock()
        mock_hidr.cadastro = _make_hidr_cadastro()

        mock_confhd = MagicMock()
        mock_confhd.usinas = _make_confhd_df()

        mock_dger = _make_prod_model_dger_mock(
            ano_inicio=ano_inicio,
            mes_inicio=mes_inicio,
            num_anos=num_anos,
            num_anos_pos=0,
        )

        parsed: dict = {"hidr": mock_hidr, "confhd": mock_confhd, "dger": mock_dger}
        file_overrides: dict = {}
        if modif is not None:
            parsed["modif"] = modif
            file_overrides["modif"] = tmp_path / "modif.dat"
        if volref_volumes is not None:
            mock_volref = MagicMock()
            mock_volref.volumes = volref_volumes
            parsed["volref_saz"] = mock_volref
            file_overrides["volref_saz"] = tmp_path / "volref_saz.dat"
        return make_case(make_nw_files(tmp_path, **file_overrides), **parsed)

    def test_null_stage_row_per_hydro_when_no_overrides(self, tmp_path: Path) -> None:
        """Without CFUGA/CMONT: one NULL-stage_id row per hydro with base
        productivity."""
        case = self._base_case(tmp_path)

        from novomodelo_bridge.newave.converters.hydro import (
            convert_hydro_energy_productivity,
        )

        table = convert_hydro_energy_productivity(case, self._make_id_map())

        assert table.num_rows == 2
        assert table.column_names[:2] == ["hydro_id", "stage_id"]
        stage_ids = table["stage_id"].to_pylist()
        assert stage_ids == [None, None]
        prods = table["equivalent_productivity_mw_per_m3s"].to_pylist()
        # USINA_A: v_65=685, poly(685)=368.5, net_drop=318.5, ρ=0.9 * 318.5
        assert prods[0] == pytest.approx(0.9 * 318.5)

    def test_per_stage_rows_for_cfuga_override(self, tmp_path: Path) -> None:
        """CFUGA at stage 3 → per-stage rows for the full horizon; stages [0..2] use
        base."""
        cfuga_rec = _make_cfuga_rec(month=4, year=2025, nivel=60.0)
        usina_rec = MagicMock()
        usina_rec.codigo = 1
        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [cfuga_rec]

        case = self._base_case(
            tmp_path, modif=mock_modif, ano_inicio=2025, mes_inicio=1, num_anos=5
        )

        from novomodelo_bridge.newave.converters.hydro import (
            convert_hydro_energy_productivity,
        )

        table = convert_hydro_energy_productivity(case, self._make_id_map())

        # USINA_A (hydro_id=0): 60 per-stage rows. USINA_B (hydro_id=1): 1 NULL row.
        rows = table.to_pylist()
        a_rows = [r for r in rows if r["hydro_id"] == 0]
        b_rows = [r for r in rows if r["hydro_id"] == 1]
        assert len(a_rows) == 60  # 5 years * 12 months
        assert len(b_rows) == 1
        assert b_rows[0]["stage_id"] is None

        # USINA_A is tipo_regulacao="M", so _compute_productivity evaluates the
        # cota polynomial at v_65 = vmin + 0.65·useful = 100 + 0.65·900 = 685.
        # h(685) = 300 + 0.1·685 = 368.5.  CFUGA overrides canal_fuga from
        # cadastro's 50.0 to 60.0 starting at stage 3.
        base = 0.9 * (368.5 - 50.0)
        override = 0.9 * (368.5 - 60.0)
        # Stages 0..2 = base, stages 3..59 = override.
        a_by_stage = {
            r["stage_id"]: r["equivalent_productivity_mw_per_m3s"] for r in a_rows
        }
        assert a_by_stage[0] == pytest.approx(base)
        assert a_by_stage[2] == pytest.approx(base)
        assert a_by_stage[3] == pytest.approx(override)
        assert a_by_stage[59] == pytest.approx(override)

    def test_seasonal_volref_emits_per_stage_rows(self, tmp_path: Path) -> None:
        """volref_saz row with non-zero values → per-stage ρ computed at
        ``vol_min + volref[month]`` for that calendar month."""
        # USINA_A: seasonal row (every month has its own useful volume).
        # USINA_B: not present in the file → falls back to altura_65 default.
        volref_df = pd.DataFrame(
            {
                "codigo_usina": [1] * 12,
                "nome_usina": ["USINA_A"] * 12,
                "mes": list(range(1, 13)),
                # Useful volumes (hm³ above vol_min=100): 100, 200, ..., 1200
                "valor": [float(100 * m) for m in range(1, 13)],
            }
        )

        case = self._base_case(
            tmp_path,
            volref_volumes=volref_df,
            ano_inicio=2025,
            mes_inicio=1,
            num_anos=1,
        )

        from novomodelo_bridge.newave.converters.hydro import (
            convert_hydro_energy_productivity,
        )

        table = convert_hydro_energy_productivity(case, self._make_id_map())
        rows = table.to_pylist()
        a_rows = [r for r in rows if r["hydro_id"] == 0]
        b_rows = [r for r in rows if r["hydro_id"] == 1]

        # USINA_A: 12 per-stage rows (1 year * 12 months); USINA_B: 1 null row.
        assert len(a_rows) == 12
        assert len(b_rows) == 1
        assert b_rows[0]["stage_id"] is None

        # USINA_A stage 0 = calendar month 1, useful=100, V=200, h(V)=320, drop=270,
        # ρ=243.
        by_stage = {
            r["stage_id"]: r["equivalent_productivity_mw_per_m3s"] for r in a_rows
        }
        # ρ_esp=0.9, cf=50, h(v)=300+0.1v
        for stage in range(12):
            useful = 100.0 * (stage + 1)
            expected = 0.9 * ((300.0 + 0.1 * (100.0 + useful)) - 50.0)
            assert by_stage[stage] == pytest.approx(expected)

    def test_seasonal_volref_all_zero_row_falls_back_to_legacy(
        self, tmp_path: Path
    ) -> None:
        """All-zero volref_saz row is the source model's "no seasonal reference"
        sentinel:
        emit a single null-stage row with the legacy altura_65 productivity."""
        volref_df = pd.DataFrame(
            {
                "codigo_usina": [1] * 12 + [2] * 12,
                "nome_usina": ["USINA_A"] * 12 + ["USINA_B"] * 12,
                "mes": list(range(1, 13)) * 2,
                # USINA_A all zeros (sentinel); USINA_B all zeros (sentinel).
                "valor": [0.0] * 24,
            }
        )

        case = self._base_case(
            tmp_path,
            volref_volumes=volref_df,
            ano_inicio=2025,
            mes_inicio=1,
            num_anos=1,
        )

        from novomodelo_bridge.newave.converters.hydro import (
            convert_hydro_energy_productivity,
        )

        table = convert_hydro_energy_productivity(case, self._make_id_map())
        rows = table.to_pylist()
        assert len(rows) == 2
        for r in rows:
            assert r["stage_id"] is None
        # Both fall back to altura_65 legacy default.
        a_row = next(r for r in rows if r["hydro_id"] == 0)
        assert a_row["equivalent_productivity_mw_per_m3s"] == pytest.approx(0.9 * 318.5)

    def test_seasonal_volref_zero_month_inside_nonzero_row_uses_vmin(
        self, tmp_path: Path
    ) -> None:
        """A zero entry within a row that has some non-zero months means
        "operate at vol_min" for that month — distinct from the legacy
        fallback used for an entirely all-zero row."""
        # USINA_A: month 1 = 0 (V=vmin); month 2 = 500 (V=vmin+500); rest = 0.
        # Plant is kept because at least one month is non-zero.
        valor = [0.0] * 12
        valor[1] = 500.0  # month 2
        volref_df = pd.DataFrame(
            {
                "codigo_usina": [1] * 12,
                "nome_usina": ["USINA_A"] * 12,
                "mes": list(range(1, 13)),
                "valor": valor,
            }
        )

        case = self._base_case(
            tmp_path,
            volref_volumes=volref_df,
            ano_inicio=2025,
            mes_inicio=1,
            num_anos=1,
        )

        from novomodelo_bridge.newave.converters.hydro import (
            convert_hydro_energy_productivity,
        )

        table = convert_hydro_energy_productivity(case, self._make_id_map())
        rows = table.to_pylist()
        a_rows = sorted(
            (r for r in rows if r["hydro_id"] == 0),
            key=lambda r: r["stage_id"],
        )
        # Stage 0 = month 1 (volref=0) → V=vmin=100, h=310, drop=260, ρ=234.
        assert a_rows[0]["equivalent_productivity_mw_per_m3s"] == pytest.approx(
            0.9 * (300.0 + 0.1 * 100.0 - 50.0)
        )
        # Stage 1 = month 2 (volref=500) → V=600, h=360, drop=310, ρ=279.
        assert a_rows[1]["equivalent_productivity_mw_per_m3s"] == pytest.approx(
            0.9 * (300.0 + 0.1 * 600.0 - 50.0)
        )

    def test_seasonal_volref_combined_with_cfuga_override(self, tmp_path: Path) -> None:
        """Seasonal volref and CFUGA must compose: each stage uses the
        month's reference volume AND the active canal_fuga override."""
        # CFUGA effective from month 6 (stage 5) onward — canal_fuga 50 → 60.
        cfuga_rec = _make_cfuga_rec(month=6, year=2025, nivel=60.0)
        usina_rec = MagicMock()
        usina_rec.codigo = 1
        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [cfuga_rec]

        # Seasonal volref: every month useful=200 → V=300, h=330.
        volref_df = pd.DataFrame(
            {
                "codigo_usina": [1] * 12,
                "nome_usina": ["USINA_A"] * 12,
                "mes": list(range(1, 13)),
                "valor": [200.0] * 12,
            }
        )

        case = self._base_case(
            tmp_path,
            modif=mock_modif,
            volref_volumes=volref_df,
            ano_inicio=2025,
            mes_inicio=1,
            num_anos=1,
        )

        from novomodelo_bridge.newave.converters.hydro import (
            convert_hydro_energy_productivity,
        )

        table = convert_hydro_energy_productivity(case, self._make_id_map())
        rows = [r for r in table.to_pylist() if r["hydro_id"] == 0]
        by_stage = {
            r["stage_id"]: r["equivalent_productivity_mw_per_m3s"] for r in rows
        }
        # Stages 0..4: cf=50, V=300, h=330, drop=280, ρ=252.
        pre = 0.9 * (300.0 + 0.1 * 300.0 - 50.0)
        # Stages 5..11: cf=60, drop=270, ρ=243.
        post = 0.9 * (300.0 + 0.1 * 300.0 - 60.0)
        assert by_stage[0] == pytest.approx(pre)
        assert by_stage[4] == pytest.approx(pre)
        assert by_stage[5] == pytest.approx(post)
        assert by_stage[11] == pytest.approx(post)

    def test_other_override_columns_are_null(self, tmp_path: Path) -> None:
        """reference_outflow_m3s / ρ_esp columns remain NULL.

        The retired ``reference_volume_hm3`` column must no longer be emitted;
        V_ref now lives in ``hydro_production_models.json``.
        """
        case = self._base_case(tmp_path)

        from novomodelo_bridge.newave.converters.hydro import (
            convert_hydro_energy_productivity,
        )

        table = convert_hydro_energy_productivity(case, self._make_id_map())

        assert "reference_volume_hm3" not in table.column_names
        for col in (
            "reference_outflow_m3s",
            "specific_productivity_mw_per_m3s_per_m",
        ):
            assert all(v is None for v in table[col].to_pylist())


# ---------------------------------------------------------------------------
# _parse_fpha_plane_reduction: structured-diagnostic coverage
# ---------------------------------------------------------------------------
#
# The parsing behaviour itself (angle/distance methods, commented lines, no
# file present) is covered by TestParseFphaPlaneReduction in
# tests/newave/test_fpha_conversion.py, the established home for the FPHA converters;
# this class covers only the multiple-active-methods diagnostic.


class TestParseFphaPlaneReductionDiagnostics:
    """Structured-diagnostic coverage for the multiple-active-methods warning."""

    def _case_with_text(self, tmp_path: Path, text: str):
        path = tmp_path / "tratamento-fpha.csv"
        path.write_text(text, encoding="utf-8")
        return make_case(make_nw_files(tmp_path, tratamento_fpha=path))

    _MULTI_METHOD_TEXT = (
        "HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES-ANGULO-PADRAO; 1.0\n"
        "HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES-DISTANCIA-PADRAO; 0.002\n"
    )

    def test_multiple_methods_emits_one_diagnostic_using_the_first(
        self, tmp_path: Path
    ) -> None:
        from novomodelo_bridge.newave.converters.hydro import (
            _parse_fpha_plane_reduction,
        )

        case = self._case_with_text(tmp_path, self._MULTI_METHOD_TEXT)

        with dx.collect() as collected:
            result = _parse_fpha_plane_reduction(case)

        assert result == {"method": "angle", "tolerance_deg": 1.0}
        assert len(collected) == 1
        diag = collected[0]
        assert diag.code == "fpha-plane-reduction-multiple-methods"
        assert diag.severity is Severity.WARNING
        assert diag.category == "Production model"
        assert diag.table is None
        assert "tratamento-fpha.csv" in diag.summary
        assert "angle" in diag.summary

        _assert_no_repo_internal_leaks(collected)

    def test_single_method_emits_no_diagnostic(self, tmp_path: Path) -> None:
        from novomodelo_bridge.newave.converters.hydro import (
            _parse_fpha_plane_reduction,
        )

        case = self._case_with_text(
            tmp_path, "HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES-ANGULO-PADRAO; 1.0\n"
        )

        with dx.collect() as collected:
            _parse_fpha_plane_reduction(case)

        assert collected == []

    def test_multiple_methods_carries_no_legacy_warning(self, tmp_path: Path) -> None:
        from novomodelo_bridge.newave.converters.hydro import (
            _parse_fpha_plane_reduction,
        )

        case = self._case_with_text(tmp_path, self._MULTI_METHOD_TEXT)

        with dx.collect() as collected:
            _parse_fpha_plane_reduction(case)

        assert not any(d.code == "legacy-warning" for d in collected)

    def test_no_sink_fallback_logs_one_warning(self, tmp_path: Path, caplog) -> None:
        """With no active collect() sink, emit() degrades to a single logging
        record — the pre-migration caplog contract keeps working."""
        import logging

        from novomodelo_bridge.newave.converters.hydro import (
            _parse_fpha_plane_reduction,
        )

        case = self._case_with_text(tmp_path, self._MULTI_METHOD_TEXT)

        with caplog.at_level(logging.WARNING):
            _parse_fpha_plane_reduction(case)

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
