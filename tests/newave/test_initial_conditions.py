"""Unit tests for the source model initial-conditions converter."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from novomodelo_bridge.newave.id_map import NewaveIdMap
from tests.conftest import make_case, make_nw_files
from tests.newave.conftest import (
    _make_confhd_df,
    _make_hidr_cadastro,
    _make_ne_cadastro,
    _make_ne_confhd_df,
    _make_ne_exph_mock,
    _ne_filling_id_map,
)

# ---------------------------------------------------------------------------
# Initial conditions conversion
# ---------------------------------------------------------------------------


class TestConvertInitialConditions:
    def _make_id_map(self) -> NewaveIdMap:
        return NewaveIdMap(
            subsystem_ids=[1],
            hydro_codes=[1, 2],
            thermal_codes=[],
        )

    def test_returns_storage_and_filling_storage(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.initial_conditions import (
            convert_initial_conditions,
        )

        result = convert_initial_conditions(_ic_case(tmp_path), self._make_id_map())
        assert "storage" in result
        assert "filling_storage" in result

    def test_storage_values_converted_from_percentage(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.initial_conditions import (
            convert_initial_conditions,
        )

        result = convert_initial_conditions(_ic_case(tmp_path), self._make_id_map())
        # New formula: (pct / 100) * (vol_max - vol_min) + vol_min
        # USINA_A: pct=50%, vol_min=100, vol_max=1000
        #   -> (0.50) * (1000 - 100) + 100 = 450 + 100 = 550 hm3.
        # USINA_B: pct=75%, vol_min=50, vol_max=500
        #   -> (0.75) * (500 - 50) + 50 = 337.5 + 50 = 387.5 hm3.
        storage = {s["hydro_id"]: s["value_hm3"] for s in result["storage"]}
        assert storage[0] == pytest.approx(550.0)
        assert storage[1] == pytest.approx(387.5)

    def test_storage_sorted_by_hydro_id(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.initial_conditions import (
            convert_initial_conditions,
        )

        result = convert_initial_conditions(_ic_case(tmp_path), self._make_id_map())
        ids = [s["hydro_id"] for s in result["storage"]]
        assert ids == sorted(ids)

    def test_out_of_range_percentage_clamped(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.initial_conditions import (
            convert_initial_conditions,
        )

        # Should not raise; pct is clamped to 100.
        result = convert_initial_conditions(
            _ic_case(tmp_path, pct_b=120.0), self._make_id_map()
        )
        storage = {s["hydro_id"]: s["value_hm3"] for s in result["storage"]}
        # pct clamped to 100 -> vol_max=500 -> 500.0 hm3.
        assert storage[1] == pytest.approx(500.0)

    def test_filling_storage_is_empty(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.initial_conditions import (
            convert_initial_conditions,
        )

        result = convert_initial_conditions(_ic_case(tmp_path), self._make_id_map())
        assert result["filling_storage"] == []

    def test_storage_uses_volmin_adjusted_min(self, tmp_path) -> None:
        """Initial-% is of the operational useful (vol_max − VOLMIN), not raw.

        A modif.dat VOLMIN override raises the operational minimum; the source model
        takes ``volume_inicial_percentual`` of the VOLMIN-adjusted useful range
        (verified against pmo.dat "VOLUME ARMAZENADO INICIAL" for I. SOLTEIRA). The
        initial storage must use the same min the bounds converter uses. Regression for
        the I. Solteira initial-storage bug.
        """
        from novomodelo_bridge.newave.converters.initial_conditions import (
            convert_initial_conditions,
        )

        # USINA_A (code 1): raw vol_min=100, vol_max=1000, pct=50%. A VOLMIN=400
        # override makes the operational useful 1000−400 = 600.
        volmin_rec = MagicMock()
        type(volmin_rec).__name__ = "VOLMIN"
        volmin_rec.volume = 400.0
        usina_rec = MagicMock()
        usina_rec.codigo = 1
        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [volmin_rec]

        mock_hidr = MagicMock()
        mock_hidr.cadastro = _make_hidr_cadastro()
        mock_confhd = MagicMock()
        mock_confhd.usinas = _make_confhd_df()
        case = make_case(tmp_path, hidr=mock_hidr, confhd=mock_confhd, modif=mock_modif)

        result = convert_initial_conditions(case, self._make_id_map())
        storage = {s["hydro_id"]: s["value_hm3"] for s in result["storage"]}
        # operational base: 0.50 * (1000 − 400) + 400 = 700 (NOT the raw-min 550).
        assert storage[0] == pytest.approx(700.0)

    def test_run_of_river_S_initial_anchored_to_volref(self, tmp_path) -> None:
        """``tipo_regulacao='S'`` initial storage is pinned to volume_referencia.

        The bounds converter collapses 'S' (fio-d'água) storage to
        ``volume_referencia``; the initial condition must match so it stays inside
        the collapsed [min,max] range. The ``volume_inicial_percentual`` (50% here)
        is ignored for 'S' plants, and the anchor is the reference volume, not
        Vmin.
        """
        from novomodelo_bridge.newave.converters.initial_conditions import (
            convert_initial_conditions,
        )

        # USINA_A: Vmin 100, Vmax 1000. Use a Vref (700) distinct from both Vmin
        # and the percentual formula result (0.50*(1000-100)+100 = 550) so the
        # assertion pins down volume_referencia specifically.
        cadastro = _make_hidr_cadastro()
        cadastro.loc[1, "tipo_regulacao"] = "S"
        cadastro.loc[1, "volume_referencia"] = 700.0
        mock_hidr = MagicMock()
        mock_hidr.cadastro = cadastro
        mock_confhd = MagicMock()
        mock_confhd.usinas = _make_confhd_df()
        case = make_case(tmp_path, hidr=mock_hidr, confhd=mock_confhd)

        result = convert_initial_conditions(case, self._make_id_map())
        storage = {s["hydro_id"]: s["value_hm3"] for s in result["storage"]}
        # 'S' plant anchored to volume_referencia (700), NOT Vmin (100) nor the
        # percentual 550.
        assert storage[0] == pytest.approx(700.0)

    def _ne_filling_ic_case(self, tmp_path, *, volume_morto: float = 0.0):
        """A ``NewaveCase`` with JURUENA (NE+filling) for IC routing tests.

        Two EX plants (codes 1, 2) plus JURUENA (code 309, ``NE``,
        ``volume_minimo == 2.93``) admitted via its exph filling row.
        """
        return make_case(
            tmp_path,
            hidr=MagicMock(cadastro=_make_ne_cadastro()),
            confhd=MagicMock(usinas=_make_ne_confhd_df()),
            exph=_make_ne_exph_mock(volume_morto=volume_morto),
        )

    def test_ne_plant_routed_to_filling_storage(self, tmp_path) -> None:
        """A filling ``NE`` plant is seeded into ``filling_storage``, not ``storage``.

        JURUENA (code 309 → novomodelo id 2) has ``volume_morto == 0`` and
        ``volume_minimo == 2.93``, so its seed is ``0.00 × 2.93 == 0.0``.
        """
        from novomodelo_bridge.newave.converters.initial_conditions import (
            convert_initial_conditions,
        )

        result = convert_initial_conditions(
            self._ne_filling_ic_case(tmp_path), _ne_filling_id_map()
        )
        assert result["filling_storage"] == [{"hydro_id": 2, "value_hm3": 0.0}]

    def test_ne_plant_excluded_from_storage(self, tmp_path) -> None:
        """The filling plant must NOT appear in ``storage`` (novomodelo rejects in-both).

        The two EX plants (novomodelo ids 0, 1) stay in ``storage``; JURUENA (id 2) is
        absent from it.
        """
        from novomodelo_bridge.newave.converters.initial_conditions import (
            convert_initial_conditions,
        )

        result = convert_initial_conditions(
            self._ne_filling_ic_case(tmp_path), _ne_filling_id_map()
        )
        storage_ids = {s["hydro_id"] for s in result["storage"]}
        assert 2 not in storage_ids
        assert {0, 1} <= storage_ids

    def test_filling_seed_scales_with_volume_morto(self, tmp_path) -> None:
        """The seed is ``(volume_morto / 100) × volume_minimo``.

        With a synthetic ``volume_morto == 50`` and ``volume_minimo == 2.93``,
        the seed is ``0.50 × 2.93 == 1.465`` hm³.
        """
        from novomodelo_bridge.newave.converters.initial_conditions import (
            convert_initial_conditions,
        )

        result = convert_initial_conditions(
            self._ne_filling_ic_case(tmp_path, volume_morto=50.0),
            _ne_filling_id_map(),
        )
        seed = {s["hydro_id"]: s["value_hm3"] for s in result["filling_storage"]}
        assert seed[2] == pytest.approx(1.465)

    def test_volume_morto_out_of_range_clamped(self, tmp_path, caplog) -> None:
        """A ``volume_morto`` above 100 is clamped to 100% with a warning.

        Synthetic ``volume_morto == 150`` clamps to 100%, giving the full
        ``volume_minimo == 2.93`` hm³ seed, and logs a clamp warning.
        """
        from novomodelo_bridge.newave.converters.initial_conditions import (
            convert_initial_conditions,
        )

        with caplog.at_level(logging.WARNING):
            result = convert_initial_conditions(
                self._ne_filling_ic_case(tmp_path, volume_morto=150.0),
                _ne_filling_id_map(),
            )
        seed = {s["hydro_id"]: s["value_hm3"] for s in result["filling_storage"]}
        assert seed[2] == pytest.approx(2.93)
        assert any("volume_morto" in rec.message for rec in caplog.records)


def _ic_case(tmp_path, pct_b: float = 75.0):
    """Build a NewaveCase with hidr/confhd readers pre-cached for IC tests."""
    mock_hidr = MagicMock()
    mock_hidr.cadastro = _make_hidr_cadastro()

    df = _make_confhd_df().copy()
    df.loc[df["codigo_usina"] == 2, "volume_inicial_percentual"] = pct_b
    mock_confhd = MagicMock()
    mock_confhd.usinas = df

    # A dger with a concrete study start so the windowed anticipated-commitment
    # dates resolve (study stage k -> month (2024-01) + k).
    mock_dger = MagicMock()
    mock_dger.ano_inicio_estudo = 2024
    mock_dger.mes_inicio_estudo = 1

    return make_case(tmp_path, hidr=mock_hidr, confhd=mock_confhd, dger=mock_dger)


class TestAnticipatedCommitmentSeeding:
    """``convert_initial_conditions`` writes real adterm MW, clamped to bounds.

    Novomodelo (>= 0.7.0) honours non-zero pre-horizon seeds, so the committed MW is
    passed through (no longer zeroed); only out-of-bounds values are clamped.
    """

    def _id_map(self) -> NewaveIdMap:
        return NewaveIdMap(
            subsystem_ids=[1],
            hydro_codes=[1, 2],
            thermal_codes=[86],
        )

    @patch(
        "novomodelo_bridge.newave.converters.initial_conditions.thermal_generation_bounds"
    )
    @patch(
        "novomodelo_bridge.newave.converters.initial_conditions.read_anticipated_dispatch"
    )
    def test_in_range_values_pass_through(
        self, mock_read, mock_bounds, tmp_path, caplog
    ) -> None:
        from novomodelo_bridge.newave.converters.anticipated import AnticipatedDispatch
        from novomodelo_bridge.newave.converters.initial_conditions import (
            convert_initial_conditions,
        )

        mock_read.return_value = {
            86: AnticipatedDispatch(lead_stages=2, values_mw=[204.5647, 0.0])
        }
        mock_bounds.return_value = {86: (0.0, 481.27)}

        with caplog.at_level(
            logging.WARNING,
            logger="novomodelo_bridge.newave.converters.initial_conditions",
        ):
            result = convert_initial_conditions(_ic_case(tmp_path), self._id_map())

        # Windowed records (novomodelo 0.14): one contiguous monthly window per
        # leading delivery stage, zero-MW stages written explicitly.
        assert result["past_anticipated_commitments"] == [
            {
                "thermal_id": 0,
                "start_date": "2024-01-01",
                "end_date": "2024-02-01",
                "value_mw": 204.5647,
            },
            {
                "thermal_id": 0,
                "start_date": "2024-02-01",
                "end_date": "2024-03-01",
                "value_mw": 0.0,
            },
        ]
        assert "clamping" not in caplog.text

    @patch(
        "novomodelo_bridge.newave.converters.initial_conditions.thermal_generation_bounds"
    )
    @patch(
        "novomodelo_bridge.newave.converters.initial_conditions.read_anticipated_dispatch"
    )
    def test_out_of_range_values_clamped_and_warned(
        self, mock_read, mock_bounds, tmp_path, caplog
    ) -> None:
        from novomodelo_bridge.newave.converters.anticipated import AnticipatedDispatch
        from novomodelo_bridge.newave.converters.initial_conditions import (
            convert_initial_conditions,
        )

        # 600 > max 481.27 -> clamp to 481.27; -5 < min 0 -> clamp to 0.
        mock_read.return_value = {
            86: AnticipatedDispatch(lead_stages=2, values_mw=[600.0, -5.0])
        }
        mock_bounds.return_value = {86: (0.0, 481.27)}

        with caplog.at_level(
            logging.WARNING,
            logger="novomodelo_bridge.newave.converters.initial_conditions",
        ):
            result = convert_initial_conditions(_ic_case(tmp_path), self._id_map())

        commitments = result["past_anticipated_commitments"]
        assert [c["value_mw"] for c in commitments] == pytest.approx([481.27, 0.0])
        assert commitments[0]["start_date"] == "2024-01-01"
        assert commitments[0]["end_date"] == "2024-02-01"
        assert "code=86" in caplog.text
        assert "clamping" in caplog.text

    @patch(
        "novomodelo_bridge.newave.converters.initial_conditions.thermal_generation_bounds"
    )
    @patch(
        "novomodelo_bridge.newave.converters.initial_conditions.read_anticipated_dispatch"
    )
    def test_code_absent_from_id_map_skipped(
        self, mock_read, mock_bounds, tmp_path, caplog
    ) -> None:
        from novomodelo_bridge.newave.converters.anticipated import AnticipatedDispatch
        from novomodelo_bridge.newave.converters.initial_conditions import (
            convert_initial_conditions,
        )

        mock_read.return_value = {
            999: AnticipatedDispatch(lead_stages=1, values_mw=[100.0])
        }
        mock_bounds.return_value = {999: (0.0, 500.0)}

        with caplog.at_level(
            logging.WARNING,
            logger="novomodelo_bridge.newave.converters.initial_conditions",
        ):
            result = convert_initial_conditions(_ic_case(tmp_path), self._id_map())

        # Unknown thermal -> skipped, so no commitments key is emitted.
        assert "past_anticipated_commitments" not in result
        assert "absent from" in caplog.text

    @patch(
        "novomodelo_bridge.newave.converters.initial_conditions.thermal_generation_bounds"
    )
    @patch(
        "novomodelo_bridge.newave.converters.initial_conditions.read_anticipated_dispatch"
    )
    def test_non_gnl_case_skips_bounds_computation(
        self, mock_read, mock_bounds, tmp_path
    ) -> None:
        from novomodelo_bridge.newave.converters.initial_conditions import (
            convert_initial_conditions,
        )

        mock_read.return_value = {}

        result = convert_initial_conditions(_ic_case(tmp_path), self._id_map())

        assert "past_anticipated_commitments" not in result
        mock_bounds.assert_not_called()

    def test_delivery_window_year_and_december_wrap(self) -> None:
        """The windowed-commitment dates wrap the year at December correctly."""
        from novomodelo_bridge.newave.converters.initial_conditions import (
            _delivery_window,
        )

        # Study starts Nov 2024.
        assert _delivery_window(2024, 11, 0) == ("2024-11-01", "2024-12-01")
        # k=1 is December -> exclusive end rolls into the next year.
        assert _delivery_window(2024, 11, 1) == ("2024-12-01", "2025-01-01")
        # k=2 crosses into the new year.
        assert _delivery_window(2024, 11, 2) == ("2025-01-01", "2025-02-01")
        # A multi-year lead stays aligned.
        assert _delivery_window(2024, 11, 14) == ("2026-01-01", "2026-02-01")


class TestReadAnticipatedDispatchHorizonTruncation:
    """``read_anticipated_dispatch`` has no post-horizon surface.

    Evidence for the twin-track decision recorded in
    ``decomp/anticipated.py``: adterm.dat commits dispatch only through the
    study horizon, so a lag beyond it is truncated and warned, never carried
    past the horizon as a post-study surface (unlike the DECOMP GNL model's
    já-comandada/signaled post-horizon windows).
    """

    def test_lag_beyond_horizon_truncated_and_warned(self, tmp_path, caplog) -> None:
        from novomodelo_bridge.newave.converters.anticipated import (
            read_anticipated_dispatch,
        )

        adterm_path = tmp_path / "adterm.dat"
        adterm_path.touch()
        files = make_nw_files(tmp_path, adterm=adterm_path)

        mock_dger = MagicMock()
        mock_dger.despacho_antecipado_gnl = 1
        mock_dger.ano_inicio_estudo = 2024
        mock_dger.mes_inicio_estudo = 1
        mock_dger.num_anos_estudo = 1
        mock_dger.num_anos_pos_estudo = 0  # 12-stage horizon, no post-study tail

        mock_patamar = MagicMock()
        mock_patamar.numero_patamares = 1
        mock_patamar.duracao_mensal_patamares = None

        mock_adterm = MagicMock()
        mock_adterm.despachos = pd.DataFrame(
            {
                "codigo_usina": [86, 86],
                "lag": [1, 15],
                "patamar": [1, 1],
                "valor": [50.0, 999.0],
            }
        )

        case = make_case(
            files, dger=mock_dger, patamar=mock_patamar, adterm=mock_adterm
        )

        with caplog.at_level(
            logging.WARNING, logger="novomodelo_bridge.newave.converters.anticipated"
        ):
            result = read_anticipated_dispatch(case)

        # lag=15 exceeds the 12-stage horizon -> truncated to it; the source
        # model surfaces nothing beyond the study horizon.
        assert result[86].lead_stages == 12
        assert len(result[86].values_mw) == 12
        assert result[86].values_mw[0] == pytest.approx(50.0)
        assert "lag=15" in caplog.text
        assert "truncating to 12" in caplog.text
