"""Unit tests for the source model thermal converter."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from novomodelo_bridge.newave.id_map import NewaveIdMap
from tests.conftest import make_case, make_nw_files
from tests.newave.conftest import (
    _make_thermal_dger,
    _thermal_readers,
)

# ---------------------------------------------------------------------------
# Thermal conversion
# ---------------------------------------------------------------------------


class TestConvertThermals:
    def _make_id_map(self) -> NewaveIdMap:
        return NewaveIdMap(
            subsystem_ids=[1, 2],
            hydro_codes=[],
            thermal_codes=[10, 20, 30],
        )

    def test_returns_thermals_key(self, tmp_path) -> None:
        conft, clast, term = _thermal_readers()
        case = make_case(
            tmp_path, conft=conft, clast=clast, term=term, dger=_make_thermal_dger()
        )
        from novomodelo_bridge.newave.converters.thermal import convert_thermals

        result = convert_thermals(case, self._make_id_map())
        assert "thermals" in result

    def test_thermal_count(self, tmp_path) -> None:
        conft, clast, term = _thermal_readers()
        case = make_case(
            tmp_path, conft=conft, clast=clast, term=term, dger=_make_thermal_dger()
        )
        from novomodelo_bridge.newave.converters.thermal import convert_thermals

        result = convert_thermals(case, self._make_id_map())
        assert len(result["thermals"]) == 3

    def test_thermal_ids_are_zero_based_sorted(self, tmp_path) -> None:
        conft, clast, term = _thermal_readers()
        case = make_case(
            tmp_path, conft=conft, clast=clast, term=term, dger=_make_thermal_dger()
        )
        from novomodelo_bridge.newave.converters.thermal import convert_thermals

        result = convert_thermals(case, self._make_id_map())
        ids = [t["id"] for t in result["thermals"]]
        assert ids == sorted(ids)
        assert ids[0] == 0

    def test_cost_per_mwh_scalar(self, tmp_path) -> None:
        conft, clast, term = _thermal_readers()
        case = make_case(
            tmp_path, conft=conft, clast=clast, term=term, dger=_make_thermal_dger()
        )
        from novomodelo_bridge.newave.converters.thermal import convert_thermals

        result = convert_thermals(case, self._make_id_map())
        for t in result["thermals"]:
            assert "cost_per_mwh" in t
            assert isinstance(t["cost_per_mwh"], float)
            assert "cost_segments" not in t
            assert "generation" in t
            assert "min_mw" in t["generation"]
            assert "max_mw" in t["generation"]

    def test_bus_id_assignment(self, tmp_path) -> None:
        conft, clast, term = _thermal_readers()
        case = make_case(
            tmp_path, conft=conft, clast=clast, term=term, dger=_make_thermal_dger()
        )
        from novomodelo_bridge.newave.converters.thermal import convert_thermals

        result = convert_thermals(case, self._make_id_map())
        # TERMO_A (code 10) and TERMO_B (code 20) are in submercado 1 -> bus 0.
        # TERMO_C (code 30) is in submercado 2 -> bus 1.
        termo_a = next(t for t in result["thermals"] if t["name"] == "TERMO_A")
        termo_c = next(t for t in result["thermals"] if t["name"] == "TERMO_C")
        assert termo_a["bus_id"] == 0
        assert termo_c["bus_id"] == 1

    def test_capacity_uses_factor(self, tmp_path) -> None:
        conft, clast, term = _thermal_readers()
        case = make_case(
            tmp_path, conft=conft, clast=clast, term=term, dger=_make_thermal_dger()
        )
        from novomodelo_bridge.newave.converters.thermal import convert_thermals

        result = convert_thermals(case, self._make_id_map())
        # TERMO_A: potencia=100, factor=0.9, teif=0.05% -> max_mw=89.955.
        termo_a = next(t for t in result["thermals"] if t["name"] == "TERMO_A")
        assert termo_a["generation"]["max_mw"] == pytest.approx(89.955)


class TestConvertThermalBoundsClastModificacoes:
    """Per-stage cost overrides from the modificacoes block in clast.dat."""

    def _make_id_map(self) -> NewaveIdMap:
        return NewaveIdMap(
            subsystem_ids=[1, 2],
            hydro_codes=[],
            thermal_codes=[10, 20, 30],
        )

    def _make_dger(self) -> MagicMock:
        dger = MagicMock()
        dger.mes_inicio_estudo = 1
        dger.ano_inicio_estudo = 2023
        dger.num_anos_estudo = 1
        dger.num_anos_pos_estudo = 0
        dger.num_anos_manutencao_utes = 0
        return dger

    def test_modificacao_overrides_year_indexed_cost_inside_window(
        self, tmp_path
    ) -> None:
        import datetime

        conft, clast, term = _thermal_readers()

        # Override TERMO_A (code 10) cost from 50.0 -> 77.0 for stages 2-4
        # of a 12-stage 2023 horizon (March-May). Other stages keep 50.0.
        modif_df = pd.DataFrame(
            {
                "codigo_usina": [10],
                "nome_usina": ["TERMO_A"],
                "data_inicio": [datetime.datetime(2023, 3, 1)],
                "data_fim": [datetime.datetime(2023, 5, 1)],
                "custo": [77.0],
            }
        )
        clast.modificacoes = modif_df
        case = make_case(
            tmp_path,
            conft=conft,
            clast=clast,
            term=term,
            dger=self._make_dger(),
        )

        from novomodelo_bridge.newave.converters.thermal import convert_thermal_bounds

        table = convert_thermal_bounds(case, self._make_id_map())
        assert table is not None

        df = table.to_pandas()
        termo_a_id = self._make_id_map().thermal_id(10)
        a_rows = df[df["thermal_id"] == termo_a_id].sort_values("stage_id")
        # 12 stages emitted for the cost-varying plant.
        assert len(a_rows) == 12
        # Inside the modification window (stages 2, 3, 4 -> Mar, Apr, May).
        assert a_rows.iloc[2]["cost_per_mwh"] == pytest.approx(77.0)
        assert a_rows.iloc[3]["cost_per_mwh"] == pytest.approx(77.0)
        assert a_rows.iloc[4]["cost_per_mwh"] == pytest.approx(77.0)
        # Outside the window the year-1 base cost is restored.
        assert a_rows.iloc[0]["cost_per_mwh"] == pytest.approx(50.0)
        assert a_rows.iloc[5]["cost_per_mwh"] == pytest.approx(50.0)
        assert a_rows.iloc[11]["cost_per_mwh"] == pytest.approx(50.0)
        # Plants without a modificacao (and uniform year cost) emit no
        # per-stage cost override — cost_per_mwh is left null.
        termo_b_id = self._make_id_map().thermal_id(20)
        b_rows = df[df["thermal_id"] == termo_b_id]
        assert b_rows["cost_per_mwh"].isna().all()

    def test_chained_potef_finite_then_open_keeps_plant_alive(self, tmp_path) -> None:
        """A finite POTEF window followed by an open-ended one keeps the plant
        alive across both: the first window's end is not a decommission date."""
        import datetime

        conft, clast, term = _thermal_readers()

        # POTEF window 1: stages 0-3 (Jan-Apr 2023) at 100 MW.
        # POTEF window 2: stage 4 onwards (May 2023+) at 200 MW.
        expt_df = pd.DataFrame(
            {
                "codigo_usina": [10, 10],
                "tipo": ["POTEF", "POTEF"],
                "modificacao": [100.0, 200.0],
                "data_inicio": [
                    datetime.datetime(2023, 1, 1),
                    datetime.datetime(2023, 5, 1),
                ],
                "data_fim": [datetime.datetime(2023, 4, 1), pd.NaT],
            }
        )
        expt_obj = MagicMock()
        expt_obj.expansoes = expt_df

        # Use a real expt file path so the optional source is wired in.
        nw = make_nw_files(tmp_path, expt=tmp_path / "expt.dat")
        case = make_case(
            nw,
            conft=conft,
            clast=clast,
            term=term,
            dger=self._make_dger(),
            expt=expt_obj,
        )

        from novomodelo_bridge.newave.converters.thermal import convert_thermal_bounds

        table = convert_thermal_bounds(case, self._make_id_map())
        assert table is not None
        df = table.to_pandas()
        termo_a_id = self._make_id_map().thermal_id(10)
        a_rows = df[df["thermal_id"] == termo_a_id].sort_values("stage_id")

        # FCMAX=90, TEIF=0.05% (fixture stores TEIF in percent units, applied
        # as (100-teif)/100), IP zeroed by step 1.
        # Window 1: max = 100 * 0.9 * (1 - 0.0005) = 89.955
        # Window 2: max = 200 * 0.9 * (1 - 0.0005) = 179.910
        assert a_rows.iloc[0]["max_generation_mw"] == pytest.approx(89.955)
        assert a_rows.iloc[3]["max_generation_mw"] == pytest.approx(89.955)
        assert a_rows.iloc[4]["max_generation_mw"] == pytest.approx(179.910)
        assert a_rows.iloc[11]["max_generation_mw"] == pytest.approx(179.910)

    def test_potef_window_gap_decommissions_plant(self, tmp_path) -> None:
        """A gap between two finite POTEF windows truly decommissions the
        plant — capacity goes to zero for stages outside any window."""
        import datetime

        conft, clast, term = _thermal_readers()
        # Only a plant CONFT marks EE/NE has its registry capacity and
        # minimum discarded, which is what makes the EXPT schedule the
        # whole timeline.
        conft.usinas.loc[conft.usinas["codigo_usina"] == 10, "usina_existente"] = "EE"

        expt_df = pd.DataFrame(
            {
                "codigo_usina": [10, 10],
                "tipo": ["POTEF", "POTEF"],
                "modificacao": [100.0, 200.0],
                "data_inicio": [
                    datetime.datetime(2023, 1, 1),
                    datetime.datetime(2023, 8, 1),
                ],
                "data_fim": [
                    datetime.datetime(2023, 3, 1),
                    datetime.datetime(2023, 10, 1),
                ],
            }
        )
        expt_obj = MagicMock()
        expt_obj.expansoes = expt_df

        nw = make_nw_files(tmp_path, expt=tmp_path / "expt.dat")
        case = make_case(
            nw,
            conft=conft,
            clast=clast,
            term=term,
            dger=self._make_dger(),
            expt=expt_obj,
        )

        from novomodelo_bridge.newave.converters.thermal import convert_thermal_bounds

        table = convert_thermal_bounds(case, self._make_id_map())
        assert table is not None
        df = table.to_pandas()
        termo_a_id = self._make_id_map().thermal_id(10)
        a_rows = df[df["thermal_id"] == termo_a_id].sort_values("stage_id")
        # Stage 3 (Apr) and Stage 6 (Jul) sit in the gap → zeroed.
        assert a_rows.iloc[3]["max_generation_mw"] == pytest.approx(0.0)
        assert a_rows.iloc[6]["max_generation_mw"] == pytest.approx(0.0)
        # Stage 0 (Jan) in window 1 → 89.955; stage 7 (Aug) in window 2 → 179.910.
        assert a_rows.iloc[0]["max_generation_mw"] == pytest.approx(89.955)
        assert a_rows.iloc[7]["max_generation_mw"] == pytest.approx(179.910)
        # Stage 11 (Dec) past window 2 → zeroed.
        assert a_rows.iloc[11]["max_generation_mw"] == pytest.approx(0.0)

    def test_ex_plant_keeps_its_registry_capacity_outside_the_windows(
        self, tmp_path
    ) -> None:
        """An ``EX`` plant's TERM.DAT capacity survives where EXPT declares none.

        The registry capacity is discarded only for ``EE``/``NE``, so a POTEF
        schedule on an ``EX`` plant modifies it inside the window instead of
        defining the only period it exists.
        """
        import datetime

        conft, clast, term = _thermal_readers()
        # The fixture's plants are EX; left as is on purpose.
        expt_df = pd.DataFrame(
            {
                "codigo_usina": [10],
                "tipo": ["POTEF"],
                "modificacao": [200.0],
                "data_inicio": [datetime.datetime(2023, 1, 1)],
                "data_fim": [datetime.datetime(2023, 3, 1)],
            }
        )
        expt_obj = MagicMock()
        expt_obj.expansoes = expt_df

        nw = make_nw_files(tmp_path, expt=tmp_path / "expt.dat")
        case = make_case(
            nw,
            conft=conft,
            clast=clast,
            term=term,
            dger=self._make_dger(),
            expt=expt_obj,
        )

        from novomodelo_bridge.newave.converters.thermal import convert_thermal_bounds

        table = convert_thermal_bounds(case, self._make_id_map())
        assert table is not None
        a_rows = (
            table.to_pandas()
            .query("thermal_id == @self._make_id_map().thermal_id(10)")
            .sort_values("stage_id")
        )
        # Inside the window the POTEF value applies: 200 * 90% * (100 - 0.05)%.
        assert a_rows.iloc[0]["max_generation_mw"] == pytest.approx(179.910)
        # Outside it the registry capacity stands, instead of being zeroed.
        assert a_rows.iloc[6]["max_generation_mw"] == pytest.approx(89.955)
        assert a_rows.iloc[11]["max_generation_mw"] == pytest.approx(89.955)

    def test_modificacao_with_open_end_extends_to_horizon(self, tmp_path) -> None:
        import datetime

        conft, clast, term = _thermal_readers()

        modif_df = pd.DataFrame(
            {
                "codigo_usina": [20],
                "nome_usina": ["TERMO_B"],
                "data_inicio": [datetime.datetime(2023, 7, 1)],
                "data_fim": [pd.NaT],
                "custo": [120.0],
            }
        )
        clast.modificacoes = modif_df
        case = make_case(
            tmp_path,
            conft=conft,
            clast=clast,
            term=term,
            dger=self._make_dger(),
        )

        from novomodelo_bridge.newave.converters.thermal import convert_thermal_bounds

        table = convert_thermal_bounds(case, self._make_id_map())
        assert table is not None

        df = table.to_pandas().sort_values(["thermal_id", "stage_id"])
        termo_b_id = self._make_id_map().thermal_id(20)
        b_rows = df[df["thermal_id"] == termo_b_id].sort_values("stage_id")
        # Stage 5 = Jun 2023 (outside the window) keeps the base 80.0.
        # Stage 6 = Jul 2023 onwards picks up the open-ended override.
        assert b_rows.iloc[5]["cost_per_mwh"] == pytest.approx(80.0)
        assert b_rows.iloc[6]["cost_per_mwh"] == pytest.approx(120.0)
        assert b_rows.iloc[11]["cost_per_mwh"] == pytest.approx(120.0)

    def test_gtmin_availability_freezes_post_study_tail(self, tmp_path) -> None:
        """The "período estático final" freezes thermal min generation at December
        of the last study year: a seasonal GTMIN window must NOT keep cycling its
        on/off pattern through the post-study tail."""
        import datetime

        conft, clast, term = _thermal_readers()
        conft.usinas.loc[conft.usinas["codigo_usina"] == 10, "usina_existente"] = "EE"

        # POTEF gives capacity across the whole horizon. GTMIN is active only
        # Jan-Apr and Sep-Dec 2023 (zero May-Aug). December — the freeze point —
        # is active, so the entire post-study tail must hold GTMIN rather than
        # re-dropping it in the post-study May-Aug months.
        expt_df = pd.DataFrame(
            {
                "codigo_usina": [10, 10, 10],
                "tipo": ["POTEF", "GTMIN", "GTMIN"],
                "modificacao": [100.0, 30.0, 30.0],
                "data_inicio": [
                    datetime.datetime(2023, 1, 1),
                    datetime.datetime(2023, 1, 1),
                    datetime.datetime(2023, 9, 1),
                ],
                "data_fim": [
                    pd.NaT,
                    datetime.datetime(2023, 4, 1),
                    datetime.datetime(2023, 12, 1),
                ],
            }
        )
        expt_obj = MagicMock()
        expt_obj.expansoes = expt_df

        dger = self._make_dger()
        dger.num_anos_pos_estudo = 1  # 12 study + 12 post-study stages

        nw = make_nw_files(tmp_path, expt=tmp_path / "expt.dat")
        case = make_case(
            nw, conft=conft, clast=clast, term=term, dger=dger, expt=expt_obj
        )

        from novomodelo_bridge.newave.converters.thermal import convert_thermal_bounds

        table = convert_thermal_bounds(case, self._make_id_map())
        assert table is not None
        a_rows = (
            table.to_pandas()
            .query("thermal_id == @self._make_id_map().thermal_id(10)")
            .sort_values("stage_id")
            .reset_index(drop=True)
        )
        assert len(a_rows) == 24  # 12 study + 12 post-study

        # In-study: active Jan-Apr (3) and Sep-Dec (11); zero May-Aug (6).
        assert a_rows.loc[3, "min_generation_mw"] == pytest.approx(30.0)
        assert a_rows.loc[6, "min_generation_mw"] == pytest.approx(0.0)
        assert a_rows.loc[11, "min_generation_mw"] == pytest.approx(30.0)
        # Post-study (12-23): frozen at December → 30 every month, including the
        # May (16) and Aug (19) 2024 stages outside the 2023 windows' months.
        assert a_rows.loc[16, "min_generation_mw"] == pytest.approx(30.0)
        assert a_rows.loc[19, "min_generation_mw"] == pytest.approx(30.0)
        assert a_rows.loc[12:23, "min_generation_mw"].tolist() == pytest.approx(
            [30.0] * 12
        )

    def test_cost_modificacao_does_not_leak_into_post_study(self, tmp_path) -> None:
        """A clast cost change dated inside the post-study tail must not apply —
        the static final period freezes cost at December of the last study year."""
        import datetime

        conft, clast, term = _thermal_readers()

        # Future cost change starting Mar 2024 (a post-study stage). Frozen at
        # December 2023, it must never take effect.
        clast.modificacoes = pd.DataFrame(
            {
                "codigo_usina": [10],
                "nome_usina": ["TERMO_A"],
                "data_inicio": [datetime.datetime(2024, 3, 1)],
                "data_fim": [pd.NaT],
                "custo": [999.0],
            }
        )
        dger = self._make_dger()
        dger.num_anos_pos_estudo = 1

        case = make_case(tmp_path, conft=conft, clast=clast, term=term, dger=dger)

        from novomodelo_bridge.newave.converters.thermal import convert_thermal_bounds

        table = convert_thermal_bounds(case, self._make_id_map())
        assert table is not None
        a_rows = (
            table.to_pandas()
            .query("thermal_id == @self._make_id_map().thermal_id(10)")
            .sort_values("stage_id")
            .reset_index(drop=True)
        )
        # The year-1 base cost (50.0) holds through the whole post-study tail;
        # the future 999.0 modification is frozen out.
        assert a_rows.loc[11, "cost_per_mwh"] == pytest.approx(50.0)  # Dec 2023
        assert a_rows.loc[14, "cost_per_mwh"] == pytest.approx(50.0)  # Mar 2024
        assert a_rows.loc[12:23, "cost_per_mwh"].tolist() == pytest.approx([50.0] * 12)

    def test_post_study_expansion_freezes_at_online_value(self, tmp_path) -> None:
        """A plant that comes online only in the post-study (POTEF dated in the first
        post-study month) freezes at its *online* terminal December value, not at the
        last study stage (where it does not yet exist) and not at its seasonal
        profile."""
        import datetime

        conft, clast, term = _thermal_readers()
        conft.usinas.loc[conft.usinas["codigo_usina"] == 10, "usina_existente"] = "EE"

        # Plant exists only from 2024 (the post-study). GTMIN: a closed seasonal window
        # Jan-May (30) plus an open-ended tail from Jun (80). The source model freezes
        # the whole tail at the terminal December value (the open tail, 80).
        expt_df = pd.DataFrame(
            {
                "codigo_usina": [10, 10, 10],
                "tipo": ["POTEF", "GTMIN", "GTMIN"],
                "modificacao": [100.0, 30.0, 80.0],
                "data_inicio": [
                    datetime.datetime(2024, 1, 1),
                    datetime.datetime(2024, 1, 1),
                    datetime.datetime(2024, 6, 1),
                ],
                "data_fim": [pd.NaT, datetime.datetime(2024, 5, 1), pd.NaT],
            }
        )
        expt_obj = MagicMock()
        expt_obj.expansoes = expt_df

        dger = self._make_dger()
        dger.num_anos_pos_estudo = 1  # study Jan-Dec 2023, post Jan-Dec 2024

        nw = make_nw_files(tmp_path, expt=tmp_path / "expt.dat")
        case = make_case(
            nw, conft=conft, clast=clast, term=term, dger=dger, expt=expt_obj
        )

        from novomodelo_bridge.newave.converters.thermal import convert_thermal_bounds

        table = convert_thermal_bounds(case, self._make_id_map())
        assert table is not None
        a = (
            table.to_pandas()
            .query("thermal_id == @self._make_id_map().thermal_id(10)")
            .sort_values("stage_id")
            .reset_index(drop=True)
        )
        assert len(a) == 24
        # Study (0-11): plant offline → zero capacity and zero must-run.
        assert a.loc[0, "max_generation_mw"] == pytest.approx(0.0)
        assert a.loc[11, "min_generation_mw"] == pytest.approx(0.0)
        # Post-study (12-23): frozen at the terminal open-ended GTMIN (80) and
        # online — NOT the last-study-stage value (0) and NOT the seasonal Jan
        # value (30). The whole tail is flat at 80.
        assert a.loc[12, "min_generation_mw"] == pytest.approx(80.0)  # Jan 2024
        assert a.loc[12, "max_generation_mw"] > 0.0  # online
        assert a.loc[12:23, "min_generation_mw"].tolist() == pytest.approx([80.0] * 12)


# ---------------------------------------------------------------------------
# TERM.DAT minimum-generation regimes and table emission
# ---------------------------------------------------------------------------


def _remaining_years_term(other_years_b: float) -> pd.DataFrame:
    """TERM.DAT rows as ``inewave`` decodes them: twelve months plus ``mes`` 13.

    TERMO_A's monthly minimum is 11 in January, 22 in February and 0 after, with
    77 for the remaining years; TERMO_B's is 33 every month, with
    ``other_years_b`` for the remaining years.
    """
    months = list(range(1, 14))
    return pd.DataFrame(
        {
            "codigo_usina": [10] * 13 + [20] * 13,
            "nome_usina": ["TERMO_A"] * 13 + ["TERMO_B"] * 13,
            "potencia_instalada": [100.0] * 13 + [200.0] * 13,
            "fator_capacidade_maximo": [100.0] * 26,
            "teif": [0.0] * 26,
            "indisponibilidade_programada": [0.0] * 26,
            "mes": months + months,
            "geracao_minima": [11.0, 22.0]
            + [0.0] * 10
            + [77.0]
            + [33.0] * 12
            + [other_years_b],
        }
    )


class TestThermalBoundsRemainingYearsMinimum:
    _ID_MAP = NewaveIdMap(subsystem_ids=[1, 2], hydro_codes=[], thermal_codes=[10, 20])

    def _case(self, tmp_path, term_df: pd.DataFrame, dger: MagicMock):
        conft, clast, term = _thermal_readers()
        term.usinas = term_df
        return make_case(tmp_path, conft=conft, clast=clast, term=term, dger=dger)

    def _plant_rows(self, case, code: int) -> pd.DataFrame:
        from novomodelo_bridge.newave.converters.thermal import convert_thermal_bounds

        table = convert_thermal_bounds(case, self._ID_MAP)
        assert table is not None
        df = table.to_pandas()
        return df[df["thermal_id"] == self._ID_MAP.thermal_id(code)].set_index(
            "stage_id"
        )

    def test_minimum_switches_to_the_remaining_years_value(self, tmp_path) -> None:
        case = self._case(tmp_path, _remaining_years_term(44.0), _make_thermal_dger())

        a = self._plant_rows(case, 10)
        assert a.loc[0, "min_generation_mw"] == pytest.approx(11.0)
        assert a.loc[1, "min_generation_mw"] == pytest.approx(22.0)
        assert a.loc[12, "min_generation_mw"] == pytest.approx(77.0)
        assert a.loc[13, "min_generation_mw"] == pytest.approx(77.0)

        b = self._plant_rows(case, 20)
        assert b.loc[11, "min_generation_mw"] == pytest.approx(33.0)
        assert b.loc[12, "min_generation_mw"] == pytest.approx(44.0)

    @pytest.mark.parametrize("maint_years", [0, 2])
    def test_boundary_is_the_first_study_year_not_the_maintenance_years(
        self, tmp_path, maint_years: int
    ) -> None:
        dger = _make_thermal_dger(mes_inicio=7)
        dger.num_anos_estudo = 3
        dger.num_anos_manutencao_utes = maint_years
        case = self._case(tmp_path, _remaining_years_term(44.0), dger)

        a = self._plant_rows(case, 10)
        # Stages 0-5 are Jul-Dec 2023, the first study year; stage 6 is Jan 2024.
        assert a.loc[5, "min_generation_mw"] == pytest.approx(0.0)
        assert a.loc[6, "min_generation_mw"] == pytest.approx(77.0)
        assert a.loc[7, "min_generation_mw"] == pytest.approx(77.0)
        assert a.loc[18, "min_generation_mw"] == pytest.approx(77.0)

    def test_blank_remaining_years_value_keeps_the_monthly_column(
        self, tmp_path
    ) -> None:
        case = self._case(
            tmp_path, _remaining_years_term(float("nan")), _make_thermal_dger()
        )

        b = self._plant_rows(case, 20)
        assert b["min_generation_mw"].tolist() == pytest.approx([33.0] * 24)

    def test_every_row_lies_inside_the_published_envelope(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.thermal import (
            convert_thermal_bounds,
            thermal_generation_bounds,
        )

        term_df = _remaining_years_term(44.0)
        term_df["indisponibilidade_programada"] = 10.0
        case = self._case(tmp_path, term_df, _make_thermal_dger())

        envelope = thermal_generation_bounds(case)
        table = convert_thermal_bounds(case, self._ID_MAP)
        assert table is not None
        for code in (10, 20):
            low, high = envelope[code]
            rows = table.to_pandas().query(
                "thermal_id == @self._ID_MAP.thermal_id(@code)"
            )
            assert (rows["min_generation_mw"] >= low).all()
            assert (rows["max_generation_mw"] <= high).all()

    def test_stage_invariant_bounds_emit_no_table(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.thermal import convert_thermal_bounds

        conft, clast, term = _thermal_readers()
        case = make_case(
            tmp_path, conft=conft, clast=clast, term=term, dger=_make_thermal_dger()
        )
        id_map = NewaveIdMap(
            subsystem_ids=[1, 2], hydro_codes=[], thermal_codes=[10, 20, 30]
        )

        assert convert_thermal_bounds(case, id_map) is None


class TestMaintenanceFileReading:
    """The maintenance file is reached through ``arquivos`` under whatever name
    the deck gives it; a file with no record means no maintenance."""

    _ID_MAP = NewaveIdMap(subsystem_ids=[1, 2], hydro_codes=[], thermal_codes=[10, 20])

    def _case(self, files, **parsed):
        conft, clast, term = _thermal_readers()
        term.usinas = _remaining_years_term(44.0)
        return make_case(
            files,
            conft=conft,
            clast=clast,
            term=term,
            dger=_make_thermal_dger(),
            **parsed,
        )

    def test_file_without_records_is_no_maintenance(self, tmp_path, caplog) -> None:
        from novomodelo_bridge.newave.converters.thermal import convert_thermal_bounds

        files = make_nw_files(tmp_path, manutt=tmp_path / "manutt.eas")
        empty = self._case(files, manutt=MagicMock(manutencoes=None))
        absent = self._case(tmp_path)

        with caplog.at_level(logging.WARNING, logger="novomodelo_bridge"):
            got = convert_thermal_bounds(empty, self._ID_MAP)

        assert got.equals(convert_thermal_bounds(absent, self._ID_MAP))
        assert "could not be parsed" not in caplog.text

    def test_unreadable_file_is_reported_by_its_deck_name(
        self, tmp_path, caplog
    ) -> None:
        from novomodelo_bridge.newave.converters.thermal import convert_thermal_bounds

        case = self._case(make_nw_files(tmp_path, manutt=tmp_path / "manutt.eas"))

        with (
            patch("novomodelo_bridge.newave.case.Manutt.read", side_effect=ValueError),
            caplog.at_level(logging.WARNING, logger="novomodelo_bridge"),
        ):
            convert_thermal_bounds(case, self._ID_MAP)

        assert "manutt.eas could not be parsed" in caplog.text


class TestThermalBoundStageSteps:
    """The per-stage bound steps in isolation."""

    @staticmethod
    def _state(**overrides: float):
        from novomodelo_bridge.newave.converters.thermal import _StageInputs

        defaults = {
            "potencia": 100.0,
            "fcmax": 100.0,
            "teif": 0.0,
            "ip": 0.0,
            "gen_min": 0.0,
        }
        defaults.update(overrides)
        return _StageInputs(**defaults)

    def test_step1_zeroes_ip_before_maintenance_end(self) -> None:
        from novomodelo_bridge.newave.converters.thermal import (
            _step1_zero_ip_before_maintenance,
        )

        state = self._state(ip=8.0)
        _step1_zero_ip_before_maintenance(state, stage_idx=2, maint_end_stage=5)
        assert state.ip == 0.0
        # At/after the maintenance end IP is left untouched.
        state2 = self._state(ip=8.0)
        _step1_zero_ip_before_maintenance(state2, stage_idx=5, maint_end_stage=5)
        assert state2.ip == 8.0

    def test_step4_applies_in_file_order_for_closed_window(self) -> None:
        from datetime import date

        from novomodelo_bridge.newave.converters.thermal import (
            _step4_apply_expt_overrides,
        )

        state = self._state()
        overrides = [
            {
                "tipo": "FCMAX",
                "modificacao": 73.38,
                "data_inicio": "2024-01-01",
                "data_fim": "2024-12-01",
            },
            {
                "tipo": "GTMIN",
                "modificacao": 469.62,
                "data_inicio": "2024-01-01",
                "data_fim": "2024-12-01",
            },
        ]
        _step4_apply_expt_overrides(
            state,
            overrides,
            ref_date=date(2024, 6, 1),
            is_post_study=False,
            last_stage_date=date(2030, 12, 1),
        )
        assert state.fcmax == pytest.approx(73.38)
        assert state.gen_min == pytest.approx(469.62)

    def test_step4_skips_window_not_covering_ref_date(self) -> None:
        from datetime import date

        from novomodelo_bridge.newave.converters.thermal import (
            _step4_apply_expt_overrides,
        )

        state = self._state(fcmax=100.0)
        overrides = [
            {
                "tipo": "FCMAX",
                "modificacao": 50.0,
                "data_inicio": "2024-01-01",
                "data_fim": "2024-03-01",
            }
        ]
        _step4_apply_expt_overrides(
            state,
            overrides,
            ref_date=date(2024, 6, 1),  # outside the window
            is_post_study=False,
            last_stage_date=date(2030, 12, 1),
        )
        assert state.fcmax == 100.0

    def test_step4_open_ended_override_blankets_post_study_tail(self) -> None:
        from datetime import date

        from novomodelo_bridge.newave.converters.thermal import (
            _step4_apply_expt_overrides,
        )

        state = self._state(potencia=100.0)
        overrides = [
            {
                "tipo": "POTEF",
                "modificacao": 250.0,
                "data_inicio": "2024-01-01",
                "data_fim": float("nan"),  # open-ended
            }
        ]
        _step4_apply_expt_overrides(
            state,
            overrides,
            ref_date=date(2026, 12, 1),
            is_post_study=True,
            last_stage_date=date(2030, 12, 1),
        )
        assert state.potencia == pytest.approx(250.0)

    def test_step4b_zeroes_out_of_window_stage(self) -> None:
        from datetime import date

        from novomodelo_bridge.newave.converters.thermal import (
            _step4b_apply_potef_availability,
        )

        state = self._state(potencia=100.0, gen_min=30.0)
        windows = [(date(2024, 1, 1), date(2024, 6, 1))]
        _step4b_apply_potef_availability(
            state, windows, stage_date=date(2024, 9, 1), nullified=True
        )
        assert state.potencia == 0.0
        assert state.gen_min == 0.0
        # Inside a window → untouched.
        s_in = self._state(potencia=100.0, gen_min=30.0)
        _step4b_apply_potef_availability(
            s_in, windows, stage_date=date(2024, 3, 1), nullified=True
        )
        assert s_in.potencia == 100.0
        # Registry not nulled (an EX plant) → the schedule does not gate it.
        s_ex = self._state(potencia=100.0, gen_min=30.0)
        _step4b_apply_potef_availability(
            s_ex, windows, stage_date=date(2024, 9, 1), nullified=False
        )
        assert s_ex.potencia == 100.0
        assert s_ex.gen_min == 30.0

    def test_step4b_zeroes_expt_plant_without_potef(self) -> None:
        """A nullified (``EE``/``NE``) plant with no POTEF window is not installed.

        Its registry capacity is discarded and EXPT declares none, so the source
        model reports a maximum of 0 instead of the TERM.DAT capacity.
        """
        from datetime import date

        from novomodelo_bridge.newave.converters.thermal import (
            _step4b_apply_potef_availability,
        )

        # No POTEF window + nullified → held out of service.
        state = self._state(potencia=204.0, gen_min=0.0)
        _step4b_apply_potef_availability(
            state, None, stage_date=date(2024, 9, 1), nullified=True
        )
        assert state.potencia == 0.0
        assert state.gen_min == 0.0

        # No POTEF window + EX → the registry capacity stands.
        s_keep = self._state(potencia=204.0, gen_min=0.0)
        _step4b_apply_potef_availability(
            s_keep, None, stage_date=date(2024, 9, 1), nullified=False
        )
        assert s_keep.potencia == 204.0

    def test_step4c_drops_gtmin_outside_window(self) -> None:
        """For a nullified plant GTMIN applies only inside EXPT windows; outside
        it is 0 (capacity kept), whatever the TERM.DAT minimum says."""
        from datetime import date

        from novomodelo_bridge.newave.converters.thermal import (
            _step4c_apply_gtmin_availability,
        )

        windows = [(date(2024, 9, 1), date(2024, 10, 1))]
        # Inside the window → minimum kept; capacity untouched.
        s_in = self._state(potencia=235.0, gen_min=218.68)
        _step4c_apply_gtmin_availability(
            s_in, windows, stage_date=date(2024, 9, 1), nullified=True
        )
        assert s_in.gen_min == 218.68
        assert s_in.potencia == 235.0
        # Outside the window → minimum dropped to 0; capacity untouched.
        s_out = self._state(potencia=235.0, gen_min=201.5)
        _step4c_apply_gtmin_availability(
            s_out, windows, stage_date=date(2024, 11, 1), nullified=True
        )
        assert s_out.gen_min == 0.0
        assert s_out.potencia == 235.0
        # Registry not nulled (an EX plant) → its minimum survives outside.
        s_ex = self._state(potencia=235.0, gen_min=201.5)
        _step4c_apply_gtmin_availability(
            s_ex, windows, stage_date=date(2024, 11, 1), nullified=False
        )
        assert s_ex.gen_min == 201.5

    def test_step4c_drops_gtmin_for_expt_plant_without_gtmin(self) -> None:
        """A nullified plant with no GTMIN window has no minimum, even with a
        nonzero TERM.DAT GTMIN."""
        from datetime import date

        from novomodelo_bridge.newave.converters.thermal import (
            _step4c_apply_gtmin_availability,
        )

        # No GTMIN window + nullified → minimum dropped, capacity untouched.
        s = self._state(potencia=75.0, gen_min=62.99)
        _step4c_apply_gtmin_availability(
            s, None, stage_date=date(2024, 9, 1), nullified=True
        )
        assert s.gen_min == 0.0
        assert s.potencia == 75.0
        # No GTMIN window + EX → the registry minimum stands.
        s_keep = self._state(potencia=75.0, gen_min=62.99)
        _step4c_apply_gtmin_availability(
            s_keep, None, stage_date=date(2024, 9, 1), nullified=False
        )
        assert s_keep.gen_min == 62.99

    def test_step5_subtracts_maint_reduction_before_maint_end(self) -> None:
        import numpy as np

        from novomodelo_bridge.newave.converters.thermal import (
            _step5_apply_maint_reduction,
        )

        state = self._state(potencia=100.0)
        reduction = np.array([10.0, 20.0, 30.0])
        _step5_apply_maint_reduction(state, reduction, stage_idx=1, maint_end_stage=3)
        assert state.potencia == pytest.approx(80.0)
        # At/after maint end → no reduction.
        s2 = self._state(potencia=100.0)
        _step5_apply_maint_reduction(s2, reduction, stage_idx=3, maint_end_stage=3)
        assert s2.potencia == 100.0

    def test_step6_normal_case(self) -> None:
        from novomodelo_bridge.newave.converters.thermal import _step6_evaluate_bounds

        state = self._state(potencia=200.0, fcmax=100.0, ip=0.0, teif=0.0, gen_min=50.0)
        min_mw, max_mw, exceeded = _step6_evaluate_bounds(state)
        assert max_mw == pytest.approx(200.0)
        assert min_mw == pytest.approx(50.0)
        assert exceeded is False

    def test_step6_honors_gtmin_above_capacity(self) -> None:
        """GTMIN (the inflexible minimum) is honored even when it exceeds the
        FCMAX-derived capacity; the cap is lifted to it, never the minimum clamped
        down: capacity 420.88 < GTMIN 469.62 gives [469.62, 469.62]."""
        from novomodelo_bridge.newave.converters.thermal import _step6_evaluate_bounds

        state = self._state(
            potencia=420.88, fcmax=100.0, ip=0.0, teif=0.0, gen_min=469.62
        )
        min_mw, max_mw, exceeded = _step6_evaluate_bounds(state)
        assert min_mw == pytest.approx(469.62)  # GTMIN honored, not clamped down
        assert max_mw == pytest.approx(469.62)  # cap lifted to GTMIN for feasibility
        assert exceeded is True

    def test_step6_rounding_excess_lifts_the_cap_without_flagging(self) -> None:
        """A GTMIN equal to the available capacity rounded to the deck's 0.01 MW
        (477.9585 written as 477.96) is not a data error."""
        from novomodelo_bridge.newave.converters.thermal import _step6_evaluate_bounds

        state = self._state(
            potencia=496.2193548387096, fcmax=100.0, ip=0.0, teif=3.68, gen_min=477.96
        )
        min_mw, max_mw, exceeded = _step6_evaluate_bounds(state)
        assert min_mw == pytest.approx(477.96)
        assert max_mw == pytest.approx(477.96)
        assert exceeded is False

    def test_step6_clamps_negative_potencia_to_zero(self) -> None:
        from novomodelo_bridge.newave.converters.thermal import _step6_evaluate_bounds

        state = self._state(potencia=-5.0, fcmax=100.0, gen_min=10.0)
        min_mw, max_mw, exceeded = _step6_evaluate_bounds(state)
        # gen_min 10 > capacity 0 → honor GTMIN, lift cap.
        assert min_mw == pytest.approx(10.0)
        assert max_mw == pytest.approx(10.0)
        assert exceeded is True


class TestGtminAboveCapacityDiagnostic:
    def test_table_reports_the_excess_at_the_deck_resolution(self, tmp_path) -> None:
        from novomodelo_bridge.core import diagnostics as dx
        from novomodelo_bridge.newave.converters.thermal import (
            _emit_gtmin_above_capacity,
            _GtminRecord,
        )

        conft, _clast, _term = _thermal_readers()
        case = make_case(tmp_path, conft=conft)
        records = [
            _GtminRecord(code=10, stage_id=2, gtmin_mw=478.03, capacity_mw=477.9585),
            _GtminRecord(code=10, stage_id=3, gtmin_mw=478.01, capacity_mw=477.99),
        ]

        with dx.collect() as collected:
            _emit_gtmin_above_capacity(records, case)

        [diag] = collected
        assert diag.table is not None
        assert diag.table.columns[-1] == "Excess MW"
        assert diag.table.rows == [["TERMO_A", 10, "2-3", 478.03, 477.96, 0.07]]


class TestThermalGenerationBounds:
    """``thermal_generation_bounds`` returns the static ``[min_mw, max_mw]``."""

    def test_pair_envelopes_both_minimum_and_availability_regimes(
        self, tmp_path
    ) -> None:
        """The pair spans the whole horizon, not the registry row alone.

        IP is zeroed inside the maintenance year and applies after it, and the
        minimum switches from the monthly columns to the ``mes``-13 value after
        the first study year, so the widest maximum and the smallest minimum come
        from different stages.
        """
        from novomodelo_bridge.newave.converters.thermal import (
            thermal_generation_bounds,
        )

        conft, clast, term = _thermal_readers()
        term.usinas = pd.DataFrame(
            {
                "codigo_usina": [10] * 13,
                "nome_usina": ["TERMO_A"] * 13,
                "potencia_instalada": [100.0] * 13,
                "fator_capacidade_maximo": [90.0] * 13,
                "teif": [0.0] * 13,
                "indisponibilidade_programada": [10.0] * 13,
                "mes": list(range(1, 14)),
                "geracao_minima": [50.0] * 12 + [20.0],
            }
        )
        case = make_case(
            tmp_path, conft=conft, clast=clast, term=term, dger=_make_thermal_dger()
        )

        # max: 100 * 90% with IP zeroed (stages 0-11), against 100 * 90% * 90%
        # after the maintenance year. min: the mes-13 value, below every month.
        assert thermal_generation_bounds(case)[10] == pytest.approx((20.0, 90.0))

    def test_registry_minimum_above_capacity_keeps_the_pair_ordered(
        self, tmp_path
    ) -> None:
        """A registry GTMIN above the registry capacity product cannot invert the pair.

        ``term.dat`` may declare a minimum the ``potencia x fcmax`` product
        cannot reach; the per-stage evaluation lifts the ceiling to the
        inflexible minimum, and the envelope inherits that ordering instead of
        publishing an empty interval no committed value could satisfy.
        """
        from novomodelo_bridge.newave.converters.thermal import (
            thermal_generation_bounds,
        )

        conft, clast, term = _thermal_readers()
        term.usinas = pd.DataFrame(
            {
                "codigo_usina": [10],
                "nome_usina": ["TERMO_A"],
                "potencia_instalada": [161.0],
                "fator_capacidade_maximo": [93.0],
                "teif": [0.0],
                "indisponibilidade_programada": [0.0],
                "mes": [1],
                "geracao_minima": [161.38],
            }
        )
        case = make_case(
            tmp_path, conft=conft, clast=clast, term=term, dger=_make_thermal_dger()
        )

        low, high = thermal_generation_bounds(case)[10]
        assert low <= high
        assert (low, high) == pytest.approx((161.38, 161.38))

    def test_no_usinas_returns_empty(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.thermal import (
            thermal_generation_bounds,
        )

        conft, clast, term = _thermal_readers()
        term.usinas = None
        case = make_case(
            tmp_path, conft=conft, clast=clast, term=term, dger=_make_thermal_dger()
        )

        assert thermal_generation_bounds(case) == {}
