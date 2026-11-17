"""Tier-1 tests for DECOMP's REE energy definition
(``comparators.decomp.ree_energy``).

Hand-built credits and rules, and the builder over a synthetic
``EffectiveCadastro``; no deck.
"""

from __future__ import annotations

import polars as pl
import pytest

from novomodelo_bridge.comparators.decomp.ree_energy import (
    EnergyCredit,
    ReeEnergyRule,
    build_ree_energy_rule,
    inflow_energy_credits,
    stored_energy_credits,
)
from novomodelo_bridge.core.units import C_M3S2HM3, MEAN_MONTH_HOURS
from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro
from tests.decomp.test_rhe import _hidr_frame, _hydro_row


class TestStoredEnergyCredits:
    def test_without_a_cut_the_whole_chain_counts_in_the_plant_s_ree(self) -> None:
        credits = stored_energy_credits((1, 2, 3), {1: 10, 2: 20, 3: 30}, cut=False)

        assert credits == (
            EnergyCredit(1, 10),
            EnergyCredit(2, 10),
            EnergyCredit(3, 10),
        )

    def test_a_cut_moves_the_rest_of_the_chain_to_the_first_ree_below(self) -> None:
        credits = stored_energy_credits((1, 2, 3), {1: 10, 2: 20, 3: 30}, cut=True)

        assert credits == (
            EnergyCredit(1, 10),
            EnergyCredit(2, 20, capped=True),
            EnergyCredit(3, 20, capped=True),
        )

    def test_a_chain_plant_without_an_ree_is_counted_by_the_current_one(self) -> None:
        credits = stored_energy_credits((1, 9, 2), {1: 10, 2: 20}, cut=True)

        assert credits == (
            EnergyCredit(1, 10),
            EnergyCredit(9, 10),
            EnergyCredit(2, 20, capped=True),
        )


class TestInflowEnergyCredits:
    def test_inside_a_subsystem_the_chain_moves_until_it_passes_a_monthly_plant(
        self,
    ) -> None:
        credits = inflow_energy_credits(
            (1, 2, 3), {1: 10, 2: 20, 3: 30}, {1: 1, 2: 1, 3: 1}, monthly={2}
        )

        assert [credit.ree for credit in credits] == [10, 20, 20]

    def test_a_monthly_first_plant_keeps_the_chain_in_its_ree(self) -> None:
        credits = inflow_energy_credits((1, 2), {1: 10, 2: 20}, {1: 1, 2: 1}, {1})

        assert [credit.ree for credit in credits] == [10, 10]

    def test_a_subsystem_crossing_moves_the_chain_and_forgets_the_monthly_plant(
        self,
    ) -> None:
        credits = inflow_energy_credits(
            (1, 2, 3), {1: 10, 2: 20, 3: 30}, {1: 1, 2: 2, 3: 2}, monthly={1}
        )

        assert [credit.ree for credit in credits] == [10, 20, 30]
        assert not any(credit.capped for credit in credits)


def _rule() -> ReeEnergyRule:
    """UP (1, REE 10, ``EZ`` 50 %) drains into DOWN (2, REE 20).

    UP's range is [100, 300] hm³ and its ρ is ``1 + useful / 100``; DOWN's
    range is [0, 1000] hm³ and its ρ is 2.0.
    """
    ranges = {1: (100.0, 300.0), 2: (0.0, 1000.0)}
    rho = {1: lambda useful: 1.0 + useful / 100.0, 2: lambda _useful: 2.0}
    return ReeEnergyRule(
        stored={
            1: (EnergyCredit(1, 10), EnergyCredit(2, 20, capped=True)),
            2: (EnergyCredit(2, 20),),
        },
        inflow={
            1: (EnergyCredit(1, 10), EnergyCredit(2, 20)),
            2: (EnergyCredit(2, 20),),
        },
        ez_fractions={1: 0.5},
        storage_range=lambda code, _stage: ranges[code],
        productivity=lambda code, _stage: rho[code],
    )


def _plants(rows: list[tuple[int, int, int, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "scenario_id": pl.Int64,
            "stage_id": pl.Int64,
            "codigo_usina": pl.Int64,
            "storage_final_hm3": pl.Float64,
            "incremental_inflow_m3s": pl.Float64,
        },
        orient="row",
    )


def _by_ree(energies: pl.DataFrame) -> dict[int, tuple[float, float]]:
    return {
        row["codigo_ree"]: (row["earm_mwmes"], row["ena_mwmes"])
        for row in energies.iter_rows(named=True)
    }


class TestReeEnergyRuleEnergies:
    def test_values_storage_at_its_head_and_caps_the_cut_share(self) -> None:
        # UP holds 160 hm³ useful (ρ 2.6), DOWN 500 hm³ (ρ 2.0). UP's share
        # below the cut is capped at 50 % of its 200 hm³ range.
        plants = _plants([(0, 0, 1, 260.0, 10.0), (0, 0, 2, 500.0, 20.0)])

        out = _by_ree(_rule().energies(plants, {}))

        assert out[10] == (
            pytest.approx(160.0 * 2.6 / C_M3S2HM3),
            pytest.approx(10.0 * 2.6),
        )
        assert out[20] == (
            pytest.approx((100.0 * 2.0 + 500.0 * 2.0) / C_M3S2HM3),
            pytest.approx(10.0 * 2.0 + 20.0 * 2.0),
        )

    def test_ena_spans_the_stage_hours_in_mean_months(self) -> None:
        plants = _plants([(0, 0, 1, 260.0, 10.0), (0, 0, 2, 500.0, 20.0)])

        out = _by_ree(_rule().energies(plants, {0: MEAN_MONTH_HOURS / 2}))

        assert out[10][1] == pytest.approx(10.0 * 2.6 / 2)
        assert out[20][1] == pytest.approx((10.0 * 2.0 + 20.0 * 2.0) / 2)

    def test_storage_below_the_floor_holds_nothing(self) -> None:
        plants = _plants([(0, 0, 1, 50.0, 10.0), (0, 0, 2, 0.0, 0.0)])

        out = _by_ree(_rule().energies(plants, {}))

        assert out[10] == (0.0, pytest.approx(10.0 * 1.0))
        assert out[20][0] == 0.0

    def test_a_plant_without_a_row_holds_no_water_and_has_no_inflow(self) -> None:
        plants = _plants([(0, 0, 1, 260.0, 10.0)])

        out = _by_ree(_rule().energies(plants, {}))

        assert out[20] == (
            pytest.approx(100.0 * 2.0 / C_M3S2HM3),
            pytest.approx(10.0 * 2.0),
        )

    def test_a_row_of_a_plant_on_no_chain_is_ignored(self) -> None:
        plants = _plants([(0, 0, 1, 260.0, 10.0), (0, 0, 99, 1e6, 1e6)])

        out = _by_ree(_rule().energies(plants, {}))

        assert out[10][0] == pytest.approx(160.0 * 2.6 / C_M3S2HM3)

    def test_each_scenario_and_stage_is_valued_alone(self) -> None:
        plants = _plants(
            [
                (0, 0, 1, 260.0, 10.0),
                (1, 0, 1, 100.0, 10.0),
                (0, 1, 1, 300.0, 10.0),
            ]
        )

        out = _rule().energies(plants, {})

        earm_10 = {
            (row["scenario_id"], row["stage_id"]): row["earm_mwmes"]
            for row in out.filter(pl.col("codigo_ree") == 10).iter_rows(named=True)
        }
        assert earm_10 == {
            (0, 0): pytest.approx(160.0 * 2.6 / C_M3S2HM3),
            (1, 0): 0.0,
            (0, 1): pytest.approx(200.0 * 3.0 / C_M3S2HM3),
        }


def _membership(rows: dict[int, tuple[int, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "codigo_usina": list(rows),
            "codigo_ree": [ree for ree, _ in rows.values()],
            "codigo_submercado": [subsystem for _, subsystem in rows.values()],
        }
    )


def _cascade(energy_links: dict[int, tuple[int | None, ...]] | None = None):
    """UP (1, "M", REE 10) -> MID (2, "S", REE 20) -> RUN (3, "D", REE 20)."""
    return EffectiveCadastro(
        base=_hidr_frame(
            {
                1: _hydro_row(tipo_regulacao="M", codigo_usina_jusante=2),
                2: _hydro_row(tipo_regulacao="S", codigo_usina_jusante=3),
                3: _hydro_row(tipo_regulacao="D", codigo_usina_jusante=0),
            }
        ),
        n_stages=1,
        stage_varying={},
        energy_links=energy_links or {},
    )


_CASCADE_MEMBERSHIP = {1: (10, 1), 2: (20, 1), 3: (20, 1)}


class TestBuildReeEnergyRule:
    def test_credits_follow_the_deck_s_chains(self) -> None:
        rule = build_ree_energy_rule(
            _cascade(), (1, 2, 3), _membership(_CASCADE_MEMBERSHIP), {1: 0.5}
        )

        assert rule.stored[1] == (
            EnergyCredit(1, 10),
            EnergyCredit(2, 20, capped=True),
            EnergyCredit(3, 20, capped=True),
        )
        assert rule.stored[2] == (EnergyCredit(2, 20), EnergyCredit(3, 20))
        # UP is monthly, so its inflow stays in REE 10 down the subsystem.
        assert [credit.ree for credit in rule.inflow[1]] == [10, 10, 10]
        assert rule.ez_fractions == {1: 0.5}

    def test_reads_ranges_and_productivity_from_the_cadastro(self) -> None:
        rule = build_ree_energy_rule(
            _cascade(), (1, 2, 3), _membership(_CASCADE_MEMBERSHIP), {}
        )

        assert rule.storage_range(1, 0) == (0.0, 1000.0)
        assert rule.storage_range(3, 0) == (500.0, 500.0)
        # Flat 100 m forebay, ρ_esp 0.01: 1.0 MW/(m³/s) at any storage.
        assert rule.productivity(1, 0)(250.0) == pytest.approx(1.0)

    def test_an_operated_plant_without_an_ree_has_no_chain_of_its_own(self) -> None:
        rule = build_ree_energy_rule(
            _cascade(), (1, 2, 3), _membership({1: (10, 1), 2: (20, 1)}), {}
        )

        assert set(rule.stored) == {1, 2}
        assert rule.stored[2] == (EnergyCredit(2, 20), EnergyCredit(3, 20))

    def test_a_chain_that_loops_back_stops_before_the_revisited_plant(self) -> None:
        rule = build_ree_energy_rule(
            _cascade({3: (1,)}), (1, 2, 3), _membership(_CASCADE_MEMBERSHIP), {}
        )

        assert [credit.plant for credit in rule.stored[1]] == [1, 2, 3]

    def test_an_energy_link_outside_hidr_raises(self) -> None:
        with pytest.raises(ValueError, match="names plant 77"):
            build_ree_energy_rule(
                _cascade({2: (77,)}), (1, 2, 3), _membership(_CASCADE_MEMBERSHIP), {}
            )
