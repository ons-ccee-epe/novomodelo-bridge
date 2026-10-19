"""Tier-1 tests for the shared stored-energy constraint terms."""

from __future__ import annotations

from novomodelo_bridge.core.stored_energy import (
    cascade_share,
    cut_cascade_share,
    downstream_share,
    share_parameters,
    stored_energy_floor,
    stored_energy_lhs,
)


def test_cascade_share_reads_novomodelo_computed_parameters_only() -> None:
    share = cascade_share(3)

    assert share.coefficient == "rho_acum_mwh_h3"
    assert share.max_energy == "earm_max_mwh_h3"
    assert share.parameters == {}


def test_cut_cascade_share_sums_equivalent_productivities_down_to_the_cut() -> None:
    share = cut_cascade_share(0, 900.0, [0, 1])

    assert share.coefficient == "rho_cut_mwh_h0"
    assert share.max_energy == "earm_max_cut_mwh_h0"
    assert share.parameters == {
        "rho_cut_mwh_h0": "@rho_eq_mwh_h0 + @rho_eq_mwh_h1",
        "earm_max_cut_mwh_h0": "900.0 * @rho_cut_mwh_h0",
    }


def test_downstream_share_values_the_volume_below_the_cut() -> None:
    share = downstream_share(0, 900.0, 2)

    assert share.coefficient == "rho_acum_mwh_h2"
    assert share.max_energy == "earm_max_fict_mwh_h0"
    assert share.parameters == {"earm_max_fict_mwh_h0": "900.0 * @rho_acum_mwh_h2"}


def test_lhs_keeps_one_term_per_share_with_its_sign() -> None:
    lhs = stored_energy_lhs([(1.0, cascade_share(0)), (-1.0, cascade_share(1))])

    assert lhs == (
        "@rho_acum_mwh_h0 * hydro_useful_volume_final(0)"
        " - @rho_acum_mwh_h1 * hydro_useful_volume_final(1)"
    )


def test_lhs_scales_a_non_unit_sign() -> None:
    assert stored_energy_lhs([(-2.0, cascade_share(0))]) == (
        "- 2.0 * @rho_acum_mwh_h0 * hydro_useful_volume_final(0)"
    )


def test_floor_of_one_share_needs_no_parentheses() -> None:
    assert stored_energy_floor([(1.0, cascade_share(0))], "pct") == (
        "@rho_acum_mwh_h0 * hydro_useful_volume_final(0) >= @pct * @earm_max_mwh_h0"
    )


def test_floor_applies_the_fraction_to_the_signed_maximum_sum() -> None:
    shares = [(1.0, cascade_share(0)), (-1.0, downstream_share(1, 50.0, 0))]

    assert stored_energy_floor(shares, "pct").split(" >= ")[1] == (
        "@pct * (@earm_max_mwh_h0 - @earm_max_fict_mwh_h1)"
    )


def test_share_parameters_merges_in_first_seen_order() -> None:
    params = share_parameters(
        [cut_cascade_share(0, 1.0, [0]), cascade_share(1), downstream_share(0, 1.0, 1)]
    )

    assert list(params) == [
        "rho_cut_mwh_h0",
        "earm_max_cut_mwh_h0",
        "earm_max_fict_mwh_h0",
    ]
