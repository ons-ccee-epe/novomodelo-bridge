"""Tier-1 tests for the shared stored-energy constraint terms."""

from __future__ import annotations

from novomodelo_bridge.core.stored_energy import (
    StoredEnergyShare,
    cascade_share,
    share_parameters,
    stored_energy_floor,
    stored_energy_lhs,
)


def _expression_share(
    hydro_id: int, coefficient: str, max_energy: str
) -> StoredEnergyShare:
    return StoredEnergyShare(
        hydro_id,
        coefficient,
        max_energy,
        {coefficient: f"@rho_eq_mwh_h{hydro_id}", max_energy: f"2.0 * @{coefficient}"},
    )


def test_cascade_share_reads_novomodelo_computed_parameters_only() -> None:
    share = cascade_share(3)

    assert share.coefficient == "rho_acum_mwh_h3"
    assert share.max_energy == "earm_max_mwh_h3"
    assert share.parameters == {}


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
    shares = [(1.0, cascade_share(0)), (-1.0, _expression_share(1, "rho1", "emax1"))]

    assert stored_energy_floor(shares, "pct").split(" >= ")[1] == (
        "@pct * (@earm_max_mwh_h0 - @emax1)"
    )


def test_share_parameters_merges_in_first_seen_order() -> None:
    params = share_parameters(
        [
            _expression_share(0, "rho0", "emax0"),
            cascade_share(1),
            _expression_share(0, "rho0", "emax0_down"),
        ]
    )

    assert list(params) == ["rho0", "emax0", "emax0_down"]
