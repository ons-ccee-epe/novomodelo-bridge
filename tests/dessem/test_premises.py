"""Tests for dessem/exporters/fcf/premises.py."""

from novomodelo_bridge.dessem.exporters.fcf.core.premises import (
    P1,
    P2,
    P3,
    P4,
    P5,
    P12,
    P14,
    PREMISES,
)


class TestPremises:
    """Test the PREMISES tuple and individual premise constants."""

    def test_premises_count(self) -> None:
        """PREMISES contains exactly 15 premises."""
        assert len(PREMISES) == 15

    def test_premises_are_strings(self) -> None:
        """All premises are non-empty strings."""
        for premise in PREMISES:
            assert isinstance(premise, str)
            assert len(premise) > 10

    def test_p1_mentions_zeros(self) -> None:
        """P1 mentions the zero-filled physical records."""
        assert "zeros" in P1.lower() or "zero" in P1.lower()

    def test_p3_mentions_travel_time(self) -> None:
        """P3 mentions travel-time or n_utv."""
        assert "travel" in P3.lower() or "n_utv" in P3.lower()

    def test_p2_exempts_pi_gnl_from_the_division(self) -> None:
        assert "pi_gnl" in P2 and "MWh" in P2

    def test_p4_is_not_dormant(self) -> None:
        assert "DORMANT" not in P4

    def test_p5_is_not_dormant(self) -> None:
        assert "DORMANT" not in P5

    def test_p12_states_the_block_hours(self) -> None:
        assert "hours" in P12

    def test_p14_states_the_exported_coefficient(self) -> None:
        assert "second stage" in P14

    def test_p14_states_the_drop_without_folding_it_into_the_intercept(self) -> None:
        assert "intercept" in P14
        assert "counted" in P14

    def test_premises_numbered_correctly(self) -> None:
        """Each premise starts with its number (P1, P2, etc.)."""
        for i, premise in enumerate(PREMISES, start=1):
            assert premise.startswith(f"P{i}:"), (
                f"Premise {i} should start with 'P{i}:'"
            )
