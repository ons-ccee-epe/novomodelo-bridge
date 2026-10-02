"""Tests for dessem/exporters/fcf/premises.py."""

from novomodelo_bridge.dessem.exporters.fcf.premises import P1, P3, P14, PREMISES


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

    def test_p14_mentions_gnl(self) -> None:
        """P14 mentions GNL or anticipated thermal."""
        assert "gnl" in P14.lower() or "anticipated" in P14.lower()

    def test_premises_numbered_correctly(self) -> None:
        """Each premise starts with its number (P1, P2, etc.)."""
        for i, premise in enumerate(PREMISES, start=1):
            assert premise.startswith(f"P{i}:"), (
                f"Premise {i} should start with 'P{i}:'"
            )
