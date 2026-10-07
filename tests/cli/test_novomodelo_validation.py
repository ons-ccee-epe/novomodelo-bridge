"""Tests for cli/validate.py."""

from __future__ import annotations


class TestPartitionValidationWarnings:
    """``_partition_validation_warnings`` — the pure whitelist filter."""

    def test_partition_whitelists_interop_warning(self) -> None:
        """The interop message is whitelisted; an unrelated one still renders."""
        from cobre_bridge.cli.app import _partition_validation_warnings

        interop = (
            "inflow lags are disabled on all study stages. This is a valid "
            "configuration for external-solver interoperability; otherwise "
            "it is likely a misconfiguration."
        )
        unrelated = "some unrelated warning"

        rendered, whitelisted = _partition_validation_warnings(
            [interop, unrelated], ("external-solver interoperability",)
        )

        assert rendered == [unrelated]
        assert whitelisted == [interop]

    def test_partition_empty_whitelist_is_identity(self) -> None:
        """An empty whitelist — what ``convert newave`` passes — changes nothing."""
        from cobre_bridge.cli.app import _partition_validation_warnings

        warnings: list[object] = ["w1", "w2", {"message": "w3"}]

        rendered, whitelisted = _partition_validation_warnings(warnings, ())

        assert rendered == warnings
        assert whitelisted == []
