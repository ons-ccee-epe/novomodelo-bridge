"""Tests for cli/validate.py."""

from __future__ import annotations


class TestPartitionValidationWarnings:
    """``_partition_validation_warnings`` — the pure whitelist filter."""

    def test_partition_whitelists_interop_warning(self) -> None:
        """The interop message is whitelisted; an unrelated one still renders."""
        from novomodelo_bridge.cli.app import _partition_validation_warnings

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
        from novomodelo_bridge.cli.app import _partition_validation_warnings

        warnings: list[object] = ["w1", "w2", {"message": "w3"}]

        rendered, whitelisted = _partition_validation_warnings(warnings, ())

        assert rendered == warnings
        assert whitelisted == []


_STD_ZERO = "Hydro {} stage {}: std_m3s is 0.0, indicating deterministic inflow"
_ABOVE_MAX = "Hydro {}: initial storage ({}) lies above the maximum storage"


class TestGroupValidationMessages:
    """``_group_validation_messages`` — alike warnings fold into one group."""

    def test_messages_differing_only_in_numbers_share_a_group(self) -> None:
        from novomodelo_bridge.cli.validate import _group_validation_messages

        warnings: list[object] = [
            _ABOVE_MAX.format(75, 2716.78),
            _STD_ZERO.format(159, 83),
            {"message": _STD_ZERO.format(12, -1)},
            _STD_ZERO.format(7, 0),
        ]

        assert _group_validation_messages(warnings) == [
            (_STD_ZERO.format(159, 83), 3),
            (_ABOVE_MAX.format(75, 2716.78), 1),
        ]

    def test_equal_counts_keep_the_order_they_came_in(self) -> None:
        from novomodelo_bridge.cli.validate import _group_validation_messages

        assert _group_validation_messages(["b", "a"]) == [("b", 1), ("a", 1)]


class TestPrintValidationWarnings:
    """``_print_validation_warnings`` — one line per group unless verbose."""

    @staticmethod
    def _print(warnings: list[object], *, verbose: bool) -> list[str]:
        import io

        from rich.console import Console

        from novomodelo_bridge.cli.validate import _print_validation_warnings

        buf = io.StringIO()
        console = Console(file=buf, width=300, no_color=True, highlight=False)
        _print_validation_warnings(warnings, verbose=verbose, err_console=console)
        return buf.getvalue().splitlines()

    def test_alike_warnings_print_once_with_their_count(self) -> None:
        warnings: list[object] = [_STD_ZERO.format(i, i) for i in range(4)]
        warnings.append(_ABOVE_MAX.format(75, 2716.78))

        assert self._print(warnings, verbose=False) == [
            f"Validation warning (4×): {_STD_ZERO.format(0, 0)}",
            f"Validation warning: {_ABOVE_MAX.format(75, 2716.78)}",
            "  -v lists all 5 validation warnings.",
        ]

    def test_distinct_warnings_print_without_the_hint(self) -> None:
        lines = self._print(["first", "second"], verbose=False)

        assert lines == ["Validation warning: first", "Validation warning: second"]

    def test_verbose_prints_every_warning(self) -> None:
        warnings: list[object] = [_STD_ZERO.format(i, i) for i in range(3)]

        assert self._print(warnings, verbose=True) == [
            f"Validation warning: {_STD_ZERO.format(i, i)}" for i in range(3)
        ]
