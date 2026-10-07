"""Unit tests for the typed CLI argument dataclasses (``novomodelo_bridge.cli.args``).

Tier 1 — pure Python, imports no novomodelo.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path

import pytest
from rich.console import Console

from novomodelo_bridge.cli.args import (
    CheckArgs,
    CommonArgs,
    CompareArgs,
    ConvertArgs,
    DashboardArgs,
)

_FAMILIES = (ConvertArgs, CompareArgs, CheckArgs, DashboardArgs)


def _common_kwargs() -> dict[str, object]:
    return {
        "json_output": False,
        "verbose": 0,
        "log_file": None,
        "no_color": True,
        "quiet": False,
    }


def test_common_args_is_frozen_dataclass() -> None:
    """CommonArgs is a dataclass and its instances refuse attribute assignment."""
    assert dataclasses.is_dataclass(CommonArgs)
    assert CommonArgs.__dataclass_params__.frozen is True


@pytest.mark.parametrize("family", _FAMILIES)
def test_family_is_frozen_dataclass_subclassing_common_args(
    family: type[CommonArgs],
) -> None:
    """Each command-family dataclass is frozen and subclasses CommonArgs."""
    assert dataclasses.is_dataclass(family)
    assert family.__dataclass_params__.frozen is True
    assert issubclass(family, CommonArgs)
    assert family is not CommonArgs


def test_err_console_propagates_no_color_true() -> None:
    """err_console() builds a stderr Console honouring no_color=True."""
    args = CommonArgs(**_common_kwargs())
    console = args.err_console()
    assert isinstance(console, Console)
    assert console.no_color is True


def test_out_console_propagates_no_color_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """out_console() with no_color=False leaves Rich's NO_COLOR auto-detection
    in force; with the env var unset that resolves to no_color=False."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    kwargs = _common_kwargs()
    kwargs["no_color"] = False
    args = CommonArgs(**kwargs)
    console = args.out_console()
    assert isinstance(console, Console)
    assert console.no_color is False


def test_common_args_frozen_instance_raises_on_assignment() -> None:
    """Assigning a field on a frozen CommonArgs instance raises FrozenInstanceError."""
    args = CommonArgs(**_common_kwargs())
    with pytest.raises(dataclasses.FrozenInstanceError):
        args.quiet = True  # type: ignore[misc]


def test_convert_args_frozen_instance_raises_on_assignment() -> None:
    """Assigning a field on a frozen ConvertArgs instance raises FrozenInstanceError."""
    args = ConvertArgs(
        **_common_kwargs(),
        src=Path("src"),
        dst=Path("dst"),
        validate=False,
        force=False,
        diagnostics_json=None,
        dry_run=False,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        args.force = True  # type: ignore[misc]


def test_compare_args_frozen_instance_raises_on_assignment() -> None:
    """Assigning a field on a frozen CompareArgs instance raises FrozenInstanceError."""
    args = CompareArgs(
        **_common_kwargs(),
        source_dir=Path("source"),
        novomodelo_output_dir=Path("novomodelo"),
        tolerance=None,
        format=None,
        out_dir=None,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        args.tolerance = 1.0  # type: ignore[misc]


def test_check_args_frozen_instance_raises_on_assignment() -> None:
    """Assigning a field on a frozen CheckArgs instance raises FrozenInstanceError."""
    args = CheckArgs(**_common_kwargs(), src=Path("src"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        args.src = Path("other")  # type: ignore[misc]


def test_dashboard_args_frozen_instance_raises_on_assignment() -> None:
    """Assigning a field on a frozen DashboardArgs instance raises FrozenInstanceError."""
    args = DashboardArgs(
        **_common_kwargs(),
        case_dir=Path("case"),
        output=None,
        open_browser=False,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        args.open_browser = True  # type: ignore[misc]


def test_convert_args_no_fcf_defaults_false() -> None:
    """ConvertArgs.no_fcf defaults to False (only `convert decomp` sets it)."""
    args = ConvertArgs(
        **_common_kwargs(),
        src=Path("src"),
        dst=Path("dst"),
        validate=False,
        force=False,
        diagnostics_json=None,
        dry_run=False,
    )
    assert args.no_fcf is False


def _type_checking_guarded_node_ids(tree: ast.Module) -> set[int]:
    """Collect ``id()`` of every AST node nested inside an ``if TYPE_CHECKING:``
    block's true branch (any nesting depth, matched by name or attribute)."""
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        is_type_checking = (
            isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"
        ) or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")
        if not is_type_checking:
            continue
        for stmt in node.body:
            for sub in ast.walk(stmt):
                guarded.add(id(sub))
    return guarded


def _rich_import_nodes(tree: ast.Module) -> list[ast.Import | ast.ImportFrom]:
    """Every ``import rich...`` / ``from rich... import`` node anywhere in the tree."""
    nodes: list[ast.Import | ast.ImportFrom] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            alias.name == "rich" or alias.name.startswith("rich.")
            for alias in node.names
        ):
            nodes.append(node)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module == "rich" or node.module.startswith("rich."):
                nodes.append(node)
    return nodes


def test_rich_import_only_inside_type_checking_guard() -> None:
    """The module never imports Rich at runtime — only under TYPE_CHECKING.

    Parses the module with ``ast`` (rather than scanning source lines) so a
    runtime ``from rich import ...`` placed before or beside the
    ``TYPE_CHECKING`` guard is still caught.
    """
    import novomodelo_bridge.cli.args as module

    tree = ast.parse(inspect.getsource(module))
    rich_imports = _rich_import_nodes(tree)
    assert rich_imports, "expected a rich import in cli/args.py"

    guarded_ids = _type_checking_guarded_node_ids(tree)
    for node in rich_imports:
        assert id(node) in guarded_ids, (
            f"rich import on line {node.lineno} must be nested inside an "
            "`if TYPE_CHECKING:` guard, never at runtime module scope"
        )


class TestParseFormats:
    """``_parse_formats`` token parsing and validation."""

    def test_parse_formats_default(self) -> None:
        from novomodelo_bridge.cli.args import _parse_formats

        assert _parse_formats(None) == {"console", "parquet", "json"}

    def test_parse_formats_comma_and_repeat(self) -> None:
        from novomodelo_bridge.cli.args import _parse_formats

        assert _parse_formats(["csv,json", "parquet"]) == {
            "csv",
            "json",
            "parquet",
        }

    def test_parse_formats_all_expands(self) -> None:
        from novomodelo_bridge.cli.args import _parse_formats

        assert _parse_formats(["all"]) == {
            "console",
            "html",
            "csv",
            "parquet",
            "json",
        }

    def test_parse_formats_unknown_raises(self) -> None:
        from novomodelo_bridge.cli.args import _parse_formats

        with pytest.raises(ValueError, match="bogus"):
            _parse_formats(["bogus"])
