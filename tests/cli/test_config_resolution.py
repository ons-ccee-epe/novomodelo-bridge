"""Unit tests for the leaf config-resolution layer (``novomodelo_bridge.cli.config``).

Every test isolates discovery from the real environment: the cwd chain is driven
through ``start=`` (or ``monkeypatch.chdir``) and ``XDG_CONFIG_HOME`` / ``HOME``
are monkeypatched under ``tmp_path``. The real ``~/.config`` is never touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from novomodelo_bridge.cli.config import (
    BridgeConfig,
    discover_config_path,
    load_config,
)


def _isolate_discovery_env(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    """Point discovery's env lookups at *home* and unset ``XDG_CONFIG_HOME``.

    Keeps the user-level fallbacks inside the test sandbox so an empty cwd chain
    cannot accidentally resolve a real ``~/.config`` file.
    """
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", str(home))


def test_no_file_returns_empty_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An empty cwd chain plus a home with no config dir: nothing is discovered.
    empty_dir = tmp_path / "work"
    empty_dir.mkdir()
    _isolate_discovery_env(monkeypatch, tmp_path / "home")

    result = load_config(start=empty_dir)

    assert result == BridgeConfig()
    assert result.results_tolerance is None
    assert result.formats is None
    assert result.out_dir is None
    assert result.source_path is None
    assert result.warnings == ()


def test_local_toml_populates_all_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "novomodelo-bridge.toml"
    config_path.write_text(
        "[compare.results]\n"
        "tolerance = 2e-2\n"
        "[compare]\n"
        'format = ["console", "html"]\n'
        'out_dir = "art"\n',
        encoding="utf-8",
    )

    result = load_config(start=tmp_path)

    assert result.results_tolerance == 2e-2
    assert result.formats == ("console", "html")
    assert result.out_dir == Path("art")
    assert result.source_path == config_path
    assert result.warnings == ()


def test_malformed_toml_warns_and_returns_none_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "novomodelo-bridge.toml"
    config_path.write_text("not = valid = toml = here\n", encoding="utf-8")

    result = load_config(start=tmp_path)

    assert result.results_tolerance is None
    assert result.formats is None
    assert result.out_dir is None
    assert result.source_path == config_path
    assert len(result.warnings) == 1


def test_wrong_typed_value_skips_only_that_key(tmp_path: Path) -> None:
    config_path = tmp_path / "novomodelo-bridge.toml"
    config_path.write_text(
        '[compare.results]\ntolerance = "loose"\n[compare]\nout_dir = "art"\n',
        encoding="utf-8",
    )

    result = load_config(start=tmp_path)

    assert result.results_tolerance is None
    assert result.out_dir == Path("art")
    assert result.source_path == config_path
    assert len(result.warnings) == 1
    assert "results" in result.warnings[0]


def test_empty_out_dir_warns_and_is_treated_as_absent(tmp_path: Path) -> None:
    # An empty out_dir must NOT silently become Path(".") (cwd); it is dropped
    # with a warning so the derived <novomodelo_output_dir>/comparison_artifacts wins.
    config_path = tmp_path / "novomodelo-bridge.toml"
    config_path.write_text('[compare]\nout_dir = "   "\n', encoding="utf-8")

    result = load_config(start=tmp_path)

    assert result.out_dir is None
    assert len(result.warnings) == 1
    assert "out_dir" in result.warnings[0]


def test_discovery_falls_back_to_home_config_when_xdg_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    user_config = home / ".config" / "novomodelo-bridge" / "config.toml"
    user_config.parent.mkdir(parents=True)
    user_config.write_text("[compare.bounds]\ntolerance = 1e-5\n", encoding="utf-8")
    _isolate_discovery_env(monkeypatch, home)

    empty_dir = tmp_path / "work"
    empty_dir.mkdir()

    assert discover_config_path(start=empty_dir) == user_config


def test_discovery_prefers_cwd_over_xdg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A local novomodelo-bridge.toml in the cwd chain must win over an XDG file.
    local_config = tmp_path / "novomodelo-bridge.toml"
    local_config.write_text("[compare.bounds]\ntolerance = 1e-3\n", encoding="utf-8")

    xdg_home = tmp_path / "xdg"
    xdg_config = xdg_home / "novomodelo-bridge" / "config.toml"
    xdg_config.parent.mkdir(parents=True)
    xdg_config.write_text("[compare.bounds]\ntolerance = 9e-9\n", encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_home))

    assert discover_config_path(start=tmp_path) == local_config
