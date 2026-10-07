"""Tests for the cobre-compat version-gate policy. Tier-1: pure Python, no
``cobre`` import."""

from __future__ import annotations

import pytest

from cobre_bridge.cobre.compat import (
    MIN_COBRE_VERSION,
    _cobre_python_supports_output,
    _installed_cobre_python_version,
)


def test_min_cobre_version_value() -> None:
    assert MIN_COBRE_VERSION == "0.18.0"
    assert isinstance(MIN_COBRE_VERSION, str)


def test_supports_output_at_the_floor() -> None:
    assert _cobre_python_supports_output("0.18.0") is True


def test_supports_output_above_the_floor() -> None:
    assert _cobre_python_supports_output("0.18.1") is True


def test_supports_output_below_the_floor() -> None:
    assert _cobre_python_supports_output("0.17.0") is False


def test_supports_output_pre_release_suffix_is_lenient() -> None:
    assert _cobre_python_supports_output("0.18.0rc1") is True


def test_installed_version_returns_the_distribution_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("importlib.metadata.version", lambda name: "0.14.3")
    assert _installed_cobre_python_version() == "0.14.3"


def test_installed_version_returns_none_when_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from importlib.metadata import PackageNotFoundError

    def _raise(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr("importlib.metadata.version", _raise)
    assert _installed_cobre_python_version() is None
