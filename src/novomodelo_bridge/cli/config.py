"""Leaf config-resolution layer: discover and parse ``cobre-bridge.toml``.

This module supplies the *file-sourced* defaults for the compare commands —
the absolute/relative tolerances, the output format set, and the artifact
output directory — as a typed, frozen :class:`BridgeConfig`. It sits at the
bottom of the precedence chain **CLI flag > env var > config file > built-in
default**: it discovers a single TOML file and reads the recognised keys,
exposing the four built-in defaults as module constants for the CLI to fall
back on.

It is a pure leaf. It imports nothing from ``cli``, Typer, ``ui``,
``diagnostics``, or any converter, and it consults the environment **only** for
file discovery (``XDG_CONFIG_HOME`` / ``HOME``) — never for tolerances, format,
or out-dir, which Typer's ``envvar=`` handles in the CLI layer. A malformed
file, an unreadable file, or a wrong-typed value never raises: the problem is
recorded as a human-readable string in :attr:`BridgeConfig.warnings` and the
affected field is left ``None`` so the CLI falls through to its next source.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

RESULTS_TOLERANCE_DEFAULT: float = 1e-2
FORMAT_DEFAULT: tuple[str, ...] = ("console", "parquet", "json")
# out-dir has no fixed default (it derives from <cobre_output_dir>); the config
# value, when absent, is None and the CLI keeps its derived default.

_LOCAL_CONFIG_NAME = "cobre-bridge.toml"
_USER_CONFIG_RELPATH = ("cobre-bridge", "config.toml")


@dataclass(frozen=True)
class BridgeConfig:
    """File-sourced compare defaults, each ``None`` when absent from the file.

    A ``None`` value field means "the config file did not set this key" (so the
    CLI can distinguish that from "the file set it to a value"); a parse or type
    problem is reported in :attr:`warnings` while the offending field stays
    ``None``.

    Attributes:
        results_tolerance: ``[compare.results] tolerance`` (relative), or ``None``.
        formats: ``[compare] format`` normalised to the raw token tuple, or
            ``None``. Token names are NOT validated here.
        out_dir: ``[compare] out_dir`` wrapped in a :class:`Path`, or ``None``.
        source_path: The file these values came from, set whenever a file was
            located (even if it failed to parse); ``None`` when none was found.
        warnings: Human-readable load warnings (malformed TOML, bad types).
    """

    results_tolerance: float | None = None
    formats: tuple[str, ...] | None = None
    out_dir: Path | None = None
    source_path: Path | None = None
    warnings: tuple[str, ...] = ()


def discover_config_path(start: Path | None = None) -> Path | None:
    """Return the first existing config file on the search chain, or ``None``.

    First-found-wins, in this exact order:

    1. ``start/cobre-bridge.toml`` then each ancestor of ``start`` up to and
       including the filesystem root (``start`` defaults to :func:`Path.cwd`).
    2. ``$XDG_CONFIG_HOME/cobre-bridge/config.toml`` when ``XDG_CONFIG_HOME`` is
       set and non-empty.
    3. ``~/.config/cobre-bridge/config.toml`` (via :func:`Path.home`), the
       ``XDG_CONFIG_HOME``-unset fallback and final lookup.

    Existence is tested with :meth:`Path.is_file`; no file is read here.
    """
    base = Path.cwd() if start is None else start
    for directory in (base, *base.parents):
        candidate = directory / _LOCAL_CONFIG_NAME
        if candidate.is_file():
            return candidate

    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        candidate = Path(xdg).joinpath(*_USER_CONFIG_RELPATH)
        if candidate.is_file():
            return candidate

    candidate = Path.home().joinpath(".config", *_USER_CONFIG_RELPATH)
    if candidate.is_file():
        return candidate

    return None


def load_config(path: Path | None = None, *, start: Path | None = None) -> BridgeConfig:
    """Parse a single ``cobre-bridge.toml`` into a :class:`BridgeConfig`.

    When ``path`` is ``None`` the file is located via
    :func:`discover_config_path` (passing ``start``); if discovery also returns
    ``None`` an empty :class:`BridgeConfig` (all-``None``, no warnings,
    ``source_path=None``) is returned. There is NO merge across files — the
    chosen single file is parsed in full.

    A :class:`tomllib.TOMLDecodeError` or :class:`OSError` aborts the parse and
    yields all-``None`` value fields, a single warning, and
    ``source_path=path``. A wrong-typed value for a recognised key records one
    warning and skips only that key, keeping the other valid keys. Unknown
    tables and keys are ignored silently. Nothing is written and nothing raises.
    """
    if path is None:
        path = discover_config_path(start)
        if path is None:
            return BridgeConfig()

    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as err:
        return BridgeConfig(
            source_path=path,
            warnings=(f"Ignoring malformed config file {path}: {err}",),
        )
    except OSError as err:
        return BridgeConfig(
            source_path=path,
            warnings=(f"Ignoring unreadable config file {path}: {err}",),
        )

    warnings: list[str] = []
    compare = _read_table(data, "compare")
    results = _read_table(compare, "results")

    return BridgeConfig(
        results_tolerance=_read_number(
            results, "tolerance", "compare.results", warnings
        ),
        formats=_read_formats(compare, "format", warnings),
        out_dir=_read_path(compare, "out_dir", "compare", warnings),
        source_path=path,
        warnings=tuple(warnings),
    )


def _read_table(table: Mapping[str, object], key: str) -> Mapping[str, object]:
    """Return ``table[key]`` if it is itself a mapping, else an empty mapping.

    A non-table value where a table is expected is treated as absent (an empty
    mapping), so the recognised leaf keys simply fall through to ``None`` rather
    than raising or producing a warning here.
    """
    value = table.get(key)
    if isinstance(value, Mapping):
        return value
    return {}


def _read_number(
    table: Mapping[str, object], key: str, where: str, warnings: list[str]
) -> float | None:
    """Read a required-number key, coercing ``int``/``float`` to ``float``.

    Returns ``None`` when the key is absent (no warning). A value of the wrong
    type appends one warning naming ``where`` and returns ``None``. A TOML
    ``bool`` is rejected as a number — it is an ``int`` subclass — so ``True``
    is a type mismatch, not the number ``1``.
    """
    if key not in table:
        return None
    value = table[key]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    warnings.append(
        f"Ignoring [{where}] {key}: expected a number, got {type(value).__name__}."
    )
    return None


def _read_path(
    table: Mapping[str, object], key: str, where: str, warnings: list[str]
) -> Path | None:
    """Read a required-string key and wrap it in a :class:`Path`.

    Returns ``None`` when the key is absent (no warning). A non-string value, or
    an empty/whitespace-only string, appends one warning naming ``where`` and
    returns ``None`` — an empty ``out_dir`` is treated as absent rather than
    silently becoming ``Path(".")`` (the current directory).
    """
    if key not in table:
        return None
    value = table[key]
    if isinstance(value, str):
        if value.strip():
            return Path(value)
        warnings.append(f"Ignoring [{where}] {key}: value is empty.")
        return None
    warnings.append(
        f"Ignoring [{where}] {key}: expected a string, got {type(value).__name__}."
    )
    return None


def _read_formats(
    table: Mapping[str, object], key: str, warnings: list[str]
) -> tuple[str, ...] | None:
    """Read the ``format`` key as a token tuple without validating token names.

    Accepts a single string (``"console"`` → ``("console",)``) or a list of
    strings (normalised to a tuple of the raw tokens). Returns ``None`` when the
    key is absent (no warning). Any other shape — a non-string scalar, or a list
    holding a non-string element — appends one warning and returns ``None``.
    Token names are passed through verbatim; the CLI validates them.
    """
    if key not in table:
        return None
    value = table[key]
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    warnings.append(
        f"Ignoring [compare] {key}: expected a string or a list of strings, "
        f"got {type(value).__name__}."
    )
    return None
