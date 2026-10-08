"""Adapter for validating a converted case with the installed novomodelo-python.

Shared by every ``convert *`` command's ``--validate`` gate; calls the version
policy through the :mod:`novomodelo_bridge.novomodelo.compat` module object (rather than
importing its functions/constant by name) so a test's
``patch("novomodelo_bridge.novomodelo.compat._installed_novomodelo_python_version", ...)``
reaches the same object this module looks up at call time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from novomodelo_bridge.novomodelo import compat as novomodelo_compat
from novomodelo_bridge.ui.console import print_status, render_error

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from rich.console import Console


def _validation_message(item: object) -> str:
    """Extract the display text from a ``novomodelo.io.validate`` warning/error item.

    Each item is either a plain string or a ``{"message": ...}`` dict; both
    forms render and partition on the same text.
    """
    text = item.get("message", item) if isinstance(item, dict) else item
    return str(text)


def _partition_validation_warnings(
    warnings: Sequence[object], whitelist_substrings: Sequence[str] = ()
) -> tuple[list[object], list[object]]:
    """Split ``novomodelo.io.validate`` ``warnings`` into ``(rendered, whitelisted)``.

    A warning whose message (:func:`_validation_message`) contains any of
    *whitelist_substrings* is whitelisted — not rendered as a "Validation
    warning" and not counted in the rendered/blocking warning count; every
    other warning goes to *rendered* exactly as before. An empty
    *whitelist_substrings* — what ``convert newave`` passes — is the identity
    partition: ``rendered == warnings`` and ``whitelisted == []``. Pure; never
    touches ``errors``, and never suppresses a non-matching warning.
    """
    if not whitelist_substrings:
        return list(warnings), []

    rendered: list[object] = []
    whitelisted: list[object] = []
    for warning in warnings:
        message = _validation_message(warning)
        if any(substring in message for substring in whitelist_substrings):
            whitelisted.append(warning)
        else:
            rendered.append(warning)
    return rendered, whitelisted


def _run_novomodelo_validation(
    dst: Path,
    *,
    command: str,
    summary: dict[str, object],
    json_output: bool,
    err_console: Console,
    whitelist_substrings: Sequence[str] = (),
) -> bool:
    """Validate *dst* with the installed novomodelo-python and render the outcome.

    Shared by every ``convert *`` command's ``--validate`` gate: the
    :data:`~novomodelo_bridge.novomodelo.compat.MIN_NOVOMODELO_VERSION` skip, the
    ``novomodelo.io.validate`` call, warning/error rendering (warnings are first
    partitioned through :func:`_partition_validation_warnings` against
    *whitelist_substrings* — ``convert newave`` passes an empty tuple, the
    identity case, so its rendering stays byte-identical), and the
    machine-readable ``summary["validation"]`` sub-object, populated only
    when *json_output* is set. *command* names the caller in the
    whitelisted-note message; *summary* is mutated in place.

    Returns whether validation FAILED (``valid`` came back ``False``, or
    ``novomodelo.io.validate`` itself raised) so the caller can flip its exit code
    to 2 — a skipped validation (old/absent novomodelo-python) is never a failure.
    Does not emit the enclosing ``--json`` verdict or raise ``typer.Exit``;
    that stays the caller's job, run immediately after this returns.
    """
    installed = novomodelo_compat._installed_novomodelo_python_version()
    if (
        installed is not None
        and not novomodelo_compat._novomodelo_python_supports_output(installed)
    ):
        print_status(
            f"Note: converted output requires novomodelo-python >= "
            f"{novomodelo_compat.MIN_NOVOMODELO_VERSION} (installed novomodelo-python {installed} is "
            f"older); skipping novomodelo-python validation.",
            console=err_console,
            style="#F5A623",
        )
        if json_output:
            summary["validation"] = {
                "ran": False,
                "valid": None,
                "warnings": 0,
                "errors": 0,
                "skipped_reason": "novomodelo-python-too-old",
            }
        return False

    try:
        import novomodelo.io  # type: ignore[import-untyped]
    except ImportError:
        print_status(
            "Warning: novomodelo package not installed, skipping validation",
            console=err_console,
            style="#F5A623",
        )
        if json_output:
            # Validation was requested but could not run; record that it was
            # skipped so the absence of a real outcome is explicit.
            summary["validation"] = {
                "ran": False,
                "valid": None,
                "warnings": 0,
                "errors": 0,
            }
        return False

    try:
        # novomodelo v0.6.x: novomodelo.io.validate is a function returning a
        # report dict; it never raises (errors are surfaced as data).
        result = novomodelo.io.validate(str(dst))
    except Exception as exc:  # noqa: BLE001
        render_error(f"Validation error: {exc}", console=err_console)
        if json_output:
            # Validation raised unexpectedly; still emit one JSON object so
            # the --json contract (exactly one verdict on stdout) holds on
            # this exit-2 path too. The conversion itself succeeded, so the
            # summary is intact; only the validation outcome is an error.
            summary["validation"] = {
                "ran": False,
                "valid": None,
                "warnings": 0,
                "errors": 1,
            }
        return True

    raw_warnings = result.get("warnings", [])
    errors = result.get("errors", [])
    valid = bool(result.get("valid", False))

    rendered_warnings, whitelisted_warnings = _partition_validation_warnings(
        raw_warnings, whitelist_substrings
    )

    for warning in rendered_warnings:
        print_status(
            f"Validation warning: {_validation_message(warning)}",
            console=err_console,
            style="#F5A623",
        )
    if whitelisted_warnings:
        # Whitelisted-but-present is worth one INFO note, never a WARNING —
        # this is what tells the whitelist apart from a real suppression.
        print_status(
            f"Note: {len(whitelisted_warnings)} validation warning(s) matched "
            "the expected external-solver-interop configuration for "
            f"{command}; not rendered.",
            console=err_console,
        )

    validation_failed = False
    if not valid:
        for err in errors:
            print_status(
                f"Validation error: {_validation_message(err)}",
                console=err_console,
                style="bold #DC4C4C",
            )
        print_status("Validation failed.", console=err_console, style="bold #DC4C4C")
        validation_failed = True

    if json_output:
        # The machine-readable outcome under ``summary``; ``status`` stays
        # derived from diagnostics only (validation never flips it).
        summary["validation"] = {
            "ran": True,
            "valid": valid,
            "warnings": len(rendered_warnings),
            "errors": len(errors),
        }

    return validation_failed
