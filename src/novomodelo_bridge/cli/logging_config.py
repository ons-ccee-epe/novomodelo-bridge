"""CLI logging configuration: the verbose ladder, ``--log-file``, and teardown."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from pathlib import Path

#: A no-op handler parked on the package logger when warnings are suppressed, so a
#: suppressed record does not fall through to ``logging.lastResort`` (which would
#: otherwise echo it to stderr, defeating the suppression).
NULL_HANDLER = logging.NullHandler()

#: The ``--log-file`` DEBUG ``FileHandler`` attached to the package logger for the
#: duration of a run, or ``None`` when no ``--log-file`` was given.
#: :func:`restore_log_file_handler` removes and closes it so it never leaks across
#: in-process invocations (the autouse test fixture restores ``propagate``/level
#: but NOT handlers).
_LOG_FILE_HANDLER: logging.FileHandler | None = None


def configure_logging(verbose: int, log_file: Path | None) -> None:
    """Configure logging for a CLI run.

    *verbose* is a graduated count selecting the live console level:
    ``0`` keeps ``cobre_bridge`` warnings recorded — the diagnostics collector and
    ``--diagnostics-json`` rely on them — but off the live console (the Rich
    diagnostics block is the single user-facing surface, and warnings are not
    printed twice); ``1`` (``-v`` / ``--verbose``) raises the console to INFO; and
    ``2`` or more (``-vv``) raises it to DEBUG.

    This is a behavior change from the previous boolean ``--verbose``: a bare
    ``--verbose`` used to mean DEBUG and now means INFO; ``-vv`` is required for the
    full DEBUG firehose. ``--log-file`` (below) gives the complete DEBUG trace to
    anyone who needs it regardless of the console level.

    When *log_file* is not ``None``, its missing parent directories are created and
    a DEBUG ``FileHandler`` is attached to the
    ``cobre_bridge`` logger and the package logger level is lowered to DEBUG, so the
    file always captures the full trace even at console verbose ``0`` (the console
    output stays at the ladder level — it is driven by ``basicConfig``/root, while
    ``NULL_HANDLER`` keeps suppressed records off ``logging.lastResort``). The
    created handler is stored in the module-level ``_LOG_FILE_HANDLER`` so
    :func:`restore_log_file_handler` can remove and close it; ``main`` also restores
    ``propagate`` afterwards.
    """
    global _LOG_FILE_HANDLER

    pkg = logging.getLogger("cobre_bridge")
    if verbose >= 1:
        level = logging.DEBUG if verbose >= 2 else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(levelname)s %(name)s: %(message)s",
        )
        pkg.setLevel(level)
        pkg.propagate = True
        # Symmetric with the suppress branch below: drop the null handler so a
        # verbose run after a suppressed one in the same process leaves the
        # package logger in a clean state (removeHandler is a no-op if absent).
        pkg.removeHandler(NULL_HANDLER)
    else:
        # Leave the package logger level untouched (root's default WARNING already
        # records warnings for the collector); just keep them off the live console.
        if NULL_HANDLER not in pkg.handlers:
            pkg.addHandler(NULL_HANDLER)
        pkg.propagate = False

    if log_file is not None:
        # Runs before any command builds its arguments, so an unwritable path is
        # reported as an option error rather than escaping as a traceback.
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(log_file, encoding="utf-8")
        except OSError as exc:
            raise typer.BadParameter(
                f"cannot write {log_file}: {exc.strerror or exc}",
                param_hint="'--log-file'",
            ) from exc
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        pkg.addHandler(handler)
        # Lower the logger threshold so DEBUG records reach the file handler even
        # when the console ladder leaves it at verbose 0.
        pkg.setLevel(logging.DEBUG)
        _LOG_FILE_HANDLER = handler


def restore_log_file_handler(logger: logging.Logger) -> None:
    """Undo :func:`configure_logging`'s ``--log-file`` handler: remove, close, forget.

    A no-op when no ``--log-file`` was given (``_LOG_FILE_HANDLER`` stays
    ``None``). Called from ``cli.main``'s ``finally`` so a real CLI run never
    leaks a ``FileHandler`` across process invocations.
    """
    global _LOG_FILE_HANDLER

    if _LOG_FILE_HANDLER is not None:
        logger.removeHandler(_LOG_FILE_HANDLER)
        _LOG_FILE_HANDLER.close()
        _LOG_FILE_HANDLER = None
