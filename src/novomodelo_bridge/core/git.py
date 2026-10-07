"""Tiny internal git helper shared by the provenance-manifest writers.

Lives at the package root (rather than inside ``comparators``) because both
the comparison manifest (:mod:`cobre_bridge.comparators.manifest`) and the
conversion manifest (:mod:`cobre_bridge.cli.conversion_manifest`) record the git
SHA, and neither should depend on the other's private internals. The git
subprocess runs only when :func:`git_sha` is called — never at import time.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def git_sha() -> str | None:
    """Return the bridge package's short git HEAD SHA, or ``None``.

    Resolved from the package's own location (``Path(__file__).parent``), not
    the caller's cwd — a pip-installed bridge run from inside a user's
    git-tracked case directory must not record that repo's SHA as provenance.
    Runs ``git rev-parse --short HEAD`` with ``check=False`` so a non-git
    directory (non-zero return code, e.g. a normal wheel install) yields
    ``None`` rather than raising. Only :class:`FileNotFoundError` /
    :class:`OSError` (git binary missing or not executable) are caught; all
    other errors propagate.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()
