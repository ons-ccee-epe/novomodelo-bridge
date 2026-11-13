"""CLI subprocess runner and output helpers used only by tests/cli/."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping


def _run_cli_subprocess(
    *args: str, env: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Invoke the novomodelo-bridge entry point as a real subprocess, with
    *env* layered over the current environment."""
    return subprocess.run(
        [sys.executable, "-m", "novomodelo_bridge.cli", *args],
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, **(env or {})},
    )


def _panel_text(rendered: str) -> str:
    """*rendered* with Rich's vertical panel borders dropped and every run of
    whitespace collapsed to one space.

    Non-TTY output renders at a fixed width, so where a message wraps inside a
    panel moves with the length of the temp path it names; match phrases
    against this text, not the raw lines.
    """
    return " ".join(rendered.replace("│", " ").split())
