"""CLI subprocess runner used only by tests/cli/."""

from __future__ import annotations

import subprocess
import sys


def _run_cli_subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the novomodelo-bridge entry point as a real subprocess."""
    return subprocess.run(
        [sys.executable, "-m", "novomodelo_bridge.cli", *args],
        capture_output=True,
        text=True,
    )
