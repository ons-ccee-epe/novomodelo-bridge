"""CLI subprocess runner used only by tests/cli/."""

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
