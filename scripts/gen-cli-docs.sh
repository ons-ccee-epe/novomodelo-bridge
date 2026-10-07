#!/usr/bin/env bash
# Regenerate docs/cli.md from the Typer app. Dev/build-time only.
# Target novomodelo_bridge.cli.app (where `app` is constructed), not the novomodelo_bridge.cli
# package: typer's `utils docs` driver re-execs the target and mutates the app's
# `_add_completion`; the package re-exports a cached `app`, so pointing at it drops
# the completion flags from the output. cli.app re-execs a fresh app each time.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-python}"

"$PYTHON" -m typer novomodelo_bridge.cli.app utils docs \
    --name novomodelo-bridge \
    --output "$REPO_ROOT/docs/cli.md"

echo "Regenerated $REPO_ROOT/docs/cli.md" >&2
