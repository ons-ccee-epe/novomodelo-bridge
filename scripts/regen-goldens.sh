#!/usr/bin/env bash
# Regenerate the tests/golden/ snapshots for the golden-file test consumers.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PYTEST="${PYTEST:-.venv/bin/pytest}"

cd "$REPO_ROOT"
NOVOMODELO_BRIDGE_UPDATE_GOLDENS=1 "$PYTEST" \
    tests/comparators/test_golden_dataset.py \
    tests/dashboard/test_chart_helpers.py \
    tests/comparators/test_report_builder.py \
    -q

echo "Goldens regenerated -- review 'git diff tests/golden/' before committing." >&2
