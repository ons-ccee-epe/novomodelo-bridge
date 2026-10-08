---
paths:
  - "tests/**/*.py"
---

# Novomodelo-Bridge Testing Rules

The standing contract for how this repo is tested. Generic Python testing
style lives in the global Python rules; these are the bridge-specific rules.

## The 2-tier convention (load-bearing — CI depends on it)

- **Tier 1 — pure Python.** Imports no `novomodelo`. Runs on every CI job. No
  module at any tier may `import novomodelo` at module scope — the suite must
  *collect* cleanly in a novomodelo-free environment.
- **Tier 2 — needs `novomodelo-python`** (a required runtime dep) but no solver
  binary. Guarded by `tests.conftest.requires_novomodelo_python`
  (`find_spec`-based) and, where relevant, `requires_writer_binding`; any
  `import novomodelo` lives inside the guarded test/helper body.

## Local-data policy (enforced by `tests/test_local_data_policy.py`)

Every test runs from the repository alone:

- **No test builds a path into the gitignored `example/` tree, and none
  resolves a path through `Path.home()`.** A test that runs on one developer's
  machine only is a snapshot, not a guard, and its permanent skip elsewhere
  reads as coverage. There is no tier 3.
- **Real-format inputs are committed as small excerpts** under
  `tests/fixtures/` (result-file heads, a schema, one example case — a few KB
  each); synthetic decks live under `tests/decks/`. Name the excerpt's origin
  in the module docstring.
- **A check that genuinely needs a whole deck, a solved novomodelo case, or the
  solver binary** is written up in `docs/real-deck-checks.md` (what it
  verified, what it needs) instead of being kept as a skipped test.

## Test architecture

- **Mock discipline: patch public entry points only.** Patching an
  underscore-private symbol couples the test to an implementation detail;
  the repo's baseline is ~5 private patches out of ~300 — keep it there.
  Pipeline tests mock converter entry points; CLI error-path tests run as
  subprocess to verify real exit codes.
- **One home per source module.** New converter/comparator tests go in a
  `test_<module>.py` named for the source module — do not grow the mega-files
  (`test_entity_conversion.py`, `test_decomp_results_compare.py`, `test_cli.py`
  are legacy exceptions scheduled for splitting, not a pattern to extend).
- **Shared case builders live in `tests/conftest.py`** (or a shared fixtures
  module) — never re-defined per file.
- **Test output goes to `tmp_path`.** No test writes into the repo tree;
  generated artifacts self-delete.
- **Golden files** (`tests/golden/`) pin rendered HTML/dataset shapes. A
  deliberate rendering change regenerates them via the documented regeneration
  path — never hand-edit a golden file, and never loosen an assertion to avoid
  regenerating.
- **CliRunner + Rich:** Typer help/rendering tests run under a dumb terminal
  (`TERM=dumb` fixture) — Rich box-drawing output differs across CI terminals
  and versions.
- **Logger state:** an autouse fixture snapshots/restores the `novomodelo_bridge`
  logger so a CliRunner invocation cannot leak logging state into a later
  `caplog` test — new tests that reconfigure logging must go through it.

## What a new feature must ship with

- A tier-1 test for the pure conversion/analysis logic (synthetic inputs).
- For a new CLI flag or exit-code path: a CliRunner or subprocess test
  asserting the exit code and, when `--json` is involved, the verdict envelope.
- For a new output artifact: an emission-shape test (schema/columns), mirrored
  on **both** tracks when the artifact exists on both (see
  `.claude/rules/bridge.md` — twin-track symmetry).
