# Novomodelo-Bridge — Agent Guidelines

novomodelo-bridge is a Python CLI that converts NEWAVE and DECOMP cases into the
input format of Novomodelo (the Rust SDDP solver at https://github.com/ons-ccee-epe/novomodelo)
and compares the two models' results. The human-facing documents are the
source of truth for what the tool does and how the code is laid out; this file
adds only what an agent working in this checkout needs beyond them.

## Read first

- `README.md` — what the tool does, quick start, commands, exit codes, config.
- `docs/architecture.md` — package map, import direction, twin-track spine,
  end-to-end data flow, extension points.
- `docs/newave.md`, `docs/decomp.md` — per-track behaviour and caveats.
- `CONTRIBUTING.md` — dev setup, test tiers, gates, conventions, releasing.
- `docs/cli.md` — generated from the Typer app; never hand-edit it.
- `docs/newave-data-map.md`, `docs/decomp-data-map.md` — generated (pt-BR)
  maps from deck file, record, and column to each converted field, plus what
  is not converted yet; edit `docs/lineage/*.toml`, never the pages.

## Quick commands

```bash
pip install -e ".[dev]"                      # or: uv sync --extra dev
pytest                                       # suite; tier-2 tests skip without novomodelo-python
ruff check . && ruff format --check .        # whole tree, as CI and the pre-commit hook run it
python3 scripts/ci/check_no_plan_leaks.py    # hard gates, also run by CI
python3 scripts/ci/check_comment_refs.py
python3 scripts/ci/check_doc_paths.py
scripts/gen-cli-docs.sh                      # regenerate docs/cli.md after any help-text change
scripts/gen-lineage-docs.py                  # regenerate docs/*-data-map.md after editing docs/lineage/*.toml
scripts/regen-goldens.sh                     # regenerate tests/golden/, then review the diff
ln -sf ../../scripts/pre-commit .git/hooks/pre-commit
```

## Rules

Path-scoped rules in `.claude/rules/` load automatically when a matching file
is read and are the authoritative statements; do not restate them here:

- `.claude/rules/comments.md` (`**/*.py`) — comment and docstring discipline.
- `.claude/rules/doc-integrity.md` (`*.md`) — prose docs: no hand-frozen
  counts or versions, every cited path resolves, sober voice.
- `.claude/rules/testing.md` (`tests/`) — the three test tiers, deck guards,
  mock discipline, golden files.
- `.claude/rules/bridge.md` (`src/`) — twin-track symmetry, the tracked
  novomodelo-gap protocol, presentation and I/O boundaries, exit-code and `--json`
  contracts.

Hard rules that are fixed before any commit:

- No planning vocabulary (`epic`, `ticket`, `sprint`, plan ids) in `src/`,
  `tests/` prose, `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, or `docs/`.
  Plans live in the gitignored `plans/`. Commit messages may reference them.
- No rot-prone references in shipped prose: no `file.py:NNN`, no `MEMORY.md`,
  no machine-local (`~/git/...`) or gitignored (`plans/`) paths. Reference
  symbols, named tests, or stable external anchors.
- Twin-track symmetry: behaviour added to one conversion track lands on both
  in the same change, or the asymmetry is recorded in the architecture-debt
  registry under `plans/`. Never import an underscore-private name across the
  `newave/` ↔ `decomp/` boundary.
- Never silently work around a novomodelo limitation: log or emit a diagnostic,
  mark the site `TRACKED NOVOMODELO-GAP (Cn)`, and register the removal condition
  in the novomodelo repository. Those comments are protected in cleanup passes.
- Comments are default-off: a comment ships only if it survives the Deletion
  Test; never delete or weaken a load-bearing unit or direction contract.

## Local decks

`example/` is gitignored. It holds real NEWAVE and DECOMP decks and their
converted-and-solved Novomodelo cases for manual runs only. No test reads it and no
test reaches outside the repository (`tests/test_local_data_policy.py`
enforces both); real-format inputs are committed as small excerpts under
`tests/fixtures/` and synthetic mini-decks under `tests/decks/`. End-to-end
checks that need a whole real deck are catalogued in `docs/real-deck-checks.md`.
When exercising a real deck by hand, convert into a fresh directory and run
novomodelo before comparing so an existing run's `output/` is not clobbered.

A local novomodelo checkout, when present, is the reference for the input contract
(serde structs and schemas); a bridge release `X.Y.Z` pairs with novomodelo
`X.Y.Z`, and the `novomodelo-python` pin in `pyproject.toml` is exactly
`MIN_NOVOMODELO_VERSION`.
