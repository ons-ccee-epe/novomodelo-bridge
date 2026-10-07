# `scripts/`

Helper scripts for novomodelo-bridge, grouped by role. `CONTRIBUTING.md` says when
to run them in a normal development loop.

- **`ci/`** — quality gates and advisory reports (see below). The gates run in
  CI (`.github/workflows/ci.yml`) and from the `pre-commit` hook; every script
  locates the repo root from its own path, so they run from anywhere.
- **`pre-commit`** — the git pre-commit hook. Install with
  `ln -sf ../../scripts/pre-commit .git/hooks/pre-commit`.
- **`gen-cli-docs.sh`** — regenerates `docs/cli.md` from the Typer app
  (content-guarded by `tests/test_docs.py`; never hand-edit the output).
- **`gen-lineage-docs.py`** — renders `docs/newave-data-map.md` and
  `docs/decomp-data-map.md` (pt-BR) from `docs/lineage/*.toml`; `--check`
  validates the TOML, reports a stale page, and cross-checks the outputs'
  declared sources against an AST trace of the pipelines. `tests/test_lineage.py`
  runs the same checks plus emission coverage against the mini decks. The
  loader, tracer, and renderer live in `lineage/`.
- **`regen-goldens.sh`** — regenerates the `tests/golden/` snapshots by running
  their consumer tests with `NOVOMODELO_BRIDGE_UPDATE_GOLDENS=1`. Goldens are
  regenerated via this path, never hand-edited; review the resulting
  `git diff tests/golden/` before committing.
- **`analyze_results.py`** — local analysis utility, not a gate.

## `ci/` — quality gates

Gates marked _advisory_ never fail the build; the rest exit 1 on violation.
The conventions each gate enforces live in `.claude/rules/` (comments,
doc-integrity, testing, bridge contracts).

| Script                   | Purpose                                                                                                                      |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| `check_no_plan_leaks.py` | No plan-structure vocabulary in shipped docs (hard) + src/ prose burndown count (advisory). `--all` lists it.                |
| `check_comment_refs.py`  | No `file.py:NNN` / `MEMORY.md` / private-tooling refs in src/ prose (hard); machine-local + gitignored path refs (advisory). |
| `check_doc_paths.py`     | Repo-relative paths cited in README/CONTRIBUTING/docs resolve against the tree (hard); CLAUDE.md (hard).                     |
| `check_comment_bloat.py` | _Advisory_: long comment blocks + repeated clauses, candidates for a comment-skeptic pass.                                   |
| `quality_report.py`      | _Advisory_: churn/size/long-fn/suppression/plan-token hotspot ranking.                                                       |
| `_scan.py`               | Shared comment/docstring extraction (tokenize + ast), imported by the gates.                                                 |
