---
paths:
  - "src/novomodelo_bridge/**/*.py"
---

# Novomodelo-Bridge Conversion & CLI Contracts

Repo-specific architectural contracts. Each is a contract, not a style
preference — a plausible deviation ships wrong numbers, silent divergence
between the two conversion tracks, or messages a pip-installed user cannot act
on. The known standing architectural violations have been catalogued and
remediated; do not introduce new ones.

## 1. Twin-track symmetry (the central contract)

The NEWAVE track (`newave/`, spine + `newave/converters/` +
`newave/pipeline.py`) and the DECOMP track (`decomp/`, spine +
`decomp/converters/` + `decomp/pipeline.py`) must honour the **same
contracts** for the same operations:

- A behaviour added to one track — a CLI flag, an exit-code rule, an emission
  self-check, a `--force`/rollback rule, a verdict field, a preflight policy —
  lands on **both tracks in the same change**, or the asymmetry is recorded as
  a finding in the audit registry. Silent one-track divergence is the repo's
  #1 recorded debt source; never extend it.
- **Never import an underscore-private name across the `newave/` ↔ `decomp/`
  track boundary.** When both tracks need it, promote it to `core/` (the
  shared foundation both tracks may import) — enforced by
  `tests/test_package_boundaries.py` (Rule B,
  `test_no_private_cross_package_imports`) — the way `decomp/preflight.py`
  reuses the public `CheckItem`/`PreflightResult`.
- Physics and calendar math (cota polynomials, productivity, stage/block
  weighting) live in **one** shared implementation (in `core/`). Two
  implementations that "agree except at the edges" is exactly the bug class
  `compare` exists to detect — in our own tool.

## 2. TRACKED NOVOMODELO-GAP workarounds

Never silently adopt a bridge-side workaround for a novomodelo limitation:

1. Emit a log/diagnostic at the workaround site.
2. Mark the site with a `TRACKED NOVOMODELO-GAP (Cn)` comment.
3. Register the gap with its **removal condition** in novomodelo's
   `conversion-found-improvements` registry (in the novomodelo repo).

These comments are **protected contracts** (`.claude/rules/comments.md` §4):
never delete or reword one away in a cleanup pass. When the gap closes in
novomodelo, the workaround and its comment leave together. Reference the registry
by name, never by a machine-local path.

## 3. Source-model prose

Comments, docstrings, and test names say **"the source model"** — not
NEWAVE/CEPEL — when describing the (single) source format being converted.
The model names stay in identifiers (`NewaveCase`), CLI/display labels, and
file names (`dger.dat`). Carve-out: comparator/report code whose *subject* is
the difference between the two models may name both ("NEWAVE uses full
submarket names; DECOMP abbreviates") — a contrastive statement cannot be
written generically.

## 4. User-facing message hygiene

CLI messages, diagnostics, and remediation hints must be **self-contained**:
no repo docs paths, ticket codes, branch names, build tooling, or private
symbols — pip-installed users have no repo checkout. Breadcrumbs for
maintainers go in comments, not in the message. Guard new remediation strings
with a no-leak test (the `test_remediation_has_no_repo_internal_references`
pattern).

## 5. Presentation and I/O boundaries

- **Rich lives only in `ui/`.** No other module constructs a Console or
  imports Rich (TYPE_CHECKING imports excepted). Library code never calls
  `print()`/`sys.stdout.write` — user-facing rendering goes through
  `novomodelo_bridge.ui.console`.
- **Converters emit structured `Diagnostic` objects** (`diagnostics.emit()`),
  never pre-formatted warning strings — the Diagnostic keeps the entity names,
  stages, and values that the rendering layer turns into detail tables.
- **Case-file writes go through the pipeline's write funnel** (the
  `_write_json`/`_write_parquet` gate that owns `dry_run`/`would_write`) —
  never a bare `json.dump`/`write_text` into the case dir that bypasses the
  dry-run contract.
- **Reads of novomodelo output route through `novomodelo_readers`/`novomodelo_io`** — do not
  re-open `stages.json`/`lines.json`/parquet partitions with ad-hoc path
  logic; per-block quantities are always weighted by `stages.json`
  `blocks[].hours` through the shared weighting kernels.

## 6. Exit codes and the `--json` envelope

- `status == "error"` and a non-zero exit code travel together — a command
  must never print an ERROR verdict and exit 0.
- Every failure path under `--json` emits exactly one
  `{schema_version, command, status, summary, diagnostics}` envelope on
  stdout via `verdict.build_verdict` — including early CLI-level failures.
- Diagnostics produced during a command run inside a `diagnostics.collect()`
  sink and land in the envelope's `diagnostics` array — an emit with no sink
  is a bug, not a fallback.
