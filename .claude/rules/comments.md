---
paths:
  - "**/*.py"
---

# Novomodelo-Bridge Comment & Docstring Rules

Governs every `#` comment and docstring in any `.py` file in this repo.
Python adaptation of novomodelo's comment discipline (`~/git/novomodelo/.claude/rules/comments.md`);
the Deletion Test and the Four Voices are identical, the directive set is adapted
to Python and to this repo's domain (file-format conversion with unit traps).

**Default is silence.** A comment is a liability — it costs reader attention and
it can rot away from the code it describes. The job of this rule is to keep the
small set of comments that prevent real bugs and delete everything else.

---

## 0. The Default-Off Discipline

You are writing code with **no comments** until a specific comment earns its
place against §1. In order, before any comment exists:

1. **Refactor first.** Rename → extract a well-named function/constant →
   introduce a dataclass/enum/NewType. A comment that a better name would carry
   is a refactor you skipped.
2. **Relocate.** The why of a _change_ goes in the commit message; a durable
   cross-cutting contract goes in `.claude/rules/*.md` or a module docstring; a
   behavioural fact goes in a **test name**. Inline comments are the residual:
   the line-local trap none of those can carry.
3. **Hoist.** A fact repeated across siblings (dataclass fields, dict keys,
   match arms, call sites) is stated **once** at the enclosing scope.
4. **Only then comment** — the shortest form that survives the Deletion Test,
   referencing owning symbols instead of copying their formulas or numbers.

### The docstring floor (terse, not absent)

Public modules, classes, and functions carry a docstring (the global Python
rules mandate this). Pay that floor with **one terse purpose line** plus only
the sections that carry a nugget (`Args:` entries with a unit/shape/ordering
contract, `Raises:` that callers must handle). Never a blow-by-blow narration
of the body, and never the same clause restated across sibling functions —
hoist shared context to the module docstring.

---

## 1. The Deletion Test (the one gate)

> **Delete the comment.** Now — reading only the code, would a competent
> engineer (a) introduce a bug, (b) "simplify" something correct into something
> wrong, or (c) be unable to recover a fact that lives **outside this file**?
> **If none → leave it deleted. If any → restore it, cut to the single clause
> that triggers the "yes".**

| If, with the comment deleted, the code…                         | Verdict                                  |
| --------------------------------------------------------------- | ---------------------------------------- |
| would be mis-edited into a bug (wrong-but-running alternative)  | **KEEP** — Contract (Voice 1), 1 clause  |
| would be "simplified" by removing a load-bearing choice         | **KEEP** — Rationale (Voice 2), 1 clause |
| loses a fact that lives outside this file (spec, sibling, test) | **KEEP** — pointer by symbol             |
| reads exactly the same to a competent engineer                  | **DELETE**                               |
| would be clearer with a better name / type / extracted fn       | **REFACTOR**, don't comment              |
| loses only "how it got here" / what the next line does          | **DELETE** (→ git / a test name)         |

The test applies to **every** comment site — module docstrings, function
docstrings, function bodies, call sites, dataclass fields. The common bloat
shapes, all DELETE:

- A docstring that narrates its own body ("then for each stage: reads X,
  computes Y, writes Z").
- A call-site comment naming the function it precedes. If the call is not
  self-documenting, **rename the function**.
- Pipeline labels (`# Step 1`, `# Phase 3`) — the function names carry the phases.
- An "explains why this is long" note with no suppression directive attached.
- The same clause repeated down a function — state it once, delete the echoes.
- An `Args:` list restating each parameter's name and type — keep only the
  entries carrying a nugget (a unit, an ordering contract, a shape).

When in doubt, **delete** and put the thought in the commit message.

---

## 2. The Four Voices — the only comments that survive §1

- **Voice 1 — Contract.** A present-tense invariant **+ the wrong-but-running
  alternative it forbids +** the owning symbol. _Exemplar (this repo):_ "must
  stay pandas, never polars — handing `line_bounds` a polars frame makes the
  bounds overlay silently disappear (`line_summary_chart` iterates
  `.iterrows()`)."
- **Voice 2 — Rationale.** Why a non-obvious choice was made; kept only while
  the wrong simplification is still plausible. _Exemplar:_ "probability-weighted
  mean, not `.mean()` — uniform averaging is wrong on a skewed scenario fan."
- **Voice 3 — ~~Narrator~~ (NOT ALLOWED).** What the next line does; how it
  used to work; which ticket changed it; project dates. Belongs in names, git,
  or test names — never a shipped comment.
- **Voice 4 — Intent/Seam.** Why a currently-unused item exists and what
  activates it — pinned to the suppression or guard that anchors it (an unused
  parameter kept for a wired-later caller must say so, or be deleted).

---

## 3. Directives

### DO

- **D1 — Contracts.** When the obvious alternative is wrong, say so: invariant +
  forbidden alternative + owning symbol. One clause.
- **D2 — Units and conventions.** Annotate values carrying physical units or a
  convention trap — this repo is made of them: `m³/s` vs `hm³`, MW vs MWmes/MWh,
  k$ vs R$ (×1000), per-block vs per-stage vs hours-weighted aggregation,
  1-based source-model ids vs 0-based novomodelo ids, calendar vs stage indexing.
  This is the one comment a good name usually cannot carry, and the direction
  of the conversion ("divide by stage hours, not multiply") is the load-bearing
  clause.
- **D3 — Rationale above suppression.** Every `# noqa: CODE`,
  `# type: ignore[code]`, `pytest.mark.xfail`, and every deliberate
  `except`-and-degrade carries a reason on the same line or the line above:
  why the fix the diagnostic asks for is inappropriate here. Blanket
  suppressions without a code are banned outright (ruff `PGH` enforces this).
- **D4 — Data-format traps.** Fixed-width column positions, sentinel values
  (`-1` block ids, `1e31`-style infinities), case-insensitive filename
  resolution, and record-layout quirks of the source-model files are
  out-of-file facts — one clause naming the trap and the format that owns it.

### DON'T

- **N1 — No what-narration.** Rename instead.
- **N2 — No history narration.** No `replaces`/`formerly`, no project-event
  dates, no plan tokens (see N4). Keep the present-tense fact. Carve-outs:
  bibliographic years, calendar/data-coverage years, a deck revision name
  (`mar-26-rv2`) naming a still-existing fixture that pins a contract.
- **N3 — No drift-prone refs.** Never a source-file line reference — this
  repo's `file.py:NNN` or novomodelo's `file.rs:NNN` / `file.rs ~NN` alike (they
  drift on every edit above the line) — nor commit hashes, `MEMORY.md`,
  `.claude/` paths other than `.claude/rules/*`, machine-local paths
  (`~/git/...`), paths into gitignored dirs (`plans/`), repo-relative paths
  into the novomodelo source tree (`crates/...`, `novomodelo-io/src/...`), or a bare
  internal design-doc section pointer (`design §5`, whose doc lives in
  `plans/`). A pip-installed reader can resolve none of them. Reference by
  symbol, by named test, or by a stable external anchor — the novomodelo book, a
  schema name, a published reference `manual §`, or a novomodelo `file.rs::symbol`
  (all durable, all allowed). For a symbol+line hybrid, keep the symbol, strip
  the line. Gate: `scripts/ci/check_comment_refs.py` (every class hard).
- **N4 — No plan/workstream leakage** in `src/`, `README.md`, `CHANGELOG.md`,
  or `docs/`. No `epic`/`ticket`/`sprint` vocabulary, no `ticket-NNN`/`epic-NN`
  ids, no acceptance-criteria tags. When a banned token is a trailing tag on a
  contract line, amputate the tag, keep the invariant. Plan refs in **test
  names** stay allowed (identifiers join with `_`, which the gate's word
  boundaries respect). Gate: `scripts/ci/check_no_plan_leaks.py`.
- **N5 — No banners fencing groups inside one long function** — extract a
  function. Dividers between top-level sections of a large module are fine.
- **N6 — No duplicated source-of-truth in prose** — a mirror restates the
  _shape_ and references the owner by symbol; it never copies a magic number
  or formula. A drifted mirror is a lie.

### TODO/FIXME

A shipped `TODO` carries a durable behavioural tag
(`TODO(post-study-withdrawal)`) and names the guard or test enforcing the
current limitation. Never a plan token. Bare ownerless `TODO`s are discouraged.

---

## 4. Protected Contracts (never delete — always considered)

A blunt minimization pass that deletes one of these is worse than the bloat it
removes:

- **`TRACKED NOVOMODELO-GAP (Cn)` comments.** Each marks a deliberate workaround for
  a novomodelo limitation, with its removal condition registered in novomodelo's
  `conversion-found-improvements` registry. Never delete or reword one away;
  when the gap closes, the workaround and its comment leave together
  (see `.claude/rules/bridge.md`).
- **Unit/weighting/direction contracts (D2).** "Weight by block hours",
  "k$ → R$ ×1000", "divide, not multiply" clauses are the difference between
  right and wrong numbers that still render.
- **Silent-failure warnings (Voice 1)** naming a wrong-but-running alternative
  whose failure mode is invisible (blank chart, empty overlay, zero-vs-zero
  match).

Tighten these to one clause + owning symbol; never lose the invariant.

---

## 5. Tooling

- `scripts/ci/check_comment_refs.py` — hard gate on N3 (rot-refs), every
  class enforced in `src/` (the pre-existing debt was burned down to zero).
- `scripts/ci/check_no_plan_leaks.py` — hard gate on N4 in both shipped docs
  and `src/` (the `src/` burndown is complete, so the scope is now enforced,
  not advisory).
- `scripts/ci/check_comment_bloat.py` — advisory: long comment blocks and
  repeated clauses, candidates for a comment-skeptic pass.
- `scripts/ci/quality_report.py` — advisory hotspot ranking (churn × size ×
  long functions × suppressions × plan tokens).
