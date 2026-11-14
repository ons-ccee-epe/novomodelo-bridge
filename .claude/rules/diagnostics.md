---
paths:
  - "src/novomodelo_bridge/**/*.py"
---

# Diagnostics: Severity and Wording

Every finding a command reports reaches a NEWAVE or DECOMP modeller as a
console panel, a one-line note, or a `--json` entry. These rules give those
messages one voice. `tests/test_diagnostic_style.py` enforces the mechanical
ones over every `Diagnostic(...)` and `DiagnosticTable(...)` call in `src/`.

## 1. Severity follows the effect on results

- **ERROR**: the output cannot be trusted or was not written. The command
  exits non-zero (`.claude/rules/bridge.md` §6).
- **WARNING**: the output will differ from what NEWAVE or DECOMP computes for
  the same deck, or a report differs from what the run did. The summary says
  how it differs.
- **INFO**: no effect on results. The bridge does what the source model does,
  or makes a change that cannot bind, such as capping a limit at a capacity
  the plant already has.

Never raise a severity to draw attention, and never lower one to quiet a
deck. Moving a finding between WARNING and INFO changes neither the exit code
nor the `--json` status; moving it to or from ERROR changes both.

## 2. A finding is a Diagnostic, never a log line

- Report a degraded input, a dropped record or a changed value with
  `emit(Diagnostic(...))` inside the command's `collect()` sink. A
  `logger.warning` reaches the user only as a generic "Conversion warning"
  panel, or as a bare log line outside `convert`.
- One diagnostic per kind of finding. Each affected entity is a row of its
  table; never emit one diagnostic, or one log line, per plant.
- Log at INFO or DEBUG only what a maintainer tracing a run needs.

## 3. Title

- Sentence case, naming the outcome the user meets ("Diversion closed:
  target plant not in the case"), not the mechanism.
- No trailing period and no `(s)` plural marker. When the finding lists
  entities, the count is a ` (N)` suffix: never `(N plant(s))`,
  `(N finding(s))` or a leading `N plant(s) ...`.
- Deck files in lowercase (`curva.dat`, `dadger.rvN`); records, registers
  and fields as the manual writes them (`AC NUMJUS`, `RHE`, `VAZMINT`).

## 4. Summary

- Full sentences, starting with a capital or the count, ending with a
  period. No `(s)` plural marker: the title carries the count, so the
  summary names "the listed plants"; a summary without a table spells the
  noun to agree with its count (`counted(n, "plant")`). The same holds for
  notes, check lines and the console's count line.
- Say what the deck has, what the conversion did, and the effect: how the
  results differ (WARNING), or why they do not (INFO, "as in DECOMP").
- Name what the user meets: deck files, records, registers and fields as the
  manual names them, and converted-case files with their path
  (`constraints/hydro_bounds.parquet`). Never parquet columns, JSON field
  paths, novomodelo validation-rule numbers, Python names, or log phrasing
  ("; skipping").
- Track code names NEWAVE or DECOMP; `core/` code shared by both tracks says
  "the source model".

## 5. Table

- Lead with the deck's own identifiers, `Code` then `Name`; a second entity
  is `Target code`, `Target name` (or `Owner ...`, `Downstream ...`). A
  finding read from the converted case or its run output, where the deck
  code is not at hand, leads with the case's id (`Hydro ID`, `Thermal ID`).
- An emitter that meets its entities one at a time emits one-row tables
  inside a local `collect()` and re-emits them through `merge_tables`, which
  gives one table per code and the ` (N)` count.
- Put units in the header: `Max gap (m³/s)`, `Capacity (MW)`.
- Stage and block ids are 0-based in every table, DECOMP's included, and the
  table carries the caption `Stage ids are 0-based.` (or `Stage and block
  ids are 0-based.`). Longer
  remarks go in `notes`, which render at full width; a caption wraps to the
  table's width.

## 6. Remediation

- Only when the user can remove the finding by changing the deck or the
  command; name the file, record and field.
- Never "no action needed", "this is informational", or a restatement of the
  summary. Leave the field out instead.

## 7. Codes

- Kebab-case and stable: a code is part of the `--json` contract. A new,
  renamed or removed code, or a severity change, gets a `CHANGELOG.md`
  entry; a reworded title or summary does not.
