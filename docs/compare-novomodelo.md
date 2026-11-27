# Comparing two Novomodelo runs

`compare novomodelo BASELINE_OUTPUT CANDIDATE_OUTPUT` sets the results of two
solved Novomodelo cases side by side. It reads the `output/` directory of each
case, pairs their entities by name, gives each variable's gap between the two
runs, and judges the final bounds of their training. The runs may come from
different novomodelo versions or from the same one. This page covers what the
command reads, how the two runs are aligned, how to read the verdict, and what
the command writes. Flags and their help text are in the
[CLI reference](cli.md).

The command is informational: a difference between the runs never changes the
exit code, and the report is where you judge it.

## Quick start

```bash
novomodelo-bridge compare novomodelo /path/to/old_case/output /path/to/new_case/output --format html
```

The first directory is the baseline and the second the candidate. The command
prints a summary and writes its files to the `version_comparison_artifacts`
directory under the candidate output directory, or to `--out-dir`. With
`--format html` that directory holds `report.html`, and `comparison.json`
beside it, which every run writes. [Output](#output) lists the files of the
other formats.

## Inputs

Each argument is the `output/` directory of a solved Novomodelo case. The first
is the baseline, the reference every gap is relative to; the second is the
candidate, the run compared with it. A directory is a Novomodelo output when it
holds `training/` or `simulation/`.

Two runs of the same novomodelo version are accepted, and so is one directory
given twice, which the `same-output-compared` note reports.

From each output directory the command reads:

- `training/` and `simulation/`;
- `hydro_models/fpha_hyperplanes.parquet`;
- `generic_constraints/resolved_echo.parquet`.

From the case directory that holds the output, which is its parent, it reads:

- `system/hydros.json` and `system/buses.json`, the two required files;
- `system/thermals.json` and `system/lines.json`;
- `stages.json`;
- `constraints/hydro_bounds.parquet` and `constraints/line_bounds.parquet`;
- `constraints/generic_constraints.json`;
- `scenarios/load_seasonal_stats.parquet`.

A `system/` file is looked up in the case directory, then in the output
directory, then in the parent of the case directory. Every file other than
`hydros.json` and `buses.json` is optional: [Results missing in one
output](#results-missing-in-one-output) says what an absent one costs.

## How the two runs are aligned

Entities pair by name, never by id, because the same plant carries different
ids in two runs. Names are compared with the blanks around them removed and
the letter case ignored, so `Furnas` and `FURNAS` name the same plant.

- An entity found in one output only is left out of the comparison and listed
  by the `entity-one-sided` warning.
- A name held by more than one entity of one output is left out of the
  comparison and listed by the `entity-name-duplicated` warning.
- Only the stage ids both outputs hold are compared, and the others are listed
  by the `stage-sets-differ` warning. An output's stages are those of its
  `stages.json`.
- A line pairs by name. When the two outputs define it between the same two
  buses, matched by bus name, in opposite directions, the baseline's flow is
  read with its sign reversed so that the same flow compares equal.

## Labels

Each run carries a label, chosen in this order:

1. The name given with `--baseline-label` or `--candidate-label`.
2. `Novomodelo <version>` when both outputs record a version in
   `training/metadata.json` and the two differ.
3. The name of the case directory: the output directory's parent when the
   output directory is literally `output`, else the output directory's own
   name. This is the default when the versions are equal or one is unknown.
4. `Output 1` for the baseline or `Output 2` for the candidate, when nothing
   above gives a name.

When the two labels come out equal, they become `<label> (1)` and
`<label> (2)`. A character other than a letter, a digit, a blank, `.`, `_`, `+`
or `-` becomes `_`.

The labels name the two runs throughout the report: in its title, candidate
first as `<candidate label> vs <baseline label>`, in the legends, axes and
tables of the charts, and in the notices and tables the comparison adds. Both
runs' series carry Novomodelo's names, such as `<label> immediate_cost`,
`<label> future_cost`, `<label> Lower` and `<label> Upper`. The
console summary carries them in its title. The keys of `comparison.json` and
of the `--json` summary keep the words `baseline` and `candidate`.

For two runs of the same version, name them with the flags:

```bash
novomodelo-bridge compare novomodelo pre/output post/output \
    --baseline-label "pre-change" --candidate-label "post-change"
```

## What is compared

Each compared variable gets a gap: the summed absolute difference between the
two runs over every stage and entity, relative to the summed baseline values. A
gap up to `--tolerance` (0.05 by default, so 5%) is close, one beyond four
times it is far, and one in between is moderate. The headline of the console
summary and the counts of the `--json` summary cover the judged variables
listed below and the two final-bound rows described under
[Convergence and the verdict](#convergence-and-the-verdict).

| Section | Covers | In the headline |
|---------|--------|-----------------|
| `hydro` | Per plant: storage at the end of the stage, generation, turbined flow, spillage, outflow, incremental and total inflow, evaporation, productivity and water value. | Yes, except the water value |
| `hydro-bounds` | The per-stage limits of each plant from `constraints/hydro_bounds.parquet`, and the water withdrawal. | Withdrawal only |
| `system-spillage` | The energy the whole system spills: total, reservoirs and run-of-river plants. | Yes |
| `thermal` | Generation of each thermal plant. | Yes |
| `bus` | Spot price and deficit of each bus. | Yes |
| `bus-net-import` | Net import of each bus: the flow its lines carry in less the flow they carry out. | Yes |
| `line` | Net flow of each line, positive from its source bus to its target bus. | Yes |
| `line-limits` | The per-stage direct and reverse limits of each line from `constraints/line_bounds.parquet`. | No |
| `energy-balance` | Hydro and thermal generation, deficit, excess and net load, by bus. | No |
| `costs` | The cost breakdown by component, and the immediate and future cost of each stage. | No |
| `generic-constraints` | Each generic constraint's left-hand side and bounds, evaluated for each output with its own definition. | No |
| `production-function` | The fitted production planes of the hydro plants, evaluated on a shared grid. | No |
| `operating-points` | Where each run operates its hydro plants, and the productivity each run's planes give at the other's points. | No |

The sections marked No are drawn in the report and judged by no gap. The water
value and the per-iteration bounds are shown in the report and in the files,
and left out of the headline and of the `--json` status.

## Results missing in one output

Which sections a comparison can draw is decided from the partitions and
columns each output holds, never from its version. A section that needs a file
or a column one output lacks is left out for both outputs, and the run goes
on. The `section-unavailable` warning, titled `Results unavailable (N)` with N
the number of rows, lists what was left out in a table with the columns
Section, Column and Missing in. The report shows the same table at the top of
each tab that lost a section.

- Missing in is `baseline` or `candidate` when only that output lacks the item,
  and `both` when neither holds it.
- The Column cell is an em dash when the whole section is left out. It names a
  column when only that column is missing: the section is then compared
  without the variables that read it, and any other section that needs the
  column is listed too.
- An optional entity file, `thermals.json`, `lines.json` or
  `generic_constraints.json`, that is present but cannot be read makes its
  sections unavailable on that side. A warning names the file and the side. An
  absent file reads as a case without that entity type.

## Convergence and the verdict

"Final" is each run's own last iteration with a finite lower bound, so runs of
different lengths still compare. When the two final iterations differ, the
`iteration-counts-differ` note says so. Two rows are judged under the entity
type `final_bounds`, and they count in the headline beside the operation
variables:

- `final_lower_bound`, the final lower bound of each run;
- `final_bound_ratio`, the final upper bound over the final lower bound, judged
  only when the upper bounds are comparable.

The optimality gap of a run is that ratio minus one. The Overview tab of the
report shows the gap of each run and their difference, and the `--json`
summary carries them as `baseline_gap_percent`, `candidate_gap_percent` and
`gap_delta_pp`, the candidate's gap minus the baseline's, in percentage
points. Each is null when it is undefined.

**Sensitivity.** The ratio row compares the two runs' upper-to-lower bound
ratios, so `--tolerance` acts on the difference between the two optimality
gaps, scaled by one plus the baseline's gap. At the default of 0.05, a baseline
gap of 0.5% against a candidate gap of 5% gives a row gap of 0.0448, which is
close, and the row leaves the close band only when the two gaps differ by
about 5 percentage points. Read a smaller difference in `gap_delta_pp`, in the
report's Overview or in the `--json` summary.

### Comparability of the upper bounds

The upper bounds are judged only when they measure the same thing in both
runs. The table lists the reasons in the order they are checked, and the first
one that applies is the one reported, in `upper_bound_comparable` and in the
`upper-bound-not-comparable` note.

| Value | Meaning |
|-------|---------|
| `comparable` | Both outputs have final bounds and record the same kind of upper bound, no run is risk-averse or has an unreadable risk measure, and every final bound is positive and finite. The ratio row is judged. |
| `bounds-missing` | An output has no training convergence data, so it has no final bounds. Neither final-bound row is judged. |
| `kind-missing` | An output was written before novomodelo 0.14 and does not record how its upper bound was computed, so the two upper bounds cannot be told to measure the same thing. |
| `kind-differs` | One run estimated its upper bound by simulation and the other computed it exactly. |
| `risk-averse` | An output comes from a risk-averse run, so the simulated mean cost it reports as its upper bound does not bound the objective of that run. |
| `risk-unknown` | The risk measure of an output could not be read, so the comparison cannot tell whether the simulated mean cost it reports as its upper bound bounds the objective of that run. |
| `bound-nonpositive` | A final bound of an output is missing, zero, negative or not finite, so the gap between its upper and lower bounds has no meaning. |

Under every value but `comparable` the ratio row is not judged. Under every
value but `bounds-missing` the final lower bound still is, and the gaps are
shown without being judged. The Overview tab appends `Not judged` and the value
to its gap line.

The `upper-bound-not-comparable` note carries a sentence on risk measures under
`risk-averse` and `risk-unknown`, and under any other value when the two cases'
risk weights are known to differ. It says that when the two cases declare
different risk measures, their lower bounds bound different objectives, so a
mismatch in the final lower bound can be genuine.

### Status

With `--json` the status is one of:

- `ok`: at least one variable was compared and every one is close;
- `mismatch`: a compared variable is moderate or far;
- `no-comparable-rows`: neither an operation row nor a final-bound row was
  compared.

The final-bound rows count in the headline, so `no-comparable-rows` cannot
occur when both runs trained. The status describes the comparison and is
independent of the exit code, which stays 0.

## Production function

When both outputs hold fitted production planes, the comparison evaluates both
cases' planes on one regular grid over the storage and turbined-flow range the
two cases share, for each paired plant and compared stage. The comparison
builds that grid. It is neither run's own fitting window, so a plane read
outside the window it was fitted for is an extrapolation.

A plant fitted on one side only is left out, and no finding names it. The exact
production function is not available between two Novomodelo outputs, so the
columns that need it, the exact productivity and the fit error of each run's
planes, stay empty. The `production-function` and `operating-points` sections
are unavailable when either output lacks
`hydro_models/fpha_hyperplanes.parquet`.

## Output

The console prints a verdict line, a title and the two directories under their
labels, then one row per compared variable with its gap and its band, and a
table with each run's final lower bound and simulated mean cost. These go to
stdout. The comparison notes, which are the findings listed under
[Diagnostics](#diagnostics), go to stderr: `-v` prints them in full, and
`--quiet` drops the summary and the info notes while warnings still print.

The files go to the `version_comparison_artifacts` directory under the
candidate output directory unless `--out-dir` names another one. `--format`
chooses them:

| File | Written when | Holds |
|------|--------------|-------|
| `comparison.json` | Every run, whatever `--format` selects | The provenance of the run, described below |
| `comparison.parquet` | `parquet` (default) | The compared values in long form, one row per entity, stage, block, variable and source |
| `summary.parquet` | `parquet` (default) | One row per variable: its gap, band and difference statistics |
| `metadata.json` | `parquet` (default) | The facts of the comparison, the convergence bounds of both runs and the data behind the report's charts |
| `summary.json` | `json` (default) | The rows of `summary.parquet` as records |
| `top_divergences.json` | `json` (default) | The largest differences between the runs |
| `comparison.csv` | `csv` | The content of `comparison.parquet` |
| `summary.csv` | `csv` | The content of `summary.parquet` |
| `report.html` | `html` | The multi-tab report |

`comparison.json` records the command and the tolerance, both directories as
typed, each run's version, label, upper-bound kind and optimality gap, the
entities found in one output only, the sections left out, whether the upper
bounds are comparable and why, the bridge version and git commit, a timestamp,
the names of the files written, the top divergences, and every finding. A data
file that cannot be written is a warning on stderr and does not change the exit
code.

With `--json`, stdout holds one JSON document and the human-readable output is
suppressed. Its keys are `schema_version`, `command`, `status`, `summary` and
`diagnostics`. The summary keys, in order, are:

- `tolerance`, `total`, `close`, `moderate`, `far` and `far_variables`: the
  tolerance and the count of compared variables by band, with each far variable
  and its gap, largest first;
- `baseline_version` and `candidate_version`: the recorded versions, null when
  an output records none;
- `only_in_baseline` and `only_in_candidate`: the names found in one output
  only, by entity type;
- `upper_bound_comparable`, `baseline_gap_percent`, `candidate_gap_percent` and
  `gap_delta_pp`: the verdict of the upper bounds and the gaps of
  [Convergence and the verdict](#convergence-and-the-verdict).

A failure prints the same envelope with the status `error`, an empty summary
and a diagnostic that describes it.

**Sharing --out-dir.** The default directory is distinct from
`comparison_artifacts`, the default of `compare newave` and `compare decomp`,
so those commands do not overwrite this one's files. An explicit directory,
set with `--out-dir`, with `NOVOMODELO_BRIDGE_OUT_DIR`, or with `out_dir`
under `[compare]` in a `novomodelo-bridge.toml`, that another comparison also
writes to has its files of the same name overwritten, `comparison.json`
included. Its other files stay in place, so the directory can then hold files
of two comparisons, and the `comparison.json` of the last run lists only the
files that run wrote.

## Exit codes

The command checks these conditions in order and the first failure sets the
code. A command line the parser rejects never reaches the command: it exits 2
before anything is read.

| Condition | Code |
|-----------|------|
| An unknown `--format` token, reported before anything is read | 2 |
| The baseline directory is not found | 1 |
| The candidate directory is not found | 1 |
| A directory holds neither `training/` nor `simulation/` | 2 |
| `hydros.json` or `buses.json` is not found | 1 |
| Either of them cannot be parsed, or has an entry without an integer id or a name | 1 |
| Another file is present and cannot be read | 2 |
| Invalid command line | 2 |
| Both outputs are read | 0 |

Both directories are checked for existence first, the baseline before the
candidate. The checks that follow, from the `training/` and `simulation/` test
to the unreadable file, run on the baseline first and then on the candidate.
A difference between the runs never changes the code.

## Diagnostics

The findings of this command have the category `Compare data`. They print as
comparison notes on stderr, and they are in `comparison.json` and in the
`diagnostics` of the `--json` document. None of them changes the exit code.

| Code | Severity | Meaning |
|------|----------|---------|
| `entity-one-sided` | warning | An entity is found by name in one output only, so the comparison leaves it out. The table gives its entity type, its name and the side. |
| `entity-name-duplicated` | warning | A name belongs to more than one entity in one output, so the comparison leaves those entities out. Names that differ only in letter case or surrounding blanks count as one. |
| `stage-sets-differ` | warning | Some stages are in one output only, so the comparison leaves them out and compares the stages both hold. Stage ids are 0-based. |
| `section-unavailable` | warning | A section or a column is missing in one output or both, so the comparison leaves it out for both. |
| `generic-constraint-differs` | warning | A generic constraint has the same name in both outputs and different terms, bounds or both over the stages both hold. The report draws it with the candidate's definition and evaluates each output with its own, so the two values are not those of one constraint. |
| `upper-bound-not-comparable` | note | The upper bounds are not judged, for the reason given by a comparability value. The notes under it give the upper-bound kind of each output. |
| `iteration-counts-differ` | note | The two runs ended at different iterations, so a difference between their final bounds can come from the number of iterations. |
| `same-output-compared` | note | Both arguments are the same directory, so every difference is zero. |

Reading an output can also raise the findings of the readers the other compare
commands share, in other categories, such as a note that `penalties.json` was
not found or a warning that a file could not be read. Each starts with
`Baseline output:` or `Candidate output:` to name its side.

## Known limitations

1. **Risk-averse and unknown upper bounds are not judged.** Under `risk-averse`
   and `risk-unknown` only the final lower bound is judged, and lower bounds of
   different risk measures bound different objectives.
2. **A plant fitted on one side only is left out.** Neither the comparison of
   the fitted planes nor the productivity gap holds it, and no finding names
   it.
3. **A null realized flow is read as 0.** A null evaporation or water
   withdrawal flow, or a null violation slack on either, reads as 0, as in
   `compare newave`.
4. **The data files name the baseline after the source models.** The report
   uses Novomodelo's names on both sides, but in the data files the baseline
   fills the columns and rows named for the source model, `newave_value`,
   `newave_code` and the source `newave`, and the candidate fills the
   `novomodelo_*` ones. In `metadata.json` the baseline's costs are keyed by the
   cost-parcel names of the source models, except Storage Bounds, Inflow
   Penalty and Pumping, which carry the Novomodelo cost column names
   `storage_violation_cost`, `inflow_penalty_cost` and `pumping_cost`.

## See also

- [CLI reference](cli.md): every command and option.
- [NEWAVE track](newave.md#comparing-results): comparing NEWAVE's results with
  Novomodelo's.
- [DECOMP track](decomp.md#comparing-results): comparing DECOMP's operation
  with Novomodelo's.
