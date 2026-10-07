# `novomodelo-bridge`

Convert power system data to Novomodelo input format.

**Usage**:

```console
$ novomodelo-bridge [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--version`: Show the version and exit.
* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `dashboard`: Generate an interactive HTML dashboard...
* `convert`: Convert data from a source format to Novomodelo...
* `compare`: Compare source model inputs/results...
* `check`: Validate source-model inputs without...

## `novomodelo-bridge dashboard`

Generate an interactive HTML dashboard from Novomodelo simulation results.

**Usage**:

```console
$ novomodelo-bridge dashboard [OPTIONS] {case_dir}
```

**Arguments**:

* `case_dir`: Path to the Novomodelo case directory.  [required]

**Options**:

* `-o, --output <path>`: Output HTML file path (default: &lt;case_dir&gt;/dashboard.html).
* `--open`: Open the generated dashboard in the default web browser after writing it.
* `--json`: Emit a single machine-readable JSON verdict to stdout and suppress the human-readable (Rich) output.
* `-v, --verbose`: Increase console log verbosity (-v INFO, -vv DEBUG).  [default: 0]
* `--log-file PATH`: Write the full DEBUG log to PATH (the console verbosity is unaffected).
* `--no-color`: Disable coloured output (also honoured via the NO_COLOR env var).
* `--quiet`: Suppress the summary and info notes; warnings/errors still show.
* `--help`: Show this message and exit.

## `novomodelo-bridge convert`

Convert data from a source format to Novomodelo JSON.

**Usage**:

```console
$ novomodelo-bridge convert [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `newave`: Convert a NEWAVE case directory to a Novomodelo...
* `decomp`: Convert a DECOMP deck revision to a Novomodelo...

### `novomodelo-bridge convert newave`

Convert a NEWAVE case directory to a Novomodelo case directory.

**Usage**:

```console
$ novomodelo-bridge convert newave [OPTIONS] {src} {dst}
```

**Arguments**:

* `src`: Path to the NEWAVE case directory.  [required]
* `dst`: Path to the output Novomodelo case directory.  [required]

**Options**:

* `--validate`: After conversion, validate the output with the novomodelo package.
* `--force`: Overwrite destination directory if it already contains files.
* `--diagnostics-json PATH`: Also write the conversion diagnostics (counts + findings) as JSON.
* `--json`: Emit a single machine-readable JSON verdict to stdout and suppress the human-readable (Rich) output.
* `--dry-run`: Run the full conversion in memory and report what would be written, without creating or modifying the destination directory.
* `-v, --verbose`: Increase console log verbosity (-v INFO, -vv DEBUG).  [default: 0]
* `--log-file PATH`: Write the full DEBUG log to PATH (the console verbosity is unaffected).
* `--no-color`: Disable coloured output (also honoured via the NO_COLOR env var).
* `--quiet`: Suppress the summary and info notes; warnings/errors still show.
* `--help`: Show this message and exit.

### `novomodelo-bridge convert decomp`

Convert a DECOMP deck revision to a Novomodelo case directory.

Deck features the conversion leaves out (``check decomp`` lists them per
deck) are reported as warnings, never dropped silently. The boundary FCF is
imported by default whenever the deck declares its cut files; ``--no-fcf``
skips it.

**Usage**:

```console
$ novomodelo-bridge convert decomp [OPTIONS] {src} {dst}
```

**Arguments**:

* `src`: Path to the DECOMP deck directory.  [required]
* `dst`: Path to the output Novomodelo case directory.  [required]

**Options**:

* `--force`: Overwrite destination directory if it already contains files.
* `--validate`: After conversion, validate the output with the novomodelo package.
* `--diagnostics-json PATH`: Also write the conversion diagnostics (counts + findings) as JSON.
* `--json`: Emit a single machine-readable JSON verdict to stdout and suppress the human-readable (Rich) output.
* `--dry-run`: Run the full conversion in memory and report what would be written, without creating or modifying the destination directory.
* `--no-fcf`: Skip importing the deck&#x27;s boundary FCF. By default, when the deck declares cortes/cortesh files (its FC records), they are imported as a terminal-stage novomodelo policy checkpoint via an in-process 1-iteration novomodelo pass (slow; requires novomodelo-python). Pass this for a quick conversion without the terminal FCF. The FCF is always skipped under --dry-run.
* `-v, --verbose`: Increase console log verbosity (-v INFO, -vv DEBUG).  [default: 0]
* `--log-file PATH`: Write the full DEBUG log to PATH (the console verbosity is unaffected).
* `--no-color`: Disable coloured output (also honoured via the NO_COLOR env var).
* `--quiet`: Suppress the summary and info notes; warnings/errors still show.
* `--help`: Show this message and exit.

## `novomodelo-bridge compare`

Compare source model inputs/results against Novomodelo.

**Usage**:

```console
$ novomodelo-bridge compare [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `decomp`: Compare a DECOMP run&#x27;s published operation...
* `newave`: Compare NEWAVE published results against...

### `novomodelo-bridge compare decomp`

Compare a DECOMP run&#x27;s published operation against Novomodelo&#x27;s simulation.

Informational: always exits 0, reporting divergences without failing.

Two caveats apply to the generated report. First, the Overview tab&#x27;s NPV
cost cards compare DECOMP&#x27;s undiscounted-nominal costs against Novomodelo&#x27;s
time-discounted costs: DECOMP&#x27;s own cost report carries no per-stage
discount factor, so none is fabricated on that side, and the two totals
are not on the same time-value footing. Second, percentile bands are
omitted (or labelled low-N where one would otherwise appear) for a
deterministic tree with too few scenarios to report a spread without
synthesizing it.

**Usage**:

```console
$ novomodelo-bridge compare decomp [OPTIONS] {decomp_dir} {novomodelo_output_dir}
```

**Arguments**:

* `decomp_dir`: Path to the DECOMP deck directory (deck + dec_oper_*.csv result files, all directly in it).  [required]
* `novomodelo_output_dir`: Path to the Novomodelo output directory.  [required]

**Options**:

* `--tolerance <float>`: Relative tolerance for results comparison (default 1e-2; overridable via NOVOMODELO_BRIDGE_RESULTS_TOLERANCE or novomodelo-bridge.toml).  [env var: NOVOMODELO_BRIDGE_RESULTS_TOLERANCE]
* `--format FORMAT`: Output format(s): console,html,csv,parquet,json,all. Comma-separated and/or repeatable. Overridable via NOVOMODELO_BRIDGE_FORMAT or novomodelo-bridge.toml. (default: console,parquet,json)  [env var: NOVOMODELO_BRIDGE_FORMAT]
* `--out-dir <path>`: Directory for file artifacts. Overridable via NOVOMODELO_BRIDGE_OUT_DIR or novomodelo-bridge.toml. (default: &lt;novomodelo_output_dir&gt;/comparison_artifacts).  [env var: NOVOMODELO_BRIDGE_OUT_DIR]
* `--json`: Emit a single machine-readable JSON verdict to stdout and suppress the human-readable (Rich) output.
* `-v, --verbose`: Increase console log verbosity (-v INFO, -vv DEBUG).  [default: 0]
* `--log-file PATH`: Write the full DEBUG log to PATH (the console verbosity is unaffected).
* `--no-color`: Disable coloured output (also honoured via the NO_COLOR env var).
* `--quiet`: Suppress the summary and info notes; warnings/errors still show.
* `--help`: Show this message and exit.

### `novomodelo-bridge compare newave`

Compare NEWAVE published results against Novomodelo simulation output.

Informational: always exits 0, reporting divergences without failing.

**Usage**:

```console
$ novomodelo-bridge compare newave [OPTIONS] {newave_dir} {novomodelo_output_dir}
```

**Arguments**:

* `newave_dir`: Path to the NEWAVE case directory (case + MEDIAS-*.CSV result files, all directly in it).  [required]
* `novomodelo_output_dir`: Path to the Novomodelo output directory.  [required]

**Options**:

* `--tolerance <float>`: Relative tolerance for results comparison (default 1e-2; overridable via NOVOMODELO_BRIDGE_RESULTS_TOLERANCE or novomodelo-bridge.toml).  [env var: NOVOMODELO_BRIDGE_RESULTS_TOLERANCE]
* `--format FORMAT`: Output format(s): console,html,csv,parquet,json,all. Comma-separated and/or repeatable. Overridable via NOVOMODELO_BRIDGE_FORMAT or novomodelo-bridge.toml. (default: console,parquet,json)  [env var: NOVOMODELO_BRIDGE_FORMAT]
* `--out-dir <path>`: Directory for file artifacts. Overridable via NOVOMODELO_BRIDGE_OUT_DIR or novomodelo-bridge.toml. (default: &lt;novomodelo_output_dir&gt;/comparison_artifacts).  [env var: NOVOMODELO_BRIDGE_OUT_DIR]
* `--json`: Emit a single machine-readable JSON verdict to stdout and suppress the human-readable (Rich) output.
* `-v, --verbose`: Increase console log verbosity (-v INFO, -vv DEBUG).  [default: 0]
* `--log-file PATH`: Write the full DEBUG log to PATH (the console verbosity is unaffected).
* `--no-color`: Disable coloured output (also honoured via the NO_COLOR env var).
* `--quiet`: Suppress the summary and info notes; warnings/errors still show.
* `--help`: Show this message and exit.

## `novomodelo-bridge check`

Validate source-model inputs without converting.

**Usage**:

```console
$ novomodelo-bridge check [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `newave`: Validate a NEWAVE case directory without...
* `decomp`: Validate a DECOMP deck revision without...

### `novomodelo-bridge check newave`

Validate a NEWAVE case directory without converting or writing any files.

**Usage**:

```console
$ novomodelo-bridge check newave [OPTIONS] {src}
```

**Arguments**:

* `src`: Path to the NEWAVE case directory.  [required]

**Options**:

* `--json`: Emit a single machine-readable JSON verdict to stdout and suppress the human-readable (Rich) output.
* `-v, --verbose`: Increase console log verbosity (-v INFO, -vv DEBUG).  [default: 0]
* `--log-file PATH`: Write the full DEBUG log to PATH (the console verbosity is unaffected).
* `--no-color`: Disable coloured output (also honoured via the NO_COLOR env var).
* `--quiet`: Suppress the summary and info notes; warnings/errors still show.
* `--help`: Show this message and exit.

### `novomodelo-bridge check decomp`

Validate a DECOMP deck revision without converting or writing any files.

Also reports what the conversion will leave behind, so a deferred feature
is never a silent omission.

**Usage**:

```console
$ novomodelo-bridge check decomp [OPTIONS] {src}
```

**Arguments**:

* `src`: Path to the DECOMP deck directory.  [required]

**Options**:

* `--json`: Emit a single machine-readable JSON verdict to stdout and suppress the human-readable (Rich) output.
* `-v, --verbose`: Increase console log verbosity (-v INFO, -vv DEBUG).  [default: 0]
* `--log-file PATH`: Write the full DEBUG log to PATH (the console verbosity is unaffected).
* `--no-color`: Disable coloured output (also honoured via the NO_COLOR env var).
* `--quiet`: Suppress the summary and info notes; warnings/errors still show.
* `--help`: Show this message and exit.
