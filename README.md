# novomodelo-bridge

`novomodelo-bridge` converts hydrothermal dispatch cases written for the Brazilian
planning models **NEWAVE** (long term) and **DECOMP** (short term) into the
input format of [Novomodelo](https://github.com/ons-ccee-epe/novomodelo), an open-source
SDDP solver, and compares the two models' results once both have been run.

It is a command-line tool. A session typically goes:

1. `check` that a source case is complete enough to convert;
2. `convert` it into a Novomodelo case directory;
3. solve that directory with the `novomodelo` solver;
4. `compare` the source model's published results against Novomodelo's simulation;
5. open an interactive `dashboard` of the Novomodelo run.

## Installation

```bash
uv tool install novomodelo-bridge    # isolated, on-PATH CLI (recommended)
pipx install novomodelo-bridge       # alternative
pip install novomodelo-bridge        # into the current environment
```

Requires Python 3.12 or newer. The install pulls in `novomodelo-python`, Novomodelo's
Python bindings, so `convert --validate`, `compare`, and the DECOMP boundary
cost-to-go import work without further setup. `novomodelo-python` ships prebuilt
wheels for common platforms; if pip reports that none matches yours, see the
[novomodelo repository](https://github.com/ons-ccee-epe/novomodelo) for build options.

The `novomodelo` solver itself is a separate install, built from the novomodelo
repository (see its README). novomodelo-bridge does not need it to convert or
compare. You need it to solve the converted case between those two steps.

### Versions

A novomodelo-bridge release `X.Y.Z` targets novomodelo `X.Y.Z`: the converted case
follows that novomodelo release's input contract, and the bridge depends on
exactly that `novomodelo-python` release. The pin is exact because novomodelo loads a
DECOMP boundary imported by `convert decomp` only in the version of the
`novomodelo-python` that wrote it (see the
[DECOMP track page](https://github.com/ons-ccee-epe/novomodelo-bridge/blob/main/docs/decomp.md#running-a-case-with-an-imported-boundary)).
`convert --validate` skips its validation step, with a note, when the
installed `novomodelo-python` is older than the paired release.

## Quick start

```bash
# 1. Preflight the source case. Exit 0 = ready, 1 = ready with warnings, 2 = will not convert.
novomodelo-bridge check newave /path/to/newave_case

# 2. Convert into a new Novomodelo case directory (--force overwrites a non-empty one).
novomodelo-bridge convert newave /path/to/newave_case ./my_case

# 3. Solve with novomodelo. By default the solver writes its results to ./my_case/output/.
novomodelo run ./my_case

# 4. Compare NEWAVE's published results with Novomodelo's simulation. Always exits 0.
novomodelo-bridge compare newave /path/to/newave_case ./my_case/output --format html

# 5. Browse the Novomodelo run.
novomodelo-bridge dashboard ./my_case --open
```

Replace `newave` with `decomp` for a DECOMP deck; the flow is the same. The
DECOMP track imports the deck's boundary cost-to-go function by default, which
adds one rule for running the case; see the
[DECOMP track page](https://github.com/ons-ccee-epe/novomodelo-bridge/blob/main/docs/decomp.md).

## Commands

| Command                                        | What it does                                                                                                        |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `convert newave SRC DST`                       | Convert a NEWAVE case directory into a Novomodelo case directory.                                                        |
| `convert decomp SRC DST`                       | Convert a DECOMP deck revision into a Novomodelo case directory.                                                         |
| `check newave SRC` / `check decomp SRC`        | Validate the source inputs without writing anything; report missing inputs and what the conversion would leave out. |
| `compare newave NEWAVE_DIR NOVOMODELO_OUTPUT_DIR`   | Compare NEWAVE's published results (`MEDIAS-*.CSV`, `pmo.dat`) against a Novomodelo simulation output directory.         |
| `compare decomp DECOMP_DIR NOVOMODELO_OUTPUT_DIR`   | Compare a DECOMP run's operation tables (`dec_oper_*.csv`) against a Novomodelo simulation output directory.             |
| `dashboard CASE_DIR`                           | Build an interactive HTML dashboard from a solved Novomodelo case (reads `CASE_DIR/output/`).                            |

Every command accepts `--json`, `-v`/`-vv`, `--log-file PATH`, `--no-color`,
and `--quiet`. `convert` adds `--validate`, `--force`, `--dry-run`, and
`--diagnostics-json PATH`; `convert decomp` also has `--no-fcf`. `compare`
adds `--tolerance`, `--format`, and `--out-dir`. `dashboard` adds `--output`
and `--open`. The generated reference with every option's help text is
[docs/cli.md](https://github.com/ons-ccee-epe/novomodelo-bridge/blob/main/docs/cli.md).
Shell completion: `novomodelo-bridge --install-completion`.

### What `convert` writes

`DST` becomes a Novomodelo case directory: `config.json`, `stages.json`,
`penalties.json`, and `initial_conditions.json` at the top, entity registries
under `system/`, inflow and load data under `scenarios/`, and bounds and
generic constraints under `constraints/`. Stage-varying tables are Parquet;
everything else is JSON. The layout is Novomodelo's documented input format (see
the [novomodelo documentation](https://docs.novomodelo.invalid/)).

Next to the case, `conversion_manifest.json` records provenance: the bridge
version and git commit, the source directory, a hash of every input file read,
the entity counts, and every diagnostic raised during the run.

Diagnostics (a missing optional file, a bound clamped to keep the LP feasible,
a feature the converter leaves out) are rendered as grouped panels on stderr
with the affected plants and stages. `--diagnostics-json` saves them to a
file; `--dry-run` runs the whole conversion in memory and lists what would be
written.

### What `compare` writes

`compare` prints a per-variable summary table and writes artifacts to
`NOVOMODELO_OUTPUT_DIR/comparison_artifacts/` (or `--out-dir`). `--format` chooses
the set: `parquet` and `json` (the tidy comparison dataset) by default, `csv`,
`html` for a multi-tab `report.html` covering costs, system and energy
balance, network, convergence, performance, and per-plant detail, or `all`.
`compare` is informational and always exits 0; the report is where you judge
the differences.

### Machine-readable output

With `--json`, a command prints exactly one JSON document to stdout,
`{schema_version, command, status, summary, diagnostics}`, and nothing else on
stdout, including when it fails. Human-readable rendering is suppressed.

| Command     | Exit codes                                                                |
| ----------- | ------------------------------------------------------------------------- |
| `check`     | 0 ready, 1 ready with warnings, 2 will not convert                        |
| `convert`   | 0 converted, 1 conversion error, 2 `--validate` reported a load failure   |
| `compare`   | always 0                                                                  |
| `dashboard` | 0 written, 1 error                                                        |

An error status and a non-zero exit code always travel together.

## Configuration

Defaults for the `compare` options can live in a `novomodelo-bridge.toml`:

```toml
[compare]
format = ["console", "html"]   # or one string; same tokens as --format
out_dir = "comparisons"

[compare.results]
tolerance = 0.01               # relative
```

The file is looked up in the working directory and its parents, then
`$XDG_CONFIG_HOME/novomodelo-bridge/config.toml`, then
`~/.config/novomodelo-bridge/config.toml`; the first one found is used. The same
settings exist as environment variables (`NOVOMODELO_BRIDGE_RESULTS_TOLERANCE`,
`NOVOMODELO_BRIDGE_FORMAT`, `NOVOMODELO_BRIDGE_OUT_DIR`). Precedence is flag >
environment > file > built-in default.

## Documentation

- [CLI reference](https://github.com/ons-ccee-epe/novomodelo-bridge/blob/main/docs/cli.md), generated from the command help.
- [NEWAVE track](https://github.com/ons-ccee-epe/novomodelo-bridge/blob/main/docs/newave.md): inputs, what the converter changes, comparing results.
- [DECOMP track](https://github.com/ons-ccee-epe/novomodelo-bridge/blob/main/docs/decomp.md): inputs, deferred features, the boundary cost-to-go import, running the case.
- [Mapa de dados NEWAVE](https://github.com/ons-ccee-epe/novomodelo-bridge/blob/main/docs/newave-data-map.md) and [Mapa de dados DECOMP](https://github.com/ons-ccee-epe/novomodelo-bridge/blob/main/docs/decomp-data-map.md), in Portuguese: which deck file, record, and column feeds each Novomodelo file and field, and what is not converted yet.
- [Architecture](https://github.com/ons-ccee-epe/novomodelo-bridge/blob/main/docs/architecture.md): how the code is organised, for contributors.
- [Contributing](https://github.com/ons-ccee-epe/novomodelo-bridge/blob/main/CONTRIBUTING.md): development setup, tests, quality gates, releasing.
- [Changelog](https://github.com/ons-ccee-epe/novomodelo-bridge/blob/main/CHANGELOG.md).

## Origin and credits

novomodelo-bridge is developed by Operador Nacional do Sistema Elétrico - ONS,
Câmara de Comercialização de Energia Elétrica - CCEE and Empresa de Pesquisa
Energética - EPE, with other contributors, as a fork of
[cobre-bridge](https://github.com/cobre-rs/cobre-bridge). The fork was created
from the cobre-bridge v0.18.0 release (commit `8ca8e38`, tagged `fork-point` in
this repository); every commit up to and including it is cobre-bridge's and is
preserved unchanged here.

## License

Apache-2.0
