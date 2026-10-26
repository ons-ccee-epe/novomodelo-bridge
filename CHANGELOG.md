# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`export dessem` writes a solved DECOMP case's policy as DESSEM's future
  cost function.** `export dessem DECOMP_DIR CASE_DIR` writes the
  `mapcut`/`cortdeco` files from the case's policy, water travel time
  included; `--audit-csv` also writes the cuts as CSV.
- **`compare newave` and `compare decomp` compare both models at their
  operating points.** The Productivity tab splits each plant's productivity
  gap into a part from the production functions and a part from where each
  model operates, with each model's fit inputs. NEWAVE's operating points
  need the `nwlistop.dat` plant list set to `999`.
- **Both `compare` reports net the penalties out of the lower bounds**, in a
  table under the Overview convergence chart.
- **`convert decomp` accepts decks with a multi-month trunk or with monthly
  stages only.**
- **`convert decomp` converts water travel time** (`VI`). The boundary cost
  then differs from DECOMP's by the value of the water in transit, and a note
  lists the plants.
- **`convert decomp` converts the `UH` spillage ceiling**, so a plant fed
  only by a diversion channel, such as BELO MONTE, spills nothing, as in
  DECOMP.
- **`convert newave` converts pumping stations** (`ESTACAO-BOMBEAMENTO` in
  `indices.csv`) **and `modif.dat` `CDESVIO` diversion channels**, so BELO
  MONTE now receives the water diverted from PIMENTAL.
- **New warnings in `check` and `convert`:**
  - tabs in fixed-column input files, which shift the fields after them;
  - plants whose turbined-flow limit lacks part of its `hidr.dat` data
    (`turbine-limit-data-missing`);
  - FPHA plants whose spill the source model keeps out of the tailrace, which
    Cobre cannot do yet (`fpha-spill-outside-tailrace`);
  - negative incremental inflows in a DECOMP deck.

### Changed

- **The FPHA is fitted as the source model fits it.** Results change for
  most FPHA plants, and the cases need the paired cobre release.
  - `convert newave` takes the reference volume from `volumes-referencia.csv`,
    fits turbined flow over NEWAVE's range and spillage up to twice the
    station's largest monthly mean inflow, reads the levels of daily and
    run-of-river plants at the volume they are held at, and applies the plane
    reduction only when `indices.csv` declares it.
  - `convert decomp` fits over DECOMP's volume window and `FP` records,
    DECOMP's turbined range and twice the largest `mlt.dat` mean inflow, and
    gives a stage with a zero turbined limit zero productivity.
- **Turbined-flow and generation caps follow the source model.** `convert
  decomp` caps turbined flow stage by stage at DECOMP's own limit. `convert
  newave` caps generation at the plant's available power on every stage.
- **Both tracks simulate with every trained cut**, as the source models do.
  `convert newave` warns that a `dger.dat` cut window is not modeled.
- **`convert newave` selects cuts with `level1`**, as cobre no longer has
  `lml1`; results can differ from a run made with `lml1`.
- **VminOP and RHE bound stored energy as the source models do**: each plant
  counts with its own stored-energy productivity, RHE values each reservoir
  along its energy chain (`AC JUSENA`, `EZ`), and violations are priced per
  MWh. Results change for cases with these constraints.
- **LIBS files are read from the files the deck's index declares**, not by
  file name. An index entry whose file is missing fails the conversion, and
  undeclared LIBS-like files are ignored with a warning.
- **Negative historical incremental inflows are kept** rather than raised to
  zero, so NEWAVE results change where they occur.
- **The Constraints tabs show each generic constraint as the solver
  received it**, with VminOP and RHE as one constraint per REE or record.
- **idecomp 1.17.0 and inewave 1.16.2 or newer are required.**

### Fixed

- **`convert decomp` couples the boundary cuts plant by plant, as DECOMP
  does.** Notes and warnings list the plants that borrow another plant's
  coefficients and what cannot be represented.
- **`convert decomp` reads values written without a decimal point as DECOMP
  does** (a `TG` cost `299` is 2.99).
- **`convert decomp` converts two conflicting flow restrictions on one
  plant** as soft constraints, as DECOMP does, with a warning, instead of
  failing.
- **`convert decomp` reads special-constraint coefficients stage by stage.**
- **Diversions reach their target plant.** `convert decomp` gives every plant
  that can divert a channel to its target, unlimited where DECOMP sets no
  limit, and both tracks refuse a case where diverted water reaches no plant.
- **Itaipu in `convert decomp`:** `RE` limits on one frequency bound only
  that half, and an `MP` or `FD` row without a frequency applies to both.
- **`convert decomp` reads a `dadgnl` with a single `GL` register.**
- **`convert newave` reads every `modif.dat` record**, including the last one
  and every `USINA` block, and `CFUGA`/`CMONT` levels after the last record
  follow NEWAVE.
- **`convert newave` gives each plant on a shared gauging station its own
  inflow series**, and plants on the artificial station 300 get zero inflow.
- **`convert newave` keeps plants used by pumping stations**, such as
  VIGÁRIO, instead of dropping them as fictitious, and reports a cascade cycle
  with its path instead of crashing.
- **`convert newave` reads a Latin-1 `tratamento-fpha.csv`**, and both tracks
  accept the LIBS index in either encoding.
- **A failed `convert` keeps the warnings found before the failure.**
- **`compare decomp` puts DECOMP's cost breakdown at present value**, and
  adds its deficit, contracts and unitemized penalties.
- **`compare decomp` counts the ANDE load in Net Load and shows the system
  ENA in MW.**
- **`compare decomp` renders the report when DECOMP's lower bound
  overflowed.**
- **The convergence chart compares the source model's best upper bound with
  Cobre's best**, and `compare newave` also compares the per-iteration upper
  bounds.
- **`dashboard` builds for cases run with training or simulation disabled.**
- **Overview quick-look links highlight the tab they open.**

## [0.18.0] - 2026-10-07

Pairs with cobre 0.18.0, which refuses policy checkpoints written by 0.17.0 or
earlier: convert a DECOMP case with an imported boundary again.

### Changed

- **`cobre-python` is pinned to exactly the paired cobre release**, so an
  installer cannot pick a cobre that refuses the converted boundary.

### Fixed

- **`compare` records the cobre version in its comparison manifest.**

## [0.17.0] - 2026-10-01

Pairs with cobre 0.17.0. A DECOMP case with an imported boundary runs only
under the cobre release that matches the installed `cobre-python`.

### Added

- **`modif.dat` `POTEFE`, `VOLCOTA` and `COTAREA` records are applied**;
  they were skipped without a warning.

### Fixed

- **NEWAVE hydros under expansion (`EE`) are converted** instead of dropped.
- **NEWAVE thermal limits follow `conft.dat`, `term.dat` and `expt.dat` as
  NEWAVE reads them**, and no longer produce bounds cobre rejects.
- **NEWAVE `S` plants hold their reference volume**, as `D` plants do.
- **NEWAVE `VAZMINT` records marked `PRE` or `POS` are converted**; they
  aborted the conversion.
- **NEWAVE electric constraints stop at the end of their declared period.**
- **A 600-posto `vazoes.dat` is read at its width.**
- **NEWAVE decks with empty or single-block files convert without false
  warnings**, and files named with another extension in `arquivos.dat` are
  read.
- **No GTMIN warning for a minimum the deck rounded to its capacity.**
- **DECOMP `RQ` minimum outflow is read per stage**, and released for
  run-of-river headwater plants as DECOMP does.
- **DECOMP `S` plants count as reservoirs.**
- **DECOMP GNL plants follow their `TG` records stage by stage.**
- **A plant with zero rated turbined flow or rated power stays on constant
  productivity** instead of making cobre's FPHA fit abort.
- **The dashboard's Stochastic Model tab renders again.**
- **`--log-file` creates missing parent directories.**
- **Failure output names the command that ran.**

## [0.16.0] - 2026-09-22

Pairs with cobre 0.16.0.

### Added

- **Data maps in Portuguese.** `docs/newave-data-map.md` and
  `docs/decomp-data-map.md` trace each converted field to the deck file,
  record and column it comes from, and list what is not converted yet.

### Changed

- **`dger.dat` switches are honoured**: inputs a switch turns off are left
  out, as NEWAVE leaves them out, and reported.
- **`cobre run <case>` loads the DECOMP boundary** without `--output`.
- **The README walks the check → convert → solve → compare → dashboard
  flow**, with per-track pages `docs/newave.md` and `docs/decomp.md`.

### Fixed

- **`modif.dat` volume records honour their unit column** (hm³ or % of
  useful volume).
- **`check decomp` and `convert decomp --help` no longer list converted
  inputs as deferred.**

## [0.15.0] - 2026-08-24

Pairs with cobre 0.15.0.

### Added

- **`convert decomp` carries GNL deliveries committed past the horizon.**
- **The boundary policy checkpoint describes its own layout**, so a
  mismatched one fails at load.

### Changed

- **Exit codes and `--json` agree**: an error status always comes with a
  non-zero exit code, and every failure emits the `--json` envelope.
- **`convert decomp` accepts `--force`**, as `convert newave` does.

### Fixed

- **Chart titles render again** under plotly.js 3.
- **`convert decomp` splits Itaipu's 50 Hz and 60 Hz groups across their
  buses with the ANDE load**, as DECOMP does.
- **The dashboard's VminOP and RHE charts** plot the constraint and its bound
  on one scale.
- **`compare` writes its JSON artifacts** instead of failing at export.

## [0.14.3] - 2026-08-19

Pairs with cobre 0.14.3.

### Fixed

- **`convert decomp` keeps the boundary's inflow-lag terms**, which were
  dropped when cobre loaded the boundary.

## [0.14.2] - 2026-08-18

Pairs with cobre 0.14.2.

### Added

- **`compare decomp` weights DECOMP's results across its scenario fan**, and
  the Overview cost includes the monthly fan stage.
- **`compare decomp` reads DECOMP's results from the deck directory**; the
  `saidas/` subfolder is no longer used.
- **`compare decomp` file exports and `--json` match the HTML report.**

### Changed

- **`cobre-python` is a required dependency**, so `convert decomp` works on
  a plain `pip install`. The `validation` and `test-roundtrip` extras are
  removed.
- **Error messages give a `pip install` or `--no-fcf` remedy** instead of
  developer build steps.

### Fixed

- **`convert newave` clamps per-stage hydro bounds into the plant's declared
  range**, with a warning, instead of writing bounds cobre rejects.
- **`convert decomp --boundary-fcf` works on lag-coupled decks.**
- **`compare decomp --json` no longer crashes on network corridors.**
- **`compare decomp` compares REE energy and convergence bounds in matching
  units.**
- **Dashboard per-submarket charts show only the real submarkets**, and the
  Constraints tab lays out tall grids correctly.

## [0.14.1] - 2026-08-18

Pairs with cobre 0.14.1; converted cases need cobre 0.14.1 or newer.

### Added

- **`compare decomp --format html` builds the full multi-tab report**, as
  `compare newave` does, plus REE energy, generic constraints and
  evaporation.

### Changed

- **`compare decomp --json` reports a within-tolerance `status`**, set by the
  new `--tolerance` option. The command still exits 0.

## [0.12.0] - 2026-07-22

Pairs with cobre 0.12.0; converted cases need cobre 0.12.0 or newer.

### Added

- **`convert newave` sets cobre's backward-pass scheduler from the deck's
  number of openings.**

## [0.10.0] - 2026-07-10

Pairs with cobre 0.10.0; converted cases need cobre 0.10.0 or newer.

### Changed

- **`compare results` is renamed `compare newave`.**
- **`convert --validate` skips** (`cobre-python-too-old`) when the installed
  `cobre-python` is older than the converted case needs.

### Removed

- **The `compare bounds` command**, its `[compare.bounds]` config table and
  its environment variable.

## [0.9.1] - 2026-06-27

Pairs with cobre 0.9.1, which fixes the discounting of future cost. Cases with
`NE` plants need cobre 0.9.1 or newer.

### Added

- **NEWAVE `NE` plants are converted**, with their dead-volume filling and
  unit entry schedule from `exph.dat`; they were dropped before.
- **`check newave`** reports a case's diagnostics without writing output.
- **`convert --dry-run`** runs the conversion without writing files.
- **`dashboard --open`** opens the dashboard in the browser.
- **`--json` on every command** prints one machine-readable verdict.
- **`--log-file PATH`** writes the full log to a file.
- **`cobre-bridge.toml` and `COBRE_BRIDGE_*` environment variables** set the
  `compare` defaults; flags override both.
- **`conversion_manifest.json`** records each conversion's source, versions
  and time.
- **`docs/cli.md`**, the command reference generated from the CLI.

### Changed

- **Storage and filling penalties are derived from the deficit cost**
  instead of fixed values.
- **`-v` shows INFO and `-vv` DEBUG**; a single `-v` used to show DEBUG.

### Fixed

- **The turbined-flow cap uses each stage's head**, so run-of-river plants
  such as STO ANTONIO and JIRAU are no longer under-capped.

## [0.8.2] - 2026-06-17

Pairs with cobre 0.8.2; converted cases need cobre 0.8.2 or newer.

### Added

- **FPHA cases are converted with FPHA production models**, fitted by cobre
  from the plant geometry and the `polinjus` tailrace curves, with the
  `volref_saz.dat` reference volumes and the `tratamento-fpha` plane
  reduction.
- **`compare results` compares the FPHA production functions** in the
  Productivity tab.

### Changed

- **The turbined-flow cap is head-corrected and the generation cap is the
  installed power**, so an FPHA reservoir's turbined cap no longer overshoots.

## [0.8.1] - 2026-06-13

Pairs with cobre 0.8.1.

### Added

- **`compare` writes queryable artifacts**: `comparison.parquet`,
  `summary.parquet`/`summary.json`, `top_divergences.json` and the
  `comparison.json` manifest.
- **Per-REE stored energy in the VminOP comparison.**

### Changed

- **`compare` takes `--format` and `--out-dir`.** Breaking: `--output`/`-o`
  is removed.

### Fixed

- **NEWAVE `S` plants are converted as reservoirs.**
- **Fictitious plants are identified by structure**, keeping orphaned
  reservoirs.
- **The `modif.dat` `VOLMIN` override applies to the initial storage.**
- **Thermal GTMIN follows `expt.dat` maintenance windows.**
- **Load-block indices are mapped correctly** for renewables and load
  factors.
- **The NEWAVE VminOP comparison uses linear stored energy.**
- **Truncation with penalty is the default inflow treatment.**

## [0.8.0] - 2026-06-01

### Added

- **`compare results` explains productivity**: static productivities against
  `pmo.dat`, realized productivity per stage and a table of each model's
  inputs.
- **The Overview cost breakdown adds thermal and other costs**, and sums
  NEWAVE's generic-violation parcels against Cobre's.
- **Each comparison reports `WithinTol` and `sMAPE`.**
- **The Energy Balance comparison reads the full-horizon net load** from
  nwlistop.
- **A warning for `curva.dat` FIXA penalization**, which Cobre does not
  support.
- **Per-stage hydro penalty overrides.**
- **`convert` exports the visited states** when the NEWAVE case exports them.

### Changed

- **Production-size dashboards are about 20 times smaller.**
- **VminOP constraints cover only real reservoirs**, with productivities that
  match `pmo.dat`.
- **NEWAVE penalty productivities match `pmo.dat`.**
- **GNL anticipated dispatch carries the `adterm.dat` values** instead of
  zeros.

### Fixed

- **Plant, line and case names are HTML-escaped** in the dashboard and the
  report.
- **Thermal GTMIN is kept when `FCMAX` lowers the maximum.**
- **Post-study thermal, exchange-group and seasonal values** follow NEWAVE.
- **`compare results` fails on unreadable Cobre output** instead of reporting
  no divergence, and uses the right line direction and thermal `POTEF`.

## [0.7.0] - 2026-05-24

### Added

- **GNL anticipated thermal dispatch** from `adterm.dat`.
- **A head-corrected turbined-flow cap**, as NEWAVE computes it.
- **The PIMENTAL → BELO MONTE diversion.**
- **NC plants.**
- **Hydro slack charts** in the dashboard and in `compare results`, with
  NEWAVE's withdrawal violations and the plant bounds overlaid.
- **`convert` logs the risk measure** it selects.

### Changed

- **Cut selection defaults to `lml1`.**
- **Inflow non-negativity defaults to `truncation_with_penalty`**, and its
  penalty sits above the withdrawal penalty.
- **`D` plants start at `volume_referencia`.**

### Fixed

- **Withdrawal slack labels in `compare results`** match NEWAVE's sign
  convention.
- **P10–P90 bands and hover** in the slack charts.
- **Pre-horizon GNL commitments, which cobre rejected, are zeroed with a
  warning.**

## [0.6.2] - 2026-05-21

### Added

- **`compare results` gains Network and Performance tabs**, per-bus Hydro
  Operation charts, more Hydro Details panels, per-plant summary tables of the
  largest differences, and thermal-generation cost cards.

### Changed

- **Withdrawal and evaporation are compared as realized values** on both
  sides, in m³/s.
- **`dsvagua.dat` is skipped when `dger.dat` turns other water uses off**, as
  NEWAVE does.
- **Non-simulated sources are must-run**, as in NEWAVE.

### Fixed

- **`dsvagua.dat` plant codes are read as plant codes**, not gauging
  stations; reconvert cases that have the file.
- **The Network tab lays out five or more lines.**

### Removed

- **The Network capacity heatmap, the Hydro Operation spillage-energy section
  and the `GHMAX_FPHC` overlay.**

## [0.6.1] - 2026-05-18

First release for cobre 0.6.x, with every change since 0.5.1.

### Added

- **Converted cases use cobre 0.6's production-model files**, with
  per-stage productivity for `CFUGA`/`CMONT` changes.
- **Stored energy (EARM) and natural inflow energy (ENA)** in the dashboard
  and the comparison report.
- **Plants whose cascade passes through NEWAVE fictitious plants** get the
  right downstream plant.
- **The PAR(p) order selection follows `dger.dat`.**
- **The VminOP bound uses each stage's productivity.**
- **Dashboards are about 40 % smaller.**

### Changed

- **Penalties follow NEWAVE's penalty families.**
- **Comparison charts show only the stages both models share**, with a
  unified hover.

### Fixed

- **The evaporation penalty is scaled by productivity again.**
- **`config.json` no longer carries the inflow penalty cost cobre ignores.**

## Earlier releases

Releases 0.4.0 through 0.5.1 predate this changelog; their history is in the
git log (`git log v0.4.0..v0.5.1`).
