# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

novomodelo-bridge is a fork of [cobre-bridge](https://github.com/cobre-rs/cobre-bridge),
and its version numbers continue cobre-bridge's: novomodelo-bridge 0.18.0 is
cobre-bridge 0.18.0 under the new name, paired with novomodelo 0.18.0.
cobre-bridge's release history up to the fork point follows under
"cobre-bridge history" as cobre-bridge published it, so its names, commands and
links are cobre-bridge's.

## [Unreleased]

### Changed

- **BREAKING:** the product is renamed. The CLI and the package are
  `novomodelo-bridge` (`import novomodelo_bridge`), the configuration file is
  `novomodelo-bridge.toml`, environment variables such as
  `COBRE_BRIDGE_OUT_DIR` are now `NOVOMODELO_BRIDGE_OUT_DIR`, and the bridge
  converts cases for, and reads the outputs of, novomodelo 0.18.0 through
  `novomodelo-python`. Everything else behaves as in cobre-bridge 0.18.0.
- `novomodelo-python` comes from the novomodelo repository at a pinned commit,
  not from PyPI, so the bridge installs with uv from this repository (see the
  README).

# cobre-bridge history

## [cobre-bridge 0.18.0] - 2026-10-07

Pairs the bridge with the **cobre 0.18.0** release and pins `cobre-python` to
exactly that release. cobre 0.18.0 refuses policy checkpoints written by 0.17.0
or earlier, so a DECOMP case whose boundary FCF an earlier bridge imported must
be converted again.

### Changed

- **`cobre-python` is pinned to exactly the paired cobre release.** The
  dependency is now `cobre-python==0.18.0` instead of `>=0.17.0,<0.18`, and
  `MIN_COBRE_VERSION` moves to `0.18.0`. cobre loads a policy checkpoint only
  in the version that wrote it, so the boundary `convert decomp` writes loads
  only in that release; a range let an installer pick a patch release that
  refuses it. `tests/test_packaging.py` checks that the pin equals
  `MIN_COBRE_VERSION`.

### Fixed

- **`compare` records the cobre version in its comparison manifest.** It read
  a key cobre never writes, so the manifest's `cobre_version` was always
  empty. The version now comes from `software_version`, the key cobre writes
  from 0.18.0 on, or from `cobre_version` for outputs of cobre 0.17.0 and
  earlier; the dashboard's run summary reads it the same way.

## [0.17.0] - 2026-10-01

Pairs the bridge with the **cobre 0.17.0** release: the `cobre-python` pin and
`MIN_COBRE_VERSION` floor move to `0.17.0` (bridge `X.Y.Z` pairs cobre `X.Y.Z`,
guarded by `tests/test_packaging.py`). cobre 0.17.0 loads a policy checkpoint
only in the version that wrote it, which ties the boundary `convert decomp`
imports to that exact cobre release. The release also converts NEWAVE hydros
under expansion, applies the `modif.dat` `POTEFE`, `VOLCOTA` and `COTAREA`
records, and fixes NEWAVE thermal limits, minimum outflows and electric
constraints, DECOMP minimum outflow and GNL commitments, and several CLI and
dashboard defects.

### Added

- **`modif.dat` `POTEFE`, `VOLCOTA` and `COTAREA` are applied to the
  `hidr.dat` registry**, so a conjunto's effective power and both cota
  polynomials reach the generation caps, the hydro geometry, the
  productivities and the head-corrected turbined caps. They were previously
  skipped without a warning.

### Fixed

- **NEWAVE hydros under expansion (`EE`) are converted.** They were dropped as
  if not yet built, taking their reservoir out of the water balance and their
  inflow into the plant below. An `EE` plant now starts at the configuration
  `modif.dat` declares for the study start and gains capacity as its
  `exph.dat` machines enter, through per-stage bounds; `hydros.json` and FPHA
  eligibility use its final configuration.
- **NEWAVE thermal limits.** The `thermals.json` pair is the envelope of the
  plant's per-stage bounds, so it can no longer publish an inverted interval
  that cobre rejects or a maximum below a stage's bound. Registry values are
  discarded for `EE`/`NE` plants per `conft.dat`, not per presence in
  `expt.dat`. `term.dat`'s monthly minimum applies to the first study year and
  its remaining-years value afterwards (a blank value keeps the monthly one),
  and the per-stage table is written whenever a plant's bounds vary, even
  without `expt.dat` or `manutt.dat`.
- **NEWAVE weekly-regulating (`S`) plants hold their reference volume.** Like
  `D` plants, their storage range and initial storage collapse to
  `volume_referencia` instead of `volume_minimo`; with no reference volume the
  full range is kept. DECOMP is unchanged: there `S` plants are reservoirs.
- **NEWAVE `VAZMINT` records marked `PRE` or `POS` are converted.** They
  aborted the conversion. Each set is read as a step over one repeating year:
  `PRE` sets the minimum outflow in force when the study starts, until the
  first dated record, and `POS` sets the post-study months instead of
  repeating the last study value.
- **NEWAVE electric constraints stop at the end of their declared period.**
  `re.dat` limits were carried to the end of the study and
  `restricao-eletrica.csv` limits repeated by calendar month past it.
- **A 600-posto `vazoes.dat` is read at its width.** The width comes from the
  file size and the deck's history and study years; a 600-posto deck was read
  as 320 postos and either failed or silently produced a different history.
- **DECOMP `RQ` minimum outflow is read per stage**, not per block, and the
  `RQ`/`UH` floor is released for run-of-river plants from a cascade headwater
  down to the first reservoir, as DECOMP does. An explicit `RHQ` window is
  kept.
- **DECOMP weekly-regulating (`S`) plants count as reservoirs** for that
  release and in `RHE` stored-energy constraints, which also evaluate them
  with the volume-integrated productivity.
- **The dashboard's Stochastic Model tab renders again.** It still read the
  historical inflows by a single `date` column, which the windowed
  `scenarios/inflow_history.parquet` replaced, so the tab showed an error
  placeholder; it now reads each window's `start_date`. A case with no fitted
  AR model shows a note instead of an empty order chart.
- **`--log-file` creates its missing parent directories**, and a path that
  cannot be written is reported as an option error instead of a traceback.
- **Failure and diagnostics output names the command that ran.** `dashboard`
  and `compare` failures were headed "Conversion notes" and "Comparison
  failure" whatever the command, and several hints printed a doubled `→`.
- **NEWAVE decks with empty or single-block files convert without false
  warnings.** A maintenance file with no record means no maintenance, and a
  single-block `patamar` file needs no load factors; both were reported as
  unreadable or missing. Warnings name the file the deck's `arquivos` lists,
  and `compare newave` reads `sistema` and `c_adic` through it too, so decks
  that use another extension (such as `.eas`) are read in full.
- **No GTMIN warning for a minimum the deck rounded to its capacity.** A
  thermal GTMIN written as the available capacity rounded to 0.01 MW could
  exceed it by a fraction of that step and was reported as a data error, with
  both values shown as the same number. Excesses within that rounding are no
  longer reported, and the warning's table shows values to 0.01 MW with the
  excess in its own column.
- **DECOMP GNL plants follow their `TG` records stage by stage.** Only the
  first `TG` record of each plant was read, so a stage-1 inflexibility bounded
  every committed delivery and clamped commitments the deck allows. `TG` is
  now densified like `CT` (a stage inherits the last declared record): each
  commitment is checked against its own stage, post-study pricing uses the
  last declared record, and `thermals.json` declares the envelope over stages.
- **A plant with zero rated turbined flow or rated power stays on constant
  productivity** on both tracks, instead of being emitted as FPHA and making
  cobre's fit abort the run; such plants are listed in an informational
  diagnostic.

### Changed

- **cobre pairing.** `cobre-python>=0.17.0,<0.18` and `MIN_COBRE_VERSION =
  "0.17.0"`; the conversion manifest records the `0.17.0` floor.
- **The DECOMP boundary loads only in the cobre version that wrote it.**
  cobre 0.17.0 stamps each policy checkpoint with the version of the build
  that writes it and refuses one written by any other version, so a case with
  an imported boundary must run under the `cobre` release that matches the
  installed `cobre-python` exactly. A mismatched cobre names both versions;
  converting again with the matching `cobre-python` resolves it. The bridge no
  longer passes a version string to the checkpoint writer, which now ignores
  it.
- **Dependency floors.** `inewave>=1.16.1`, the first release that reads the
  `VAZMINT` `PRE`/`POS` markers the fix above relies on, plus
  `pyarrow>=25.0.1`, `pandas>=3.0.6`, `plotly>=7.1.0`, `polars>=1.44.2` and
  `typer>=0.27.2`.

## [0.16.0] - 2026-09-22

Pairs the bridge with the **cobre 0.16.0** release: the `cobre-python` pin and
`MIN_COBRE_VERSION` floor move to `0.16.0` (bridge `X.Y.Z` pairs cobre `X.Y.Z`,
guarded by `tests/test_packaging.py`). `convert decomp` now authors the terminal
boundary with cobre 0.16.0's date-driven contract — the change that requires the
pairing — and the release also honours the `dger.dat` input-gating switches,
fixes `modif.dat` volume units, ships generated Portuguese data maps, and
reorganises the internal package layout.

### Added

- **Data maps, in Portuguese.** `docs/newave-data-map.md` and
  `docs/decomp-data-map.md` trace every file and field of the converted case
  back to the deck file, record, and column it comes from, state the
  transformation applied, and list what the converter does not convert yet.
  Both pages are generated from `docs/lineage/` and gated by tests against a
  real conversion of the mini decks, so they cannot drift from the code.

### Fixed

- `modif.dat` volume records (`VOLMAX`, `VOLMIN`, `VMAXT`, `VMINT`) now honour
  their unit column: `h` is hm³ and `%` is a percentage of the plant's useful
  volume, resolved against the `hidr.dat` registry. Before, permanent records
  were always read as hm³ and dated records always as a percentage, so a
  `VOLMAX` declared in percent became an absolute volume of a few hm³. A
  record with no recognisable unit keeps the old reading and raises a warning.
- `check decomp` no longer reports GNL anticipated dispatch and `MP`
  availability factors as deferred; both are converted, and the stale
  warnings padded the list of what the conversion leaves out.

### Changed

- **cobre pairing.** `cobre-python>=0.16.0,<0.17` and `MIN_COBRE_VERSION =
  "0.16.0"`; the conversion manifest records the `0.16.0` floor.
- **`convert decomp` authors the terminal boundary with cobre's date-driven
  contract.** The boundary cost-to-go function is now priced by calendar date —
  the boundary pool carries the study horizon's terminal date, and each
  anticipated delivery carries an explicit `interval_start`/`interval_end` date
  window instead of a single delivery stage — and the case's season descriptor
  round-trips through the policy checkpoint so cobre's season-compatibility gate
  accepts the reconciled boundary. cobre now resolves `policy.boundary.path`
  against the case directory, so a plain `cobre run <case>` loads the boundary;
  the earlier `--output <case_dir>` guidance and its warning are retired. This is
  the change that pairs the release with the next cobre.
- **`dger.dat` switches are honoured.** `agrint.dat`, `c_adic.dat`,
  `ghmin.dat`, `re.dat`, `restricao-eletrica.csv`, minimum outflow, and the
  dated `TURBMAXT`/`TURBMINT` records of `modif.dat` are now left out when the
  corresponding switch line (`AGRUPAMENTO LIVRE`, `CONS. CARGA ADICIONAL`,
  `CONSIDERA GHMIN`, the two `RESTRICOES ELETRICAS` lines, `DESCONSIDERA
  VAZMIN`, `REST. TURBINAMENTO`) turns them off, as NEWAVE leaves them out.
  Each ignored input is reported by `check newave` and `convert newave` with an
  informational `dger-switch-off` diagnostic; an absent switch line counts as
  on, so decks without the line convert as before.
- Internal package layout reorganised (`core/`, `cobre/`, `newave/`, `cli/`,
  `comparators/{newave,decomp}/`, `ui/html/`); no CLI or output change.
- **Documentation overhaul for onboarding.** The README now walks the
  check → convert → solve → compare → dashboard flow and states every
  command's exit codes, the `--json` envelope, and the configuration file
  shape. New per-track pages `docs/newave.md` and `docs/decomp.md` (the latter
  absorbs the former boundary-FCF page), a rewritten `docs/architecture.md`
  with the end-to-end data flow and extension points, a new `CONTRIBUTING.md`,
  and a project `CLAUDE.md` reduced to pointers instead of a stale module map.
- `convert decomp --help` no longer describes the exchange network,
  renewables, and GNL anticipation as deferred; all three are converted.

### Removed

- The unused per-REE `penalid.dat` reader in the NEWAVE hydro converters;
  penalties are converted once, system-wide, into `penalties.json`.
- **The real-deck test tier.** Forty-six tests skipped unless real decks were
  present under the gitignored `example/` tree (some also needed a locally
  built cobre binary at a fixed home-directory path), so they ran on one
  machine at best and never in CI. They are removed and catalogued, with what
  each verified and what it needs, in `docs/real-deck-checks.md`;
  `tests/test_local_data_policy.py` now fails the build on any test that
  reaches outside the repository. Small real-format excerpts under
  `tests/fixtures/` replace the checks that only needed one, and the contract
  tests read cobre's schema and example case from vendored copies instead of a
  sibling checkout.

## [0.15.0] - 2026-08-24

Pairs the bridge with the **cobre 0.15.0** release: the `cobre-python` pin and
`MIN_COBRE_VERSION` floor move to `0.15.0` (bridge `X.Y.Z` pairs cobre `X.Y.Z`,
guarded by `tests/test_packaging.py`). The release adds DECOMP anticipated
post-horizon support and lands a broad correctness and structural pass across
both conversion tracks.

### Added

- **DECOMP anticipated post-horizon commitments.** `convert decomp` emits the
  fixed post-horizon anticipated-thermal deliveries cobre 0.15.0 models: the
  already-decided (já-comandada) generation is pinned as fixed past-horizon
  commitments, the freshly-signaled post-study generation is left for the solver
  to re-derive as `post_study_stages.json` thermal bounds, and the anticipation
  lead is emitted uncapped. The boundary cost-to-go function is authored against
  the excised anticipated ring and loads into the fanned terminal.
- **Self-describing `manifest.bin` policy checkpoint.** The boundary policy
  checkpoint carries its own per-pool provenance — cost scale, node and graph
  stage ids, entity manifest — so a later layout change fails loudly at load
  rather than silently; the sidecar `metadata.json` is retired.

### Changed

- **cobre pairing.** `cobre-python>=0.15.0,<0.16` and `MIN_COBRE_VERSION =
"0.15.0"`; the conversion manifest records the `0.15.0` floor.
- **Twin-track symmetry.** Physics, calendar/stage weighting, generic-constraint
  emission, the case write path, and the `DecompCase` parse are single shared
  implementations across the NEWAVE and DECOMP tracks; neither track imports the
  other's private names.
- **Exit-code and `--json` contracts.** An error status and a non-zero exit code
  travel together; every failure path — early CLI-level ones included — emits one
  `{schema_version, command, status, summary, diagnostics}` envelope.
- **Presentation boundary.** Converters emit structured `Diagnostic` objects
  through the `collect()` sink instead of pre-formatted strings, and Rich
  rendering is confined to the `ui` package.
- **Internals.** The comparators' chart code and the hydro converter are split
  into packages; the compare data layer converges on one dataset model; the
  Plotly model is unified across the dashboard and the compare report.
- `convert decomp` gains `--force` pre-clear and rollback parity with the NEWAVE
  track.

### Fixed

- **Chart and axis titles render again under plotly.js 3.x.** plotly.js 3.0
  dropped bare-string `title` support, so raw-dict layouts reaching the browser —
  the compare report's charts and the dashboard's client-side charts — lost their
  titles; titles are normalized to the object form at both the Python and the
  client-side seams.
- **Itaipu 50/60 Hz split and the ANDE load.** `convert decomp` places the 50 Hz
  unit group on the Sudeste bus with the ANDE load netted into it, and the 60 Hz
  group on the Ivaiporã transshipment node, matching the source model.
- **Dashboard VminOP / RHE constraint scale.** The dashboard evaluates a
  storage-scaled constraint's left-hand side with the solver's per-stage
  `rho_acum` override, so the plotted band and its bound share one scale.
- **`compare` artifact export.** The comparison metadata stores each plant's bus
  label as a JSON-native list, so `compare` writes its JSON artifacts instead of
  raising at export.
- Dashboard KPI weighting and guard corrections, and the migration of the NEWAVE
  converter warnings to structured diagnostics.

### Removed

- The comparator's standalone bounds engine; bounds now compare through the
  shared constraint evaluator both tracks use.

## [0.14.3] - 2026-08-19

Pairs the bridge with the **cobre 0.14.3** release and restores the boundary
cost-to-go function's inflow-lag coupling for DECOMP-style decks. `convert decomp`
now declares the boundary's inflow-lag depth so cobre reserves the matching state
slots when it loads the checkpoint; without it those slots were absent and every
inflow-lag term was silently dropped. The pin and `MIN_COBRE_VERSION` floor at
`0.14.3` (the 0.14 input contract is unchanged).

### Fixed

- **`convert decomp` keeps the boundary cost-to-go function's inflow-lag
  coupling.** A terminal cut carries a storage gradient plus per-lag inflow
  gradients, but the deck has no autoregressive inflow model for the solver to
  size the inflow-lag state from, so the written checkpoint reserved no lag slots
  and every inflow-lag term was dropped on load — the reconciled boundary
  collapsed to storage-only (a few hundred state slots where thousands were
  expected). The checkpoint now declares its inflow-lag depth, so cobre reserves
  the canonical inflow-lag slots and places the per-lag gradients; the loaded
  boundary prices the inflow-lag coupling again.

### Changed

- **`MIN_COBRE_VERSION` raised to `0.14.3`** (pin `cobre-python>=0.14.3,<0.15`).
  The 0.14 input contract is unchanged (a case converted here still loads on
  0.14.1), but 0.14.3 is the first release whose checkpoint writer reserves the
  canonical inflow-lag state slots the emitted boundary now depends on; on an
  older cobre those slots are absent and the inflow-lag coupling is silently
  dropped (recorded in the conversion manifest and enforced by the `--validate`
  gate).

## [0.14.2] - 2026-08-18

Pairs the bridge with the **cobre 0.14.2** solver bug-fix release and makes a
fresh `pip install cobre-bridge` work out of the box. The emitted policy (the
CVaR gap rule and the loaded terminal boundary cost-to-go function) now converges
and prices correctly against cobre 0.14.2's fixes, so the pin and
`MIN_COBRE_VERSION` floor at `0.14.2` (the 0.14 input contract is unchanged).
`cobre-python` is now a required runtime dependency, so `convert decomp` — which
imports the deck's boundary cost-to-go function by default — no longer fails on a
plain install. This release also completes `compare decomp` result parity and
fixes several `compare`/dashboard rendering bugs.

### Changed

- **`cobre-python` is now a required runtime dependency** (`>=0.14.2,<0.15`),
  moved from the optional `validation` / `test-roundtrip` extras into the core
  `dependencies`. A plain `pip install cobre-bridge` now installs a cobre that
  can validate converted output and import the boundary cost-to-go function. The
  two extras, which only ever carried `cobre-python`, are **removed** (a plain
  install now supersedes them); CI installs the dependency via `.[dev]` on every
  supported Python.
- **`MIN_COBRE_VERSION` raised to `0.14.2`.** The 0.14 input contract is
  unchanged (a case converted here still loads on 0.14.1), but the emitted CVaR
  gap rule and loaded terminal boundary cost-to-go function rely on cobre
  0.14.2's solver fixes to converge and price correctly, so the bridge requires
  it (recorded in the conversion manifest and enforced by the `--validate` gate).
- **End-user-facing messages are self-contained.** The boundary cost-to-go
  capability and writer diagnostics no longer point users at developer build
  recipes or repo-internal documents (worktrees, `maturin` commands); they now
  give an actionable `pip install` / `--no-fcf` remediation.

### Fixed

- **`convert newave` clamps per-stage hydro bounds into the plant's declared
  envelope (cobre rule 43).** A per-stage override (TURBMAXT, VAZMAXT, or an RE
  ceiling) could raise a `hydro_bounds` MAX column above the plant's own declared
  value in `system/hydros.json`, which cobre rejects on load. A post-emission
  pass now ceils each such value at the declared max and floors it at the
  declared min, so every row stays a valid tightening of the envelope, and emits
  a **warning** naming what was clamped rather than adjusting silently.
- **`convert decomp` no longer aborts on a fresh install.** With the boundary
  cost-to-go function imported by default and `cobre-python` merely optional, a
  plain install failed on any deck that declares its cut files; making the
  dependency required fixes it.
- **`compare decomp --json` no longer crashes on network corridors.** The
  unmapped-entity summary coerced every code with `int()`, but the `line` level
  lists `[from, to]` submarket corridor _pairs_; those now serialize as lists,
  so the JSON verdict is emitted correctly.
- **`compare decomp` REE energy (ENA) is now a fair comparison.** cobre's
  `incremental_inflow_energy_mw` is a stage-mean MW rate; DECOMP's `ena_MWmes`
  is energy over the stage. The comparator converts the cobre rate to MWmês
  (×stage-hours/730, mirroring the existing EARM handling), collapsing a
  several-fold weekly-stage discrepancy to the residual model-coefficient offset.
- **Dashboard per-submarket facets** are restricted to the four real submarkets
  (a clean 2×2), no longer faceting fictitious/interconnection nodes — for both
  the NEWAVE and DECOMP conventions.
- **Dashboard Constraints tab** facets only constraints that carry a reference
  LHS, and `facet_grid` no longer marches y-domains negative on tall grids
  (the row-gap is capped so rows + gaps fit `[0, 1]`).
- **`compare decomp` convergence bounds** are reconciled from DECOMP's k$ to
  cobre's R$ (×1e3) so the Convergence section compares like for like.
- **`convert decomp`** no longer emits a `state_space.inflow_lag_depth` override
  that cobre 0.14 rejects on lag-coupled decks (cobre now sizes the inflow-lag
  state from the PAR(p) order and boundary); this unblocks
  `convert decomp --boundary-fcf` on those decks.

### Added

- **`compare decomp` probability-weights all physical DECOMP variables** across
  the terminal scenario fan, so the reference series aggregate the stochastic
  stage the same way the cost figures already did.
- **The monthly scenario-fan stage is represented in the Overview cost**
  (`relato2`, probability-weighted), completing the cost breakdown across the
  weekly trunk and the monthly fan.
- **Root-only result discovery** — `compare decomp` reads DECOMP results
  (`dec_oper_*`, `relato*`, cut files) directly from the deck root; the legacy
  `saidas/` subfolder convention is retired.
- **`compare decomp` export / `--json` unified onto the canonical
  `ComparisonDataset`** seam (same source of truth as the HTML report), with docs
  and tests brought along.
- Packaging-guard tests: `cobre-python` must be a core dependency (not an extra)
  and its pin must floor at `MIN_COBRE_VERSION`; and a guard that the
  boundary-FCF remediation never leaks a repo-internal reference.

## [0.14.1] - 2026-08-18

Grows `compare decomp` to feature parity with `compare newave` — the rich
multi-tab HTML report now ships from the CLI, correctly labelled "DECOMP" — and
syncs the bridge to the **cobre 0.14.1** input contract.

### Added

- **`compare decomp --format html` now renders the full multi-tab comparison
  report.** Previously the CLI emitted a thin single-page report; it now builds
  the same shared report `compare newave` produces, from the same canonical
  `ComparisonDataset` seam: Overview/cost breakdown, System (per-bus + SIN),
  Energy Balance, Network (corridor→line), Convergence, a DECOMP-vs-cobre
  Performance benchmark, Hydro/Thermal Operation + Plant Details, Productivity
  (realized), and FPHA — plus a DECOMP-specific **REE energy** rollup, a
  **Generic Constraints (LHS vs bound)** section, and an **evaporation**
  comparison. Every reference series/label/title reads "DECOMP" (a new additive
  `reference_label` indirection; `compare newave` output is byte-identical at its
  default). Modeling caveats are surfaced honestly, not papered over: DECOMP
  costs are undiscounted-nominal vs cobre's discounted NPV; FPHA is metrics-only
  (no fitted-surface overlay); the Productivity static/pmo half and the
  constraints' interchange/pumping terms render cobre-only where the source model
  has no counterpart; the cobre sub-monthly-stage evaporation over-scaling gap
  (C11) is shown rather than corrected.

### Changed

- **`MIN_COBRE_VERSION` is now `0.14.1`** (was `0.13.0`), and the `cobre-python`
  `validation` / `test-roundtrip` pins move to `>=0.14.1,<0.15`. cobre 0.14
  carries breaking input-contract changes (sense-free generic constraints,
  `constraints/generic_parameters.json`, the NCS `value` → `availability_factor`
  rename); converted cases require cobre **>= 0.14.1**.
- **`compare decomp --json` now emits a within-tolerance `status`.** A new
  `--tolerance` option (env `COBRE_BRIDGE_RESULTS_TOLERANCE`, default `1e-2`,
  same precedence chain as `compare newave`'s `--tolerance`) drives it: `status`
  is now `ok`/`mismatch`/`no-comparable-rows` (was `ok`/`no-comparable-rows`) —
  `mismatch` when any compared variable's per-row sMAPE exceeds the tolerance,
  `ok` when every variable is within it, `no-comparable-rows` unchanged for an
  empty comparison. The `summary` object gains `within_tol`/`total`/
  `all_within_tol` (mirroring `compare newave`'s `summary`), appended after the
  existing `stages`/`variables`/`unmapped` keys. Backward-compatible: only new
  keys were added, `schema_version` is unchanged, and the command still always
  exits 0 regardless of status.

## [0.12.0] - 2026-07-22

Syncs the bridge to the **cobre 0.12.0** ecosystem release (skipping the 0.11.x
line, which carried no input-contract or output-format change relevant to the
converter). An audit of the full 0.10.0 → 0.12.0 delta found that all 17
system/entity/stage/constraint schemas are byte-identical to 0.10.0; only
`config.json` gained fields, and every addition is optional. The v0.11.0 PAR
`residual_std_ratio` derivation does not apply — the bridge ships inflow
**history** and lets cobre fit PAR(p), never AR coefficients. Cobre's simulation
and `training/` output columns are unchanged, so `compare newave` needs no work.

### Added

- **`convert newave` now emits `training.parallelism.backward_scheduler`** in
  `config.json` (cobre 0.12.0+). The bridge opts the training backward pass into
  cobre's opening-block scheduler — `{ "method": "opening_block", "block_size":
⌈openings / 2⌉ }` — where `openings` is the source deck's backward opening
  count (`dger.dat`'s número de aberturas). The block size is half the openings
  rounded up; it coincides with cobre's own per-stage `opening_block` default but
  records the value taken from the deck.

### Changed

- **`convert --validate` now requires `cobre-python>=0.12,<0.13`** (was
  `>=0.10,<0.11`), pairing with the cobre 0.12.0 release used to validate this
  version.
- **`MIN_COBRE_VERSION` is now `0.12.0`** (was `0.10.0`). Because every converted
  `config.json` now carries the 0.12.0-only `training.parallelism.backward_scheduler`
  block, and the config rejects unknown keys, the output is loadable only by
  cobre **>= 0.12.0**. The value is recorded in `conversion_manifest.json` and the
  `--validate` gate skips gracefully against an older `cobre-python`
  (`skipped_reason: "cobre-python-too-old"`).

### Notes

- **`modeling.cost_scale_factor` (new in cobre 0.12.0) is intentionally not
  emitted.** The source model has no equivalent, and omitting it reproduces
  cobre's previous hard-coded objective scaling byte-for-byte.
- Verified against the cobre 0.12.0 binary and `cobre-python` 0.12.0: `convert`
  output validates with **0 errors** (`cobre validate`), the `--validate` path
  runs clean, and the full test suite (1551 tests) passes.

## [0.10.0] - 2026-07-10

Syncs the bridge to the **cobre 0.10.0** ecosystem release. The conversion
already emitted the two breaking 0.10.0 input-contract changes
(`operational_start_date` on every entity; the `hydro_storage_final` generic
constraint vocabulary); this release updates the version pin, the compatibility
floor, and the documentation to match.

Because every converted `system/*.json` entity now carries a required
`operational_start_date` (introduced in cobre 0.10.0, absent in 0.9.x), a
converted case now requires **cobre >= 0.10.0** — including EX-only cases, which
previously loaded on cobre >= 0.8.2. This is a compatibility narrowing.

Verified against the cobre 0.10.0 binary: `convert` output validates with **0
errors** for both deterministic and stochastic (PAR(p)) cases, and `compare
results` reads cobre 0.10.0 run output unchanged. No new converter work was
required — cobre 0.10.0's optional `travel_time_hours` and chronological
`block_mode` do not apply to a monthly NEWAVE model with parallel load blocks,
so the bridge keeps cobre's defaults (instantaneous transfer, `"parallel"`
blocks).

### Changed

- **`convert --validate` now requires `cobre-python>=0.10,<0.11`** (was
  `>=0.9.1,<0.10`), pairing with the cobre 0.10.0 release used to validate this
  version.
- **The minimum-cobre-version floor is now universal.** `MIN_COBRE_VERSION`
  (`= "0.10.0"`, replacing the filling-only `MIN_COBRE_FILLING_VERSION`) is
  recorded in `conversion_manifest.json` for _every_ case, and the `--validate`
  gate skips gracefully whenever the installed `cobre-python` predates it (JSON
  `skipped_reason: "cobre-python-too-old"`) — previously that skip fired only for
  `NE`-with-filling cases, so an EX-only case validated against an older
  cobre-python would have produced a false failure.
- **`compare results` is renamed to `compare newave`.** The comparison subcommand
  now names its source model, mirroring `convert newave` / `check newave`, so the
  `convert` and `compare` interfaces are uniform and ready for a future
  `compare decomp`. The `--json` verdict and provenance-manifest `command` field is
  now `"compare newave"`; the command's options and behavior are otherwise unchanged.

### Removed

- **The `compare bounds` command is removed.** The workflow standardized on
  `compare newave` (the modelling-divergence tool), so the round-trip LP-bounds
  check is no longer exposed as a CLI command. Its `--summary` / `--variables`
  flags, the `[compare.bounds]` config table, and the
  `COBRE_BRIDGE_BOUNDS_TOLERANCE` environment variable are gone with it. The
  underlying comparator (`comparators/bounds.py`, `bounds_from_inputs.py`) is
  retained as an internal library.

## [0.9.1] - 2026-06-27

First release of the `NE` (future, will-be-built) hydro feature line: it adds
first-class support for NEWAVE `NE` plants via cobre's dead-volume **filling**
schema, and lands a broad CLI overhaul (preflight `check`, unified `--json`
verdicts, config-file defaults, and provenance manifests). The planned 0.9.0
was never released; 0.9.1 supersedes it.

Paired with **cobre 0.9.1** and **cobre-python 0.9.1** — the versions used to
validate this release — which fix a future-cost discounting bug in cobre's LP
(the discount rate was not applied to the future-cost term). `convert
--validate` now requires `cobre-python>=0.9.1,<0.10`.

The filling schema requires **cobre >= 0.9.1**; EX-only conversions are
unchanged on the wire and still load on cobre >= 0.8.2. With
`cobre-python >= 0.9.1` installed, `convert --validate` validates
`NE`-with-filling cases; an older cobre-python that predates the filling schema
is skipped gracefully (an informational diagnostic, never a failure).

### Added

- **`NE` (dead-volume-filling) hydro support.** `NE` plants — future units with
  a dead-volume filling schedule and gradual generating-unit entry — were
  previously dropped (and had to be hand-rewritten to `NC`). They are now
  admitted as real cobre nodes that carry a dead-volume **filling** contract and
  per-unit entry schedule read from `exph.dat`. The new
  `example/newave_rodada_2001_completo` case exercises this with one `NE` plant
  (309 JURUENA).
- **`filling{}` + `entry_stage_id` emitted in `convert_hydros`.** Each `NE` plant
  declares its filling window (start stage from `data_inicio_enchimento`, length
  from `duracao_enchimento`) with `entry_stage_id` set to the fill-complete
  stage, matching cobre's filling/entry-stage contract.
- **Per-stage unit-ramp capacity bounds.** A `max_generation_mw` column is added
  and per-stage generation caps ramp capacity in over the unit-entry schedule
  (`data_entrada_operacao` rows in `exph.dat`) instead of switching the full
  plant on at once.
- **`filling_storage` initial condition.** `NE` plants are seeded into
  `filling_storage` (from the `volume_morto` fraction impounded at filling start)
  rather than `storage`.
- **`min_cobre_version` in `conversion_manifest.json`** plus a per-`NE`
  `Diagnostic`, both recording the cobre >= 0.9.1 requirement for cases that emit
  the filling schema (`null` / absent for EX-only cases).
- **`check newave` preflight command.** Validates NEWAVE inputs without
  writing any output files — surfaces the same diagnostics as `convert`
  but exits before the write phase, so authors can audit a case without
  producing partial output.
- **`convert --dry-run`.** Runs the full conversion pipeline in memory and
  reports diagnostics and a summary without writing any files to the
  destination directory.
- **`dashboard --open`.** After building the HTML dashboard, opens it in
  the default browser automatically.
- **Unified `--json` verdict on every command.** Passing `--json` emits a
  single machine-readable envelope
  `{schema_version, command, status, summary, diagnostics}` on stdout, making
  every command scriptable without parsing human-readable output.
- **`--log-file PATH`.** Writes the full log stream to a file in addition
  to stderr, shared across all commands.
- **`cobre-bridge.toml` config file + `COBRE_BRIDGE_*` env vars** for
  compare defaults (`tolerance`, `format`, `out_dir`). The file is
  discovered from the working directory upward, then
  `$XDG_CONFIG_HOME/cobre-bridge/config.toml`, then
  `~/.config/cobre-bridge/config.toml`. Environment variables
  (`COBRE_BRIDGE_BOUNDS_TOLERANCE`, `COBRE_BRIDGE_FORMAT`,
  `COBRE_BRIDGE_OUT_DIR`) take precedence over the config file; explicit
  flags take precedence over both.
- **Conversion manifest (`conversion_manifest.json`).** Written to the
  destination directory on every successful `convert newave` run: records
  the source path, bridge version, cobre-python version, git sha, and
  timestamp so the provenance of a converted case is always recoverable.
- **`docs/cli.md` autogenerated command reference.** Full per-command,
  per-flag reference generated from the live CLI; cross-linked from the
  README.

### Changed

- **Storage / filling penalties derived from deficit × productivity.**
  `storage_violation_below_cost` and `filling_target_violation_cost` were
  hard-coded placeholders (`1e6`, `1e7`). They are now derived on the same
  energy-equivalent basis as the other hydro penalties (deficit cost ×
  ρ_max_acum, in $/hm³ with no time weighting, matching cobre's LP): the
  storage-below tier at 10× the deficit cost (the greatest hydro penalty) and
  the filling-target tier just below it. The `PENALID`/`VOLMIN` path now also
  uses ρ_max_acum.
- **`-v/--verbose` now raises the console log level to INFO** (`-vv` for
  DEBUG). Previously a single `--verbose` went straight to DEBUG, making
  it impractical as a routine "show progress" flag. Scripts that relied
  on DEBUG output from a single `-v` must add a second `-v`.

### Fixed

- **Per-stage head in the max-turbined (swallowing) cap.** `max_turbined` is now
  evaluated with per-stage tailrace/forebay head, honouring temporal `CFUGA` /
  `CMONT` overrides, instead of a single static head. Run-of-river plants whose
  head varies by stage (e.g. STO. ANTÔNIO, JIRAU) were previously under-capped,
  so cobre spilled flow the source model turbines (inflating thermal dispatch);
  the cap is now emitted per stage in `constraints/hydro_bounds.parquet`.

## [0.8.2] - 2026-06-17

Paired with **cobre 0.8.2** and **cobre-python 0.8.2** — the versions used to
validate this release. `convert --validate` now requires `cobre-python>=0.8.2,<0.9`.

cobre 0.8.2 ships **breaking config changes**: the top-level `energy` section is
removed in favour of a per-production-model `reference_volume`, and
`training.cut_selection` is restructured into a tagged `selection` object.
`convert newave` now emits the v0.8.2 schema, so **converted cases load only on
cobre / cobre-python >= 0.8.2**.

### Added

- **Computed-FPHA production models.** In FPHA cases (`dger`
  `funcao_producao_uhe == 0`), reservoir plants are emitted with `model: "fpha"`
  and `fpha_config: { source: "computed" }` plus a per-plant volume fitting
  window, so cobre fits the production function from geometry + tailrace.
  Run-of-river and non-FPHA plants keep `constant_productivity`.
- **`system/tailrace_curves.parquet`.** The source model's `polinjus`
  downstream-level curve families are converted into cobre's exact
  piecewise-quartic tailrace curves (eleven descriptive columns —
  `downstream_reference_level_m`, `outflow_min_m3s`/`outflow_max_m3s`,
  `coefficient_0`..`coefficient_4`, keyed by `hydro_id`/`family_id`/`segment_id`).
  Drives the v0.8.2 exact-tailrace FPHA fit; omitted (and inert) when the case
  ships no `polinjus`.
- **`fpha_plane_reduction` config.** `tratamento-fpha` is parsed into cobre's
  file-level `fpha_plane_reduction` block on `hydro_production_models.json`
  (`angle` → `tolerance_deg`, or `distance` → `tolerance_pct` + `n_samples`).
- **Per-model `reference_volume`.** The FPHA reference volume (V_ref) is now
  declared per production model: seasonal references from `volref_saz.dat` are
  emitted as `selection_mode: "seasonal"` with one absolute `reference_volume`
  per season; plants without a seasonal row fall back to `percentile 0.65`. This
  replaces reliance on cobre's removed top-level `energy.reference_volume_fraction`
  default.
- **FPHA production-function comparison** in `compare results` (Productivity
  tab). Evaluates each model's lower-envelope production surface
  `GH = min_k(g0 + gv·V + gq·Q + gs·S)` on the source model's own (V, Q) fitting
  grid at S = 0 and compares the surfaces — per-plant fidelity metrics (NMAE /
  bias / GHmax-ratio) plus a rotatable 3D NEWAVE/Cobre/Difference view (and a
  Q-curve for run-of-river plants).

### Changed

- **`training.cut_selection` migrated to the v0.8.2 tagged `selection` object.**
  The emitted block keeps the always-on `row_activity_tolerance` at the top level
  and nests the method under `selection` (`{ "method": "lml1", "check_frequency":
1 }`); `selection` is omitted (disabling row selection) when the source model
  turns cut selection off in both passes.
- **Hydro turbined/generation bounds unified** and computed independently of the
  production function: `max_turbined` is the head-corrected swallowing capacity
  (the operational dispatch cap that binds in the LP) and `max_generation` is the
  rated installed power (Σ n·p_nom). Corrects FPHA reservoirs whose turbined cap
  previously overshot.
- **Source-model-neutral prose.** Comments, docstrings, and descriptive test
  names drop explicit NEWAVE/CEPEL mentions in favour of "the source model";
  identifiers, the `convert newave` CLI, the `source: "newave"` comparison label,
  and user-facing display labels are intentionally unchanged.

## [0.8.1] - 2026-06-13

Paired with **cobre 0.8.1** and **cobre-python 0.8.1** — the versions used to
validate this release. `convert --validate` now requires `cobre-python>=0.8.1,<0.9`.

### Added

- **Three-layer architecture for the `compare` command.** `compare bounds` and
  `compare results` now flow through READ → ANALYZE → RENDER, with a canonical,
  serializable `ComparisonDataset` (a tidy/long value frame + per-variable summary
  frame + a typed metadata side-table) and a `ComparisonManifest` provenance object
  as the single source of truth. Console, HTML, and file export are all pure
  consumers of one dataset, ending the historical console-vs-HTML two-path divergence.
- **Machine-readable comparison artifacts.** Every `compare` run now emits queryable
  data — `comparison.parquet` (tidy frame), `summary.parquet`/`summary.json`,
  `top_divergences.json`, and a `comparison.json` provenance manifest (case paths,
  tolerance, bridge/cobre versions, git sha, timestamp) — so downstream tools and
  agents can query the comparison directly instead of scraping HTML.
- **Per-REE EARM** added to the VminOP generic-constraint comparison.

### Changed

- **Unified `compare` output flags.** The overloaded `--output` (Parquet for
  `bounds`, HTML for `results`) is **replaced** by `--format {console,html,csv,
parquet,json,all}` + `--out-dir`, shared across both subcommands. A plain run
  still writes the queryable data artifacts by default (`console,parquet,json`);
  `--format html` builds the single self-contained report.
  **Breaking:** scripts passing `--output`/`-o` to `compare` must migrate to
  `--format`/`--out-dir`.
- **`charts.py` slimmed.** Per-bus / SIN / percentile aggregation moved out of the
  chart functions into the analyze layer; the duplicated subplot-domain math is now
  a single tested `facet_grid` helper. The 11-tab HTML report renders identically.
- **Converters read parsed inputs through `NewaveCase`** (id-map, horizon,
  active-hydro, constraints, network, stochastic), and the Cobre cost taxonomy is
  single-sourced.

### Fixed

- **Run-of-river `'S'` plants** treated as fio-d'água (storage collapsed to Vmin,
  head-corrected turbine cap) instead of as reservoirs.
- **Fictitious plants** identified structurally, keeping orphaned reservoirs.
- **`modif.dat` VOLMIN override** now applied to the initial-storage base.
- **Thermal GTMIN** follows EXPT maintenance windows rather than TERM.DAT, removing
  spurious must-run; no-POTEF thermals and anticipated thermal cost handled.
- **Patamar (block) index mapping** corrected for NCS per-source load blocks and for
  load-factor per-submarket blocks (global → per-block indices).
- **NEWAVE VminOP LHS** compared on linear stored energy (`ρ_int × VARMUH`).
- **Inflow handling**: post-horizon truncation with penalty made the default.

## [0.8.0] - 2026-06-01

### Added

- **Productivity tab overhaul** in `compare results`. The tab now explains
  productivity in both models across three sections: (A) static-fidelity
  scatters comparing NEWAVE pmo productivities against the converted Cobre
  values, (B) a per-stage **realized productivity** (gen / turbined) chart
  driven by the shared per-plant dropdown widget (all plants, one at a time),
  and (C) a **building-blocks table** pairing each NEWAVE/Cobre input column
  2-by-2 with Δ% highlighting. Productivity is shown as constant within a
  stage and varying across stages.
- **Operational hydro productivity** (gen / turbined) comparison, surfaced
  per plant alongside the existing operation variables.
- **Overview cost breakdown** gains thermal-cost and other-costs charts, and
  the generic-constraint violation row now sums the matching NEWAVE parcelas
  (`VIOLACAO CAR`, `VIOLACAO SAR`, `VIOL. RESTELETRICA`, `VIOL. INTERC. MIN.`,
  `VIOLACAO RHQ`, `VIOLACAO RHV`, `VIOL.RLPP *`) so generic violations compare
  like-for-like with Cobre's `storage_violation_cost`.
- **Trustworthy results metrics**: each `ResultComparison` now reports
  `WithinTol` and `sMAPE`, so large relative errors on tiny quantities are no
  longer mistaken for structural divergences.
- **Full-horizon net load** read from NWLISTOP `mercl` files for the Energy
  Balance comparison.
- **FIXA VminOP penalization warning**. `curva.dat`'s
  `configuracoes_penalizacao` TIPO=0 (FIXA) is not supported by Cobre; the
  conversion now warns when it is encountered. The unmodelled-`modif.dat`
  `DefaultRegister` log was downgraded from a warning to debug.
- **Per-stage hydro penalty overrides** with post-study productivity freeze.
- **Visited cut-generation state export** from `convert`, emitted when the
  source NEWAVE case exports them.

### Changed

- **Production-scale dashboard size**. The cut-cost diagnostic scatter
  (`chart_cuts_vs_solve_time_scatter`) embedded one marker per backward LP
  sample (~12 M points on a 2000-scenario / 50-iteration run → 191 MB and
  unrenderable). It now collapses each `(iteration, stage)` group — where the
  cut count and stage are constant — to a **median ms/LP with a p25–p75 error
  bar**, preserving the cost-of-cut curve while cutting that chart from
  ~191 MB to ~0.1 MB. The Spatial Correlation heatmaps (§D) now round their
  matrices to 4 decimals (the hover shows `.3f`). Combined, a production
  dashboard drops from **~208 MB to ~10 MB** with no perceptible visual loss.
- **VminOP generic constraints** are now built only for genuine
  stored-energy reservoirs (`tipo_regulacao == "M"` with a non-degenerate
  `[vmin, vmax]`) rather than gating on a positive accumulated productivity,
  and their coefficients use per-regulation reference productivities matched
  to the pmo.dat `produtibilidade_acumulada_calculo_earm` values to the 4
  decimals NEWAVE prints (equivalent own-productivity exact, accumulated
  within ~0.01 %).
- **NEWAVE penalty productivities** matched to pmo.dat: PRODT mean for
  VAZMIN/turbined penalties, accumulated maximum-height productivity for
  withdrawal/evaporation, with per-configuration CFUGA/CMONT variation.
- **GNL anticipated dispatch** now seeds the real block-weighted `adterm`
  MW values (honoured by cobre ≥ 0.7.0) instead of zeroing them.
- **Theme single-source-of-truth**: one `hex_to_rgba`, an honest theme SSOT,
  and a fix for the hydro colour collision across charts.
- **Dependencies**: `cobre-python>=0.8.0` (validation extra), `pyarrow>=24.0.0`,
  `pandas>=3.0.3`, `polars>=1.41.2`.
- Large internal refactor toward architectural cleanliness — extracted
  productivity / constraint-expression / study-horizon domain modules,
  single-sourced the case↔output directory convention, replaced
  hand-threaded constraint offsets with named result types and an ID
  allocator, and promoted several cross-module reach-ins to public APIs.

### Fixed

- **Security**: NEWAVE-derived strings (plant / line / case names) are now
  HTML-escaped in the dashboard and comparison report, so a crafted name
  cannot inject markup.
- **Thermal `GTMIN`** is honoured independently of `FCMAX` — the converter no
  longer clamps the minimum-generation bound down to a lowered maximum.
- **Thermal post-study `EXPT` bounds** are frozen at the last study stage.
- **AGRINT post-study exchange-group limits** are frozen at the last study
  value rather than dropping to zero.
- **Post-study seasonalization** now follows the `dger` flag, seasonalizing
  flagged quantities and freezing only the flagless ones.
- **`compare results`** robustness: fail loudly on unreadable Cobre output
  (no false "no divergence"), corrected line direction and thermal `POTEF`
  in the computed bounds, and fixed an `evaluate_lhs_cobre` crash on a real
  simulation `LazyFrame`.

## [0.7.0] - 2026-05-24

### Added

- **GNL anticipated thermal dispatch**. New `converters/anticipated.py`
  reads `adterm.dat` (gated by `dger.despacho_antecipado_gnl`) and
  aggregates per-(thermal, lag, patamar) MW values via a
  block-duration-weighted mean, preserving total committed MWh under
  Cobre's constant-MW-per-stage convention. Each thermal now declares
  `anticipated_config` (with `lead_stages`) in `thermals.json`,
  replacing the dead `gnl_config: null` field. `adterm` is added as an
  optional file to `NewaveFiles` resolved via `arquivos.dat`.
- **Head-corrected turbined-flow cap** for hydro plants. NEWAVE's
  effective turbination cap applies an affinity-law correction with the
  volume-integrated head and a `pinst/prodt` clamp, not the nameplate
  `Σ(n·q_nom)`. New `_compute_max_turbined_hypothesis` implements

      qtur_max = min(Σ_c n_c · q_nom_c · (h_op / h_nom_c)^k_turb,
                     pinst / (ρ_esp · h_int)) · (1 - teif) · (1 - ip)

  with `h_op = mean_cota(V_min, V_65) - cota_jus - perdas` for reservoir
  plants and the machine-count-weighted nominal head for D/F/S plants.
  Reproduces M. DE MORAES's binding peak to within 0.0001 % across all
  28 stages, closing a persistent ~108 m³/s gap that distorted hydro
  operation. Legacy `_compute_max_turbined_simple` preserved as
  fallback when `hidr.dat` columns are missing.

- **PIMENTAL → BELO MONTE diversion** (13 000 m³/s nameplate)
  restored. Without the explicit channel the Cobre LP had nowhere to
  route the upstream water NEWAVE accounts for via the fictitious-plant
  cascade, producing spurious spillage and downstream starvation.
- **NC plant support** in converters.
- **Operational-slack visibility** in both `cobre-bridge dashboard`
  and `cobre-bridge compare results`:
  - Plant-detail tab now plots `water_withdrawal_violation_{pos,neg}`
    and `inflow_nonnegativity_slack` with p10/p50/p90 bands, so the
    user can localize which plants/stages the LP had to relax under
    stochastic noise.
  - Plant-detail tab overlays NEWAVE `VIOL_POS/NEG_VRETIRUH` on the two
    withdrawal-slack panels (converted /2.63 from hm³ to m³/s),
    matching NEWAVE's sign convention.
  - Hydro Operation tab gains per-bus + SIN-total aggregates for all
    four paired hydro slacks (withdrawal + evaporation) plus the
    Cobre-only inflow non-negativity slack — driven by
    per-(entity_id, stage_id) frames since slacks don't go through the
    `ResultComparison` pipeline.
  - Dashed bound overlays for storage / generation / turbined / outflow
    on the compare-results tab (the dashboard tab already had them).
    Static values come from `hydros.json` via
    `read_cobre_hydro_metadata` and are shadowed per-stage by overrides
    from `constraints/hydro_bounds.parquet` via the new
    `read_cobre_hydro_per_stage_bounds` reader.
- **Risk-measure selection logging** in `convert_stages`. INFO log at
  the top names the selected mode (expectation / constant-CVaR /
  per-stage CVaR, from `dger.cvar`) and resolved alpha/lambda when CVaR
  is in play. Per-stage branch refactored into four explicit cases
  mirroring the log; output is bit-identical.
- **Spec doc** `docs/findings/cobre-anticipated-thermal-pre-horizon-
limitation.md` for the Cobre maintainers: self-contained spec of the
  "non-zero `past_anticipated_commitments` rejected" limitation, the
  bridge-side workaround, and the functional/acceptance requirements
  that would let cobre-bridge restore NEWAVE-parity GNL dispatch by
  flipping a single line.

### Changed

- **Default cut selection** now emits `method: "lml1"` with
  `memory_window: 0` (was `domination` + `domination_epsilon: 0.0`).
  Aligns with the `RowSelectionConfig` schema (`memory_window` is
  required for `lml1`).
- **Default inflow non-negativity** now `truncation_with_penalty`:
  clamp negative PAR(p) draws to zero before LP patching and keep the
  non-negativity slack columns as a backstop. Closes the exploit where
  the LP would otherwise route negative inflow noise through the
  withdrawal-neg slack (priced 1 R$/(m³/s) below the nonneg slack on
  the cobre-bridge calibration).
- **`inflow_nonnegativity_cost`** anchored to
  `water_withdrawal_violation_cost + 1` R$/m³/s (was 1.01 × max of
  flow-domain slacks, which made it cheaper than withdrawal and let
  the LP buy "free" water to dodge withdrawal violations).
- **D-regulation plant initial storage** anchored to
  `volume_referencia`, consistent with the collapsed
  `[vmin, vmax] = vref` bounds (NEWAVE freezes D reservoirs across
  stages).
- Stronger guard rails in deterministic mode.

### Fixed

- **Withdrawal slack sign labels** in `compare results` HTML. Cobre's
  `water_withdrawal_violation_pos/neg` columns use the inverse sign
  convention of NEWAVE's `VIOL_POS/NEG_VRETIRUH`; the NEWAVE → Cobre
  column mapping and display labels are swapped so each "Pos" / "Neg"
  panel pairs the columns that mean the same physical violation.
  Evaporation slacks already shared NEWAVE's convention and are left
  as-is.
- **Slack p10/p90 emission**. `read_cobre_hydro_percentiles`'
  `flow_cols` list now includes the three operational slacks added
  above, so the percentile frame ships
  `water_withdrawal_violation_{pos,neg}_m3s_{p10,p90}` and
  `inflow_nonnegativity_slack_m3s_{p10,p90}` columns. Without them,
  `_build_interactive_detail_html` emitted only the bare `Cobre Mean`
  trace per slack chart and the `x unified` hover had nothing to lock
  onto, breaking the band+P10/P90 tooltip every other flow variable
  enjoys.
- **P10–P90 unified-hover tooltip**: every band-trace site (5 in
  `charts.py` plus the interactive plant-detail JS) now sets
  `hoverinfo: "skip"` on the closing-polygon trace, and the visible
  p10/p90 lines are renamed to "Cobre P10" / "Cobre P90". Previously
  the unified hover showed the literal text `Cobre P10–P90` instead of
  values at the cursor x.
- **`past_anticipated_commitments` rejected by Cobre validator**.
  `convert_initial_conditions` now zeroes the `values_mw` array
  (length still `lead_stages`, as Cobre requires) and emits a WARNING
  naming the `adterm.dat` code and the MW values being dropped — so
  the user knows exactly what pre-horizon NEWAVE dispatch is not being
  honoured. `read_anticipated_dispatch` keeps computing the true
  block-weighted MW so the warning is informative; the zero-out
  policy lives at the conversion site, not in the reader. Flipping a
  single line restores genuine values when Cobre lifts the limitation.
- Miscellaneous converter fixes uncovered during the `pmo_set_24`
  stochastic-case investigation.

## [0.6.2] - 2026-05-21

### Added

- New **Network** tab in `compare results` HTML comparing directional
  submarket flow per line: NWLISTOP `int*.out` files vs Cobre
  `output/simulation/exchanges`. Per-line small-multiples with Cobre
  P10/P90 band, NEWAVE mean, and finite capacity bounds (the ±99999
  NEWAVE big-M sentinel for fictitious connections is filtered so it
  doesn't compress real flows into a flat strip).
- New **Performance** tab comparing wall-clock timings: NEWAVE
  `newave.tim` stage totals + per-iteration forward/backward breakdown
  (parsed directly — inewave doesn't expose it) vs Cobre
  `training/convergence.parquet` + `training/metadata.json`. Headline
  cards for total/policy/training duration and a speedup ratio.
- **Hydro Operation** tab now opens with per-bus facets
  (SUDESTE / SUL / NORDESTE / NORTE) for storage, generation, spillage,
  turbined, inflow, and water value, then stacks the EARM/ENA aggregate
  charts and finally a **System Totals (SIN)** section that brings the
  six original system-aggregate charts back at the bottom.
- **Hydro Details** picker gains four new comparison panels per plant:
  evaporation, water withdrawal, total inflow (≡ NEWAVE `QAFLUH`,
  computed Cobre-side from incremental + upstream cascade outflow), and
  total outflow (turbined + spilled). Generation panel carries a dashed
  Cobre LP `gen_max` overlay. Tab renamed from "Plant Details".
- **Hydro Details** and **Thermal Details** tabs open with a
  per-plant **max relative difference** summary table — one column per
  variable, NEWAVE-referenced, colored green/amber/red at ≤1%/≤10%/>10%
  thresholds, sorted worst-first. Bold last row is a per-column
  **Median** across plants (robust to outliers like LAJES's water-value
  divergence).
- **Overview** tab cards switched to thermal-generation NPV
  (NEWAVE/Cobre/Δ R$/Δ%) — the prior meta cards
  (Total Comparisons / Entity Types / Variables) offered no operational
  insight. The cost-breakdown chart now uses a vertical right-side
  legend (the previous horizontal layout collided with the chart
  title), and the companion table has been restyled with proper CSS.
- New readers: `read_nwlistop_intercambio`,
  `read_cobre_line_means/percentiles`,
  `read_cobre_hydro_total_flows`, `read_cobre_hydro_withdrawal`,
  `read_cobre_lp_max_generation`, `read_cobre_spillage_energy`,
  `read_cobre_training_duration`, `read_cobre_iteration_timing`,
  `read_newave_tim_iterations`, `read_newave_tim_stages`.

### Changed

- `compare results` reconstructs _realized_ water-withdrawal and
  evaporation on both sides via the LP slack convention
  `realized = scheduled + violation_pos − violation_neg`. Sign matches
  Cobre's `lp_builder/matrix.rs` water-balance row. Applied symmetrically
  to NEWAVE's `VRETIRUH`/`VEVAPUH` + `VIOL_POS/VIOL_NEG` so the
  comparison is realized-vs-realized.
- NEWAVE `VEVAPUH` and `VRETIRUH` (reported in hm³/month, not flow) are
  divided by 2.63 — NEWAVE's rounded `730 h × 3600 s / 10⁶` factor —
  before comparison against Cobre's m³/s values. The converter side
  still uses the exact 2.628 since it operates on input data with no
  analogous rounding.

- `convert_water_withdrawal` honours `dger.outros_usos_da_agua`: when
  the flag is `0`, the converter short-circuits and emits no
  withdrawal rows, matching NEWAVE's own behaviour (the solver
  ignores `dsvagua.dat` regardless of its content when the flag is
  off). When the flag is `1` (the default) it proceeds normally.

### Fixed

- `convert_water_withdrawal` was treating `dsvagua.dat::codigo_usina`
  as a **posto** and routing it through `confhd.dat`'s posto→plant
  map. The field is actually the **plant code** directly. The
  miscoding swapped data between any plant pair whose code/posto values
  collided (e.g. PICADA's plant code = SIMPLICIO's posto = 126 in the
  bundled case) and silently dropped entries for plants whose code
  didn't coincidentally exist as a posto. On the bundled example,
  SIMPLICIO's withdrawal target moved from 1.09 m³/s to 89.71 m³/s,
  dropping the `withdrawal_m3s` mean abs diff from 4.17 to 0.0009 m³/s
  (r = 0.65 → 1.00). Users with `dsvagua.dat` in their case should
  regenerate after upgrading.
- Network tab small-multiples y-domain formula assumed ≤2 rows and
  positioned panels 5–6 outside the chart area for cases with 5–6
  lines. Now distributes any number of rows evenly across `[0, 1]`.

### Removed

- The dashboard-style Cobre-only capacity-utilisation heatmap from the
  Network tab — a Cobre-only view doesn't belong in the comparison.
- The "System Spillage (Energy Units)" section from the Hydro
  Operation tab — superseded by the new per-bus facets.
- The NEWAVE `GHMAX_FPHC` overlay trace from the Hydro Details
  generation panel — wasn't helping interpretation.

### Notes (carried forward from earlier "Unreleased")

- `convert_non_controllable_sources` emits
  `"allow_curtailment": false` on every NCS entity derived from
  `sistema.dat::geracao_usinas_nao_simuladas`. NEWAVE pre-nets these
  aggregates (PCH, PCT, EOL, UFV, MMGD) from MERC before the dispatch
  LP runs, which makes them effectively must-run; setting
  `allow_curtailment=false` instructs Cobre's LP to pin dispatch to
  the realized per-scenario availability instead of leaving curtailment
  as a cheap LP slack. On the bundled deterministic 1983 case this
  restores parity with NEWAVE — eliminates ≈ 18 % of total NCS supply
  being artificially curtailed, a ≈ +15 % hydro-dispatch swing, and a
  ≈ −23 % spillage divergence. Requires Cobre with the
  `non_controllable_sources.allow_curtailment` field. See
  `docs/findings/ncs-must-run-treatment.md`.
- Regenerated `example/convertido/system/non_controllable_sources.json`
  to reflect the new emission. The 32 NCS aggregates in the bundled
  NEWAVE-derived case now carry `"allow_curtailment": false`; all
  other fields are unchanged.

## [0.6.1] - 2026-05-18

First public release for the cobre v0.6.x line — bundles every change
since v0.5.1 (the v0.6.0 milestone was never published to PyPI).

### Added

- **Cobre v0.6 compatibility**: schema migration to
  `hydro_production_models.json` + `hydro_energy_productivity.parquet`,
  per-(hydro, stage) productivity overrides for CFUGA/CMONT temporal
  changes, scalar parameters, and `scalar_parameters.json` emission.
- **Energy-based hydro outputs**: dashboards and comparison reports now
  surface EARM (stored energy, MWh) and ENA (natural energy from
  incremental inflow, MW) at system, per-bus, and per-plant level.
- **FICT-plant cascade resolution**
  (`src/cobre_bridge/converters/fict_cascade.py`): walks NEWAVE
  fictitious-plant chains so that real plants whose energy cascade
  traverses `FICT.<NAME>` topology bridges (e.g. `TRES MARIAS → FICT.TRES
MA → SOBRADINHO`) are correctly wired in `hydros.json::downstream_id`.
  Restores cobre `ρ_acum` to within 0.004 of NEWAVE's
  `produtibilidade_acumulada_calculo_earm` on the bundled case (was off
  by up to 2.77 MW/(m³/s) on 7 plants). Includes ambiguity warnings
  when the 7-char name-truncation key is shared by multiple real plants.
- **PAR(p)-A `order_selection` mapping**: `dger.consideracao_media_anual`
  → `"pacf"` (classic PAR(p)) or `"pacf_annual"` (NEWAVE option 3) so
  cobre's stochastic estimator matches NEWAVE's configured method.
- **Per-stage VminOP RHS**: bound now uses per-stage ρ_acum (built from
  per-stage own productivities) rather than a static base productivity,
  so the curva.dat percentage targets translate to correct absolute
  bounds for plants with CFUGA temporal overrides.
- **Lighter dashboards**: cached chart data, reduced JSON payload, and
  smaller box-plot point counts in the performance tab. ~40 %
  reduction in HTML file size on a 60-stage case.
- **Comparison-report energy charts**: new aggregate-vs-NEWAVE SIN
  comparison of EARM and ENA in the Hydro Operation and Energy Balance
  tabs, plus per-plant cobre-only entries in the Plant Details tab.
- Findings documentation in `docs/findings/`:
  `evaporation-unidirectional-q_ev.md` (handoff that drove the cobre
  v0.6.1 hotfix) and `fict-cascade-resolution.md` (rationale and
  quantitative verification of the cascade fix).

### Changed

- **`cobre-python>=0.6.1`** in the `validation` extra — required so
  `cobre-bridge convert --validate` picks up cobre's signed `Q_ev`
  evaporation fix.
- **NEWAVE-aligned penalty conversion**
  (`src/cobre_bridge/converters/network.py`): full rewrite of the
  PENALID-to-cobre mapping using the four-family taxonomy from the
  NEWAVE manual v29 §3.24. Documented module-level constants for
  ρ_avg / ρ_max_acum derivation, `_EVAPORATION_MULT = 1.1` (down from a
  literal 10× per the manual to keep the LP coefficient range below
  HiGHS's 1e10 conditioning warning), tie-breaking factors on
  flow-domain slacks that would otherwise share ρ_avg, and a 100×
  uplift on NEWAVE micro-penalties to lift the LP coefficient floor
  off HiGHS's noise threshold.
- **Cost Breakdown chart** in the comparison report now truncates
  Cobre to NEWAVE's reported stage range so the totals compare
  like-for-like when Cobre simulates a longer horizon.
- **All comparison-report charts** now use `hovermode: 'x unified'`
  matching the dashboard convention so NEWAVE-vs-Cobre lines snap to
  the same x-tooltip.
- **Truncation to common-stage horizon** applied to every chart in the
  comparison report (energy balance, hydro operation, plant details)
  so visualizations never show unmatched Cobre-only stages.

### Fixed

- **Evaporation penalty conversion**: restored the `× ρ_max_acum`
  factor on `evaporation_violation_cost` (was dropped when the
  docstring on cobre's side appeared to suggest hm³ units), and
  derived `inflow_nonnegativity_cost` as 1 % above the strictest
  flow-domain slack so the slack is the LP's last resort.
- **`config.json::modeling.inflow_non_negativity.penalty_cost`** is no
  longer emitted — cobre treats this field as deprecated and reads the
  live value from `penalties.json::hydro.inflow_nonnegativity_cost`.
  The legacy field stayed in the converted output and confused users
  about which value the LP would actually use.

## Earlier releases

Releases 0.4.0 through 0.5.1 predate this changelog; their history is in the
git log (`git log v0.4.0..v0.5.1`).
