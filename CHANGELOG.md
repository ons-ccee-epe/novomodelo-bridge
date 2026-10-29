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

### Fixed

- **`export dessem` exports a case whose water in transit outlives the study**
  instead of failing with `year -214749 is out of range`; a cut coefficient on
  water that would arrive after the study ends is dropped with a
  `dessem-fcf-undated-buckets-priced` warning.

## [0.19.1] - 2026-10-26

Pairs the bridge with the **novomodelo 0.19.1** release and pins
`novomodelo-python` to it; convert again a DECOMP case whose boundary FCF an
earlier bridge imported.

### Added

- **`export dessem` exports water travel time** in the `mapcut`/`cortdeco`
  files; a case with water travel time used to be refused.
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
- **New warnings in `convert`:**
  - plants whose turbined-flow limit lacks part of its `hidr.dat` data
    (`turbine-limit-data-missing`);
  - FPHA plants whose spill the source model keeps out of the tailrace, which
    Novomodelo cannot do yet (`fpha-spill-outside-tailrace`);
  - negative incremental inflows in a DECOMP deck.

### Changed

- **The FPHA is fitted as the source model fits it.** Results change for
  most FPHA plants, and the cases need the paired novomodelo release.
  - `convert newave` takes the reference volume from `volumes-referencia.csv`,
    fits turbined flow over NEWAVE's range and spillage up to twice the
    station's largest monthly mean inflow, and reads the levels of daily and
    run-of-river plants at the volume they are held at.
  - `convert decomp` fits over DECOMP's volume window and `FP` records,
    DECOMP's turbined range and twice the largest `mlt.dat` mean inflow, and
    gives a stage with a zero turbined limit zero productivity.
- **Turbined-flow and generation caps follow the source model.** `convert
  decomp` caps turbined flow stage by stage at DECOMP's own limit. `convert
  newave` caps generation at the plant's available power on every stage.
- **The bridge requires idecomp 1.17.1, pandas 3.1 and polars 2.1 or newer**,
  the releases it is tested with.
- **`convert decomp` lets `centralizado` renewable parks be curtailed**, from
  their `PEE-CONFIG-PER` state, as the cheapest way to absorb a surplus;
  results change for decks with such parks.

### Fixed

- **`convert decomp` couples the boundary cuts plant by plant, as DECOMP
  does.** Notes and warnings list the plants that borrow another plant's
  coefficients and what cannot be represented.
- **`convert decomp` reads values written without a decimal point as DECOMP
  does** (a `TG` cost `299` is 2.99).
- **`convert decomp` converts two conflicting flow restrictions on one
  plant** as soft constraints, as DECOMP does, with a warning, instead of
  failing.
- **Itaipu in `convert decomp`:** an `MP` or `FD` row without a frequency
  applies to both halves.
- **`convert decomp` reads a `dadgnl` with a single `GL` register.**
- **`convert newave` scores `level1` cut selection over the last two
  iterations**, as `lml1` did; 0.19.0 scored the whole run, which kept about
  three times as many cuts active and slowed training.
- **`compare decomp` puts DECOMP's cost breakdown at present value**, and
  adds its deficit, contracts and unitemized penalties.
- **`compare decomp` counts the ANDE load in Net Load and shows the system
  ENA in MW.**
- **`compare decomp` renders the report when DECOMP's lower bound
  overflowed.**
- **The convergence chart compares the source model's best upper bound with
  Novomodelo's best**, and `compare newave` also compares the per-iteration
  upper bounds.
- **`convert decomp` keeps the renewable parks of a `renovaveis` file
  that uses the `PEE-GER-PER-PAT` card**; they used to be dropped with no
  warning.
- **`convert decomp` converts more LIBs electrical restrictions**: those
  with `peq_` terms or with the `ger_usit` of a `dadgnl` GNL plant (its `GL`
  commanded generation), those declared only by
  `RESTRICAO-ELETRICA-INEQUACAO-PERIODO-PATAMAR` rows, and numbers such as
  `1E+31`.

## [0.19.0] - 2026-10-20

Pairs the bridge with the **novomodelo 0.19.0** release. novomodelo loads a
policy checkpoint only in the version that wrote it, so a DECOMP case whose
boundary FCF an earlier bridge imported must be converted again.

### Added

- **`export dessem` command** exports a solved Novomodelo case's policy checkpoint
  to the DECOMP-format `mapcut`/`cortdeco` binaries DESSEM reads as its future
  cost function. The command takes the DECOMP deck the case was converted
  from, `export dessem DECOMP_DIR CASE_DIR`. The deck supplies the output
  revision, the `hidr.dat` plant codes and each GNL plant's lag months from
  its `NL` record, and a deck whose operated hydro or GNL plants differ from
  the case's is refused. The checkpoint is read through
  `novomodelo-python`, cut values are converted from the run's cost scale to R$
  and then to DECOMP's 10³ R$, and each export records the conversion premises it applied in
  `export_manifest.json`; `--audit-csv` adds per-pool CSVs of the cuts as read
  and as written. GNL coefficients are written as zeros, and a policy carrying
  travel-time state is refused.
- **DECOMP: the `UH` spillage ceiling is converted.** Field 9 of the `UH`
  register, the plant's maximum spillage in m³/s, becomes a
  `max_spillage_m3s` bound in `constraints/hydro_bounds.parquet` for every
  stage; the field was ignored before. A blank field or DECOMP's default of
  1.0e+21 adds no bound, and where a single-plant `HQ` restriction on `QVER`
  also limits the plant, the lower ceiling applies. A zero ceiling is kept,
  so a plant fed only through a diversion channel, such as BELO MONTE, spills
  nothing, as in DECOMP.
- **NEWAVE: Pumping station conversion.** When the deck declares
  `ESTACAO-BOMBEAMENTO` and `ESTACAO-BOMBEAMENTO-SUBMERCADO` in `indices.csv`
  and the `ESTACOES BOMBEAMENTO` switch is on, the converter generates
  `system/pumping_stations.json` and `constraints/pumping_bounds.parquet`.
  Open-ended limit periods (blank `data_fim`) extend to the last stage.
- **NEWAVE: diversion channels.** A `CDESVIO` record in a plant's `USINA`
  block of `modif.dat` now gives that plant a diversion channel in
  `hydros.json`, to the record's target plant and up to its maximum flow, when
  the `CANAL DE DESVIO` switch of `dger.dat` is on. A maximum flow of
  9 999 999 m³/s or more, NEWAVE's value for no limit, gives a channel
  without a limit and is reported. Plants fed only by a
  channel, such as BELO MONTE below PIMENTAL, now receive that water. The
  `desvio` code of `hidr.dat` alone declares no channel. A record without a
  maximum flow, or whose target plant is not in the case, converts to no
  channel and is reported; with the switch off the records are ignored and
  reported.

### Changed

- **`novomodelo-python` is pinned to 0.19.0**, and `MIN_NOVOMODELO_VERSION`
  moves with it.
- **The bridge requires polars 2, pyarrow 26 and typer 0.27.3 or newer**, the
  releases it is tested with, so installing it into an existing environment
  upgrades them.
- **Both tracks simulate with the cuts active when training ended.**
  `config.json` now sets `simulation.cut_set` to `active` rather than leaving
  it to novomodelo's default, `all`, which also loads the cuts the selection
  deactivated and makes the simulation LPs larger than the training ones.
  `convert decomp` turns on no cut selection, so its simulation loads every
  trained cut either way. `convert newave` warns when `dger.dat` turns on the
  cut window (`janela_de_cortes`), which Novomodelo does not have: every cut is
  kept.
- **Both tracks supply each plant's own stored-energy productivity to
  novomodelo.** `system/hydro_energy_productivity.parquet` has a new
  `integrated_equivalent_productivity_mw_per_m3s` column, which novomodelo now
  requires. novomodelo sums it down the cascade for the integrated accumulated
  productivity and the maximum stored energy that VminOP and RHE read. The
  value is the source model's own convention: NEWAVE integrates the head of
  monthly reservoirs between minimum and maximum volume and takes the point
  at `volume_referencia` for the other plants, with each stage's `CFUGA` and
  `CMONT`, matching `pmo.dat`; DECOMP integrates the head of `M` and `S`
  reservoirs and takes the point for `D` plants, with each stage's `AC`
  overrides. The `equivalent_productivity_mw_per_m3s` column now sets only
  the generation productivity, so without the new column VminOP and RHE
  would weigh storage by novomodelo's integral of the plant geometry. A DECOMP
  FPHA plant now has a row as well, with a NULL generation productivity.
- **LIBS files are read from the files the deck declares, not by name.** The
  CSV inputs of the LIBS family were found by fixed file name or prefix; now
  each record is read from the file the deck's LIBS index lists for its key,
  as the source models read them. The index is `indices.csv` in a NEWAVE case
  directory and the file the `FA` register names in a DECOMP deck (the `FJ`
  register still names the tailrace-curve file on its own). A deck without an
  index has no LIBS records; an index key whose file is missing, an `FJ` that
  disagrees with the index, or renewable records spread over several files
  now fail the conversion. A file with a usual LIBS name that nothing declares
  is ignored, and `check` and `convert` warn about it on both tracks.
- **`convert decomp` pins the inflow non-negativity method to `penalty`.**
  `config.json` now carries `modeling.inflow_non_negativity.method` =
  `"penalty"` instead of `"truncation_with_penalty"`, the method 0.18.1 set
  and novomodelo's default. Its unbounded, priced inflow correction keeps the
  first forward pass feasible when a single-plant volume target (`HV`)
  becomes a hard storage floor at a later stage; under the bounded correction
  the boundary FCF import's one-iteration run can fail with an infeasible
  subproblem. `convert newave` keeps `truncation_with_penalty`.
- **VminOP (NEWAVE) and RHE (DECOMP) bound stored energy in MWh with novomodelo's
  MWh parameters.** The bridge no longer computes the integrated
  accumulated productivity itself: expressions reference novomodelo's
  `integrated_accumulated_productivity_mwh_per_hm3` (`@rho_acum_mwh_h{id}`,
  MWh/hm³) and `max_stored_energy_mwh` (`@earm_max_mwh_h{id}`, MWh). VminOP
  bounds useful stored energy,
  `Σ @rho_acum_mwh_h{id} * hydro_useful_volume_final(id) >=
  @pct_vminop_ree{code} * (Σ @earm_max_mwh_h{id})`, with the curve as a
  per-stage fraction. RHE bounds useful stored energy the same way, the base
  of the source model's `%EAMAX`: a percentage limit is a per-stage fraction
  `@pct_rhe{code}` of the maximum, and an absolute limit is converted from
  MWmês to MWh with the stage hours.
- **Every generic-constraint slack declares how novomodelo charges its penalty.**
  Each soft constraint now carries `slack.penalty_basis`, which novomodelo
  requires. VminOP and RHE are `per_unit`: the source model's penalty is
  charged per MWh of violation, so each stays one constraint per REE or
  record (`VminOP_<REE>`, `RHE_<code>`) on months of 672–744 h and on weekly
  DECOMP stages alike. Electric, `RE`, `HQ`, `HV` and LIBS constraints are
  `per_hour`, novomodelo's previous charge of penalty × the row's hours, so their
  costs do not change. The constraint families emitted after VminOP and RHE
  get lower ids than with one constraint per stage.
- **RHE (DECOMP) values each reservoir along its energy chain, as DECOMP
  does.** The chain follows the `AC JUSENA` energy link where the deck
  declares one, else the water route, so the Paraíba do Sul reservoirs count
  through NILO PECANHA, GUARAPIRANGA and BILLINGS through HENRY BORDEN, and
  ITAPARICA through COMP PAF-MOX (176), a plant outside the configuration
  that enters as a `per_stage` constant. A reservoir with an `EZ` record has
  its chain cut where it leaves its REE: the plants above the cut count in its
  own REE, and up to `EZ` % of its useful volume counts in the REE below the
  cut (SERRA MESA 55% in NORTE). A reservoir without its own `EZ` record
  counts its whole chain in its own REE. Chains that depart from the
  `hydros.json` water route are `expression` parameters over novomodelo's
  productivities, and their maximum stored energy reads the new computed
  `min_storage`/`max_storage` parameters (`@min_storage_h{id}`,
  `@max_storage_h{id}`, declared for every hydro on both tracks) instead of
  volumes written into the expression. On mar-26-rv2 every REE's maximum
  stored energy matches DECOMP's to within 0.01%; before, SUDESTE was 8.6% and NORDESTE 2.4% low and NORTE 37% high. A
  `decomp-rhe-ree-hydraulic-coupling` note lists the cut reservoirs with
  their `EZ` %. Because a generic constraint cannot cap a volume term, the
  share counts the whole volume; a `decomp-rhe-ez-cap-unmodeled` warning names
  any stage where that can make the constraint looser than DECOMP's. The
  `check decomp` note that `EZ` records are ignored is gone.
- **`compare` and `dashboard` read generic constraints from novomodelo's echo.**
  The Constraints tabs take each term's coefficient and each row's bounds
  from `output/generic_constraints/resolved_echo.parquet`, as the LP received
  them, instead of re-deriving them from the expression text and
  `generic_parameters.json`. VminOP and RHE are shown as one constraint per
  REE or record across their stages, and each penalty says whether it is
  charged per hour or per unit; VminOP reads as useful stored energy in MWh
  against the curve fraction of its maximum, and the source-model side
  weights `VARMUH` by the same coefficients. The DECOMP RHE achieved value is
  converted from stage-average MW to MWh. A constraint with a term the
  simulation output cannot value is left out of the novomodelo-side LHS instead of
  being evaluated partially.
- **`generic_parameters.json` declares the MWh per-hydro parameters.** Every
  hydro also gets `rho_eq_mwh_h{id}`, `rho_acum_mwh_h{id}` and
  `earm_max_mwh_h{id}` (novomodelo's `integrated_equivalent_productivity_mwh_per_hm3`,
  `integrated_accumulated_productivity_mwh_per_hm3` and `max_stored_energy_mwh`),
  and `rho_acum_h{id}` is always novomodelo's computed point productivity.
- **`convert newave` turns cut selection on with `level1`.** novomodelo removed
  the `lml1` method; the converted `config.json` now names `level1`, checked
  every iteration over the trial states of the whole run. Level-1 selection
  keeps exactly the cuts that are highest at some stored trial state, so
  results can differ from a run made with `lml1`.
- **`convert decomp` writes the boundary FCF in novomodelo's current policy
  format.** The checkpoint no longer carries a cut id or a pool's list of
  active cuts; each cut states its own activity.
- **`dashboard` reads novomodelo's per-pool cut-selection report.** The cut counts
  of a stage now sum the row pools of its nodes, and the pool size is read
  from `cuts_in_pool`.

### Fixed

- **The README links the published novomodelo documentation**,
  <https://docs.modelossetoreletrico.org.br/>, which the package's PyPI page
  shows too.
- **DECOMP: Itaipu's 50 Hz and 60 Hz `RE` limits bound only their own half.**
  A single-term `RE` whose `FU` term names a frequency now bounds that
  frequency's unit group in `constraints/hydro_unit_group_bounds.parquet`.
  Before, it bounded the whole plant in `constraints/hydro_bounds.parquet`,
  which capped Itaipu's total at one half's limit once novomodelo started capping
  a plant's total across its buses, and made novomodelo warn that the plant
  minimum had no effect. A ceiling above the half's installed capacity is
  clamped to it with one warning per unit group, and a frequency on any plant
  other than Itaipu stops the conversion.
- **NEWAVE: `CFUGA`/`CMONT` levels after the last record now follow the
  source model.** During the study, each `modif.dat` record holds until the
  next one. With the `dger.dat` seasonal flag on, the post-study repeats the
  last study year month by month. Previously, every month after the last
  record took the latest record naming that calendar month in any year, so a
  month declared only in an early year came back in the last study year and
  in every post-study year. That record could be, for example, Tucuruí's
  September and October tailrace in the first year. Per-stage productivity in
  `hydro_energy_productivity.parquet`, the generation caps and the
  stored-energy productivity change for those plants and stages.
- **`convert newave` gives every plant on a shared gauging station its own
  inflow series, and zero inflow on the artificial gauge.** Inflows were
  attributed per station, so when several plants shared one, only the last
  received a history and the others failed Novomodelo's validation, and the upstream
  stations of all of them were subtracted from that one plant. A deck that puts
  several plants on posto 300, NEWAVE's artificial gauge for plants without
  natural inflow, gave one plant a negative series and left the rest without
  history. Each plant now subtracts only its own upstream stations; a plant on
  posto 300 gets zero history, seasonal statistics and recent inflows, and an
  artificial plant upstream is not subtracted. Both tracks list the plants on
  the artificial gauge, and warn when a gauged plant sits below an artificial
  plant that has gauged plants further upstream. Cases without shared stations
  convert as before.
- **NEWAVE: every record of `modif.dat` is read.** The last record of the
  file was dropped, a plant with more than one `USINA` block kept only its
  first, and a file with a single block failed to convert. The records of all
  blocks now apply in file order, so PIMENTAL in a deck whose second block
  carries the `VAZMINT` series now gets those minimum outflows. A dated
  record that repeats a month now replaces the earlier one, where the larger
  value used to win, and before the study start the latest record is the one
  in force.
- **`convert decomp` routes bounded diversions to their target plant.** A
  plant whose single-term `QDES` restriction allowed a positive diversion but
  set no positive floor got diversion bounds and no Novomodelo diversion channel,
  so Novomodelo removed the diverted water from the system instead of passing it
  to the plant that receives it (such as TOCOS to LAJES and VIGARIO to
  FONTES AB). Every plant whose diversion can be positive now gets the
  channel to its `AC DESVIO` or `hidr.dat` target, limited by the `AC DESVIO`
  flow, and the `QDES` bounds still apply stage by stage. A floored plant
  whose target has no `AC DESVIO` limit now gets its channel instead of being
  reported. When the target plant is not in the case, the diversion ceiling
  is closed and reported, so the water stays in the plant's own river.
- **`convert decomp` leaves a diversion channel unlimited when DECOMP does.**
  A `desvio` without an `AC DESVIO` limit, or with the "no limit" value of
  9 999 999 m³/s or more, was capped at the receiving plant's turbine
  capacity, which DECOMP's own solutions exceed (MOXOTO to P.AFONSO 4). The
  channel now carries no limit, and a base `desvio` no longer writes a
  `max_diversion_m3s` bound. Channels given the "no limit" value are reported
  on both tracks.
- **`convert` refuses a case where diverted water reaches no plant.** Both
  `convert newave` and `convert decomp` now fail before writing when a plant
  can divert water (a positive diversion floor or ceiling) but has no
  diversion channel to a receiving plant, which Novomodelo rejects. The failure
  names each plant, the bound and its stages.
- **NEWAVE: Hydro plants referenced by pumping stations are no longer
  classified as fictitious.** Previously, zero-productivity reservoirs used
  for pumped storage (e.g., VIGÁRIO) were removed, which could break cascade
  topology and cause a RecursionError.
- **NEWAVE: Cascade cycle detection now raises a clear error instead of
  RecursionError.** The error names the cycle path and points to the fictitious
  classification as the likely cause.
- **`convert decomp` reads special-constraint coefficients stage by stage.**
  Each `FU`/`FT`/`FI`, `CQ` and `CV` line declares a coefficient from its
  `estagio` onward, but the reader kept a term's first coefficient for every
  stage and counted a term from the constraint's first stage even when the
  deck declares it later. A coefficient that changes between stages, or a
  term that joins later and is absent before that stage, as DECOMP reads it,
  is now a `per_stage` parameter `@coef_<family><code>_t<k>` in
  `generic_parameters.json`; a constant coefficient stays a literal, so decks
  without such lines convert as before. A single-term constraint lowers to an
  entity bound only when its coefficient is ±1 at every stage. A `RE`, `HQ`
  or `HV` with no coefficient at its first stage, a deck DECOMP rejects, now
  fails the conversion.
- **`convert newave` applies the FPHA plane reduction only when the case
  declares it.** A `tratamento-fpha.csv` in the case directory set
  `fpha_plane_reduction` even when `indices.csv` left out or commented out
  `HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES`, the line that makes NEWAVE use
  it. Such a case now converts without the block, so Novomodelo keeps every FPHA
  plane, and FPHA results change for those cases.
- **A commented-out LIBS index line no longer triggers the DECOMP
  electrical-constraint warning.** The check scanned the raw index text, so
  an `&RESTRICAO-ELETRICA-ESPECIAL` line counted as present.
- **`convert newave` reads a Latin-1 `tratamento-fpha.csv`.** The file was
  read as UTF-8 only, so a Latin-1 one failed the conversion, while the
  electrical-constraint file was read as Latin-1 only. Decks mix the two
  encodings, so each of these files, and the LIBS index on both tracks, is
  now read as UTF-8 when it is valid UTF-8 and as Latin-1 otherwise.

## [0.18.1] - 2026-10-15

Pairs the bridge with the **novomodelo 0.18.1** release. novomodelo loads a
policy checkpoint only in the version that wrote it, so a DECOMP case whose
boundary FCF an earlier bridge imported must be converted again.

### Added

- **`check` and `convert` warn about tabs in fixed-column inputs**, on both
  tracks. The NEWAVE text files `arquivos.dat` names and DECOMP's
  `dadger`/`dadgnl` are read by column position, so a tab where the layout expects spaces shifts the fields
  after it; a DECOMP register line that starts with a tab is not read at all.
  The warning names each file and its lines (DECOMP comment lines are
  skipped), and a `check` that finds one reports ready with warnings.

### Changed

- **`novomodelo-python` is pinned to 0.18.1**, and `MIN_NOVOMODELO_VERSION`
  moves with it.
- **`convert decomp` sets the inflow non-negativity method.** `config.json`
  now carries `modeling.inflow_non_negativity.method` =
  `"truncation_with_penalty"`, the method `convert newave` already sets,
  instead of leaving it to novomodelo's default, which 0.18.1 moves from
  `penalty` to that method. The inflow slack is capped at the amount a plant's
  incremental inflow falls below zero, so it is zero wherever the tree's
  inflow is non-negative; under `penalty` it could add water at any inflow.
  DECOMP results can change where the LP used that slack.
- **NEWAVE comparisons can show negative novomodelo inflows.** From 0.18.1,
  novomodelo's truncation methods keep the PAR inflow raw: `inflow_m3s`,
  `incremental_inflow_m3s` and `incremental_inflow_energy_mw` report the raw
  draw, which can be negative, and `inflow_nonnegativity_slack_m3s` reports the
  correction that lifts it to zero in the water balance. `compare newave`
  compares the values novomodelo reports, so where the correction acts, the
  incremental inflow, the total inflow and the natural inflow energy are lower
  than under 0.18.0.

### Fixed

- **`convert decomp` fits each plant's FPHA over its whole storage range.**
  The fitting window was a band of ±10 % of useful volume around the initial
  storage. Below the band, novomodelo's production planes are negative at zero
  turbined flow, so a plant drawn down past the band's lower edge had to
  turbine a minimum flow; where its inflow could not supply that flow, the
  stage was infeasible and the boundary FCF import failed with "infeasible
  subproblem". Under the `penalty` inflow method the case solved, but the
  inflow slack supplied the missing water and inflated the costs. The window
  and the `hydro_geometry.parquet` volume grid now span the plant's storage
  envelope over the study (a single reference volume for a run-of-river
  plant), as `convert newave` fits its reservoirs over their whole storage
  range. DECOMP results change for every FPHA plant whose band did not cover
  its storage range.
- **A failed `convert` keeps the warnings found before the failure.** Both
  tracks reported only the error, so a warning that explained it, such as a
  tab that misaligned a fixed-column file, was lost. The rendered notes and the
  `--json` verdict now list those warnings next to the error.
- **`dashboard` builds for a case run with the simulation disabled.**
  novomodelo creates the `output/simulation/` directories even when it does not
  simulate, and the dashboard failed on the empty inflow-lag and
  generic-violation directories instead of treating the case as
  training-only.
- **`dashboard` builds for a simulation-only case.** A run with training
  disabled, simulating from a stored policy, writes no
  `output/training/convergence.parquet`, which the dashboard required. Such a
  case now gets no Training tab, its Overview takes the version, run date and
  duration from the simulation metadata, and its Performance tab leaves out
  the training sections and cards. A training-only case likewise gets no
  Simulation section or cards in the Performance tab.
- **Overview quick-look links highlight the tab they open.** They addressed
  the navigation buttons by position, so the wrong button was highlighted
  whenever the set of tabs differed from the one the positions assumed, as
  it does when the Stochastic Model tab is present.

## [0.18.0] - 2026-10-08

### Changed

- **BREAKING:** the product is renamed. The CLI and the package are
  `novomodelo-bridge` (`import novomodelo_bridge`), the configuration file is
  `novomodelo-bridge.toml`, environment variables such as
  `COBRE_BRIDGE_OUT_DIR` are now `NOVOMODELO_BRIDGE_OUT_DIR`, and the bridge
  converts cases for, and reads the outputs of, novomodelo 0.18.0 through
  `novomodelo-python`. Everything else behaves as in cobre-bridge 0.18.0.

# cobre-bridge history

## [cobre-bridge 0.18.0] - 2026-10-07

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
