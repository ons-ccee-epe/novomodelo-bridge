# Real-deck checks

This page is for contributors. It catalogues the end-to-end checks that used
to live in the test suite as tests guarded on real NEWAVE and DECOMP decks,
explains why they were removed, and records what each one needs so it can be
brought back in a form that runs for everyone. It also records checks that
were only ever run by hand on a real deck.

## Why they were removed

The suite carried tests that skipped unless a real deck was present under the
gitignored `example/` tree, and in some cases unless a locally built `novomodelo`
binary sat at a fixed path under the developer's home directory. Four more
tests read a schema and an example case from a sibling checkout of the novomodelo
repository. None of those inputs is part of the repository, so:

- CI never ran any of them, and on any checkout whose local decks differed
  from the author's they skipped silently. A test that has passed on one
  machine at one point in time is a snapshot, not a regression guard, and a
  permanent skip reads as coverage in the tree.
- Their pinned numbers came from decks no reviewer could open, so nobody could
  tell whether a pin encoded correct behaviour or a bug of the day.

The code they exercised keeps its unit tests on synthetic inputs. What was
lost is the check against real data, which is what this page exists to
restore properly.

## The policy now

Every test runs from the repository alone. `tests/test_local_data_policy.py`
fails the build if any test module builds a path into `example/` or resolves
one through the developer's home directory. Real-format inputs are committed
as small excerpts under `tests/fixtures/` (NEWAVE result files, DECOMP result
files, novomodelo's contract schema and its example case), and the synthetic
mini-decks under `tests/decks/` drive the end-to-end conversions.

The removed test bodies remain in git history. To read one:

```bash
git log -S"<test function name>" --oneline -- tests   # the commit that removed it
git show <that commit>^:tests/<path to the module>    # the module as it was
```

## What was removed, and what each check needs

The deck identifiers below name directories that lived under `example/` on
the author's machine. They are not in the repository and may not exist
anywhere any more; treat each as "a deck with these properties".

### A full NEWAVE case (`newave_rodada`, with its `cortesh.dat` and `cortes.dat`)

| Removed check | Verified |
| --- | --- |
| `test_rule43_regression.py` (whole module) | Converting the case yields no `hydro_bounds.parquet` row whose `max_turbined_m3s` or `max_generation_mw` exceeds the plant's declared value in `hydros.json` (novomodelo's semantic rule 43), over at least a thousand rows. The in-pipeline mirror, `check_hydro_bounds_no_raising` in `src/novomodelo_bridge/core/emission_checks.py`, keeps its unit tests in `tests/core/test_emission_checks.py`. |
| `TestConvertLineBoundsRealDeckFidelity` in `test_convert_network.py` | Every per-block line-bound row equals the base capacity times the block factor recomputed independently from `sistema.dat` and `patamar.dat`, and the row count equals the number of genuinely differing (line, stage, block) combinations. |
| Six real-deck tests in `test_fcf_cortes.py` | Cut-file readers on a non-GNL deck: header facts (plant count, individualized layout, no GNL lag), record shapes at the boundary stage (one storage coefficient per plant, twelve inflow lags per plant, no GNL slots), per-cut provenance fields, which plants carry nonzero storage and lag coefficients, and the cut-family summary. |

### A full DECOMP deck with its cut files (`decomp-mar-26-rv2`), plus a novomodelo binary

| Removed check | Verified |
| --- | --- |
| Three real-deck tests in `test_fcf_cortes.py` | Cut-file readers on a hybrid deck (aggregated stages then plant-space cuts): plant count after excluding fictitious plants, GNL slot width, boundary stage taken from the trailer, record count and raw column count, zero SAR columns, right-hand-side magnitude. |
| Three end-to-end tests in `test_fcf_importer.py` | Convert plus boundary import: every anticipated plant's covered ring lane is priced and its in-study lane is not; the post-study calendar has the expected shape; `novomodelo validate` reports zero case errors; a bounded `novomodelo run <case>` loads the boundary with no panic, no empty-lag warning, and no dropped delivery; the operator command `novomodelo-bridge convert decomp` run as a subprocess authors a populated boundary and reports it in the `--json` verdict. |

### The reduced DECOMP deck (`decomp-mar-26-rv2-reduced`, a two-leaf terminal fan)

| Removed check | Verified |
| --- | --- |
| One end-to-end test in `test_fcf_importer.py` | The bootstrap resolves a real terminal node id (never the -1 sentinel), `policy.boundary.source_stage` equals that graph stage, and the already-committed months carry no dated ring slot while a signalled month does. |
| Two tests in `test_anticipated.py` | The already-committed set derived from the `gl` records matches the prior-revision markers in `relgnl` for SANTA CRUZ; a full conversion emits the seven-stage post-study calendar, prunes the post-study thermal bounds to the signalled stages, tiles the committed windows with the deck's own MW, and keeps the lead uncapped. |
| `test_anticipated_e2e.py` (whole module; also needs `novomodelo-python`'s checkpoint writer) | The CLI conversion with boundary import trains and simulates in-process; the anticipated-lanes output partition is populated; the post-horizon split matches the deck's declaration and the run reaches every post-study delivery. |
| Nine result-reader tests in `tests/comparators/test_decomp_readers.py` | Each reader returns rows with the expected columns on the deck's real result files. Four of them (`dec_oper_gnl.csv`, `dec_oper_ree.csv`, `decomp.tim`, `dec_desvfpha.rvN`) now run against committed excerpts; the rest are listed under "Excerpts still to capture". |

### The reduced deck plus its solved novomodelo case (`novomodelo-mar-26-rv2-reduced`)

| Removed check | Verified |
| --- | --- |
| Ten tests in the former `test_decomp_full_report_smoke.py` (now `tests/comparators/test_retired_decomp_symbols.py`) | `compare decomp` builds the full multi-tab report end to end: every navigation tab is present; the energy balance, network, cost, performance, constraints, REE, evaporation, and FPHA sections render with data; the `DECOMP` reference label leaves no `NEWAVE` literal. |
| One test in `tests/dashboard/test_energy_balance.py` | The block-hours-weighted system spillage rate on the solved case equals a pinned value (59,001.5 m³/s at capture), guarding against summing per-block rows without weighting. |

### A DECOMP deck with electrical-constraint files (`decomp-abr-26-lpp`), plus a novomodelo binary

| Removed check | Verified |
| --- | --- |
| One test each in `test_libs_electrical_emit.py` and `test_libs_electrical_pipeline.py` | Conversion emits at least one `LIBS_ELEC_*` generic constraint with matching bound rows, the unresolved-operand census stays below the pre-fix baseline of 20, and `novomodelo validate` exits 0 on the converted case. |

### A DECOMP deck carrying `VI` water travel time, plus a novomodelo binary

| Check | Verifies |
| --- | --- |
| Convert a deck with at least one operated `VI` arc, then `novomodelo run` the case to convergence | The transit-bucket cut generation produces a valid policy: training converges on a non-negative gap (no lower bound driven past the upper bound), and the solved case prices the in-transit water. `novomodelo validate` alone does not exercise this — it checks the `past_defluences` coverage rule at config time but not the cut arithmetic — so this needs the solver binary, not just the validator. The emission shape is unit-tested in `test_travel_time.py` and the loadability in `test_convert_decomp_e2e.py` (tier-2, `novomodelo validate`); this whole-deck run is what closed the former novomodelo cut-generation defect (`C-travel-time`). |

### Sibling novomodelo checkout (replaced, not removed)

Four tests in `tests/decomp/test_contracts.py` read novomodelo's
`energy_contracts.schema.json` and its `d41-energy-contracts` example from a
checkout under the developer's home directory. Those files are now vendored
under `tests/fixtures/novomodelo_schemas/` and `tests/fixtures/novomodelo_d41/`, so the
tests run everywhere. Refresh both copies from the novomodelo repository whenever
`MIN_NOVOMODELO_VERSION` moves.

## Checks run by hand

These were never tests. They are recorded so they can be repeated.

### The reduced DECOMP deck, run by DECOMP with its coupling report

| Check | Verified |
| --- | --- |
| Boundary coupling convention | Run with `IR ACOPLA` and `IR CORTES` added to a copy of `dadger.rvN`, DECOMP reports each boundary cut's terms and the inflows it evaluates them on. In each terminal scenario, the largest cut re-evaluated from the deck alone reproduces the cost-to-go value DECOMP reports to within a relative 5×10⁻⁷: the constant term, plus the storage terms of the `M` and `S` plants above their minimum volume, plus the inflow-lag terms on the coupling gauges, all scaled by the mean month, plus the GNL terms. Copying a downstream reservoir's storage price onto in-transit water breaks the match, so that deck's boundary prices no water in transit. Captured with DECOMP 32.0.1 in October 2026. |

### The reduced DECOMP deck with water travel time, exported back to DECOMP's cut files

| Check | Verified |
| --- | --- |
| `export dessem` against DECOMP's own `mapcut` and `cortdeco` | DECOMP run on the reduced deck writes its own pair, and `convert decomp`, `novomodelo run` and `export dessem` on the same deck write the bridge's. Record 6's integers through the lag-start array and every travel-time record match byte for byte, and each node's latest cut values the in-transit water within 0.2% of DECOMP's. Two runs on edited copies of `dadger.rvN`, one with a `VI` travel time of 100 h and one with the two `VI` records swapped, fixed the conventions the unit tests pin in `test_travel_time.py`, `test_layout.py` and `test_mapcut_writer.py`: the lag array is stage-major with plants inner, the plants are numbered in `VI` record order, and the in-transit coefficients are lag-major with plants inner. Captured with DECOMP 32.0.1 in October 2026. |

### A full DECOMP deck with water travel time, whose last stages leave transit buckets undated

| Check | Verified |
| --- | --- |
| `export dessem` on transit buckets past the study end | On a seven-stage deck whose two plants, each with a 360 h travel time, leave transit buckets undated in its last stages, the export completes, reports three undated buckets in the travel-time note and, in its log, one in the fifth stage's pool and two in the sixth's, raises no dropped-coefficient warning (every cut's coefficient on them is zero), and writes a pair that passes the read-back check. The first four stages' pools carry no undated bucket, so their in-transit coefficients come from the dated path, which this change leaves as it was. Captured with novomodelo 0.19.1 in October 2026. |

### The reduced DECOMP deck and a monthly-stage deck, run by DECOMP with its turbined-flow report

| Check | Verified |
| --- | --- |
| Turbined-flow limit against DECOMP's `avl_turb_max.csv` | With `IR ARQCSV`, DECOMP writes `avl_turb_max.csv`, whose `Qmax dc pl` column is the turbined-flow limit it applies to each plant on each stage. The bridge side is the sum of the plant's unit-group limits for the stage, read from `hydro_unit_group_bounds.parquet` or else the group's envelope in `hydros.json`; for Itaipu it is capped by the plant's `max_turbined_m3s` row in `hydro_bounds.parquet`. Every plant matches within 0.5% on every stage DECOMP reports, on both decks, and no fixed point fails to converge. The largest difference is Itaipu, above DECOMP's limit, because the conversion leaves its `VL` lateral inflow out of the tailrace level. Captured with DECOMP 32.0.1 in October 2026. |

### The reduced DECOMP deck and a monthly-stage deck, run by DECOMP with their FPHA reports

| Check | Verified |
| --- | --- |
| DECOMP's plane form and exact production function | DECOMP's planes in `avl_cortesfpha_dec.rvN`, evaluated with the storage above the `hidr.dat` minimum and the `VU` lateral flow of the same block, reproduce the `Ghid (FPHA)` column of `dec_desvfpha.rvN`, so the comparison reads them in that form. The exact production function rebuilt from the deck (cota polynomial, `polinjus` families read at the downstream plant's printed forebay, `VL`, `VU` and `VA` flows, `hidr.dat` losses, the `eco_fpha_.rvN` generation limit) reproduces its `Ghid (FPH)` column on the reduced deck within 0.004 MW at the median point and 0.62 MW at the worst, a plant with a drowned tailrace. The monthly deck's run writes no `dec_desvfpha.rvN`, which is why the comparison does not depend on it. The unit tests in `tests/comparators/test_decomp_operating_points.py` pin the same conventions on one plant's committed excerpt. Captured with DECOMP 32.0.1 in October 2026. |

### A NEWAVE FPHA run with every plant's nwlistop tables, plus its solved novomodelo case

| Check | Verified |
| --- | --- |
| NEWAVE's operating-point conventions against its printed heads and exact production | With the `nwlistop.dat` plant list set to `999`, nwlistop writes each plant's per-block generation, flows, storage, heads (`hmont`, `hjus`, `hliq`) and exact production (`gh_fphexat`). The planes in `fpha_cortes.csv`, evaluated with the stage's mean storage above the `hidr.dat` minimum, reproduce `ghiduh`. The exact production function rebuilt from the deck (cota polynomial at that storage, `polinjus` families read at the downstream plant's forebay at the same point, spill in the tailrace only where `hidr.dat` says it reaches it, `hidr.dat` losses, the `fpha_eco.csv` generation limit) reproduces `gh_fphexat` for every plant and stage within 0.07 MW, the print resolution of a block's MWmes divided by its share of the month. `compare newave` reports that gap in the fit-inputs table, so rerunning it on such a run repeats the check. The unit tests in `tests/comparators/test_newave_operating_points.py` pin the storage, generation and tie conventions on the mini-deck. Captured on a September 2024 case with a historical final simulation, NEWAVE 30.0.4, in October 2026. |

### A LIBs shadow-process DECOMP deck, with its parks on either generation card

| Check | Verified |
| --- | --- |
| `PEE-GER-PER-PAT` read like `PEE-GER-PER-PAT-CEN` | The deck declares its parks' generation on the scenario-free `PEE-GER-PER-PAT` card. A copy with the same generation rewritten on `PEE-GER-PER-PAT-CEN`, the scenario column added, converts to the same renewable registry, `non_controllable_stats.parquet` and `external_ncs_scenarios.parquet`. A bridge that reads only `PEE-GER-PER-PAT-CEN` converts the original deck with its `PQ` series alone, and warns about nothing. Captured in October 2026. |

### The same shadow-process deck, with `centralizado` and `fixo` parks

| Check | Verified |
| --- | --- |
| `PEE-CONFIG-PER` operating state against curtailment | The deck's `PEE-CONFIG-PER` declares each park `centralizado` or `fixo`. Matching each park to its converted source by name, every `centralizado` park and no other carries `allow_curtailment: true`; the `fixo` parks and every `PQ` series stay must-run, and `penalties.json` prices curtailment at `PCORTEOL`. Captured in October 2026. |

### Two LIBs DECOMP decks, with `peq_` terms, period-only inequalities and load-driven activation rules

| Check | Verified |
| --- | --- |
| LIBs electrical restrictions | Every restriction of the shadow-process deck's `lib_restricao-eletrica-especial.csv` converts, among them the ones reading `peq_<name>_<subsystem>` series and the ones declared only by `RESTRICAO-ELETRICA-INEQUACAO-PERIODO-PATAMAR` rows. On a copy of `DEC_ONS_062026_RV0_VE_LPPs_COMPLETAS`, whose `RESTRICAO-ELETRICA-HABILITA` rules compare the `DP` load and aliases, the restrictions the conversion leaves inactive are the ones an independent evaluation of every rule, stage and block finds inactive, and the restrictions whose limit is `1E+31` convert with that side unbounded. Captured in October 2026. |

### The same copy of the complete deck, with the GNL plant's `GL` generation varied

| Check | Verified |
| --- | --- |
| GNL `ger_usit` abated with its `GL` commanded generation | On the copy of `DEC_ONS_062026_RV0_VE_LPPs_COMPLETAS`, the restrictions whose `GPC`/`GPC2` expressions include the `ger_usit` of the `dadgnl` GNL plant convert, and their bound is the deck's limit less the formula coefficient times the plant's commanded generation. A further copy with that plant's `GL` generation made different per block, and in one week of the monthly stage, moves each such bound in every active cell by its formula coefficient times the change in the block's commanded value, the monthly stage taking the `duracao`-weighted mean of its `GL` weeks; no other restriction moves. Captured in October 2026. |

## Excerpts still to capture

These would restore the remaining reader checks as ordinary tests. Take the
file from any deck that has it, keep the header and the first few stages'
rows, confirm the reader parses it, and commit it under
`tests/fixtures/decomp_results/`.

| Reader | File | Columns the removed test asserted |
| --- | --- | --- |
| `read_relato_balance`, `read_relato_costs`, `read_relato_expected_cost`, `read_relato_membership` | `relato.rvN` | rows present; membership has `codigo_usina` and `codigo_ree`. The parser needs the complete report, so a head excerpt does not work; a block-wise excerpt tool or a deliberately larger fixture is required. |
| `read_dec_estatfpha` | `dec_estatfpha.rvN` | `variavel`, `valor` |
| `read_dec_oper_evap` | `dec_oper_evap.csv` | `evaporacao_calculada_hm3`; no `patamar` column |
| `read_dec_oper_rhesoft` | `dec_oper_rhesoft.csv` | `codigo_restricao`, `limite_MW`, `valor_MW`, `violacao_absoluta_MW` |

The removed `read_eco_fpha` check asserted only that a particular deck had no
such file. The absent-file contract is covered synthetically, and the reader
now also runs against a committed `eco_fpha_.rvN` excerpt.

## Restoring the whole-deck checks

The checks in the tables above genuinely need a whole deck, a solved novomodelo
case, or the solver binary. They are worth having, as a separate suite that
runs where the data is rather than as skipped tests in the default one:

1. Host the decks and solved cases outside git, in a location every
   contributor can fetch from, with one script that downloads them to a
   directory of the caller's choice.
2. Read that directory and the solver path from environment variables (for
   example `NOVOMODELO_BRIDGE_DECK_ROOT` and `NOVOMODELO_BIN`) instead of fixed paths,
   and select the suite with a dedicated pytest marker so it is never mixed
   into the default run.
3. Run it on a schedule or on demand in CI with the fetch step, so a failure
   is seen and a skip is impossible.
4. Attach the deck identifier and the capture date to every pinned value, so
   a reviewer knows what the number is a fact about.

Until that exists, this page is the record of what those checks proved.
