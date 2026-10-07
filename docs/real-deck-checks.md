# Real-deck checks

This page is for contributors. It catalogues the end-to-end checks that used
to live in the test suite as tests guarded on real NEWAVE and DECOMP decks,
explains why they were removed, and records what each one needs so it can be
brought back in a form that runs for everyone.

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
| One test in `test_inflow_mlt.py` | Incremental long-term-mean inflows cover every operated plant, read zero for the diversion-fed plant on the artificial station (code 288), and carry no increment below -30 m³/s. |
| Nine result-reader tests in `tests/comparators/test_decomp_readers.py` | Each reader returns rows with the expected columns on the deck's real result files. Three of them (`dec_oper_gnl.csv`, `dec_oper_ree.csv`, `decomp.tim`) now run against committed excerpts; the rest are listed under "Excerpts still to capture". |

### The reduced deck plus its solved novomodelo case (`novomodelo-mar-26-rv2-reduced`)

| Removed check | Verified |
| --- | --- |
| Ten tests in the former `test_decomp_full_report_smoke.py` (now `tests/comparators/test_retired_decomp_symbols.py`) | `compare decomp` builds the full multi-tab report end to end: every navigation tab is present; the energy balance, network, cost, performance, constraints, REE, evaporation, and FPHA sections render with data; the `DECOMP` reference label leaves no `NEWAVE` literal. |
| One test in `tests/dashboard/test_energy_balance.py` | The block-hours-weighted system spillage rate on the solved case equals a pinned value (59,001.5 m³/s at capture), guarding against summing per-block rows without weighting. |

### A DECOMP deck with electrical-constraint files (`decomp-abr-26-lpp`), plus a novomodelo binary

| Removed check | Verified |
| --- | --- |
| One test each in `test_libs_electrical_emit.py` and `test_libs_electrical_pipeline.py` | Conversion emits at least one `LIBS_ELEC_*` generic constraint with matching bound rows, the unresolved-operand census stays below the pre-fix baseline of 20, and `novomodelo validate` exits 0 on the converted case. |

### Sibling novomodelo checkout (replaced, not removed)

Four tests in `tests/decomp/test_contracts.py` read novomodelo's
`energy_contracts.schema.json` and its `d41-energy-contracts` example from a
checkout under the developer's home directory. Those files are now vendored
under `tests/fixtures/novomodelo_schemas/` and `tests/fixtures/novomodelo_d41/`, so the
tests run everywhere. Refresh both copies from the novomodelo repository whenever
`MIN_NOVOMODELO_VERSION` moves.

## Excerpts still to capture

These would restore the remaining reader checks as ordinary tests. Take the
file from any deck that has it, keep the header and the first few stages'
rows, confirm the reader parses it, and commit it under
`tests/fixtures/decomp_results/`.

| Reader | File | Columns the removed test asserted |
| --- | --- | --- |
| `read_relato_balance`, `read_relato_costs`, `read_relato_expected_cost`, `read_relato_membership` | `relato.rvN` | rows present; membership has `codigo_usina` and `codigo_ree`. The parser needs the complete report, so a head excerpt does not work; a block-wise excerpt tool or a deliberately larger fixture is required. |
| `read_dec_desvfpha` | `dec_desvfpha.rvN` | `codigo_usina`, `estagio`, `volume_total_hm3`, `vazao_turbinada_m3s`, `vazao_vertida_m3s`, `geracao_hidraulica_fpha` |
| `read_dec_estatfpha` | `dec_estatfpha.rvN` | `variavel`, `valor` |
| `read_dec_oper_evap` | `dec_oper_evap.csv` | `evaporacao_calculada_hm3`; no `patamar` column |
| `read_dec_oper_rhesoft` | `dec_oper_rhesoft.csv` | `codigo_restricao`, `limite_MW`, `valor_MW`, `violacao_absoluta_MW` |

The removed `read_eco_fpha` check asserted only that a particular deck had no
such file; the absent-file contract is covered synthetically and needs no
excerpt.

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
