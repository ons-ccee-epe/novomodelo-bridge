# The NEWAVE track

`convert newave`, `check newave`, and `compare newave` work on a NEWAVE case
directory. This page describes what the converter reads, what it changes on
the way to a Novomodelo case, and how to read a comparison. Flags and their help
text are in the [CLI reference](cli.md).

## Inputs

A case is discovered the way NEWAVE itself does it: `caso.dat` names the file
index (`arquivos.dat` by default) and the index names every other data file.
File-name lookups are case-insensitive. The two binary registries the index
does not list, `hidr.dat` and `vazoes.dat`, are found by scanning the
directory.

Many inputs are optional (`modif.dat`, `cvar.dat`, `curva.dat`, `agrint.dat`,
and others). `check newave` prints one line per optional input that is absent
and states that the conversion proceeds without it. A required input that is
missing or unreadable makes `check` exit 2, and `convert` fails before writing
anything. Run `check newave` first when a case comes from an unfamiliar
source.

Several optional inputs are also gated by a switch line in `dger.dat`, such
as `CONSIDERA GHMIN` for `ghmin.dat`, `AGRUPAMENTO LIVRE` for `agrint.dat`,
`CONS. CARGA ADICIONAL` for `c_adic.dat`, the two `RESTRICOES ELETRICAS`
lines for `re.dat` and `restricao-eletrica.csv`, `DESCONSIDERA VAZMIN` for
minimum outflow, and `REST. TURBINAMENTO` for the dated turbining records of
`modif.dat`. A file that is present but switched off is ignored, as NEWAVE
ignores it, and both `check newave` and `convert newave` say so with an
informational diagnostic. A switch line absent from `dger.dat` counts as on.

The field-by-field map from deck files to the converted case, including what
the converter does not convert yet, is the [NEWAVE data map](newave-data-map.md)
(in Portuguese).

## What the converter changes

Novomodelo's input format differs from NEWAVE's in ways the converter has to
resolve. Every step is deterministic: converting the same case twice produces
the same output.

- **Entity ids.** NEWAVE identifies plants, subsystems, and interchanges by
  arbitrary 1-based codes; Novomodelo uses dense 0-based ids. The converter sorts
  the source codes and assigns 0-based ids in that order, consistently across
  every output file. `compare newave` rebuilds the same mapping from the
  source case, so results trace back to the source codes.
- **Which plants exist.** A hydro becomes a Novomodelo entity when `confhd.dat`
  marks it in service — either already operating (`EX`) or operating with an
  expansion still to come (`EE`). Plants that do not yet exist are left out,
  except a future plant whose `exph.dat` schedule has it filling its dead
  volume during the horizon. NEWAVE's fictitious accounting plants are also
  removed. They are identified structurally, as a zero-productivity plant
  sharing its inflow gauge with a generating plant, rather than by the `FICT.`
  name prefix, so a cascade reduced to a subset of plants still classifies
  correctly. Cascade links that ran through a removed plant are rewired to the
  next real plant downstream, preserving the water-balance topology.
- **Plants under expansion.** An `EE` plant operates from the first stage at
  the machine configuration `modif.dat` declares for the study start, and
  reaches the `hidr.dat` configuration as the machines listed in `exph.dat`
  enter service. The capacity before each entry is written as a per-stage
  bound, and the plant declares the configuration it ends with. When the deck
  declares no study-start configuration, the converter takes the registry minus
  the machines still to enter rather than crediting them twice.
- **Horizon.** The study horizon comes from `dger.dat`: start month and year,
  study years, post-study years. Every per-stage table is sized to it. Data
  NEWAVE provides only for the study years (loads, block factors, some bounds)
  is extended into the post-study years by repeating the last study year's
  seasonal values.
- **Constraints.** The minimum-storage security curve (`curva.dat`), electric
  constraints (`restricao-eletrica.csv`), and interchange group limits
  (`agrint.dat`) all become Novomodelo generic constraints: linear expressions over
  storage, generation, and exchange variables with per-stage bounds. A term
  whose entity is absent from the converted case, such as an interchange line
  missing from a reduced system, is dropped with a diagnostic naming it.
- **Risk.** The CVaR setting in `dger.dat` selects expectation, constant CVaR,
  or per-stage CVaR from `cvar.dat`. If `dger.dat` asks for CVaR and
  `cvar.dat` is absent, the converter falls back to expectation and says so.
- **Penalties.** NEWAVE's flow-domain penalties (`penalid.dat`) are converted
  to Novomodelo's cost basis with the same system-mean productivity NEWAVE applies,
  so a violation is priced the same way in both models.
- **Stochastic data.** Historical inflows (`vazoes.dat`, `vazpast.dat`), load
  (`sistema.dat`, `c_adic.dat`), and block factors (`patamar.dat`) are written
  under `scenarios/`, stage-varying tables as Parquet.

Anything the converter cannot carry over faithfully is reported rather than
dropped silently: `convert newave` renders diagnostics as grouped panels,
`--diagnostics-json` saves them, and `--json` includes them in the verdict.

## Comparing results

`compare newave NEWAVE_DIR NOVOMODELO_OUTPUT_DIR` needs, on the NEWAVE side, the
`MEDIAS-*.CSV` result files and `pmo.dat` directly in `NEWAVE_DIR`, and on the
Novomodelo side the `output/` directory `novomodelo run` produced. It aligns entities
through the conversion's id mapping, aggregates Novomodelo's scenarios to NEWAVE's
published level, and reports per-variable agreement as a symmetric percentage
error and the share of points within tolerance. It also re-evaluates the
converted generic constraints against both models' operation. With
`--format html` it writes a multi-tab report.

Before treating a difference as a converter defect, confirm the two runs are
comparable:

- The security curve uses fixed penalization in `curva.dat`. The bridge models
  the curve as a per-stage slack at the fixed cost and does not reproduce
  NEWAVE's iterative penalization mode.
- NEWAVE's final simulation has preventive rationing enabled in `dger.dat`.
  With it disabled, NEWAVE drains reservoirs to avoid deficit while Novomodelo
  follows the converted policy, and the two diverge by construction.
- The hydro production function is linear (constant productivity). A
  head-dependent run is a different production model.
- You know whether the final simulation is deterministic or stochastic. A
  single historical series makes the inflow model irrelevant on both sides.
- No input `.dat` file is newer than the NEWAVE outputs. Otherwise the
  converted case and the published results come from different decks.

Then read the report cost-first: the cost breakdown says which component
differs, the per-stage costs say where, and the system and per-bus operation
tabs localize it. Load and non-controllable generation are converted directly
from the inputs and are rarely the cause.
