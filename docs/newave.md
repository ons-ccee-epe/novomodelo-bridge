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

The CSV inputs of the LIBS family are found through their own index,
`indices.csv` in the case directory. Each line of it names a record key and the
file that holds it, and the converter reads each LIBS record from the file
listed for its key: `HIDRELETRICA-CADASTRO-RESERVATORIO-CURVAJUSANTE` (the
tailrace curves), `HIDRELETRICA-CADASTRO-RESERVATORIO` (the per-period
reference volumes), `HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES` (the FPHA plane
reduction method) and `RESTRICAO-ELETRICA-ESPECIAL` (the electrical
constraints). Lines starting with `&` are comments. Without `indices.csv`, or
with a key left out or commented out, that record is not part of the case: no
tailrace curves, no electrical constraints from the CSV, and no plane
reduction, which Novomodelo then leaves off. A key whose file is not in the
directory makes the case fail to convert. A file with a usual LIBS name, such
as `polinjus.csv`, that the index does not list is ignored, and `check newave`
and `convert newave` warn about it. `indices.csv`, the plane-reduction file and
the electrical-constraint file are each read as UTF-8 when they are valid UTF-8
and as Latin-1 otherwise.

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
minimum outflow, `REST. TURBINAMENTO` for the dated turbining records of
`modif.dat`, and `CANAL DE DESVIO` for its `CDESVIO` records. A file that is present but switched off is ignored, as NEWAVE
ignores it, and both `check newave` and `convert newave` say so with an
informational diagnostic. A switch line absent from `dger.dat` counts as on.

NEWAVE's text files are read by fixed columns, so a tab where the layout
expects spaces shifts every field after it on its line, and a data line can
read with wrong or missing values. `check newave` and `convert newave` warn
about every text file `arquivos.dat` names that contains a tab, and name its
lines; replace each tab with the spaces that restore the columns.

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
- **Diversion channels.** A `CDESVIO` record in a plant's block of
  `modif.dat` becomes that plant's Novomodelo diversion channel, to the record's
  target plant and capped at its maximum flow; a maximum flow of 9 999 999
  m³/s or more, NEWAVE's "no limit" value, gives a channel without a limit
  and is reported. Only `CDESVIO`
  declares a channel; the `desvio` code of `hidr.dat` alone does not. A record
  with no maximum flow, or whose target plant is not in the case, converts to
  no channel and is reported, and so the plant releases only to its
  downstream plant. The minimum storage for operating the channel
  (`volume_desvio` in `hidr.dat`) is not converted.
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
- **Incremental inflows.** NEWAVE records natural inflow per gauging station
  (posto); Novomodelo takes each plant's incremental inflow. The converter
  subtracts, for each plant, the natural inflow of the stations of the plants
  directly upstream of it. Plants that share a station each get their own
  series. A plant on posto 300, NEWAVE's artificial gauge, has no natural
  inflow: it gets a zero series, so Novomodelo treats its inflow as deterministic
  zero, and an artificial plant upstream is not subtracted. `convert newave`
  lists these plants, and warns when a gauged plant sits below an artificial
  plant that has gauged plants further upstream, since their inflow may then
  be counted twice.
- **Production function.** When `dger.dat` selects the hydro production
  function (FPHA), Novomodelo fits each eligible plant's planes itself. The
  converter sets the spillage range of that fit to twice the largest
  calendar-month mean natural inflow of the plant's station in `vazoes.dat`,
  over the record before the study start. DECOMP applies the same rule to its
  long-term means. A plant on posto 300 gets no range, and Novomodelo fits over
  twice its maximum turbined flow.
- **FPHA reference volume.** Each plant of an FPHA case declares, per stage,
  the volume NEWAVE reads its forebay level at: the head of its own turbined
  fit range and the downstream level of the plant above it. A daily or
  run-of-river plant reads `volume_referencia`. A monthly reservoir reads the
  `CADH-VOL-REF-PER` row of `volumes-referencia.csv` that covers the stage's
  month when `VOLUME-REFERENCIAL-TIPO-PADRAO` is 1, or `volume_referencia`
  when it has no row. Without the file, with type 0 or without the record,
  every monthly reservoir reads its initial storage. Rows that leave a stage
  uncovered or give one month twice fail the conversion, as they stop NEWAVE.
  `volref_saz.dat` and the `modif.dat` temporal records do not move this
  volume; `volref_saz.dat` sets the evaporation reference and the head of the
  equivalent productivity. A volume outside the plant's storage range widens
  its sampled geometry to cover it.
- **Turbined fit range.** Each FPHA plant's fit runs up to the flow at which
  every machine set reaches its rated power or its turbine limit, NEWAVE's
  rule, with no outage derating: the turbine head runs from the forebay at the
  reference volume down to the `polinjus` tailrace at that flow, read at the
  forebay of the plant `hidr.dat` names downstream, and the generator head
  from the mean forebay over the plant's operating range.
- **Turbine and generation limits.** Under FPHA, a plant turbines at most its
  rated flow and generates at most its rated power, both scaled by its TEIF
  and IP availability, with no head correction. `max_turbined_m3s` declares
  that flow and `hydro_bounds.parquet` gives the available power on every
  stage, while the declared `max_generation_mw` stays the rated power the fit
  is capped at. A constant-productivity plant is capped at its rated flow
  corrected for the operating head, which also keeps it under its available
  power.

Anything the converter cannot carry over faithfully is reported rather than
dropped silently: `convert newave` renders diagnostics as grouped panels,
`--diagnostics-json` saves them, and `--json` includes them in the verdict.

## Pumping stations

The converter reads pumping station data from LIBS files declared in
`indices.csv`:

| Index key | Content |
|-----------|---------|
| `ESTACAO-BOMBEAMENTO` | Station definitions and temporal limits |
| `ESTACAO-BOMBEAMENTO-SUBMERCADO` | Station-to-subsystem mapping |

Both keys must be listed for pumping conversion to proceed. Files present
in the deck directory but not declared in `indices.csv` are ignored (with
a warning).

### Switch

The feature respects the `ESTACOES BOMBEAMENTO` register in `dger.dat`:

| Value | Behavior |
|-------|----------|
| `1` (or absent) | Pumping converted |
| `0` | Pumping skipped; `pumping_stations.json` has an empty list |

When the switch is off but the files are present, a diagnostic notes that
the input was skipped.

### Fictitious filter exemption

Hydro plants referenced as origin or destination by a pumping station are
excluded from the fictitious-plant filter. Without this exemption, a
zero-productivity reservoir used for pumped storage would be removed,
potentially breaking cascade topology.

### Open-ended periods

A temporal limit row with a blank `data_fim` applies from `data_inicio`
to the last stage of the study horizon.

### Validation

The converter raises `SourceFileError` when:

- A station has no entry (or multiple entries) in the submercado file
- `codigo_usina_origem` or `codigo_usina_destino` references a hydro not
  in `confhd.dat`
- A limits row references an undefined station
- `patamar` is outside the horizon's block range

### Output files

| File | When written |
|------|--------------|
| `system/pumping_stations.json` | Always (empty list when no stations) |
| `constraints/pumping_bounds.parquet` | Only when temporal limits exist |

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
- Both runs use the same kind of hydro production function. When `dger.dat`
  selects FPHA, Novomodelo fits its own planes rather than reading NEWAVE's; the
  Productivity tab compares the two.
- You know whether the final simulation is deterministic or stochastic. A
  single historical series makes the inflow model irrelevant on both sides.
- No input `.dat` file is newer than the NEWAVE outputs. Otherwise the
  converted case and the published results come from different decks.

Then read the report cost-first: the cost breakdown says which component
differs, the per-stage costs say where, and the system and per-bus operation
tabs localize it. Below the convergence chart, a table nets each model's
expected constraint-violation penalties out of its final lower bound, so a
slack priced far above any real cost on one side does not hide how close the
priced operation is. Load and non-controllable generation are converted directly
from the inputs and are rarely the cause.

In an FPHA run, the Productivity tab compares the two models' fitted planes on
NEWAVE's fit grid, read from `fpha_cortes.csv` and `fpha_eco.csv`, and the two
models at the operating points each one reached, per plant, stage, series and
block. The operating points come from the nwlistop plant tables (`ghiduh`,
`qturuh`, `qvertuh`, `gh_fphexat`, `varmuh` and `cmarg`), which nwlistop
writes only for the plants in the plant list of `nwlistop.dat`. Set that list
to `999` (every plant) and rerun nwlistop; without the tables the tab leaves
the operating points out and says so. For each plant and stage the tab splits
the gap between the two realized productivities into a part from the fitted
planes, Novomodelo's planes evaluated at NEWAVE's operating points, and a part from
where Novomodelo operates, and plots generation against turbined flow at NEWAVE's
mean operating point for both models' planes and the exact production
function.

The exact function follows NEWAVE's conventions: the forebay at the stage's
mean storage, the mean of its initial and final storage in `varmuh`, starting
from the `confhd.dat` initial storage, with a daily or run-of-river plant held
at its `volume_referencia`; the tailrace at the turbined flow, plus the spill
where `hidr.dat` says it reaches the tailrace, read between `polinjus`
families at the downstream plant's forebay; the losses of `hidr.dat`; and the
`fpha_eco.csv` generation limit. The fit-inputs table gives the largest gap
between that function and NEWAVE's printed `gh_fphexat` for each plant and
stage. A point where NEWAVE's marginal cost (`cmarg`) is zero, or where
Novomodelo's spot price or water value is, is a tie, handled as on the DECOMP
report (see [the DECOMP guide](decomp.md#comparing-results)).
