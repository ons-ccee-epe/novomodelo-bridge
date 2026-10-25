# The DECOMP track

`convert decomp`, `check decomp`, and `compare decomp` work on a DECOMP deck
revision. This page covers what the converter reads, what it leaves out and
how it tells you, the boundary cost-to-go import that runs by default, and the
one extra rule for running the converted case. Flags and their help text are
in the [CLI reference](cli.md).

## Inputs

`SRC` is the deck directory. `caso.dat` names the revision (`rv0`, `rv1`, and
so on) and the revision's index file names that revision's data files:
`dadger.rvN` and `vazoes.rvN`, the shared `hidr.dat` registry, the
long-term mean inflows `mlt.dat` when the index lists them, `dadgnl.rvN`
when the deck has fuel-constrained thermals, and the cut files when the deck
declares them. The deck is parsed once and shared by every converter.

The CSV inputs of the LIBS family are found through the LIBS index that the
`FA` register of `dadger.rvN` names. Each line of the index names a record key
and the file that holds it, and the converter reads each LIBS record from the
file listed for its key: the `PARQUE-EOLICO-EQUIVALENTE-*` records (the
renewable plants, which must all be in one file),
`HIDRELETRICA-CADASTRO-RESERVATORIO-CURVAJUSANTE` (the tailrace curves) and
`RESTRICAO-ELETRICA-ESPECIAL` (the electrical constraints). Lines starting with
`&` are comments. The `FJ` register can name the tailrace-curve file too, with
or without an index; when both name one, they must agree. Apart from that file,
a deck without `FA` has no LIBS records. A declared file that is not in the
deck directory makes the deck fail to convert. A
file with a usual LIBS name, such as `renovaveis.csv`, that nothing declares is
ignored, and `check decomp` and `convert decomp` warn about it. The index is
read as UTF-8 when it is valid UTF-8 and as Latin-1 otherwise.

`check decomp` validates the deck without writing anything and also lists what
the conversion will leave behind for this particular deck. Exit 0 means ready,
1 ready with warnings, 2 the deck will not convert.

`dadger.rvN` and `dadgnl.rvN` are read by fixed columns, so a tab where the
layout expects spaces shifts the fields after it, and a register line that
starts with one is not read at all. `check decomp` and `convert decomp` warn
about every tab outside a comment line (one whose first non-blank character
is `&`) and name the file and lines.

The register-by-register map from the deck to the converted case, including
what the converter does not convert yet, is the
[DECOMP data map](decomp-data-map.md) (in Portuguese).

## What is converted, and what is deferred

The converter emits buses with their deficit curves plus the transshipment
node, exchange lines, thermals (including the fuel-constrained anticipated
dispatch declared in `dadgnl`, with its post-horizon commitments), hydros with
the deck's registry overrides applied, small plants as must-run
non-controllable sources, the explicit scenario tree (inflow, load, and
non-controllable generation per node), pumping stations, energy contracts,
generic constraints from the deck's restriction registers, and head-dependent
production planes where the deck uses them.

A plant's diverted flow (`QDES`) reaches its diversion target, the plant named
by `AC DESVIO` or else by the `desvio` field of `hidr.dat`, through a Novomodelo
diversion channel. The channel carries the `AC DESVIO` flow limit; a plant
with no such limit, or with the deck's "no limit" value of 9 999 999 m³/s
or more, gets a channel without one, as in DECOMP, and the `QDES` bounds
still apply stage by stage. When that target is not in the case, a diversion
ceiling is closed and reported, so the water stays in the plant's own river;
a positive diversion floor cannot be closed, and the conversion fails.

A restriction register on one plant variable, such as an `HQ` on `QDEF`,
becomes a bound on that plant. Two such `HQ` restrictions on the same plant
variable that no value meets at once, such as a 0 m³/s ceiling and a 6 m³/s floor, become generic constraints
instead, each with its own slack at the same penalty. DECOMP prices every
`RHQ` violation per unit and violates the cheaper restriction, whatever their
order or numbers, so Novomodelo then makes the same choice. The conversion reports
each such pair as a warning.

Incremental inflows are converted as the inflow file gives them, negative
values included. DECOMP reports a negative incremental inflow as zero in its
operation output; Novomodelo's water balance removes that water unless its priced
inflow slack adds it back. The conversion lists every negative value as a
warning.

A plant's turbined flow is capped stage by stage at the limit DECOMP computes
before the optimization. The rule:

- The units' rated flow is corrected for the head at the stage's initial
  storage. The tailrace level rises with the flow, so the flow is iterated
  until head and flow agree.
- The result is scaled by the `MP` maintenance and `FD` availability factors.
  It never exceeds the flow that produces the plant's available power.
- A plant that cannot hold back its stage inflow turbines up to that inflow
  when it exceeds the head-limited flow. This covers a run-of-river plant and
  a reservoir without enough free room. The inflow is the deck's, negative
  values included; the terminal stage takes the probability-weighted mean of
  its scenarios.
- The tailrace curves come from `polinjus`. For a deck without that file they
  come from the `hidr.dat` downstream polynomials. That path follows the same
  rule but has not been checked against a DECOMP run.
- Itaipu's limit is plant-wide: its two frequency groups share the
  head-limited flow, and each group is also held to its own available power.
- DECOMP raises the tailrace level by the lateral inflows of `VL` and `VU`
  registers; the conversion leaves them out. A plant with such a register,
  such as Itaipu or Belo Monte, may therefore turbine slightly more than
  DECOMP allows.
- `AC ALTEFE` overrides of the rated head are not applied; the conversion
  warns about them.

Novomodelo fits the planes of the hydro production function (FPHA) itself. These
inputs follow DECOMP:

- Each stage is fitted over DECOMP's volume window: the `UH` initial storage,
  as a percentage of the stage's useful volume, minus a reduction and plus an
  increase, within the stage's storage limits. Both are 20 % of useful volume
  on a weekly stage and 50 % on a monthly one unless `FP` records set them,
  in % of useful volume or in hm³, from their stage on. A run-of-river plant
  is fitted at its reference volume. The turbined-flow window and point
  counts of `FP` are not converted; the conversion warns when a record sets
  them away from DECOMP's defaults.
- The spillage range of the fit is twice the largest monthly long-term mean
  inflow in `mlt.dat` of the plant's station (`AC NUMPOS` included). A plant
  on the artificial station 300, or on a station with a zero mean, is fitted
  over twice its maximum turbined flow, as in DECOMP. A deck without `mlt.dat`
  gets that fallback for every plant, and the conversion warns.
- A stage where the plant's turbined limit is zero, for example with every
  machine under maintenance, gets a zero constant productivity in place of
  the production function. DECOMP builds no planes for such a stage.

Some deck features have no Novomodelo counterpart yet, or are held back behind a
novomodelo issue. None is dropped silently: `check decomp` lists them per deck,
`convert decomp` reports each as a warning naming the affected records, and
the same diagnostics land in `--diagnostics-json` and the `--json` verdict.
Examples at the time of writing are the unreadable special-constraint
families (`FE`, `RHA`) and a few restriction-register term types; the
[DECOMP data map](decomp-data-map.md)
lists every register and field the converter leaves out. Each warning's
remediation line states whether any action is needed; usually none, and the
conversion proceeds.

## The boundary cost-to-go function

A DECOMP deck ends at a horizon where a longer-term cost-to-go function takes
over, delivered as cut files. By default `convert decomp` imports those cuts
into a Novomodelo policy checkpoint under the case's `boundary/` directory and
points `config.json` at it, so Novomodelo prices its terminal stage the way DECOMP
did.

- The import runs after the case is written and needs `novomodelo-python`, which
  the bridge installs as a dependency. It bootstraps the terminal cut layout
  with a short in-process novomodelo pass, so it is the slowest part of the
  conversion.
- Before importing, the bridge checks that the installed `novomodelo-python` can
  write and reload the checkpoint format by doing a real round trip, not by
  reading a version string. If the probe fails, the error says so;
  reinstalling or upgrading `novomodelo-python` resolves it.
- `--no-fcf` skips the import for a quick conversion. `--dry-run` always
  skips it.

### How the cuts attach to the converted plants

The cuts come from NEWAVE, which represents some plants differently from
DECOMP. The import couples them plant by plant, as DECOMP does:

- A plant with its own row in the cuts takes it. A plant inside a complex
  that a `CX` register of `dadger.rvN` names takes the complex's row. Any
  other plant takes the row of the first plant downstream that has one;
  DECOMP reports this rule in its output, but its manual does not describe
  it. The import lists every plant that takes another plant's row. A plant
  that finds no row is reported as an error, because the boundary gives its
  water no value.
- Only a reservoir (`tipo_regulacao` `M` or `S` in `hidr.dat`) takes a
  storage term, through its own row or its complex's, and the term prices
  the storage above the plant's minimum volume.
- The inflow-lag terms are read on each plant's coupling gauge: the one an
  `AC NPOSNW` register names, otherwise the `hidr.dat` gauge, with the
  artificial gauges computed. Novomodelo's lag state follows the inflow the plant
  operates on. Over the stages before the last one, the difference between
  the two is known at conversion and is folded into each cut's constant
  term, as are the lag terms of cut plants that no converted plant takes.
  The pre-study observations that seed Novomodelo's lag state are read on the
  coupling gauges too.
- The cut values are scaled by the mean month of the DECOMP manual
  (365.25 days × 24 h / 12), whatever the length of the last stage.
- For a deck with fuel-constrained thermals, each anticipated delivery is
  priced in the month the `GS` register of `dadgnl.rvN` books its week to
  (in a deck of monthly stages only, its own month), with that month's
  patamar shares from `cortesh.dat`. A delivery the deck
  has already committed is folded into the constant term.
- Water still in transit toward a plant when the study ends (the `VI`
  register's travel time) is valued with that plant's storage term when it
  takes one, otherwise with its first inflow-lag term. This is the in-channel
  rule CEPEL documents for the cost-to-go function DECOMP hands to DESSEM.
  DECOMP gives that water no value at the NEWAVE boundary, so the two
  boundary costs differ by its value; the import lists the plants concerned.

Two parts of the coupling cannot be represented yet, and the import warns
about each. In the last stage, the coupling inflow's difference from the
operated inflow, and the inflow of cut plants outside the case, vary by
scenario and are not folded; the warning names the plants and the largest
gap. When the cut's month begins before the last
stage, Novomodelo evaluates the first inflow lag on the last stage's inflow
alone; the warning states the share of the month this affects. A third note
reports how far the per-patamar GNL prices spread, since Novomodelo holds one MW
per delivery window rather than one per patamar.

The [DECOMP data map](decomp-data-map.md) traces each term to the deck
record it is read from.

### Running a case with an imported boundary

The boundary checkpoint path resolves relative to the case directory, so a
case with an imported boundary runs like any other:

```bash
novomodelo run <case_dir>
```

`convert decomp` confirms the imported boundary and this command after a
successful import.

The checkpoint records the version of the `novomodelo-python` that wrote it, and
novomodelo loads it only when that version equals its own. Run the case with the
`novomodelo` release whose version matches the installed `novomodelo-python`
(`pip show novomodelo-python`); a different novomodelo refuses the boundary and names
both versions. To use another novomodelo version, install the novomodelo-bridge
release paired with it, which pins the matching `novomodelo-python`, and convert
the deck again.

## Comparing results

`compare decomp DECOMP_DIR NOVOMODELO_OUTPUT_DIR` reads the deck plus the
`dec_oper_*.csv` operation tables directly in `DECOMP_DIR`, and Novomodelo's
`output/` directory. Percentile bands are omitted, or labelled low-N, when the
deterministic tree has too few scenarios to report a spread without
synthesizing it.

The overview's cost breakdown is at present value on both sides. DECOMP's
cost report prints each stage's parcels at their nominal value and the stage
total at both nominal and present value, so each parcel is discounted by that
stage's ratio of the two. The report prints no deficit parcel: the deficit is
priced from `dec_oper_sist` and the deck's `CD` costs. What is left of the
stage total after the printed parcels and the deficit is the penalty on the
constraint slacks the report does not itemize, such as `RHQ`, `RHV`, `RHE`,
evaporation and irrigation; it is shown as one unitemized-slack category. The
two turbining parcels are the price DECOMP puts on turbined water and compare
with Novomodelo's turbined cost. DECOMP's stage total counts the take-or-pay
minimum of an export contract at its price as a cost, and only the energy
above it as revenue. Novomodelo counts every exported MWh as revenue, so on a deck
with an export minimum the two contract costs differ by twice the minimum's
value; the comparison logs that amount.

DECOMP's `Zsup` is the lowest upper bound reached so far, and the relato
prints no upper bound per iteration, so the convergence chart sets it against
Novomodelo's running minimum.

Below the convergence chart, a table gives each model's final lower bound with
and without its expected constraint-violation penalties. One slack priced far
above any real cost, such as a pair of conflicting constraints, can dominate a
model's bound; the bounds net of penalties compare the operation both models
price. The table names the categories it counts and any cost category it
cannot classify. `compare newave` shows the same table.

On the Energy Balance tab, the ANDE load of the deck's `RI` register is added
to the demand of Itaipu's submarket, as the conversion adds it to that bus:
DECOMP's `dec_oper_sist` demand leaves it out, while its hydro generation
includes the Itaipu output that serves it. DECOMP prints the system's natural
inflow energy (ENA) as energy over the stage, in MWmês of 730 hours; the
chart converts it to the stage's mean MW, the unit of Novomodelo's series.

The Productivity tab compares the two models at the operating points each one
reached, per plant, stage, scenario and block. It reads DECOMP's fitted planes
(`avl_cortesfpha_dec.rvN`) and fit windows (`eco_fpha_.rvN`). For each plant
and stage it splits the gap between the two realized productivities into a
part from the fitted planes, Novomodelo's planes evaluated at DECOMP's operating
points, and a part from where Novomodelo operates. A chart draws generation against
turbined flow at DECOMP's mean operating point for both models' planes and the
exact production function, beside each model's fit inputs and head components.
The exact function follows DECOMP's conventions: the forebay at the stage's
mean storage, the tailrace at the plant's outflow plus the lateral flows of its
`VU` and `VA` registers, read between `polinjus` families at the downstream
plant's forebay, and the losses of `hidr.dat`. DECOMP holds the `VA` share at
the first stage's inflow, and so does the comparison. When the run wrote
`dec_desvfpha.rvN`, the table also gives the largest gap between that function
and DECOMP's own printed value.

A point where the plant's energy or stored water is worth nothing in either
model is a tie: DECOMP's marginal cost is zero, Novomodelo's spot price is zero at
every bus the plant feeds, or Novomodelo's water value is zero, all at DECOMP's
two-decimal print resolution. Either model may then turbine or spill at no
cost. Ties are left out of the split and of the per-plant ranking of
generation, turbined flow, productivity, spillage and outflow on the Hydro
Plant Details tab, and the tab reports their share per stage. DECOMP prints no
water value, so on its side a tie is detected through the marginal cost only.
