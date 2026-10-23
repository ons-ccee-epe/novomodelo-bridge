# The DECOMP track

`convert decomp`, `check decomp`, and `compare decomp` work on a DECOMP deck
revision. This page covers what the converter reads, what it leaves out and
how it tells you, the boundary cost-to-go import that runs by default, and the
one extra rule for running the converted case. Flags and their help text are
in the [CLI reference](cli.md).

## Inputs

`SRC` is the deck directory. `caso.dat` names the revision (`rv0`, `rv1`, and
so on) and the revision's index file names that revision's data files:
`dadger.rvN` and `vazoes.rvN`, the shared `hidr.dat` registry, `dadgnl.rvN`
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
  priced in the month the `GS` register of `dadgnl.rvN` books its week to,
  with that month's patamar shares from `cortesh.dat`. A delivery the deck
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
`output/` directory. Two caveats apply to the report:

- The overview's net-present-value cost cards compare DECOMP's undiscounted
  nominal costs with Novomodelo's time-discounted costs. DECOMP's cost report
  carries no per-stage discount factor and the bridge does not invent one, so
  the two totals are not on the same time-value footing.
- Percentile bands are omitted, or labelled low-N, when the deterministic tree
  has too few scenarios to report a spread without synthesizing it.
