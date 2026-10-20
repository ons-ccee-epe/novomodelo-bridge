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
Examples at the time of writing are water travel time (the `VI` register)
and a few restriction-register term types; the [DECOMP data map](decomp-data-map.md)
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
