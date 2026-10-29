# DESSEM FCF Export

`export dessem` converts the policy checkpoint of a solved Novomodelo case into the
DECOMP-format binary cut files (`mapcut`, `cortdeco`) that DESSEM reads as its
future cost function. This page covers the workflow, the output artifacts, the
conversion premises, and the current limitations.

## Quick start

```bash
novomodelo-bridge export dessem /path/to/decomp-deck /path/to/case
```

The output appears in `<case>/dessem_fcf/` unless `--output` names another
directory.

## Input

`DECOMP_DIR` is the DECOMP deck the case was converted from with
`convert decomp`. The export reads from it:

- the revision named in `caso.dat`, which sets the output file suffix;
- the operated hydro plants (`UH` records with an initial volume);
- their `hidr.dat` codes, which label the plants in `mapcut`;
- each GNL plant's lag in months, from its `NL` record in `dadgnl`, which
  `mapcut` carries per GNL plant;
- the order of its `VI` records, which numbers the plants with a water travel
  time in `mapcut`.

The deck must operate exactly the case's hydro plants, under the names the
case gives them. Otherwise the export fails with a message naming the first
difference. Novomodelo hydro id `i` is the `i`-th smallest operated plant code,
the numbering `convert decomp` uses. Likewise the deck's GNL plants, in
ascending code order, must match the case's anticipated thermals in id order
and by name, and each must have an `NL` record.

`CASE_DIR` is a Novomodelo case that novomodelo has solved, so `CASE_DIR/output/policy/`
holds its policy checkpoint. The checkpoint is read through `novomodelo-python`, so
it must have been written by the novomodelo release this bridge pairs with; an
older checkpoint is refused with an error naming its format version. The
export also reads the case's `stages.json`, `system/hydros.json`,
`system/thermals.json`, and `system/buses.json`.

A plant whose `hydros.json` entry carries `travel_time_hours` is one of
`mapcut`'s travel-time plants, and it must have a `VI` record in the deck.
Each cut then also values the water those plants released that has not yet
reached the plant downstream, per stage of release, from the cut's
transit-bucket coefficients (premise P3).

Novomodelo stores cut values in the run's scaled cost units and records the scale
in each pool's `cost_scale_factor`. The export multiplies every intercept and
coefficient by that factor to get R$, then divides by 1000 for DECOMP's
10³ R$.

## Options

| Option | Description |
|--------|-------------|
| `--output, -o` | Output directory. Default: `CASE_DIR/dessem_fcf/`. |
| `--force` | Overwrite existing outputs. |
| `--audit-csv` | Write the audit CSVs (`eco/` and `content/`). |
| `--include-terminal` | Also write the terminal pool's audit CSV (with `--audit-csv`). |
| `--json` | Print one machine-readable JSON verdict to stdout. |
| `-v`, `-vv` | Increase console log verbosity (INFO, DEBUG). |

`docs/cli.md` lists every option with its help text.

## Outputs

| Artifact | Description |
|----------|-------------|
| `mapcut.<rev>` | DECOMP cut-header binary: iteration and cut counts, hydro plant codes in Novomodelo's plant order, each plant's downstream plant (as its 1-based position in that list, as DECOMP writes it), the node tree, discount factors, the per-node cut-chain heads, and, per travel-time plant, the share of each stage's release that arrives in each later stage. |
| `cortdeco.<rev>` | DECOMP cut binary: one record per cut with its right-hand side, storage coefficients, coefficients of the water released and still in transit, and GNL coefficients. |
| `eco/*.csv` | With `--audit-csv`: one CSV per pool with every cut as read from the checkpoint, already converted to R$. |
| `content/*.csv` | With `--audit-csv`: the values actually written to `mapcut` and `cortdeco`, read back from the binaries. |
| `export_manifest.json` | Provenance: tool and library versions, the case and deck paths, settings, the full text of every premise applied, output paths, and status. |

A successful export writes `export_manifest.json` with `"status": "ok"`. A
failed one writes it with `"status": "failed"` and a `"failed_step"` field,
and the command prints the reason (or, under `--json`, carries it in the
verdict's `diagnostics`).

After writing both binaries the export reads them back and checks that they
parse and agree. If that check fails, both files are renamed with a
`.rejected` suffix so a later DESSEM run cannot pick them up.

## Premises

The conversion follows a fixed set of premises, P1 to P15. Their full text is
the `PREMISES` tuple in `novomodelo_bridge.dessem.exporters.fcf.core.premises`,
and every export copies it into `export_manifest.json`. In brief:

| Premise | Effect |
|---------|--------|
| P1 | `mapcut` physical records (indices 4–17) are written as zeros. |
| P2 | Intercepts and coefficients are divided by 1000 (R$ to 10³ R$). |
| P3 | Travel-time plants follow the deck's `VI` records; each release's in-transit water is valued from the transit buckets it arrives in. A bucket with no arrival window (water that would arrive after the study ends) takes no weight; a coefficient found on one is dropped with a warning. |
| P4 | GNL coefficient sign flip; dormant while P14 holds. |
| P5 | GNL split across load blocks by stage hours; dormant while P14 holds. |
| P6 | Submarket is `bus_id + 1`; the `IV` transhipment bus is excluded. |
| P7 | Only trunk pools are converted; the terminal pool is excluded. |
| P8 | Inflow-lag coefficients are dropped, with a counted audit log. |
| P9 | Discount factors are `(1+r)^(-days/365.25)` from the stage dates. |
| P10 | Every populated cut past the warm-start slots is written; `is_active` is recorded, not used as a filter. |
| P11 | `mapcut` record 10's operational cost parcels are written as zeros. |
| P12 | `mapcut` record 9's trailing GNL hours block is written as zeros. |
| P13 | `cortdeco` holds one extra record duplicating the last cut. |
| P14 | GNL coefficients are written as zeros. |
| P15 | GNL lag months are read from the deck's `NL` records; a GNL plant with no `NL` record is refused. |

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Export succeeded. |
| 1 | Export failed: deck or case not found, the deck is not the one the case was converted from, outputs exist without `--force`, unreadable checkpoint, a premise refused the case, or the read-back check failed. |
| 2 | Invalid command line, such as a missing `DECOMP_DIR`. |

## Known limitations

1. **GNL coefficients are zeros (P14).** The exported cuts carry the value of
   stored water but not of GNL commitments.
2. **Terminal pool excluded (P7).** DESSEM supplies its own terminal value.

## See also

- [DECOMP track](decomp.md): converting DECOMP decks to Novomodelo.
- [CLI reference](cli.md): every command and option.
