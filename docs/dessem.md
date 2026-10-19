# DESSEM FCF Export

`export dessem` converts the policy checkpoint of a solved Novomodelo case into the
DECOMP-format binary cut files (`mapcut`, `cortdeco`) that DESSEM reads as its
future cost function. This page covers the workflow, the output artifacts, the
conversion premises, and the current limitations.

## Quick start

```bash
novomodelo-bridge export dessem /path/to/case --revision rv0 \
    --hydro-codes /path/to/hydro_codes.json
```

The output appears in `<case>/dessem_fcf/` unless `--output` names another
directory.

## Input

`CASE_DIR` is a Novomodelo case that novomodelo has solved, so `CASE_DIR/output/policy/`
holds its policy checkpoint. The checkpoint is read through `novomodelo-python`, so
it must have been written by the novomodelo release this bridge pairs with; an
older checkpoint is refused with an error naming its format version. The
export also reads the case's `stages.json`, `system/hydros.json`,
`system/thermals.json`, and `system/buses.json`.

Novomodelo stores cut values in the run's scaled cost units and records the scale
in each pool's `cost_scale_factor`. The export multiplies every intercept and
coefficient by that factor to get R$, then divides by 1000 for DECOMP's
10³ R$.

## Options

| Option | Description |
|--------|-------------|
| `--revision, -r` | PMO revision (required, e.g. `rv0`). Sets the output file suffix. |
| `--hydro-codes` | JSON file mapping Novomodelo hydro positions to DECOMP plant codes. Default: the bundled `decomp_hydro_codes.json`. |
| `--output, -o` | Output directory. Default: `CASE_DIR/dessem_fcf/`. |
| `--force` | Overwrite existing outputs. |
| `--audit-csv` | Write the audit CSVs (`eco/` and `content/`). |
| `--include-terminal` | Also write the terminal pool's audit CSV (with `--audit-csv`). |
| `--json` | Print one machine-readable JSON verdict to stdout. |
| `-v`, `-vv` | Increase console log verbosity (INFO, DEBUG). |

`docs/cli.md` lists every option with its help text.

### Hydro codes file

The file maps Novomodelo's 0-based hydro positions to DECOMP plant codes. Keys are
position strings, values are integer codes, and position 0 is the first
storage coefficient:

```json
{
  "0": 1,
  "1": 2,
  "2": 4
}
```

The map must cover exactly the case's hydro plants. The bundled default holds
one specific plant set, so a case converted from a deck with a different set
of plants needs its own file; a mismatch fails the export with a message
naming both counts. For a case converted with `convert decomp`, Novomodelo hydro
ids are the deck's operated plants (`UH` records) in ascending code order, so
position `i` maps to the `i`-th smallest operated plant code.

## Outputs

| Artifact | Description |
|----------|-------------|
| `mapcut.<rev>` | DECOMP cut-header binary: iteration and cut counts, hydro codes, the node tree, discount factors, and the per-node cut-chain heads. |
| `cortdeco.<rev>` | DECOMP cut binary: one record per cut with its right-hand side, storage coefficients, and GNL coefficients. |
| `eco/*.csv` | With `--audit-csv`: one CSV per pool with every cut as read from the checkpoint, already converted to R$. |
| `content/*.csv` | With `--audit-csv`: the values actually written to `mapcut` and `cortdeco`, read back from the binaries. |
| `export_manifest.json` | Provenance: tool and library versions, settings, the full text of every premise applied, output paths, and status. |

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
| P3 | No travel-time block: `n_utv = 0`. |
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
| P15 | GNL lag months are written as 2; any GNL plant whose lead time is not 1608 h is refused. |

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Export succeeded. |
| 1 | Export failed: case or hydro codes file not found, outputs exist without `--force`, unreadable checkpoint, a premise refused the case, or the read-back check failed. |
| 2 | Invalid command line, such as a missing `--revision`. |

## Known limitations

1. **GNL coefficients are zeros (P14).** The exported cuts carry the value of
   stored water but not of GNL commitments.
2. **No travel time (P3).** A case whose cuts carry water travel-time state
   is refused.
3. **One GNL lead time (P15).** A case whose GNL plants declare a lead time
   other than 1608 h is refused rather than mapped by a guessed rule.
4. **Terminal pool excluded (P7).** DESSEM supplies its own terminal value.

## See also

- [DECOMP track](decomp.md): converting DECOMP decks to Novomodelo.
- [CLI reference](cli.md): every command and option.
