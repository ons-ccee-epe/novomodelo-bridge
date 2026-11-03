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
export also reads the case's `stages.json`, `initial_conditions.json`,
`system/hydros.json`, `system/thermals.json`, and `system/buses.json`.

A plant whose `hydros.json` entry carries `travel_time_hours` is one of
`mapcut`'s travel-time plants, and it must have a `VI` record in the deck.
Each cut then also values the water those plants released that has not yet
reached the plant downstream, per stage of release, from the cut's
transit-bucket coefficients (premise P3).

Each GNL plant's cuts price its commitment for the delivery of the case's
second stage (premise P14). The export finds that delivery in the policy
checkpoint by its start date, divides the plant's coefficient by the sum of
that stage's load-block hours in `stages.json`, changes its sign, and writes
the result in R$/MWh, the same on every load block. Only the cuts of the
first cut-building node carry it, and `mapcut` record 9 carries the same
load-block hours for every GNL plant. A plant for which the policy holds no
coefficient for that delivery, or a coefficient of exactly zero, is written
as 0.0 with a warning. The export is refused when a plant has two delivery
slots for that stage, when a slot's dates disagree with the stage's
load-block hours, when a case with GNL plants has a single stage, or when
the first cut-building node is not at the case's first stage.

`mapcut` record 10 carries, for each study stage, the cost of the GNL
generation committed before the study (premise P11). For every GNL plant, the
export takes the commitment windows in the case's `initial_conditions.json`.
It multiplies the MW of each window that covers the whole stage by the sum of
the stage's load-block hours in `stages.json` and by the plant's
`cost_per_mwh` in `system/thermals.json`. It sums the plants and writes the
result in 10³ R$, not discounted. A stage no window covers gets 0.0. The
export is refused when:

- a window covers a stage only in part;
- a covered stage's load-block hours differ from its calendar span;
- a GNL plant has no window;
- a window names a plant that is not a GNL plant;
- a window starts before the study's end and runs past it;
- two windows of one plant cover the same stage.

A window that lies entirely before the study, or starts at or after the
study's end, counts for no stage and adds nothing.

Novomodelo stores cut values in the run's scaled cost units and records the scale
in each pool's `cost_scale_factor`. The export multiplies every intercept and
coefficient by that factor to get R$. It then divides the intercepts and the
storage and in-transit water coefficients by 1000 for DECOMP's 10³ R$; the GNL
coefficients are written in R$/MWh and are not divided.

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
| `mapcut.<rev>` | DECOMP cut-header binary: iteration and cut counts, hydro plant codes in Novomodelo's plant order, each plant's downstream plant (as its 1-based position in that list, as DECOMP writes it), the node tree, discount factors, the per-node cut-chain heads, and, per travel-time plant, the share of each stage's release that arrives in each later stage. Per GNL plant it also holds the submarket, the lag in months and the load-block hours of the delivery the GNL coefficients price. Each stage's record 10 holds the discount factor and the cost of the GNL generation committed before the study, in 10³ R$. |
| `cortdeco.<rev>` | DECOMP cut binary: one record per cut with its right-hand side, storage coefficients, coefficients of the water released and still in transit, and GNL coefficients. |
| `eco/*.csv` | With `--audit-csv`: one CSV per pool with every cut as read from the checkpoint, already converted to R$. |
| `content/*.csv` | With `--audit-csv`: the values actually written to `mapcut` and `cortdeco`, read back from the binaries. `mapcut` record 9's GNL hours take one row per GNL plant and load block. |
| `export_manifest.json` | Provenance: tool and library versions, the case and deck paths, settings, the full text of every premise applied, output paths, and status. |

A successful export writes `export_manifest.json` with `"status": "ok"`. A
failed one writes it with `"status": "failed"` and a `"failed_step"` field,
and the command prints the reason (or, under `--json`, carries it in the
verdict's `diagnostics`).

After writing both binaries the export reads them back and checks that they
parse and agree. If that check fails, both files are renamed with a
`.rejected` suffix so a later DESSEM run cannot pick them up.

## Premises

The conversion follows a fixed set of numbered premises. Their full text is
the `PREMISES` tuple in `novomodelo_bridge.dessem.exporters.fcf.core.premises`,
and every export copies it into `export_manifest.json`. In brief:

| Premise | Effect |
|---------|--------|
| P1 | `mapcut` physical records (indices 4–17) are written as zeros. |
| P2 | Intercepts and the storage and in-transit water coefficients are divided by 1000 (R$ to 10³ R$); GNL coefficients stay in R$/MWh. |
| P3 | Travel-time plants follow the deck's `VI` records; each release's in-transit water is valued from the transit buckets it arrives in. |
| P4 | GNL coefficients change sign (Novomodelo negative, DECOMP positive). |
| P5 | A delivery's GNL value is written unchanged on every load block, never split by block hours. |
| P6 | Submarket is `bus_id + 1`; the `IV` transhipment bus is excluded. |
| P7 | Only trunk pools are converted; the terminal pool is excluded. |
| P8 | Inflow-lag coefficients are dropped, with a counted audit log. |
| P9 | Discount factors are `(1+r)^(-days/365.25)` from the stage dates. |
| P10 | Every populated cut past the warm-start slots is written; `is_active` is recorded, not used as a filter. |
| P11 | `mapcut` record 10 holds the committed GNL cost from the case's commitment windows, load-block hours and plant costs, in 10³ R$ and not discounted; its other four cost parcels are written as zeros. |
| P12 | `mapcut` record 9's trailing GNL hours block holds the delivery stage's load-block hours from `stages.json`. |
| P13 | `cortdeco` holds one extra record duplicating the last cut. |
| P14 | The first cut-building node's cuts carry each GNL plant's coefficient for the delivery of the case's second stage, divided by that stage's hours; the other GNL coefficients are dropped, with a counted audit log. |
| P15 | GNL lag months are read from the deck's `NL` records; a GNL plant with no `NL` record is refused. |

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Export succeeded. |
| 1 | Export failed: deck or case not found, the deck is not the one the case was converted from, outputs exist without `--force`, unreadable checkpoint, a premise refused the case, or the read-back check failed. |
| 2 | Invalid command line, such as a missing `DECOMP_DIR`. |

## Known limitations

1. **GNL coefficients for one delivery only (P14).** Only the first
   cut-building node's cuts carry GNL coefficients, and only for the
   delivery of the case's second stage. The policy's other GNL coefficients
   are not folded into the cut intercept, so the cuts price those
   commitments as zero. Each run logs their count per pool, which `-v`
   shows and `--log-file` records.
2. **One GNL value per delivery (P5).** Novomodelo prices one value per GNL
   delivery, so the file holds the same value on every load block, where
   DECOMP prices each load block separately.
3. **`mapcut` record 10's other cost parcels are zeros (P11).** The
   thermal minimum-generation cost, the import and export contract minimum
   costs and the signalled GNL generation are DECOMP quantities that a
   Novomodelo policy does not hold.
4. **The committed GNL cost is priced from the case's values (P11).** The
   export prices every committed stage at the plant's `cost_per_mwh` in
   `system/thermals.json` and does not read the stage-level costs in
   `constraints/thermal_bounds.parquet`. For a deck whose GNL plant's `TG`
   unit cost varies across the committed stages, the value therefore
   differs from DECOMP's. The [DECOMP data map](decomp-data-map.md) shows
   where the case's thermal costs come from. The export also prices the
   committed MW the case holds: when a deck's `GL` record commits a GNL
   plant outside the range between the inflexibility and the availability
   of its `TG` record, `convert decomp` clamps the committed MW into that
   range and logs a warning, and the value then differs from DECOMP's on
   those stages. How DESSEM uses this field is not documented, so matching
   DECOMP's value does not show an effect on dispatch.
5. **The GNL hours basis is not confirmed (P12).** `mapcut` record 9
   carries the load-block hours of the delivery stage, where a `mapcut`
   written by DECOMP carries those of an average month. How DESSEM combines
   these hours with the GNL coefficients is not documented.
6. **Terminal pool excluded (P7).** DESSEM supplies its own terminal value.

## See also

- [DECOMP track](decomp.md): converting DECOMP decks to Novomodelo.
- [CLI reference](cli.md): every command and option.
