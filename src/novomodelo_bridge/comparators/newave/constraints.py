"""Generic constraint LHS evaluation from the source model and Novomodelo simulation outputs.

Takes the converted case's generic constraints as novomodelo resolved them
(:class:`~novomodelo_bridge.novomodelo.constraint_expr.ResolvedGenericConstraints`: the
LP's own per-stage coefficients and bounds) and evaluates each constraint's LHS
using both:

- the source model outputs — ``MEDIAS-USIH.CSV`` (``GHIDUH``, ``VARMUH`` per plant
  per stage) and ``int*.out`` (per-line per-stage interchange).
- Novomodelo simulation outputs — per-(scenario, stage, block) hydro and
  exchange data, collapsed to one value per (constraint, stage) by
  averaging across scenarios and blocks.

The output is consumed by the compare-results report's Constraints tab, mirroring the
dashboard's per-constraint LHS-vs-Bound visualisation with a the source model overlay so
the user can see whether a binding constraint matches across the two solvers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from novomodelo_bridge.comparators.newave.alignment import EntityAlignment
from novomodelo_bridge.novomodelo.constraint_expr import (
    USEFUL_VOLUME_KINDS,
    ResolvedGenericConstraints,
    unsupported_constraint_ids,
)

if TYPE_CHECKING:
    from novomodelo_bridge.newave.id_map import NewaveIdMap


def evaluate_lhs_newave(
    resolved: ResolvedGenericConstraints,
    nw_hydro_df: pl.DataFrame,
    nw_line_means: pl.DataFrame,
    alignment: EntityAlignment,
    id_map: NewaveIdMap,
    nw_offset: int,
) -> pl.DataFrame:
    """Evaluate each constraint's LHS from the source model simulation outputs.

    The constraint terms reference Novomodelo entity IDs. We translate via ``id_map``
    (hydro Novomodelo id → the source model code) and ``alignment.lines`` (Novomodelo line id →
    the source model submarket pair) before looking values up in the source model data.

    The source model outputs are stage-level (block-collapsed) — MEDIAS values are
    monthly means and ``int*.out`` ``TOTAL`` rows already average over blocks weighted
    by their duration. The result therefore has one value per (constraint, stage_0based)
    pair.

    Each term is weighted by novomodelo's own coefficient for that stage, so a
    VminOP term carries the same integrated stored-energy productivity (MWh/hm³)
    on both sides. A useful-volume term reads ``VARMUH`` (useful stored volume)
    directly; an absolute-storage term adds back the plant's ``min_storage_hm3``.

    Parameters
    ----------
    resolved:
        The case's resolved generic constraints.
    nw_hydro_df:
        Output of :func:`novomodelo_bridge.comparators.newave.readers.read_medias_hydro`.
        Columns ``newave_code`` (Int64), ``stage`` (Int64, 1-based as in
        MEDIAS — Sep-start studies use 9 for the first stage),
        ``variable`` (Utf8), ``value`` (Float64).
    nw_line_means:
        Output of
        :func:`novomodelo_bridge.comparators.newave.readers.read_nwlistop_intercambio`.
        One row per directional pair × stage.  We map each Novomodelo line to
        the matching directional row via the alignment's ``newave_de`` /
        ``newave_para`` fields (respecting the ``reversed`` flag for
        sign).
    alignment:
        Pre-built entity alignment.  Provides Novomodelo-line → (newave_de,
        newave_para) mappings and the ``reversed`` flag.
    id_map:
        Used to translate Novomodelo hydro IDs back to the source model plant codes.
    nw_offset:
        MEDIAS stage offset (e.g. 9 for a September-start study).  Used to convert the
        source model 1-based MEDIAS stages to Novomodelo 0-based.

    Returns
    -------
    polars.DataFrame
        Columns: ``constraint_id`` (Int32), ``stage_id`` (Int32, 0-based),
        ``lhs_value`` (Float64).  One row per (constraint, stage) pair that has at least
        one resolvable term.  Stages with missing variables produce no row (rather than
        a zero); this avoids misrepresenting "no source-model data" as "LHS = 0".
    """
    schema = {
        "constraint_id": pl.Int32,
        "stage_id": pl.Int32,
        "lhs_value": pl.Float64,
    }
    skipped = unsupported_constraint_ids(resolved.terms)
    terms = resolved.terms.filter(~pl.col("constraint_id").is_in(list(skipped)))
    if terms.is_empty():
        return pl.DataFrame(schema=schema)

    # --- Build hydro generation lookup: (novomodelo_hydro_id, stage_0based) -> MW
    hydro_gen: dict[tuple[int, int], float] = {}
    if not nw_hydro_df.is_empty():
        ghiduh = nw_hydro_df.filter(pl.col("variable") == "GHIDUH")
        for row in ghiduh.iter_rows(named=True):
            nw_code = int(row["newave_code"])
            try:
                novomodelo_id = id_map.hydro_id(nw_code)
            except KeyError:
                # Skip plants that are not in the LP (e.g. FICT, NE/NC).
                continue
            stage_0based = int(row["stage"]) - nw_offset
            if stage_0based < 0:
                continue
            hydro_gen[(novomodelo_id, stage_0based)] = float(row["value"])

    # --- Build hydro storage lookup: (novomodelo_hydro_id, stage_0based) -> hm3
    hydro_storage: dict[tuple[int, int], float] = {}
    if not nw_hydro_df.is_empty():
        varmuh = nw_hydro_df.filter(pl.col("variable") == "VARMUH")
        for row in varmuh.iter_rows(named=True):
            nw_code = int(row["newave_code"])
            try:
                novomodelo_id = id_map.hydro_id(nw_code)
            except KeyError:
                continue
            stage_0based = int(row["stage"]) - nw_offset
            if stage_0based < 0:
                continue
            hydro_storage[(novomodelo_id, stage_0based)] = float(row["value"])

    # --- Build line exchange lookup: (novomodelo_line_id, stage_0based) -> MW Aligned via
    # EntityAlignment.lines: each Novomodelo line records the source model directional pair
    # (newave_de → newave_para), which matches the Novomodelo (source, target) orientation by
    # construction.  NWLISTOP rows for the opposite (para, de) ordering carry the
    # opposite sign, so they are negated.
    line_flow: dict[tuple[int, int], float] = {}
    if not nw_line_means.is_empty() and alignment.lines:
        # Build (de, para, stage_0based) -> value for both directions.
        # NWLISTOP files are per (de, para) ordered pair; if our alignment
        # uses the opposite ordering we negate.
        df = nw_line_means.with_columns(
            (pl.col("stage") - nw_offset).cast(pl.Int64).alias("stage_0based")
        ).filter(pl.col("stage_0based") >= 0)
        nw_by_pair: dict[tuple[int, int, int], float] = {}
        for row in df.iter_rows(named=True):
            key = (
                int(row["from_submarket_code"]),
                int(row["to_submarket_code"]),
                int(row["stage_0based"]),
            )
            nw_by_pair[key] = float(row["value"])
        for line in alignment.lines:
            de = line.newave_de
            para = line.newave_para
            if de is None or para is None:
                continue
            for (de_k, para_k, s), val in nw_by_pair.items():
                if de_k == de and para_k == para:
                    line_flow[(line.novomodelo_line_id, s)] = val
                elif de_k == para and para_k == de:
                    # Reversed-ordering NWLISTOP row supplies the opposite sign
                    # of what our alignment expects.  Only fill if the
                    # canonical-direction row hasn't already populated this slot.
                    line_flow.setdefault((line.novomodelo_line_id, s), -val)

    # --- Per-(constraint, stage) LHS evaluation ---
    # The source model outputs are stage-level, so a per-block row's terms are
    # read once per stage (the lowest row block carries the stage's coefficients).
    stage_terms = (
        terms.sort(["constraint_id", "stage_id", "block_id"], nulls_last=False)
        .unique(subset=["constraint_id", "stage_id", "term_index"], keep="first")
        .sort(["constraint_id", "stage_id", "term_index"])
    )
    rows: list[dict] = []
    for (cid, stage), group in stage_terms.group_by(
        ["constraint_id", "stage_id"], maintain_order=True
    ):
        lhs = 0.0
        complete = True
        for term in group.iter_rows(named=True):
            vtype = term["variable_kind"]
            eid = int(term["entity_id"])
            if vtype in USEFUL_VOLUME_KINDS:
                val = hydro_storage.get((eid, stage))
            elif vtype in ("hydro_storage", "hydro_storage_final"):
                useful = hydro_storage.get((eid, stage))
                val = (
                    useful + resolved.min_storage.get(eid, 0.0)
                    if useful is not None
                    else None
                )
            elif vtype == "hydro_generation":
                val = hydro_gen.get((eid, stage))
            elif vtype == "line_direct":
                # ``line_direct`` is the non-negative flow in the canonical
                # (src<tgt) direction. The source model int*.out gives a single
                # signed value per (stage, pair); we take its positive part as the
                # stage-mean approximation. Per-block decomposition would require
                # parsing the per-patamar rows separately.
                signed = line_flow.get((eid, stage))
                val = max(0.0, signed) if signed is not None else None
            elif vtype == "line_reverse":
                signed = line_flow.get((eid, stage))
                val = max(0.0, -signed) if signed is not None else None
            else:  # line_exchange (signed net flow)
                val = line_flow.get((eid, stage))
            if val is None:
                complete = False
                break
            lhs += float(term["coefficient"]) * val
        if complete:
            rows.append({"constraint_id": cid, "stage_id": stage, "lhs_value": lhs})

    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows).with_columns(
        pl.col("constraint_id").cast(pl.Int32),
        pl.col("stage_id").cast(pl.Int32),
        pl.col("lhs_value").cast(pl.Float64),
    )
