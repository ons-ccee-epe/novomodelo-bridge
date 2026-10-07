"""Generic-constraint expression parsing and LHS evaluation (presentation-free).

Cobre encodes each generic constraint's left-hand side as a small textual
expression over LP variables, e.g.::

    "5.68 * hydro_storage(78)"
    "@rho_acum_h78 * hydro_storage(78) - line_exchange(4)"

This module owns the *domain* logic for those expressions — the term parser, the
scalar-parameter resolver, and the routine that evaluates the LHS from a Cobre
simulation's parquet output. It has **no** presentation dependency, so it is the single
shared home for both the dashboard (which renders the result) and the comparator (which
checks it against the source model). It deliberately lives at the package top level so
neither consumer has to import across the comparator↔UI boundary.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd
import polars as pl

_LOG = logging.getLogger(__name__)

# Matches terms like: [+/-] [coeff *] [@name *] variable_type(id)
# Examples:
#   "5.68 * hydro_storage_final(78)"       — literal coefficient
#   "hydro_generation(145)"                — implicit 1.0
#   "- line_exchange(4)"                   — implicit -1.0
#   "@rho_acum_h78 * hydro_storage_final(78)"  — @name coefficient (cobre sigil)
#   "0.5 * @rho_eq_h47 * hydro_generation(47)"  — literal × @name scale
#
# ``hydro_storage_final`` (cobre's end-of-stage stored volume Sᴷ) is what the
# converter emits for VminOP; the bare ``hydro_storage`` alias is still accepted
# for older cobre cases and normalised to it in :func:`parse_expression`. The
# longer name is listed first so the alternation matches it before the prefix.
_TERM_RE = re.compile(
    r"([+-]?\s*\d*\.?\d*)\s*\*?\s*(?:@([A-Za-z_][A-Za-z0-9_]*)\s*\*\s*)?"
    r"(hydro_storage_final|hydro_storage|hydro_generation"
    r"|line_exchange|line_direct|line_reverse)"
    r"\((\d+)\)"
)

# A scalar-parameter reference resolved against simulation columns:
#   rho_eq_h{id}    → equivalent_productivity_mw_per_m3s for that hydro
#   rho_acum_h{id}  → accumulated_productivity_mw_per_m3s for that hydro
_RHO_EQ_RE = re.compile(r"^rho_eq_h(\d+)$")
_RHO_ACUM_RE = re.compile(r"^rho_acum_h(\d+)$")


def resolve_param_to_column(name: str) -> tuple[str, int] | None:
    """Map a parameter ``name`` to the (column, hydro_id) used to look it up.

    Returns ``(simulation_column, hydro_id)`` for the two computed parameters
    cobre-bridge declares (``rho_eq_h{id}``, ``rho_acum_h{id}``), or ``None``
    when the name is unrecognised. Callers treat unrecognised parameters as if
    they had value 0 to avoid a hard error.
    """
    m = _RHO_ACUM_RE.match(name)
    if m is not None:
        return "accumulated_productivity_mw_per_m3s", int(m.group(1))
    m = _RHO_EQ_RE.match(name)
    if m is not None:
        return "equivalent_productivity_mw_per_m3s", int(m.group(1))
    return None


def scales_storage_by_rho_acum(constraint: dict) -> bool:
    """True when the constraint's LHS scales ``hydro_storage`` by ``@rho_acum_h{id}``.

    Both the source model's VminOP (security-curve) and DECOMP's RHE
    (stored-energy) constraints take this shape; RE/AGRINT/HQ/HV never
    reference ``@rho_acum_h{id}``, so this cleanly partitions the set.
    """
    for _, param_name, vtype, _ in parse_expression(constraint.get("expression", "")):
        if (
            vtype == "hydro_storage"
            and param_name is not None
            and param_name.startswith("rho_acum_h")
        ):
            return True
    return False


def load_rho_acum_overrides(cobre_case_dir: Path) -> dict[int, dict[int, float]]:
    """Load per-stage ρ_acum overrides from ``constraints/generic_parameters.json``.

    Returns ``{hydro_id: {stage_id: ρ_acum}}`` for every ``rho_acum_h{id}``
    entry the writer declared ``kind: "per_stage"`` — the energy-scaled
    coefficient (MWmonth/hm³) the VminOP/RHE LP actually uses at
    ``@rho_acum_h{id}`` in place of cobre's ``computed`` default (the point
    productivity ``accumulated_productivity_mw_per_m3s``). Pass the result as
    :func:`evaluate_constraint_expressions`'s ``rho_acum_overrides`` so the
    evaluated LHS matches what the LP actually solved, not the simulation's
    default productivity column.
    """
    path = cobre_case_dir / "constraints" / "generic_parameters.json"
    out: dict[int, dict[int, float]] = {}
    if not path.exists():
        return out
    try:
        with path.open() as f:
            params = json.load(f).get("scalar_parameters", [])
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("generic_parameters.json could not be parsed: %s", exc)
        return out
    for entry in params:
        m = re.fullmatch(r"rho_acum_h(\d+)", str(entry.get("name", "")))
        if m is None or entry.get("kind") != "per_stage":
            continue
        out[int(m.group(1))] = {int(s): float(v) for s, v in entry.get("values", [])}
    return out


def parse_expression(expr: str) -> list[tuple[float, str | None, str, int]]:
    """Parse a constraint LHS expression into term tuples.

    Returns a list of ``(literal_coeff, param_name_or_None, variable_type,
    entity_id)`` tuples. The effective coefficient on the variable is
    ``literal_coeff × value_of(param_name)`` when a parameter is present,
    or ``literal_coeff`` otherwise.

    Handles:
    - Leading ``-`` with no explicit coefficient → -1.0
    - No coefficient → 1.0
    - ``0.5 * hydro_generation(47)`` → 0.5
    - ``@rho_acum_h78 * hydro_storage(78)``
      → (1.0, "rho_acum_h78", "hydro_storage", 78)
    - ``- @rho_eq_h12 * hydro_generation(12)``
      → (-1.0, "rho_eq_h12", "hydro_generation", 12)
    """
    terms: list[tuple[float, str | None, str, int]] = []
    for m in _TERM_RE.finditer(expr):
        raw_coeff = m.group(1).replace(" ", "")
        param_name = m.group(2)
        var_type = m.group(3)
        # Bare ``hydro_storage_final`` is the end-of-stage volume Sᴷ, identical to
        # the legacy ``hydro_storage``; collapse to one vtype so every downstream
        # consumer (storage-only detection, column lookup) stays unchanged.
        if var_type == "hydro_storage_final":
            var_type = "hydro_storage"
        entity_id = int(m.group(4))
        if raw_coeff in ("", "+"):
            coeff = 1.0
        elif raw_coeff == "-":
            coeff = -1.0
        else:
            coeff = float(raw_coeff)
        terms.append((coeff, param_name, var_type, entity_id))
    return terms


def _apply_param_scale(
    sub: pd.DataFrame,
    param_name: str | None,
    entity_id: int,
    rho_acum_overrides: dict[int, dict[int, float]] | None,
) -> None:
    """Scale ``sub["_val"]`` in place by the resolved ``@param_name`` value.

    Mirrors cobre's own ``@name`` resolution at solve time. When the
    parameter resolves to ``accumulated_productivity_mw_per_m3s`` and
    *rho_acum_overrides* supplies a per-stage value for *entity_id*, the
    override wins row-by-row over the simulation's default productivity
    column — the VminOP/RHE writer always installs a ``kind: "per_stage"``
    override that the LP uses at solve time (see
    :func:`load_rho_acum_overrides`), so evaluating against the default
    column alone puts the LHS on a different scale than its own bound.
    """
    if param_name is None:
        return
    resolved = resolve_param_to_column(param_name)
    if resolved is None:
        # Unknown / unresolved parameter contributes zero so we don't
        # pollute LHS with stale unscaled values.
        sub["_val"] = 0.0
        return
    col, _ = resolved
    overrides = rho_acum_overrides.get(entity_id) if rho_acum_overrides else None
    has_override = col == "accumulated_productivity_mw_per_m3s" and overrides
    if col not in sub.columns and not has_override:
        sub["_val"] = 0.0
        return
    default = (
        sub[col].fillna(0.0) if col in sub.columns else pd.Series(0.0, index=sub.index)
    )
    if has_override:
        override_series = sub["stage_id"].map(overrides)
        scale = override_series.where(override_series.notna(), default)
    else:
        scale = default
    sub["_val"] = sub["_val"] * scale


def evaluate_constraint_expressions(
    constraints: list[dict],
    hydros_lf: pl.LazyFrame,
    exchanges_lf: pl.LazyFrame,
    rho_acum_overrides: dict[int, dict[int, float]] | None = None,
) -> pd.DataFrame:
    """Evaluate LHS of all generic constraints from simulation output.

    Variable lookups (``hydro_storage_final`` is normalised to ``hydro_storage``
    by :func:`parse_expression`, so both map to the same column):
    - ``hydro_storage(id)``    → ``storage_final_hm3`` where hydro_id=id, block_id=0
    - ``hydro_generation(id)`` → ``generation_mw``     where hydro_id=id  (per block)
    - ``line_exchange(id)``    → ``net_flow_mw``        where line_id=id   (per block)
    - ``line_direct(id)``      → ``direct_flow_mw``     where line_id=id   (per block)
    - ``line_reverse(id)``     → ``reverse_flow_mw``    where line_id=id   (per block)

    Storage-only constraints produce one row per (scenario, stage) with
    block_id=0. Mixed / generation / exchange constraints produce one row per
    (scenario, stage, block).

    Accepts LazyFrames and only collects the specific entity IDs referenced
    in constraint expressions, keeping memory usage minimal.

    ``rho_acum_overrides`` (``{hydro_id: {stage_id: value}}``, typically from
    :func:`load_rho_acum_overrides`) overrides the ``@rho_acum_h{id}``
    resolution per (hydro, stage) instead of the simulation's default
    ``accumulated_productivity_mw_per_m3s`` column — see
    :func:`_apply_param_scale`. ``None`` (the default) preserves prior
    behaviour.

    Returns DataFrame with columns:
        constraint_id, scenario_id, stage_id, block_id, lhs_value
    """
    # Parse ALL constraints to find which entity IDs / parameters are referenced.
    hydro_ids_needed: set[int] = set()
    line_ids_needed: set[int] = set()
    needs_rho_eq = False
    needs_rho_acum = False
    for c in constraints:
        for _coeff, param_name, vtype, eid in parse_expression(c["expression"]):
            if vtype.startswith("hydro"):
                hydro_ids_needed.add(eid)
            elif vtype in ("line_exchange", "line_direct", "line_reverse"):
                line_ids_needed.add(eid)
            if param_name is not None:
                resolved = resolve_param_to_column(param_name)
                if resolved is not None:
                    col, _ = resolved
                    if col == "equivalent_productivity_mw_per_m3s":
                        needs_rho_eq = True
                    elif col == "accumulated_productivity_mw_per_m3s":
                        needs_rho_acum = True

    # Collect only the referenced entities (tiny subset of full data). We pull
    # productivity columns when any @name reference depends on them so the
    # LHS evaluator can multiply the literal coefficient by the resolved
    # productivity at solve time (mirrors cobre's @name resolution).
    schema = hydros_lf.collect_schema()
    h0_cols = ["scenario_id", "stage_id", "hydro_id", "storage_final_hm3"]
    if needs_rho_eq and "equivalent_productivity_mw_per_m3s" in schema:
        h0_cols.append("equivalent_productivity_mw_per_m3s")
    if needs_rho_acum and "accumulated_productivity_mw_per_m3s" in schema:
        h0_cols.append("accumulated_productivity_mw_per_m3s")
    h0_pd = (
        hydros_lf.filter(
            (pl.col("block_id") == 0) & pl.col("hydro_id").is_in(list(hydro_ids_needed))
        )
        .select(h0_cols)
        .collect(engine="streaming")
        .to_pandas()
    )
    hg_cols = [
        "scenario_id",
        "stage_id",
        "block_id",
        "hydro_id",
        "generation_mw",
    ]
    if needs_rho_eq and "equivalent_productivity_mw_per_m3s" in schema:
        hg_cols.append("equivalent_productivity_mw_per_m3s")
    if needs_rho_acum and "accumulated_productivity_mw_per_m3s" in schema:
        hg_cols.append("accumulated_productivity_mw_per_m3s")
    hg_pd = (
        hydros_lf.filter(pl.col("hydro_id").is_in(list(hydro_ids_needed)))
        .select(hg_cols)
        .collect(engine="streaming")
        .to_pandas()
    )
    # Pull all three flow columns when line-touching terms are referenced;
    # the per-term branch below picks whichever column matches the
    # variable type.  ``direct_flow_mw`` and ``reverse_flow_mw`` are
    # the non-negative LP primitives behind cobre's ``line_direct`` /
    # ``line_reverse`` variables; ``net_flow_mw = direct - reverse`` is
    # the signed shorthand referenced by ``line_exchange``.
    ex_cols = [
        "scenario_id",
        "stage_id",
        "block_id",
        "line_id",
        "net_flow_mw",
        "direct_flow_mw",
        "reverse_flow_mw",
    ]
    ex_pd = (
        (
            exchanges_lf.filter(pl.col("line_id").is_in(list(line_ids_needed)))
            .select(ex_cols)
            .collect(engine="streaming")
            .to_pandas()
        )
        if line_ids_needed
        else pd.DataFrame(columns=ex_cols)
    )

    all_results: list[pd.DataFrame] = []

    for c in constraints:
        cid = c["id"]
        expr = c["expression"]
        terms = parse_expression(expr)
        if not terms:
            continue

        var_types = {t[2] for t in terms}
        storage_only = var_types == {"hydro_storage"}

        if storage_only:
            # One LHS value per (scenario, stage) — use block_id 0
            base = h0_pd[["scenario_id", "stage_id"]].drop_duplicates().copy()
            base["_lhs"] = 0.0
            for coeff, param_name, _vtype, eid in terms:
                sub = h0_pd[h0_pd["hydro_id"] == eid].copy()
                if sub.empty:
                    continue
                sub["_val"] = sub["storage_final_hm3"]
                _apply_param_scale(sub, param_name, eid, rho_acum_overrides)
                sub = sub[["scenario_id", "stage_id", "_val"]]
                merged = base.merge(sub, on=["scenario_id", "stage_id"], how="left")
                merged["_val"] = merged["_val"].fillna(0.0)
                base["_lhs"] = base["_lhs"].values + coeff * merged["_val"].values
            base["constraint_id"] = cid
            base["block_id"] = 0
            base = base.rename(columns={"_lhs": "lhs_value"})
            all_results.append(
                base[
                    [
                        "constraint_id",
                        "scenario_id",
                        "stage_id",
                        "block_id",
                        "lhs_value",
                    ]
                ]
            )
        else:
            # Per (scenario, stage, block) — need consistent block grid.
            base = (
                hg_pd[["scenario_id", "stage_id", "block_id"]].drop_duplicates().copy()
            )
            base["_lhs"] = 0.0

            for coeff, param_name, vtype, eid in terms:
                if vtype == "hydro_storage":
                    sub = h0_pd[h0_pd["hydro_id"] == eid].copy()
                    if sub.empty:
                        continue
                    sub["_val"] = sub["storage_final_hm3"]
                    join_cols = ["scenario_id", "stage_id"]
                elif vtype == "hydro_generation":
                    sub = hg_pd[hg_pd["hydro_id"] == eid].copy()
                    if sub.empty:
                        continue
                    sub["_val"] = sub["generation_mw"]
                    join_cols = ["scenario_id", "stage_id", "block_id"]
                else:  # line_exchange / line_direct / line_reverse
                    sub = ex_pd[ex_pd["line_id"] == eid].copy()
                    if sub.empty:
                        continue
                    if vtype == "line_direct":
                        sub["_val"] = sub["direct_flow_mw"]
                    elif vtype == "line_reverse":
                        sub["_val"] = sub["reverse_flow_mw"]
                    else:
                        sub["_val"] = sub["net_flow_mw"]
                    join_cols = ["scenario_id", "stage_id", "block_id"]

                _apply_param_scale(sub, param_name, eid, rho_acum_overrides)

                sub = sub[join_cols + ["_val"]]
                merged = base.merge(sub, on=join_cols, how="left")
                merged["_val"] = merged["_val"].fillna(0.0)
                base["_lhs"] = base["_lhs"].values + coeff * merged["_val"].values

            base["constraint_id"] = cid
            base = base.rename(columns={"_lhs": "lhs_value"})
            all_results.append(
                base[
                    [
                        "constraint_id",
                        "scenario_id",
                        "stage_id",
                        "block_id",
                        "lhs_value",
                    ]
                ]
            )

    if not all_results:
        return pd.DataFrame(
            columns=[
                "constraint_id",
                "scenario_id",
                "stage_id",
                "block_id",
                "lhs_value",
            ]
        )
    return pd.concat(all_results, ignore_index=True)
