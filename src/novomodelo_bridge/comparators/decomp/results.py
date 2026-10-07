"""Side-by-side comparison of a DECOMP run against its converted Cobre run.

This is the minimal slice: it aligns the source model's shipped operation
tables (``dec_oper_*.csv``, directly in the deck directory) onto Cobre's
simulation output and reports
per-variable divergence. No tolerance verdict and no HTML — the point is that
both sides are read, aligned, and compared on the same quantities.

Three conventions, each chosen so neither side is silently privileged:

- **Stage aggregation is taken, not recomputed.** Every ``dec_oper_*`` table
  carries a ``patamar``-null row per (stage, node) holding the source model's
  own duration-weighted stage value; the comparison reads that row instead of
  re-deriving one from the per-block rows, so a block-weighting convention
  cannot drift between the two products.
- **Scenario averaging is probability-weighted on the source side** for
  every operation variable this module compares -- hydro/thermal/bus/
  interchange, the REE energy rollup, evaporation, and the Constraints tab's
  DECOMP-side LHS, not only the Overview operating cost. The monthly
  scenario-fan stage's openings are not equiprobable, so the source model's
  expected value at each (stage, entity) must use the real tree
  ``probabilidade`` the ``relato``/``relato2`` cost tables carry per row --
  :func:`_scenario_probabilities` builds that per-(stage, scenario) lookup
  once, in :func:`build_decomp_dataset`, and every :func:`_scenario_mean`
  call site threads it through via the ``probabilities`` parameter. A
  deterministic (single-scenario) stage's weighted mean over its one row is
  identically that row's value, so only a genuine scenario fan's aggregate
  changes; a deck with no ``relato``/``relato2`` probability source at all
  degrades to :func:`_scenario_mean`'s original unweighted mean rather than
  raising. cobre's per-stage statistics are already an expectation over its
  own (equiprobable) simulation scenarios, so the two sides stay comparable.
  **The Overview operating cost** (:func:`_cost_frames`) predates this and
  keeps its own, independent path: it reads the same real ``probabilidade``
  directly off its already-unioned ``relato``/``relato2`` cost frame via
  :func:`_probability_weighted_stage_cost`, rather than joining in
  :func:`_scenario_probabilities`'s lookup.
- **Storage is compared as useful volume.** The source model reports useful
  volume; Cobre's ``storage_final_hm3`` is absolute, so the registry's
  ``min_storage_hm3`` is subtracted before comparing.

Entities the id map cannot resolve are counted and reported, never dropped in
silence.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from cobre_bridge.cobre import readers as cobre_readers
from cobre_bridge.cobre.case_io import case_dir_for
from cobre_bridge.comparators import fpha
from cobre_bridge.comparators.decomp.readers import (
    read_dec_desvfpha,
    read_dec_estatfpha,
    read_dec_oper_evap,
    read_dec_oper_interc,
    read_dec_oper_ree,
    read_dec_oper_rhesoft,
    read_dec_oper_sist,
    read_dec_oper_usih,
    read_dec_oper_usit,
    read_decomp_tim,
    read_eco_fpha,
    read_relato2_costs,
    read_relato_convergence,
    read_relato_costs,
    read_relato_membership,
)
from cobre_bridge.comparators.model import PercentileData, ResultComparison
from cobre_bridge.core.diagnostics import Diagnostic, Severity, emit

if TYPE_CHECKING:
    from cobre_bridge.comparators.dataset import ComparisonDataset
    from cobre_bridge.decomp.case import DecompCase
    from cobre_bridge.decomp.constraint_registers import (
        ConstraintRecord,
        ConstraintTerm,
    )
    from cobre_bridge.decomp.id_map import DecompIdMap

_LOG = logging.getLogger(__name__)

#: Minimum turbined flow (m³/s) for the derived gen/turbined productivity
#: to be meaningful -- mirrors
#: :data:`cobre_bridge.comparators.newave.results._PRODUCTIVITY_TURB_EPS`: near-zero
#: turbining makes generation/turbined an undefined 0/0 on both sides.
_PRODUCTIVITY_TURBINED_EPS: float = 1.0e-6

#: cobre's ``stored_energy_final_mwh`` is MWh; the source model's
#: own REE ``earm_final_MWmes`` is MWmês (average MW sustained over a month).
#: 730 h/month is the same implicit hours-per-month convention
#: `report_builder`'s Energy Balance chart already applies to the SIN-total
#: EARM overlay (``cobre_aggregate_chart(..., nw_factor=730.0, ...)``) --
#: reused here rather than re-derived so the two EARM reconciliations never
#: drift apart.
_EARM_MWH_TO_MWMES: float = 730.0


@dataclass(frozen=True)
class _Variable:
    """One comparable quantity and its column on each side."""

    level: str
    name: str
    source_column: str
    cobre_column: str
    unit: str


_HYDRO_VARIABLES: tuple[_Variable, ...] = (
    _Variable("hydro", "generation", "geracao_MW", "generation_mw", "MW"),
    _Variable("hydro", "turbined", "vazao_turbinada_m3s", "turbined_m3s", "m³/s"),
    _Variable("hydro", "spillage", "vazao_vertida_m3s", "spillage_m3s", "m³/s"),
    _Variable("hydro", "outflow", "vazao_defluente_m3s", "outflow_m3s", "m³/s"),
    _Variable("hydro", "storage", "volume_util_final_hm3", "useful_storage_hm3", "hm³"),
)

_THERMAL_VARIABLES: tuple[_Variable, ...] = (
    _Variable("thermal", "generation", "geracao_MW", "generation_mw", "MW"),
)

_BUS_VARIABLES: tuple[_Variable, ...] = (
    _Variable("bus", "deficit", "deficit_MW", "deficit_mw", "MW"),
    _Variable("bus", "spot_price", "cmo", "spot_price", "$/MWh"),
)

#: (level, ``_Variable.name``) -> the canonical variable name the shared
#: chart layer keys off (``analyze.build_results_dataset`` downstream), not
#: the decomp-local ``_Variable.name``. Storage maps to ``storage_final_hm3``
#: to match the newave-side comparator, though the comparison is on *useful*
#: volume — do not "fix" the value to the ``_Variable``/column spelling.
_CANONICAL_VARIABLE: dict[tuple[str, str], str] = {
    ("hydro", "generation"): "generation_mw",
    ("hydro", "turbined"): "turbined_m3s",
    ("hydro", "spillage"): "spillage_m3s",
    ("hydro", "outflow"): "outflow_m3s",
    ("hydro", "storage"): "storage_final_hm3",
    ("thermal", "generation"): "generation_mw",
    ("bus", "deficit"): "deficit_mw",
    ("bus", "spot_price"): "spot_price",
}


def _stage_rows(frame: pl.DataFrame) -> pl.DataFrame:
    """Keep the source model's own stage-aggregate rows (``patamar`` null).

    Falls back to a duration-weighted fold of the per-block rows if a deck
    ever ships without the aggregate row, so the comparison degrades to a
    computed value instead of an empty frame.
    """
    if "patamar" not in frame.columns:
        return frame
    aggregate = frame.filter(pl.col("patamar").is_null())
    if not aggregate.is_empty():
        return aggregate.drop("patamar")
    _LOG.warning(
        "No stage-aggregate rows (patamar null) found; folding per-block rows "
        "by duration instead"
    )
    return frame


def _weighted_group_mean(
    frame: pl.DataFrame,
    group: list[str],
    columns: list[str],
    *,
    weight_col: str = "probabilidade",
) -> pl.DataFrame:
    """Per-group weighted mean with a zero-weight fallback -- never divide by ~0."""
    weight = pl.col(weight_col)
    weight_sum = weight.sum()
    aggs = [
        pl.when(weight_sum.abs() > 1e-12)
        .then((pl.col(c) * weight).sum() / weight_sum)
        .otherwise(pl.col(c).mean())
        .alias(c)
        for c in columns
    ]
    return frame.group_by(group).agg(aggs).sort(group)


def _scenario_mean(
    frame: pl.DataFrame,
    key: str,
    columns: list[str],
    *,
    entity_column: str | list[str] | None,
    probabilities: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Average the stage rows over the source model's nodes.

    ``key`` is the stage column; ``entity_column`` narrows the grouping to one
    entity per stage: ``None`` for tables that are already one row per stage,
    a single column name for a single-code entity, or a list of column names
    for a composite key -- e.g. an interchange corridor's
    ``(codigo_submercado_de, codigo_submercado_para)`` pair.

    ``probabilities`` -- a :func:`_scenario_probabilities`-shaped
    ``(estagio, cenario, probabilidade)`` lookup -- turns the fold into a
    probability-weighted expectation over *frame*'s own ``cenario`` openings
    instead of a plain mean: the monthly scenario-fan stage's openings are
    not equiprobable, so the source model's own tree probabilities give the
    correct expected value there. ``None``, an empty frame, or a *frame*
    that carries no ``cenario`` column of its own each keep the existing
    unweighted ``.mean()`` behaviour verbatim -- every pre-existing caller is
    unaffected by this parameter's default. Otherwise *frame* is left-joined
    onto *probabilities* by ``[key, "cenario"]`` (both sides cast to
    ``Int64`` first, so a dtype mismatch never breaks the join); a scenario
    absent from *probabilities* defaults to weight ``1.0`` (``fill_null``)
    rather than being dropped, and each present column becomes the
    per-group weighted mean ``sum(col * probabilidade) / sum(probabilidade)``
    -- falling back to the plain mean for any group whose weight sum is ~0,
    never dividing by zero. A deterministic (single-scenario) stage's
    weighted mean over its one row is identically that row's own value, so
    only a genuine scenario fan's aggregate changes.
    """
    present = [c for c in columns if c in frame.columns]
    if entity_column is None:
        entity_columns: list[str] = []
    elif isinstance(entity_column, str):
        entity_columns = [entity_column]
    else:
        entity_columns = list(entity_column)
    group = [key, *entity_columns]

    if (
        probabilities is None
        or probabilities.is_empty()
        or "cenario" not in frame.columns
    ):
        return (
            frame.group_by(group)
            .agg([pl.col(c).mean().alias(c) for c in present])
            .sort(group)
        )

    joined = (
        frame.with_columns(pl.col(key).cast(pl.Int64), pl.col("cenario").cast(pl.Int64))
        .join(
            probabilities.rename({"estagio": key}),
            on=[key, "cenario"],
            how="left",
        )
        .with_columns(pl.col("probabilidade").fill_null(1.0))
    )
    return _weighted_group_mean(joined, group, present)


#: Canonical schema :func:`_scenario_probabilities` returns -- the
#: ``_scenario_mean``-consumable ``(estagio, cenario, probabilidade)``
#: lookup, independent of which ``dec_oper_*`` table a given caller folds.
_SCENARIO_PROBABILITY_SCHEMA: dict[str, type[pl.DataType]] = {
    "estagio": pl.Int64,
    "cenario": pl.Int64,
    "probabilidade": pl.Float64,
}


def _union_relato_reports(raw: pl.DataFrame, monthly: pl.DataFrame) -> pl.DataFrame:
    """Union ``relato``/``relato2`` -- relato2 wins for stages it covers."""
    if monthly.is_empty():
        return raw
    if raw.is_empty():
        return monthly
    fan_stages = monthly["estagio"].unique().to_list()
    return pl.concat(
        [raw.filter(~pl.col("estagio").is_in(fan_stages)), monthly],
        how="diagonal_relaxed",
    )


def _scenario_probabilities(decomp_dir: Path) -> pl.DataFrame:
    """Real per-(stage, scenario) tree probabilities, from ``relato``/``relato2``.

    Every physical ``dec_oper_*`` table (``dec_oper_usih``/``_usit``/
    ``_sist``/``_interc``/``_ree``/``_evap``/``_rhesoft``) carries a
    ``cenario`` index but no ``probabilidade`` column of its own -- the real
    tree probabilities live only in the ``relatorio_operacao_custos`` table
    the ``relato``/``relato2`` general reports carry, and the ``cenario``
    indices align between the two (verified: ``dec_oper_usit``'s stage-4
    per-scenario thermal cost equals ``relato2``'s per-scenario values for
    the same stage/scenario). This builds the ``(estagio, cenario,
    probabilidade)`` lookup :func:`_scenario_mean` consumes, unioning the two
    reports exactly the way :func:`_cost_frames` does: ``relato``
    (:func:`read_relato_costs`) supplies the deterministic weekly stages;
    ``relato2`` (:func:`read_relato2_costs`, optional) supplies the real,
    non-equiprobable opening probabilities of the scenario-fan (monthly)
    stage(s) ``relato`` omits, and is authoritative for any stage it covers
    -- that stage is dropped from the ``relato`` side before the union, so it
    is never double-counted.

    Built once by :func:`build_decomp_dataset` and threaded through every
    :func:`_scenario_mean` call site, rather than re-read per site -- parsing
    ``relato2`` is not free.

    Returns an empty (schema-valid) frame -- :func:`_scenario_mean` then
    falls back to its unweighted mean -- when :func:`read_relato_costs`
    raises ``FileNotFoundError``/``ValueError`` (no source-model cost report
    at all) AND ``relato2`` is also empty. Never raises on its own.
    """
    try:
        raw = read_relato_costs(decomp_dir)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No relato cost table for scenario probabilities: %s", exc)
        raw = pl.DataFrame()

    monthly = read_relato2_costs(decomp_dir)

    if raw.is_empty() and monthly.is_empty():
        return pl.DataFrame(schema=_SCENARIO_PROBABILITY_SCHEMA)
    combined = _union_relato_reports(raw, monthly)

    return (
        combined.select("estagio", "cenario", "probabilidade")
        .unique()
        .cast(_SCENARIO_PROBABILITY_SCHEMA)
        .sort(["estagio", "cenario"])
    )


def _result_diff(nw_value: float, cobre_value: float) -> tuple[float, float | None]:
    """Absolute and relative difference for one ``ResultComparison`` row.

    Mirrors :func:`cobre_bridge.comparators.newave.results._compute_diff` exactly, so
    the two per-variable stat kernels in play here (this module's own tidy
    rows and the shared ``ResultComparison`` one) agree on what "the
    difference" means.
    """
    abs_diff = abs(nw_value - cobre_value)
    rel_diff = abs_diff / abs(nw_value) if abs(nw_value) > 1e-10 else None
    return abs_diff, rel_diff


def _result_comparisons(
    source: pl.DataFrame,
    cobre: pl.DataFrame,
    variables: tuple[_Variable, ...],
    *,
    names: dict[int, str],
) -> list[ResultComparison]:
    """Join one level's two frames into ``ResultComparison`` rows.

    Emits the canonical
    :class:`~cobre_bridge.comparators.model.ResultComparison` shape
    :func:`build_decomp_dataset` hands to
    :func:`cobre_bridge.comparators.analyze.build_results_dataset`, keyed by
    :data:`_CANONICAL_VARIABLE`.
    """
    joined = source.join(cobre, on=["entity_id", "stage_id"], how="inner")
    if joined.is_empty():
        return []

    applicable = [
        var
        for var in variables
        if var.source_column in joined.columns and var.cobre_column in joined.columns
    ]
    if not applicable:
        return []

    results: list[ResultComparison] = []
    for row in joined.iter_rows(named=True):
        entity_id = int(row["entity_id"])
        stage = int(row["stage_id"])
        newave_code = int(row["newave_code"])
        entity_name = names.get(entity_id, "")
        for var in applicable:
            nw_val = row[var.source_column]
            cobre_val = row[var.cobre_column]
            if nw_val is None or cobre_val is None:
                continue
            nw_value = float(nw_val)
            cobre_value = float(cobre_val)
            abs_diff, rel_diff = _result_diff(nw_value, cobre_value)
            results.append(
                ResultComparison(
                    entity_type=var.level,
                    entity_name=entity_name,
                    newave_code=newave_code,
                    cobre_id=entity_id,
                    stage=stage,
                    variable=_CANONICAL_VARIABLE[(var.level, var.name)],
                    newave_value=nw_value,
                    cobre_value=cobre_value,
                    abs_diff=abs_diff,
                    rel_diff=rel_diff,
                )
            )
    return results


def _hydro_productivity_results(
    hydro_results: Sequence[ResultComparison],
) -> list[ResultComparison]:
    """Derive per-(plant, stage) realized productivity = generation / turbined.

    Consumes the ``generation_mw``/``turbined_m3s`` hydro
    ``ResultComparison`` rows :func:`_result_comparisons` already produced
    (the source values come from ``dec_oper_usih``'s stage-aggregate
    rows, the Cobre values from the cobre hydro means, joined on the
    id-map-resolved ``(entity_id, stage_id)`` pair) -- so this reuses that
    exact alignment/restriction instead of a separate lookup or filter.
    Mirrors :mod:`cobre_bridge.comparators.newave.results`'s own ``_compare_hydros``
    productivity derivation. TRACKED zero-guard semantics: a plant/stage
    where either side's turbined flow is at/near zero -- or one side's row is
    simply absent -- is DROPPED, never null-kept, matching the NEWAVE-side
    behaviour for the exact same widget (gen/turbined is an undefined 0/0
    there).

    Returns a list meant to be appended onto the caller's main ``results``
    list, not held separately: there is no dedicated ``PercentileData``
    field for the per-stage productivity frame --
    :func:`cobre_bridge.comparators.analyze.build_results_dataset` derives
    ``dataset.metadata["productivity_per_stage"]`` by filtering the *same*
    ``results`` list it was handed for ``entity_type == "hydro"`` /
    ``variable == "productivity_mw_per_m3s"`` rows.
    """
    generation: dict[tuple[int, int], tuple[float, float, int, str]] = {}
    turbined: dict[tuple[int, int], tuple[float, float]] = {}
    for r in hydro_results:
        if r.entity_type != "hydro":
            continue
        key = (r.newave_code, r.stage)
        if r.variable == "generation_mw":
            generation[key] = (
                r.newave_value,
                r.cobre_value,
                r.cobre_id,
                r.entity_name,
            )
        elif r.variable == "turbined_m3s":
            turbined[key] = (r.newave_value, r.cobre_value)

    productivity: list[ResultComparison] = []
    for key, (nw_gen, cb_gen, cobre_id, name) in sorted(generation.items()):
        turb = turbined.get(key)
        if turb is None:
            continue
        nw_turb, cb_turb = turb
        if (
            nw_turb <= _PRODUCTIVITY_TURBINED_EPS
            or cb_turb <= _PRODUCTIVITY_TURBINED_EPS
        ):
            continue
        nw_code, stage = key
        nw_value = nw_gen / nw_turb
        cobre_value = cb_gen / cb_turb
        abs_diff, rel_diff = _result_diff(nw_value, cobre_value)
        productivity.append(
            ResultComparison(
                entity_type="hydro",
                entity_name=name,
                newave_code=nw_code,
                cobre_id=cobre_id,
                stage=stage,
                variable="productivity_mw_per_m3s",
                newave_value=nw_value,
                cobre_value=cobre_value,
                abs_diff=abs_diff,
                rel_diff=rel_diff,
            )
        )
    return productivity


def reconcile_kdollars_to_reais(value: float) -> float:
    """Convert a native k$ cost value to R$ (×10³).

    Every cost the source model reports — `read_relato_costs`,
    `read_relato_expected_cost`, `read_dec_oper_gnl`'s ``custo_geracao`` — is
    in k$ (thousands of BRL), while cobre reports costs in R$. Silently
    mixing the two is the same unit trap documented in
    `project_decomp_fcf_unit_conversion_bug`: cobre's boundary FCF coefficients
    were consumed verbatim in k$ against a R$-denominated model, undervaluing
    water by three orders of magnitude. This helper is the single conversion
    site — readers stay in native k$, and callers convert once, explicitly.

    Downstream, two different R$ conventions apply: the Overview cost dict
    uses plain R$ (this factor, ×10³), while `nw_sin` uses
    10⁶ R$ (an additional ÷10⁶ on top of this factor).
    """
    return value * 1e3


#: Canonical convergence-chart schema -- matches
#: :func:`cobre_readers.read_cobre_convergence`'s own return schema exactly,
#: so :func:`~cobre_bridge.comparators.charts.convergence_chart` can read
#: either side without a source-specific branch.
_CONVERGENCE_SCHEMA: dict[str, type[pl.DataType]] = {
    "iteration": pl.Int64,
    "lower_bound": pl.Float64,
    "upper_bound_mean": pl.Float64,
}


def _decomp_convergence_frame(decomp_dir: Path) -> pl.DataFrame:
    """DECOMP-side convergence bounds in the canonical chart schema.

    Renames ``relato.convergencia``'s ``iteracao``/``zinf``/
    ``zsup`` onto the ``iteration``/``lower_bound``/``upper_bound_mean``
    columns :func:`cobre_readers.read_cobre_convergence` emits -- the shape
    :func:`~cobre_bridge.comparators.charts.convergence_chart` reads on each
    side separately (not a per-iteration join shape). The
    iteration axis stays 1-based as reported, no offset -- the chart plots
    it directly. ``zinf``/``zsup`` are native k$ and are reconciled to R$
    (:func:`reconcile_kdollars_to_reais`, x1e3) so the bounds are
    unit-comparable to cobre's R$-denominated convergence.

    A missing/empty relato (``read_relato_convergence`` raising
    ``FileNotFoundError``/``ValueError``) degrades to an empty frame with
    this same canonical schema instead of aborting
    :func:`build_decomp_dataset` -- the chart then renders Cobre-only.
    """
    try:
        source = read_relato_convergence(decomp_dir)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.warning("No source-model convergence table: %s", exc)
        return pl.DataFrame(schema=_CONVERGENCE_SCHEMA)

    columns = {c.lower(): c for c in source.columns}
    # zinf/zsup are native k$ (like every DECOMP cost); cobre's bounds are R$.
    # Convert once via the shared factor (x1e3) so the two convergence lines are
    # unit-comparable -- without it the DECOMP bounds sit ~1000x below cobre's.
    k_to_r = reconcile_kdollars_to_reais(1.0)
    return source.select(
        pl.col(columns["iteracao"]).cast(pl.Int64).alias("iteration"),
        (pl.col(columns["zinf"]).cast(pl.Float64) * k_to_r).alias("lower_bound"),
        (pl.col(columns["zsup"]).cast(pl.Float64) * k_to_r).alias("upper_bound_mean"),
    )


def _hydro_side(
    decomp_dir: Path,
    id_map_hydro: dict[int, int],
    *,
    probabilities: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, list[int]]:
    """Per-(hydro, stage) ``dec_oper_usih`` means. *probabilities* -- a
    :func:`_scenario_probabilities` lookup -- weights the scenario fold by
    the real tree probabilities instead of :func:`_scenario_mean`'s plain
    mean; ``None`` (the default) keeps the unweighted fold."""
    frame = _stage_rows(read_dec_oper_usih(decomp_dir))
    columns = [v.source_column for v in _HYDRO_VARIABLES]
    aggregated = _scenario_mean(
        frame,
        "estagio",
        columns,
        entity_column="codigo_usina",
        probabilities=probabilities,
    )
    return _map_entities(aggregated, "codigo_usina", id_map_hydro)


def _thermal_side(
    decomp_dir: Path,
    id_map_thermal: dict[int, int],
    *,
    probabilities: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, list[int]]:
    """Per-(thermal, stage) ``dec_oper_usit`` means. See :func:`_hydro_side`
    for *probabilities*'s scenario-weighting contract."""
    frame = _stage_rows(read_dec_oper_usit(decomp_dir))
    columns = [v.source_column for v in _THERMAL_VARIABLES]
    aggregated = _scenario_mean(
        frame,
        "estagio",
        columns,
        entity_column="codigo_usina",
        probabilities=probabilities,
    )
    return _map_entities(aggregated, "codigo_usina", id_map_thermal)


def _bus_side(
    decomp_dir: Path,
    id_map_bus: dict[int, int],
    *,
    probabilities: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, list[int]]:
    """Per-(bus, stage) ``dec_oper_sist`` means. See :func:`_hydro_side` for
    *probabilities*'s scenario-weighting contract."""
    frame = _stage_rows(read_dec_oper_sist(decomp_dir))
    columns = [v.source_column for v in _BUS_VARIABLES]
    aggregated = _scenario_mean(
        frame,
        "estagio",
        columns,
        entity_column="codigo_submercado",
        probabilities=probabilities,
    )
    return _map_entities(aggregated, "codigo_submercado", id_map_bus)


# --- interchange corridor -> cobre line alignment ---
#
# DECOMP publishes exchange per submarket-pair corridor (``dec_oper_interc``);
# the converted case publishes it per directed cobre line
# (``system/lines.json``, read as ``net_flow_mw`` by
# ``cobre_readers.read_cobre_line_means``). Usually one corridor -> one line,
# but the converter-created transhipment bus
# (``DecompIdMap.transhipment_bus_id``) can realize a corridor as a pair of
# legs; this block reconciles the two.
#
# Out-of-file column/direction contracts this alignment depends on:
# - ``system/lines.json`` keys endpoints ``source_bus_id``/``target_bus_id``,
#   not ``from_bus``/``to_bus`` (``convert_lines`` on both tracks).
# - ``net_flow_mw`` is positive in the line's own ``source_bus_id ->
#   target_bus_id`` direction (``comparators/newave/alignment.py``'s ``LineEntity``).
# - ``dec_oper_interc`` (``idecomp`` 1.14) names its corridor endpoints
#   ``codigo_submercado_de``/``codigo_submercado_para`` (not
#   ``submercado_de``/``submercado_para``), plus ``intercambio_origem_MW``/
#   ``intercambio_destino_MW``/``perdas_MW``.


def _read_cobre_lines_index(cobre_output_dir: Path) -> dict[tuple[int, int], int]:
    """``{(source_bus_id, target_bus_id): line_id}`` from ``system/lines.json``.

    A missing ``system/lines.json`` -- e.g. a case predating the ``IA``
    exchange-network converter (``decomp/converters/network.py::convert_lines``)
    -- yields an empty index rather than raising: :func:`_corridor_line_alignment`
    then resolves nothing, and :func:`_interc_side` reports every corridor as
    unresolved instead of failing the comparison (the Network tab degrades to
    empty). Built from :func:`cobre_readers.read_cobre_lines`'s raw ``"lines"``
    list -- this function owns only the bus-pair index transform.

    [ASSUMPTION] (star topology): more than one
    line declared between the *same ordered* bus pair makes that leg
    ambiguous -- both are dropped from the index so any corridor that would
    resolve through it is reported unresolved rather than guessing which
    line applies.
    """
    index: dict[tuple[int, int], int] = {}
    ambiguous: set[tuple[int, int]] = set()
    for line in cobre_readers.read_cobre_lines(cobre_output_dir):
        key = (int(line["source_bus_id"]), int(line["target_bus_id"]))
        if key in index:
            ambiguous.add(key)
            continue
        index[key] = int(line["id"])
    for key in ambiguous:
        index.pop(key, None)
        _LOG.warning(
            "Multiple cobre lines declared for bus pair %s; excluding it from "
            "the corridor alignment instead of guessing which line applies",
            key,
        )
    return index


def _resolve_leg(
    lines_index: dict[tuple[int, int], int], bus_a: int, bus_para: int
) -> tuple[int, int] | None:
    """Return the ``(line_id, sign)`` leg realizing flow ``bus_a -> bus_para``.

    ``sign`` orients the cobre line's own ``source_bus_id -> target_bus_id``
    convention onto the requested direction: ``+1`` when a line already runs
    ``bus_a -> bus_para``, ``-1`` when the matching line runs the reverse.
    ``None`` when no cobre line connects the pair in either direction.
    """
    line_id = lines_index.get((bus_a, bus_para))
    if line_id is not None:
        return line_id, 1
    line_id = lines_index.get((bus_para, bus_a))
    if line_id is not None:
        return line_id, -1
    return None


def _corridor_line_alignment(
    cobre_output_dir: Path, id_map: DecompIdMap
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Map every resolvable DECOMP corridor onto its cobre line leg(s).

    Keyed by cobre bus-id pair ``(bus_de, bus_para)`` -- built over every
    ordered pair of *declared* subsystem buses (the transhipment bus is
    never a corridor endpoint in its own right, D-UNITS) -- to the ordered
    list of cobre ``(line_id, sign)`` legs realizing it:

    - a single leg when a direct cobre line connects the pair, in either
      direction (requirement 3's "direct" branch);
    - else the two-leg ``bus_de -> transhipment_bus_id -> bus_para`` path,
      when both legs exist (requirement 3's transhipment branch);
    - else the pair is simply absent from the returned mapping.

    :func:`_corridor_line_alignment` itself never raises and never reports:
    it is total in the sense of trying every declared bus pair, but a pair
    with no realizing line(s) is left out of the dict rather than tracked
    here. :func:`_interc_side` is the layer that turns "absent from this
    mapping" into a reported, never-silently-dropped unresolved corridor,
    scoped to the corridors ``dec_oper_interc`` actually carries.

    [ASSUMPTION]: the two-leg
    reconciliation assumes the converted case's exchange topology is a star
    through the transhipment bus, with at most one ``de<->IV<->para`` path
    per corridor -- guaranteed structurally here since ``DecompIdMap`` has
    exactly one transhipment bus and :func:`_read_cobre_lines_index` drops
    (rather than arbitrarily picks) any bus pair wired by more than one
    line. Alternatives considered: (a) direct submarket-pair match only --
    fails on the star topology, rejected; (b) transhipment-aware two-leg
    reconciliation -- chosen; (c) compare at the bus-injection level instead
    of per line -- out of scope, not implemented here. A deck whose
    exchange graph is genuinely non-star (e.g. two different corridors that
    would need to share one leg) is not guessed at by this function -- each
    corridor is resolved independently, so a shared leg is assigned the same
    value by both corridors rather than an aggregate; that residual gap is a
    known limitation of the star assumption, not something this function
    reconciles further.
    """
    lines_index = _read_cobre_lines_index(cobre_output_dir)
    if not lines_index:
        return {}

    transhipment = id_map.transhipment_bus_id
    real_bus_ids = sorted({id_map.bus_id(code) for code in id_map.bus_codes})

    alignment: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for bus_de in real_bus_ids:
        for bus_para in real_bus_ids:
            if bus_de == bus_para:
                continue
            direct = _resolve_leg(lines_index, bus_de, bus_para)
            if direct is not None:
                alignment[(bus_de, bus_para)] = [direct]
                continue
            leg_in = _resolve_leg(lines_index, bus_de, transhipment)
            leg_out = _resolve_leg(lines_index, transhipment, bus_para)
            if leg_in is not None and leg_out is not None:
                alignment[(bus_de, bus_para)] = [leg_in, leg_out]
    return alignment


def _interc_side(
    decomp_dir: Path,
    cobre_output_dir: Path,
    id_map: DecompIdMap,
    *,
    probabilities: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, list[tuple[int, int]]]:
    """Per-(cobre ``line_id``, 0-based stage) DECOMP interchange net-flow.

    Folds ``dec_oper_interc`` the way every other ``_*_side`` helper folds
    its table -- :func:`_stage_rows`'s patamar-null aggregate, then
    :func:`_scenario_mean` averaged over the source model's nodes (real
    tree-probability-weighted when *probabilities* is given -- see
    :func:`_hydro_side`), here keyed
    by the corridor's composite ``(codigo_submercado_de,
    codigo_submercado_para)`` pair rather than a single entity code -- aligns
    each corridor onto its cobre line leg(s) via
    :func:`_corridor_line_alignment`, and orients the net-flow value to each
    leg's ``sign``. ``estagio - 1`` rebases the 1-based DECOMP stage the same
    way :func:`_map_entities` does for every other level.

    The net-flow value is ``intercambio_origem_MW`` -- the flow measured at
    the corridor's origin, already in its own ``de -> para`` direction.
    ``perdas_MW`` (transmission losses) is read as part of the same fold but
    intentionally NOT subtracted from it: a cobre line has no loss model (its
    ``net_flow_mw`` is a single, lossless figure -- see
    ``decomp/converters/network.py::convert_lines``), so the origin-side reading, not
    the post-loss ``intercambio_destino_MW``, is the one quantity comparable
    to it.

    A corridor whose endpoints the id map cannot resolve, or whose resolved
    bus pair has no entry in :func:`_corridor_line_alignment`'s mapping
    (including every corridor when ``system/lines.json`` is absent), is
    collected into the returned unresolved list as
    ``(codigo_submercado_de, codigo_submercado_para)`` code pairs --
    mirroring :func:`_map_entities`'s ``unmapped`` convention -- and excluded
    from the returned frame. It is reported, never silently dropped.

    The source model reports every physical interface as two corridor rows,
    one per direction (e.g. both ``SE -> S`` and ``S -> SE``); both align
    onto the very same cobre line leg with opposite ``sign``. The
    per-``(entity_id, stage_id)`` rows built above are therefore grouped and
    summed before being returned, collapsing each pair down to one row equal
    to ``flow(de -> para) - flow(para -> de)`` -- exactly the net figure a
    lossless cobre line's own ``net_flow_mw`` represents. The same grouping
    also nets more than one corridor's contribution onto a shared two-leg
    transhipment leg (the star-topology "shared leg" case) instead of
    colliding as duplicate ``(entity_id, stage_id)`` keys.
    """
    empty = pl.DataFrame(
        schema={
            "entity_id": pl.Int64,
            "stage_id": pl.Int64,
            "net_flow_mw": pl.Float64,
        }
    )
    frame = _stage_rows(read_dec_oper_interc(decomp_dir))
    aggregated = _scenario_mean(
        frame,
        "estagio",
        ["intercambio_origem_MW", "perdas_MW"],
        entity_column=["codigo_submercado_de", "codigo_submercado_para"],
        probabilities=probabilities,
    )
    if aggregated.is_empty():
        return empty, []

    alignment = _corridor_line_alignment(cobre_output_dir, id_map)

    unresolved: set[tuple[int, int]] = set()
    rows: list[dict[str, float | int]] = []
    for row in aggregated.iter_rows(named=True):
        de_raw = row["codigo_submercado_de"]
        para_raw = row["codigo_submercado_para"]
        if de_raw is None or para_raw is None:
            # The Itaipu 60 Hz "IV" node -- the transshipment corridor into
            # Ivaiporã -- reports its corridor by ``nome_submercado_*`` only,
            # no ``SB``-declared numeric code, so ``codigo_submercado_de``/
            # ``_para`` comes back null. Report it with the ``-1`` "no SB
            # code" sentinel rather than crashing on the ``int()`` cast below.
            unresolved.add(
                (
                    int(de_raw) if de_raw is not None else -1,
                    int(para_raw) if para_raw is not None else -1,
                )
            )
            continue
        de_code = int(de_raw)
        para_code = int(para_raw)
        try:
            bus_de = id_map.bus_id(de_code)
            bus_para = id_map.bus_id(para_code)
        except KeyError:
            unresolved.add((de_code, para_code))
            continue

        legs = alignment.get((bus_de, bus_para))
        if legs is None:
            unresolved.add((de_code, para_code))
            continue

        stage_id = int(row["estagio"]) - 1
        net_flow = float(row["intercambio_origem_MW"])
        for line_id, sign in legs:
            rows.append(
                {
                    "entity_id": line_id,
                    "stage_id": stage_id,
                    "net_flow_mw": sign * net_flow,
                }
            )

    if not rows:
        return empty, sorted(unresolved)
    result = (
        pl.DataFrame(rows)
        .group_by("entity_id", "stage_id")
        .agg(pl.col("net_flow_mw").sum())
        .cast({"entity_id": pl.Int64, "stage_id": pl.Int64, "net_flow_mw": pl.Float64})
        .sort("entity_id", "stage_id")
    )
    return result, sorted(unresolved)


# --- line ResultComparison rows + Network tab metadata ---
#
# Wires the corridor alignment into ``build_decomp_dataset``: line
# ``ResultComparison`` rows (mirroring ``results._compare_lines``'s emitted
# shape) plus the three ``PercentileData`` fields the shared Network tab
# (``report_builder``, ``charts.line_summary_chart``) consumes.


def _line_entity_names(line_meta: list[dict], id_map: DecompIdMap) -> dict[int, str]:
    """Display name per cobre line id, straight from ``lines.json``'s own
    ``name`` field (built by ``decomp/converters/network.py::convert_lines`` as
    ``f"{pair[0]}-{pair[1]}"``).

    Falls back to a bus-name pair label when an entry carries no usable
    ``name`` -- defensive; every line the converter emits does carry one --
    so a line's ``entity_name`` is never blank.
    """
    names: dict[int, str] = {}
    for entry in line_meta:
        line_id = int(entry["id"])
        name = str(entry.get("name") or "").strip()
        if not name:
            try:
                src_name = id_map.bus_name(int(entry["source_bus_id"]))
                tgt_name = id_map.bus_name(int(entry["target_bus_id"]))
                name = f"{src_name}<->{tgt_name}"
            except KeyError:
                name = f"line_{line_id}"
        names[line_id] = name
    return names


def _line_result_comparisons(
    decomp_dir: Path,
    cobre_output_dir: Path,
    id_map: DecompIdMap | None,
    line_meta: list[dict],
    *,
    probabilities: pl.DataFrame | None = None,
) -> tuple[list[ResultComparison], list[list[int]]]:
    """Join the aligned DECOMP line flow onto Cobre's per-line means.

    Mirrors :func:`_result_comparisons`'s join shape, but keyed by cobre line
    id rather than a source-model entity code: a single cobre line can be the
    shared leg of more than one corridor under the star topology (the
    two-leg transhipment path), so ``newave_code`` -- unlike every other
    level's rows -- carries no single canonical source-model code; it stays
    at the constant ``0`` placeholder, mirroring the SIN row's own
    convention (:func:`_energy_balance_frames`) rather than fabricating one.
    *probabilities* is forwarded to :func:`_interc_side` unchanged (see
    :func:`_hydro_side` for its scenario-weighting contract).

    Returns ``([], [])`` when the deck could not be read (*id_map* is
    ``None``) or the source-model interchange table is unavailable, and
    ``([], unresolved)`` when corridors resolve but the join with Cobre's own
    per-line means produces no row -- matching :func:`_interc_side`'s own
    never-silently-drop convention for unresolved corridors.
    """
    if id_map is None:
        return [], []
    try:
        source_lines, unresolved = _interc_side(
            decomp_dir, cobre_output_dir, id_map, probabilities=probabilities
        )
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No source-model interchange table for line alignment: %s", exc)
        return [], []

    unresolved_lists = [list(pair) for pair in unresolved]
    cobre_lines = cobre_readers.read_cobre_line_means(cobre_output_dir)
    if source_lines.is_empty() or cobre_lines.is_empty():
        return [], unresolved_lists

    names = _line_entity_names(line_meta, id_map)
    joined = (
        source_lines.rename({"net_flow_mw": "source_net_flow_mw"})
        .join(cobre_lines, on=["entity_id", "stage_id"], how="inner")
        .sort("entity_id", "stage_id")
    )

    results: list[ResultComparison] = []
    for row in joined.iter_rows(named=True):
        line_id = int(row["entity_id"])
        nw_value = float(row["source_net_flow_mw"])
        cobre_value = float(row["net_flow_mw"])
        abs_diff, rel_diff = _result_diff(nw_value, cobre_value)
        results.append(
            ResultComparison(
                entity_type="line",
                entity_name=names.get(line_id, f"line_{line_id}"),
                newave_code=0,
                cobre_id=line_id,
                stage=int(row["stage_id"]),
                variable="net_flow_mw",
                newave_value=nw_value,
                cobre_value=cobre_value,
                abs_diff=abs_diff,
                rel_diff=rel_diff,
            )
        )
    return results, unresolved_lists


def _line_bounds_and_meta(cobre_output_dir: Path) -> tuple[pl.DataFrame, list[dict]]:
    """Cobre-side line capacity bounds + metadata for the Network tab.

    Both are pure Cobre-case artifacts, not DECOMP-derived, so this reads
    them exactly the way ``results.compare_results`` does for the source
    model's own Network tab, via the shared ``cobre_readers`` readers:

    - ``line_bounds``: :func:`cobre_readers.read_cobre_line_bounds` verbatim.
      This is already per-stage: ``decomp/converters/network.py::convert_lines`` writes
      one base row (``block_id`` null) per ``(line, stage)`` carrying the
      resolved max-of-blocks ``IA`` capacity for that stage, so a case whose
      capacity genuinely never changes across stages naturally produces
      identical per-stage rows here -- no separate static-capacity broadcast
      needs implementing on top.
    - ``line_meta``: :func:`cobre_readers.read_cobre_lines`'s ``"lines"``
      list verbatim -- each entry's own nested ``capacity.direct_mw``/
      ``capacity.reverse_mw`` is exactly the shape ``line_summary_chart``
      reads.

    The Network tab is optional: a missing ``line_bounds.parquet`` degrades
    to an empty frame via the reader's own typed-empty contract; a missing
    or unparseable ``lines.json`` degrades to an empty ``line_meta`` (with a
    warning) rather than raising, unchanged from before this module routed
    through the shared readers.
    """
    line_bounds = cobre_readers.read_cobre_line_bounds(cobre_output_dir)

    try:
        line_meta = cobre_readers.read_cobre_lines(cobre_output_dir)
    except cobre_readers.CobreReadError as exc:
        _LOG.warning("Failed to read lines.json for network tab: %s", exc)
        line_meta = []

    return line_bounds, line_meta


#: ``dec_oper_sist`` columns the Energy Balance tab's reference-model overlay
#: needs: the two generation families it sums into GHTOT/GTERM, the demand
#: and non-controllable-generation basis for NET_LOAD, and the two system
#: energy series (EARM/ENA). Read once via :func:`_scenario_mean` so scenario
#: averaging matches the rest of the module (real tree-probability-weighted
#: when a ``relato``/``relato2`` probability source is available, else the
#: plain unweighted mean across nodes).
_NW_MARKET_COLUMNS: tuple[str, ...] = (
    "demanda_MW",
    "geracao_hidroeletrica_MW",
    "geracao_termica_MW",
    "geracao_termica_antecipada_MW",
    "geracao_eolica_MW",
    "geracao_pequenas_usinas_MW",
    "deficit_MW",
    "ena_MWmes",
    "earm_final_MWmes",
)


def _empty_market_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "newave_code": pl.Int64,
            "stage": pl.Int64,
            "variable": pl.Utf8,
            "value": pl.Float64,
        }
    )


def _energy_balance_frames(
    decomp_dir: Path,
    bus_codes: dict[int, int],
    *,
    probabilities: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build the Energy Balance tab's DECOMP-side reference frames.

    Returns ``(nw_market, nw_net_load, nw_sin)`` — the long
    ``newave_code``/``stage``/``variable``/``value`` frames
    :func:`~cobre_bridge.comparators.report_builder.build_energy_balance_tab` and
    :func:`~cobre_bridge.comparators.charts.cobre_aggregate_chart` read.
    *probabilities* is forwarded to the single :func:`_scenario_mean` fold
    below (see :func:`_hydro_side` for its scenario-weighting contract).

    Only the tokens the tab actually consumes are emitted — no dead rows:
    ``GHTOT``/``GTERM``/``DEFT`` in ``nw_market`` (mirroring
    ``charts._BALANCE_VARS``), ``NET_LOAD`` in ``nw_net_load``, and
    ``EARMF``/``ENA`` in ``nw_sin``. The tab's fifth token, ``EXCESSO``
    (system energy excess), has no DECOMP source — neither ``dec_oper_sist``
    nor ``relato.balanco_energetico`` carries an excess/curtailment column —
    so it is intentionally omitted rather than fabricated; that panel still
    renders (Cobre-only, no NEWAVE overlay line), matching the tab's existing
    graceful-degradation path for a variable absent on one side.

    ``newave_code`` here is the **Cobre bus id**, not the raw DECOMP
    ``codigo_submercado`` — ``nw_bus_names`` (built by
    ``build_decomp_dataset``) is populated from ``read_cobre_bus_metadata``
    and is therefore keyed by Cobre bus id, and ``build_energy_balance_tab``
    joins ``nw_market``/``nw_net_load`` against it by that same key.
    ``bus_codes`` (``{codigo_submercado: cobre_bus_id}``, already built by
    ``_read_aligned_frames``) performs that translation; ``nw_sin`` needs no
    such translation since it sums across every submarket rather than
    keying per-bus.

    D-STAGE-OFFSET: ``stage`` stays the raw 1-based ``estagio`` (unlike
    :func:`_map_entities`, this does *not* rebase to Cobre's 0-based
    ``stage_id``), so the consuming charts' own stage-alignment (``nw_offset
    = 1``, set on ``PercentileData`` by ``build_decomp_dataset``) lines up.
    """
    frame = _stage_rows(read_dec_oper_sist(decomp_dir))
    if frame.is_empty() or "codigo_submercado" not in frame.columns:
        empty = _empty_market_frame()
        return empty, empty, empty

    aggregated = _scenario_mean(
        frame,
        "estagio",
        list(_NW_MARKET_COLUMNS),
        entity_column="codigo_submercado",
        probabilities=probabilities,
    )
    present = set(aggregated.columns)

    mapped = aggregated.with_columns(
        pl.col("codigo_submercado")
        .map_elements(lambda c: bus_codes.get(int(c)), return_dtype=pl.Int64)
        .alias("newave_code"),
        pl.col("estagio").cast(pl.Int64).alias("stage"),
    ).filter(pl.col("newave_code").is_not_null())

    # --- nw_market: GHTOT (hydro gen) / GTERM (live + anticipated thermal
    # gen) / DEFT (deficit), keyed by the mapped Cobre bus id. GTERM sums the
    # two DECOMP thermal-generation columns because Cobre's own
    # ``thermal_gen_mw`` (the counterpart this overlays) already includes
    # anticipated GNL generation — the same live+anticipated fold the cost
    # breakdown applies (see charts._COST_MAP's "Thermal Generation" entry).
    market_exprs: list[tuple[str, pl.Expr]] = []
    if "geracao_hidroeletrica_MW" in present:
        market_exprs.append(("GHTOT", pl.col("geracao_hidroeletrica_MW")))
    if "geracao_termica_MW" in present and "geracao_termica_antecipada_MW" in present:
        market_exprs.append(
            (
                "GTERM",
                pl.col("geracao_termica_MW") + pl.col("geracao_termica_antecipada_MW"),
            )
        )
    if "deficit_MW" in present:
        market_exprs.append(("DEFT", pl.col("deficit_MW")))

    if market_exprs and not mapped.is_empty():
        wide = mapped.select(
            "newave_code",
            "stage",
            *[expr.alias(name) for name, expr in market_exprs],
        )
        nw_market = wide.unpivot(
            on=[name for name, _ in market_exprs],
            index=["newave_code", "stage"],
            variable_name="variable",
            value_name="value",
        ).with_columns(pl.col("value").cast(pl.Float64))
    else:
        nw_market = _empty_market_frame()

    # --- nw_net_load: demand minus the non-controllable sources (wind +
    # small plants) — the tab's NET_LOAD basis.
    non_controllable = {"demanda_MW", "geracao_eolica_MW", "geracao_pequenas_usinas_MW"}
    if non_controllable <= present and not mapped.is_empty():
        nw_net_load = mapped.select(
            "newave_code",
            "stage",
            pl.lit("NET_LOAD").alias("variable"),
            (
                pl.col("demanda_MW")
                - pl.col("geracao_eolica_MW")
                - pl.col("geracao_pequenas_usinas_MW")
            )
            .cast(pl.Float64)
            .alias("value"),
        )
    else:
        nw_net_load = _empty_market_frame()

    # nw_sin: EARMF / ENA summed across every submarket the deck declares. The
    # transhipment bus never appears among ``codigo_submercado`` values
    # (D-UNITS) — referenced only by name in ``IA`` records — so the sum
    # already excludes it, no filter needed (``TestBusSideExcludesTranshipment``
    # pins this down). ``newave_code`` is the SIN placeholder ``0``, matching
    # ``read_medias_sin``'s newave-side convention.
    system_energy = {"earm_final_MWmes", "ena_MWmes"}
    if system_energy <= present:
        sin_agg = aggregated.group_by("estagio").agg(
            pl.col("earm_final_MWmes").sum().alias("EARMF"),
            pl.col("ena_MWmes").sum().alias("ENA"),
        )
        nw_sin = (
            sin_agg.unpivot(
                on=["EARMF", "ENA"], index=["estagio"], variable_name="variable"
            )
            .with_columns(
                pl.lit(0).cast(pl.Int64).alias("newave_code"),
                pl.col("estagio").cast(pl.Int64).alias("stage"),
                pl.col("value").cast(pl.Float64),
            )
            .select("newave_code", "stage", "variable", "value")
            .sort("stage", "variable")
        )
    else:
        nw_sin = _empty_market_frame()

    return nw_market, nw_net_load, nw_sin


# --- Overview cost metadata (nw_costs / nw_sin cost rows) ---
#
# The single place the native k$ -> R$ reconciliation happens, for both target
# unit conventions: the NPV breakdown dict is R$
# (``reconcile_kdollars_to_reais``, x1e3); the per-stage ``nw_sin`` cost rows
# are 10^6 R$ (an additional /1e6 on top of that factor).

#: ``relatorio_operacao_custos`` numeric cost columns aggregated here
#: -- every cost column the table carries except the per-submarket
#: ``cmo_<sbm>`` marginal-*price* columns (not a cost) and the
#: estagio/cenario/probabilidade bookkeeping columns.
_RELATO_COST_COLUMNS: tuple[str, ...] = (
    "custo_presente",
    "custo_futuro",
    "geracao_termica",
    "violacao_desvio",
    "penalidade_vertimento_reservatorio",
    "penalidade_vertimento_fio",
    "violacao_turbinamento_reservatorio",
    "violacao_turbinamento_fio",
    "penalidade_intercambio",
)

#: ``violacao_desvio`` has no ``charts._COST_MAP`` analogue -- surfaced under
#: its own label (matching the Portuguese-abbreviation style every other
#: ``_COST_MAP`` key uses) rather than being silently dropped;
#: ``charts._resolve_cost_categories`` appends it as an unmapped residual
#: category.
_DEVIATION_VIOLATION_LABEL = "VIOLACAO DESVIO"

#: ``(display_label, source_columns...)`` for the ``nw_costs`` NPV dict --
#: mirrors ``charts._COST_MAP``'s the source model label strings. VERTIMENTO
#: sums two source columns (reservoir + run-of-river spillage penalty); every
#: other category is a single column.
_NW_COST_LABELS: tuple[tuple[str, ...], ...] = (
    ("GERACAO TERMICA", "geracao_termica"),
    ("INTERCAMBIO", "penalidade_intercambio"),
    (
        "VERTIMENTO",
        "penalidade_vertimento_reservatorio",
        "penalidade_vertimento_fio",
    ),
    ("VIOL. TURB. MINIMO", "violacao_turbinamento_reservatorio"),
    ("VIOL. TURB. MAXIMO", "violacao_turbinamento_fio"),
    (_DEVIATION_VIOLATION_LABEL, "violacao_desvio"),
)

#: ``nw_sin`` variable token -> source column, for the per-stage cost rows
#: ``charts._extract_stage_cost_series`` reads. Matches MEDIAS-SIN's the
#: source model tokens on the newave side of ``compare newave``.
_NW_SIN_COST_TOKENS: tuple[tuple[str, str], ...] = (
    ("COPER", "custo_presente"),
    ("CUSTO_FUTURO", "custo_futuro"),
    ("CTERM", "geracao_termica"),
)


def _probability_weighted_stage_cost(
    frame: pl.DataFrame, columns: list[str]
) -> pl.DataFrame:
    """Per-stage **expected** cost, weighting scenarios by the DECOMP tree
    ``probabilidade``.

    One row per ``estagio``; each cost column is
    ``sum(probabilidade * col) / sum(probabilidade)`` across that stage's
    scenarios. A deterministic stage (single row, ``probabilidade`` = 1.0)
    reduces to the value itself, so ``relato``'s weekly stages are unchanged
    and only a genuine scenario fan (``relato2``'s monthly stage) is averaged.
    This is the same probability-weighted expectation
    :func:`_scenario_mean`'s own ``probabilities`` parameter now applies to
    every physical variable, but reads ``probabilidade`` directly off *this*
    already-unioned ``relato``/``relato2`` cost frame instead of joining in a
    separate :func:`_scenario_probabilities` lookup -- the cost path predates
    that lookup and every one of its cost columns is already in *frame*.
    Falls back to an unweighted mean for any stage whose ``probabilidade`` is
    absent or sums to ~0 (a malformed report), never dividing by zero.
    """
    present = [c for c in columns if c in frame.columns]
    if frame.is_empty() or "estagio" not in frame.columns:
        return frame
    if "probabilidade" not in frame.columns:
        return (
            frame.group_by("estagio")
            .agg([pl.col(c).mean().alias(c) for c in present])
            .sort("estagio")
        )
    return _weighted_group_mean(frame, ["estagio"], present)


def _cost_frames(decomp_dir: Path) -> tuple[dict[str, float], pl.DataFrame]:
    """DECOMP-side Overview cost metadata: the NPV dict + ``nw_sin`` rows.

    Reads ``relatorio_operacao_custos`` from **both** reports and unions them
    so every stage is covered: ``relato`` (:func:`read_relato_costs`) carries
    the deterministic study-period weeks, and ``relato2``
    (:func:`read_relato2_costs`, optional) carries the per-realization rows of
    the scenario-fan (monthly) stage(s) ``relato`` omits. Where ``relato2``
    covers a stage it is authoritative (that stage is dropped from ``relato``
    before the union, so a fan stage is never counted twice). Aggregated in
    two passes, both built on the same per-stage expected cost:

    - **Per-stage expected cost** -- :func:`_probability_weighted_stage_cost`,
      weighting each stage's scenarios by the real tree ``probabilidade``.
      This is a deliberate departure from the module's unweighted-scenario
      convention **for cost only**: an *expected* operating cost on the fan
      stage must use the true opening probabilities (which are not
      equiprobable), and both reports carry ``probabilidade`` per row. A
      deterministic week (single row, prob 1.0) is unaffected. No stage
      discount is applied: unlike cobre's ``discount_factor`` column
      (:func:`cobre_readers.read_cobre_cost_breakdown`),
      ``relatorio_operacao_custos`` exposes no per-row discount factor, so the
      per-stage values are DECOMP's own raw nominal $ costs at that node.
    - **``nw_costs``** (R$): the per-stage expected costs summed across every
      stage, then :func:`reconcile_kdollars_to_reais` (x1e3). Categories below
      0.01 R$ are excluded, mirroring ``newave_readers.read_pmo_cost_
      breakdown`` and :func:`cobre_readers.read_cobre_cost_breakdown`'s own
      floor.
    - **``nw_sin`` rows** (10^6 R$): the same per-stage expected costs,
      unpivoted to ``COPER``/``CUSTO_FUTURO``/``CTERM`` rows with the 1-based
      ``stage`` column kept as-is (the consuming charts subtract ``nw_offset``).

    Raises whatever :func:`read_relato_costs` raises on a missing/empty
    ``relato`` parse (the mandatory report); a missing ``relato2`` is not an
    error -- :func:`read_relato2_costs` returns empty and only the fan stage(s)
    are absent.
    """
    raw = read_relato_costs(decomp_dir)
    monthly = read_relato2_costs(decomp_dir)
    combined = _union_relato_reports(raw, monthly)
    per_stage = _probability_weighted_stage_cost(combined, list(_RELATO_COST_COLUMNS))
    present = [c for c in _RELATO_COST_COLUMNS if c in per_stage.columns]

    totals: dict[str, float] = (
        per_stage.select([pl.col(c).sum().alias(c) for c in present]).row(0, named=True)
        if present
        else {}
    )

    nw_costs: dict[str, float] = {}
    for label, *columns in _NW_COST_LABELS:
        value = reconcile_kdollars_to_reais(sum(totals.get(c, 0.0) for c in columns))
        if abs(value) > 0.01:
            nw_costs[label] = value

    sin_tokens = [(var, col) for var, col in _NW_SIN_COST_TOKENS if col in present]
    if not sin_tokens or per_stage.is_empty():
        nw_sin_rows = _empty_market_frame()
    else:
        # A scalar factor derived from the single conversion site rather
        # than a re-derived literal: k$ -> R$ (x1e3) then R$ -> 10^6 R$
        # (/1e6), collapsing to x1e-3 -- but expressed via
        # reconcile_kdollars_to_reais so the x1e3 itself is never repeated.
        k_to_r6 = reconcile_kdollars_to_reais(1.0) / 1e6
        nw_sin_rows = (
            per_stage.select(
                "estagio",
                *[pl.col(col).alias(var) for var, col in sin_tokens],
            )
            .unpivot(
                on=[var for var, _ in sin_tokens],
                index=["estagio"],
                variable_name="variable",
                value_name="value",
            )
            .with_columns(
                pl.lit(0).cast(pl.Int64).alias("newave_code"),
                pl.col("estagio").cast(pl.Int64).alias("stage"),
                (pl.col("value") * k_to_r6).cast(pl.Float64).alias("value"),
            )
            .select("newave_code", "stage", "variable", "value")
            .sort("stage", "variable")
        )

    return nw_costs, nw_sin_rows


def _union_cost_rows(nw_sin: pl.DataFrame, cost_rows: pl.DataFrame) -> pl.DataFrame:
    """Union :func:`_cost_frames`'s cost rows onto the existing ``nw_sin``.

    ``nw_sin`` (from :func:`_energy_balance_frames`) already
    carries the EARM/ENA rows -- this must add the cost rows alongside them,
    never overwrite. A columnless frame on either side (the
    ``_AlignedDecompFrames``/``PercentileData`` dataclass default
    ``pl.DataFrame()``) cannot be vertically concatenated against a typed
    frame, so it is treated as "nothing to union" instead of raising a
    schema-mismatch error.
    """
    if not nw_sin.columns:
        return cost_rows
    if not cost_rows.columns:
        return nw_sin
    return pl.concat([nw_sin, cost_rows], how="vertical", rechunk=True)


# --- Performance tab timing metadata ---
#
# DECOMP is nested Benders over an explicit tree, not sampled SDDP -- no
# forward/backward pass structure, so ``nw_tim_iterations`` never carries
# ``forward_seconds``/``backward_seconds`` and
# ``performance_fwd_bwd_split_chart`` renders cobre-only (its ``has_nw`` guard
# checks for those columns, not just frame emptiness).

#: ``decomp.tim``'s ``Etapa`` phase name -> the two keys
#: ``charts.performance_metric_cards`` reads, so the shared card renderer needs
#: no source-specific branch. ``"Convergencia"`` is DECOMP's nested-Benders
#: loop -- its analog of ``newave.tim``'s ``"Calculo da Politica"`` (policy
#: training).
_DECOMP_TIM_PHASE_MAP: dict[str, str] = {
    "Tempo Total": "Tempo Total",
    "Convergencia": "Calculo da Politica",
}


def _decomp_tim_stages(decomp_dir: Path) -> dict[str, float]:
    """``nw_tim_stages``: DECOMP phase totals mapped onto the shared card keys.

    ``read_decomp_tim``'s ``Tempo`` column is a ``datetime.timedelta`` (one
    row per ``iter_rows``); :meth:`~datetime.timedelta.total_seconds` gives
    the float seconds :func:`~cobre_bridge.comparators.charts.
    performance_metric_cards` expects. A missing/empty ``decomp.tim``
    (``read_decomp_tim`` raising ``FileNotFoundError``/``ValueError``)
    degrades to ``{}`` instead of failing :func:`build_decomp_dataset` --
    the metric cards then render with their ``0.0``/"—" fallback. When
    ``decomp.tim`` *is* read but one of the two mapped ``Etapa`` rows is
    absent from it, that key is still returned, explicitly set to ``0.0``
    (see :data:`_DECOMP_TIM_PHASE_MAP`).
    """
    try:
        table = read_decomp_tim(decomp_dir)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No source-model timing table (decomp.tim): %s", exc)
        return {}

    seconds_by_etapa: dict[str, float] = {}
    for row in table.iter_rows(named=True):
        tempo = row["Tempo"]
        seconds_by_etapa[str(row["Etapa"]).strip()] = (
            tempo.total_seconds() if tempo is not None else 0.0
        )
    return {
        key: seconds_by_etapa.get(etapa, 0.0)
        for etapa, key in _DECOMP_TIM_PHASE_MAP.items()
    }


#: Canonical ``nw_tim_iterations`` schema -- deliberately only ``iteration``/
#: ``total_seconds`` (Caveat #2): unlike the source model's own
#: ``newave.tim``, DECOMP has no forward/backward pass to split, so this
#: frame must never carry ``forward_seconds``/``backward_seconds`` --
#: fabricating either would misrepresent a source with no such structure.
_DECOMP_TIM_ITERATIONS_SCHEMA: dict[str, type[pl.DataType]] = {
    "iteration": pl.Int64,
    "total_seconds": pl.Float64,
}


def _decomp_tim_iterations(decomp_dir: Path) -> pl.DataFrame:
    """``nw_tim_iterations``: per-iteration wall-clock, DECOMP-shaped.

    ``relato.convergencia``'s ``iteracao``/``tempo`` rename straight onto
    ``iteration``/``total_seconds`` -- no forward/backward split is derived
    or fabricated (Caveat #2). Degrades to the empty canonical frame when
    ``read_relato_convergence`` raises ``FileNotFoundError``/``ValueError``,
    matching :func:`_decomp_convergence_frame`'s own degrade-to-empty
    contract for the same reader.
    """
    empty = pl.DataFrame(schema=_DECOMP_TIM_ITERATIONS_SCHEMA)
    try:
        source = read_relato_convergence(decomp_dir)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No source-model convergence table for timing: %s", exc)
        return empty

    columns = {c.lower(): c for c in source.columns}
    if "iteracao" not in columns or "tempo" not in columns:
        return empty
    return source.select(
        pl.col(columns["iteracao"]).cast(pl.Int64).alias("iteration"),
        pl.col(columns["tempo"]).cast(pl.Float64).alias("total_seconds"),
    )


def _decomp_max_stage(aligned: _AlignedDecompFrames) -> int | None:
    """Max DECOMP stage (0-based, Cobre convention) seen in this run.

    Neither of :func:`_decomp_tim_stages`'/:func:`_decomp_tim_iterations`'s
    two sources (``decomp.tim``, ``relato.convergencia``) carries a stage
    axis, so ``nw_max_stage`` is derived from the already-aligned per-level
    frames (:func:`_hydro_side`/:func:`_thermal_side`/:func:`_bus_side`, via
    :func:`_map_entities`) instead of an extra read -- their ``stage_id`` is
    already 0-based, matching every other Cobre-convention stage-axis field
    on :class:`~cobre_bridge.comparators.model.PercentileData`. ``None``
    when every level is empty, mirroring
    :func:`~cobre_bridge.comparators.newave.results.compare_results`'s own
    ``nw_max_stage_1based is None`` case.
    """
    candidates: list[int] = []
    for frame in (aligned.source_hydro, aligned.source_thermal, aligned.source_bus):
        if frame.is_empty() or "stage_id" not in frame.columns:
            continue
        values = frame["stage_id"].drop_nulls()
        if not values.is_empty():
            candidates.append(int(values.max()))  # type: ignore[arg-type]
    return max(candidates) if candidates else None


def _map_entities(
    frame: pl.DataFrame, code_column: str, mapping: dict[int, int]
) -> tuple[pl.DataFrame, list[int]]:
    """Translate source codes to Cobre ids; report the codes with no mapping.

    Keeps the original code alongside the mapped ``entity_id`` (renamed to
    the generic ``newave_code`` — D-SOURCE-TOKEN: the field name means "the
    reference model's code", not NEWAVE-specific) so callers that need the
    reference code back (:func:`_result_comparisons`) don't have to invert
    ``mapping`` themselves.
    """
    codes = {int(c) for c in frame[code_column].unique()}
    unmapped = sorted(code for code in codes if code not in mapping)
    mapped = (
        frame.with_columns(
            pl.col(code_column)
            .map_elements(lambda c: mapping.get(int(c)), return_dtype=pl.Int64)
            .alias("entity_id"),
            pl.col(code_column).cast(pl.Int64).alias("newave_code"),
            # Stage ids are 0-based in Cobre and 1-based in the source deck.
            (pl.col("estagio").cast(pl.Int64) - 1).alias("stage_id"),
        )
        .filter(pl.col("entity_id").is_not_null())
        .drop(code_column, "estagio")
    )
    return mapped, unmapped


def _cobre_hydro(cobre_output_dir: Path) -> tuple[pl.DataFrame, dict[int, str]]:
    """Cobre hydro means with absolute storage converted to useful volume."""
    means = cobre_readers.read_cobre_hydro_means(cobre_output_dir)
    metadata = cobre_readers.read_cobre_hydro_metadata(cobre_output_dir)
    names = {i: str(m.get("name", "")) for i, m in metadata.items()}

    if means.is_empty():
        return means, names

    floors = pl.DataFrame(
        {
            "entity_id": list(metadata.keys()),
            "min_storage_hm3": [
                float(m.get("min_storage_hm3") or 0.0) for m in metadata.values()
            ],
        },
        schema={"entity_id": pl.Int64, "min_storage_hm3": pl.Float64},
    )
    if "storage_final_hm3" not in means.columns or floors.is_empty():
        return means, names
    return (
        means.join(floors, on="entity_id", how="left").with_columns(
            (
                pl.col("storage_final_hm3") - pl.col("min_storage_hm3").fill_null(0.0)
            ).alias("useful_storage_hm3")
        ),
        names,
    )


@dataclass(frozen=True)
class _AlignedDecompFrames:
    """One run's DECOMP-side + Cobre-side frames, aligned to Cobre ids/stages.

    The shared read/align result :func:`build_decomp_dataset` builds on, via
    :func:`_read_aligned_frames` — per-level scenario averaging and the
    matching Cobre means, off the shared :class:`~cobre_bridge.decomp.case.DecompCase`
    parse, run exactly once.
    """

    source_hydro: pl.DataFrame
    source_thermal: pl.DataFrame
    source_bus: pl.DataFrame
    cobre_hydro: pl.DataFrame
    cobre_thermal: pl.DataFrame
    cobre_bus: pl.DataFrame
    hydro_names: dict[int, str]
    thermal_names: dict[int, str]
    bus_names: dict[int, str]
    unmapped: dict[str, list[int]]
    # --- Energy Balance tab reference frames ---
    # Long newave_code/stage/variable/value frames built by
    # :func:`_energy_balance_frames`; defaulted so fixtures that construct
    # this dataclass without them keep working unchanged.
    nw_market: pl.DataFrame = field(default_factory=pl.DataFrame)
    nw_net_load: pl.DataFrame = field(default_factory=pl.DataFrame)
    nw_sin: pl.DataFrame = field(default_factory=pl.DataFrame)


def _read_aligned_frames(
    case: DecompCase,
    cobre_output_dir: Path,
    *,
    probabilities: pl.DataFrame | None = None,
) -> _AlignedDecompFrames:
    """Read and align both sides of one DECOMP-vs-Cobre comparison run.

    *case* is the shared, once-parsed deck
    (:class:`~cobre_bridge.decomp.case.DecompCase`, built once by
    :func:`build_decomp_dataset`); its directory must also contain the
    ``dec_oper_*.csv`` result tables read directly below.
    ``cobre_output_dir`` is Cobre's output directory, whose case directory
    supplies the entity registries. Entities the id map cannot resolve are
    reported via ``unmapped`` rather than dropped in silence. *probabilities*
    -- a :func:`_scenario_probabilities` lookup, built once by
    :func:`build_decomp_dataset` -- is forwarded unchanged to every per-level
    fold (:func:`_hydro_side`/:func:`_thermal_side`/:func:`_bus_side`/
    :func:`_energy_balance_frames`); ``None`` keeps their unweighted default.
    """
    decomp_dir = case.files.dadger.parent
    id_map = case.id_map

    hydro_codes = {code: id_map.hydro_id(code) for code in id_map.hydro_codes}
    thermal_codes = {code: id_map.thermal_id(code) for code in id_map.thermal_codes}
    bus_codes = {code: id_map.bus_id(code) for code in id_map.bus_codes}

    source_hydro, unmapped_hydro = _hydro_side(
        decomp_dir, hydro_codes, probabilities=probabilities
    )
    source_thermal, unmapped_thermal = _thermal_side(
        decomp_dir, thermal_codes, probabilities=probabilities
    )
    source_bus, unmapped_bus = _bus_side(
        decomp_dir, bus_codes, probabilities=probabilities
    )
    nw_market, nw_net_load, nw_sin = _energy_balance_frames(
        decomp_dir, bus_codes, probabilities=probabilities
    )

    cobre_hydro, hydro_names = _cobre_hydro(cobre_output_dir)
    cobre_thermal = cobre_readers.read_cobre_thermal_means(cobre_output_dir)
    cobre_bus = cobre_readers.read_cobre_bus_means(cobre_output_dir)

    thermal_names = {
        i: str(m.get("name", ""))
        for i, m in cobre_readers.read_cobre_thermal_metadata(cobre_output_dir).items()
    }
    bus_names = {
        i: str(m.get("name", ""))
        for i, m in cobre_readers.read_cobre_bus_metadata(cobre_output_dir).items()
    }

    return _AlignedDecompFrames(
        source_hydro=source_hydro,
        source_thermal=source_thermal,
        source_bus=source_bus,
        cobre_hydro=cobre_hydro,
        cobre_thermal=cobre_thermal,
        cobre_bus=cobre_bus,
        hydro_names=hydro_names,
        thermal_names=thermal_names,
        bus_names=bus_names,
        unmapped={
            "hydro": unmapped_hydro,
            "thermal": unmapped_thermal,
            "bus": unmapped_bus,
        },
        nw_market=nw_market,
        nw_net_load=nw_net_load,
        nw_sin=nw_sin,
    )


# --- Hydro Operation + Plant Details tab metadata ---
#
# Fills the four remaining hydro ``PercentileData`` fields the shared Hydro
# Operation tab (``hydro_per_bus_chart``/``hydro_aggregate_chart``/
# ``hydro_slack_per_bus_chart``/``hydro_slack_aggregate_chart``) and the
# Hydro Plant Details tab (``build_hydro_detail_tab``) both read.


def _merge_hydro_bus_ids(
    meta: dict[int, dict], labels: dict[int, frozenset[int]]
) -> dict[int, dict]:
    """Copy each hydro's metadata dict and inject its cobre bus ids.

    ``cobre_readers.read_cobre_hydro_metadata`` carries plant physics only
    -- no bus information (see that reader's docstring); the
    plant -> bus *label* is re-sourced from
    ``cobre_readers.read_cobre_hydro_bus_labels``'s
    ``hydro_bus_generation``-partition-derived map, exactly the way
    ``results.compare_results`` merges the two for the source model's own
    Hydro Operation/Details tabs.

    ``hydro_per_bus_chart``/``hydro_slack_per_bus_chart`` (via
    ``analyze._bus_name_lookups``) key off ``meta[hydro_id]["bus_ids"]``
    and KeyError without it -- every
    returned entry carries the key, an empty list when the plant has no
    label in the partition, rather than leaving it absent. Returns a new
    dict of new per-plant dicts; neither *meta* nor *labels* is mutated.
    """
    return {
        # Sorted list, not the reader's frozenset: bus_ids lands in the
        # JSON-serialized metadata side-table, which rejects a frozenset.
        hydro_id: {**entry, "bus_ids": sorted(labels.get(hydro_id, []))}
        for hydro_id, entry in meta.items()
    }


# --- Productivity tab's "Fitted production functions (FPHA)"
# section (`fpha_metrics`/`fpha_surface`/`fpha_spill`) ---
#
# [ASSUMPTION] (Confidence Low): the full 3D FPHA overlay the newave side
# renders (`analyze.build_fpha_comparison`, option (a)) needs the source side's
# fitted PLANE coefficients on a shared ``(V, Q)`` grid. None of the three
# declared readers carries them (`read_dec_estatfpha` is a deck-wide summary,
# `read_dec_desvfpha` the realized ``(V, Q, S)`` operating points,
# `read_eco_fpha` only the fitting-grid bounds/counts); the coefficients live
# in a fourth file (``avl_cortesfpha.rvN`` / idecomp's ``AvlCortesFpha``) that
# is not a declared reader. So (a) is infeasible and this ships fallback (b):
# `fpha_metrics` only, `fpha_surface`/`fpha_spill` always `None` (never
# fabricated; the ``if not fpha_metrics.is_empty()`` gate in `report_builder`
# renders the empty-surface placeholder, so no exception follows).
#
# Within (b), `_fpha_metrics` evaluates Cobre's own fitted envelope
# (`read_cobre_fpha_planes`) AT the source model's realized operating points
# (`read_dec_desvfpha`) and compares it to the source's own realized
# ``geracao_hidraulica_fpha`` -- a genuine cross-solver comparison sampled at
# the trajectory, not a dense grid (so it never claims (a)'s full-surface
# fidelity). ``n_planes_newave`` stays null (no declared reader counts source
# planes).


def _log_decomp_fpha_deck_summary(decomp_dir: Path) -> None:
    """Log the source model's deck-wide FPHA deviation summary.

    `read_dec_estatfpha`'s table carries no per-hydro or per-stage key (a
    single deck-wide variavel/valor summary), so it cannot join into the
    per-(hydro, stage) `fpha_metrics` frame `_fpha_metrics` builds -- it is
    surfaced as log context only. Never raises: a missing/empty parse is
    logged and dropped, exactly like every other optional FPHA source here.
    """
    try:
        summary = read_dec_estatfpha(decomp_dir)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No source-model FPHA deck-wide summary (dec_estatfpha): %s", exc)
        return
    _LOG.info(
        "Source-model deck-wide FPHA deviation summary: %s",
        dict(
            zip(
                summary["variavel"].to_list(),
                summary["valor"].to_list(),
                strict=True,
            )
        ),
    )


def _cobre_fpha_plane_counts(cb_planes: pl.DataFrame) -> pl.DataFrame:
    """One row per (cobre_id, stage) with Cobre's own fitted plane count."""
    return (
        cb_planes.rename({"hydro_id": "cobre_id", "stage_id": "stage"})
        .cast({"cobre_id": pl.Int64, "stage": pl.Int64})
        .group_by(["cobre_id", "stage"])
        .agg(pl.len().cast(pl.Int64).alias("n_planes_cobre"))
    )


def _decomp_fpha_grid_nv(
    decomp_dir: Path, code_to_cobre: dict[int, int]
) -> pl.DataFrame:
    """One row per (cobre_id, stage) with the source model's own volume
    fitting-grid node count (``numero_pontos_volume_armazenado``), from
    `read_eco_fpha`.

    Degrades to an empty frame -- never raises -- when the deck ships no
    ``eco_fpha`` table (confirmed absent on the reduced deck): ``n_v`` then
    stays null on every metrics row, honest about the value being genuinely
    unavailable rather than fabricating one.
    """
    empty = pl.DataFrame(
        schema={"cobre_id": pl.Int64, "stage": pl.Int64, "n_v": pl.Int64}
    )
    try:
        grid = read_eco_fpha(decomp_dir)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No source-model FPHA grid table (eco_fpha): %s", exc)
        return empty

    mapped = grid.with_columns(
        pl.col("codigo_usina")
        .map_elements(lambda c: code_to_cobre.get(int(c)), return_dtype=pl.Int64)
        .alias("cobre_id"),
        (pl.col("estagio").cast(pl.Int64) - 1).alias("stage"),
    ).filter(pl.col("cobre_id").is_not_null())
    if mapped.is_empty():
        return empty
    return mapped.select(
        "cobre_id",
        "stage",
        pl.col("numero_pontos_volume_armazenado").cast(pl.Int64).alias("n_v"),
    )


def _fpha_metrics(
    decomp_dir: Path,
    cobre_output_dir: Path,
    id_map: DecompIdMap | None,
    hydro_names: dict[int, str],
) -> pl.DataFrame | None:
    """Per-(hydro, stage) FPHA fit-fidelity metrics -- fallback (b), see the
    [ASSUMPTION] comment above this section.

    ``None`` whenever either side has no fitted planes to compare at all:
    Cobre fitted none (`cobre_readers.read_cobre_fpha_planes` returns
    ``None``), the deck has no id map to resolve hydro codes onto Cobre ids
    (`id_map` is ``None``), the source model's own deviation table is
    absent/empty, or no realized point resolves onto a Cobre hydro with
    fitted planes. `build_decomp_dataset` passes ``None`` straight through
    to `~cobre_bridge.comparators.model.PercentileData.fpha_metrics`; the
    report's FPHA section gate (`report_builder`) reads it as the empty
    `RenderInputs.fpha_metrics` default, so the section is omitted, never a
    crash.
    """
    cb_planes = cobre_readers.read_cobre_fpha_planes(cobre_output_dir)
    if cb_planes is None or id_map is None:
        return None

    try:
        deviations = read_dec_desvfpha(decomp_dir)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No source-model FPHA deviation table: %s", exc)
        return None

    code_to_cobre = {code: id_map.hydro_id(code) for code in id_map.hydro_codes}
    points = deviations.with_columns(
        pl.col("codigo_usina")
        .map_elements(lambda c: code_to_cobre.get(int(c)), return_dtype=pl.Int64)
        .alias("cobre_id"),
        (pl.col("estagio").cast(pl.Int64) - 1).alias("stage"),
    ).filter(pl.col("cobre_id").is_not_null())
    if points.is_empty():
        return None
    points = (
        points.select(
            "cobre_id",
            "stage",
            pl.col("volume_total_hm3").cast(pl.Float64).alias("v_hm3"),
            pl.col("vazao_turbinada_m3s").cast(pl.Float64).alias("q_m3s"),
            pl.col("vazao_vertida_m3s").cast(pl.Float64).alias("s_m3s"),
            pl.col("geracao_hidraulica_fpha").cast(pl.Float64).alias("decomp_gh_mw"),
        )
        .with_row_index("_point_id")
        .with_columns(pl.col("_point_id").cast(pl.Int64))
    )

    # Cobre's own kappa/gamma_v already multiply absolute volume, and the
    # source model's own volume_total_hm3 is likewise absolute (not useful)
    # volume -- fpha.point_cloud's default volume_offset=0.0 applies here.
    envelope = fpha.point_cloud(cb_planes, points)
    if envelope.is_empty():
        return None

    _log_decomp_fpha_deck_summary(decomp_dir)

    joined = points.join(envelope, on="_point_id", how="inner").with_columns(
        (pl.col("cobre_gh_mw") - pl.col("decomp_gh_mw")).alias("diff")
    )

    metrics = (
        joined.group_by(["cobre_id", "stage"])
        .agg(
            pl.col("decomp_gh_mw").max().abs().alias("_denom"),
            pl.col("diff").abs().mean().alias("_mean_abs_diff"),
            pl.col("diff").mean().alias("_mean_diff"),
            pl.col("diff").abs().max().alias("max_abs_dev"),
            pl.col("cobre_gh_mw").max().alias("_cobre_max"),
            pl.col("decomp_gh_mw").max().alias("_decomp_max"),
        )
        .with_columns((pl.col("_denom") > 1e-9).alias("_scaled"))
        .with_columns(
            pl.when(pl.col("_scaled"))
            .then(pl.col("_mean_abs_diff") / pl.col("_denom"))
            .otherwise(None)
            .alias("nmae"),
            pl.when(pl.col("_scaled"))
            .then(pl.col("_mean_diff") / pl.col("_denom"))
            .otherwise(None)
            .alias("bias"),
            pl.when(pl.col("_scaled"))
            .then(pl.col("_cobre_max") / pl.col("_decomp_max"))
            .otherwise(None)
            .alias("gh_max_ratio"),
        )
        .join(_cobre_fpha_plane_counts(cb_planes), on=["cobre_id", "stage"], how="left")
        .join(
            _decomp_fpha_grid_nv(decomp_dir, code_to_cobre),
            on=["cobre_id", "stage"],
            how="left",
        )
        .with_columns(
            pl.col("cobre_id")
            .map_elements(
                lambda c: hydro_names.get(int(c), f"hydro_{c}"), return_dtype=pl.Utf8
            )
            .alias("plant_name"),
            pl.lit(None).cast(pl.Int64).alias("n_planes_newave"),
        )
        .select(list(fpha.FPHA_METRICS_SCHEMA))
        .sort(["cobre_id", "stage"])
    )
    return metrics if not metrics.is_empty() else None


# --- REE energy rollup via membership ---
#
# DECOMP reports energy at REE (reservoir-equivalent-energy) granularity
# (`read_dec_oper_ree`); Cobre has no REE entity, so the counterpart is built
# by rolling each REE's member plants' energy up through the
# `relato.uhes_rees_submercados` membership table (`read_relato_membership`).


def _ree_membership_map(
    membership: pl.DataFrame, hydro_codes: dict[int, int]
) -> tuple[dict[int, int], list[int]]:
    """``{cobre_hydro_id: codigo_ree}`` via membership, restricted to the
    operated hydro codes this module already resolves
    (``{codigo_usina: cobre_id}``, from `DecompIdMap.hydro_id`).

    A hydro code with no row in *membership* cannot be attributed to any REE
    -- it is excluded from every REE sum and returned (sorted) as the second
    element instead of being silently dropped.
    """
    code_to_ree: dict[int, int] = {}
    if not membership.is_empty():
        for row in membership.iter_rows(named=True):
            code_to_ree[int(row["codigo_usina"])] = int(row["codigo_ree"])

    ree_by_cobre_id: dict[int, int] = {}
    unmapped: list[int] = []
    for code, cobre_id in hydro_codes.items():
        ree_code = code_to_ree.get(code)
        if ree_code is None:
            unmapped.append(code)
            continue
        ree_by_cobre_id[cobre_id] = ree_code
    return ree_by_cobre_id, sorted(unmapped)


_EMPTY_COBRE_REE_SUMS_SCHEMA: dict[str, type[pl.DataType]] = {
    "entity_id": pl.Int64,
    "stage_id": pl.Int64,
    "ena_mw": pl.Float64,
    "earm_mwh": pl.Float64,
}

#: Columns :func:`_cobre_ree_sums` reads off *cobre_hydro* besides
#: ``entity_id``/``stage_id``. A frame missing either -- e.g. a trimmed
#: test fixture that only carries the base hydro-comparison columns --
#: degrades to an empty sum instead of a Polars ``ColumnNotFoundError``.
_REE_COBRE_ENERGY_COLUMNS: tuple[str, ...] = (
    "incremental_inflow_energy_mw",
    "stored_energy_final_mwh",
)


def _cobre_ree_sums(
    cobre_hydro: pl.DataFrame, ree_by_cobre_id: dict[int, int]
) -> pl.DataFrame:
    """Membership-weighted sum of Cobre hydro ENA/EARM per (codigo_ree, stage).

    *cobre_hydro* is `_AlignedDecompFrames.cobre_hydro` (the aligned
    Cobre hydro means, reused rather than re-read); *ree_by_cobre_id* inverts
    :func:`_ree_membership_map`'s own map, one entry per member plant. A
    Cobre hydro id absent from it -- every plant this rollup could not
    attribute to a membership row -- contributes to no REE sum. Returned
    ``entity_id`` is the REE code (``codigo_ree``), matching the join key
    :func:`_decomp_ree_frame` emits; ``earm_mwh`` is still raw MWh here --
    the ÷730 MWmês reconciliation happens once, at the
    :class:`~cobre_bridge.comparators.model.ResultComparison` emission site.
    """
    empty = pl.DataFrame(schema=_EMPTY_COBRE_REE_SUMS_SCHEMA)
    if cobre_hydro.is_empty() or not ree_by_cobre_id:
        return empty
    if not set(_REE_COBRE_ENERGY_COLUMNS) <= set(cobre_hydro.columns):
        return empty
    mapped = cobre_hydro.with_columns(
        pl.col("entity_id")
        .map_elements(lambda i: ree_by_cobre_id.get(int(i)), return_dtype=pl.Int64)
        .alias("codigo_ree")
    ).filter(pl.col("codigo_ree").is_not_null())
    if mapped.is_empty():
        return empty
    return (
        mapped.group_by(["codigo_ree", "stage_id"])
        .agg(
            pl.col("incremental_inflow_energy_mw").sum().alias("ena_mw"),
            pl.col("stored_energy_final_mwh").sum().alias("earm_mwh"),
        )
        .rename({"codigo_ree": "entity_id"})
        .cast({"entity_id": pl.Int64, "stage_id": pl.Int64})
        .sort(["entity_id", "stage_id"])
    )


def _decomp_ree_frame(
    decomp_dir: Path, *, probabilities: pl.DataFrame | None = None
) -> tuple[pl.DataFrame, dict[int, str]]:
    """DECOMP-side per-(codigo_ree, stage) ENA/EARM, scenario-averaged over
    the deck's own nodes, plus the REE display-name lookup.

    Mirrors every other ``_*_side`` helper's fold
    (:func:`_stage_rows`/:func:`_scenario_mean`), keyed by ``codigo_ree``
    (`read_dec_oper_ree` carries no ``patamar`` column, so :func:`_stage_rows`
    is a no-op here -- called anyway for the same fold shape every other level
    uses). *probabilities* is forwarded to :func:`_scenario_mean` unchanged
    (see :func:`_hydro_side` for its scenario-weighting contract). Rebases
    ``estagio`` onto the 0-based ``stage_id`` the same way
    :func:`_map_entities` does. Raises whatever `read_dec_oper_ree` raises on
    a missing/empty parse -- the caller degrades that to "no REE section".
    """
    frame = _stage_rows(read_dec_oper_ree(decomp_dir))

    names: dict[int, str] = {}
    if "nome_ree" in frame.columns:
        for row in (
            frame.select("codigo_ree", "nome_ree").unique().iter_rows(named=True)
        ):
            names[int(row["codigo_ree"])] = str(row["nome_ree"])

    aggregated = _scenario_mean(
        frame,
        "estagio",
        ["ena_MWmes", "earm_final_MWmes"],
        entity_column="codigo_ree",
        probabilities=probabilities,
    )
    if aggregated.is_empty():
        return aggregated, names
    mapped = aggregated.with_columns(
        pl.col("codigo_ree").cast(pl.Int64).alias("entity_id"),
        (pl.col("estagio").cast(pl.Int64) - 1).alias("stage_id"),
    ).drop("codigo_ree", "estagio")
    return mapped, names


def _ree_result_comparisons(
    decomp_dir: Path,
    cobre_hydro: pl.DataFrame,
    id_map: DecompIdMap | None,
    *,
    probabilities: pl.DataFrame | None = None,
    stage_hours: dict[int, float] | None = None,
) -> tuple[list[ResultComparison], list[int]]:
    """REE-level ``ena_mwmes``/``earm_final_mwmes`` ``ResultComparison`` rows.

    The source-model side is `_decomp_ree_frame` (`read_dec_oper_ree`,
    scenario-averaged per REE/stage -- *probabilities* is forwarded to it
    unchanged, see :func:`_hydro_side` for its scenario-weighting contract);
    the Cobre side is
    :func:`_cobre_ree_sums`'s membership-weighted plant rollup, with EARM
    converted MWh -> MWmês (:data:`_EARM_MWH_TO_MWMES`) and ENA converted from
    a stage-mean MW (average power) to MWmês energy via the stage's own
    duration (*stage_hours* ÷ 730 months), both at emission -- so each is on
    DECOMP's MW-month footing. ``newave_code``/``cobre_id`` are both the
    REE's own ``codigo_ree`` -- Cobre has no independent REE id to diverge
    from it.

    Returns ``([], [])`` when *id_map* is ``None`` (deck unreadable -- no
    hydro code -> Cobre id mapping exists to build the membership map with).
    A missing/empty membership table or `dec_oper_ree` table degrades to no
    REE rows (a genuinely unavailable section, logged at INFO) rather than
    raising -- matching every other optional field this module builds. A
    hydro code absent from the membership table is excluded from every REE
    sum and returned (sorted) as the second element, with a WARNING
    diagnostic recording the codes (never silently dropped).
    """
    if id_map is None:
        return [], []

    hydro_codes = {code: id_map.hydro_id(code) for code in id_map.hydro_codes}

    try:
        membership = read_relato_membership(decomp_dir)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No source-model REE membership table (relato): %s", exc)
        return [], []

    ree_by_cobre_id, unmapped = _ree_membership_map(membership, hydro_codes)
    if unmapped:
        emit(
            Diagnostic(
                code="ree-membership-plant-unmapped",
                severity=Severity.WARNING,
                category="Compare data",
                title="Hydro plant absent from REE membership table",
                summary=(
                    f"{len(unmapped)} source-model hydro plant code(s) carry "
                    "no row in the relato uhes_rees_submercados membership "
                    "table; excluded from every REE energy sum rather than "
                    "silently dropped."
                ),
                notes=[f"codigo_usina: {unmapped}"],
            ),
            logger=_LOG,
        )

    try:
        source_ree, names = _decomp_ree_frame(decomp_dir, probabilities=probabilities)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No source-model REE table (dec_oper_ree): %s", exc)
        return [], unmapped

    cobre_sums = _cobre_ree_sums(cobre_hydro, ree_by_cobre_id)
    if source_ree.is_empty() or cobre_sums.is_empty():
        return [], unmapped

    joined = source_ree.join(cobre_sums, on=["entity_id", "stage_id"], how="inner")
    if joined.is_empty():
        return [], unmapped

    results: list[ResultComparison] = []
    for row in joined.iter_rows(named=True):
        ree_code = int(row["entity_id"])
        stage = int(row["stage_id"])
        name = names.get(ree_code, "")
        cobre_earm_mwmes = float(row["earm_mwh"]) / _EARM_MWH_TO_MWMES
        # cobre's ``ena_mw`` is a stage-mean MW (average inflow power); DECOMP's
        # ``ena_MWmes`` is energy *over the stage* (MW-month), scaling with stage
        # duration. Multiply the rate by the stage's duration in months
        # (``stage_hours / 730``, mirroring EARM's ÷730) — else a weekly stage's
        # mean power is compared to a week's energy, a ~stage_hours/730 mismatch
        # that inflates the weekly ENA several-fold. Falls back to one month when
        # hours are unavailable. The residual per-REE offset is the
        # productivity/FPHA difference vs cobre's per-plant model, not a unit error.
        stage_months = (
            stage_hours.get(stage, _EARM_MWH_TO_MWMES) / _EARM_MWH_TO_MWMES
            if stage_hours is not None
            else 1.0
        )
        cobre_ena_mwmes = float(row["ena_mw"]) * stage_months
        for variable, nw_raw, cobre_value in (
            ("ena_mwmes", row["ena_MWmes"], cobre_ena_mwmes),
            ("earm_final_mwmes", row["earm_final_MWmes"], cobre_earm_mwmes),
        ):
            if nw_raw is None:
                continue
            nw_value = float(nw_raw)
            abs_diff, rel_diff = _result_diff(nw_value, cobre_value)
            results.append(
                ResultComparison(
                    entity_type="ree",
                    entity_name=name,
                    newave_code=ree_code,
                    cobre_id=ree_code,
                    stage=stage,
                    variable=variable,
                    newave_value=nw_value,
                    cobre_value=cobre_value,
                    abs_diff=abs_diff,
                    rel_diff=rel_diff,
                )
            )
    return results, unmapped


# --- evaporation comparison (hydro, "evaporation_m3s") ---
#
# Source model: evaporation as a per-stage *volume* in hm³
# (`read_dec_oper_evap`'s `evaporacao_calculada_hm3` -- the run's own water
# balance, not the fitted `evaporacao_modelo_hm3`). Cobre: a mean *flow* in
# m³/s (`evaporation_m3s`). The divisor reconciling them must come from the
# stage's actual hours (`_cobre_stage_hours`), not the fixed monthly 2.63 the
# source's own MEDIAS report uses (`results.py`'s `_compare_hydros`) -- stages
# here are not always full calendar months (sub-monthly patamares). Reuses
# `cobre_readers._load_block_hours` so the two block-hours readings can't drift.


def _cobre_stage_hours(cobre_output_dir: Path) -> dict[int, float]:
    """``{stage_id: total_hours}`` from the Cobre case's own ``stages.json``.

    Reuses `cobre_readers._load_block_hours` -- the project's one
    ``stages.json`` block-hours reader, already the source every
    hours-weighted `cobre_readers` aggregation goes through -- rather than
    re-parsing the file a second time; sums each stage's block hours into a
    flat per-stage lookup. Empty when ``stages.json`` cannot be found or
    parsed (`_load_block_hours` returns ``None``), which
    :func:`_evaporation_result_comparisons` treats as "no hours denominator
    available" and degrades to no evaporation rows, rather than fabricating
    one.
    """
    block_hours = cobre_readers._load_block_hours(cobre_output_dir)  # noqa: SLF001
    if block_hours is None or block_hours.is_empty():
        return {}
    totals = block_hours.group_by("stage_id").agg(pl.col("hours").sum().alias("hours"))
    return {
        int(row["stage_id"]): float(row["hours"])
        for row in totals.iter_rows(named=True)
    }


#: hm³ -> m³/s conversion factor per hour of stage duration -- the inverse of
#: `core.units.C_M3S2HM3` (``hours * 3600 / 1e6``), generalized to the
#: stage's own ``hours`` instead of the fixed ``MONTH_HOURS`` (730) that
#: constant assumes.
_HM3_TO_M3S_HOUR_FACTOR: float = 3600.0 / 1e6


def _hm3_to_m3s(volume_hm3: float, stage_hours: float) -> float:
    """Convert a stage's evaporated volume (hm³) into a mean flow (m³/s).

    ``m3s = hm3 * 1e6 / (stage_hours * 3600)`` -- volume/time with the hm³ ->
    m³ and hour -> second unit changes folded into
    :data:`_HM3_TO_M3S_HOUR_FACTOR`, the same physical relationship
    `core.units.C_M3S2HM3` expresses in the opposite direction.
    """
    return volume_hm3 / (stage_hours * _HM3_TO_M3S_HOUR_FACTOR)


def _evap_side(
    decomp_dir: Path,
    hydro_codes: dict[int, int],
    *,
    probabilities: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, list[int]]:
    """The source model's per-(hydro, stage) evaporated volume (hm³),
    scenario-averaged over the deck's own nodes.

    Mirrors :func:`_hydro_side`'s exact fold (:func:`_stage_rows` is a no-op
    here -- `read_dec_oper_evap` carries no ``patamar`` column, unlike
    `dec_oper_usih` -- called anyway for the same fold shape every other
    level uses); *probabilities* is forwarded to :func:`_scenario_mean`
    unchanged (see :func:`_hydro_side` for its scenario-weighting contract).
    Raises whatever `read_dec_oper_evap` raises on a missing/empty parse;
    the caller degrades that to "no evaporation section".
    """
    frame = _stage_rows(read_dec_oper_evap(decomp_dir))
    aggregated = _scenario_mean(
        frame,
        "estagio",
        ["evaporacao_calculada_hm3"],
        entity_column="codigo_usina",
        probabilities=probabilities,
    )
    return _map_entities(aggregated, "codigo_usina", hydro_codes)


def _evaporation_result_comparisons(
    decomp_dir: Path,
    cobre_output_dir: Path,
    cobre_hydro: pl.DataFrame,
    id_map: DecompIdMap | None,
    names: dict[int, str],
    *,
    probabilities: pl.DataFrame | None = None,
) -> tuple[list[ResultComparison], list[int]]:
    """Per-(hydro, stage) ``evaporation_m3s`` ``ResultComparison`` rows.

    The source-model side is :func:`_evap_side` (`read_dec_oper_evap`,
    scenario-averaged -- *probabilities* is forwarded to it unchanged, see
    :func:`_hydro_side` for its scenario-weighting contract -- mapped onto
    Cobre ids via *id_map*); the Cobre side is
    *cobre_hydro*'s own ``evaporation_m3s`` column (already carried by
    `_read_aligned_frames`, not re-read here). A plant carrying an
    evaporation series on only one side -- the source model's
    ``dec_oper_evap`` but no Cobre ``evaporation_m3s`` for that plant, or the
    reverse -- is excluded from the paired join and returned (sorted, by
    Cobre id) as the second element instead of being silently dropped, with
    a WARNING diagnostic recording the count.

    TRACKED COBRE-GAP WORKAROUND (C11): Cobre's own evaporation model
    over-scales the volume it deposits on a sub-monthly stage -- it applies
    (pre-fix) a full calendar month's worth of evaporation regardless of the
    stage's actual duration (see
    `decomp.converters.hydro.entity._evaporation_coefficients_mm`'s
    docstring; its removal condition is
    registered in the cobre repository's conversion-found-improvements
    registry). The reconciliation below ONLY rescales the source model's own
    hm³ volume into
    a directly comparable m³/s flow via :func:`_hm3_to_m3s`; Cobre's
    ``evaporation_m3s`` value is compared UNCHANGED. Any residual divergence
    this surfaces on a sub-monthly stage is that known Cobre gap (C11), not a
    conversion error -- it must not be "corrected away" by rescaling Cobre's
    side to match.

    Returns ``([], [])`` when *id_map* is ``None`` (deck unreadable). A
    missing/empty `dec_oper_evap` table, an absent ``evaporation_m3s`` column
    on the Cobre side, or no resolvable stage-hours denominator
    (:func:`_cobre_stage_hours`) each degrade to "no evaporation section"
    (logged at INFO) rather than raising, matching every other optional
    field this module builds.
    """
    if id_map is None:
        return [], []

    hydro_codes = {code: id_map.hydro_id(code) for code in id_map.hydro_codes}

    try:
        source_evap, _unmapped_hydro_codes = _evap_side(
            decomp_dir, hydro_codes, probabilities=probabilities
        )
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No source-model evaporation table (dec_oper_evap): %s", exc)
        return [], []
    if source_evap.is_empty():
        return [], []

    if cobre_hydro.is_empty() or "evaporation_m3s" not in cobre_hydro.columns:
        _LOG.info("Cobre output has no evaporation_m3s column; skipping")
        return [], []
    cobre_evap = cobre_hydro.select(
        "entity_id", "stage_id", "evaporation_m3s"
    ).drop_nulls("evaporation_m3s")

    stage_hours = _cobre_stage_hours(cobre_output_dir)
    if not stage_hours:
        _LOG.info(
            "No stages.json block-hours available; skipping the evaporation "
            "hm3->m3s reconciliation"
        )
        return [], []

    decomp_ids = set(source_evap["entity_id"].unique().to_list())
    cobre_ids = set(cobre_evap["entity_id"].unique().to_list())
    one_sided = sorted(int(i) for i in decomp_ids.symmetric_difference(cobre_ids))
    if one_sided:
        emit(
            Diagnostic(
                code="evaporation-plant-one-sided",
                severity=Severity.WARNING,
                category="Compare data",
                title="Hydro plant has evaporation data on only one side",
                summary=(
                    f"{len(one_sided)} hydro plant(s) carry an evaporation "
                    "series on only one side (the source model's "
                    "dec_oper_evap or Cobre's evaporation_m3s, not both); "
                    "excluded from the paired evaporation comparison rather "
                    "than silently dropped."
                ),
                notes=[f"cobre_id: {one_sided}"],
            ),
            logger=_LOG,
        )

    joined = source_evap.join(cobre_evap, on=["entity_id", "stage_id"], how="inner")
    if joined.is_empty():
        return [], one_sided

    results: list[ResultComparison] = []
    for row in joined.iter_rows(named=True):
        entity_id = int(row["entity_id"])
        stage = int(row["stage_id"])
        hours = stage_hours.get(stage)
        if hours is None or hours <= 0:
            continue
        nw_value = _hm3_to_m3s(float(row["evaporacao_calculada_hm3"]), hours)
        cobre_value = float(row["evaporation_m3s"])
        abs_diff, rel_diff = _result_diff(nw_value, cobre_value)
        results.append(
            ResultComparison(
                entity_type="hydro",
                entity_name=names.get(entity_id, ""),
                newave_code=int(row["newave_code"]),
                cobre_id=entity_id,
                stage=stage,
                variable="evaporation_m3s",
                newave_value=nw_value,
                cobre_value=cobre_value,
                abs_diff=abs_diff,
                rel_diff=rel_diff,
            )
        )
    return results, one_sided


# --- Constraints tab (gc_* metadata, DECOMP-side LHS) ---
#
# `constraints` supplies the source-agnostic pieces (the converted
# case's own constraint/bounds tables, and `evaluate_lhs_cobre`). Only the
# DECOMP-side LHS is missing -- `evaluate_lhs_newave` is
# MEDIAS/NewaveIdMap-coupled and cannot serve DECOMP -- so
# `_generic_constraint_lhs_decomp` below evaluates each constraint against
# `dec_oper_usih`/`dec_oper_usit`/`dec_oper_rhesoft`, recovering via the
# special-constraint register (`decomp.constraint_registers.read_constraints`)
# the SAME coefficients the conversion-time emitters
# (`decomp.converters.constraints.emit_re_generics`/`emit_rhq_rhv_generics`/
# `emit_rhe_generics`) used to author each constraint's expression.


def _dec_oper_hydro_stage_frame(
    decomp_dir: Path, *, probabilities: pl.DataFrame | None = None
) -> pl.DataFrame:
    """One row per (source hydro code, stage): the stage-aggregate
    ``dec_oper_usih`` columns the constraint terms need, averaged
    over node/scenario.

    Reuses `_stage_rows` (keep the duration-weighted ``patamar``-null
    aggregate row) + `_scenario_mean` exactly like `_hydro_side`, but stops
    short of `_map_entities`'s translation to cobre ids -- the
    special-constraint register terms this feeds carry the source model's
    own plant codes, not cobre ids. *probabilities* is forwarded to
    `_scenario_mean` unchanged (see `_hydro_side` for its
    scenario-weighting contract). Degrades to an empty frame (never
    raises) when the deck ships no ``dec_oper_usih.csv`` -- every caller
    already treats an empty frame as "nothing to look up".
    """
    try:
        frame = _stage_rows(read_dec_oper_usih(decomp_dir))
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No dec_oper_usih.csv for the Constraints tab LHS: %s", exc)
        return pl.DataFrame()
    columns = [
        "geracao_MW",
        "vazao_defluente_m3s",
        "vazao_turbinada_m3s",
        "vazao_desviada_m3s",
        "vazao_vertida_m3s",
        "volume_util_final_hm3",
    ]
    return _scenario_mean(
        frame,
        "estagio",
        columns,
        entity_column="codigo_usina",
        probabilities=probabilities,
    )


def _dec_oper_thermal_stage_frame(
    decomp_dir: Path, *, probabilities: pl.DataFrame | None = None
) -> pl.DataFrame:
    """One row per (source thermal code, stage): ``dec_oper_usit``'s
    generation, averaged over node/scenario. Mirrors
    `_dec_oper_hydro_stage_frame`, including the same degrade-to-empty and
    the same *probabilities* scenario-weighting contract."""
    try:
        frame = _stage_rows(read_dec_oper_usit(decomp_dir))
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No dec_oper_usit.csv for the Constraints tab LHS: %s", exc)
        return pl.DataFrame()
    return _scenario_mean(
        frame,
        "estagio",
        ["geracao_MW"],
        entity_column="codigo_usina",
        probabilities=probabilities,
    )


def _dec_oper_rhesoft_frame(decomp_dir: Path) -> pl.DataFrame:
    """Best-effort ``dec_oper_rhesoft.csv`` read.

    Empty when the deck carries no RHE (soft minimum-stored-energy)
    constraints -- the source model omits the file entirely in that case,
    which `read_dec_oper_rhesoft`'s own empty-is-error contract (for a
    genuinely broken/empty parse) does not distinguish on its own.
    """
    try:
        return read_dec_oper_rhesoft(decomp_dir)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No dec_oper_rhesoft.csv (no RHE constraints?): %s", exc)
        return pl.DataFrame()


def _stage_frame_to_lookup(
    frame: pl.DataFrame, value_col: str, *, code_col: str = "codigo_usina"
) -> dict[tuple[int, int], float]:
    """One stage-aggregate ``dec_oper_*`` frame column -> ``{(code,
    stage_0based): value}``.

    *frame* is already collapsed to one row per (code, stage) by
    `_scenario_mean`; this only reshapes it into a lookup and converts the
    source model's 1-based ``estagio`` to the 0-based ``stage_id`` every
    other ``gc_*`` frame uses (D-STAGE-OFFSET, offset=1).
    """
    if (
        frame.is_empty()
        or value_col not in frame.columns
        or code_col not in frame.columns
    ):
        return {}
    return {
        (int(row[code_col]), int(row["estagio"]) - 1): float(row[value_col])
        for row in frame.iter_rows(named=True)
        if row[value_col] is not None
    }


def _storage_lookup(
    hydro_frame: pl.DataFrame,
    id_map: DecompIdMap,
    min_storage_by_cobre_id: dict[int, float],
) -> dict[tuple[int, int], float]:
    """``{(source code, stage_0based): absolute storage hm3}`` = useful + Vmin.

    The source model's own operation export (``volume_util_final_hm3``)
    reports *useful* volume (above the plant's minimum operative storage);
    cobre's ``hydro_storage(id)`` -- what an ``HV``/``VARM`` generic
    constraint's expression is built from -- is absolute. This reuses the
    ``min_storage_hm3`` the converter already wrote to the case's own
    ``system/hydros.json`` (via `cobre_readers.read_cobre_hydro_metadata`) --
    the same floor `_cobre_hydro` subtracts in the other direction for the
    Hydro Operation tab's own storage row -- rather than rebuilding
    ``EffectiveCadastro`` from the deck a second time. A single static value
    per plant, not per-stage: the same simplification
    `constraints_compare._load_hydro_min_storage` already accepts on the
    source-model side.
    """
    useful = _stage_frame_to_lookup(hydro_frame, "volume_util_final_hm3")
    out: dict[tuple[int, int], float] = {}
    for (code, stage), value in useful.items():
        try:
            cobre_id = id_map.hydro_id(code)
        except KeyError:
            continue
        vmin = min_storage_by_cobre_id.get(cobre_id)
        if vmin is None:
            continue
        out[(code, stage)] = value + vmin
    return out


@dataclass(frozen=True)
class _DecompConstraintLookups:
    """Per-(source code, stage) means feeding
    `_generic_constraint_lhs_decomp`'s register-term resolution."""

    hydro_generation: dict[tuple[int, int], float] = field(default_factory=dict)
    thermal_generation: dict[tuple[int, int], float] = field(default_factory=dict)
    flow: dict[str, dict[tuple[int, int], float]] = field(default_factory=dict)
    storage: dict[tuple[int, int], float] = field(default_factory=dict)


#: Register term variables `_generic_constraint_lhs_decomp` cannot re-derive from
#: ``dec_oper_*`` output: ``interchange`` (RE ``FI``) needs the bus/line direction
#: resolution `decomp.converters.constraints.resolve_fi_term` does at conversion time;
#: ``QBOM`` (HQ pumping) names a pumping-station code (not a hydro code), and no
#: ``dec_oper_*`` table reports a pumping station's flow. A constraint carrying either
#: is skipped whole (skip-not-partial), never a partial sum.
_UNSUPPORTED_TERM_VARIABLES = frozenset({"interchange", "QBOM"})

#: Matches a cobre generic constraint's ``name`` field as authored by the
#: conversion-time emitters (``decomp.converters.constraints.emit_re_generics`` ->
#: ``"RE_<id>"``, ``emit_rhq_rhv_generics`` -> ``"HQ_<id>"``/``"HV_<id>"``,
#: ``emit_rhe_generics`` -> ``"RHE_<id>"``), recovering the special-constraint
#: register's own family + ``codigo_restricao`` without re-running the
#: conversion-time id-assignment sequence.
_CONSTRAINT_NAME_RE = re.compile(r"^(RE|HQ|HV|RHE)_(\d+)$")


def _term_lookup_value(
    term: ConstraintTerm, lookups: _DecompConstraintLookups, stage: int
) -> float | None:
    """One register term's per-stage value, or ``None`` when unavailable."""
    if term.variable == "generation":
        return lookups.hydro_generation.get((term.code, stage))
    if term.variable == "thermal_generation":
        return lookups.thermal_generation.get((term.code, stage))
    if term.variable == "VARM":
        return lookups.storage.get((term.code, stage))
    flow_lookup = lookups.flow.get(term.variable)
    return flow_lookup.get((term.code, stage)) if flow_lookup is not None else None


def _rhe_lhs_lookup(
    dec_oper_rhesoft: pl.DataFrame, *, probabilities: pl.DataFrame | None = None
) -> dict[int, dict[int, float]]:
    """``{codigo_restricao: {stage_0based: valor_MW}}``.

    RHE constraints report their achieved LHS directly in
    ``DecOperRheSoft``: ``valor_MW`` is the achieved Σρ_acum·storage in the
    operation -- the same raw physical quantity cobre's own (slack-free)
    expression evaluation produces, so the two overlaid series compare on
    the same basis. Read it directly rather than re-deriving the
    ρ_acum-weighted cascade sum ``decomp.converters.constraints.emit_rhe_generics``
    computes at conversion time, which would duplicate a large, easily
    drifting piece of machinery.

    The file's ``violacao_absoluta_MW`` is the shortfall against the target
    ``Meta`` (``valor_MW + violacao_absoluta_MW == Meta``, the constraint's
    *bound*, not its LHS) -- adding it would collapse the reference trace
    onto the bound line during every violation episode, exactly the moments
    the LHS-vs-bound chart exists to surface. Verified against real
    ``dec_oper_rhesoft.csv`` output. *probabilities* is forwarded to
    `_scenario_mean` unchanged (see `_hydro_side` for its
    scenario-weighting contract).
    """
    out: dict[int, dict[int, float]] = {}
    if dec_oper_rhesoft.is_empty():
        return out
    collapsed = _scenario_mean(
        dec_oper_rhesoft,
        "estagio",
        ["valor_MW"],
        entity_column="codigo_restricao",
        probabilities=probabilities,
    )
    for row in collapsed.iter_rows(named=True):
        valor = row["valor_MW"]
        if valor is None:
            continue
        cid = int(row["codigo_restricao"])
        stage0 = int(row["estagio"]) - 1
        out.setdefault(cid, {})[stage0] = float(valor)
    return out


_GC_LHS_SCHEMA = {
    "constraint_id": pl.Int32,
    "stage_id": pl.Int32,
    "lhs_value": pl.Float64,
}


def _generic_constraint_lhs_decomp(
    case: DecompCase,
    cobre_output_dir: Path,
    gc_constraints: list[dict],
    *,
    probabilities: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Evaluate each generic constraint's LHS from the source model's own
    operation output.

    *case* is the shared, once-parsed deck :func:`build_decomp_dataset`
    builds: its ``id_map`` resolves a plant code to its Cobre hydro id (the
    HV storage floor), and ``dadger`` feeds the special-constraint census
    (:func:`~cobre_bridge.decomp.constraint_registers.read_constraints`) --
    neither is re-parsed here. The ``dec_oper_*`` tables are read directly
    off *case*'s own directory. *probabilities* is forwarded unchanged to
    every underlying :func:`_scenario_mean` fold this derivation relies on
    (:func:`_dec_oper_hydro_stage_frame`/:func:`_dec_oper_thermal_stage_frame`/
    :func:`_rhe_lhs_lookup`) -- see :func:`_hydro_side` for its
    scenario-weighting contract.

    The DECOMP sibling of ``constraints_compare.evaluate_lhs_newave``: that
    function reads MEDIAS-USIH/int*.out through an ``EntityAlignment``/
    ``NewaveIdMap`` this source has neither of, so this evaluates each
    constraint's participating entities directly against
    ``dec_oper_usih``/``dec_oper_usit``/``dec_oper_rhesoft``.

    Per-family strategy:

    - **RE** (hydro/thermal generation): ``generation``/
      ``thermal_generation`` register terms resolve against
      ``dec_oper_usih``/``dec_oper_usit``'s own stage-aggregate row.
    - **HQ** (flow): ``QDEF``/``QTUR``/``QDES``/``QVER`` register terms
      resolve against the matching ``dec_oper_usih`` flow column.
    - **HV** (storage): ``VARM`` register terms resolve against
      ``dec_oper_usih``'s *useful* stage-final volume, floor-adjusted to
      cobre's absolute basis (see `_storage_lookup`).
    - **RHE** (soft stored-energy): read directly from ``DecOperRheSoft``
      (see `_rhe_lhs_lookup`) -- no register/census lookup needed.

    Deferred (cobre-only render, logged, never fabricated): RE
    ``interchange`` (FI) and HQ ``QBOM`` (pumping) terms -- see
    :data:`_UNSUPPORTED_TERM_VARIABLES`. A constraint whose name does not
    match a known family/id (:data:`_CONSTRAINT_NAME_RE`), or whose id has
    no matching register record (deck unavailable, or the record was not
    found), is likewise skipped whole.
    """
    if not gc_constraints:
        return pl.DataFrame(schema=_GC_LHS_SCHEMA)

    from cobre_bridge.decomp.constraint_registers import read_constraints

    decomp_dir = case.files.dadger.parent
    census = read_constraints(case.dadger)
    census_by_key: dict[tuple[str, int], ConstraintRecord] = {}
    for records in census.by_family.values():
        for record in records:
            census_by_key[(record.family, record.constraint_id)] = record

    hydro_frame = _dec_oper_hydro_stage_frame(decomp_dir, probabilities=probabilities)
    min_storage = {
        i: float(m.get("min_storage_hm3") or 0.0)
        for i, m in cobre_readers.read_cobre_hydro_metadata(cobre_output_dir).items()
    }
    storage_lookup = _storage_lookup(hydro_frame, case.id_map, min_storage)

    thermal_frame = _dec_oper_thermal_stage_frame(
        decomp_dir, probabilities=probabilities
    )
    lookups = _DecompConstraintLookups(
        hydro_generation=_stage_frame_to_lookup(hydro_frame, "geracao_MW"),
        thermal_generation=_stage_frame_to_lookup(thermal_frame, "geracao_MW"),
        flow={
            "QDEF": _stage_frame_to_lookup(hydro_frame, "vazao_defluente_m3s"),
            "QTUR": _stage_frame_to_lookup(hydro_frame, "vazao_turbinada_m3s"),
            "QDES": _stage_frame_to_lookup(hydro_frame, "vazao_desviada_m3s"),
            "QVER": _stage_frame_to_lookup(hydro_frame, "vazao_vertida_m3s"),
        },
        storage=storage_lookup,
    )
    rhe_lhs = _rhe_lhs_lookup(
        _dec_oper_rhesoft_frame(decomp_dir), probabilities=probabilities
    )

    rows: list[dict[str, object]] = []
    n_skipped = 0
    for c in gc_constraints:
        name = str(c.get("name", ""))
        cobre_cid = int(c["id"])
        match = _CONSTRAINT_NAME_RE.match(name)
        if match is None:
            _LOG.info(
                "Generic constraint %d (%r) has no recognized source-model "
                "family/id in its name; rendering cobre-only.",
                cobre_cid,
                name,
            )
            n_skipped += 1
            continue
        prefix, raw_cid = match.group(1), int(match.group(2))

        if prefix == "RHE":
            stage_values = rhe_lhs.get(raw_cid)
            if not stage_values:
                _LOG.info(
                    "RHE constraint %d (%s) has no dec_oper_rhesoft rows; "
                    "rendering cobre-only.",
                    cobre_cid,
                    name,
                )
                n_skipped += 1
                continue
            rows.extend(
                {"constraint_id": cobre_cid, "stage_id": stage, "lhs_value": value}
                for stage, value in stage_values.items()
            )
            continue

        record = census_by_key.get((prefix, raw_cid))
        if record is None:
            _LOG.info(
                "Generic constraint %d (%s) has no matching source-model "
                "register record (deck unavailable or record not found); "
                "rendering cobre-only.",
                cobre_cid,
                name,
            )
            n_skipped += 1
            continue
        unsupported = sorted(
            {
                t.variable
                for t in record.terms
                if t.variable in _UNSUPPORTED_TERM_VARIABLES
            }
        )
        if unsupported:
            _LOG.info(
                "Generic constraint %d (%s) carries a term type this "
                "comparator cannot re-derive from dec_oper_* output (%s); "
                "rendering cobre-only.",
                cobre_cid,
                name,
                ", ".join(unsupported),
            )
            n_skipped += 1
            continue

        constraint_rows = 0
        for stage in range(record.stage_start, record.stage_end + 1):
            lhs = 0.0
            complete = True
            for term in record.terms:
                value = _term_lookup_value(term, lookups, stage)
                if value is None:
                    complete = False
                    break
                lhs += term.coefficient * value
            if complete:
                rows.append(
                    {"constraint_id": cobre_cid, "stage_id": stage, "lhs_value": lhs}
                )
                constraint_rows += 1
        if constraint_rows == 0:
            n_skipped += 1

    if n_skipped:
        _LOG.info(
            "DECOMP-side generic-constraint LHS: %d of %d constraint(s) "
            "rendered cobre-only (no source-model LHS derivable).",
            n_skipped,
            len(gc_constraints),
        )
    if not rows:
        return pl.DataFrame(schema=_GC_LHS_SCHEMA)
    return pl.DataFrame(rows).with_columns(
        pl.col("constraint_id").cast(pl.Int32),
        pl.col("stage_id").cast(pl.Int32),
        pl.col("lhs_value").cast(pl.Float64),
    )


def build_decomp_dataset(
    decomp_dir: Path, cobre_output_dir: Path, *, tolerance: float = 1e-2
) -> ComparisonDataset:
    """Build the canonical results dataset for a DECOMP-vs-Cobre comparison.

    One :class:`~cobre_bridge.decomp.case.DecompCase` is built and shared
    across the read/align (:func:`_read_aligned_frames`), Network/
    Productivity/REE/evaporation, and Constraints sections below -- a single
    deck parse for the whole dataset build.

    Reads and aligns both sides via :func:`_read_aligned_frames`, then emits
    the canonical :class:`~cobre_bridge.comparators.model.ResultComparison`
    shape and assembles it through the shared, source-agnostic
    :func:`~cobre_bridge.comparators.analyze.build_results_dataset` kernel.

    Per D-PERCENTILEDATA, the returned dataset's ``PercentileData`` carries:

    - Entity-name dicts (``nw_bus_names``/``nw_hydro_names``) and the
      cobre-side ``bus``/``hydro``/``thermal``/``line`` percentile bands, read
      verbatim from the matching ``cobre_readers`` percentile functions.
    - The Energy Balance tab's DECOMP-side frames ``nw_market``/
      ``nw_net_load``/``nw_sin`` (`_energy_balance_frames`) plus the cobre-side
      ``bus_aggregates``/``cobre_bus_meta``/``cobre_hydro_means``.
    - The Network tab's ``line_bounds``/``line_meta`` (`_line_bounds_and_meta`).
    - The Overview tab's ``nw_costs`` (`_cost_frames`), ``cobre_costs``/
      ``cobre_stage_costs``, and the Convergence overlay ``nw_convergence``/
      ``cobre_convergence`` (`_decomp_convergence_frame`). The per-stage
      ``COPER``/``CUSTO_FUTURO``/``CTERM`` rows are unioned onto ``nw_sin``'s
      EARM/ENA rows via `_union_cost_rows`, never overwriting them.
    - The Performance tab's ``nw_tim_stages``/``nw_tim_iterations``/
      ``nw_max_stage`` (`_decomp_tim_stages`/`_decomp_tim_iterations`/
      `_decomp_max_stage`) plus ``cobre_training_seconds``/
      ``cobre_iteration_timing``. ``nw_tim_iterations`` never carries a
      ``forward_seconds``/``backward_seconds`` split -- DECOMP is nested
      Benders over an explicit tree, with no forward/backward pass to split.
    - The Hydro/Thermal Operation + Plant Details tabs' ``cobre_hydro_meta``
      (`_merge_hydro_bus_ids`, so every entry carries a ``"bus_ids"`` key) and
      ``cobre_hydro_per_stage_bounds``.
    - The Productivity tab's ``fpha_metrics`` (`_fpha_metrics`; its preceding
      [ASSUMPTION] comment covers why this is fit-fidelity statistics at the
      source model's realized points, not a full grid-reconstructed surface).
    - The Constraints tab's ``gc_constraints``/``gc_bounds``/``gc_lhs_cobre``
      (verbatim from ``constraints``) and the DECOMP-side
      ``gc_lhs_newave`` (`_generic_constraint_lhs_decomp`).

    ``results`` also carries a derived ``productivity_mw_per_m3s`` hydro row
    per (plant, stage) (`_hydro_productivity_results`), ``entity_type="ree"``
    ENA/EARM rows (`_ree_result_comparisons`), and a per-(hydro, stage)
    ``evaporation_m3s`` row (`_evaporation_result_comparisons`); these last
    three have no dedicated ``PercentileData`` field and flow through
    ``results``/``dataset.metadata["unmapped"]``.

    Unit reconciliations (D2): REE EARM MWh -> MWmês via
    :data:`_EARM_MWH_TO_MWMES`, REE ENA (a stage-mean MW) -> MWmês via the
    stage's own hours, and evaporation hm³ -> m³/s via the stage's actual
    hours -- the last surfaces a TRACKED COBRE-GAP WORKAROUND (C11) rather
    than correcting it away (see `_evaporation_result_comparisons`).
    ``nw_offset=1`` (D-STAGE-OFFSET): DECOMP's ``estagio`` is 1-based from the
    deck's first stage, so the offset is fixed at 1, not derived from the data.

    Nothing is fabricated to fill a gap: ``bus``/``hydro``/``thermal``/
    ``line`` stay empty whenever the Cobre run has no percentile output (e.g.
    the deterministic 2-node tree), and the ``nw_hydro_slacks``,
    ``productivity_detail``, ``fpha_surface``, and ``fpha_spill`` fields stay
    at their empty defaults, so those sections render their documented "No
    data" path (``report_builder``'s ``has_newave=False`` branch) rather than
    a fabricated comparison -- no percentile spread is ever invented.

    Scenario weighting: :func:`_scenario_probabilities` is built exactly once,
    here, and threaded as the ``probabilities`` keyword into every
    physical-variable fold (``_read_aligned_frames``,
    ``_line_result_comparisons``, ``_ree_result_comparisons``,
    ``_evaporation_result_comparisons``, ``_generic_constraint_lhs_decomp``),
    so a genuine scenario-fan stage is probability-weighted rather than
    plainly averaged -- see the module docstring's "Scenario averaging"
    bullet. A deck with no probability source degrades every fold to its
    unweighted mean.

    Args:
        decomp_dir: The deck directory (results resolved via the deck's
            union discovery).
        cobre_output_dir: Cobre's output directory.
        tolerance: Relative tolerance (default ``1e-2``) forwarded to the
            summary builder.

    Returns:
        The validated dataset. ``metadata["unmapped"]`` carries the per-level
        (hydro/thermal/bus/line/ree) reference codes/corridors the id map,
        the corridor alignment, or the REE membership table could not
        resolve, so they survive into the dataset instead of being silently
        dropped.
    """
    from cobre_bridge.comparators.analyze import build_results_dataset
    from cobre_bridge.decomp.case import DecompCase

    case = DecompCase.from_directory(decomp_dir)
    probabilities = _scenario_probabilities(decomp_dir)

    aligned = _read_aligned_frames(case, cobre_output_dir, probabilities=probabilities)

    results: list[ResultComparison] = []
    # Captured separately so the derived realized productivity
    # (generation / turbined, per plant/stage) can be appended right after
    # -- see _hydro_productivity_results.
    hydro_results = _result_comparisons(
        aligned.source_hydro,
        aligned.cobre_hydro,
        _HYDRO_VARIABLES,
        names=aligned.hydro_names,
    )
    results.extend(hydro_results)
    results.extend(_hydro_productivity_results(hydro_results))
    results.extend(
        _result_comparisons(
            aligned.source_thermal,
            aligned.cobre_thermal,
            _THERMAL_VARIABLES,
            names=aligned.thermal_names,
        )
    )
    results.extend(
        _result_comparisons(
            aligned.source_bus,
            aligned.cobre_bus,
            _BUS_VARIABLES,
            names=aligned.bus_names,
        )
    )

    # --- Network tab (line rows + line/line_bounds/line_meta) ---
    line_bounds, line_meta = _line_bounds_and_meta(cobre_output_dir)
    # The Network/Productivity/REE/evaporation sections below all reuse the
    # shared case's id map rather than each re-parsing the deck.
    line_id_map = case.id_map
    line_results, unresolved_lines = _line_result_comparisons(
        decomp_dir,
        cobre_output_dir,
        line_id_map,
        line_meta,
        probabilities=probabilities,
    )
    results.extend(line_results)

    # --- Overview cost metadata (nw_costs / cobre_costs /
    # nw_sin cost rows / cobre_stage_costs) ---
    nw_costs, nw_cost_rows = _cost_frames(decomp_dir)
    cobre_costs = cobre_readers.read_cobre_cost_breakdown(cobre_output_dir)
    cobre_stage_costs = cobre_readers.read_cobre_stage_costs(cobre_output_dir)
    nw_sin = _union_cost_rows(aligned.nw_sin, nw_cost_rows)

    # --- Performance tab timing metadata (no fabricated DECOMP
    # forward/backward split) ---
    nw_tim_stages = _decomp_tim_stages(decomp_dir)
    nw_tim_iterations = _decomp_tim_iterations(decomp_dir)
    nw_max_stage = _decomp_max_stage(aligned)
    cobre_training_seconds = cobre_readers.read_cobre_training_duration(
        cobre_output_dir
    )
    cobre_iteration_timing = cobre_readers.read_cobre_iteration_timing(cobre_output_dir)

    # --- Hydro Operation + Plant Details tab metadata ---
    cobre_hydro_meta = _merge_hydro_bus_ids(
        cobre_readers.read_cobre_hydro_metadata(cobre_output_dir),
        cobre_readers.read_cobre_hydro_bus_labels(cobre_output_dir),
    )

    # --- Productivity tab's "Fitted production functions (FPHA)" section.
    fpha_metrics = _fpha_metrics(
        decomp_dir, cobre_output_dir, line_id_map, aligned.hydro_names
    )

    # --- REE energy rollup via membership. ---
    ree_results, unmapped_ree = _ree_result_comparisons(
        decomp_dir,
        aligned.cobre_hydro,
        line_id_map,
        probabilities=probabilities,
        stage_hours=_cobre_stage_hours(cobre_output_dir) or None,
    )
    results.extend(ree_results)

    # --- evaporation comparison (hydro, "evaporation_m3s"). See
    # ``_evaporation_result_comparisons``'s docstring for the hm³ -> m³/s
    # reconciliation and the TRACKED COBRE-GAP WORKAROUND (C11) it surfaces.
    evaporation_results, unmapped_evaporation = _evaporation_result_comparisons(
        decomp_dir,
        cobre_output_dir,
        aligned.cobre_hydro,
        line_id_map,
        aligned.hydro_names,
        probabilities=probabilities,
    )
    results.extend(evaporation_results)

    # --- Constraints tab (gc_* metadata). The cobre-side pieces
    # (constraint/bounds tables, simulation LHS) are source-agnostic and
    # reused verbatim from `constraints`; only the DECOMP-side LHS
    # (`_generic_constraint_lhs_decomp`) is new -- see that function's
    # docstring for the per-family derivation.
    from cobre_bridge.cobre.constraint_expr import load_rho_acum_overrides
    from cobre_bridge.comparators.constraints import (
        evaluate_lhs_cobre,
        load_generic_constraint_bounds,
        load_generic_constraints,
    )

    cobre_case_dir = case_dir_for(cobre_output_dir)
    gc_constraints = load_generic_constraints(cobre_case_dir)
    gc_bounds_df = load_generic_constraint_bounds(cobre_case_dir)
    if gc_constraints:
        # RHE's ``@rho_acum_h{id}`` resolves against the LP's per-stage
        # override (same mechanism as VminOP), not the simulation's default
        # productivity column -- unlike the source-model VminOP path, RHE's
        # bound is never useful-energy-shifted (dead volume can exceed the
        # bound here), so only the LHS scaling is corrected.
        rho_acum_overrides = load_rho_acum_overrides(cobre_case_dir)
        gc_lhs_cb = evaluate_lhs_cobre(
            gc_constraints, cobre_output_dir, rho_acum_overrides
        )
        gc_lhs_nw = _generic_constraint_lhs_decomp(
            case, cobre_output_dir, gc_constraints, probabilities=probabilities
        )
    else:
        gc_lhs_cb = pl.DataFrame()
        gc_lhs_nw = pl.DataFrame()

    pct = PercentileData(
        nw_bus_names=aligned.bus_names,
        nw_hydro_names=aligned.hydro_names,
        bus=cobre_readers.read_cobre_bus_percentiles(cobre_output_dir),
        nw_market=aligned.nw_market,
        nw_net_load=aligned.nw_net_load,
        nw_sin=nw_sin,
        # D-STAGE-OFFSET: fixed at 1, not derived — see the docstring above.
        nw_offset=1,
        # Overview tab's Convergence overlay.
        nw_convergence=_decomp_convergence_frame(decomp_dir),
        cobre_convergence=cobre_readers.read_cobre_convergence(cobre_output_dir),
        nw_costs=nw_costs,
        cobre_costs=cobre_costs,
        cobre_stage_costs=cobre_stage_costs,
        bus_aggregates=cobre_readers.read_cobre_bus_aggregates(cobre_output_dir),
        cobre_bus_meta=cobre_readers.read_cobre_bus_metadata(cobre_output_dir),
        cobre_hydro_means=cobre_readers.read_cobre_hydro_means(cobre_output_dir),
        line=cobre_readers.read_cobre_line_percentiles(cobre_output_dir),
        line_bounds=line_bounds,
        line_meta=line_meta,
        # Performance tab timing metadata.
        nw_tim_stages=nw_tim_stages,
        nw_tim_iterations=nw_tim_iterations,
        nw_max_stage=nw_max_stage,
        cobre_training_seconds=cobre_training_seconds,
        cobre_iteration_timing=cobre_iteration_timing,
        # Hydro Operation + Plant Details tab metadata. Never
        # fabricate a percentile spread when the Cobre run has no hydro
        # percentile output (e.g. the deterministic 2-node tree) --
        # ``read_cobre_hydro_percentiles`` already degrades to an empty
        # frame in that case, passed through verbatim.
        hydro=cobre_readers.read_cobre_hydro_percentiles(cobre_output_dir),
        cobre_hydro_meta=cobre_hydro_meta,
        cobre_hydro_per_stage_bounds=cobre_readers.read_cobre_hydro_per_stage_bounds(
            cobre_output_dir
        ),
        # DECOMP has no direct withdrawal/evaporation-slack analog --
        # stays at the dataclass empty-frame default (never set to a
        # DECOMP-derived source) so the slack charts render the cobre
        # Mean + p10/p90 band only (report_builder's has_newave=False path).
        nw_hydro_slacks=pl.DataFrame(),
        # Thermal Operation + Plant Details tab metadata. Never fabricate a
        # percentile spread when the Cobre run has no thermal percentile output
        # (e.g. the deterministic 2-node tree): ``read_cobre_thermal_percentiles``
        # degrades to an empty frame, passed through verbatim.
        thermal=cobre_readers.read_cobre_thermal_percentiles(cobre_output_dir),
        # DECOMP ships no pmo.dat, so the static productivity scatter/
        # building-blocks table have no counterpart -- stays at the empty-frame
        # default (never fabricated), rendering the "No productivity data
        # available" path. Realized per-stage productivity is still filled, via
        # the ``productivity_mw_per_m3s`` rows appended to ``results`` above
        # (`_hydro_productivity_results`), not this field.
        productivity_detail=pl.DataFrame(),
        # Productivity tab's Fitted production functions (FPHA)
        # section -- fallback (b), see the [ASSUMPTION] comment above
        # `_fpha_metrics`. ``fpha_surface``/``fpha_spill`` stay at their
        # dataclass ``None`` default (never fabricated): the declared FPHA
        # readers carry no fitted plane coefficients to reconstruct a dense
        # surface from, only per-hydro/stage fit-fidelity statistics.
        fpha_metrics=fpha_metrics,
        # Constraints tab LHS-vs-bound metadata. ``nw_max_stage``
        # is unchanged (already set above by `_decomp_max_stage`) and
        # doubles as this section's stage clamp (D-STAGE-OFFSET).
        gc_constraints=gc_constraints,
        gc_bounds=gc_bounds_df,
        gc_lhs_newave=gc_lhs_nw,
        gc_lhs_cobre=gc_lhs_cb,
    )
    dataset = build_results_dataset(results, pct, tolerance)
    dataset.metadata["unmapped"] = {
        level: [int(code) for code in codes]
        for level, codes in aligned.unmapped.items()
    }
    dataset.metadata["unmapped"]["line"] = unresolved_lines
    dataset.metadata["unmapped"]["ree"] = unmapped_ree
    dataset.metadata["unmapped"]["evaporation"] = unmapped_evaporation
    return dataset
