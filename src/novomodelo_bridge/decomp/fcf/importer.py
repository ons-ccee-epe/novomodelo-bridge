"""Boundary FCF importer: reads the source model's cut files and authors a
novomodelo policy checkpoint.

:func:`import_boundary_fcf` is the single entry point: it composes the
cut reader (``fcf/cortes.py``), the terminal-manifest bootstrap
(``fcf/bootstrap.py``), the coupling resolution (``fcf/coupling.py``), the
manifest-to-manifest mapper (``fcf/mapper.py``), and the checkpoint writer
(``fcf/writer.py``) in order, then patches the converted case's
``config.json`` so novomodelo loads the authored ``boundary/`` checkpoint at its
terminal stage. Beyond reading the deck's GNL calendar it is orchestration
only.

The importer is a **post-conversion** step: it runs against an already
converted case directory (``convert_decomp_case``'s output), never inside the
conversion itself, because the bootstrap stage needs a real ``novomodelo run`` on
the converted case to read back its terminal state-vector layout.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from idecomp.decomp import Dadgnl, Vazoes
from inewave.newave import Cortesh

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.pandas_utils import as_int
from novomodelo_bridge.core.units import MEAN_MONTH_HOURS
from novomodelo_bridge.decomp.converters.anticipated import (
    classify_gnl_windows,
    operative_week_months,
    read_gnl_model,
)
from novomodelo_bridge.decomp.converters.cadastro import build_effective_cadastro
from novomodelo_bridge.decomp.fcf.bootstrap import (
    bootstrap_terminal_manifest,
    ensure_writer_binding,
)
from novomodelo_bridge.decomp.fcf.cortes import (
    cut_month,
    read_cortes,
    read_patamar_fractions,
    required_inflow_lag_depth,
    summarize_cut_families,
)
from novomodelo_bridge.decomp.fcf.coupling import (
    carried_lag_share,
    coupling_lag_offsets,
    lag_months,
    resolve_couplings,
    source_only_lag_inflows,
    terminal_shares,
)
from novomodelo_bridge.decomp.fcf.mapper import (
    GnlCommandedDelivery,
    GnlDeliveryMonth,
    GnlRingPlan,
    GnlThermalTarget,
    map_boundary_cuts,
    month_key,
)
from novomodelo_bridge.decomp.fcf.writer import (
    build_metadata,
    build_stage_cuts_payload,
    write_boundary_checkpoint,
)
from novomodelo_bridge.decomp.scenarios import convert_recent_observation_windows
from novomodelo_bridge.novomodelo.case_writer import CaseWriter

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from idecomp.decomp import Dadger

    from novomodelo_bridge.decomp.case import DecompCase
    from novomodelo_bridge.decomp.converters.anticipated import GnlCommitmentModel
    from novomodelo_bridge.decomp.fcf.cortes import BoundaryCuts
    from novomodelo_bridge.decomp.fcf.coupling import CouplingResolution, TerminalShare
    from novomodelo_bridge.decomp.fcf.mapper import MappingResult
    from novomodelo_bridge.decomp.temporal import OperativeStage

_LOG = logging.getLogger(__name__)

#: a GNL deviation group whose carried |Σ| is below this FRACTION of the
#: panel's own max |Σ| is treated as numerically vanished relative to the
#: group that actually carries weight — its relative spread
#: `(max_p c_p - min_p c_p) / |Σ_p c_p|` would otherwise inflate into noise.
#: Relative, not a fixed-magnitude floor: an absolute floor cannot track how
#: "small" a Σ is without knowing the panel's own scale (a `1e-6` floor left a
#: Σ≈-4e-05 group in, since `4e-05 >= 1e-6`). Excluded from the
#: `boundary-fcf-gnl-anticipated-deviation` diagnostic's relative `max_spread`
#: HEADLINE only; its per-row table values are unaffected.
_GNL_DEVIATION_REL_FLOOR = 1e-3

_CARRIED_SHARE_TOLERANCE = 1e-9


def _gnl_targets_from(
    model: GnlCommitmentModel, thermals_doc: Mapping[str, object]
) -> tuple[GnlThermalTarget, ...]:
    """Join the ``dadgnl`` plants onto the converted case's GNL thermal ids.

    ``thermals_doc["thermals"]`` entries carrying ``anticipated_config``,
    sorted, are ``convert_decomp_case``'s ``first_thermal_id + i`` assignment
    ascending by code, so zipping them with the codes reproduces it.

    Raises
    ------
    ValueError
        If the case's GNL-fleet count disagrees with the ``dadgnl`` registry's,
        which would otherwise mis-zip ids onto the wrong codes.
    """
    thermals_entries = thermals_doc["thermals"]
    if not isinstance(thermals_entries, list):
        raise TypeError(
            f"thermals.json 'thermals' is {type(thermals_entries).__name__}, not a list"
        )
    gnl_ids = sorted(
        int(entry["id"])
        for entry in thermals_entries
        if isinstance(entry, dict) and "anticipated_config" in entry
    )
    codes = sorted(thermal.code for thermal in model.thermals)
    if len(gnl_ids) != len(codes):
        raise ValueError(
            f"thermals.json carries {len(gnl_ids)} GNL thermal(s) "
            "(anticipated_config), but the dadgnl registry declares "
            f"{len(codes)} thermal(s); the GNL fleet must match exactly"
        )
    id_of = dict(zip(codes, gnl_ids, strict=True))
    return tuple(
        GnlThermalTarget(
            thermal_id=id_of[thermal.code], submercado=thermal.submarket_code
        )
        for thermal in model.thermals
    )


def _date_int(day: date) -> int:
    return day.year * 10_000 + day.month * 100 + day.day


def _commanded_deliveries(
    model: GnlCommitmentModel,
    week_months: Mapping[date, tuple[int, int]],
    horizon_end: date,
) -> tuple[GnlCommandedDelivery, ...]:
    """The já-comandada (class-4) weeks with a nonzero ``gl`` dispatch.

    Read off :func:`classify_gnl_windows`, the same windows the conversion
    hands novomodelo as past commitments; the zero-MW grid stub carries nothing.
    """
    submercado_of = {thermal.code: thermal.submarket_code for thermal in model.thermals}
    block_mw_of = {
        (code, stage.start_date): stage.block_mw
        for code, commitment in model.commitments.items()
        for stage in commitment.stages
    }
    deliveries: list[GnlCommandedDelivery] = []
    classification = classify_gnl_windows(model, horizon_end=horizon_end)
    for code, plant in sorted(classification.plants.items()):
        for start, end, _mw in plant.class4_windows:
            block_mw = block_mw_of.get((code, start), ())
            if not any(block_mw):
                continue
            year, month = week_months.get(start, (start.year, start.month))
            deliveries.append(
                GnlCommandedDelivery(
                    submercado=submercado_of[code],
                    month=month_key(year, month),
                    block_mw=block_mw,
                    hours=(end - start).days * 24.0,
                )
            )
    return tuple(deliveries)


def _build_gnl_ring_plan(
    case_dir: Path,
    case: DecompCase,
    cortesh: Cortesh,
    cut: tuple[int, int],
    lag_maximo_gnl: int,
) -> GnlRingPlan | None:
    """The deck's GNL delivery calendar, or ``None`` when the deck has no GNL."""
    if case.files.dadgnl is None or lag_maximo_gnl == 0:
        return None
    model = read_gnl_model(Dadgnl.read(str(case.files.dadgnl)))
    if model is None:
        return None
    with (case_dir / "system" / "thermals.json").open(encoding="utf-8") as handle:
        targets = _gnl_targets_from(model, json.load(handle))
    if not targets:
        return None
    return _plan_gnl_ring(
        model,
        targets,
        read_patamar_fractions(cortesh),
        cut,
        lag_maximo_gnl,
        case.calendar,
    )


def _plan_gnl_ring(
    model: GnlCommitmentModel,
    targets: tuple[GnlThermalTarget, ...],
    fractions: Mapping[tuple[int, int], tuple[float, ...]],
    cut: tuple[int, int],
    lag_maximo_gnl: int,
    calendar: Sequence[OperativeStage],
) -> GnlRingPlan:
    """Lag ``l`` prices the ``l``-th month after the cut's, ``1..lag_maximo_gnl``.

    Each priced month carries its *fractions* entry (the ``cortesh.dat``
    patamar shares); a month the file does not cover is left out, so its
    slots drop.
    """
    cut_key = month_key(*cut)
    months: dict[int, GnlDeliveryMonth] = {}
    for lag in range(1, lag_maximo_gnl + 1):
        year, month_index = divmod(cut_key + lag, 12)
        shares = fractions.get((year, month_index + 1))
        if shares is not None:
            months[cut_key + lag] = GnlDeliveryMonth(lag, shares)

    weeks = operative_week_months(model.weeks_per_month, calendar)
    return GnlRingPlan(
        targets=targets,
        months=months,
        week_months={_date_int(day): month_key(*ym) for day, ym in weeks.items()},
        commanded=_commanded_deliveries(model, weeks, calendar[-1].end_date),
    )


def _complexo_map(dadger: Dadger) -> dict[int, list[int]]:
    """The deck's ``CX`` register as ``{complexo_code: [component_codes]}``.

    ``CX`` couples a complex of the boundary cut (absent from the DECOMP
    model) to the individual DECOMP plants inside it. Empty when the deck has
    no ``CX`` line.
    """
    complexo_map: dict[int, list[int]] = {}
    cx = dadger.cx(df=True)
    if cx.empty:
        return complexo_map
    for complexo, component in zip(
        cx["codigo_newave"], cx["codigo_decomp"], strict=True
    ):
        complexo_map.setdefault(as_int(complexo), []).append(as_int(component))
    return complexo_map


@dataclass(frozen=True)
class _DeckCoupling:
    """What the deck adds to the cut: who takes which row, and the constants.

    ``cut`` is the cut's ``(year, month)``; ``recent_observations`` the
    ``initial_conditions`` windows read on the coupling postos.
    """

    cut: tuple[int, int]
    resolution: CouplingResolution
    inflow_lag_offsets: Mapping[int, tuple[float, ...]]
    source_only_inflows: Mapping[int, tuple[float, ...]]
    terminal_shares: tuple[TerminalShare, ...]
    carried_share: float
    gnl_plan: GnlRingPlan | None
    recent_observations: list[dict]


def _read_deck_coupling(
    case_dir: Path, case: DecompCase, cortesh: Cortesh, cuts: BoundaryCuts
) -> _DeckCoupling:
    cut = cut_month(cortesh, int(cuts.boundary_stage))
    months = lag_months(*cut)
    calendar = case.calendar
    effective, _ = build_effective_cadastro(case.dadger, case.hidr, calendar)
    vazoes = Vazoes.read(str(case.files.vazoes))
    plant_codes = cuts.header.plant_codes
    resolution = resolve_couplings(
        plant_codes, case.id_map, effective, _complexo_map(case.dadger)
    )
    coupled = {coupling.plant_index for coupling in resolution.couplings}
    source_only = source_only_lag_inflows(
        vazoes,
        effective,
        plant_codes,
        (index for index in range(len(plant_codes)) if index not in coupled),
        calendar,
        months,
    )
    return _DeckCoupling(
        cut=cut,
        resolution=resolution,
        inflow_lag_offsets=coupling_lag_offsets(
            vazoes, effective, case.id_map, calendar, months, resolution.couplings
        ),
        source_only_inflows=source_only,
        terminal_shares=terminal_shares(
            vazoes,
            effective,
            case.id_map,
            resolution.couplings,
            (plant_codes[index] for index in source_only),
            calendar,
        ),
        carried_share=carried_lag_share(calendar, *cut),
        gnl_plan=_build_gnl_ring_plan(
            case_dir, case, cortesh, cut, cuts.header.lag_maximo_gnl
        ),
        recent_observations=convert_recent_observation_windows(
            vazoes, effective, case.id_map, calendar
        ),
    )


def _seed_recent_observations(
    writer: CaseWriter, windows: list[dict], initial_conditions: dict
) -> None:
    """Add ``recent_observations`` to the case's ``initial_conditions.json``.

    Mutates the pipeline's in-memory ``initial_conditions`` (the dict already
    written) and writes it back through *writer*, rather than re-reading the
    file.
    """
    initial_conditions["recent_observations"] = windows
    writer.write_json("initial_conditions.json", initial_conditions)


def _gnl_deviation_rows(
    cuts: BoundaryCuts, gnl_plan: GnlRingPlan
) -> list[tuple[int, int, float, float, float]]:
    """The patamar spread of each priced ``(submercado, lag)`` GNL group.

    Uses ``mapper.py::_GnlLayout.column``'s flat-column layout. Each row is
    ``(submercado, lag, Σ_p c_p, relative spread, absolute spread)`` from the
    active record with the largest relative spread
    ``(max_p c_p - min_p c_p) / |Σ_p c_p|`` (``0.0`` when ``|Σ| <= 1e-9``).
    Groups outside the ``pi_gnl`` shape are skipped; rows are sorted.
    """
    n_patamares = cuts.header.n_patamares
    lag_maximo_gnl = cuts.header.lag_maximo_gnl
    active_records = tuple(record for record in cuts.records if record.is_active)
    if not active_records or lag_maximo_gnl == 0:
        return []
    width = len(active_records[0].pi_gnl)
    block = n_patamares * lag_maximo_gnl
    if width == 0 or block == 0 or width % block != 0:
        return []
    n_submercados = width // block

    def col(submercado: int, patamar: int, lag: int) -> int:
        """Flat pi_gnl column for (submercado, patamar, lag), 1-based axes."""
        return ((submercado - 1) * n_patamares + (patamar - 1)) * lag_maximo_gnl + (
            lag - 1
        )

    groups = sorted(
        {
            (target.submercado, month.lag)
            for target in gnl_plan.targets
            for month in gnl_plan.months.values()
        }
    )
    rows: list[tuple[int, int, float, float, float]] = []
    for submercado, lag in groups:
        if not (1 <= submercado <= n_submercados) or not (1 <= lag <= lag_maximo_gnl):
            continue
        cols = tuple(
            col(submercado, patamar, lag) for patamar in range(1, n_patamares + 1)
        )
        best_sum = 0.0
        best_spread = 0.0
        best_abs_spread = 0.0
        for record in active_records:
            values = tuple(record.pi_gnl[c] for c in cols)
            total = math.fsum(values)
            abs_spread = max(values) - min(values)
            spread = abs_spread / abs(total) if abs(total) > 1e-9 else 0.0
            if spread >= best_spread:
                best_spread = spread
                best_sum = total
                best_abs_spread = abs_spread
        rows.append((submercado, lag, best_sum, best_spread, best_abs_spread))
    return rows


def _emit_import_diagnostics(
    cuts: BoundaryCuts,
    mapping: MappingResult,
    gnl_plan: GnlRingPlan | None = None,
) -> None:
    """Surface the mapping's accepted approximations.

    Always emits ``boundary-fcf-cut-family-summary``; adds
    ``boundary-fcf-source-only-plants-dropped`` when a cut plant has no taker,
    and ``boundary-fcf-gnl-anticipated-deviation`` when *gnl_plan* is given and
    the cut carries a GNL block.
    """
    summary = summarize_cut_families(cuts)
    dx.emit(
        dx.Diagnostic(
            code="boundary-fcf-cut-family-summary",
            severity=dx.Severity.INFO,
            category="Boundary FCF",
            title=f"Boundary FCF authors {summary.n_active_cuts} cut(s)",
            summary=(
                f"{summary.n_active_cuts} active cut(s) authored from the "
                f"source model's boundary cuts; {summary.storage_nonzero_plants} "
                "plant(s) carry a nonzero storage coefficient; nonzero "
                f"inflow-lag plants by depth {summary.lag_nonzero_by_depth}; "
                f"RHS range [{summary.rhs_min:.6g}, {summary.rhs_max:.6g}]"
            ),
        ),
        logger=_LOG,
    )

    if mapping.dropped:
        dx.emit(
            dx.Diagnostic(
                code="boundary-fcf-source-only-plants-dropped",
                severity=dx.Severity.INFO,
                category="Boundary FCF",
                title=(
                    f"{len(mapping.dropped)} source-only plant(s) dropped "
                    "from the boundary FCF state"
                ),
                summary=(
                    f"{len(mapping.dropped)} plant(s) of the source model's "
                    "boundary cuts are not in the converted case; their "
                    "storage terms are omitted, and their inflow-lag terms "
                    "are folded into each cut's intercept from the deck's "
                    "inflow history where the plant has a posto (the "
                    "terminal stage's share stays out)"
                ),
                table=dx.DiagnosticTable(
                    columns=["Plant code", "β (pi_varm)", "Inflow folded"],
                    rows=[
                        [
                            term.plant_code,
                            round(term.beta, 6),
                            "yes" if term.inflow_folded else "no",
                        ]
                        for term in mapping.dropped
                    ],
                    justify=["right", "right", "left"],
                ),
            ),
            logger=_LOG,
        )

    if gnl_plan is not None and cuts.header.lag_maximo_gnl > 0:
        _emit_gnl_deviation(cuts, mapping, gnl_plan)


def _emit_gnl_deviation(
    cuts: BoundaryCuts, mapping: MappingResult, gnl_plan: GnlRingPlan
) -> None:
    rows = _gnl_deviation_rows(cuts, gnl_plan)
    max_abs_sum = max((abs(row[2]) for row in rows), default=0.0)
    headline_rows = (
        [row for row in rows if abs(row[2]) >= _GNL_DEVIATION_REL_FLOOR * max_abs_sum]
        if max_abs_sum > 1e-12
        else []
    )
    relative_headline = (
        f"{max(row[3] for row in headline_rows):.4g}" if headline_rows else "n/a"
    )
    max_absolute_spread = max((row[4] for row in rows), default=0.0)
    dx.emit(
        dx.Diagnostic(
            code="boundary-fcf-gnl-anticipated-deviation",
            severity=dx.Severity.INFO,
            category="Boundary FCF",
            title="GNL anticipated ring prices one MW per slot, not per patamar",
            summary=(
                "the source model commits GNL per patamar, the converted case "
                "one MW per ring slot, so each slot is priced at the patamar "
                "prices weighted by the source's patamar shares; max patamar "
                f"spread {relative_headline} relative / "
                f"{max_absolute_spread:.4g} absolute across {len(rows)} "
                f"(submercado, lag) group(s) (a group carrying less than "
                f"{_GNL_DEVIATION_REL_FLOOR:g} of the largest |Σ| is left out "
                f"of the relative figure); {len(mapping.gnl_dropped)} GNL "
                "term(s) reach no ring slot"
            ),
            table=dx.DiagnosticTable(
                columns=["Submercado", "Lag", "Σ pi_gnl (carried)", "Patamar spread"],
                rows=[
                    [submercado, lag, round(carried_sum, 6), round(spread, 6)]
                    for submercado, lag, carried_sum, spread, _abs_spread in rows
                ],
                justify=["right", "right", "right", "right"],
            ),
            notes=[
                f"thermal {term.thermal_id}, submercado {term.submercado}: "
                f"{term.reason}"
                if term.thermal_id is not None
                else f"submercado {term.submercado}, lag {term.lag}: {term.reason} "
                f"(Σ pi_gnl {term.coefficient:.6g})"
                for term in mapping.gnl_dropped
            ],
        ),
        logger=_LOG,
    )


def _emit_transit_pricing(mapping: MappingResult, hydro_codes: Sequence[int]) -> None:
    """Surface the boundary's price for water in transit, which the source
    model's own boundary leaves unpriced."""
    if not mapping.transit:
        return
    dx.emit(
        dx.Diagnostic(
            code="boundary-fcf-transit-water-priced",
            severity=dx.Severity.INFO,
            category="Boundary FCF",
            title=(
                f"{len(mapping.transit)} plant(s) value the water still in "
                "transit toward them at the boundary"
            ),
            summary=(
                "water released upstream that has not reached the plant when "
                "the study ends is valued with the plant's boundary storage "
                "term when it takes one, otherwise with its first inflow-lag "
                "term; the source model gives that water no value at this "
                "boundary, so the boundary cost differs from the source "
                "model's by the value of the water in transit"
            ),
            table=dx.DiagnosticTable(
                columns=["Plant", "Valued as"],
                rows=[
                    [
                        hydro_codes[entry.hydro_id],
                        "storage" if entry.by_storage else "first inflow lag",
                    ]
                    for entry in mapping.transit
                ],
                justify=["right", "left"],
            ),
        ),
        logger=_LOG,
    )


def _emit_coupling_diagnostics(
    resolution: CouplingResolution,
    shares: Sequence[TerminalShare],
    carried_share: float,
) -> None:
    """Surface how the converted hydros take the cut's plant terms."""
    if resolution.borrowed:
        dx.emit(
            dx.Diagnostic(
                code="boundary-fcf-coefficient-owners",
                severity=dx.Severity.INFO,
                category="Boundary FCF",
                title=(
                    f"{len(resolution.borrowed)} plant(s) take another cut "
                    "plant's boundary terms"
                ),
                summary=(
                    "a plant without its own boundary-cut row takes the row of "
                    "its complex (CX) or, failing that, of the first downstream "
                    "plant that has one; it takes the storage term only through "
                    "a complex, and only as a reservoir"
                ),
                table=dx.DiagnosticTable(
                    columns=["Plant", "Owner", "Rule"],
                    rows=[
                        [
                            entry.plant_code,
                            entry.owner_code,
                            "complex (CX)" if entry.via_complex else "downstream",
                        ]
                        for entry in resolution.borrowed
                    ],
                    justify=["right", "right", "left"],
                ),
            ),
            logger=_LOG,
        )

    if resolution.unowned:
        dx.emit(
            dx.Diagnostic(
                code="boundary-fcf-unowned-plants",
                severity=dx.Severity.ERROR,
                category="Boundary FCF",
                title=(f"{len(resolution.unowned)} plant(s) take no boundary-cut term"),
                summary=(
                    "these plants have no row in the boundary cuts, belong to "
                    "no complex, and reach no downstream plant that has one, so "
                    "the boundary cost-to-go gives their water no value"
                ),
                table=dx.DiagnosticTable(
                    columns=["Plant"],
                    rows=[[code] for code in resolution.unowned],
                    justify=["right"],
                ),
            ),
            logger=_LOG,
        )

    # TRACKED NOVOMODELO-GAP (C25): the coupling stage's outgoing lag 1 is its own
    # inflow alone; the share of the month earlier stages carry is left out.
    if carried_share > _CARRIED_SHARE_TOLERANCE:
        dx.emit(
            dx.Diagnostic(
                code="boundary-fcf-coupling-lag-carried-share",
                severity=dx.Severity.WARNING,
                category="Boundary FCF",
                title="The coupling month starts before the last stage",
                summary=(
                    f"{carried_share:.1%} of the boundary cut's month lies in "
                    "stages before the last one; novomodelo evaluates the cut's "
                    "first inflow lag on the last stage's inflow alone, so that "
                    "share is priced at the last stage's inflow instead of the "
                    "inflow that occurred"
                ),
            ),
            logger=_LOG,
        )

    # TRACKED NOVOMODELO-GAP (C26): novomodelo's lag state follows each hydro's
    # operation inflow; the terminal stage's coupling-inflow difference and
    # the source-only plants' terminal inflow cannot be folded.
    if shares:
        dx.emit(
            dx.Diagnostic(
                code="boundary-fcf-coupling-inflow-unrepresented",
                severity=dx.Severity.WARNING,
                category="Boundary FCF",
                title=(
                    f"{len(shares)} plant(s) couple on an inflow novomodelo's lag "
                    "state does not carry in the last stage"
                ),
                summary=(
                    "the source model evaluates the boundary cut's inflow terms "
                    "on each plant's coupling posto; the deterministic stages' "
                    "difference is folded into each cut's intercept, but the "
                    "last stage's scenario-dependent difference is not "
                    "represented"
                ),
                table=dx.DiagnosticTable(
                    columns=[
                        "Plant",
                        "Coupling posto",
                        "Operation posto",
                        "Max gap (m³/s)",
                    ],
                    rows=[
                        [
                            share.plant_code,
                            share.coupling_posto,
                            "-"
                            if share.operation_posto is None
                            else share.operation_posto,
                            round(share.max_gap_m3s, 3),
                        ]
                        for share in shares
                    ],
                    justify=["right", "right", "right", "right"],
                ),
            ),
            logger=_LOG,
        )


def _patch_policy_boundary(writer: CaseWriter, config: dict) -> None:
    """Set ``["policy"]["boundary"]`` in ``config.json``, preserving the rest.

    The block carries only ``path``: novomodelo's boundary loader selects the
    source pool by calendar date (the pool whose ``priced_state_date`` equals
    the loading study's own boundary date), so no stage/pool index is written.
    ``policy.boundary.source_stage`` is removed from novomodelo's config contract
    and rejected by its deny-unknown-fields validation; ``strict`` is left
    unset (its ``false`` default), so a source pricing more state than the
    study models loads and records the superset in the reconciliation report
    rather than rejecting.

    Mutates the pipeline's own in-memory ``config`` dict (the one already
    written to ``config.json``, ``state_space``/``training``/``simulation``
    included) — creating ``["policy"]`` if the case predates any policy
    section — and writes it back through *writer*, rather than re-reading the
    file, so the patched file matches the rest of the case's JSON output
    byte-for-byte in style.
    """
    policy = config.setdefault("policy", {})
    policy["boundary"] = {"path": "boundary"}
    writer.write_json("config.json", config)


def import_boundary_fcf(
    case_dir: Path,
    case: DecompCase,
    *,
    work_dir: Path,
    cost_scale_factor: float,
    config: dict | None = None,
    initial_conditions: dict | None = None,
) -> Path | None:
    """Import the source model's boundary FCF into the converted case at
    ``case_dir``.

    *config*/*initial_conditions* are the exact in-memory dict objects the
    pipeline authored and wrote to ``config.json``/``initial_conditions.json``
    pre-bootstrap; :func:`_patch_policy_boundary`/:func:`_seed_recent_observations`
    mutate and rewrite them through *writer* rather than re-reading either file
    off disk. ``None`` (the default) is only valid on the no-cut-files no-op
    path below — once cut files are present and the import actually proceeds,
    both are required (see the ``ValueError`` below), since patching either
    file from an absent dict would silently drop the rest of its content
    rather than mutate it.

    Gated on cut-file presence: if either ``case.files.cortesh`` or
    ``case.files.cortes`` is ``None``, this is a no-op — no ``boundary/``
    directory is written, and ``config.json`` is left untouched. Otherwise:

    1. Reads the shared case's ``dadger``/``id_map`` (parsed once by the
       caller, not re-parsed here) — same-study: the boundary cut source is
       the very deck whose ``dadger`` produced ``case_dir``, so no separate
       deck path is needed.
    2. Reads the header (``case.files.cortesh``) and the boundary-stage cut
       records (``case.files.cortes``), deriving the boundary stage from the
       cut file's own trailer when it is a single-stage partition export.
    3. Checks the writer binding, then runs a 1-iteration in-process
       ``novomodelo.run.run`` pass on ``case_dir`` (checkpoint under ``work_dir``,
       the case is never mutated) to read back its terminal state-vector
       layout.
    4. Resolves which converted hydro takes which cut plant's terms
       (:func:`~novomodelo_bridge.decomp.fcf.coupling.resolve_couplings`), the
       inflow-lag constants novomodelo's state cannot carry, and the deck's GNL
       delivery calendar (:func:`_build_gnl_ring_plan`), then maps every
       boundary cut onto that layout (:func:`~novomodelo_bridge.decomp.fcf.mapper.
       map_boundary_cuts`).
    5. Surfaces the mapping's approximations as ``Diagnostic``s.
    6. Writes ``case_dir/boundary/``, points ``config.json``'s
       ``["policy"]["boundary"]`` at it, and seeds ``initial_conditions.json``'s
       ``recent_observations`` from the coupling postos
       (:func:`_seed_recent_observations`).

    Returns the ``case_dir/boundary`` path, or ``None`` on the no-cut-files
    no-op.

    Raises
    ------
    RuntimeError
        Propagated verbatim from ``fcf.bootstrap``'s ``ensure_writer_binding``
        (writer binding missing) or ``bootstrap_terminal_manifest`` (the
        in-process ``novomodelo.run.run`` failed or its checkpoint was malformed).
    ValueError
        Raised directly when cut files are present but *config* or
        *initial_conditions* is ``None`` (a caller must thread the pipeline's
        own in-memory dicts once the import actually proceeds — see
        ``convert_decomp_case``'s ``fcf_inputs_out``). Also propagated
        verbatim from the cut reader (``fcf/cortes.py``, e.g. a
        non-individualized deck or a nonzero SAR coefficient), the mapper
        (``fcf/mapper.py``, e.g. no ``HydroStorage`` slots in the target
        manifest), or the writer (``fcf/writer.py``, e.g. a mapped
        coefficient vector length mismatch).
    """
    if case.files.cortesh is None or case.files.cortes is None:
        _LOG.info("boundary FCF skipped — no cut files")
        return None
    if config is None or initial_conditions is None:
        raise ValueError(
            "boundary FCF import requires the converted case's config and "
            "initial_conditions once cut files are present"
        )

    # A real, non-dry-run writer: the importer runs a real ``novomodelo`` pass and
    # never runs under ``--dry-run``.
    writer = CaseWriter(case_dir)

    cortesh = Cortesh.read(str(case.files.cortesh))
    cuts = read_cortes(case.files.cortes, cortesh, boundary_stage=None)
    # `BoundaryCuts.boundary_stage` is typed `int`, but a single-stage export's
    # derived value inherits `numpy.int32` from `cortesh.ano_inicio_estudo`'s
    # own numpy dtype (confirmed against this deck) — narrow to a plain `int`
    # here, at the boundary between the numpy-sourced reader and the
    # JSON/novomodelo-FFI payloads this function builds below.
    boundary_stage = int(cuts.boundary_stage)

    ensure_writer_binding()

    # No explicit inflow-lag depth is supplied to the bootstrap: novomodelo sizes the
    # HydroInflowLag state block from the case's own PAR(p) model order (the same
    # autoregressive structure the boundary cuts' pi_qafl terms derive from), so
    # the terminal manifest already carries the slots the mapper places those
    # terms onto. The former state_space.inflow_lag_depth override was redundant
    # with that sizing and is rejected by novomodelo >= 0.14.
    manifest = bootstrap_terminal_manifest(case_dir, work_dir=work_dir)
    coupling = _read_deck_coupling(case_dir, case, cortesh, cuts)
    # The DECOMP case has no PAR(p) model, so the bootstrap manifest carries no
    # HydroInflowLag slots; the deepest lag the cuts reference is declared to
    # the writer, which reserves the canonical slots.
    inflow_lag_depth = required_inflow_lag_depth(summarize_cut_families(cuts))
    mapping = map_boundary_cuts(
        cuts,
        manifest,
        coupling.resolution.couplings,
        inflow_lag_depth=inflow_lag_depth,
        inflow_lag_offsets=coupling.inflow_lag_offsets,
        source_only_inflows=coupling.source_only_inflows,
        gnl_plan=coupling.gnl_plan,
    )
    _LOG.info(
        "boundary FCF coupled at %04d-%02d: cut terms scaled over the source "
        "model's %.1f h mean month (inflow lags additionally x C_M3S2HM3 for "
        "novomodelo's m3/s lag state)",
        *coupling.cut,
        MEAN_MONTH_HOURS,
    )
    _emit_import_diagnostics(cuts, mapping, coupling.gnl_plan)
    _emit_transit_pricing(mapping, case.id_map.hydro_codes)
    _emit_coupling_diagnostics(
        coupling.resolution, coupling.terminal_shares, coupling.carried_share
    )

    stage_cuts_payload = build_stage_cuts_payload(
        mapping,
        manifest,
        stage_id=boundary_stage,
        cost_scale_factor=cost_scale_factor,
        node_id=manifest.node_id,
        graph_stage_id=manifest.graph_stage_id,
        priced_state_date=manifest.priced_state_date,
    )
    completed_iterations = max((cut.iteration for cut in mapping.cuts), default=0)
    metadata = build_metadata(
        num_stages=1,
        cost_scale_factor=cost_scale_factor,
        completed_iterations=completed_iterations,
        final_lower_bound=0.0,
        max_iterations=completed_iterations,
        forward_passes=0,
        warm_start_cuts=0,
        rng_seed=0,
        created_at=datetime.now(tz=UTC).isoformat(),
        season_manifest=manifest.season_manifest,
    )

    boundary_dir = case_dir / "boundary"
    write_boundary_checkpoint(
        boundary_dir,
        stage_cuts_payload,
        metadata,
        inflow_lag_depth=inflow_lag_depth,
    )

    _patch_policy_boundary(writer, config)
    _seed_recent_observations(writer, coupling.recent_observations, initial_conditions)
    _LOG.info(
        "boundary FCF: seeded %d recent-observation window(s) on the coupling postos",
        len(coupling.recent_observations),
    )

    return boundary_dir
