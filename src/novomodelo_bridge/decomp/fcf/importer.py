"""Boundary FCF importer: reads the source model's cut files and authors a
cobre policy checkpoint.

:func:`import_boundary_fcf` is the single entry point: it composes the
cut reader (``fcf/cortes.py``), the terminal-manifest bootstrap
(``fcf/bootstrap.py``), the manifest-to-manifest mapper (``fcf/mapper.py``),
and the checkpoint writer (``fcf/writer.py``) in order, then patches the
converted case's ``config.json`` so cobre loads the authored ``boundary/``
checkpoint at its terminal stage. It is thin orchestration only — every
algorithm it calls already exists in one of the four modules above; this
module adds no new cut-mapping or checkpoint-authoring logic of its own.

The importer is a **post-conversion** step: it runs against an already
converted case directory (``convert_decomp_case``'s output), never inside the
conversion itself, because the bootstrap stage needs a real ``cobre run`` on
the converted case to read back its terminal state-vector layout.
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from idecomp.decomp import Dadgnl, Mlt, Vazoes
from inewave.newave import Cortesh

from cobre_bridge.cobre.case_writer import CaseWriter
from cobre_bridge.core import diagnostics as dx
from cobre_bridge.decomp.converters.anticipated import read_gnl_model
from cobre_bridge.decomp.converters.cadastro import build_effective_cadastro
from cobre_bridge.decomp.fcf.bootstrap import (
    bootstrap_terminal_manifest,
    ensure_writer_binding,
)
from cobre_bridge.decomp.fcf.cortes import (
    read_cortes,
    required_inflow_lag_depth,
    summarize_cut_families,
)
from cobre_bridge.decomp.fcf.mapper import (
    GnlRingPlan,
    GnlThermalTarget,
    map_boundary_cuts,
)
from cobre_bridge.decomp.fcf.writer import (
    build_metadata,
    build_stage_cuts_payload,
    write_boundary_checkpoint,
)
from cobre_bridge.decomp.files import DecompFiles
from cobre_bridge.decomp.inflow_mlt import build_incremental_mlt, coupling_lag_means
from cobre_bridge.decomp.scenarios import convert_recent_observation_windows

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from cobre_bridge.decomp.case import DecompCase
    from cobre_bridge.decomp.converters.anticipated import GnlCommitmentModel
    from cobre_bridge.decomp.converters.cadastro import EffectiveCadastro
    from cobre_bridge.decomp.fcf.cortes import BoundaryCuts
    from cobre_bridge.decomp.fcf.mapper import MappingResult
    from cobre_bridge.decomp.temporal import OperativeStage

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


def _gnl_targets_from(
    model: GnlCommitmentModel, thermals_doc: Mapping[str, object]
) -> GnlRingPlan | None:
    """Build the submercado -> GNL-thermal ring plan from the deck + case.

    Joins ``model.thermals`` (ascending by ``code``, read from ``dadgnl``'s
    ``tg`` registry by
    :func:`~cobre_bridge.decomp.converters.anticipated.read_gnl_model` —
    reconciled with, never re-derived) onto the converted case's GNL
    thermal ids: ``thermals_doc["thermals"]`` entries carrying
    ``anticipated_config``, sorted ascending, are exactly
    ``convert_decomp_case``'s ``first_thermal_id + i`` assignment (ascending
    by code — see ``decomp/pipeline.py``/``decomp/anticipated.py::convert_gnl``).
    Zipping the two ascending-by-code sequences positionally reproduces that
    assignment exactly, without re-deriving it.

    A plant absent from ``model.nl_lag_months`` has no dispatch-anticipation
    lag to key its ring slot(s) on (no ``pi_gnl`` lag axis), so it is
    *skipped* — its ring stays at coefficient ``0.0`` via the mapper's own
    D3-like drop path — with a single INFO log line naming every skipped
    plant, never raised on. Returns ``None`` when every plant is skipped (no
    live target at all).

    Raises
    ------
    ValueError
        If the converted case's GNL-fleet count (``thermals_doc["thermals"]``
        entries carrying ``anticipated_config``) disagrees with the
        ``dadgnl`` registry's thermal count — a converter/reader
        inconsistency that would otherwise silently mis-zip ids onto the
        wrong codes.
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

    targets: list[GnlThermalTarget] = []
    skipped: list[str] = []
    for thermal in model.thermals:
        lag = model.nl_lag_months.get(thermal.code)
        if lag is None:
            skipped.append(thermal.name)
            continue
        targets.append(
            GnlThermalTarget(
                thermal_id=id_of[thermal.code],
                submercado=thermal.submarket_code,
                nl_lag=lag,
            )
        )

    if skipped:
        _LOG.info(
            "%d GNL plant(s) declare no nl dispatch-anticipation lag, so "
            "their AnticipatedThermalState ring slot(s) stay at coefficient "
            "0.0 (no pi_gnl lag axis to key on): %s",
            len(skipped),
            ", ".join(skipped),
        )

    return GnlRingPlan(tuple(targets)) if targets else None


def _post_horizon_start(case_dir: Path) -> int | None:
    """The ``YYYYMM01`` month-anchor of the earliest post-study stage start,
    or ``None``.

    Reads ``case_dir / "post_study_stages.json"`` (the
    ``decomp/anticipated.py::GnlEmission.post_study_stages`` payload,
    written by ``decomp/pipeline.py`` only when some GNL delivery lands
    post-horizon). Returns ``None`` when the file is absent or its
    ``stages`` list is empty — a case shape as legitimate as "no post-study
    horizon at all"; the mapper's ``GnlRingPlan.post_horizon_start=None``
    disables the covered-lane filter entirely for that case.

    The earliest stage's ``year*10000 + month*100 + 1`` month-anchor mirrors
    the granularity cobre's excised anticipated ring dates every surviving
    slot at (its own ``year_month_day_anchor``): the já-comandada (class-4)
    window is excised from the ring entirely — cobre's ``ring_index`` returns
    ``None`` for it, so it never reaches the terminal manifest — which under
    the source model's month-boundary study makes this anchor the first
    signaled (class-3) month.

    A malformed ``start_date`` string is NOT swallowed here: the
    ``ValueError`` from ``date.fromisoformat`` propagates verbatim rather
    than silently disabling the filter — a corrupt horizon is a real
    problem, never a "no horizon" case.
    """
    path = case_dir / "post_study_stages.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        doc = json.load(handle)
    stages = doc.get("stages", [])
    if not stages:
        return None
    earliest = min(date.fromisoformat(stage["start_date"]) for stage in stages)
    return earliest.year * 10000 + earliest.month * 100 + 1


def _final_stage_block_hours(case_dir: Path) -> list[float]:
    """The coupling (terminal) stage's per-block hours, in file order, from
    ``stages.json``.

    The boundary FCF attaches at the end of the last modelled stage, so the
    coupling period is the final ``stages.json`` stage. Returns its per-block
    ``hours`` values in file order — patamar order, matching the source
    model's own cut axis (a weekly deck's ~168 h split across blocks, or the
    coupling month's post-exclusion hours — e.g. 648 h for a 27-day April) —
    so :func:`~cobre_bridge.decomp.fcf.mapper.map_boundary_cuts`'s GNL branch
    can weight each ``pi_gnl`` patamar column by its own coupling block's
    hours, rather than the stage's total.
    :func:`_coupling_stage_hours` sums this same vector for the
    intercept/storage/inflow-lag terms.

    Raises
    ------
    ValueError
        If ``stages.json`` is missing, carries no stages, or the final stage
        has no positive total block-hours — a boundary import cannot scale
        the cut coefficients to a cost without the coupling stage's duration.
    """
    path = case_dir / "stages.json"
    if not path.is_file():
        raise ValueError(f"stages.json not found at {path}; cannot scale boundary FCF")
    with path.open(encoding="utf-8") as handle:
        doc = json.load(handle)
    stages = doc.get("stages", [])
    if not stages:
        raise ValueError(f"{path} carries no stages; cannot scale boundary FCF")
    hours = [float(block["hours"]) for block in stages[-1].get("blocks", [])]
    total = math.fsum(hours)
    if total <= 0.0:
        raise ValueError(
            f"{path} final stage has non-positive total block hours ({total}); "
            "cannot scale boundary FCF"
        )
    return hours


def _coupling_stage_hours(case_dir: Path) -> float:
    """The coupling (terminal) stage's total duration in hours, from ``stages.json``.

    The source model's FCF coefficients are a *per-hour* cost rate (see
    ``fcf/mapper.py``'s header): the source integrates that rate over its
    coupling period's actual hours to obtain the future-cost value, so the
    mapper needs those same hours (``cost_unit_hours``) to reproduce it in
    cobre's plain-$ objective. Using the actual stage hours (not a fixed 730-h
    month) reproduces the source's ``E(CF)`` to ~2-3 %, where a fixed month
    overshoots by ~15 % on a short coupling period.

    A thin ``math.fsum`` wrapper over :func:`_final_stage_block_hours` — see
    that function for the per-block vector and the validation the two share.

    Raises
    ------
    ValueError
        Propagated verbatim from :func:`_final_stage_block_hours`.
    """
    return math.fsum(_final_stage_block_hours(case_dir))


#: A ``CX`` register line: ``CX <complexo_code> <component_code>``. idecomp
#: exposes no typed accessor for ``CX``, so it is parsed directly.
_CX_LINE = re.compile(r"^CX\s+(\d+)\s+(\d+)")


def _read_complexo_map(dadger_path: Path) -> dict[int, list[int]]:
    """Parse the deck's ``CX`` register into ``{complexo_code: [component_codes]}``.

    The ``CX`` register (*acoplamento de usinas que representam complexos no
    NEWAVE com o DECOMP*) maps a NEWAVE **complexo** — an aggregated plant
    carrying a single set of future-cost coefficients in the boundary cuts — to
    the individual DECOMP plants that share it. The boundary cut header prices
    the complexo (which is absent from the DECOMP model), so
    :func:`~cobre_bridge.decomp.fcf.mapper.map_boundary_cuts` replicates its
    coefficients onto these components. Each ``CX <complexo> <component>`` line
    appends ``component`` to ``complexo``'s list; returns an empty map when the
    deck carries no ``CX`` register or the file is absent (the shared case's
    ``dadger`` parse validates the deck's presence upstream — this raw pass only
    reads the ``CX`` lines idecomp does not model).
    """
    complexo_map: dict[int, list[int]] = {}
    if not dadger_path.is_file():
        return complexo_map
    with dadger_path.open(encoding="latin-1") as handle:
        for line in handle:
            match = _CX_LINE.match(line)
            if match is not None:
                complexo_map.setdefault(int(match.group(1)), []).append(
                    int(match.group(2))
                )
    return complexo_map


def _find_mlt(deck_dir: Path) -> Path | None:
    """Locate the deck's ``mlt.dat`` (média de longo termo), case-insensitively.

    ``mlt.dat`` is not one of :class:`~cobre_bridge.decomp.files.DecompFiles`'
    resolved inputs (it feeds only the boundary FCF's inflow-lag mean fold, not
    the conversion), so it is discovered here directly. Returns ``None`` when the
    deck carries none.
    """
    for path in deck_dir.iterdir():
        if path.is_file() and path.name.lower() == "mlt.dat":
            return path
    return None


def _boundary_inflow_context(
    case: DecompCase,
    *,
    coupling_month: int,
) -> (
    tuple[EffectiveCadastro, list[OperativeStage], dict[int, tuple[float, ...]]] | None
):
    """The effective cadastro, calendar, and per-plant inflow-lag mean fold.

    Reads the deck's ``mlt.dat`` and builds the seasonal-mean fold vector
    (:func:`~cobre_bridge.decomp.inflow_mlt.build_incremental_mlt` incrementalises
    the natural MLT, :func:`~cobre_bridge.decomp.inflow_mlt.coupling_lag_means`
    aligns it to the cut's lag-depth axis at ``coupling_month``), returning it
    alongside the ``effective``/``calendar`` the recent-observation seed also
    needs — the two ship together (mean fold + raw seed), sharing this one
    cadastro build.

    Returns ``None`` when the deck carries no ``mlt.dat``: without the seasonal
    means the deviation fold cannot be applied, so the boundary FCF stays on the
    unseeded lags-0 approximation (a logged tracked gap) rather than a raw seed
    the RHS would fail to offset — over-draining the reservoirs.
    """
    mlt_path = _find_mlt(case.files.dadger.parent)
    if mlt_path is None:
        _LOG.warning(
            "no mlt.dat in %s; the boundary FCF's inflow-lag mean fold and the "
            "recent-observation seed are both skipped (the lag state stays at 0 "
            "-- a near-mean approximation, tracked gap), since seeding raw "
            "inflows without the seasonal-mean fold over-drains the reservoirs",
            case.files.dadger.parent,
        )
        return None
    effective, _ = build_effective_cadastro(case.dadger, case.hidr, case.calendar)
    incremental_mlt = build_incremental_mlt(
        Mlt.read(str(mlt_path)).valores, effective, case.id_map
    )
    means = coupling_lag_means(incremental_mlt, coupling_month)
    return effective, case.calendar, means


def _seed_recent_observations(
    writer: CaseWriter,
    case: DecompCase,
    effective: EffectiveCadastro,
    calendar: Sequence[OperativeStage],
    initial_conditions: dict,
) -> int:
    """Add ``recent_observations`` to the case's ``initial_conditions.json``.

    Seeds cobre's PAR inflow-lag accumulator with the deck's pre-study observed
    (already-incremental) inflows
    (:func:`~cobre_bridge.decomp.scenarios.convert_recent_observation_windows`),
    so the boundary cut's inflow-lag terms are evaluated against the real recent
    inflows rather than 0. Mutates the pipeline's own in-memory
    ``initial_conditions`` dict (the one already written to
    ``initial_conditions.json``) and writes it back through *writer* — rather
    than re-reading the file — so the raw seed and the RHS mean fold (which
    offsets it) are authored together and never ship apart. Returns the number of
    observation windows written (0 when the deck carries no observation tables).
    """
    windows = convert_recent_observation_windows(
        Vazoes.read(str(case.files.vazoes)), effective, case.id_map, calendar
    )
    initial_conditions["recent_observations"] = windows
    writer.write_json("initial_conditions.json", initial_conditions)
    return len(windows)


def _build_gnl_ring_plan(case_dir: Path, deck_files: DecompFiles) -> GnlRingPlan | None:
    """Read the deck's ``dadgnl`` and build the GNL ring plan, or ``None``.

    Deck-reading wrapper around :func:`_gnl_targets_from`: returns ``None``
    when the deck carries no ``dadgnl`` file at all, or when
    :func:`~cobre_bridge.decomp.converters.anticipated.read_gnl_model` reports the deck
    is GNL-off (no committed dispatch, the G6 gate) — reconciled with that
    reader's own gate, never re-derived here. Otherwise threads
    :func:`_post_horizon_start` into the resolved plan so
    ``fcf/mapper.py::_resolve_gnl_targets`` restricts placement to covered
    post-horizon lanes without itself reading any deck file.
    """
    if deck_files.dadgnl is None:
        return None
    model = read_gnl_model(Dadgnl.read(str(deck_files.dadgnl)))
    if model is None:
        return None
    thermals_path = case_dir / "system" / "thermals.json"
    with thermals_path.open(encoding="utf-8") as handle:
        thermals_doc = json.load(handle)
    plan = _gnl_targets_from(model, thermals_doc)
    if plan is None:
        return None
    return GnlRingPlan(plan.targets, post_horizon_start=_post_horizon_start(case_dir))


def _gnl_deviation_rows(
    cuts: BoundaryCuts, gnl_plan: GnlRingPlan
) -> list[tuple[int, int, float, float, float]]:
    """The pre-fan-out patamar spread carried into each live GNL ring group.

    Mirrors ``mapper.py::_resolve_gnl_targets``'s ``col(s, p, l)`` flat-column
    formula exactly — never the placement/drop logic itself, which stays the
    mapper's job — to recompute, from the raw ``pi_gnl`` coefficients, how
    far each active cut's per-patamar sensitivities deviate from the uniform
    rate cobre's hours-weighted fan-out reconstructs. For each unique
    ``(submercado, lag)`` pair named by ``gnl_plan.targets``, returns
    ``(submercado, lag, carried_sum, relative_spread, absolute_spread)`` from
    whichever active record maximises the relative spread
    ``(max_p c_p - min_p c_p) / |sum_p c_p|`` (guarded to ``0.0`` when
    ``|sum_p c_p| <= 1e-9``) — a worst-case snapshot, not an aggregate across
    records; ``absolute_spread`` (``max_p c_p - min_p c_p``) is that same
    selected record's un-normalised spread, so
    a caller can report it alongside the relative figure without the
    near-zero-Σ inflation the ratio alone is prone to. Skips a group whose
    ``(submercado, lag)`` falls outside ``cuts``' own ``pi_gnl`` shape (the
    mapper already records that as a target-side ``GnlDroppedTerm``).

    Returns rows sorted ascending by ``(submercado, lag)``; empty when
    ``cuts`` has no active records, or the boundary carries no GNL block
    (``lag_maximo_gnl == 0`` or an empty/misshapen ``pi_gnl``).
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

    groups = sorted({(target.submercado, target.nl_lag) for target in gnl_plan.targets})
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
    """Surface the importer's documented, accepted approximations.

    Always emits ``boundary-fcf-cut-family-summary`` — the importer always
    authors cuts, so the family triage (built from
    :func:`~cobre_bridge.decomp.fcf.cortes.summarize_cut_families`, never
    re-implemented here) is always informative. Additionally emits
    ``boundary-fcf-source-only-plants-dropped`` when ``mapping.dropped`` is
    non-empty — a source-only plant has no target ``HydroStorage`` slot, so
    its storage/lag terms are omitted (D3: dropped, never folded). When
    ``gnl_plan`` is given and the boundary carries a GNL block
    (``cuts.header.lag_maximo_gnl > 0``), additionally emits
    ``boundary-fcf-gnl-anticipated-deviation`` — the per-``(submercado, lag)``
    pre-fan-out patamar spread the mapper's chain-rule sum collapses (see
    :func:`_gnl_deviation_rows`), headlined as a relative/absolute spread
    pair (the relative figure excludes a near-zero-Σ group per
    ``_GNL_DEVIATION_REL_FLOOR``) plus a
    dropped-coverage count read verbatim from ``mapping.gnl_dropped``
    (source-submercado drops plus non-covered
    dated-slot drops); ``gnl_plan=None`` (the default) gates it off
    entirely, so 2-arg callers are unchanged.

    Pure side effect via :func:`cobre_bridge.core.diagnostics.emit`: reads
    ``cuts``/``mapping``/``gnl_plan`` but does not alter any of them, so it
    can run before the checkpoint is written without changing the checkpoint
    bytes or the importer's return value. Mirrors ``decomp/pipeline.py``'s
    gated-INFO-``Diagnostic`` idiom; relies on the ambient-sink/log-fallback
    contract of ``diagnostics.emit`` rather than opening its own
    ``dx.collect()`` sink.
    """
    summary = summarize_cut_families(cuts)
    dx.emit(
        dx.Diagnostic(
            code="boundary-fcf-cut-family-summary",
            severity=dx.Severity.INFO,
            category="Boundary FCF",
            title=f"Boundary FCF authors {summary.n_active_cuts} cut(s)",
            # `lag_nonzero_by_depth` is
            # the one fact not already in this string; every other figure a
            # `notes` bullet used to restate (n_active_cuts, storage_nonzero_
            # plants, rhs_min/max) is already here, so `notes` is dropped
            # rather than duplicating them. Full-precision RHS belongs in
            # `--diagnostics-json`, not a duplicate human bullet.
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
                    "from the boundary FCF"
                ),
                summary=(
                    f"{len(mapping.dropped)} plant(s) present in the source "
                    "model's boundary cuts have no target HydroStorage slot "
                    "in the converted case; their storage and inflow-lag "
                    "terms are omitted from every authored cut, never "
                    "folded into a neighbouring plant"
                ),
                table=dx.DiagnosticTable(
                    columns=["Plant code", "β (pi_varm)"],
                    rows=[
                        [term.plant_code, round(term.beta, 6)]
                        for term in mapping.dropped
                    ],
                    justify=["right", "right"],
                ),
            ),
            logger=_LOG,
        )

    if gnl_plan is not None and cuts.header.lag_maximo_gnl > 0:
        rows = _gnl_deviation_rows(cuts, gnl_plan)
        # "Dropped" is every GNL term that reached no *covered* target: a
        # source submercado with no live thermal at all (`thermal_id is
        # None`), or a target's dated slot that fell in-study, priced by
        # the committed window instead of the ring
        # (`mapper.py::GnlDroppedTerm.reason` names it) — read straight
        # from `mapping.gnl_dropped`, never recomputing the covered/
        # uncovered split independently here. A já-comandada (class-4)
        # slot is excised from the ring entirely, so it never appears in
        # `gnl_dropped` and needs no arm of its own.
        dropped_coverage_terms = [
            term
            for term in mapping.gnl_dropped
            if term.thermal_id is None or "in-study committed window" in term.reason
        ]
        # See `_GNL_DEVIATION_REL_FLOOR`'s docstring for the relative-headline
        # filter. Guard the degenerate case where every group's |Σ| is itself
        # ~0 (no group carries any real weight at all): no group qualifies for
        # a meaningful relative headline, which then reports "n/a" rather than
        # a misleading 0.
        max_abs_sum = max((abs(row[2]) for row in rows), default=0.0)
        if max_abs_sum <= 1e-12:
            headline_rows: list[tuple[int, int, float, float, float]] = []
        else:
            headline_rows = [
                row
                for row in rows
                if abs(row[2]) >= _GNL_DEVIATION_REL_FLOOR * max_abs_sum
            ]
        relative_headline = (
            f"{max(row[3] for row in headline_rows):.4g}" if headline_rows else "n/a"
        )
        max_absolute_spread = max((row[4] for row in rows), default=0.0)
        dx.emit(
            dx.Diagnostic(
                code="boundary-fcf-gnl-anticipated-deviation",
                severity=dx.Severity.INFO,
                category="Boundary FCF",
                title="GNL anticipated ring carries a per-patamar sum",
                summary=(
                    f"max pre-fan-out patamar spread {relative_headline} "
                    f"relative / {max_absolute_spread:.4g} absolute across "
                    f"{len(rows)} live GNL ring target group(s) (a group "
                    f"carrying less than {_GNL_DEVIATION_REL_FLOOR:g} of the "
                    "panel's max |Σ| is excluded from the relative headline); "
                    f"{len(dropped_coverage_terms)} GNL term(s) dropped for "
                    "no covered target (no live thermal in that submercado, "
                    "or an in-study delivery priced by the committed window, "
                    "not the anticipated ring)"
                ),
                table=dx.DiagnosticTable(
                    columns=[
                        "Submercado",
                        "Lag",
                        "Σ pi_gnl (carried)",
                        "Patamar spread",
                    ],
                    rows=[
                        [submercado, lag, round(carried_sum, 6), round(spread, 6)]
                        for submercado, lag, carried_sum, spread, _abs_spread in rows
                    ],
                    justify=["right", "right", "right", "right"],
                ),
            ),
            logger=_LOG,
        )


def _patch_policy_boundary(writer: CaseWriter, config: dict) -> None:
    """Set ``["policy"]["boundary"]`` in ``config.json``, preserving the rest.

    The block carries only ``path``: cobre's boundary loader selects the
    source pool by calendar date (the pool whose ``priced_state_date`` equals
    the loading study's own boundary date), so no stage/pool index is written.
    ``policy.boundary.source_stage`` is removed from cobre's config contract
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
       ``cobre.run.run`` pass on ``case_dir`` (checkpoint under ``work_dir``,
       the case is never mutated) to read back its terminal state-vector
       layout.
    4. Builds the deck's GNL ring plan (:func:`_build_gnl_ring_plan`) and the
       inflow-lag mean fold (:func:`_boundary_inflow_context`: the
       incrementalised ``mlt.dat`` aligned to the cut's lag-depth axis at the
       coupling month), then maps every boundary cut onto that layout: storage
       terms by plant code, inflow-lag terms by calendar-month lag depth (with
       each cut's RHS reduced by ``Σ lag_coef · mu`` so the loaded cut prices
       the raw lag state as the *deviation* from the seasonal mean —
       ``fcf/mapper.py``'s ``inflow_lag_means``), and ``AnticipatedThermalState``
       GNL-ring terms via the plan's per-block hours-weighted patamar sum;
       ``HydroTransitBucket`` slots are left at coefficient 0
       regardless.
    5. Surfaces the mapping's documented approximations as ``Diagnostic``s
       (:func:`_emit_import_diagnostics`): the always-on cut-family triage,
       the D3-dropped source-only plants (gated on non-empty), and — when the
       deck carries a GNL block — the per-``(submercado, lag)`` deviation the
       ring's chain-rule sum collapses.
    6. Assembles and writes ``case_dir/boundary/{manifest.bin,
       cuts/<pool>.bin, basis/}``, patches ``case_dir/config.json``'s
       ``["policy"]["boundary"]`` to point at it, and — when the deck carries an
       ``mlt.dat`` (so the mean fold above was applied) — patches
       ``case_dir/initial_conditions.json`` with the pre-study
       ``recent_observations`` seed (:func:`_seed_recent_observations`), the raw
       inflow-lag values the folded RHS is built to offset. Fold and seed ship
       together or not at all.

    Returns the ``case_dir/boundary`` path, or ``None`` on the no-cut-files
    no-op.

    Raises
    ------
    RuntimeError
        Propagated verbatim from ``fcf.bootstrap``'s ``ensure_writer_binding``
        (writer binding missing) or ``bootstrap_terminal_manifest`` (the
        in-process ``cobre.run.run`` failed or its checkpoint was malformed).
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

    # A real, non-dry-run writer: the importer runs a real ``cobre`` pass and
    # never runs under ``--dry-run``.
    writer = CaseWriter(case_dir)

    cortesh = Cortesh.read(str(case.files.cortesh))
    cuts = read_cortes(case.files.cortes, cortesh, boundary_stage=None)
    # `BoundaryCuts.boundary_stage` is typed `int`, but a single-stage export's
    # derived value inherits `numpy.int32` from `cortesh.ano_inicio_estudo`'s
    # own numpy dtype (confirmed against this deck) — narrow to a plain `int`
    # here, at the boundary between the numpy-sourced reader and the
    # JSON/cobre-FFI payloads this function builds below.
    boundary_stage = int(cuts.boundary_stage)

    ensure_writer_binding()

    # No explicit inflow-lag depth is supplied to the bootstrap: cobre sizes the
    # HydroInflowLag state block from the case's own PAR(p) model order (the same
    # autoregressive structure the boundary cuts' pi_qafl terms derive from), so
    # the terminal manifest already carries the slots the mapper places those
    # terms onto. The former state_space.inflow_lag_depth override was redundant
    # with that sizing and is rejected by cobre >= 0.14.
    manifest = bootstrap_terminal_manifest(case_dir, work_dir=work_dir)
    gnl_plan = _build_gnl_ring_plan(case_dir, case.files)
    # Inflow-lag mean fold + recent-observation seed (built together, shipped
    # together — see `_boundary_inflow_context`). The coupling month is the cut
    # stage's own calendar month (`boundary_stage` is calendar-anchored as
    # `(Y - Y0)*12 + M`, so `(boundary_stage - 1) % 12 + 1 == M`); the fold
    # aligns each `pi_qafl` lag depth to `coupling_month - depth`. `None` (no
    # mlt.dat) leaves the lag state at 0 — no fold, no seed.
    coupling_month = ((boundary_stage - 1) % 12) + 1
    inflow_context = _boundary_inflow_context(case, coupling_month=coupling_month)
    inflow_lag_means = inflow_context[2] if inflow_context is not None else None
    # Read stages.json's per-block hours once; cost_unit_hours (the
    # intercept/storage/inflow-lag scale) is this same vector's sum, never a
    # second stages.json read.
    coupling_block_hours = _final_stage_block_hours(case_dir)
    cost_unit_hours = math.fsum(coupling_block_hours)
    # CX register (complexo -> DECOMP components): the header prices a NEWAVE
    # complexo absent from the DECOMP model, so its coefficients replicate onto
    # the individual plants that share it (see map_boundary_cuts) rather than
    # being dropped.
    complexo_components = _read_complexo_map(case.files.dadger)
    # The DECOMP case has no PAR(p) model, so the bootstrap manifest carries no
    # HydroInflowLag slots; the deepest lag the boundary cuts reference is
    # declared to the writer, which reserves the canonical slots (cobre-side
    # design B — see the module's cortes.required_inflow_lag_depth). The mapper
    # then emits the lag terms keyed by hydro rather than into the (absent) slots.
    inflow_lag_depth = required_inflow_lag_depth(summarize_cut_families(cuts))
    mapping = map_boundary_cuts(
        cuts,
        manifest,
        case.id_map,
        cost_unit_hours=cost_unit_hours,
        gnl_plan=gnl_plan,
        coupling_block_hours=coupling_block_hours,
        inflow_lag_means=inflow_lag_means,
        complexo_components=complexo_components,
        inflow_lag_depth=inflow_lag_depth,
    )
    _LOG.info(
        "scaling boundary FCF coefficients to cobre cost units over the "
        "coupling stage's %.0f h (the source's per-hour cut rate integrated "
        "over the coupling period; inflow-lag additionally x C_M3S2HM3 for "
        "cobre's m3/s lag state)",
        cost_unit_hours,
    )
    _emit_import_diagnostics(cuts, mapping, gnl_plan)

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

    # Seed the pre-study inflow-lag state and record the mean fold — both gated
    # on the same mlt.dat presence as the fold above, so the raw seed never
    # ships without the RHS mean that offsets it (`_boundary_inflow_context`).
    if inflow_context is not None:
        effective, calendar, _ = inflow_context
        n_windows = _seed_recent_observations(
            writer, case, effective, calendar, initial_conditions
        )
        _LOG.info(
            "boundary FCF inflow-lag coupling: folded the seasonal-mean (MLT) "
            "deviation into %d cut RHS(es) and seeded %d recent-observation "
            "window(s) at coupling month %d, so the loaded cut prices the raw "
            "inflow-lag state as the deviation Q - mu",
            len(mapping.cuts),
            n_windows,
            coupling_month,
        )

    return boundary_dir
