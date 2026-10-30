"""Read the source model's GNL (fuel-constrained) anticipated dispatch.

The source model declares its GNL thermals entirely in ``dadgnl`` — a separate file from
the ``CT`` thermal registry the main thermal converter reads — so these plants are
invisible to ``decomp/converters/thermal.py`` and must be modelled here.
The **read/model layer** (:func:`read_gnl_model` and its helpers) is pure: it
turns ``dadgnl`` into a structured commitment model and does nothing else — no
``novomodelo`` import, no filesystem writes, no clamping, no decision about lead
declaration or ring placement; it returns the true committed values as data. The
**emission layer** (:func:`convert_gnl`) turns that model into novomodelo's
anticipated-dispatch inputs and owns the bounds policy the reader defers: it
clamps each committed MW into the plant's ``tg`` capability at its own delivery
stage, warning (via the module logger) on an out-of-range value, so the
converted case never pins a delivery novomodelo would reject.

``dadgnl`` has three register families:

* ``tg`` — the GNL thermal registry, per ``(codigo_usina, estagio)``:
  ``codigo_submercado``, ``nome``, and per-block ``cvu`` (fuel cost, $/MWh),
  ``disponibilidade`` (max MW), ``inflexibilidade`` (min MW). Like ``CT`` it is
  sparse by stage — stage 1 is mandatory and a later stage inherits the last
  declared record — and is densified by the same routine
  (:func:`~novomodelo_bridge.decomp.converters.thermal.dense_stage_records`). Fixed
  3-block shape, so ``tg(df=True)`` is well-formed.
* ``gl`` — the committed weekly dispatch: one register per ``(codigo_usina,
  estagio)`` carrying ``data_inicio`` (the delivery-stage start date, a
  ``ddmmyyyy`` string), a per-block ``duracao`` list, and a per-block ``geracao``
  (committed MW) list. Block counts vary across weekly stages, so ``gl(df=True)``
  is **not** usable (the ragged per-block lists raise "All arrays must be of the
  same length"); the registers are iterated directly instead.
* ``gs`` — the weeks-per-month calendar map (``mes`` → ``semanas``).
* ``nl`` — the per-plant dispatch-anticipation lag in whole months
  (``codigo_usina`` → ``lag``): a plant's dispatch is decided ``lag`` months
  ahead of its delivery, and this is what sizes its physical ``lead_time_hours``
  — or, in a study of monthly stages only, where each ``gl`` register commits
  one month, its ``lead_stages`` (:func:`_convert_monthly_gnl`).

A commitment's delivery date is recorded per stage so the emission track can
place it: a ``gl`` week ``gs`` books to the study is folded onto the study
stages as a left-boundary ``past_anticipated_commitment``, and one it books
past the study becomes a fixed já-comandada ``past_anticipated_commitment``
too (:func:`classify_gnl_windows`). Any
remaining post-study stage not covered by a já-comandada window is priced via
``post_study_stages.json`` ``thermal_bounds`` instead — the deck does not
declare a synthesised free lane past what ``gl`` actually commits.

Committed MW per stage is the block-duration-weighted mean of ``geracao`` over
that stage's own ``duracao`` blocks (``Σ_b duracao_b·geracao_b / Σ_b duracao_b``),
self-normalising so the committed MWh is preserved exactly regardless of block
count.

Post-horizon anticipated delivery — the já-comandada (class-4) windows and the
signaled (class-3) ``thermal_bounds`` this module emits — is a feature of the
source model with no counterpart in the sibling conversion track: its
committed-dispatch reader
(:func:`novomodelo_bridge.newave.converters.anticipated.read_anticipated_dispatch`)
truncates any lag past the study horizon rather than surfacing one past it, so
this asymmetry is registered, not an oversight or a missing port.
"""

from __future__ import annotations

import calendar as _calendar
import dataclasses
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING, NamedTuple

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.decomp.converters.thermal import dense_stage_records
from novomodelo_bridge.decomp.temporal import hours_weighted, weekly_stage_count

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import pandas as pd
    from idecomp.decomp import Dadgnl
    from idecomp.decomp.modelos.dadgnl import GL

    from novomodelo_bridge.decomp.temporal import OperativeStage

_LOG = logging.getLogger(__name__)

_NONZERO_TOLERANCE = 1e-9
_HOURS_PER_OPERATIVE_WEEK = 168  # 7 days x 24 h; the study/post-study grid step


@dataclass(frozen=True)
class GnlThermal:
    """One GNL plant's identity in the ``tg`` registry. Its per-stage cost and
    capability stay in :attr:`GnlCommitmentModel.tg`, densified against the
    operative calendar at emission."""

    code: int
    name: str
    submarket_code: int


@dataclass(frozen=True)
class GnlStageCommitment:
    """One ``gl`` register: a committed MW at one delivery stage, with its date.

    ``start_date`` is the parsed ``data_inicio`` (the delivery stage's start);
    the emission track compares it against the study horizon to route the
    commitment to the left or right temporal boundary. ``committed_mw`` is the
    block-duration-weighted mean of that register's ``geracao``, ``block_mw`` the
    ``geracao`` per patamar itself and ``block_hours`` its ``duracao``
    (:func:`gnl_block_commitments`); ``hours`` is the stage span (sum of its
    ``duracao`` blocks), its weight when folded onto a study stage.
    """

    estagio: int
    start_date: date
    committed_mw: float
    hours: float
    block_mw: tuple[float, ...] = ()
    block_hours: tuple[float, ...] = ()


@dataclass(frozen=True)
class GnlCommitment:
    """One plant's committed dispatch across every ``gl`` stage it declares.

    ``stages`` is ascending by ``estagio`` and may be empty for a plant present
    in ``tg`` but absent from ``gl`` (registry-only, no committed dispatch —
    never dropped, never fabricated).
    """

    code: int
    stages: tuple[GnlStageCommitment, ...]


@dataclass(frozen=True)
class GnlCommitmentModel:
    """The GNL registry, its committed dispatch, and the weeks-per-month map.

    ``weeks_per_month`` is the ``gs`` block's ``{month_index: weeks}`` map (the
    number of operative weeks in each month, 1-based month index ascending);
    the study's own months count only the weeks left in the study, so their
    sum falls short of the ``gl`` run whenever ``gl`` reaches into the month
    after the study.

    ``nl_lag_months`` is the ``nl`` block's per-plant dispatch-anticipation lag
    (``{codigo_usina: months}``): the number of months by which a GNL plant's
    dispatch is decided ahead of its delivery (the LNG supply lead time).
    Every anticipated plant shares one lead, so on a study with operative
    weeks this drives only the uniform-lag consistency check
    (:func:`_warn_on_nonuniform_lag`) — a deck mixing lags is surfaced, not
    silently averaged.

    ``tg`` is every declared ``tg`` record, one per ``(codigo_usina, estagio)``.
    """

    thermals: tuple[GnlThermal, ...]
    commitments: dict[int, GnlCommitment]
    weeks_per_month: dict[int, int]
    nl_lag_months: dict[int, int]
    tg: pd.DataFrame


def _as_floats(value: object) -> list[float]:
    """Coerce a register field to a list of floats (scalar → 1-list, None → [])."""
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [float(v) for v in value]
    return [float(value)]  # type: ignore[arg-type]


def _gl_registers(dadgnl: Dadgnl) -> list[GL]:
    """Every ``gl`` register of ``dadgnl``, in deck order."""
    registers = dadgnl.gl()
    if registers is None:
        return []
    # idecomp returns a lone register bare, not as a one-item list.
    return registers if isinstance(registers, list) else [registers]


def _block_weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    """``Σ w·v / Σ w`` over aligned blocks; uniform mean when weights are unusable.

    Falls back to the plain mean of ``values`` when ``weights`` is empty, sums to
    zero, or has a different length than ``values`` — the value must survive a
    missing/degenerate block-hours vector rather than vanish.
    """
    if not values:
        return 0.0
    if len(weights) == len(values):
        total_w = sum(weights)
        if total_w > 0.0:
            return sum(v * w for v, w in zip(values, weights, strict=True)) / total_w
    return sum(values) / len(values)


def _parse_data_inicio(raw: object) -> date:
    """Parse a ``gl`` register's ``data_inicio`` into a :class:`datetime.date`.

    Accepts an already-parsed ``date``, or a ``ddmmyyyy`` string / integer (the
    on-file form, e.g. ``"14032026"`` → 2026-03-14). Integers are zero-padded to
    eight digits first, so a dropped leading-zero day (``4042026``) parses as
    ``2026-04-04``.
    """
    if isinstance(raw, date):
        return raw
    text = f"{int(raw):08d}" if isinstance(raw, int) else str(raw).strip()
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"unparseable gl data_inicio {raw!r} (expected ddmmyyyy)")
    day, month, year = int(text[0:2]), int(text[2:4]), int(text[4:8])
    return date(year, month, day)


def is_gnl_enabled(dadgnl: Dadgnl | None) -> bool:
    """Whether ``dadgnl`` declares any GNL plant.

    ``True`` iff ``dadgnl`` is present and its ``tg`` registry lists a plant.
    The committed ``geracao`` does not gate it: a plant whose ``gl`` commits
    0 MW, as in the first revisions of a month, is still modelled by the source
    model, which holds it at 0 MW where committed and decides the later months
    within its ``tg`` capability.
    """
    if dadgnl is None:
        return False
    tg = dadgnl.tg(df=True)
    return tg is not None and not tg.empty


def read_gnl_model(dadgnl: Dadgnl) -> GnlCommitmentModel | None:
    """Read ``dadgnl`` into a :class:`GnlCommitmentModel`, or ``None`` without
    GNL plants.

    Returns ``None`` when :func:`is_gnl_enabled` is ``False``. Otherwise builds
    the ``tg`` plants (ascending by code) and their declared records, the ``gl``
    commitments (keyed by code, ascending by ``estagio``, each stage carrying its
    parsed delivery date and block-weighted committed MW), and the ``gs``
    weeks-per-month map.

    Raises
    ------
    ValueError
        If a ``gl`` register names a ``codigo_usina`` with no ``tg`` registry
        entry (a committed dispatch for an unknown plant — a malformed deck).
    """
    if not is_gnl_enabled(dadgnl):
        return None

    tg = dadgnl.tg(df=True)
    first_records = tg.sort_values(["codigo_usina", "estagio"]).drop_duplicates(
        "codigo_usina", keep="first"
    )
    thermals = tuple(
        GnlThermal(
            code=int(row["codigo_usina"]),
            name=str(row["nome"]).strip(),
            submarket_code=int(row["codigo_submercado"]),
        )
        for _, row in first_records.iterrows()
    )
    known_codes = {t.code for t in thermals}

    commitments = _read_gl_commitments(dadgnl, known_codes)
    # Registry-only plants (in tg, absent from gl) still get an empty commitment.
    for thermal in thermals:
        commitments.setdefault(
            thermal.code, GnlCommitment(code=thermal.code, stages=())
        )

    return GnlCommitmentModel(
        thermals=thermals,
        commitments=commitments,
        weeks_per_month=_read_weeks_per_month(dadgnl),
        nl_lag_months=_read_nl_lags(dadgnl),
        tg=tg,
    )


def _read_gl_commitments(
    dadgnl: Dadgnl, known_codes: set[int]
) -> dict[int, GnlCommitment]:
    """Build ``{code: GnlCommitment}`` by iterating ``gl`` registers.

    Iterates registers (never ``gl(df=True)`` — the ragged per-block lists make
    it unusable). Each register contributes one :class:`GnlStageCommitment` with
    its parsed date and block-weighted committed MW.
    """
    by_code: dict[int, list[GnlStageCommitment]] = {}
    for register in _gl_registers(dadgnl):
        code = int(register.codigo_usina)
        if code not in known_codes:
            raise ValueError(
                f"gl declares a committed dispatch for code {code} with no tg "
                "registry entry (a dispatch for an unknown plant)"
            )
        geracao = _as_floats(register.geracao)
        duracao = _as_floats(register.duracao)
        by_code.setdefault(code, []).append(
            GnlStageCommitment(
                estagio=int(register.estagio),
                start_date=_parse_data_inicio(register.data_inicio),
                committed_mw=_block_weighted_mean(geracao, duracao),
                hours=sum(duracao),
                block_mw=tuple(geracao),
                block_hours=tuple(duracao),
            )
        )
    return {
        code: GnlCommitment(
            code=code, stages=tuple(sorted(stages, key=lambda s: s.estagio))
        )
        for code, stages in by_code.items()
    }


def gnl_block_commitments(
    model: GnlCommitmentModel, calendar: Sequence[OperativeStage]
) -> dict[int, list[tuple[float, ...] | None]]:
    """Each GNL plant's committed MW per study stage and block, from ``gl``.

    A stage's block value is the ``duracao``-weighted mean of that block's
    ``geracao`` over every study ``gl`` week (:func:`_split_at_study_end`)
    starting inside the stage — one week for a weekly stage, every week of the
    month for a monthly one, the same stage assignment :func:`convert_gnl`
    folds its per-stage commitment with.
    A stage no ``gl`` week starts in maps to ``None``: the plant's dispatch
    there is not commanded.

    Raises
    ------
    ValueError
        If a ``gl`` week inside a stage declares a different block count than
        the stage itself.
    """
    study_week_count = _study_gl_week_count(model.weeks_per_month, calendar)
    horizon_end = calendar[-1].end_date
    commitments: dict[int, list[tuple[float, ...] | None]] = {}
    for thermal in model.thermals:
        stages, _ = _split_at_study_end(
            thermal.code,
            model.commitments[thermal.code].stages,
            study_week_count,
            horizon_end,
        )
        per_stage: list[tuple[float, ...] | None] = []
        for stage in calendar:
            weeks = [
                c for c in stages if stage.start_date <= c.start_date < stage.end_date
            ]
            if not weeks:
                per_stage.append(None)
                continue
            n_blocks = len(stage.block_hours)
            for week in weeks:
                if len(week.block_mw) != n_blocks:
                    raise ValueError(
                        f"gl for GNL plant {thermal.code} at "
                        f"{week.start_date.isoformat()} declares "
                        f"{len(week.block_mw)} block(s); stage {stage.index} "
                        f"has {n_blocks}"
                    )
            per_stage.append(
                tuple(
                    _block_weighted_mean(
                        [week.block_mw[b] for week in weeks],
                        [week.block_hours[b] for week in weeks],
                    )
                    for b in range(n_blocks)
                )
            )
        commitments[thermal.code] = per_stage
    return commitments


def _read_weeks_per_month(dadgnl: Dadgnl) -> dict[int, int]:
    """The ``gs`` weeks-per-month map ``{mes: semanas}`` (empty when absent)."""
    frame = dadgnl.gs(df=True)
    if frame is None or frame.empty:
        return {}
    return {int(row["mes"]): int(row["semanas"]) for _, row in frame.iterrows()}


def operative_week_months(
    weeks_per_month: Mapping[int, int], calendar: Sequence[OperativeStage]
) -> dict[date, tuple[int, int]]:
    """The ``(year, month)`` the source model books each ``gs`` interval to,
    keyed by the interval's start.

    Walks the ``gs`` counts in ascending month order from the study start, the
    first count being the study-start month's. The intervals are operative
    weeks, and a week belongs to the month ``gs`` books it to even when it
    ends in the next one (mar-26-rv2: the week starting 2026-05-30 is a May
    week); a study of monthly stages only has one interval per month, the
    month itself. Empty without a ``gs`` block.
    """
    study_start = calendar[0].start_date
    monthly_only = weekly_stage_count(calendar) == 0
    weeks: dict[date, tuple[int, int]] = {}
    week_start = study_start
    year, month = study_start.year, study_start.month
    for key in sorted(weeks_per_month):
        if monthly_only:
            weeks[date(year, month, 1)] = (year, month)
        else:
            for _ in range(weeks_per_month[key]):
                weeks[week_start] = (year, month)
                week_start += _ONE_OPERATIVE_WEEK
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return weeks


def _read_nl_lags(dadgnl: Dadgnl) -> dict[int, int]:
    """The ``nl`` dispatch-anticipation lags ``{codigo_usina: months}``.

    Each ``nl`` register carries ``codigo_usina``, ``codigo_submercado`` and
    ``lag`` (whole months of dispatch anticipation). Unlike ``gl``, ``nl`` is a
    fixed-shape register, so ``nl(df=True)`` is well-formed. Empty when the deck
    declares no ``nl`` block.
    """
    frame = dadgnl.nl(df=True)
    if frame is None or frame.empty:
        return {}
    return {int(row["codigo_usina"]): int(row["lag"]) for _, row in frame.iterrows()}


def fill_undeclared_block_hours(
    model: GnlCommitmentModel,
    calendar: Sequence[OperativeStage],
    patamar_fractions: Mapping[tuple[int, int], Sequence[float]] | None,
) -> GnlCommitmentModel:
    """*model* with patamar hours on every ``gl`` week that declares no ``duracao``.

    The source model leaves ``duracao`` blank on the weeks past its horizon and
    computes them itself. Such a week takes the patamar
    shares of the month ``gs`` books it to (:func:`operative_week_months`) from
    *patamar_fractions* (``(year, month)`` → shares summing to 1, the boundary
    FCF header's — the same weights its cuts price a commanded week with), times
    one operative week, and its ``committed_mw`` is re-weighted with them. When
    the shares are missing or do not cover that month, the last study stage's
    patamar hours stand in, with a warning; a week whose ``geracao`` differs
    across patamares is warned on as well, since only there does the weighting
    move its MW.
    """
    names = {t.code: t.name for t in model.thermals}
    week_months = operative_week_months(model.weeks_per_month, calendar)
    last_stage_hours = calendar[-1].block_hours
    fallback_rows: list[list[object]] = []
    nonuniform_rows: list[list[object]] = []
    commitments: dict[int, GnlCommitment] = {}
    for code, commitment in model.commitments.items():
        stages: list[GnlStageCommitment] = []
        for week in commitment.stages:
            if not week.block_mw or len(week.block_hours) == len(week.block_mw):
                stages.append(week)
                continue
            year, month = week_months.get(
                week.start_date, (week.start_date.year, week.start_date.month)
            )
            shares = (patamar_fractions or {}).get((year, month))
            source = "boundary FCF patamar shares"
            if shares is None or len(shares) != len(week.block_mw):
                shares = [h / sum(last_stage_hours) for h in last_stage_hours]
                source = "last study stage patamar hours"
                if len(shares) != len(week.block_mw):
                    shares = [1.0 / len(week.block_mw)] * len(week.block_mw)
                    source = "equal patamar weights"
                fallback_rows.append(
                    [
                        code,
                        names.get(code, ""),
                        week.start_date.isoformat(),
                        f"{year}-{month:02d}",
                        source,
                    ]
                )
            block_hours = tuple(s * _HOURS_PER_OPERATIVE_WEEK for s in shares)
            committed_mw = _block_weighted_mean(week.block_mw, block_hours)
            if max(week.block_mw) - min(week.block_mw) > _NONZERO_TOLERANCE:
                nonuniform_rows.append(
                    [
                        code,
                        names.get(code, ""),
                        week.start_date.isoformat(),
                        " / ".join(f"{mw:g}" for mw in week.block_mw),
                        round(committed_mw, 3),
                        source,
                    ]
                )
            stages.append(
                dataclasses.replace(
                    week,
                    committed_mw=committed_mw,
                    hours=sum(block_hours),
                    block_hours=block_hours,
                )
            )
        commitments[code] = dataclasses.replace(commitment, stages=tuple(stages))

    if fallback_rows:
        dx.emit(
            dx.Diagnostic(
                code="decomp-gnl-block-hours-fallback",
                severity=dx.Severity.WARNING,
                category="GNL",
                title=(
                    f"{len(fallback_rows)} GNL week(s) past the horizon weighted "
                    "without their month's patamar shares"
                ),
                summary=(
                    "These committed GNL weeks declare no patamar duration and the "
                    "boundary FCF header (cortesh) is absent or does not cover "
                    "their month, so their patamares are weighted by the last "
                    "study stage's patamar hours (equal weights when its patamar "
                    "count differs) instead of that month's shares"
                ),
                table=dx.DiagnosticTable(
                    columns=["Code", "Name", "Week start", "Month", "Weights"],
                    rows=fallback_rows,
                    justify=["right", "left", "left", "left", "left"],
                ),
            ),
            logger=_LOG,
        )
    if nonuniform_rows:
        dx.emit(
            dx.Diagnostic(
                code="decomp-gnl-undeclared-block-hours-nonuniform",
                severity=dx.Severity.WARNING,
                category="GNL",
                title=(
                    f"{len(nonuniform_rows)} GNL week(s) past the horizon commit "
                    "different MW per patamar"
                ),
                summary=(
                    "These committed GNL weeks declare no patamar duration and "
                    "commit different MW per patamar, so the single MW rate "
                    "converted for each depends on the patamar weights used"
                ),
                table=dx.DiagnosticTable(
                    columns=[
                        "Code",
                        "Name",
                        "Week start",
                        "MW per patamar",
                        "Converted MW",
                        "Weights",
                    ],
                    rows=nonuniform_rows,
                    justify=["right", "left", "left", "left", "right", "left"],
                ),
            ),
            logger=_LOG,
        )
    return dataclasses.replace(model, commitments=commitments)


_ONE_OPERATIVE_WEEK = timedelta(hours=_HOURS_PER_OPERATIVE_WEEK)


def _study_gl_week_count(
    weeks_per_month: Mapping[int, int], calendar: Sequence[OperativeStage]
) -> int | None:
    """How many leading ``gl`` weeks ``gs`` books to the study's own months.

    The study's months are the operative weeks' month (when there are any) and
    one per monthly stage, and ``gl`` numbers its weeks (``estagio``) in ``gs``
    order. ``None`` without a ``gs`` block.
    """
    if not weeks_per_month:
        return None
    n_weekly = weekly_stage_count(calendar)
    n_months = len(calendar) - n_weekly + (1 if n_weekly else 0)
    return sum(weeks_per_month[m] for m in sorted(weeks_per_month)[:n_months])


def _split_at_study_end(
    code: int,
    stages: Sequence[GnlStageCommitment],
    study_week_count: int | None,
    horizon_end: date,
) -> tuple[tuple[GnlStageCommitment, ...], tuple[GnlStageCommitment, ...]]:
    """A plant's ``gl`` weeks split into the study's and those after it.

    By ``gs`` position (:func:`_study_gl_week_count`), never by date: the first
    week of the month after the study starts on the Saturday grid, which can
    fall before the horizon end (the study's monthly stage ends on the 1st), yet
    the source model books it — and leaves its ``duracao`` blank — as a week
    past its horizon. Without ``gs`` the split falls back to ``start_date <
    horizon_end``.

    Such a week also lands in the study as its head ``[start_date,
    horizon_end)``: same ``geracao``, ``hours`` and ``block_hours`` cut to the
    head's share of the operative week, so the study's last stage weighs it by
    the hours it actually covers.

    Raises
    ------
    ValueError
        If ``gs`` books to the study a week starting at or after the horizon
        end, or past the study a week ending at or before it.
    """
    study: list[GnlStageCommitment] = []
    after: list[GnlStageCommitment] = []
    for week in stages:
        in_study = (
            week.start_date < horizon_end
            if study_week_count is None
            else week.estagio <= study_week_count
        )
        if in_study and week.start_date >= horizon_end:
            raise ValueError(
                f"gs books gl week {week.estagio} of GNL plant {code} "
                f"({week.start_date.isoformat()}) to the study, but it starts at "
                f"or after the study horizon end {horizon_end.isoformat()}"
            )
        if not in_study and week.start_date + _ONE_OPERATIVE_WEEK <= horizon_end:
            raise ValueError(
                f"gs books gl week {week.estagio} of GNL plant {code} "
                f"({week.start_date.isoformat()}) past the study, but it ends at "
                f"or before the study horizon end {horizon_end.isoformat()}"
            )
        (study if in_study else after).append(week)
        if not in_study and week.start_date < horizon_end:
            study.append(_head_before(week, horizon_end))
    return tuple(study), tuple(after)


def _head_before(week: GnlStageCommitment, horizon_end: date) -> GnlStageCommitment:
    """The part of *week* before *horizon_end*, its hours cut pro rata."""
    head_hours = (horizon_end - week.start_date) / timedelta(hours=1)
    share = head_hours / _HOURS_PER_OPERATIVE_WEEK
    return dataclasses.replace(
        week,
        hours=head_hours,
        block_hours=tuple(h * share for h in week.block_hours),
    )


class GnlClass4Window(NamedTuple):
    """One já-comandada (class-4) window and the ``gl`` week it delivers.

    ``week`` is ``None`` for the ``0 MW`` grid stub; ``start_date`` is later
    than ``week.start_date`` for a week clipped at the horizon end
    (:func:`classify_gnl_windows`).
    """

    start_date: date
    end_date: date
    committed_mw: float
    week: GnlStageCommitment | None


@dataclass(frozen=True)
class GnlPlantClassification:
    """One plant's já-comandada (class-4) run and where it hands off to class-3.

    ``class4_windows`` is ascending and tiles ``[horizon_end, class4_end)`` at
    coverage 1.0: a ``0 MW`` stub ``[horizon_end, first_start)`` when the first
    já-comandada week starts after ``horizon_end``, then one window per
    já-comandada week. ``class4_end`` — where the freshly-signaled span
    starts — is, when the plant declares none, the first Saturday from
    ``horizon_end`` with a ``gs`` calendar (behind a ``0 MW`` stub) and
    ``horizon_end`` itself without one.
    ``study_weeks`` are the plant's ``gl`` weeks ``gs`` books to the study.
    """

    class4_windows: tuple[GnlClass4Window, ...]
    class4_end: date
    study_weeks: tuple[GnlStageCommitment, ...] = ()


@dataclass(frozen=True)
class GnlClassification:
    """Per-plant já-comandada (class-4) vs signaled (class-3) partition.

    Keyed by ``codigo_usina`` (:attr:`GnlThermal.code`); built by
    :func:`classify_gnl_windows` from ``gl`` alone — the source model never
    separately labels a signaled week, so class-3 is simply whatever
    class-4 does not cover.
    """

    plants: dict[int, GnlPlantClassification]


def classify_gnl_windows(
    model: GnlCommitmentModel, calendar: Sequence[OperativeStage]
) -> GnlClassification:
    """Split each plant's post-horizon ``gl`` weeks into já-comandada vs signaled.

    Já-comandada (class-4) is every ``gl`` week ``gs`` books past the study
    (:func:`_split_at_study_end`), ordered ascending. A week starting before the
    horizon end is clipped to start at it: its leading days are folded into the
    study's last stage instead (:func:`_split_at_study_end`). A
    non-terminal window ends at the next já-comandada week's start; the
    terminal window is one operative week (``start + _HOURS_PER_OPERATIVE_WEEK``)
    — never ``c.hours``, which the source model leaves at ``0`` past the
    horizon. Pure: no ``novomodelo`` import, no I/O, no MW clamping (the emission
    site's job).

    Raises
    ------
    ValueError
        If a plant's já-comandada weeks are not spaced exactly one operative
        week apart — the run has a hole or overlap this function refuses to
        silently repair — or :func:`_split_at_study_end` rejects a week.
    """
    study_week_count = _study_gl_week_count(model.weeks_per_month, calendar)
    horizon_end = calendar[-1].end_date
    return GnlClassification(
        plants={
            code: _classify_plant(
                code,
                *_split_at_study_end(
                    code, commitment.stages, study_week_count, horizon_end
                ),
                horizon_end,
                grid_aligned=study_week_count is not None,
            )
            for code, commitment in model.commitments.items()
        }
    )


def _classify_plant(
    code: int,
    study_weeks: tuple[GnlStageCommitment, ...],
    post_study: Sequence[GnlStageCommitment],
    horizon_end: date,
    *,
    grid_aligned: bool,
) -> GnlPlantClassification:
    """One plant's já-comandada run — see :func:`classify_gnl_windows`.

    With a ``gs`` calendar and no já-comandada week, the month after the study
    starts with the week after the plant's last ``gl`` week. When that week
    starts past *horizon_end* (the last ``gl`` week was cut at the month end),
    the days before it are a ``0 MW`` stub the source model commits nothing
    to; when it starts at or before *horizon_end*, the study decides it from
    *horizon_end* on and there is no stub.
    """
    post_horizon = sorted(post_study, key=lambda s: s.start_date)
    if not post_horizon:
        next_week = (
            max(s.start_date for s in study_weeks) + _ONE_OPERATIVE_WEEK
            if grid_aligned and study_weeks
            else horizon_end
        )
        if next_week <= horizon_end:
            return GnlPlantClassification(
                class4_windows=(), class4_end=horizon_end, study_weeks=study_weeks
            )
        return GnlPlantClassification(
            class4_windows=(GnlClass4Window(horizon_end, next_week, 0.0, None),),
            class4_end=next_week,
            study_weeks=study_weeks,
        )

    windows: list[GnlClass4Window] = []
    cursor = horizon_end
    if post_horizon[0].start_date > cursor:
        windows.append(GnlClass4Window(cursor, post_horizon[0].start_date, 0.0, None))
        cursor = post_horizon[0].start_date

    last = len(post_horizon) - 1
    for i, stage in enumerate(post_horizon):
        if max(stage.start_date, horizon_end) != cursor:
            raise ValueError(
                f"ambiguous já-comandada classification for GNL plant {code}: "
                f"week {stage.start_date.isoformat()} does not continue from "
                f"{cursor.isoformat()}"
            )
        nominal_end = stage.start_date + _ONE_OPERATIVE_WEEK
        end = post_horizon[i + 1].start_date if i < last else nominal_end
        if end != nominal_end:
            raise ValueError(
                f"ambiguous já-comandada classification for GNL plant {code}: "
                f"week {stage.start_date.isoformat()} is followed by "
                f"{end.isoformat()}, not the one-operative-week boundary "
                f"{nominal_end.isoformat()}"
            )
        windows.append(GnlClass4Window(cursor, end, stage.committed_mw, stage))
        cursor = end

    return GnlPlantClassification(
        class4_windows=tuple(windows), class4_end=cursor, study_weeks=study_weeks
    )


# ---------------------------------------------------------------------------
# Emission: GNL model -> novomodelo anticipated-dispatch inputs (both boundaries)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GnlEmission:
    """The novomodelo inputs an anticipated GNL fleet contributes to a converted case.

    ``thermals`` are created ``system/thermals.json`` entries (GNL plants are
    absent from ``CT``); ``past_anticipated_commitments`` extends
    ``initial_conditions.json`` with every temporal piece already fixed by
    ``gl`` (the in-study left boundary plus the post-horizon já-comandada run);
    ``post_study_stages`` is the standalone ``post_study_stages.json`` payload,
    ``None`` when the model declares no GS calendar (``weeks_per_month`` empty)
    and non-``None`` (its ``thermal_bounds`` possibly empty) whenever a GS
    calendar exists.
    """

    thermals: list[dict]
    past_anticipated_commitments: list[dict]
    post_study_stages: dict | None


def _warn_on_nonuniform_lag(nl_lag_months: Mapping[int, int]) -> None:
    """Warn when GNL plants declare differing anticipation lags.

    Every anticipated plant shares one lead, the já-comandada cutoff
    (:func:`_shared_class4_end`). A deck mixing ``nl`` lags would need per-plant
    windows — out of scope — so a non-uniform ``nl`` is surfaced (not silently
    averaged into one shift).
    """
    distinct = set(nl_lag_months.values())
    if len(distinct) > 1:
        _LOG.warning(
            "GNL plants declare differing anticipation lags %s; the lags are "
            "not used: every plant's anticipated_config.lead_time_hours runs "
            "to the shared já-comandada cutoff",
            sorted(distinct),
        )


def _reject_straddling_windows(
    past: Sequence[Mapping[str, object]], horizon_end: date
) -> None:
    """Raise if any ``past_anticipated_commitments`` row straddles ``horizon_end``.

    Class-2 windows end at or before ``horizon_end``; class-4 windows
    (:func:`classify_gnl_windows`) start at or after it. A row with
    ``start_date < horizon_end < end_date`` means the two classes overlapped
    the horizon instead of tiling either side of it.
    """
    for row in past:
        start = date.fromisoformat(row["start_date"])
        end = date.fromisoformat(row["end_date"])
        if start < horizon_end < end:
            raise ValueError(
                f"past_anticipated_commitment for thermal {row['thermal_id']} "
                f"straddles the horizon: [{row['start_date']}, "
                f"{row['end_date']}) crosses horizon_end={horizon_end.isoformat()}"
            )


class _StageCapability(NamedTuple):
    """A GNL plant's hours-weighted ``tg`` cost and ``[min, max]`` at one stage."""

    cost_per_mwh: float
    min_mw: float
    max_mw: float


def _stage_capabilities(
    tg: pd.DataFrame, calendar: Sequence[OperativeStage]
) -> dict[int, list[_StageCapability]]:
    """Each GNL plant's capability per calendar stage, from its ``tg`` records
    densified like ``CT`` (a stage inherits the last declared record)."""
    dense = dense_stage_records(tg, calendar, register="TG", name_column="nome")
    return {
        code: [
            _StageCapability(
                cost_per_mwh=hours_weighted(values["cvu"], stage),
                min_mw=hours_weighted(values["inflex"], stage),
                max_mw=hours_weighted(values["disp"], stage),
            )
            for stage, values in zip(calendar, plant["stages"], strict=True)
        ]
        for code, plant in dense.items()
    }


def _clamp_committed(
    value: float, name: str, lo: float, hi: float, context: str
) -> float:
    """Clamp a committed MW into the plant's ``[lo, hi]`` capability at its
    delivery stage.

    The emission site owns bounds policy (the reader returns the true committed
    values): the source model's ``gl`` geração and ``tg`` disponibilidade are
    independent fields, so a commitment can exceed capability, and a
    ``past_anticipated_commitment`` whose ``value_mw`` falls outside the plant's
    static generation bounds is rejected by novomodelo's semantic validator
    (``initial_conditions.rs`` requires every ``value_mw`` in ``[min, max]``). It
    is clamped into range with a warning instead — mirroring the sibling NEWAVE
    path (``converters/initial_conditions.py``).
    """
    if lo > hi:
        _LOG.warning(
            "GNL %s: inflexibility %.4g > availability %.4g (degenerate bounds); "
            "clamping commitments to <= %.4g",
            name,
            lo,
            hi,
            hi,
        )
        lo = hi
    clamped = min(max(value, lo), hi)
    if abs(clamped - value) > _NONZERO_TOLERANCE:
        _LOG.warning(
            "GNL %s: committed %.4g MW (%s) outside [%.4g, %.4g]; clamped to %.4g",
            name,
            value,
            context,
            lo,
            hi,
            clamped,
        )
    return clamped


def _lead_delivery_stage_count(
    lead_hours: float, cumulative_hours: Sequence[float]
) -> int:
    """Leading study stages novomodelo treats as pre-study-committed for *lead_hours*.

    Mirrors novomodelo-io's ``lead_delivery_stage_count`` for ``LeadTime``: the count
    of leading stages whose stage-end cumulative hours are ``<= lead_hours``
    (tie-inclusive). The bridge tiles exactly these with
    ``past_anticipated_commitments``, so it must be handed the emitted
    ``lead_time_hours`` — for a lead reaching the horizon this is every study
    stage.
    """
    count = 0
    for boundary in cumulative_hours[1:]:
        if boundary > lead_hours:
            break
        count += 1
    return count


def _record_bound(
    bounds: list[dict],
    seen: set[tuple[int, int]],
    tid: int,
    idx: int,
    capability: _StageCapability,
) -> None:
    """Append a ``(tid, idx)`` ``thermal_bounds`` row once (idempotent on ``seen``)."""
    if (tid, idx) in seen:
        return
    seen.add((tid, idx))
    bounds.append(
        {
            "thermal_id": tid,
            "post_study_stage_index": idx,
            "cost_per_mwh": capability.cost_per_mwh,
            "min_mw": capability.min_mw,
            "max_mw": capability.max_mw,
        }
    )


def _month_end_duration_hours(start: date) -> float:
    """Hours from ``start`` to the end of its civil month, the next month's
    first day exclusive (``2026-06-06`` -> ``2026-07-01``, 600 h)."""
    _, last_day = _calendar.monthrange(start.year, start.month)
    return float(last_day - start.day + 1) * 24.0


def _days_to_saturday(day: date) -> int:
    """Days from *day* to the next Saturday after it (``7`` from a Saturday)."""
    # Saturday is weekday() 5.
    return (5 - day.weekday()) % 7 or 7


def _build_post_study_calendar(
    stage_spans: Sequence[tuple[date, date]],
    class4_end: date,
) -> list[dict]:
    """The post-study calendar: a class-4 fill, then the study mirrored from
    ``class4_end``.

    Two phases, both tiled on the operative-week (Saturday) grid:

    * **class-4 fill** over ``[horizon_end, class4_end)`` — a grid-alignment
      stub ``[horizon_end, next_saturday)`` (``0`` h when ``horizon_end`` is
      already a Saturday), then ``168`` h operative weeks tiling up to
      ``class4_end``. Empty when ``class4_end == horizon_end``
      (:func:`classify_gnl_windows` found nothing commanded past the study).
    * **class-3 study-mirror** starting at ``class4_end`` — one stage per
      study stage, same type: a study week mirrors to an operative week
      ending on the next Saturday (shorter than ``168`` h only when
      ``class4_end`` is off the grid, the source model's first post-study week
      having started before ``horizon_end``), and the study's own trailing
      monthly stage mirrors to
      a stage spanning to the end of *its own* (mirrored) calendar month
      (:func:`_month_end_duration_hours`), never the study stage's own
      recorded duration. Anchoring this mirror at ``horizon_end`` instead of
      ``class4_end`` collapses every já-comandada week onto the mirror's own
      dates, so novomodelo would see one class-3 (study-decided) stage instead of
      one per study stage.

    Returns ``[]`` when ``stage_spans`` is empty. Pure: no I/O, no ``novomodelo``
    import.
    """
    if not stage_spans:
        return []
    horizon_end = stage_spans[-1][1]

    stages: list[dict] = []
    cursor = horizon_end
    stub_hours = (_days_to_saturday(horizon_end) % 7) * 24.0
    if stub_hours > 0.0 and cursor < class4_end:
        stages.append({"start_date": cursor.isoformat(), "duration_hours": stub_hours})
        cursor += timedelta(hours=stub_hours)
    while cursor < class4_end:
        stages.append(
            {
                "start_date": cursor.isoformat(),
                "duration_hours": float(_HOURS_PER_OPERATIVE_WEEK),
            }
        )
        cursor += _ONE_OPERATIVE_WEEK

    for s_start, s_end in stage_spans:
        span_hours = (s_end - s_start).days * 24.0
        duration = (
            _days_to_saturday(cursor) * 24.0
            if span_hours == _HOURS_PER_OPERATIVE_WEEK
            else _month_end_duration_hours(cursor)
        )
        stages.append({"start_date": cursor.isoformat(), "duration_hours": duration})
        cursor += timedelta(hours=duration)

    return stages


def _shared_class4_end(
    classification: GnlClassification,
    thermals: Sequence[GnlThermal],
    horizon_end: date,
) -> date:
    """The one já-comandada cutoff every anticipated thermal must share.

    ``post_study_stages`` is emitted once, globally (:class:`GnlEmission`), so
    its calendar needs a single ``class4_end``; a deck whose plants disagree
    would leave the class-3/class-4 split ambiguous for at least one of them,
    so this raises instead of picking one plant's boundary silently. Falls
    back to ``horizon_end`` when there are no thermals to check.

    Raises
    ------
    ValueError
        If the plants' ``class4_end`` values disagree.
    """
    ends = {classification.plants[t.code].class4_end for t in thermals}
    if len(ends) > 1:
        raise ValueError(
            "GNL plants declare differing já-comandada cutoffs "
            f"{sorted(ends)}; post_study_stages is emitted once, globally, "
            "and needs one shared class4_end"
        )
    return next(iter(ends), horizon_end)


def _fold_onto_stages(
    study_weeks: Sequence[GnlStageCommitment],
    stage_spans: Sequence[tuple[date, date]],
) -> list[float]:
    """The study ``gl`` weeks (:func:`_split_at_study_end`) starting inside each
    study stage, folded into one hours-weighted MW rate per stage (``0`` where
    none starts)."""
    folded: list[float] = []
    for s_start, s_end in stage_spans:
        windows = [
            (c.hours, c.committed_mw)
            for c in study_weeks
            if s_start <= c.start_date < s_end
        ]
        total_h = sum(h for h, _ in windows)
        folded.append(
            sum(h * mw for h, mw in windows) / total_h if total_h > 0 else 0.0
        )
    return folded


def _gnl_thermal(
    thermal: GnlThermal,
    tid: int,
    bus_id: int,
    study_capability: Sequence[_StageCapability],
    horizon_start: date,
    anticipated_config: Mapping[str, float],
) -> dict:
    """A created ``thermals.json`` entry for a GNL plant.

    The ``generation`` pair is the envelope of the plant's per-study-stage
    capability (smallest minimum, largest maximum): novomodelo validates every
    ``past_anticipated_commitments`` value against that static pair, so a
    stage-1 pair would reject a commitment the plant's own stage allows. Its
    ``cost_per_mwh`` is the first stage's, as for ``CT`` plants.
    """
    return {
        "id": tid,
        "name": thermal.name,
        "operational_start_date": horizon_start.isoformat(),
        "bus_id": bus_id,
        "cost_per_mwh": study_capability[0].cost_per_mwh,
        "generation": {
            "min_mw": min(c.min_mw for c in study_capability),
            "max_mw": max(c.max_mw for c in study_capability),
        },
        "anticipated_config": dict(anticipated_config),
        "entry_stage_id": None,
        "exit_stage_id": None,
    }


def _study_tiles(
    tid: int,
    thermal: GnlThermal,
    folded: Sequence[float],
    study_capability: Sequence[_StageCapability],
    calendar: Sequence[OperativeStage],
    count: int,
) -> list[dict]:
    """The in-study left boundary: the leading *count* study stages, each
    pinned to its folded ``gl`` MW clamped into that stage's capability."""
    return [
        {
            "thermal_id": tid,
            "start_date": calendar[j].start_date.isoformat(),
            "end_date": calendar[j].end_date.isoformat(),
            "value_mw": _clamp_committed(
                folded[j],
                thermal.name,
                study_capability[j].min_mw,
                study_capability[j].max_mw,
                f"in-horizon study stage {j}",
            ),
        }
        for j in range(count)
    ]


def _monthly_lead_stages(
    model: GnlCommitmentModel, thermal: GnlThermal, horizon_end: date, n_stages: int
) -> int:
    """A plant's ``nl`` lag, in stages of a study of monthly stages only.

    Raises
    ------
    ValueError
        If the plant declares no lag, a lag outside ``1..n_stages``, or a
        ``gl`` dispatch dated past the horizon: a delivery past the horizon
        decided before the study has no in-study stage to carry it.
    """
    label = f"GNL plant {thermal.code} ({thermal.name})"
    lag = model.nl_lag_months.get(thermal.code)
    if lag is None:
        raise ValueError(f"{label} declares no nl dispatch-anticipation lag")
    if not 1 <= lag <= n_stages:
        raise ValueError(
            f"{label}: an nl lag of {lag} month(s) is not supported for a study "
            f"of {n_stages} monthly stage(s); the lag must be between 1 and the "
            "stage count"
        )
    late = [
        s.start_date.isoformat()
        for s in model.commitments[thermal.code].stages
        if s.start_date >= horizon_end
    ]
    if late:
        raise ValueError(
            f"{label}: gl dispatch dated past the study horizon ({', '.join(late)}) "
            "is not supported for a study of monthly stages only"
        )
    return lag


def _post_study_months(horizon_end: date, count: int) -> list[dict]:
    """*count* whole calendar months from *horizon_end*, the first of a month."""
    stages: list[dict] = []
    start = horizon_end
    for _ in range(count):
        _, days = _calendar.monthrange(start.year, start.month)
        stages.append({"start_date": start.isoformat(), "duration_hours": days * 24.0})
        start += timedelta(days=days)
    return stages


def _pricing_axis(
    calendar: Sequence[OperativeStage], post_count: int
) -> list[OperativeStage]:
    """The calendar extended by *post_count* stages, so the ``tg`` records of the
    post-study months densify like in-study ones.

    Each extension stage carries the last study stage's patamar hours: no
    register declares the source model's post-horizon patamar durations.
    """
    last = calendar[-1]
    return [
        *calendar,
        *(
            dataclasses.replace(last, index=last.index + 1 + j)
            for j in range(post_count)
        ),
    ]


def _post_study_month_hours(
    horizon_end: date, post_calendar: Sequence[dict]
) -> list[dict[int, float]]:
    """Each post-study stage's hours per calendar month it overlaps, keyed by
    the 0-based month after the horizon end's (``{0: 144.0, 1: 24.0}`` for a
    week crossing into the second post-study month)."""
    shares: list[dict[int, float]] = []
    for stage in post_calendar:
        start = date.fromisoformat(stage["start_date"])
        end = start + timedelta(hours=stage["duration_hours"])
        hours: dict[int, float] = {}
        cursor = start
        while cursor < end:
            _, days = _calendar.monthrange(cursor.year, cursor.month)
            month_end = min(
                end, date(cursor.year, cursor.month, days) + timedelta(days=1)
            )
            offset = (
                (cursor.year - horizon_end.year) * 12 + cursor.month - horizon_end.month
            )
            hours[offset] = hours.get(offset, 0.0) + (month_end - cursor).days * 24.0
            cursor = month_end
        shares.append(hours)
    return shares


def _blend_capability(
    capability: Sequence[_StageCapability], hours: Mapping[int, float]
) -> _StageCapability:
    """The hours-weighted mean of the capabilities at each month in *hours*."""
    distinct = {capability[k] for k in hours}
    if len(distinct) == 1:
        return distinct.pop()
    total = sum(hours.values())
    return _StageCapability(
        *(
            sum(getattr(capability[k], field) * h for k, h in hours.items()) / total
            for field in _StageCapability._fields
        )
    )


def _convert_monthly_gnl(
    model: GnlCommitmentModel,
    gnl_id: Mapping[int, int],
    bus_id_of: Callable[[int], int],
    calendar: Sequence[OperativeStage],
) -> GnlEmission:
    """:func:`convert_gnl` for a study of monthly stages only.

    Each ``gs`` month is one interval, so a plant's ``nl`` lag is a whole number
    of stages, emitted as ``anticipated_config = {"lead_stages": lag}``: the
    dispatch delivered at stage ``m`` is decided at stage ``m - lag``. The
    leading ``lag`` study stages were decided before the study, so they tile
    ``past_anticipated_commitments`` with their ``gl`` MW. The post-study
    calendar is the ``max(lag)`` calendar months after the horizon; a plant's
    study stages decide its first ``lag`` of them, each priced by a
    ``thermal_bounds`` row from that month's ``tg`` record (the ``tg`` stage
    axis runs on past the study, one stage per post-study month).

    Raises
    ------
    ValueError
        If ``gs`` books more than one interval to a month, or a plant's lag or
        ``gl`` dispatch fails :func:`_monthly_lead_stages`.
    """
    split_months = {m: n for m, n in model.weeks_per_month.items() if n != 1}
    if split_months:
        raise ValueError(
            "a study of monthly stages only books one gs interval per month; gs "
            f"declares {split_months} (month: intervals)"
        )
    n_stages = len(calendar)
    horizon_start = calendar[0].start_date
    horizon_end = calendar[-1].end_date
    stage_spans = [(stage.start_date, stage.end_date) for stage in calendar]
    lags = {
        t.code: _monthly_lead_stages(model, t, horizon_end, n_stages)
        for t in model.thermals
    }
    post_calendar = _post_study_months(horizon_end, max(lags.values(), default=0))
    capabilities = _stage_capabilities(
        model.tg, _pricing_axis(calendar, len(post_calendar))
    )

    thermals: list[dict] = []
    past: list[dict] = []
    bounds: list[dict] = []
    seen_bounds: set[tuple[int, int]] = set()
    for thermal in model.thermals:
        tid = gnl_id[thermal.code]
        lag = lags[thermal.code]
        stage_capability = capabilities[thermal.code]
        study_capability = stage_capability[:n_stages]
        thermals.append(
            _gnl_thermal(
                thermal,
                tid,
                bus_id_of(thermal.submarket_code),
                study_capability,
                horizon_start,
                {"lead_stages": lag},
            )
        )
        study_weeks, _ = _split_at_study_end(
            thermal.code,
            model.commitments[thermal.code].stages,
            _study_gl_week_count(model.weeks_per_month, calendar),
            horizon_end,
        )
        folded = _fold_onto_stages(study_weeks, stage_spans)
        past.extend(_study_tiles(tid, thermal, folded, study_capability, calendar, lag))
        for j in range(lag):
            _record_bound(bounds, seen_bounds, tid, j, stage_capability[n_stages + j])

    past.sort(key=lambda w: (w["thermal_id"], w["start_date"]))
    bounds.sort(key=lambda b: (b["thermal_id"], b["post_study_stage_index"]))
    return GnlEmission(
        thermals=thermals,
        past_anticipated_commitments=past,
        post_study_stages=(
            {"stages": post_calendar, "thermal_bounds": bounds}
            if post_calendar
            else None
        ),
    )


def convert_gnl(
    model: GnlCommitmentModel,
    *,
    first_thermal_id: int,
    bus_id_of: Callable[[int], int],
    calendar: Sequence[OperativeStage],
) -> GnlEmission:
    """Convert a :class:`GnlCommitmentModel` into novomodelo's anticipated-GNL inputs.

    Each GNL plant is *created* (absent from ``CT``, :func:`_gnl_thermal`) with
    a dense id assigned after the existing thermals (``first_thermal_id``
    onward, ascending by code). A study of monthly stages only takes
    :func:`_convert_monthly_gnl`; one split into operative weeks marks each
    plant anticipated via ``anticipated_config = {"lead_time_hours":
    lead}``. ``lead`` is ``(class4_end − horizon_start).days * 24.0``, long
    enough that novomodelo's own study-reachable boundary lands exactly at the
    já-comandada cutoff ``class4_end`` (:func:`classify_gnl_windows`): every
    class-4 window then stays inside the study's reach, and only the class-3
    (signaled) stages at or after ``class4_end`` are left needing
    ``thermal_bounds`` pricing. Every anticipated thermal must agree on
    ``class4_end``, or this raises (:func:`_shared_class4_end`), since the
    post-study calendar (:func:`_build_post_study_calendar`) anchored at it is
    emitted once, globally. Every anticipated thermal then gets:

    * ``past_anticipated_commitments`` carrying both temporal pieces already
      fixed by ``gl``: the class-2 in-study left boundary, tiling the leading
      ``_lead_delivery_stage_count(lead)`` study stages (every study stage,
      since ``class4_end`` is never before the horizon end) with the
      hours-weighted committed MW folded from the
      (weekly) ``gl`` deliveries onto each study stage (explicit ``0`` where
      none); and the class-4 já-comandada run (:func:`classify_gnl_windows`),
      tiling ``[horizon_end, class4_end)`` at coverage 1.0 (an explicit
      ``0 MW`` stub included) with each window's own committed MW. Every
      window, either class, is clamped into the plant's ``tg`` capability at
      its own stage — an in-study tile at that study stage, a post-horizon
      window at the last study stage, which every later stage inherits
      (:func:`_clamp_committed`); no window may straddle the horizon
      (:func:`_reject_straddling_windows` raises otherwise) — class-2 always
      ends at or before it, class-4 always starts at or after it.

    Every plant also gets a ``thermal_bounds`` row for each **class-3
    (signaled)** post-study calendar stage — one whose ``start_date >=
    class4_end`` — carrying its ``cvu`` (fuel-inclusive) as ``cost_per_mwh``
    and its ``[min_mw, max_mw]`` capability, the carrier a signaled stage
    needs to be priced at all. Both are the ``tg`` records of the calendar
    months the stage overlaps, weighted by its hours in each: the ``tg`` stage
    axis runs on past the study, one stage per month
    (:func:`_post_study_month_hours`, :func:`_pricing_axis`). A class-4
    (já-comandada) stage gets none: its delivery is already fixed by the
    ``past_anticipated_commitments`` window above, and a ``thermal_bounds`` row
    there would let novomodelo re-optimize a cell the source model has already
    committed. ``post_study_stages`` is ``None`` when the model declares no
    ``GS`` calendar (``model.weeks_per_month`` empty) — the deck's own signal
    that there is no post-study month to price; otherwise it is emitted with its
    ``thermal_bounds`` possibly empty, when every post-study stage is
    class-4.

    ``calendar`` is the operative calendar the case's ``stages.json`` is built
    from.
    """
    gnl_id = {t.code: first_thermal_id + i for i, t in enumerate(model.thermals)}
    if weekly_stage_count(calendar) == 0:
        return _convert_monthly_gnl(model, gnl_id, bus_id_of, calendar)

    horizon_start = calendar[0].start_date
    stage_spans = [(stage.start_date, stage.end_date) for stage in calendar]
    stage_hours = [stage.total_hours for stage in calendar]
    # Cumulative operative-stage boundaries S_0=0, S_1, .., S_n, matching novomodelo's
    # `cumulative_stage_boundaries(study_stage_durations)` — the clock the
    # anticipated-delivery decider is resolved against.
    cumulative_hours = [0.0]
    for h in stage_hours:
        cumulative_hours.append(cumulative_hours[-1] + h)

    _warn_on_nonuniform_lag(model.nl_lag_months)

    horizon_end = stage_spans[-1][1]
    classification = classify_gnl_windows(model, calendar)

    # The post-study calendar exists exactly when the deck declares a GS
    # calendar (model.weeks_per_month), the deck's signal that there is a
    # post-study month to price.
    if model.weeks_per_month:
        class4_end = _shared_class4_end(classification, model.thermals, horizon_end)
        post_calendar = _build_post_study_calendar(stage_spans, class4_end)
        lead_hours = (class4_end - horizon_start).days * 24.0
    else:
        # No GS calendar (a real dadgnl always declares GS; this is a degenerate
        # deck): there is no post-study horizon to anticipate into, so the plant
        # keeps a first-stage physical lead — a valid anticipated thermal with
        # no free deliveries.
        post_calendar = []
        lead_hours = stage_hours[0]
    # Sized from the emitted lead itself, never from the GS weeks of the study
    # months: GL commits the first weeks of the month after the study too, and
    # novomodelo derives the tiled prefix from lead_time_hours alone.
    tile_k = _lead_delivery_stage_count(lead_hours, cumulative_hours)

    thermals: list[dict] = []
    past: list[dict] = []
    bounds: list[dict] = []
    seen_bounds: set[tuple[int, int]] = set()

    n_stages = len(calendar)
    month_hours = _post_study_month_hours(horizon_end, post_calendar)
    capabilities = _stage_capabilities(
        model.tg,
        _pricing_axis(calendar, max((max(h) for h in month_hours), default=-1) + 1),
    )
    for thermal in model.thermals:
        tid = gnl_id[thermal.code]
        stage_capability = capabilities[thermal.code]
        study_capability = stage_capability[:n_stages]
        last_capability = study_capability[-1]
        post_capability = [
            _blend_capability(stage_capability[n_stages:], hours)
            for hours in month_hours
        ]
        plant_classification = classification.plants[thermal.code]
        folded = _fold_onto_stages(plant_classification.study_weeks, stage_spans)
        thermals.append(
            _gnl_thermal(
                thermal,
                tid,
                bus_id_of(thermal.submarket_code),
                study_capability,
                horizon_start,
                {"lead_time_hours": lead_hours},
            )
        )
        past.extend(
            _study_tiles(tid, thermal, folded, study_capability, calendar, tile_k)
        )
        for window in plant_classification.class4_windows:
            past.append(
                {
                    "thermal_id": tid,
                    "start_date": window.start_date.isoformat(),
                    "end_date": window.end_date.isoformat(),
                    "value_mw": _clamp_committed(
                        window.committed_mw,
                        thermal.name,
                        last_capability.min_mw,
                        last_capability.max_mw,
                        f"post-horizon class-4 window {window.start_date.isoformat()}",
                    ),
                }
            )

        # A class-4 stage's delivery is already fixed by the class-4 window
        # above; giving it a thermal_bounds row too would let novomodelo
        # re-optimize an already-committed cell. Only class-3 (signaled)
        # stages get the carrier.
        class4_end = plant_classification.class4_end
        for m, stage in enumerate(post_calendar):
            if date.fromisoformat(stage["start_date"]) >= class4_end:
                _record_bound(bounds, seen_bounds, tid, m, post_capability[m])

    _reject_straddling_windows(past, horizon_end)
    past.sort(key=lambda w: (w["thermal_id"], w["start_date"]))
    bounds.sort(key=lambda b: (b["thermal_id"], b["post_study_stage_index"]))
    post_study = (
        {"stages": post_calendar, "thermal_bounds": bounds} if post_calendar else None
    )

    return GnlEmission(
        thermals=thermals,
        past_anticipated_commitments=past,
        post_study_stages=post_study,
    )
