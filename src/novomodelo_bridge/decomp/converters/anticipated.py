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
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.anticipated import commitment_clamped_diagnostic
from novomodelo_bridge.core.errors import FieldParseError, UnsupportedSourceError
from novomodelo_bridge.core.pandas_utils import (
    as_float,
    as_frame,
    as_int,
    require_frame,
)
from novomodelo_bridge.core.policy_dates import hours_between
from novomodelo_bridge.core.units import MEAN_MONTH_HOURS
from novomodelo_bridge.decomp.converters.thermal import dense_stage_records
from novomodelo_bridge.decomp.temporal import hours_weighted, weekly_stage_count

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    import pandas as pd

    from novomodelo_bridge.decomp.temporal import OperativeStage

_LOG = logging.getLogger(__name__)
_DADGNL = "dadgnl.rvN"

_NONZERO_TOLERANCE = 1e-9
_HOURS_PER_OPERATIVE_WEEK = 168  # 7 days x 24 h; the study/post-study grid step


@runtime_checkable
class _GlRegister(Protocol):
    @property
    def codigo_usina(self) -> int: ...

    @property
    def estagio(self) -> int: ...

    @property
    def data_inicio(self) -> object: ...

    @property
    def geracao(self) -> object: ...

    @property
    def duracao(self) -> object: ...


class _GnlRegisters(Protocol):
    def gl(self) -> object: ...

    def tg(self, *, df: bool) -> object: ...

    def gs(self, *, df: bool) -> object: ...

    def nl(self, *, df: bool) -> object: ...


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
    weeks this drives only the uniform-lag check
    (:func:`_report_nonuniform_lag`) — a deck mixing lags is reported, not
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
    return [as_float(value)]


def _gl_registers(dadgnl: _GnlRegisters) -> list[_GlRegister]:
    """Every ``gl`` register of ``dadgnl``, in deck order."""
    registers = dadgnl.gl()
    if isinstance(registers, list):
        return registers
    # idecomp returns a lone register bare, not as a one-item list.
    return [registers] if isinstance(registers, _GlRegister) else []


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


def is_gnl_enabled(dadgnl: _GnlRegisters | None) -> bool:
    """Whether ``dadgnl`` declares any GNL plant.

    ``True`` iff ``dadgnl`` is present and its ``tg`` registry lists a plant.
    The committed ``geracao`` does not gate it: a plant whose ``gl`` commits
    0 MW, as in the first revisions of a month, is still modelled by the source
    model, which holds it at 0 MW where committed and decides the later months
    within its ``tg`` capability.
    """
    if dadgnl is None:
        return False
    tg = as_frame(dadgnl.tg(df=True))
    return tg is not None and not tg.empty


def read_gnl_model(dadgnl: _GnlRegisters) -> GnlCommitmentModel | None:
    """Read ``dadgnl`` into a :class:`GnlCommitmentModel`, or ``None`` without
    GNL plants.

    Returns ``None`` when :func:`is_gnl_enabled` is ``False``. Otherwise builds
    the ``tg`` plants (ascending by code) and their declared records, the ``gl``
    commitments (keyed by code, ascending by ``estagio``, each stage carrying its
    parsed delivery date and block-weighted committed MW), and the ``gs``
    weeks-per-month map.

    Raises
    ------
    FieldParseError
        If a ``gl`` register names a ``codigo_usina`` with no ``tg`` registry
        entry (a committed dispatch for an unknown plant — a malformed deck),
        a ``gl`` ``data_inicio`` is not a ``ddmmyyyy`` date, or a ``gs`` or
        ``nl`` field is blank.
    """
    if not is_gnl_enabled(dadgnl):
        return None

    tg = require_frame(as_frame(dadgnl.tg(df=True)), "tg registry in dadgnl")
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
    dadgnl: _GnlRegisters, known_codes: set[int]
) -> dict[int, GnlCommitment]:
    """Build ``{code: GnlCommitment}`` by iterating ``gl`` registers.

    Iterates registers (never ``gl(df=True)`` — the ragged per-block lists make
    it unusable). Each register contributes one :class:`GnlStageCommitment` with
    its parsed date and block-weighted committed MW.
    """
    by_code: dict[int, list[GnlStageCommitment]] = {}
    for register in _gl_registers(dadgnl):
        code = int(register.codigo_usina)
        estagio = int(register.estagio)
        if code not in known_codes:
            raise FieldParseError(
                f"{_DADGNL} has a GL record for plant {code} at stage {estagio}, "
                "but no TG record declares that plant",
                field="GL plant code",
            )
        try:
            start_date = _parse_data_inicio(register.data_inicio)
        except ValueError as exc:
            raise FieldParseError(
                f"{_DADGNL} has a GL record for plant {code} at stage {estagio} "
                f"whose start date {register.data_inicio!r} is not a ddmmyyyy date",
                field="GL start date",
            ) from exc
        geracao = _as_floats(register.geracao)
        duracao = _as_floats(register.duracao)
        by_code.setdefault(code, []).append(
            GnlStageCommitment(
                estagio=estagio,
                start_date=start_date,
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
    folds its per-stage commitment with. Each week's value is first held
    inside the block's ``tg`` range at the stage, since the source model's
    ``tg`` minimum and capacity prevail over a ``gl`` value outside them.
    A stage no ``gl`` week starts in maps to ``None``: the plant's dispatch
    there is not commanded. ``tg`` is read on the axis :func:`convert_gnl`
    reads it on (:func:`_tg_axis`), so a record for a post-study month is
    accepted here exactly when the conversion accepts it.

    Raises
    ------
    ValueError
        If a ``gl`` week inside a stage declares a different block count than
        the stage itself, a ``tg`` record lies past every post-study month, or
        :func:`_post_study_calendar` refuses the plants' ``nl`` lags or ``gl``
        weeks (an :class:`UnsupportedSourceError` where the deck is valid).
    """
    study_week_count = _study_gl_week_count(model.weeks_per_month, calendar)
    post_calendar, _ = _post_study_calendar(model, calendar)
    capabilities = dense_stage_records(
        model.tg,
        _tg_axis(calendar, post_calendar),
        register="TG",
        name_column="nome",
    )
    commitments: dict[int, list[tuple[float, ...] | None]] = {}
    for thermal in model.thermals:
        stages, _ = _split_at_study_end(
            thermal.code,
            model.commitments[thermal.code].stages,
            study_week_count,
            calendar[-1],
        )
        capability = capabilities[thermal.code]["stages"]
        per_stage: list[tuple[float, ...] | None] = []
        for position, stage in enumerate(calendar):
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
                        f"{dx.counted(len(week.block_mw), 'block')}; stage {stage.index} "
                        f"has {n_blocks}"
                    )
            limits = capability[position]
            per_stage.append(
                tuple(
                    _block_weighted_mean(
                        [
                            _within_capability(
                                week.block_mw[b], limits["inflex"][b], limits["disp"][b]
                            )
                            for week in weeks
                        ],
                        [week.block_hours[b] for week in weeks],
                    )
                    for b in range(n_blocks)
                )
            )
        commitments[thermal.code] = per_stage
    return commitments


def gnl_submarkets(model: GnlCommitmentModel) -> dict[int, int]:
    """Each GNL plant's subsystem, from its ``tg`` record."""
    return {thermal.code: thermal.submarket_code for thermal in model.thermals}


def gnl_patamar_values_diagnostic(
    code: str,
    reads: Iterable[tuple[int, int, int, Sequence[float]]],
    *,
    noun: str,
    column: str,
) -> dx.Diagnostic | None:
    """The INFO for the kept constraints that read a GNL plant's commanded
    generation at a stage where it differs by patamar, or ``None`` when none
    does.

    *reads* holds each ``(constraint, thermal code, 0-based stage, MW per
    patamar)`` a kept constraint reads; *noun* names the constraints, plural,
    and *column* their table column.
    """
    stages: dict[tuple[int, int], set[int]] = {}
    for constraint, thermal_code, stage_index, blocks in reads:
        if max(blocks) - min(blocks) > _NONZERO_TOLERANCE:
            stages.setdefault((constraint, thermal_code), set()).add(stage_index)
    if not stages:
        return None
    return dx.Diagnostic(
        code=code,
        severity=dx.Severity.INFO,
        category="Special constraints",
        title=(
            f"{noun[:1].upper()}{noun[1:]} reading a GNL generation that "
            f"differs by patamar ({len(stages)})"
        ),
        summary=(
            f"In the listed stages, the listed {noun} read a GNL plant's "
            "commanded generation patamar by patamar, as DECOMP does, and it "
            "differs by patamar. DECOMP also holds the plant to it in each "
            "patamar; the case holds the plant only to its commitment for the "
            "stage, so its dispatch in a patamar can differ from the generation "
            "the constraint read there."
        ),
        table=dx.DiagnosticTable(
            columns=[column, "Thermal code", "Stages"],
            rows=[
                [constraint, thermal_code, dx.format_stage_ranges(found)]
                for (constraint, thermal_code), found in sorted(stages.items())
            ],
            justify=["right", "right", "left"],
        ),
    )


def commanded_gnl_generation(
    gnl_generation: Mapping[int, Sequence[Sequence[float] | None]],
    code: int,
    stage_index: int,
    *,
    reference: str,
    dadgnl_path: str | None,
) -> Sequence[float]:
    """GNL plant *code*'s commanded MW per block at 0-based *stage_index*,
    from :func:`gnl_block_commitments`, for the constraint term *reference*
    names.

    Raises
    ------
    FieldParseError
        When the plant's generation is not commanded at the stage (no ``gl``
        week starts in it), on the ``gl`` field of *dadgnl_path*: DECOMP
        stops on such a term too.
    """
    # TRACKED NOVOMODELO-GAP (C24): the anticipated ring holds one MW per slot,
    # so the case holds the plant's stage energy, not these per-block values; a
    # limit folded with them is exact only where the case dispatches each block
    # at its GL value. gnl_patamar_values_diagnostic lists where they differ.
    per_stage = gnl_generation.get(code)
    blocks = per_stage[stage_index] if per_stage is not None else None
    if blocks is None:
        raise FieldParseError(
            f"{reference} is on GNL thermal {code}, but no dadgnl GL week "
            f"starts in stage {stage_index + 1}, so the thermal has no "
            "commanded generation there to move into the limits; DECOMP stops "
            "on such a term too",
            path=dadgnl_path,
            field="GL",
        )
    return blocks


def _read_weeks_per_month(dadgnl: _GnlRegisters) -> dict[int, int]:
    """The ``gs`` weeks-per-month map ``{mes: semanas}`` (empty when absent)."""
    frame = as_frame(dadgnl.gs(df=True))
    if frame is None or frame.empty:
        return {}
    weeks: dict[int, int] = {}
    for _, row in frame.iterrows():
        month = _whole_number(row["mes"], "GS month", "the month of a GS record")
        weeks[month] = _whole_number(
            row["semanas"], "GS week count", f"the week count of GS month {month}"
        )
    return weeks


def operative_week_months(
    weeks_per_month: Mapping[int, int], calendar: Sequence[OperativeStage]
) -> dict[date, tuple[int, int]]:
    """The ``(year, month)`` the source model books each ``gs`` interval to,
    keyed by the interval's start.

    Walks the ``gs`` counts in ascending month order from the study start, the
    first count being the first stage's operative month's, which an RV0 starts
    in the month before (May's RV0 on 2026-04-25). The intervals are operative
    weeks, and a week belongs to the month ``gs`` books it to even when it
    ends in the next one (mar-26-rv2: the week starting 2026-05-30 is a May
    week); a study of monthly stages only has one interval per month, the
    month itself. Empty without a ``gs`` block.
    """
    monthly_only = weekly_stage_count(calendar) == 0
    weeks: dict[date, tuple[int, int]] = {}
    week_start = calendar[0].start_date
    year, month = calendar[0].operative_year, calendar[0].season_id + 1
    for key in sorted(weeks_per_month):
        if monthly_only:
            weeks[date(year, month, 1)] = (year, month)
        else:
            for _ in range(weeks_per_month[key]):
                weeks[week_start] = (year, month)
                week_start += _ONE_OPERATIVE_WEEK
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return weeks


def booked_month(
    week_months: Mapping[date, tuple[int, int]], day: date
) -> tuple[int, int]:
    """The ``(year, month)`` :func:`operative_week_months` books the interval
    starting on *day* to, or *day*'s calendar month when no interval starts
    on it."""
    return week_months.get(day, (day.year, day.month))


def _read_nl_lags(dadgnl: _GnlRegisters) -> dict[int, int]:
    """The ``nl`` dispatch-anticipation lags ``{codigo_usina: months}``.

    Each ``nl`` register carries ``codigo_usina``, ``codigo_submercado`` and
    ``lag`` (whole months of dispatch anticipation). Unlike ``gl``, ``nl`` is a
    fixed-shape register, so ``nl(df=True)`` is well-formed. Empty when the deck
    declares no ``nl`` block.
    """
    frame = as_frame(dadgnl.nl(df=True))
    if frame is None or frame.empty:
        return {}
    lags: dict[int, int] = {}
    for _, row in frame.iterrows():
        code = _whole_number(
            row["codigo_usina"], "NL plant code", "the plant code of an NL record"
        )
        lags[code] = _whole_number(row["lag"], "NL lag", f"the NL lag of plant {code}")
    return lags


def _whole_number(value: object, field: str, what: str) -> int:
    """*value* as an ``int``; *what* names it in the error a blank field raises."""
    try:
        return as_int(value)
    except (TypeError, ValueError) as exc:
        raise FieldParseError(
            f"{_DADGNL} leaves {what} blank or unreadable",
            field=field,
        ) from exc


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
            year, month = booked_month(week_months, week.start_date)
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
                    "GNL weeks weighted without their month's patamar shares "
                    f"({len(fallback_rows)})"
                ),
                summary=(
                    "The listed GNL weeks past the study horizon declare no "
                    "patamar duration, and cortesh.dat is absent or does not "
                    "cover their month. The conversion weights their patamares "
                    "by the last study stage's patamar hours, or equally when "
                    "that stage has another patamar count, so the MW each week "
                    "commits can differ from DECOMP's."
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
                    "GNL weeks committing a different MW per patamar "
                    f"({len(nonuniform_rows)})"
                ),
                summary=(
                    "The listed GNL weeks past the study horizon declare no "
                    "patamar duration and commit a different MW in each "
                    "patamar. The case commits one MW for the week, the mean "
                    "under the listed weights, so the dispatch of each patamar "
                    "can differ from DECOMP's."
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
    last_stage: OperativeStage,
) -> tuple[tuple[GnlStageCommitment, ...], tuple[GnlStageCommitment, ...]]:
    """A plant's ``gl`` weeks split into the study's and those after it.

    By ``gs`` position (:func:`_study_gl_week_count`), never by date: the first
    week of the month after the study starts on the Saturday grid, which can
    fall before the horizon end (the study's monthly stage ends on the 1st), yet
    the source model books it — and leaves its ``duracao`` blank — as a week
    past its horizon. Without ``gs`` the split falls back to ``start_date <
    horizon_end``. Such a week's days before the horizon end count in no study
    stage; the last study week covers them instead (:func:`_stretch_last_week`).

    Raises
    ------
    ValueError
        If ``gs`` books to the study a week starting at or after the horizon
        end, or past the study a week ending at or before it.
    """
    horizon_end = last_stage.end_date
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
    return _stretch_last_week(study, last_stage), tuple(after)


def _stretch_last_week(
    study: Sequence[GnlStageCommitment], last_stage: OperativeStage
) -> tuple[GnlStageCommitment, ...]:
    """*study* with its last week held to the end of *last_stage*.

    When the weeks ``gs`` books to the last stage end less than one operative
    week before the horizon end, the source model holds the last of them over
    the stage's remaining days at its own MW per patamar: its patamar hours
    become the stage's less the other weeks' in the stage. A week whose
    patamar count differs from the stage's keeps its patamar proportions over
    the stage's remaining hours.
    """
    in_stage = [
        w for w in study if last_stage.start_date <= w.start_date < last_stage.end_date
    ]
    if not in_stage:
        return tuple(study)
    last = max(in_stage, key=lambda w: w.start_date)
    shortfall = last_stage.end_date - (last.start_date + timedelta(hours=last.hours))
    if last.hours <= 0 or not timedelta(0) < shortfall < _ONE_OPERATIVE_WEEK:
        return tuple(study)
    others = [w for w in in_stage if w is not last]
    stage_hours = last_stage.block_hours
    if all(len(w.block_hours) == len(stage_hours) for w in in_stage):
        block_hours = tuple(
            h - sum(w.block_hours[b] for w in others) for b, h in enumerate(stage_hours)
        )
    else:
        remaining = sum(stage_hours) - sum(w.hours for w in others)
        block_hours = tuple(h * remaining / last.hours for h in last.block_hours)
    stretched = dataclasses.replace(
        last,
        committed_mw=_block_weighted_mean(last.block_mw, block_hours),
        hours=sum(block_hours),
        block_hours=block_hours,
    )
    return tuple(stretched if w is last else w for w in study)


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
    horizon end is clipped to start at it; the study's last week covers its
    leading days instead (:func:`_split_at_study_end`). A
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
                    code, commitment.stages, study_week_count, calendar[-1]
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
    calendar exists. ``thermal_id_by_code`` maps each plant's ``dadgnl`` code to
    the id of its created entry.
    """

    thermals: list[dict]
    past_anticipated_commitments: list[dict]
    post_study_stages: dict | None
    thermal_id_by_code: dict[int, int]


def _report_nonuniform_lag(model: GnlCommitmentModel) -> None:
    """Report GNL plants that declare differing anticipation lags.

    Every anticipated plant shares one lead, the já-comandada cutoff
    (:func:`_shared_class4_end`). A deck mixing ``nl`` lags would need per-plant
    windows, so a non-uniform ``nl`` is reported, not averaged into one shift.
    """
    lags = model.nl_lag_months
    if len(set(lags.values())) <= 1:
        return
    rows = [
        [thermal.code, thermal.name, lags[thermal.code]]
        for thermal in model.thermals
        if thermal.code in lags
    ]
    dx.emit(
        dx.Diagnostic(
            code="decomp-gnl-lag-nonuniform",
            severity=dx.Severity.WARNING,
            category="GNL",
            title=f"GNL plants with differing anticipation lags ({len(rows)})",
            summary=(
                "The listed GNL plants declare different NL anticipation lags, "
                "but on a study of operative weeks the case anticipates every "
                "plant up to the same já-comandada cutoff, so the dispatch "
                "decided ahead for these plants can differ from DECOMP's."
            ),
            table=dx.DiagnosticTable(
                columns=["Code", "Name", "Lag (months)"],
                rows=rows,
                justify=["right", "left", "right"],
            ),
        ),
        logger=_LOG,
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
        start = date.fromisoformat(str(row["start_date"]))
        end = date.fromisoformat(str(row["end_date"]))
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


def _within_capability(value: float, lo: float, hi: float) -> float:
    """*value* held inside ``[lo, hi]``; an inflexibility above the
    availability (``lo > hi``) holds it at the availability."""
    return min(max(value, min(lo, hi)), hi)


def _clamp_committed(
    value: float, thermal: GnlThermal, lo: float, hi: float, delivery: str
) -> float:
    """Clamp a committed MW into the plant's ``[lo, hi]`` capability at its
    delivery stage, reporting the move.

    The source model's ``gl`` geração and ``tg`` disponibilidade are
    independent fields, so a commitment can exceed capability, and novomodelo
    rejects a ``past_anticipated_commitment`` outside the plant's generation
    bounds.
    """
    clamped = _within_capability(value, lo, hi)
    if abs(clamped - value) > _NONZERO_TOLERANCE:
        row = [
            thermal.code,
            thermal.name,
            delivery,
            round(value, 4),
            round(lo, 4),
            round(hi, 4),
            round(clamped, 4),
        ]
        dx.emit(commitment_clamped_diagnostic([row], source="dadgnl.rvN"), logger=_LOG)
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
    return hours_between(start, start.replace(day=last_day) + timedelta(days=1))


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
        span_hours = hours_between(s_start, s_end)
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
    UnsupportedSourceError
        If the plants' ``class4_end`` values disagree.
    """
    ends = {classification.plants[t.code].class4_end for t in thermals}
    if len(ends) > 1:
        dates = ", ".join(end.isoformat() for end in sorted(ends))
        raise UnsupportedSourceError(
            "dadgnl.rvN commits the dispatch of the GNL plants past the study "
            f"horizon up to different dates ({dates}). GNL plants committed past "
            "the horizon up to different dates are not converted yet, so give "
            "each GNL plant GL weeks past the horizon up to the same date.",
            path=_DADGNL,
            field="GL data_inicio",
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
                thermal,
                study_capability[j].min_mw,
                study_capability[j].max_mw,
                f"stage {j}",
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
        If the plant declares no lag.
    UnsupportedSourceError
        If the lag is outside ``1..n_stages``, or a ``gl`` dispatch is dated
        past the horizon: a delivery past the horizon decided before the study
        has no in-study stage to carry it.
    """
    label = f"GNL plant {thermal.code} ({thermal.name})"
    lag = model.nl_lag_months.get(thermal.code)
    if lag is None:
        raise ValueError(f"{label} declares no nl dispatch-anticipation lag")
    if not 1 <= lag <= n_stages:
        raise UnsupportedSourceError(
            f"dadgnl.rvN gives {label} an NL lag of {dx.counted(lag, 'month')}, "
            f"but the study has {dx.counted(n_stages, 'monthly stage')}. In a "
            "study of monthly stages only, a lag outside its stages is not "
            f"converted yet, so give the plant an NL lag from 1 to {n_stages}.",
            path=_DADGNL,
            field="NL lag",
        )
    late = [
        s.start_date.isoformat()
        for s in model.commitments[thermal.code].stages
        if s.start_date >= horizon_end
    ]
    if late:
        raise UnsupportedSourceError(
            f"dadgnl.rvN commits {label} to GL dispatch past the study horizon "
            f"({', '.join(late)}). In a study of monthly stages only, GL dispatch "
            "past the horizon is not converted yet, so remove the plant's GL "
            "records dated past the horizon.",
            path=_DADGNL,
            field="GL data_inicio",
        )
    return lag


def _post_study_months(horizon_end: date, count: int) -> list[dict]:
    """*count* whole calendar months from *horizon_end*, the first of a month."""
    stages: list[dict] = []
    start = horizon_end
    for _ in range(count):
        _, days = _calendar.monthrange(start.year, start.month)
        end = start + timedelta(days=days)
        stages.append(
            {
                "start_date": start.isoformat(),
                "duration_hours": hours_between(start, end),
            }
        )
        start = end
    return stages


def _monthly_lags(
    model: GnlCommitmentModel, calendar: Sequence[OperativeStage]
) -> dict[int, int]:
    """Each plant's :func:`_monthly_lead_stages` on a study of monthly stages."""
    horizon_end = calendar[-1].end_date
    return {
        t.code: _monthly_lead_stages(model, t, horizon_end, len(calendar))
        for t in model.thermals
    }


def _post_study_calendar(
    model: GnlCommitmentModel, calendar: Sequence[OperativeStage]
) -> tuple[list[dict], date | None]:
    """The post-study stages :func:`convert_gnl` emits for *model*, and the
    já-comandada cutoff a weekly study anchors them at.

    A study of monthly stages only takes the whole months its longest ``nl``
    lag reaches, with no cutoff. A weekly study without a ``gs`` calendar, the
    deck's signal that there is no post-study month to price, has neither.

    Raises
    ------
    UnsupportedSourceError
        If :func:`_monthly_lead_stages` or :func:`_shared_class4_end` refuses
        the plants' ``nl`` lags or ``gl`` weeks.
    ValueError
        If a plant declares no ``nl`` lag, or :func:`classify_gnl_windows`
        refuses its ``gl`` weeks.
    """
    horizon_end = calendar[-1].end_date
    if weekly_stage_count(calendar) == 0:
        lags = _monthly_lags(model, calendar)
        return _post_study_months(horizon_end, max(lags.values(), default=0)), None
    if not model.weeks_per_month:
        return [], None
    cutoff = _shared_class4_end(
        classify_gnl_windows(model, calendar), model.thermals, horizon_end
    )
    spans = [(stage.start_date, stage.end_date) for stage in calendar]
    return _build_post_study_calendar(spans, cutoff), cutoff


def _tg_axis(
    calendar: Sequence[OperativeStage], post_calendar: Sequence[dict]
) -> list[OperativeStage]:
    """The stage axis the ``tg`` records are read on: *calendar*, then one
    stage per calendar month *post_calendar* reaches.

    The ``tg`` stage axis runs on past the study, one stage per month: stage
    ``n + 1 + k`` of an ``n``-stage study is the k-th month after the horizon
    end's. Each extension stage carries the last study stage's patamar hours:
    no register declares the source model's post-horizon patamar durations.
    """
    count = _post_study_month_count(calendar[-1].end_date, post_calendar)
    last = calendar[-1]
    return [
        *calendar,
        *(dataclasses.replace(last, index=last.index + 1 + j) for j in range(count)),
    ]


def _months_after(horizon_end: date, year: int, month: int) -> int:
    """How many calendar months ``(year, month)`` lies after *horizon_end*'s."""
    return (year - horizon_end.year) * 12 + month - horizon_end.month


def _month_label(horizon_end: date, offset: int) -> str:
    """The ``YYYY-MM`` of the month *offset* months after *horizon_end*'s."""
    year, month = divmod(horizon_end.year * 12 + horizon_end.month - 1 + offset, 12)
    return f"{year}-{month + 1:02d}"


def _stage_end(stage: dict) -> date:
    """The exclusive end of a post-study stage."""
    start = date.fromisoformat(stage["start_date"])
    return start + timedelta(hours=stage["duration_hours"])


def _post_study_month_count(horizon_end: date, post_calendar: Sequence[dict]) -> int:
    """How many calendar months the post-study stages span, from the horizon
    end's month on (``0`` without post-study stages)."""
    if not post_calendar:
        return 0
    last_day = _stage_end(post_calendar[-1]) - timedelta(days=1)
    return _months_after(horizon_end, last_day.year, last_day.month) + 1


def _post_study_tg_months(
    post_calendar: Sequence[dict],
    class4_end: date,
    week_months: Mapping[date, tuple[int, int]],
    horizon_end: date,
) -> dict[int, int]:
    """Each class-3 post-study stage's ``tg`` month, keyed by the stage's
    position, as the 0-based month after the horizon end's.

    A stage takes the month ``gs`` books the interval it starts to
    (:func:`booked_month`), the month the boundary cut prices its delivery in,
    even when the stage ends in the next calendar month or starts in the month
    after that one.

    Raises
    ------
    UnsupportedSourceError
        If that month lies outside the months the post-study stages span.
    """
    count = _post_study_month_count(horizon_end, post_calendar)
    months: dict[int, int] = {}
    for position, stage in enumerate(post_calendar):
        start = date.fromisoformat(stage["start_date"])
        if start < class4_end:
            continue
        year, month = booked_month(week_months, start)
        offset = _months_after(horizon_end, year, month)
        if not 0 <= offset < count:
            first, last = (
                _month_label(horizon_end, 0),
                _month_label(horizon_end, count - 1),
            )
            span = first if first == last else f"{first} to {last}"
            raise UnsupportedSourceError(
                f"{_DADGNL} GS books the post-study stage starting "
                f"{start.isoformat()} to {year}-{month:02d}, outside the months the "
                f"post-study stages span ({span}). A post-study stage booked "
                "outside them is not converted yet, so set the GS interval counts "
                "to book each interval past the study horizon to a month it "
                "overlaps.",
                path=_DADGNL,
                field="GS semanas",
            )
        months[position] = offset
    return months


def _report_post_study_interval_hours(
    model: GnlCommitmentModel,
    calendar: Sequence[OperativeStage],
    post_calendar: Sequence[dict],
    priced: Iterable[int],
) -> None:
    """Report the priced post-study stages the source model weighs over other
    hours.

    The source model gives each ``gs`` interval of a month an equal share of a
    :data:`MEAN_MONTH_HOURS` month, an operative week included; a post-study
    stage spans its calendar hours. A stage's source hours sum the shares of
    the intervals it overlaps, pro rata; a stage that overlaps none has no
    counterpart and no row.
    """
    week_months = operative_week_months(model.weeks_per_month, calendar)
    per_month = Counter(week_months.values())
    monthly_only = weekly_stage_count(calendar) == 0
    intervals = [
        (
            start,
            start + timedelta(hours=_month_end_duration_hours(start))
            if monthly_only
            else start + _ONE_OPERATIVE_WEEK,
            month,
        )
        for start, month in sorted(week_months.items())
    ]
    rows: list[list[object]] = []
    for position in sorted(priced):
        stage = post_calendar[position]
        start = date.fromisoformat(stage["start_date"])
        end = _stage_end(stage)
        covered = 0.0
        source_hours = 0.0
        for interval_start, interval_end, month in intervals:
            overlap = hours_between(max(start, interval_start), min(end, interval_end))
            if overlap <= 0.0:
                continue
            covered += overlap
            share = MEAN_MONTH_HOURS / per_month[month]
            source_hours += (
                share * overlap / hours_between(interval_start, interval_end)
            )
        if covered == 0.0:
            continue
        year, month = booked_month(week_months, start)
        rows.append(
            [
                position,
                start.isoformat(),
                f"{year}-{month:02d}",
                per_month[(year, month)],
                round(source_hours, 2),
                round(covered, 2),
            ]
        )
    if not rows:
        return
    dx.emit(
        dx.Diagnostic(
            code="decomp-gnl-post-study-interval-hours",
            severity=dx.Severity.WARNING,
            category="GNL",
            title=f"GNL post-study stages whose hours differ from DECOMP's ({len(rows)})",
            summary=(
                "DECOMP divides each month past the study horizon into the GS "
                "intervals of dadgnl.rvN and gives each interval an equal share of "
                "a 730.5-hour month. The case holds each post-study stage in "
                "post_study_stages.json over its calendar hours instead, since "
                "Novomodelo weighs a post-study stage by its dates alone. In the "
                "listed stages, the GNL energy committed and its cost therefore "
                "differ from DECOMP's in the ratio of the two hour columns."
            ),
            table=dx.DiagnosticTable(
                columns=[
                    "Post-study stage",
                    "Start date",
                    "GS month",
                    "GS intervals",
                    "DECOMP (h)",
                    "Case (h)",
                ],
                rows=rows,
                justify=["right", "left", "left", "right", "right", "right"],
                caption="Stage ids are 0-based.",
            ),
        ),
        logger=_LOG,
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
    UnsupportedSourceError
        If ``gs`` books more than one interval to a month, or a plant's lag or
        ``gl`` dispatch fails :func:`_monthly_lead_stages`.
    ValueError
        If a plant declares no ``nl`` lag.
    """
    split_months = {m: n for m, n in model.weeks_per_month.items() if n != 1}
    if split_months:
        months = ", ".join(
            f"month {month} into {n}" for month, n in sorted(split_months.items())
        )
        raise UnsupportedSourceError(
            f"dadgnl.rvN splits GS months into several intervals ({months}) in a "
            "study of monthly stages only. The case decides one GNL dispatch per "
            "monthly stage, so give every GS month one interval.",
            path=_DADGNL,
            field="GS semanas",
        )
    n_stages = len(calendar)
    horizon_start = calendar[0].start_date
    stage_spans = [(stage.start_date, stage.end_date) for stage in calendar]
    lags = _monthly_lags(model, calendar)
    post_calendar, _ = _post_study_calendar(model, calendar)
    capabilities = _stage_capabilities(model.tg, _tg_axis(calendar, post_calendar))

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
            calendar[-1],
        )
        folded = _fold_onto_stages(study_weeks, stage_spans)
        past.extend(_study_tiles(tid, thermal, folded, study_capability, calendar, lag))
        for j in range(lag):
            _record_bound(bounds, seen_bounds, tid, j, stage_capability[n_stages + j])

    # TRACKED NOVOMODELO-GAP (C44): each post-study month keeps its calendar
    # hours, where the source model weighs a GS month at 730.5 h.
    _report_post_study_interval_hours(
        model, calendar, post_calendar, {b["post_study_stage_index"] for b in bounds}
    )
    past.sort(key=lambda w: (w["thermal_id"], w["start_date"]))
    bounds.sort(key=lambda b: (b["thermal_id"], b["post_study_stage_index"]))
    return GnlEmission(
        thermals=thermals,
        past_anticipated_commitments=past,
        post_study_stages=(
            {"thermal_stages": post_calendar, "thermal_bounds": bounds}
            if post_calendar
            else None
        ),
        thermal_id_by_code=dict(gnl_id),
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
    needs to be priced at all. Both come from the ``tg`` record of the month
    ``gs`` books the stage to (:func:`_post_study_tg_months`): the ``tg`` stage
    axis runs on past the study, one stage per month (:func:`_tg_axis`). A class-4
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
    found: list[dx.Diagnostic] = []
    try:
        with dx.collect() as found:
            emission = _convert_gnl(
                model,
                first_thermal_id=first_thermal_id,
                bus_id_of=bus_id_of,
                calendar=calendar,
            )
    finally:
        for diagnostic in dx.merge_tables(found):
            dx.emit(diagnostic, logger=_LOG)
    return emission


def _convert_gnl(
    model: GnlCommitmentModel,
    *,
    first_thermal_id: int,
    bus_id_of: Callable[[int], int],
    calendar: Sequence[OperativeStage],
) -> GnlEmission:
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

    _report_nonuniform_lag(model)

    horizon_end = stage_spans[-1][1]
    classification = classify_gnl_windows(model, calendar)
    post_calendar, cutoff = _post_study_calendar(model, calendar)
    # No GS calendar (a real dadgnl always declares GS; this is a degenerate
    # deck): there is no post-study horizon to anticipate into, so the plant
    # keeps a first-stage physical lead — a valid anticipated thermal with no
    # free deliveries.
    lead_hours = (
        stage_hours[0] if cutoff is None else hours_between(horizon_start, cutoff)
    )
    # Sized from the emitted lead itself, never from the GS weeks of the study
    # months: GL commits the first weeks of the month after the study too, and
    # novomodelo derives the tiled prefix from lead_time_hours alone.
    tile_k = _lead_delivery_stage_count(lead_hours, cumulative_hours)

    thermals: list[dict] = []
    past: list[dict] = []
    bounds: list[dict] = []
    seen_bounds: set[tuple[int, int]] = set()

    n_stages = len(calendar)
    capabilities = _stage_capabilities(model.tg, _tg_axis(calendar, post_calendar))
    tg_months = (
        {}
        if cutoff is None
        else _post_study_tg_months(
            post_calendar,
            cutoff,
            operative_week_months(model.weeks_per_month, calendar),
            horizon_end,
        )
    )
    for thermal in model.thermals:
        tid = gnl_id[thermal.code]
        stage_capability = capabilities[thermal.code]
        study_capability = stage_capability[:n_stages]
        last_capability = study_capability[-1]
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
                        thermal,
                        last_capability.min_mw,
                        last_capability.max_mw,
                        f"window from {window.start_date.isoformat()}",
                    ),
                }
            )

        # A class-4 stage's delivery is already fixed by the class-4 window
        # above; giving it a thermal_bounds row too would let novomodelo
        # re-optimize an already-committed cell. Only class-3 (signaled)
        # stages get the carrier.
        for m, offset in tg_months.items():
            _record_bound(
                bounds, seen_bounds, tid, m, stage_capability[n_stages + offset]
            )

    # TRACKED NOVOMODELO-GAP (C44): each post-study stage keeps its calendar
    # hours, where the source model weighs a GS interval at 730.5 / N h.
    _report_post_study_interval_hours(
        model, calendar, post_calendar, {b["post_study_stage_index"] for b in bounds}
    )
    _reject_straddling_windows(past, horizon_end)
    past.sort(key=lambda w: (w["thermal_id"], w["start_date"]))
    bounds.sort(key=lambda b: (b["thermal_id"], b["post_study_stage_index"]))
    post_study = (
        {"thermal_stages": post_calendar, "thermal_bounds": bounds}
        if post_calendar
        else None
    )

    return GnlEmission(
        thermals=thermals,
        past_anticipated_commitments=past,
        post_study_stages=post_study,
        thermal_id_by_code=dict(gnl_id),
    )
