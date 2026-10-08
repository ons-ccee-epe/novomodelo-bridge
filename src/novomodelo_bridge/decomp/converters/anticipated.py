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
  ahead of its delivery, and this is what sizes its physical ``lead_time_hours``.

A commitment's delivery date is recorded per stage so the emission track can
place it: an in-study ``gl`` commitment is folded onto the study stages as a
left-boundary ``past_anticipated_commitment``, and a post-horizon ``gl`` week
becomes a fixed já-comandada ``past_anticipated_commitment`` too. Any
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
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING, NamedTuple

from novomodelo_bridge.decomp.converters.thermal import dense_stage_records
from novomodelo_bridge.decomp.temporal import hours_weighted

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import pandas as pd
    from idecomp.decomp import Dadgnl

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
    block-duration-weighted mean of that register's ``geracao``; ``hours`` is the
    stage span (sum of its ``duracao`` blocks), so the emission track can size a
    post-horizon delivery window ``[start_date, start_date + hours)``.
    """

    estagio: int
    start_date: date
    committed_mw: float
    hours: float


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
    number of operative weeks in each month, 1-based month index ascending).
    :func:`_study_lead_hours` sums it over the study's own months to size
    ``H``, a single global anticipation lead (a whole number of operative
    weeks) whose only role is sizing the class-2 left-boundary tile count
    (:func:`_lead_delivery_stage_count`) — the leading study stages folded
    into ``past_anticipated_commitments``. ``H`` shapes neither the post-study
    calendar, which is anchored at the study horizon end and the plants'
    shared já-comandada cutoff independent of it (:func:`_build_post_study_calendar`),
    nor the per-plant lead each thermal actually emits as its own
    ``anticipated_config.lead_time_hours``
    — see :func:`convert_gnl` for that.

    ``nl_lag_months`` is the ``nl`` block's per-plant dispatch-anticipation lag
    (``{codigo_usina: months}``): the number of months by which a GNL plant's
    dispatch is decided ahead of its delivery (the LNG supply lead time).
    ``H``'s tile-sizing role assumes every anticipated plant shares one
    study-span lead, so this drives only the uniform-lag consistency check
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
    """Whether ``dadgnl`` declares any committed GNL dispatch (the G6 gate).

    ``True`` iff ``dadgnl`` is present and at least one ``gl`` register carries a
    nonzero ``geracao``. There is no source-model ``dger``-equivalent activation
    flag for GNL, so presence of real committed generation is the gate. A deck
    with only a ``tg`` registry (or all-zero ``gl``) is treated as GNL-off.
    """
    if dadgnl is None:
        return False
    registers = dadgnl.gl()
    if not registers:
        return False
    return any(
        abs(g) > _NONZERO_TOLERANCE
        for register in registers
        for g in _as_floats(register.geracao)
    )


def read_gnl_model(dadgnl: Dadgnl) -> GnlCommitmentModel | None:
    """Read ``dadgnl`` into a :class:`GnlCommitmentModel`, or ``None`` if GNL-off.

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
    for register in dadgnl.gl():
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
            )
        )
    return {
        code: GnlCommitment(
            code=code, stages=tuple(sorted(stages, key=lambda s: s.estagio))
        )
        for code, stages in by_code.items()
    }


def _read_weeks_per_month(dadgnl: Dadgnl) -> dict[int, int]:
    """The ``gs`` weeks-per-month map ``{mes: semanas}`` (empty when absent)."""
    frame = dadgnl.gs(df=True)
    if frame is None or frame.empty:
        return {}
    return {int(row["mes"]): int(row["semanas"]) for _, row in frame.iterrows()}


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


_ONE_OPERATIVE_WEEK = timedelta(hours=_HOURS_PER_OPERATIVE_WEEK)


@dataclass(frozen=True)
class GnlPlantClassification:
    """One plant's já-comandada (class-4) run and where it hands off to class-3.

    ``class4_windows`` is ascending and tiles ``[horizon_end, class4_end)`` at
    coverage 1.0: a ``0 MW`` stub ``[horizon_end, first_start)`` when the first
    já-comandada week starts after ``horizon_end``, then one window per
    já-comandada week. ``class4_end`` — where the freshly-signaled span
    starts — is ``horizon_end`` itself when the plant declares none.
    """

    class4_windows: tuple[tuple[date, date, float], ...]
    class4_end: date


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
    model: GnlCommitmentModel, *, horizon_end: date
) -> GnlClassification:
    """Split each plant's post-horizon ``gl`` weeks into já-comandada vs signaled.

    Já-comandada (class-4) is ``commitment.stages`` with ``start_date >=
    horizon_end``, ordered ascending. A non-terminal window's width is the
    exact date gap to the next já-comandada week; the terminal window is one
    operative week (``start + _HOURS_PER_OPERATIVE_WEEK``) — never ``c.hours``,
    which the source model leaves at ``0`` past the horizon (``duracao`` is
    only tracked for weeks inside its own operative calendar). Pure: no
    ``novomodelo`` import, no I/O, no MW clamping (the emission site's job).

    Raises
    ------
    ValueError
        If a plant's já-comandada weeks are not spaced exactly one operative
        week apart — the run has a hole or overlap this function refuses to
        silently repair.
    """
    return GnlClassification(
        plants={
            code: _classify_plant(code, commitment.stages, horizon_end)
            for code, commitment in model.commitments.items()
        }
    )


def _classify_plant(
    code: int, stages: Sequence[GnlStageCommitment], horizon_end: date
) -> GnlPlantClassification:
    """One plant's já-comandada run — see :func:`classify_gnl_windows`."""
    post_horizon = sorted(
        (s for s in stages if s.start_date >= horizon_end),
        key=lambda s: s.start_date,
    )
    if not post_horizon:
        return GnlPlantClassification(class4_windows=(), class4_end=horizon_end)

    windows: list[tuple[date, date, float]] = []
    cursor = horizon_end
    if post_horizon[0].start_date > cursor:
        windows.append((cursor, post_horizon[0].start_date, 0.0))
        cursor = post_horizon[0].start_date

    last = len(post_horizon) - 1
    for i, stage in enumerate(post_horizon):
        if stage.start_date != cursor:
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
        windows.append((stage.start_date, end, stage.committed_mw))
        cursor = end

    return GnlPlantClassification(class4_windows=tuple(windows), class4_end=cursor)


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


def _study_lead_hours(
    stage_spans: Sequence[tuple[date, date]],
    weeks_per_month: Mapping[int, int],
) -> float:
    """The single global anticipation lead ``H``, a whole number of operative weeks.

    ``H = (Σ GS weeks over the study's months) × 168 h``. The study's month count
    is the number of distinct ``(year, month)`` among the study stage **start**
    dates; the first that many ``weeks_per_month`` values (ascending by month key)
    are the study months' operative-week counts. ``H`` sizes every plant's
    ``anticipated_config.lead_time_hours`` and the in-study left-boundary tile
    count (:func:`_lead_delivery_stage_count`); it does not shape the post-study
    calendar itself, which is grid-anchored independently of it
    (:func:`_build_post_study_calendar`).

    Returns ``0.0`` when ``weeks_per_month`` is empty (no GS calendar).
    """
    if not weeks_per_month:
        return 0.0
    study_month_count = len({(s.year, s.month) for s, _ in stage_spans})
    ordered_weeks = [weeks_per_month[k] for k in sorted(weeks_per_month)]
    return float(sum(ordered_weeks[:study_month_count]) * _HOURS_PER_OPERATIVE_WEEK)


def _warn_on_nonuniform_lag(nl_lag_months: Mapping[int, int]) -> None:
    """Warn when GNL plants declare differing anticipation lags (spec §7 A1).

    The single global lead ``H`` assumes every anticipated plant shares the
    study-span anticipation lag. A deck mixing ``nl`` lags would need per-plant
    windows — out of scope — so a non-uniform ``nl`` is surfaced (not silently
    averaged into one shift).
    """
    distinct = set(nl_lag_months.values())
    if len(distinct) > 1:
        _LOG.warning(
            "GNL plants declare differing anticipation lags %s; only the "
            "class-2 left-boundary tiling depth is sized from one study-span "
            "lead — each plant's anticipated_config.lead_time_hours is "
            "derived independently from its own já-comandada cutoff",
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
    """Leading study stages novomodelo treats as pre-study-committed for lead ``H``.

    Mirrors novomodelo-io's ``lead_delivery_stage_count`` for ``LeadTime``: the count
    of leading stages whose stage-end cumulative hours are ``<= H`` (tie-
    inclusive). The bridge tiles exactly these with
    ``past_anticipated_commitments`` so the left boundary matches the depth novomodelo
    derives from ``H`` — for a lead reaching past the horizon this is every study
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
    """Hours from ``start`` to the last day of its own calendar month.

    Uses the month's own last day-of-month number as the exclusive boundary
    directly (``2026-06-06`` -> ``2026-06-30``, 24 d/576 h) — not the
    following month's first day, which would add a spurious extra day
    (600 h) and miss novomodelo's own post-study e2e fixture by one day.
    """
    _, last_day = _calendar.monthrange(start.year, start.month)
    return float(last_day - start.day) * 24.0


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
      ``class4_end``. Empty when ``class4_end == horizon_end`` (no
      já-comandada run to fill).
    * **class-3 study-mirror** starting at ``class4_end`` — one stage per
      study stage, same type: a ``168`` h study week mirrors to a ``168`` h
      operative week, and the study's own trailing monthly stage mirrors to
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
    # Saturday = weekday() 5; 0 when horizon_end is already on the grid.
    stub_hours = ((5 - horizon_end.weekday()) % 7) * 24.0
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
            float(_HOURS_PER_OPERATIVE_WEEK)
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


def convert_gnl(
    model: GnlCommitmentModel,
    *,
    first_thermal_id: int,
    bus_id_of: Callable[[int], int],
    calendar: Sequence[OperativeStage],
) -> GnlEmission:
    """Convert a :class:`GnlCommitmentModel` into novomodelo's anticipated-GNL inputs.

    Each GNL plant is *created* (absent from ``CT``) with a dense id assigned
    after the existing thermals (``first_thermal_id`` onward, ascending by
    code) and marked anticipated via ``anticipated_config = {"lead_time_hours":
    lead}``. ``lead`` is **per plant** — ``(class4_end − horizon_start).days *
    24.0`` — long enough that novomodelo's own study-reachable boundary lands
    exactly at that plant's já-comandada cutoff ``class4_end``
    (:func:`classify_gnl_windows`): every class-4 window then stays inside the
    study's reach, and only the class-3 (signaled) stages at or after
    ``class4_end`` are left needing ``thermal_bounds`` pricing. This is
    unrelated to the single global ``H`` (:func:`_study_lead_hours`, a whole
    number of study operative weeks) that still sizes the class-2 left
    boundary's tile count below, and to the post-study calendar
    (:func:`_build_post_study_calendar`), which is anchored at the study
    horizon end and the plants' shared ``class4_end`` independent of any lead
    — every anticipated thermal must agree on ``class4_end``, or this raises
    (:func:`_shared_class4_end`), since the calendar is emitted once,
    globally. Every anticipated thermal then gets:

    * ``past_anticipated_commitments`` carrying both temporal pieces already
      fixed by ``gl``: the class-2 in-study left boundary, tiling the leading
      ``_lead_delivery_stage_count(H)`` study stages (every study stage when
      ``H > horizon``) with the hours-weighted committed MW folded from the
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
    and its ``[min_mw, max_mw]`` capability at the last study stage, the
    carrier a signaled stage needs to be priced at all. A class-4
    (já-comandada) stage gets none: its delivery is already fixed by the
    ``past_anticipated_commitments`` window above, and a ``thermal_bounds`` row
    there would let novomodelo re-optimize a cell the source model has already
    committed. ``post_study_stages`` is ``None`` when the model declares no
    ``GS`` calendar (``model.weeks_per_month`` empty) — the deck's own signal
    that there is no post-study month to price; otherwise it is emitted with its
    ``thermal_bounds`` possibly empty, when every post-study stage is
    class-4.

    Each plant's ``thermals.json`` ``generation`` pair is the envelope of its
    per-stage capability (smallest minimum, largest maximum): novomodelo validates
    every ``past_anticipated_commitments`` value against that static pair, so a
    stage-1 pair would reject a commitment the plant's own stage allows. Its
    ``cost_per_mwh`` is the first stage's, as for ``CT`` plants.

    ``calendar`` is the operative calendar the case's ``stages.json`` is built
    from.
    """
    horizon_start = calendar[0].start_date
    stage_spans = [(stage.start_date, stage.end_date) for stage in calendar]
    stage_hours = [stage.total_hours for stage in calendar]
    # Cumulative operative-stage boundaries S_0=0, S_1, .., S_n, matching novomodelo's
    # `cumulative_stage_boundaries(study_stage_durations)` — the clock the
    # anticipated-delivery decider is resolved against.
    cumulative_hours = [0.0]
    for h in stage_hours:
        cumulative_hours.append(cumulative_hours[-1] + h)

    lead_hours = _study_lead_hours(stage_spans, model.weeks_per_month)
    _warn_on_nonuniform_lag(model.nl_lag_months)

    gnl_id = {t.code: first_thermal_id + i for i, t in enumerate(model.thermals)}
    horizon_end = stage_spans[-1][1]
    classification = classify_gnl_windows(model, horizon_end=horizon_end)

    # The post-study calendar exists exactly when the deck declares a GS
    # calendar (model.weeks_per_month) -- the same "is there a post-study
    # month to price" signal _study_lead_hours uses for H; the calendar's own
    # shape no longer depends on H, only on the plants' shared class4_end.
    post_calendar = (
        _build_post_study_calendar(
            stage_spans,
            _shared_class4_end(classification, model.thermals, horizon_end),
        )
        if model.weeks_per_month
        else []
    )

    if post_calendar:
        # The left boundary tiles exactly the leading stages novomodelo derives from H
        # (every study stage, since H spans the whole study horizon).
        tile_k = _lead_delivery_stage_count(lead_hours, cumulative_hours)
    else:
        # No GS calendar (a real dadgnl always declares GS; this is a degenerate
        # deck): there is no post-study horizon to anticipate into, so the plant
        # keeps only the mandatory single leading commitment and a first-stage
        # physical lead — a valid anticipated thermal with no free deliveries.
        tile_k = 1
        lead_hours = stage_hours[0]

    thermals: list[dict] = []
    past: list[dict] = []
    bounds: list[dict] = []
    seen_bounds: set[tuple[int, int]] = set()

    capabilities = _stage_capabilities(model.tg, calendar)
    for thermal in model.thermals:
        tid = gnl_id[thermal.code]
        commitment = model.commitments[thermal.code]
        stage_capability = capabilities[thermal.code]
        last_capability = stage_capability[-1]

        # Fold weekly gl deliveries onto study stages (hours-weighted MW rate).
        folded: list[float] = []
        for s_start, s_end in stage_spans:
            windows = [
                (c.hours, c.committed_mw)
                for c in commitment.stages
                if s_start <= c.start_date < s_end
            ]
            total_h = sum(h for h, _ in windows)
            folded.append(
                sum(h * mw for h, mw in windows) / total_h if total_h > 0 else 0.0
            )

        plant_classification = classification.plants[thermal.code]
        # Per plant, not the tile-sizing global H: long enough that novomodelo's own
        # study-reachable boundary lands exactly at this plant's já-comandada
        # cutoff, so every class-4 window stays inside the study's reach and
        # only class-3 (signaled) stages need thermal_bounds pricing.
        emitted_lead_hours = (
            (plant_classification.class4_end - horizon_start).days * 24.0
            if post_calendar
            else lead_hours
        )
        thermals.append(
            {
                "id": tid,
                "name": thermal.name,
                "operational_start_date": horizon_start.isoformat(),
                "bus_id": bus_id_of(thermal.submarket_code),
                "cost_per_mwh": stage_capability[0].cost_per_mwh,
                "generation": {
                    "min_mw": min(c.min_mw for c in stage_capability),
                    "max_mw": max(c.max_mw for c in stage_capability),
                },
                "anticipated_config": {"lead_time_hours": emitted_lead_hours},
                "entry_stage_id": None,
                "exit_stage_id": None,
            }
        )
        for j in range(tile_k):
            past.append(
                {
                    "thermal_id": tid,
                    "start_date": calendar[j].start_date.isoformat(),
                    "end_date": calendar[j].end_date.isoformat(),
                    "value_mw": _clamp_committed(
                        folded[j],
                        thermal.name,
                        stage_capability[j].min_mw,
                        stage_capability[j].max_mw,
                        f"in-horizon study stage {j}",
                    ),
                }
            )
        for start, end, committed_mw in plant_classification.class4_windows:
            past.append(
                {
                    "thermal_id": tid,
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "value_mw": _clamp_committed(
                        committed_mw,
                        thermal.name,
                        last_capability.min_mw,
                        last_capability.max_mw,
                        f"post-horizon class-4 window {start.isoformat()}",
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
                _record_bound(bounds, seen_bounds, tid, m, last_capability)

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
