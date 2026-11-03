"""Operative-calendar temporal conversion for DECOMP-like decks.

Builds the stage calendar (dates, blocks, seasons) from the deck's start
date (``DT``) and per-stage block durations (``DP``). The operative-calendar
rules:

- a study whose first month is split into operative weeks starts on a
  Saturday; one without weeks (every stage monthly) starts on the first of a
  month. The shape and durations are owned by
  :func:`build_operative_calendar`; the traps that a reader cannot recover
  from the code are: the first monthly stage starts mid-month at the week
  boundary, because the first month's last week keeps the days that overlap
  the second month; a civil-month-boundary week's ``NH = 168 − 24·Ndias`` is
  only the NEWAVE-coupling energy weight, so a straddle week is still a 168 h
  ``DP`` stage; and summer time (``VR``, ``168 ± 1 h``) is the one thing that
  moves a week's real ``DP`` duration off 168 h, which this whole-day calendar
  rejects;
- the terminal fan is the only branching point (see :func:`build_node_graph`),
  so the deterministic trunk may run any number of monthly stages before it.
  That the trunk is actually deterministic is a *scenario* contract checked
  against the inflow file (``scenarios.terminal_fan_probabilities``), not a
  calendar rule: a deck whose tree branches before the last stage (DECOMP's
  growing tree from the second month on) is rejected there, never flattened;
- a weekly stage carries the **first operative month's** season (the month of
  its first week's Friday), even when it straddles a civil-month boundary; a
  monthly stage carries its own calendar month's season.

Season ids follow the shared 0-based convention (Jan=0 … Dec=11) via
:func:`novomodelo_bridge.core.season_calendar.monthly_season_definitions`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING, Protocol

from novomodelo_bridge.core.pandas_utils import as_frame, as_int
from novomodelo_bridge.core.season_calendar import (
    block_names,
    monthly_season_definitions,
)
from novomodelo_bridge.novomodelo import schemas as novomodelo_schemas

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from idecomp.decomp import Dadger

    from novomodelo_bridge.decomp.case import DecompCase

_SATURDAY = 5
_WEEK_HOURS = 168.0
_HOURS_PER_DAY = 24.0
# Only the first operative month is split into weeks, and it spans four to six
# of them depending on where the Saturday start falls in the civil month; every
# later month is a single monthly stage.
_MAX_FIRST_MONTH_WEEKS = 6


@dataclass(frozen=True)
class OperativeStage:
    """One stage of the operative calendar."""

    index: int
    start_date: date
    end_date: date  # exclusive
    season_id: int  # 0-based calendar month (Jan=0 … Dec=11)
    block_hours: tuple[float, ...]

    @property
    def total_hours(self) -> float:
        """Total stage duration in hours."""
        return sum(self.block_hours)


def hours_weighted(values: Sequence[float], stage: OperativeStage) -> float:
    """Hours-weighted mean of per-block *values* over *stage*'s block hours.

    Shared by every converter that folds a per-block declaration down to one
    stage-level number (``decomp/converters/thermal.py``'s ``CT`` base row,
    ``decomp/anticipated.py``'s ``tg`` registry, ...) — the single canonical
    implementation of the convention.
    """
    return (
        sum(v * h for v, h in zip(values, stage.block_hours, strict=True))
        / stage.total_hours
    )


def build_operative_calendar(
    start_date: date,
    stage_block_hours: Sequence[Sequence[float]],
) -> list[OperativeStage]:
    """Derive the operative stage calendar from block durations.

    ``stage_block_hours[s][b]`` is the duration in hours of block *b* of
    stage *s*. The first operative month is either a run of 168 h weekly
    stages that tiles it to a calendar-month boundary or a single monthly
    stage; each subsequent month is one monthly stage spanning a full
    calendar month. Weekly stages form a prefix and monthly stages the suffix
    (DECOMP discretizes at most the first month into weeks); the last stage is
    the terminal monthly fan.

    Raises
    ------
    ValueError
        If the calendar violates the operative rules: a weekly first stage
        not starting on a Saturday, a monthly first stage not starting on the
        first of a month, a fractional-day stage, a non-final stage that is
        neither a 168 h week nor a full calendar month, a weekly stage
        following a monthly one,
        more than ``_MAX_FIRST_MONTH_WEEKS`` weekly stages, or a stage that
        does not close its calendar month. The deterministic trunk may run any
        number of months; the terminal fan being the only branch point is a
        *scenario* contract verified against the inflow
        file (``scenarios.terminal_fan_probabilities``), not here.
    """
    if len(stage_block_hours) < 2:
        raise ValueError(
            "an operative calendar has at least two stages "
            f"(a trunk + the terminal month); got {len(stage_block_hours)}"
        )

    first_month = (start_date + timedelta(days=6)).month

    stages: list[OperativeStage] = []
    cursor = start_date
    last_index = len(stage_block_hours) - 1
    seen_monthly = False
    weeks = 0
    for index, hours in enumerate(stage_block_hours):
        block_hours = tuple(float(h) for h in hours)
        total = sum(block_hours)
        days = total / _HOURS_PER_DAY
        if abs(days - round(days)) > 1e-9:
            raise ValueError(
                f"stage {index}: total duration {total} h is not a whole number of days"
            )
        end = cursor + timedelta(days=round(days))

        # A monthly stage covers exactly one civil month, so it ends on the
        # first of the month after its start. Testing the end (not the start)
        # is deliberate: the first monthly stage begins mid-month, where the
        # weekly prefix left off, since the first month's last week carries the
        # overlapping days — only later monthly stages begin on day 1. Keying
        # on the end admits both and still rejects a stage spanning two civil
        # months. A weekly stage is a full 168 h and only valid in the prefix,
        # before any monthly stage.
        is_month = end == _first_of_next_month(cursor)
        is_week = total == _WEEK_HOURS and not seen_monthly and not is_month
        if index == 0:
            _check_study_start(start_date, is_week=is_week, is_month=is_month)
        if index < last_index:
            if not (is_week or is_month):
                raise ValueError(
                    f"stage {index}: a non-final stage is either a 168 h "
                    f"operative week of the first month or a stage closing a "
                    f"calendar month; got {total} h spanning {cursor} to "
                    f"{end}. A summer-time week (VR, 168 ± 1 h) is not "
                    f"representable on this whole-day calendar, and weekly "
                    f"stages may not follow a monthly one."
                )
            weeks += is_week
            if weeks > _MAX_FIRST_MONTH_WEEKS:
                raise ValueError(
                    f"the first operative month has at most "
                    f"{_MAX_FIRST_MONTH_WEEKS} weekly stages; got more. Only "
                    f"the first month is weekly — later months are monthly "
                    f"stages."
                )
            season_id = cursor.month - 1 if is_month else first_month - 1
        else:
            if not is_month:
                raise ValueError(
                    f"the final stage closes a calendar month (the terminal "
                    f"fan aggregates one month); got {total} h spanning "
                    f"{cursor} to {end}"
                )
            season_id = cursor.month - 1
        seen_monthly = seen_monthly or is_month
        stages.append(
            OperativeStage(
                index=index,
                start_date=cursor,
                end_date=end,
                season_id=season_id,
                block_hours=block_hours,
            )
        )
        cursor = end

    return stages


def weekly_stage_count(calendar: Sequence[OperativeStage]) -> int:
    """Number of operative weeks, the 168 h prefix of the calendar (no monthly
    stage spans 168 h)."""
    count = 0
    for stage in calendar:
        if stage.total_hours != _WEEK_HOURS:
            break
        count += 1
    return count


def _check_study_start(start_date: date, *, is_week: bool, is_month: bool) -> None:
    if is_week and start_date.weekday() != _SATURDAY:
        raise ValueError(
            f"operative weeks begin on Saturdays; start date {start_date} "
            f"is a {start_date:%A}"
        )
    if is_month and start_date.day != 1:
        raise ValueError(
            "a study without operative weeks starts on the first of a month; "
            f"start date {start_date}"
        )


def _first_of_next_month(day: date) -> date:
    """The first day of the calendar month after *day*'s month."""
    return date(day.year + (day.month == 12), day.month % 12 + 1, 1)


class _StudyDate(Protocol):
    @property
    def ano(self) -> int | None: ...

    @property
    def mes(self) -> int | None: ...

    @property
    def dia(self) -> int | None: ...


class _CalendarRegisters(Protocol):
    @property
    def dt(self) -> _StudyDate | None: ...

    def dp(self, *, df: bool) -> object: ...


def operative_calendar_from_dadger(dadger: _CalendarRegisters) -> list[OperativeStage]:
    """Build the operative calendar from a deck's ``DT`` + ``DP`` records.

    Block durations must agree across subsystems for each stage (the deck
    defines one temporal grid); a mismatch is a hard error.
    """
    dt = dadger.dt
    if dt is None or dt.ano is None or dt.mes is None or dt.dia is None:
        raise ValueError(
            "the deck has no complete DT record; cannot build the calendar"
        )
    start = date(int(dt.ano), int(dt.mes), int(dt.dia))

    dp = as_frame(dadger.dp(df=True))
    if dp is None or dp.empty:
        raise ValueError("the deck has no DP records; cannot build the calendar")

    stage_block_hours: list[list[float]] = []
    expected = 1
    for estagio, group in dp.groupby("estagio", sort=True):
        if as_int(estagio) != expected:
            raise ValueError(
                f"DP stages must be contiguous from 1; expected {expected}, "
                f"got {as_int(estagio)}"
            )
        expected += 1

        n_blocks = int(group["numero_patamares"].iloc[0])
        columns = [f"duracao_{k}" for k in range(1, n_blocks + 1)]
        reference: list[float] | None = None
        for _, row in group.iterrows():
            if int(row["numero_patamares"]) != n_blocks:
                raise ValueError(
                    f"stage {as_int(estagio)}: numero_patamares differs across subsystems"
                )
            values = [float(row[c]) for c in columns]
            if reference is None:
                reference = values
            elif values != reference:
                raise ValueError(
                    f"stage {as_int(estagio)}: block durations differ across "
                    f"subsystems ({values} vs {reference}); the deck defines "
                    "one temporal grid"
                )
        assert reference is not None
        stage_block_hours.append(reference)

    return build_operative_calendar(start, stage_block_hours)


@dataclass(frozen=True)
class CVaRConfig:
    """Resolved CVaR risk measure for the DECOMP study.

    ``from_stage_index`` is the 0-based novomodelo stage from which DECOMP's ``AR``
    register starts CVaR (starting period − 1). It is recorded for the
    conversion diagnostic only: :func:`stage_records` emits the measure
    *uniformly* across all stages (CVaR collapses to expectation on the
    deterministic trunk, and novomodelo's gap rule under CVaR requires a uniform
    measure), so the starting period does not gate emission. ``alpha`` (the
    worst-fraction quantile — novomodelo's α-convention equals DECOMP's, no ``1−α``
    flip) and ``lambda_`` (the risk-aversion weight) are fractions in ``(0, 1]``.
    """

    from_stage_index: int
    alpha: float
    lambda_: float


def _first_active(values: Sequence[float] | None) -> float | None:
    """The first strictly-positive entry of a per-period array, or ``None``."""
    for v in values or ():
        if v is not None and float(v) > 0.0:
            return float(v)
    return None


def _cortesh_cvar(cortesh_path: Path | None) -> tuple[float | None, float | None]:
    """The FCF header's active ``(alpha, lambda)`` CVaR fractions.

    Reads ``cortesh.dat`` (``Cortesh.alfa_cvar``/``lambda_cvar``, already
    fractions), returning the first active (nonzero) pair, or
    ``(None, None)`` when the header is absent or CVaR is disabled there.
    """
    if cortesh_path is None:
        return None, None
    from inewave.newave import Cortesh

    cortesh = Cortesh.read(str(cortesh_path))
    if cortesh.usa_cvar != 1:
        return None, None
    return _first_active(cortesh.alfa_cvar), _first_active(cortesh.lambda_cvar)


def resolve_cvar(dadger: Dadger, cortesh_path: Path | None) -> CVaRConfig | None:
    """Resolve the DECOMP CVaR risk measure, or ``None`` for expectation.

    DECOMP applies CVaR via the ``AR`` register (a starting period and, on the
    same record, λ and α). Per the DECOMP manual §3.4.4.1, **blank** λ/α mean
    "use the values NEWAVE employed" — carried in the FCF header ``cortesh.dat``
    (``alfa_cvar``/``lambda_cvar``). Resolution order:

    1. no ``AR`` register → expectation (``None``);
    2. ``AR`` with explicit λ/α → those (DECOMP stores them as percentages, so a
       value ``> 1`` is divided by 100);
    3. ``AR`` with blank λ/α → the FCF header's fractions, if ``cortesh.dat`` is
       present and CVaR-enabled there.

    Returns ``None`` when no admissible ``(alpha, lambda)`` with both ``> 0``
    resolves. CVaR of a deterministic (single-opening) stage collapses to
    expectation, so applying it from ``AR``'s starting period across the
    deterministic trunk is a no-op there and only binds the stochastic fan.
    """
    from idecomp.decomp.modelos.dadger import AR

    ar = next((r for r in dadger.data.of_type(AR)), None)
    if ar is None:
        return None
    estagio = int(ar.estagio) if ar.estagio is not None else 1

    def _frac(value: float | None) -> float | None:
        if value is None:
            return None
        value = float(value)
        return value / 100.0 if value > 1.0 else value

    alpha, lam = _frac(ar.alfa), _frac(ar.lamb)
    if alpha is None or lam is None:
        alpha_c, lam_c = _cortesh_cvar(cortesh_path)
        alpha = alpha if alpha is not None else alpha_c
        lam = lam if lam is not None else lam_c
    if alpha is None or lam is None or alpha <= 0.0 or lam <= 0.0:
        return None
    return CVaRConfig(from_stage_index=max(0, estagio - 1), alpha=alpha, lambda_=lam)


def stage_records(
    calendar: Sequence[OperativeStage], cvar: CVaRConfig | None = None
) -> list[dict]:
    """Build the ``stages.json`` stage entries for an operative calendar.

    Every DECOMP stage draws its openings from the external inflow library
    (trunk column 0, terminal fan columns ``0..N-1``), so no stage declares
    ``num_openings`` — the node graph's per-node ``scenario_id`` binds the
    openings, and novomodelo rejects a ``num_openings`` on an external-only stage.
    State variables follow the lag-blind convention: storage only, no
    inflow-lag state (only the boundary FCF prices lags).

    ``cvar`` (when the deck runs risk-averse) emits a ``{"cvar": {...}}`` risk
    measure on **every** stage, else ``"expectation"`` on every stage. The
    measure is emitted uniformly rather than only from ``cvar.from_stage_index``
    onward for two coincident reasons: CVaR on a deterministic (single-opening)
    stage collapses to expectation, so a uniform emission is identical in
    effect to gating it at DECOMP's ``AR`` starting period (only the stochastic
    fan is actually bound); and novomodelo admits the ``gap`` stopping rule under
    CVaR + enumerated forwards only when the risk measure is **uniform across
    all stages** (``setup/mod.rs::reject_gap_under_nonuniform_risk``) — a
    per-stage mix of expectation and CVaR would force the ``bound_stalling``
    fallback instead.
    """
    records: list[dict] = []
    for stage in calendar:
        names = block_names(len(stage.block_hours))
        if cvar is not None:
            risk_measure: object = {
                "cvar": {"alpha": cvar.alpha, "lambda": cvar.lambda_}
            }
        else:
            risk_measure = "expectation"
        records.append(
            {
                "id": stage.index,
                "start_date": stage.start_date.isoformat(),
                "end_date": stage.end_date.isoformat(),
                "season_id": stage.season_id,
                "blocks": [
                    {"id": b, "name": names[b], "hours": hours}
                    for b, hours in enumerate(stage.block_hours)
                ],
                "risk_measure": risk_measure,
                "state_variables": {"storage": True, "inflow_lags": False},
            }
        )
    return records


def build_node_graph(
    n_stages: int,
    fan_probabilities: Sequence[float],
) -> tuple[list[dict], list[dict]]:
    """Node-native graph for the trunk-as-external + terminal-fan tree.

    The deterministic trunk (stages ``0..T-2``) is one node per stage bound
    to external inflow column 0; the terminal stage ``T-1`` fans into
    ``len(fan_probabilities)`` leaf nodes bound to columns ``0..N-1``. Node
    ids are their own id space: trunk node ``id == stage_id`` (``0..T-2``),
    fan node ``id == (T-1) + k`` ascending with ``scenario_id`` (the
    canonical successor order). Trunk edges carry probability 1.0; each
    terminal branch edge carries its DECOMP per-scenario weight (novomodelo
    re-normalizes out-edges at load).
    """
    terminal = n_stages - 1
    if terminal < 1:
        raise ValueError(
            "the DECOMP node graph needs at least a trunk stage and a "
            f"terminal fan stage; got {n_stages} stage(s)"
        )

    nodes: list[dict] = [
        {"id": s, "stage_id": s, "scenario_id": 0, "label": f"trunk-{s}"}
        for s in range(terminal)
    ]
    fan_base = terminal  # ids after the T-1 trunk nodes (0..terminal-1)
    nodes.extend(
        {
            "id": fan_base + k,
            "stage_id": terminal,
            "scenario_id": k,
            "label": f"fan-{k}",
        }
        for k in range(len(fan_probabilities))
    )

    transitions: list[dict] = [
        {"source_id": s, "target_id": s + 1, "probability": 1.0}
        for s in range(terminal - 1)
    ]
    last_trunk = terminal - 1
    transitions.extend(
        {
            "source_id": last_trunk,
            "target_id": fan_base + k,
            "probability": probability,
        }
        for k, probability in enumerate(fan_probabilities)
    )
    return nodes, transitions


def convert_stages(
    case: DecompCase,
    *,
    annual_discount_rate: float,
    fan_probabilities: Sequence[float],
    cvar: CVaRConfig | None = None,
) -> dict:
    """Build the full ``stages.json`` dict for an operative calendar.

    A node-native finite-horizon graph: a deterministic trunk that fans into
    the terminal stage (see :func:`build_node_graph`), under the shared
    calendar-monthly season map. No pre-study stages: the inflow model is
    order-0 and pre-study inflows travel as dated windows. ``cvar`` (from
    :func:`resolve_cvar`) applies the deck's CVaR risk measure per stage.
    """
    stages = stage_records(case.calendar, cvar)
    nodes, transitions = build_node_graph(len(stages), fan_probabilities)
    return {
        "$schema": novomodelo_schemas.schema_url_for("stages.json"),
        "season_definitions": monthly_season_definitions(),
        "policy_graph": {
            "type": "finite_horizon",
            "annual_discount_rate": annual_discount_rate,
            "nodes": nodes,
            "transitions": transitions,
        },
        "stages": stages,
    }
