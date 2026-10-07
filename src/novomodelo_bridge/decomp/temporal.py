"""Operative-calendar temporal conversion for DECOMP-like decks.

Builds the stage calendar (dates, blocks, seasons) from the deck's start
date (``DT``) and per-stage block durations (``DP``). The operative-calendar
rules:

- the study starts on a Saturday; weekly stages are exactly 168 h and break
  on Saturdays; the single final stage aggregates the second operative
  month and ends at a calendar month boundary;
- every weekly stage carries the **first operative month's** season (the
  month containing its first week's Friday). A straddling head week's
  previous-month days carry no accumulation weight — that month is closed —
  and a straddling tail week's next-month days spill over into the second
  month's accumulator, which the final stage then completes;
- the final stage carries its own calendar month's season.

Season ids follow the shared 0-based convention (Jan=0 … Dec=11) via
:func:`cobre_bridge.core.season_calendar.monthly_season_definitions`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

from cobre_bridge.cobre import schemas as cobre_schemas
from cobre_bridge.core.season_calendar import block_names, monthly_season_definitions

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from idecomp.decomp import Dadger

    from cobre_bridge.decomp.case import DecompCase

_SATURDAY = 5
_WEEK_HOURS = 168.0
_HOURS_PER_DAY = 24.0


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
    stage *s*. All stages but the last must total exactly one operative
    week (168 h); the last stage aggregates the second operative month and
    must end at a calendar month boundary.

    Raises
    ------
    ValueError
        If the calendar violates the operative rules (non-Saturday start,
        non-weekly stage, fractional days, or a final stage that does not
        close the second operative month).
    """
    if len(stage_block_hours) < 2:
        raise ValueError(
            "an operative calendar has at least two stages "
            f"(N weekly + 1 monthly); got {len(stage_block_hours)}"
        )
    if start_date.weekday() != _SATURDAY:
        raise ValueError(
            f"operative weeks begin on Saturdays; start date {start_date} "
            f"is a {start_date:%A}"
        )

    first_month_date = start_date + timedelta(days=6)
    first_month = first_month_date.month

    stages: list[OperativeStage] = []
    cursor = start_date
    last_index = len(stage_block_hours) - 1
    for index, hours in enumerate(stage_block_hours):
        block_hours = tuple(float(h) for h in hours)
        total = sum(block_hours)
        days = total / _HOURS_PER_DAY
        if abs(days - round(days)) > 1e-9:
            raise ValueError(
                f"stage {index}: total duration {total} h is not a whole number of days"
            )
        if index < last_index and total != _WEEK_HOURS:
            raise ValueError(
                f"stage {index}: every stage before the last must be one "
                f"operative week (168 h); got {total} h"
            )
        end = cursor + timedelta(days=round(days))
        if index < last_index:
            season_id = first_month - 1
        else:
            if end.day != 1:
                raise ValueError(
                    f"final stage must end at a calendar month boundary; ends {end}"
                )
            second_month = first_month % 12 + 1
            if cursor.month != second_month:
                raise ValueError(
                    f"final stage must aggregate the second operative month "
                    f"(month {second_month}); starts {cursor}"
                )
            season_id = cursor.month - 1
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


def operative_calendar_from_dadger(dadger: Dadger) -> list[OperativeStage]:
    """Build the operative calendar from a deck's ``DT`` + ``DP`` records.

    Block durations must agree across subsystems for each stage (the deck
    defines one temporal grid); a mismatch is a hard error.
    """
    dt = dadger.dt
    start = date(int(dt.ano), int(dt.mes), int(dt.dia))

    dp = dadger.dp(df=True)
    if dp is None or dp.empty:
        raise ValueError("the deck has no DP records; cannot build the calendar")

    stage_block_hours: list[list[float]] = []
    expected = 1
    for estagio, group in dp.groupby("estagio", sort=True):
        if int(estagio) != expected:
            raise ValueError(
                f"DP stages must be contiguous from 1; expected {expected}, "
                f"got {int(estagio)}"
            )
        expected += 1

        n_blocks = int(group["numero_patamares"].iloc[0])
        columns = [f"duracao_{k}" for k in range(1, n_blocks + 1)]
        reference: list[float] | None = None
        for _, row in group.iterrows():
            if int(row["numero_patamares"]) != n_blocks:
                raise ValueError(
                    f"stage {int(estagio)}: numero_patamares differs across subsystems"
                )
            values = [float(row[c]) for c in columns]
            if reference is None:
                reference = values
            elif values != reference:
                raise ValueError(
                    f"stage {int(estagio)}: block durations differ across "
                    f"subsystems ({values} vs {reference}); the deck defines "
                    "one temporal grid"
                )
        assert reference is not None
        stage_block_hours.append(reference)

    return build_operative_calendar(start, stage_block_hours)


@dataclass(frozen=True)
class CVaRConfig:
    """Resolved CVaR risk measure for the DECOMP study.

    ``from_stage_index`` is the 0-based cobre stage from which DECOMP's ``AR``
    register starts CVaR (starting period − 1). It is recorded for the
    conversion diagnostic only: :func:`stage_records` emits the measure
    *uniformly* across all stages (CVaR collapses to expectation on the
    deterministic trunk, and cobre's gap rule under CVaR requires a uniform
    measure), so the starting period does not gate emission. ``alpha`` (the
    worst-fraction quantile — cobre's α-convention equals DECOMP's, no ``1−α``
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

    Reads ``cortesh.dat`` (``SecaoDadosCortesh.alfa_cvar``/``lambda_cvar``,
    already fractions), returning the first active (nonzero) pair, or
    ``(None, None)`` when the header is absent or CVaR is disabled there.
    """
    if cortesh_path is None:
        return None, None
    from inewave.newave import Cortesh
    from inewave.newave.modelos.cortesh import SecaoDadosCortesh

    ch = Cortesh.read(str(cortesh_path))
    sec = next((s for s in ch.data.of_type(SecaoDadosCortesh)), None)
    if sec is None or getattr(sec, "usa_cvar", 0) != 1:
        return None, None
    return _first_active(sec.alfa_cvar), _first_active(sec.lambda_cvar)


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
    openings, and cobre rejects a ``num_openings`` on an external-only stage.
    State variables follow the lag-blind convention: storage only, no
    inflow-lag state (only the boundary FCF prices lags).

    ``cvar`` (when the deck runs risk-averse) emits a ``{"cvar": {...}}`` risk
    measure on **every** stage, else ``"expectation"`` on every stage. The
    measure is emitted uniformly rather than only from ``cvar.from_stage_index``
    onward for two coincident reasons: CVaR on a deterministic (single-opening)
    stage collapses to expectation, so a uniform emission is identical in
    effect to gating it at DECOMP's ``AR`` starting period (only the stochastic
    fan is actually bound); and cobre admits the ``gap`` stopping rule under
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
    terminal branch edge carries its DECOMP per-scenario weight (cobre
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
        "$schema": cobre_schemas.schema_url_for("stages.json"),
        "season_definitions": monthly_season_definitions(),
        "policy_graph": {
            "type": "finite_horizon",
            "annual_discount_rate": annual_discount_rate,
            "nodes": nodes,
            "transitions": transitions,
        },
        "stages": stages,
    }
