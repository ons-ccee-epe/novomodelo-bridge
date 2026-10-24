"""Scenario conversion for DECOMP-like decks (the explicit inflow tree).

The inflow file carries the deterministic trunk and the terminal fan:

- **Weekly stages** (first operative month, when it is split into weeks):
  ``previsoes`` holds one row per stage, each a deterministic forecast
  (scenario 0); it is empty for a deck of monthly stages only.
- **Monthly stages** (later months): ``cenarios_gerados`` holds one row per
  (stage, scenario). Deterministic trunk stages have a single row (scenario 1
  → index 0); the terminal fan has one row per opening.

DECOMP's inflow file is the *arquivo de vazões incrementais*: each
gauging-station column already holds the plant's own incremental (local)
inflow, so each plant's value is read straight through from its (post-``AC
NUMPOS``) gauge column — no upstream subtraction. novomodelo re-derives the natural
flow at each plant by routing upstream releases down the ``downstream_id``
cascade.

The tree is emitted node-natively: every stage draws its openings from
``external_inflow_scenarios.parquet`` (trunk column 0, terminal fan columns
``0..N-1``), and the per-scenario weights become the terminal branch-edge
probabilities on the ``policy_graph`` (see ``temporal.convert_stages``). Under
the identity stochastic convention (μ = 0, σ = 1, order 0) the standardized
noise equals the natural value. ``terminal_fan_probabilities`` supplies those
fan weights; ``convert_scenario_probabilities`` remains a validation view for
``check decomp`` and is no longer written to the case directory.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa

from novomodelo_bridge.core.diagnostics import emit
from novomodelo_bridge.core.incremental_inflow import (
    artificial_gauge_plants_diagnostic,
    is_artificial_gauge,
)
from novomodelo_bridge.core.inflow_windows import (
    format_observation_windows,
    month_window,
    previous_months,
)
from novomodelo_bridge.decomp.converters.hydro import _downstream_operated
from novomodelo_bridge.decomp.temporal import weekly_stage_count

if TYPE_CHECKING:
    from collections.abc import Sequence

    from idecomp.decomp import Vazoes

    from novomodelo_bridge.decomp.case import DecompCase
    from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro
    from novomodelo_bridge.decomp.id_map import DecompIdMap
    from novomodelo_bridge.decomp.temporal import OperativeStage

_LOG = logging.getLogger(__name__)

_PROBABILITY_ATOL = 1e-4


def _incremental_context(
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
) -> tuple[dict[int, str], dict[int, list[int]]]:
    """Per-plant station column and direct operated-upstream stations, off
    the *effective* (post-``AC NUMJUS``/``NUMPOS``) topology.

    Stage-agnostic by design (one cascade for the whole horizon): both the
    station column and the downstream link are read at stage 0
    (:meth:`~novomodelo_bridge.decomp.converters.cadastro.effective.
    EffectiveCadastro.inflow_gauge`/
    :func:`~novomodelo_bridge.decomp.converters.hydro.entity.
    _downstream_operated`'s own default).
    A plant whose effective gauge varies across stages (a temporal ``AC
    NUMPOS``) gets a tracked-gap warning here; the downstream sibling gap is
    logged inside ``_downstream_operated`` itself, which this function also
    walks per plant.
    """
    operated = set(id_map.hydro_codes)
    station_by_code: dict[int, str] = {}
    for code in id_map.hydro_codes:
        if effective.inflow_gauge_varies(code):
            _LOG.warning(
                "plant %d's inflow gauge (AC NUMPOS) varies across stages; "
                "per-stage gauge attribution is not modeled (deferred "
                "fidelity) -- using the stage-0 effective gauge for the "
                "whole horizon",
                code,
            )
        station_by_code[code] = str(effective.inflow_gauge(code, 0))
    parents: dict[int, list[int]] = {code: [] for code in id_map.hydro_codes}
    for code in id_map.hydro_codes:
        downstream = _downstream_operated(effective, code, operated)
        if downstream is not None:
            parents[downstream].append(code)
    return station_by_code, parents


def _incremental_values(
    row: pd.Series,
    id_map: DecompIdMap,
    station_by_code: dict[int, str],
) -> list[float]:
    """Per-plant incremental inflows in hydro-id order.

    DECOMP's inflow file is the *arquivo de vazões incrementais*: each
    gauging-station column already holds the plant's own incremental (local)
    inflow, so the value is read straight through — no upstream subtraction.
    Verified against the deck's ``dec_oper_usih`` reported
    ``vazao_incremental``: the direct per-posto read matches every operated
    plant to 0 m³/s, and the reported natural flow reconstructs exactly as the
    running sum of these incrementals down each cascade. (DECOMP operates
    every plant — no fictitious/non-operated intermediates — so there is no
    upstream drainage to re-attribute downstream; ``station_by_code`` already
    resolves any ``AC NUMPOS`` gauge relink.)
    """
    return [float(row[station_by_code[code]]) for code in id_map.hydro_codes]


def recent_observation_rows(
    vazoes: Vazoes, study_start: date
) -> list[tuple[date, date, pd.Series]]:
    """The deck's pre-study observation windows, oldest first, each with its row.

    Each row is one ``vazoes`` observation, a column per posto. The tables are:

    - ``observacoes_mensais`` — the ``N`` full calendar months before the
      study, ``mes`` oldest first (``mes = N`` is the month before the
      study-start month);
    - ``observacoes_semanais`` — the ``W`` 7-day weeks of the study-start month
      before the study begins, ``semana`` oldest first, ending at the study
      start.

    The most recent month is clipped where the weeks begin and a month wholly
    inside them is dropped, so no two windows overlap (a novomodelo requirement of
    ``recent_observations``; ``start == previous end`` is fine).
    """
    monthly = vazoes.observacoes_mensais
    weekly = vazoes.observacoes_semanais
    windows: list[tuple[date, date, pd.Series]] = []

    weekly_floor = study_start
    if weekly is not None and not weekly.empty:
        ordered_weeks = weekly.sort_values("semana")
        n_weeks = len(ordered_weeks)
        weekly_floor = study_start - timedelta(days=n_weeks * 7)
        for offset, (_, row) in enumerate(ordered_weeks.iterrows()):
            start = study_start - timedelta(days=(n_weeks - offset) * 7)
            windows.append((start, start + timedelta(days=7), row))

    if monthly is not None and not monthly.empty:
        ordered_months = monthly.sort_values("mes")
        months = previous_months(
            study_start.year, study_start.month, len(ordered_months)
        )
        for (year, month), (_, row) in zip(
            months, ordered_months.iterrows(), strict=True
        ):
            start, end = month_window(year, month)
            if start < weekly_floor:
                windows.append((start, min(end, weekly_floor), row))

    windows.sort(key=lambda window: window[0])
    return windows


def convert_recent_observation_windows(
    vazoes: Vazoes,
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
) -> list[dict]:
    """``initial_conditions.recent_observations``: the pre-study observed inflows
    (:func:`recent_observation_rows`) seeding novomodelo's PAR inflow-lag accumulator.

    Read on each plant's boundary-cut coupling gauge
    (:meth:`~novomodelo_bridge.decomp.converters.cadastro.effective.EffectiveCadastro.coupling_gauge`),
    not its operation gauge: the seed only feeds the lag terms of the boundary
    cut, which the source model evaluates there. novomodelo casts these windows onto
    the monthly lag periods before stage 0 (``derive_inflow_seeds``) and the
    forward pass completes the accumulator at the coupling.

    The monthly history is deliberately not routed to
    ``scenarios/inflow_history.parquet``: novomodelo fits AR seasonal statistics from
    that file (≥ 2 observations per season), which the single-realization
    DECOMP tendency cannot satisfy. Returns ``[]`` when the deck carries neither
    observation table.
    """
    stations = [str(effective.coupling_gauge(code)) for code in id_map.hydro_codes]
    rows = [
        (hydro_id, start, end, float(row[station]))
        for start, end, row in recent_observation_rows(vazoes, calendar[0].start_date)
        for hydro_id, station in enumerate(stations)
    ]
    rows.sort(key=lambda entry: (entry[0], entry[1]))
    return format_observation_windows(rows)


def _tree_values(
    vazoes: Vazoes,
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
) -> dict[tuple[int, int], list[float]]:
    """``{(stage_index, scenario_index): incrementals}`` for the whole tree.

    DECOMP's inflow file splits forecasts across two tables:

    - ``previsoes`` — one row per **weekly** stage (the first operative month)
    - ``cenarios_gerados`` — one row per (stage, scenario) for **monthly**
      stages (the deterministic monthly trunk stages + the terminal fan)

    For a multi-month deck the monthly trunk stages (deterministic, scenario 1)
    come from ``cenarios_gerados``, not ``previsoes``. The terminal fan's
    multiple scenarios also come from ``cenarios_gerados``.
    """
    station_by_code, _ = _incremental_context(effective, id_map)
    terminal = len(calendar) - 1
    n_weekly = weekly_stage_count(calendar)

    previsoes = vazoes.previsoes
    n_forecasts = 0 if previsoes is None else len(previsoes)
    if n_forecasts != n_weekly:
        raise ValueError(
            f"{n_forecasts} weekly forecast rows (previsoes) for {n_weekly} "
            "weekly stages"
        )

    values: dict[tuple[int, int], list[float]] = {}
    for _, row in () if previsoes is None else previsoes.iterrows():
        stage_index = int(row["estagio"]) - 1
        if not 0 <= stage_index < n_weekly:
            raise ValueError(
                f"trunk forecast stage {int(row['estagio'])} outside the "
                f"weekly range (1..{n_weekly})"
            )
        values[(stage_index, 0)] = _incremental_values(row, id_map, station_by_code)

    cenarios = vazoes.cenarios_gerados
    if cenarios is None or cenarios.empty:
        raise ValueError("the inflow file has no terminal fan (cenarios_gerados)")
    for _, row in cenarios.iterrows():
        stage_index = int(row["estagio"]) - 1
        scenario_index = int(row["cenario"]) - 1
        if not n_weekly <= stage_index <= terminal:
            raise ValueError(
                f"cenarios_gerados row at stage {int(row['estagio'])} outside "
                f"the monthly range ({n_weekly + 1}..{terminal + 1})"
            )
        if stage_index < terminal and scenario_index != 0:
            raise ValueError(
                f"monthly trunk stage {int(row['estagio'])} has scenario "
                f"{int(row['cenario'])}; only the terminal stage may fan"
            )
        values[(stage_index, scenario_index)] = _incremental_values(
            row, id_map, station_by_code
        )
    return values


def _report_artificial_gauge_plants(
    effective: EffectiveCadastro, id_map: DecompIdMap
) -> None:
    plants = [
        (
            code,
            str(effective.base.loc[code, "nome_usina"]).strip(),
            id_map.hydro_id(code),
        )
        for code in id_map.hydro_codes
        if is_artificial_gauge(effective.inflow_gauge(code, 0))
    ]
    if plants:
        emit(artificial_gauge_plants_diagnostic(plants), logger=_LOG)


def convert_external_inflows(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    vazoes: Vazoes,
    effective: EffectiveCadastro,
) -> pa.Table:
    """``external_inflow_scenarios.parquet``: the tree of per-plant incremental
    inflows, read off each plant's *effective* (post-``AC NUMPOS``) gauge
    column."""
    _report_artificial_gauge_plants(effective, id_map)
    values = _tree_values(vazoes, effective, id_map, case.calendar)

    stage_ids: list[int] = []
    scenario_ids: list[int] = []
    hydro_ids: list[int] = []
    flows: list[float] = []
    for (stage_index, scenario_index), incrementals in sorted(values.items()):
        for hydro_id, value in enumerate(incrementals):
            stage_ids.append(stage_index)
            scenario_ids.append(scenario_index)
            hydro_ids.append(hydro_id)
            flows.append(value)

    return pa.table(
        {
            "stage_id": pa.array(stage_ids, type=pa.int32()),
            "scenario_id": pa.array(scenario_ids, type=pa.int32()),
            "hydro_id": pa.array(hydro_ids, type=pa.int32()),
            "value_m3s": pa.array(flows, type=pa.float64()),
        }
    )


def _reject_branching_trunk(prob: pd.DataFrame, terminal: int) -> None:
    """Fail unless every stage before *terminal* is deterministic.

    The node graph fans only at the terminal stage (``build_node_graph``), so
    the trunk — every 1-based stage ``1..terminal-1`` — must carry a single
    inflow opening. A trunk stage with more than one ``cenario`` is DECOMP's
    growing scenario tree (branching before the last month), which this
    terminal-fan conversion cannot represent and must not silently flatten.
    """
    stage = prob["estagio"].astype(int)
    trunk = prob[(stage >= 1) & (stage < terminal)]
    branching = sorted(
        {
            int(s)
            for s, n in trunk.groupby("estagio")["cenario"].nunique().items()
            if n > 1
        }
    )
    if branching:
        raise ValueError(
            f"inflow stages {branching} branch into multiple scenarios, but "
            f"only the terminal stage {terminal} may fan. DECOMP's scenario "
            f"tree branches from the second month on; a deck that branches "
            f"before its last stage is not convertible by the terminal-fan "
            f"model."
        )


def terminal_fan_probabilities(
    vazoes: Vazoes,
    calendar: Sequence[OperativeStage],
) -> list[float]:
    """Terminal-stage per-scenario probabilities, ordered by scenario id.

    The DECOMP fan probabilities (``vazoes.probabilidades`` at the terminal
    stage) become the terminal branch-edge weights on the ``policy_graph``
    node graph; novomodelo re-normalizes each source's out-edges at load. The
    returned list is indexed by 0-based scenario id (``cenario - 1``), the
    same id the external inflow library binds via each fan node's
    ``scenario_id``. Validated to be a contiguous 0-based range summing to 1.

    Rejects (via :func:`_reject_branching_trunk`) a deck whose trunk already
    branches, since only the terminal stage may fan.
    """
    prob = vazoes.probabilidades
    if prob is None or prob.empty:
        raise ValueError("the inflow file has no probability table")

    terminal = len(calendar)  # 1-based id of the terminal (fan) stage
    _reject_branching_trunk(prob, terminal)
    rows = prob[prob["estagio"].astype(int) == terminal]
    if rows.empty:
        raise ValueError(
            f"the inflow probability table has no rows for the terminal "
            f"stage {terminal}"
        )

    by_scenario: dict[int, float] = {}
    for _, row in rows.iterrows():
        by_scenario[int(row["cenario"]) - 1] = float(row["probabilidade"])

    if set(by_scenario) != set(range(len(by_scenario))):
        raise ValueError(
            "terminal fan scenarios are not a contiguous 0-based range: "
            f"{sorted(by_scenario)}"
        )
    ordered = [by_scenario[k] for k in range(len(by_scenario))]

    total = sum(ordered)
    if abs(total - 1.0) > _PROBABILITY_ATOL:
        raise ValueError(f"terminal fan probabilities sum to {total}, expected 1.0")
    return ordered


def deterministic_external_scenarios(
    stats: pa.Table,
    *,
    entity_column: str,
    value_in: str,
    value_out: str,
    scenario_counts: Sequence[int],
) -> pa.Table:
    """Replicate a deterministic per-(entity, stage) value across external columns.

    ``stats`` carries one row per (``entity_column``, ``stage_id``) with the
    deterministic value in ``value_in`` (load base MW, or NCS availability
    fraction). Each row expands to ``scenario_counts[stage_id]`` external rows
    (``scenario_id`` ``0..n-1``) with the value repeated, so the library's
    per-stage column count matches the inflow library (1 on the deterministic
    trunk, the terminal fan width). This satisfies novomodelo's node-native rule
    that every non-empty class at an external-column node be external: the
    DECOMP load and NCS are deterministic, so a single value fans out unchanged.
    """
    entities = stats.column(entity_column).to_pylist()
    stages = stats.column("stage_id").to_pylist()
    values = stats.column(value_in).to_pylist()

    rows: list[tuple[int, int, int, float]] = []
    for entity, stage, value in zip(entities, stages, values, strict=True):
        for scenario_id in range(scenario_counts[int(stage)]):
            rows.append((int(stage), scenario_id, int(entity), float(value)))
    # Sort by (stage_id, scenario_id, entity) to match the inflow emitter and
    # novomodelo's canonical external-library order.
    rows.sort()

    return pa.table(
        {
            "stage_id": pa.array([r[0] for r in rows], type=pa.int32()),
            "scenario_id": pa.array([r[1] for r in rows], type=pa.int32()),
            entity_column: pa.array([r[2] for r in rows], type=pa.int32()),
            value_out: pa.array([r[3] for r in rows], type=pa.float64()),
        }
    )


def convert_inflow_stats_identity(
    case: DecompCase,
    id_map: DecompIdMap,
) -> pa.Table:
    """Identity stats (μ = 0, σ = 1) — the explicit-inflow convention.

    The stochastic model is plumbing here: with these values the
    standardized noise is the incremental inflow itself, and no fan stage can
    hit the zero-σ pathology.
    """
    hydro_ids: list[int] = []
    stage_ids: list[int] = []
    for hydro_id in range(len(id_map.hydro_codes)):
        for stage in case.calendar:
            hydro_ids.append(hydro_id)
            stage_ids.append(stage.index)
    n = len(hydro_ids)
    return pa.table(
        {
            "hydro_id": pa.array(hydro_ids, type=pa.int32()),
            "stage_id": pa.array(stage_ids, type=pa.int32()),
            "mean_m3s": pa.array([0.0] * n, type=pa.float64()),
            "std_m3s": pa.array([1.0] * n, type=pa.float64()),
        }
    )


def convert_scenario_probabilities(
    vazoes: Vazoes,
    calendar: Sequence[OperativeStage],
) -> pa.Table:
    """Per-(stage, scenario) probability view for validation.

    Trunk rows carry probability 1.0; validated to sum to 1 per stage. No
    longer written to the case directory (the terminal weights now live on
    the ``policy_graph`` transitions); ``check decomp`` still uses this to
    validate the deck's probability table.
    """
    prob = vazoes.probabilidades
    if prob is None or prob.empty:
        raise ValueError("the inflow file has no probability table")

    stage_ids: list[int] = []
    scenario_ids: list[int] = []
    probabilities: list[float] = []
    for _, row in prob.iterrows():
        stage_index = int(row["estagio"]) - 1
        if not 0 <= stage_index < len(calendar):
            raise ValueError(
                f"probability row at stage {int(row['estagio'])} outside the "
                f"calendar (1..{len(calendar)})"
            )
        stage_ids.append(stage_index)
        scenario_ids.append(int(row["cenario"]) - 1)
        probabilities.append(float(row["probabilidade"]))

    sums: dict[int, float] = {}
    for stage_index, probability in zip(stage_ids, probabilities, strict=True):
        sums[stage_index] = sums.get(stage_index, 0.0) + probability
    for stage_index, total in sorted(sums.items()):
        if abs(total - 1.0) > _PROBABILITY_ATOL:
            raise ValueError(
                f"stage {stage_index}: scenario probabilities sum to {total}, "
                "expected 1.0"
            )

    return pa.table(
        {
            "stage_id": pa.array(stage_ids, type=pa.int32()),
            "scenario_id": pa.array(scenario_ids, type=pa.int32()),
            "probability": pa.array(probabilities, type=pa.float64()),
        }
    )
