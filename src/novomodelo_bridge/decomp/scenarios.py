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
cascade. The one exception is a group of NW plants (:class:`NwInflowSplit`),
which takes NEWAVE's split of the flow from the coupling tables.

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
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING, Protocol

import pandas as pd
import pyarrow as pa

from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    emit,
)
from novomodelo_bridge.core.errors import FieldParseError, UnsupportedSourceError
from novomodelo_bridge.core.incremental_inflow import (
    artificial_gauge_plants_diagnostic,
    is_artificial_gauge,
)
from novomodelo_bridge.core.inflow_windows import (
    format_observation_windows,
    month_window,
    previous_months,
)
from novomodelo_bridge.decomp.converters.hydro import _downstream_in
from novomodelo_bridge.decomp.temporal import weekly_stage_count

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from novomodelo_bridge.decomp.case import DecompCase
    from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro
    from novomodelo_bridge.decomp.id_map import DecompIdMap
    from novomodelo_bridge.decomp.temporal import OperativeStage

_LOG = logging.getLogger(__name__)

_PROBABILITY_ATOL = 1e-4
_VAZOES = "vazoes.rvN"

#: How far the operation and coupling inflow tables may differ, on any row, for an
#: NW plant's inflow to be split from its downstream plant's: the rounding of the
#: tables' whole-m³/s values over one NW plant and its absorber.
NW_SPLIT_TOLERANCE_M3S = 1.0

_WEEKLY_KEYS = ("estagio",)
_MONTHLY_KEYS = ("estagio", "cenario")
_PAIRED_TABLES = (
    ("previsoes", "previsoes_com_postos_artificiais", _WEEKLY_KEYS),
    (
        "cenarios_gerados",
        "cenarios_calculados_com_postos_artificiais",
        _MONTHLY_KEYS,
    ),
)


class ObservationTables(Protocol):
    """The pre-study observation tables of a ``vazoes`` file."""

    @property
    def observacoes_mensais(self) -> pd.DataFrame | None: ...

    @property
    def observacoes_semanais(self) -> pd.DataFrame | None: ...


class TreeTables(Protocol):
    """The operation and coupling trunk tables of a ``vazoes`` file."""

    @property
    def previsoes(self) -> pd.DataFrame | None: ...

    @property
    def cenarios_gerados(self) -> pd.DataFrame | None: ...

    @property
    def previsoes_com_postos_artificiais(self) -> pd.DataFrame | None: ...

    @property
    def cenarios_calculados_com_postos_artificiais(self) -> pd.DataFrame | None: ...


class _ProbabilityTable(Protocol):
    @property
    def probabilidades(self) -> pd.DataFrame | None: ...


class _CouplingGauges(Protocol):
    def coupling_gauge(self, code: int) -> int: ...


def _incremental_context(
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
) -> tuple[dict[int, str], dict[int, list[int]]]:
    """Per-plant station column and direct upstream stations of the converted
    plants, off the *effective* (post-``AC NUMJUS``/``NUMPOS``) topology.

    Stage-agnostic by design (one cascade for the whole horizon): both the
    station column and the downstream link are read at stage 0
    (:meth:`~novomodelo_bridge.decomp.converters.cadastro.effective.
    EffectiveCadastro.inflow_gauge`/
    :func:`~novomodelo_bridge.decomp.converters.hydro.entity.
    _downstream_in`'s own default).
    A gauge or link that changes on later stages is reported
    (:func:`novomodelo_bridge.decomp.converters.cadastro.topology_varies_diagnostic`).
    """
    converted = set(id_map.hydro_codes)
    station_by_code: dict[int, str] = {}
    for code in id_map.hydro_codes:
        station_by_code[code] = str(effective.inflow_gauge(code, 0))
    parents: dict[int, list[int]] = {code: [] for code in id_map.hydro_codes}
    for code in id_map.hydro_codes:
        downstream = _downstream_in(effective, code, converted)
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
    plant to 0 m³/s, except that a negative value is reported there as zero
    (:func:`_report_negative_incrementals`), and the reported natural flow
    reconstructs exactly as the running sum of these incrementals down each
    cascade. ``station_by_code`` already resolves any ``AC NUMPOS`` gauge
    relink. A dormant plant is read the same way, and the plants of a passed NW
    group are then replaced by :meth:`NwInflowSplit.carried`.
    """
    return [float(row[station_by_code[code]]) for code in id_map.hydro_codes]


def posto_value(row: pd.Series, posto: int) -> float:
    """*row*'s value on the ``posto`` column (named by the posto number)."""
    column = str(posto)
    if column not in row.index:
        raise ValueError(f"the inflow file has no posto {posto} column")
    return float(row[column])


@dataclass(frozen=True)
class NwInflowSplit:
    """How the inflow of each NW plant is split from the plant below it.

    An NW plant is one DECOMP does not operate. The operation tables give the
    flow of a *group* (the NW plants whose first operated plant below is the
    same *absorber*) to the absorber, and give each NW plant none. NEWAVE's
    coupling tables split that flow plant by plant, and the boundary cuts price
    that split. A group that passes the guard (:func:`split_nw_inflows`) takes
    it: each NW plant carries its coupling inflow and the absorber the rest of
    its operation inflow. A group that fails keeps the operation series.

    Attributes
    ----------
    passed:
        Absorber code -> the codes of its NW plants, for the groups that pass.
    failed:
        The same, for the groups that fail.
    max_balance_gap_m3s:
        Per failed absorber, the largest ``|op[absorber] - Σ c[absorber and
        its NW plants]|`` over the paired rows.
    max_nw_operation_m3s:
        Per failed absorber, the largest ``|op[NW plant]|`` over the paired
        rows and the group's NW plants.
    unabsorbed:
        The NW plants with no operated plant below, which keep the operation
        series.
    """

    passed: Mapping[int, tuple[int, ...]] = field(default_factory=dict)
    failed: Mapping[int, tuple[int, ...]] = field(default_factory=dict)
    max_balance_gap_m3s: Mapping[int, float] = field(default_factory=dict)
    max_nw_operation_m3s: Mapping[int, float] = field(default_factory=dict)
    unabsorbed: tuple[int, ...] = ()

    @property
    def carried_codes(self) -> tuple[int, ...]:
        """The absorbers and NW plants of the passed groups."""
        return tuple(
            code
            for absorber, nw_codes in self.passed.items()
            for code in (absorber, *nw_codes)
        )

    def carried(
        self,
        code: int,
        operation_row: pd.Series,
        coupling_row: pd.Series,
        effective: EffectiveCadastro,
    ) -> float:
        """The inflow the case carries for plant *code* on one paired row.

        An NW plant of a passed group carries its coupling inflow, and its
        absorber the operation inflow less its NW plants' coupling inflows.
        Every other plant carries its operation inflow.
        """

        def operation(plant: int) -> float:
            return posto_value(operation_row, effective.inflow_gauge(plant, 0))

        def coupling(plant: int) -> float:
            return posto_value(coupling_row, effective.coupling_gauge(plant))

        nw_codes = self.passed.get(code)
        if nw_codes is not None:
            return operation(code) - sum(coupling(nw) for nw in nw_codes)
        if code in self.carried_codes:
            return coupling(code)
        return operation(code)


def _row_key(row: pd.Series, keys: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(int(row[key]) for key in keys)


def _coupling_rows(
    table: pd.DataFrame | None, keys: tuple[str, ...]
) -> dict[tuple[int, ...], pd.Series]:
    if table is None:
        return {}
    return {_row_key(row, keys): row for _, row in table.iterrows()}


def _paired_rows(vazoes: TreeTables) -> list[tuple[pd.Series, pd.Series]]:
    """Every operation row with its coupling row: ``previsoes`` on ``estagio``,
    ``cenarios_gerados`` on ``estagio`` and ``cenario``."""
    pairs: list[tuple[pd.Series, pd.Series]] = []
    for operation_name, coupling_name, keys in _PAIRED_TABLES:
        operation: pd.DataFrame | None = getattr(vazoes, operation_name)
        if operation is None:
            continue
        coupling_rows = _coupling_rows(getattr(vazoes, coupling_name), keys)
        for _, row in operation.iterrows():
            key = _row_key(row, keys)
            if key not in coupling_rows:
                at = ", ".join(f"{k} {v}" for k, v in zip(keys, key, strict=True))
                raise FieldParseError(
                    f"the inflow file's {operation_name} row at {at} has no "
                    f"row in {coupling_name}",
                    path=_VAZOES,
                    field=operation_name,
                )
            pairs.append((row, coupling_rows[key]))
    return pairs


def split_nw_inflows(
    vazoes: TreeTables,
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
) -> NwInflowSplit:
    """Group the NW plants by absorber and run the guard on every paired row.

    A group passes when, on every row, both checks hold within
    :data:`NW_SPLIT_TOLERANCE_M3S` (inclusive): the absorber's operation
    inflow equals the sum of the coupling inflows of the absorber and its NW
    plants, and each NW plant's own operation inflow, which the split
    discards, is near zero. A deck without NW plants gets an empty split and
    its coupling tables are not read.

    Raises
    ------
    FieldParseError
        If an NW plant's coupling table is missing, or an operation row has no
        coupling row to pair with.
    """
    dormant = id_map.dormant_hydro_codes
    if not dormant:
        return NwInflowSplit()
    for _, coupling_name, _ in _PAIRED_TABLES:
        if getattr(vazoes, coupling_name) is None:
            plural = "s" if len(dormant) > 1 else ""
            raise FieldParseError(
                f"cannot split the inflow of NW plant{plural} "
                f"{', '.join(map(str, dormant))}: the inflow file has no "
                f"{coupling_name} table",
                path=_VAZOES,
                field=coupling_name,
            )

    operated = set(id_map.operated_hydro_codes)
    groups: dict[int, list[int]] = defaultdict(list)
    unabsorbed: list[int] = []
    for code in dormant:
        absorber = _downstream_in(effective, code, operated)
        if absorber is None:
            unabsorbed.append(code)
        else:
            groups[absorber].append(code)

    max_gap = dict.fromkeys(groups, 0.0)
    max_nw = dict.fromkeys(groups, 0.0)
    for operation_row, coupling_row in _paired_rows(vazoes):
        for absorber, nw_codes in groups.items():
            coupled = sum(
                posto_value(coupling_row, effective.coupling_gauge(code))
                for code in (absorber, *nw_codes)
            )
            operated_flow = posto_value(
                operation_row, effective.inflow_gauge(absorber, 0)
            )
            max_gap[absorber] = max(max_gap[absorber], abs(operated_flow - coupled))
            max_nw[absorber] = max(
                max_nw[absorber],
                *(
                    abs(posto_value(operation_row, effective.inflow_gauge(code, 0)))
                    for code in nw_codes
                ),
            )

    passes = {
        absorber: max_gap[absorber] <= NW_SPLIT_TOLERANCE_M3S
        and max_nw[absorber] <= NW_SPLIT_TOLERANCE_M3S
        for absorber in groups
    }
    return NwInflowSplit(
        passed={a: tuple(nw) for a, nw in groups.items() if passes[a]},
        failed={a: tuple(nw) for a, nw in groups.items() if not passes[a]},
        max_balance_gap_m3s={a: g for a, g in max_gap.items() if not passes[a]},
        max_nw_operation_m3s={a: v for a, v in max_nw.items() if not passes[a]},
        unabsorbed=tuple(unabsorbed),
    )


def recent_observation_rows(
    vazoes: ObservationTables, first_stage: OperativeStage
) -> list[tuple[date, date, pd.Series]]:
    """The deck's pre-study observation windows, oldest first, each with its row.

    Each row is one ``vazoes`` observation, a column per posto. The tables are:

    - ``observacoes_mensais`` — the ``N`` calendar months before the first
      operative month, ``mes`` oldest first (``mes = N`` is the month before
      it). The operative month is the first stage's season, not the month of
      its start: an RV0 whose first week starts on 27/12/2025 studies January
      2026, so its ``mes = N`` is December 2025;
    - ``observacoes_semanais`` — the ``W`` 7-day weeks of the operative month
      before the study begins, ``semana`` oldest first, ending at the study
      start.

    The windows stop at the operative month's start: a first week that starts
    in the previous month is clipped to it, and the most recent month ends
    there (or where the study begins, if earlier), keeping its whole observed
    value: the source model drops a first week's previous-month days from
    every lag month. No two windows overlap, as
    novomodelo requires of ``recent_observations`` (``start == previous end`` is
    fine).
    """
    study_start = first_stage.start_date
    operative_start = first_stage.operative_month_start
    monthly = vazoes.observacoes_mensais
    weekly = vazoes.observacoes_semanais
    windows: list[tuple[date, date, pd.Series]] = []

    if weekly is not None and not weekly.empty:
        ordered_weeks = weekly.sort_values("semana")
        n_weeks = len(ordered_weeks)
        for offset, (_, row) in enumerate(ordered_weeks.iterrows()):
            start = study_start - timedelta(days=(n_weeks - offset) * 7)
            end = start + timedelta(days=7)
            if end > operative_start:
                windows.append((max(start, operative_start), end, row))

    if monthly is not None and not monthly.empty:
        ordered_months = monthly.sort_values("mes")
        months = previous_months(
            operative_start.year, operative_start.month, len(ordered_months)
        )
        month_ceiling = min(operative_start, study_start)
        for (year, month), (_, row) in zip(
            months, ordered_months.iterrows(), strict=True
        ):
            start, end = month_window(year, month)
            if start < month_ceiling:
                windows.append((start, min(end, month_ceiling), row))

    windows.sort(key=lambda window: window[0])
    return windows


def convert_recent_observation_windows(
    vazoes: ObservationTables,
    effective: _CouplingGauges,
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
        for start, end, row in recent_observation_rows(vazoes, calendar[0])
        for hydro_id, station in enumerate(stations)
    ]
    rows.sort(key=lambda entry: (entry[0], entry[1]))
    return format_observation_windows(rows)


def _tree_values(
    vazoes: TreeTables,
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
    split: NwInflowSplit,
) -> dict[tuple[int, int], list[float]]:
    """``{(stage_index, scenario_index): incrementals}`` for the whole tree.

    DECOMP's inflow file splits forecasts across two tables:

    - ``previsoes`` — one row per **weekly** stage (the first operative month)
    - ``cenarios_gerados`` — one row per (stage, scenario) for **monthly**
      stages (the deterministic monthly trunk stages + the terminal fan)

    For a multi-month deck the monthly trunk stages (deterministic, scenario 1)
    come from ``cenarios_gerados``, not ``previsoes``. The terminal fan's
    multiple scenarios also come from ``cenarios_gerados``.

    Each plant reads its operation gauge, except the plants of a passed NW
    group (:meth:`NwInflowSplit.carried`).
    """
    station_by_code, _ = _incremental_context(effective, id_map)
    weekly_coupling = _coupling_rows(
        vazoes.previsoes_com_postos_artificiais if split.passed else None,
        _WEEKLY_KEYS,
    )
    monthly_coupling = _coupling_rows(
        vazoes.cenarios_calculados_com_postos_artificiais if split.passed else None,
        _MONTHLY_KEYS,
    )

    def row_values(
        row: pd.Series,
        coupling_rows: dict[tuple[int, ...], pd.Series],
        keys: tuple[str, ...],
    ) -> list[float]:
        incrementals = _incremental_values(row, id_map, station_by_code)
        for code in split.carried_codes:
            incrementals[id_map.hydro_id(code)] = split.carried(
                code, row, coupling_rows[_row_key(row, keys)], effective
            )
        return incrementals

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
        values[(stage_index, 0)] = row_values(row, weekly_coupling, _WEEKLY_KEYS)

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
            raise FieldParseError(
                f"vazoes.rvN gives monthly stage {int(row['estagio'])} (counted "
                f"from 1) inflow scenario {int(row['cenario'])}, but its "
                "probability table gives every stage before the last one a "
                "single scenario.",
                path=_VAZOES,
                field="cenarios_gerados cenario",
            )
        values[(stage_index, scenario_index)] = row_values(
            row, monthly_coupling, _MONTHLY_KEYS
        )
    return values


def _report_artificial_gauge_plants(
    effective: EffectiveCadastro, id_map: DecompIdMap
) -> None:
    plants = [
        (code, str(effective.base.loc[code, "nome_usina"]).strip())
        for code in id_map.hydro_codes
        if is_artificial_gauge(effective.inflow_gauge(code, 0))
    ]
    if plants:
        emit(artificial_gauge_plants_diagnostic(plants), logger=_LOG)


def _report_negative_incrementals(
    values: Mapping[tuple[int, int], Sequence[float]],
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
) -> None:
    """WARN on the tree's negative incremental inflows, kept as the deck has them."""
    negatives: dict[tuple[int, int], list[float]] = defaultdict(list)
    for (stage_index, _), incrementals in values.items():
        for code, value in zip(id_map.hydro_codes, incrementals, strict=True):
            if value < 0.0:
                negatives[(code, stage_index)].append(value)
    if not negatives:
        return
    rows = [
        [
            code,
            str(effective.base.loc[code, "nome_usina"]).strip(),
            effective.inflow_gauge(code, 0),
            stage_index,
            len(found),
            min(found),
        ]
        for (code, stage_index), found in sorted(negatives.items())
    ]
    plants = len({code for code, _ in negatives})
    emit(
        Diagnostic(
            code="decomp-negative-incremental-inflows",
            severity=Severity.WARNING,
            category="Inflow scenarios",
            title=f"Negative incremental inflows kept ({plants})",
            summary=(
                "The listed plants have a negative incremental inflow in "
                "vazoes.rvN. The conversion keeps each value, so the case's "
                "water balance removes that water unless its priced inflow "
                "slack adds it back. DECOMP reports these incremental inflows "
                "as zero in its operation output."
            ),
            table=DiagnosticTable(
                columns=[
                    "Code",
                    "Name",
                    "Posto",
                    "Stage",
                    "Scenarios",
                    "Lowest (m³/s)",
                ],
                rows=rows,
                justify=["right", "left", "right", "right", "right", "right"],
                caption="Stage ids are 0-based.",
            ),
            notes=[
                "A negative inflow below a plant whose outflow is capped can "
                "force that plant to spill or violate its outflow limit."
            ],
        ),
        logger=_LOG,
    )


def _report_nw_split_fallback(
    split: NwInflowSplit, effective: EffectiveCadastro
) -> None:
    """WARN on the NW plants that keep the operation inflows."""

    def name(code: int) -> str:
        return str(effective.base.loc[code, "nome_usina"]).strip()

    rows: list[list[object]] = [
        [
            absorber,
            name(absorber),
            ", ".join(map(str, nw_codes)),
            split.max_balance_gap_m3s[absorber],
            split.max_nw_operation_m3s[absorber],
        ]
        for absorber, nw_codes in split.failed.items()
    ]
    rows.extend([code, name(code), str(code), "-", "-"] for code in split.unabsorbed)
    if not rows:
        return
    emit(
        Diagnostic(
            code="decomp-nw-inflow-split-fallback",
            severity=Severity.WARNING,
            category="Inflow scenarios",
            title=f"NW plant inflows keep the operation series ({len(rows)})",
            summary=(
                "DECOMP's operation inflows (vazoes.rvN) give the flow of an NW "
                "plant to the first operated plant below it, but its boundary "
                "cuts read each NW plant on a coupling posto of its own. The "
                "conversion gives each NW plant its own coupling inflow, and "
                "the operated plant below it the rest, only when the operation "
                f"and coupling inflows agree within {NW_SPLIT_TOLERANCE_M3S:g} "
                "m³/s on every row. The listed plants do not, or have no "
                "operated plant below them, so they keep the operation inflows "
                "and the boundary cost of their inflow terms can differ from "
                "DECOMP's."
            ),
            table=DiagnosticTable(
                columns=[
                    "Code",
                    "Name",
                    "NW plants",
                    "Largest balance gap (m³/s)",
                    "Largest NW operation (m³/s)",
                ],
                rows=rows,
                justify=["right", "left", "left", "right", "right"],
            ),
            notes=[
                "The balance gap is the greatest difference, over the rows of "
                "vazoes.rvN, between the operated plant's operation inflow and "
                "the sum of its group's coupling inflows. The NW operation is "
                "the greatest operation inflow of the group's NW plants, which "
                "the split would discard.",
                "An NW plant with no operated plant below it has no group to "
                "split and is listed on its own row.",
            ],
        ),
        logger=_LOG,
    )


def convert_external_inflows(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    vazoes: TreeTables,
    effective: EffectiveCadastro,
) -> pa.Table:
    """``external_inflow_scenarios.parquet``: the tree of per-plant incremental
    inflows, read off each plant's *effective* (post-``AC NUMPOS``) gauge
    column, except the plants of an NW group that passes the guard
    (:func:`split_nw_inflows`)."""
    split = split_nw_inflows(vazoes, effective, id_map)
    _report_artificial_gauge_plants(effective, id_map)
    values = _tree_values(vazoes, effective, id_map, case.calendar, split)
    _report_nw_split_fallback(split, effective)
    _report_negative_incrementals(values, effective, id_map)

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


def expected_incremental_inflows(
    vazoes: TreeTables,
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
    fan_probabilities: Sequence[float],
) -> dict[tuple[int, int], float]:
    """``{(plant code, stage_index): incremental inflow m³/s}``, as the deck
    gives it (negative values included), with the NW split of
    :func:`split_nw_inflows`. It reports nothing: the conversion's own read
    (:func:`convert_external_inflows`) does.

    A trunk stage takes its single forecast; the terminal stage takes its fan's
    mean weighted by *fan_probabilities* (from
    :func:`terminal_fan_probabilities`), never the uniform mean.
    """
    terminal = len(calendar) - 1
    split = split_nw_inflows(vazoes, effective, id_map)
    expected: dict[tuple[int, int], float] = defaultdict(float)
    for (stage_index, scenario_index), incrementals in _tree_values(
        vazoes, effective, id_map, calendar, split
    ).items():
        weight = fan_probabilities[scenario_index] if stage_index == terminal else 1.0
        for code, value in zip(id_map.hydro_codes, incrementals, strict=True):
            expected[(code, stage_index)] += weight * value
    return dict(expected)


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
        stages = ", ".join(str(stage) for stage in branching)
        raise UnsupportedSourceError(
            f"vazoes.rvN branches the inflow scenario tree at stages {stages} "
            f"(counted from 1), before its last stage {terminal}. A tree that "
            "branches before its last stage is not converted yet, so generate "
            "vazoes.rvN with a tree that branches only there.",
            path=_VAZOES,
            field="probabilidades cenario",
        )


def terminal_fan_probabilities(
    vazoes: _ProbabilityTable,
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
    vazoes: _ProbabilityTable,
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
