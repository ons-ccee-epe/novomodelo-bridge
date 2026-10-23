"""How the converted hydros take the boundary cut's plant terms.

The source model couples its boundary cut plant by plant:

- a plant with its own cut row takes it;
- a plant inside a complex of the cut takes the complex's row (``CX``);
- any other plant takes the row of the first plant downstream that has one,
  which the source model reports as "seu acoplamento sera feito com a usina
  de jusante". This rule is inferred from that report and the coupling
  memory of one deck, not from the manual.

Only a reservoir (``tipo_regulacao`` ``M``/``S``) takes a storage term, priced
above its minimum volume. The inflow-lag terms are evaluated on each plant's
coupling posto (``AC NPOSNW``, else the ``hidr.dat`` posto) and on the
``vazoes`` tables with the artificial postos computed, while novomodelo's lag state
follows the plant's operation inflow. The deterministic part of that
difference is returned as a per-depth offset for the mapper to fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from novomodelo_bridge.core.inflow_windows import month_window
from novomodelo_bridge.decomp.converters.cadastro import is_reservoir
from novomodelo_bridge.decomp.fcf.mapper import HydroCoupling
from novomodelo_bridge.decomp.scenarios import recent_observation_rows

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from datetime import date

    import pandas as pd
    from idecomp.decomp import Vazoes

    from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro
    from novomodelo_bridge.decomp.id_map import DecompIdMap
    from novomodelo_bridge.decomp.temporal import OperativeStage

#: Calendar-month depths of the cut's inflow-lag axis.
LAG_DEPTHS = 12

_OFFSET_TOLERANCE = 1e-9


@dataclass(frozen=True)
class BorrowedOwner:
    """A converted plant taking another cut plant's terms."""

    plant_code: int
    owner_code: int
    via_complex: bool


@dataclass(frozen=True)
class CouplingResolution:
    """Every converted hydro's coupling, plus the plants that borrow or have none."""

    couplings: tuple[HydroCoupling, ...]
    borrowed: tuple[BorrowedOwner, ...]
    unowned: tuple[int, ...]


@dataclass(frozen=True)
class TerminalShare:
    """A plant whose terminal-stage coupling inflow novomodelo's lag state misses.

    ``max_gap_m3s`` is the largest difference over the terminal scenarios
    between the coupling inflow and the inflow novomodelo's state carries.
    """

    plant_code: int
    coupling_posto: int
    operation_posto: int | None
    max_gap_m3s: float


def lag_months(cut_year: int, cut_month: int) -> tuple[tuple[int, int], ...]:
    """The calendar month of each lag depth; depth 1 is the cut's own month."""
    months: list[tuple[int, int]] = []
    year, month = cut_year, cut_month
    for _ in range(LAG_DEPTHS):
        months.append((year, month))
        year, month = (year - 1, 12) if month == 1 else (year, month - 1)
    return tuple(months)


def resolve_couplings(
    plant_codes: Sequence[int],
    id_map: DecompIdMap,
    effective: EffectiveCadastro,
    complexo_components: Mapping[int, Sequence[int]],
) -> CouplingResolution:
    """Resolve the cut row each converted hydro takes.

    The downstream walk and the storage floor read the last stage, where the
    cut attaches. A walk that reaches a complex component stops at the
    complex. A plant whose walk finds no row is left uncoupled and listed in
    ``unowned``.
    """
    row_of = {code: index for index, code in enumerate(plant_codes)}
    complex_of = {
        component: complexo
        for complexo, components in complexo_components.items()
        if complexo in row_of
        for component in components
    }
    stage = effective.n_stages - 1

    couplings: list[HydroCoupling] = []
    borrowed: list[BorrowedOwner] = []
    unowned: list[int] = []
    for hydro_id, code in enumerate(id_map.hydro_codes):
        if code in row_of:
            owner, via_complex = code, False
        elif code in complex_of:
            owner, via_complex = complex_of[code], True
        else:
            found = _first_downstream_owner(code, effective, row_of, complex_of, stage)
            if found is None:
                unowned.append(code)
                continue
            owner, via_complex = found, False
            borrowed.append(BorrowedOwner(code, owner, via_complex=False))
        if via_complex:
            borrowed.append(BorrowedOwner(code, owner, via_complex=True))
        takes_storage = (owner == code or via_complex) and is_reservoir(effective, code)
        couplings.append(
            HydroCoupling(
                hydro_id=hydro_id,
                plant_index=row_of[owner],
                storage_floor_hm3=(
                    effective.value(code, "volume_minimo", stage)
                    if takes_storage
                    else None
                ),
            )
        )
    return CouplingResolution(tuple(couplings), tuple(borrowed), tuple(unowned))


def _first_downstream_owner(
    code: int,
    effective: EffectiveCadastro,
    row_of: Mapping[int, int],
    complex_of: Mapping[int, int],
    stage: int,
) -> int | None:
    """The first plant below *code* with a cut row, through plants with none.

    The walk crosses plants absent from the converted case (they stay in the
    ``hidr.dat`` cascade); ``0`` is the sink and a cycle ends it.
    """
    visited = {code}
    current = effective.downstream_plant(code, stage)
    while current != 0 and current not in visited and current in effective.base.index:
        if current in row_of:
            return current
        if current in complex_of:
            return complex_of[current]
        visited.add(current)
        current = effective.downstream_plant(current, stage)
    return None


def _posto_value(row: pd.Series, posto: int) -> float:
    column = str(posto)
    if column not in row.index:
        raise ValueError(f"the inflow file has no posto {posto} column")
    return float(row[column])


def _month_share(start: date, end: date, year: int, month: int) -> float:
    """The fraction of ``year``-``month``'s days that ``[start, end)`` covers."""
    month_start, month_end = month_window(year, month)
    overlap = (min(end, month_end) - max(start, month_start)).days
    return max(overlap, 0) / (month_end - month_start).days


def _monthly_means(
    windows: Iterable[tuple[date, date, float]], months: Sequence[tuple[int, int]]
) -> tuple[float, ...]:
    """Each month's day-weighted mean over *windows* (uncovered days count 0)."""
    materialized = tuple(windows)
    return tuple(
        sum(
            _month_share(start, end, year, month) * value
            for start, end, value in materialized
        )
        for year, month in months
    )


def _trunk_rows(
    table: pd.DataFrame, calendar: Sequence[OperativeStage]
) -> list[tuple[date, date, pd.Series]]:
    """The deterministic stages' rows (every stage before the terminal one)."""
    terminal = len(calendar) - 1
    rows: list[tuple[date, date, pd.Series]] = []
    for _, row in table.iterrows():
        stage_index = int(row["estagio"]) - 1
        if 0 <= stage_index < terminal:
            stage = calendar[stage_index]
            rows.append((stage.start_date, stage.end_date, row))
    return rows


def coupling_lag_offsets(
    vazoes: Vazoes,
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
    months: Sequence[tuple[int, int]],
    couplings: Iterable[HydroCoupling],
) -> dict[int, tuple[float, ...]]:
    """Per-depth m³/s the cut's lag value exceeds novomodelo's, for each coupled hydro.

    The recent-observation seed is already read on the coupling posto, so only
    the deterministic study stages differ: the coupling inflow
    (``previsoes_com_postos_artificiais`` on the coupling posto) minus the
    operation inflow novomodelo carries (``previsoes`` on the operation posto). A
    hydro whose offsets all vanish is omitted.
    """
    coupling_rows = _trunk_rows(vazoes.previsoes_com_postos_artificiais, calendar)
    operation_rows = _trunk_rows(vazoes.previsoes, calendar)
    offsets: dict[int, tuple[float, ...]] = {}
    for coupling in couplings:
        code = id_map.hydro_codes[coupling.hydro_id]
        coupling_posto = effective.coupling_gauge(code)
        operation_posto = effective.inflow_gauge(code, 0)
        gaps = tuple(
            (
                start,
                end,
                _posto_value(c_row, coupling_posto)
                - _posto_value(o_row, operation_posto),
            )
            for (start, end, c_row), (_, _, o_row) in zip(
                coupling_rows, operation_rows, strict=True
            )
        )
        offset = _monthly_means(gaps, months)
        if any(abs(value) > _OFFSET_TOLERANCE for value in offset):
            offsets[coupling.hydro_id] = offset
    return offsets


def source_only_lag_inflows(
    vazoes: Vazoes,
    effective: EffectiveCadastro,
    plant_codes: Sequence[int],
    plant_indices: Iterable[int],
    calendar: Sequence[OperativeStage],
    months: Sequence[tuple[int, int]],
) -> dict[int, tuple[float, ...]]:
    """Per-depth m³/s lag values of cut plants no converted hydro takes.

    Read on each plant's coupling posto over the pre-study observations and
    the deterministic study stages; a plant absent from ``hidr.dat`` has no
    posto and is skipped.
    """
    study_start = calendar[0].start_date
    observation_rows = recent_observation_rows(vazoes, study_start)
    trunk = _trunk_rows(vazoes.previsoes_com_postos_artificiais, calendar)
    inflows: dict[int, tuple[float, ...]] = {}
    for index in plant_indices:
        code = plant_codes[index]
        if code not in effective.base.index:
            continue
        posto = effective.coupling_gauge(code)
        windows = (
            (start, end, _posto_value(row, posto))
            for start, end, row in (*observation_rows, *trunk)
        )
        inflows[index] = _monthly_means(windows, months)
    return inflows


def terminal_shares(
    vazoes: Vazoes,
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
    couplings: Iterable[HydroCoupling],
    source_only_codes: Iterable[int],
) -> tuple[TerminalShare, ...]:
    """The plants whose terminal-stage coupling inflow novomodelo's state misses.

    A coupled hydro is listed when its coupling inflow
    (``cenarios_calculados_com_postos_artificiais`` on the coupling posto)
    differs from its operation inflow (``cenarios_gerados`` on the operation
    posto) in some scenario; a source-only plant always is, with its whole
    coupling inflow as the gap.
    """
    order = ["estagio", "cenario"]
    coupling_fan = [
        row
        for _, row in vazoes.cenarios_calculados_com_postos_artificiais.sort_values(
            order
        ).iterrows()
    ]
    operation_fan = [
        row for _, row in vazoes.cenarios_gerados.sort_values(order).iterrows()
    ]
    shares: list[TerminalShare] = []
    for coupling in couplings:
        code = id_map.hydro_codes[coupling.hydro_id]
        coupling_posto = effective.coupling_gauge(code)
        operation_posto = effective.inflow_gauge(code, 0)
        gap = max(
            (
                abs(
                    _posto_value(c_row, coupling_posto)
                    - _posto_value(o_row, operation_posto)
                )
                for c_row, o_row in zip(coupling_fan, operation_fan, strict=True)
            ),
            default=0.0,
        )
        if gap > _OFFSET_TOLERANCE:
            shares.append(TerminalShare(code, coupling_posto, operation_posto, gap))
    for code in source_only_codes:
        if code not in effective.base.index:
            continue
        posto = effective.coupling_gauge(code)
        gap = max((abs(_posto_value(row, posto)) for row in coupling_fan), default=0.0)
        shares.append(TerminalShare(code, posto, None, gap))
    return tuple(shares)


def carried_lag_share(
    calendar: Sequence[OperativeStage], cut_year: int, cut_month: int
) -> float:
    """The fraction of the cut month the stages before the terminal one cover."""
    return sum(
        _month_share(stage.start_date, stage.end_date, cut_year, cut_month)
        for stage in calendar[:-1]
    )
