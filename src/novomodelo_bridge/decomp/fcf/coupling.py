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
follows the inflow the case carries: the plant's operation inflow, except for
the plants of an NW group that takes NEWAVE's split
(:class:`~novomodelo_bridge.decomp.scenarios.NwInflowSplit`), which carry that
split. The deterministic part of the difference is returned as a per-depth
offset for the mapper to fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from novomodelo_bridge.core.coupling_auxiliary import (
    coupling_auxiliary_id,
    coupling_auxiliary_name,
)
from novomodelo_bridge.core.inflow_windows import month_window
from novomodelo_bridge.core.pandas_utils import as_int
from novomodelo_bridge.decomp.converters.cadastro import is_reservoir
from novomodelo_bridge.decomp.fcf.mapper import HydroCoupling
from novomodelo_bridge.decomp.scenarios import (
    ObservationTables,
    TreeTables,
    posto_value,
    recent_observation_rows,
    split_nw_inflows,
    terminal_case_inflows,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from datetime import date

    import pandas as pd
    from idecomp.decomp import Dadger

    from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro
    from novomodelo_bridge.decomp.id_map import DecompIdMap
    from novomodelo_bridge.decomp.temporal import OperativeStage

#: Calendar-month depths of the cut's inflow-lag axis.
LAG_DEPTHS = 12

_OFFSET_TOLERANCE = 1e-9


class _CouplingTrunkTables(Protocol):
    @property
    def previsoes_com_postos_artificiais(self) -> pd.DataFrame | None: ...

    @property
    def cenarios_calculados_com_postos_artificiais(self) -> pd.DataFrame | None: ...


class _OperationTrunkTables(Protocol):
    @property
    def previsoes(self) -> pd.DataFrame | None: ...

    @property
    def cenarios_gerados(self) -> pd.DataFrame | None: ...


class _SourceOnlyTables(_CouplingTrunkTables, ObservationTables, Protocol):
    pass


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
class CouplingAuxiliary:
    """An auxiliary hydro carrying the last-stage coupling inflow one plant's
    cut row reads beyond the inflow the case carries for it.

    ``terminal_inflows_m3s`` holds, per last-stage scenario, ``y - x + shift_m3s``:
    ``y`` the coupling inflow and ``x`` the inflow the case carries (0 for a
    cut plant no converted hydro takes). ``shift_m3s`` is the least constant
    keeping every value non-negative, and the cut's constant gives it back
    (:func:`coupling_auxiliary_lag_offsets`). ``operation_posto`` is ``None``
    for a cut plant not in the case.
    """

    plant_code: int
    hydro_id: int
    name: str
    plant_index: int
    coupling_posto: int
    operation_posto: int | None
    shift_m3s: float
    terminal_inflows_m3s: tuple[float, ...]
    max_gap_m3s: float

    @property
    def coupling(self) -> HydroCoupling:
        """The auxiliary taking its plant's cut row, without a storage term."""
        return HydroCoupling(hydro_id=self.hydro_id, plant_index=self.plant_index)


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


def _covered_means(
    windows: Iterable[tuple[date, date, float]], months: Sequence[tuple[int, int]]
) -> tuple[float, ...]:
    """Each month's mean over the days *windows* cover (0 when none is covered),
    as novomodelo casts the ``recent_observations`` seed onto a lag period."""
    materialized = tuple(windows)
    means: list[float] = []
    for year, month in months:
        shares = [
            (_month_share(start, end, year, month), value)
            for start, end, value in materialized
        ]
        covered = sum(share for share, _ in shares)
        means.append(
            sum(share * value for share, value in shares) / covered if covered else 0.0
        )
    return tuple(means)


def _trunk_rows(
    weekly: pd.DataFrame | None,
    monthly: pd.DataFrame | None,
    calendar: Sequence[OperativeStage],
) -> list[tuple[date, date, pd.Series]]:
    """The deterministic stages' rows (every stage before the terminal one), in
    stage order: the *weekly* forecasts, then the monthly trunk stages, whose
    single scenario sits in the *monthly* scenario table.

    A first week that starts before the operative month is clipped to it: the
    source model excludes those days from every lag month, and novomodelo's lag
    periods start at the first stage's season."""
    terminal = len(calendar) - 1
    operative_start = calendar[0].operative_month_start
    rows: list[tuple[date, date, pd.Series]] = []
    for table in (weekly, monthly):
        if table is None:
            continue
        for _, row in table.iterrows():
            stage_index = int(row["estagio"]) - 1
            if 0 <= stage_index < terminal:
                stage = calendar[stage_index]
                rows.append(
                    (max(stage.start_date, operative_start), stage.end_date, row)
                )
    rows.sort(key=lambda entry: entry[0])
    return rows


def _coupling_trunk(
    vazoes: _CouplingTrunkTables, calendar: Sequence[OperativeStage]
) -> list[tuple[date, date, pd.Series]]:
    return _trunk_rows(
        vazoes.previsoes_com_postos_artificiais,
        vazoes.cenarios_calculados_com_postos_artificiais,
        calendar,
    )


def _operation_trunk(
    vazoes: _OperationTrunkTables, calendar: Sequence[OperativeStage]
) -> list[tuple[date, date, pd.Series]]:
    return _trunk_rows(vazoes.previsoes, vazoes.cenarios_gerados, calendar)


def _terminal_fan(table: pd.DataFrame, terminal_estagio: int) -> list[pd.Series]:
    """The terminal stage's rows, in scenario order."""
    fan = table.loc[table["estagio"] == terminal_estagio].sort_values("cenario")
    return [row for _, row in fan.iterrows()]


def coupling_lag_offsets(
    vazoes: TreeTables,
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
    months: Sequence[tuple[int, int]],
    couplings: Iterable[HydroCoupling],
) -> dict[int, tuple[float, ...]]:
    """Per-depth m³/s the cut's lag value exceeds novomodelo's, for each coupled hydro.

    The recent-observation seed is already read on the coupling posto, so only
    the deterministic study stages differ: the coupling inflow (the tables with
    the artificial postos computed, on the coupling posto) minus the inflow
    novomodelo carries (``previsoes`` and ``cenarios_gerados``, on the operation
    posto, or the NW split of a group that passes the guard,
    :meth:`~novomodelo_bridge.decomp.scenarios.NwInflowSplit.carried`). A
    hydro whose offsets all vanish is omitted.
    """
    split = split_nw_inflows(vazoes, effective, id_map)
    coupling_rows = _coupling_trunk(vazoes, calendar)
    operation_rows = _operation_trunk(vazoes, calendar)
    offsets: dict[int, tuple[float, ...]] = {}
    for coupling in couplings:
        code = id_map.hydro_codes[coupling.hydro_id]
        coupling_posto = effective.coupling_gauge(code)
        gaps = tuple(
            (
                start,
                end,
                posto_value(c_row, coupling_posto)
                - split.carried(code, o_row, c_row, effective),
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
    vazoes: _SourceOnlyTables,
    effective: EffectiveCadastro,
    plant_codes: Sequence[int],
    plant_indices: Iterable[int],
    calendar: Sequence[OperativeStage],
    months: Sequence[tuple[int, int]],
) -> dict[int, tuple[float, ...]]:
    """Per-depth m³/s lag values of cut plants no converted hydro takes.

    Read on each plant's coupling posto, the way novomodelo builds a coupled
    hydro's lag: a month before the first operative month is the mean of the
    pre-study observations over the days they cover, and a later month the
    day-weighted mean over those observations and the deterministic study
    stages. A plant absent from ``hidr.dat`` has no posto and is skipped.
    """
    operative_start = calendar[0].operative_month_start
    observation_rows = recent_observation_rows(vazoes, calendar[0])
    trunk = _coupling_trunk(vazoes, calendar)
    seeded = [month_window(*month)[0] < operative_start for month in months]
    inflows: dict[int, tuple[float, ...]] = {}
    for index in plant_indices:
        code = plant_codes[index]
        if code not in effective.base.index:
            continue
        posto = effective.coupling_gauge(code)
        observed = tuple(
            (start, end, posto_value(row, posto))
            for start, end, row in observation_rows
        )
        study = tuple(
            (start, end, posto_value(row, posto)) for start, end, row in trunk
        )
        inflows[index] = tuple(
            seed if before else mean
            for before, seed, mean in zip(
                seeded,
                _covered_means(observed, months),
                _monthly_means((*observed, *study), months),
                strict=True,
            )
        )
    return inflows


def plan_coupling_auxiliaries(
    vazoes: TreeTables,
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
    plant_codes: Sequence[int],
    complexo_components: Mapping[int, Sequence[int]],
) -> tuple[CouplingAuxiliary, ...]:
    """The auxiliary coupling plants the boundary cuts need, in hydro-id order.

    A coupled hydro needs one when its last-stage coupling inflow
    (``cenarios_calculados_com_postos_artificiais`` on the coupling posto)
    differs in some scenario from the inflow the case carries
    (:func:`~novomodelo_bridge.decomp.scenarios.terminal_case_inflows`). A
    cut plant no converted hydro takes needs one when its coupling inflow is
    nonzero; one absent from ``hidr.dat`` has no posto and is skipped, as in
    :func:`source_only_lag_inflows`.

    Raises
    ------
    UnsupportedSourceError
        A plant that needs one has a code of 1000 or more
        (:func:`~novomodelo_bridge.core.coupling_auxiliary.coupling_auxiliary_id`).
    """
    coupling_table = vazoes.cenarios_calculados_com_postos_artificiais
    if coupling_table is None:
        raise ValueError(
            "the inflow file has no terminal coupling fan "
            "(cenarios_calculados_com_postos_artificiais)"
        )
    fan = _terminal_fan(coupling_table, len(calendar))
    carried = terminal_case_inflows(vazoes, effective, id_map, calendar)
    if len(fan) != len(carried):
        raise ValueError(
            f"{len(fan)} terminal coupling scenarios for {len(carried)} terminal "
            "inflow scenarios"
        )
    resolution = resolve_couplings(plant_codes, id_map, effective, complexo_components)

    auxiliaries: list[CouplingAuxiliary] = []

    def add(
        code: int,
        plant_index: int,
        operation_posto: int | None,
        carried_m3s: Sequence[float],
    ) -> None:
        posto = effective.coupling_gauge(code)
        gaps = tuple(
            posto_value(row, posto) - x for row, x in zip(fan, carried_m3s, strict=True)
        )
        max_gap = max((abs(gap) for gap in gaps), default=0.0)
        if max_gap <= _OFFSET_TOLERANCE:
            return
        shift = max(0.0, -min(gaps))
        auxiliaries.append(
            CouplingAuxiliary(
                plant_code=code,
                hydro_id=coupling_auxiliary_id(code),
                name=coupling_auxiliary_name(
                    code, str(effective.base.at[code, "nome_usina"])
                ),
                plant_index=plant_index,
                coupling_posto=posto,
                operation_posto=operation_posto,
                shift_m3s=shift,
                terminal_inflows_m3s=tuple(gap + shift for gap in gaps),
                max_gap_m3s=max_gap,
            )
        )

    for coupling in resolution.couplings:
        code = id_map.hydro_codes[coupling.hydro_id]
        add(
            code,
            coupling.plant_index,
            effective.inflow_gauge(code, 0),
            [inflows[coupling.hydro_id] for inflows in carried],
        )
    coupled = {coupling.plant_index for coupling in resolution.couplings}
    for index, code in enumerate(plant_codes):
        if index not in coupled and code in effective.base.index:
            add(code, index, None, [0.0] * len(fan))
    return tuple(sorted(auxiliaries, key=lambda auxiliary: auxiliary.hydro_id))


def coupling_auxiliary_lag_offsets(
    auxiliaries: Iterable[CouplingAuxiliary],
    calendar: Sequence[OperativeStage],
    months: Sequence[tuple[int, int]],
) -> dict[int, tuple[float, ...]]:
    """Per-depth m³/s the cut's lag value falls short of each auxiliary's state.

    The auxiliary's inflow is 0 before the last stage, so its lag state
    exceeds the coupling difference only by the shift the last stage carries,
    over the share of each lag month that stage covers. An auxiliary without a
    shift is omitted.
    """
    terminal = calendar[-1]
    shares = tuple(
        _month_share(terminal.start_date, terminal.end_date, year, month)
        for year, month in months
    )
    return {
        auxiliary.hydro_id: tuple(-auxiliary.shift_m3s * share for share in shares)
        for auxiliary in auxiliaries
        if auxiliary.shift_m3s > 0.0
    }


def complexo_components(dadger: Dadger) -> dict[int, list[int]]:
    """The deck's ``CX`` register as ``{complexo_code: [component_codes]}``.

    ``CX`` couples a complex of the boundary cut (absent from the DECOMP
    model) to the individual DECOMP plants inside it. Empty when the deck has
    no ``CX`` line.
    """
    components: dict[int, list[int]] = {}
    cx = dadger.cx(df=True)
    if cx.empty:
        return components
    for complexo, component in zip(
        cx["codigo_newave"], cx["codigo_decomp"], strict=True
    ):
        components.setdefault(as_int(complexo), []).append(as_int(component))
    return components
