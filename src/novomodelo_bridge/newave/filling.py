"""Stage-math, ζ (zeta), and machine-count pure helpers for the ``exph`` mapping.

This foundation module holds the calendar/horizon arithmetic shared by the
filling converters and the fill-rate helpers: mapping a
NEWAVE filling date to a Novomodelo 0-based stage index (:func:`stage_id`) and the
per-month hm³-per-m³/s weight (:func:`zeta`). It is also the one home for "how
many machines are in service at stage *t*", for both kinds of ``exph`` plant:
a not-yet-built one filling its dead volume (:func:`online_machines`) and one
already in operation gaining machines (:class:`ExpansionConfig`), both reading
the machine entries through :func:`exph_unit_rows`.

All functions are pure (no logging, no I/O) so they can be reused and tested in
isolation. The module deliberately depends on nothing under
``novomodelo_bridge.newave.converters`` — the calendar-hours basis is re-implemented locally
in :func:`month_hours` rather than imported, so a foundation module never pulls a
converter into its import graph.
"""

from __future__ import annotations

import calendar
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd


def month_hours(year: int, month: int) -> float:
    """Total number of hours in the given calendar month.

    Uses the same calendar-hours basis as
    ``novomodelo_bridge.newave.converters.temporal._month_hours`` (the basis of record):
    ``calendar.monthrange(year, month)[1] * 24``. Re-implemented here to keep
    this foundation module free of any converter dependency.

    A *month* outside 1-12 propagates ``calendar.monthrange``'s ``ValueError``.
    """
    days_in_month = calendar.monthrange(year, month)[1]
    return float(days_in_month * 24)


def zeta(year: int, month: int) -> float:
    """ζ for the given calendar month: hm³ stored per unit of m³/s inflow.

    Converts the month's calendar hours (:func:`month_hours`) into a volume
    weight via ``× 3600 / 1e6`` (seconds per hour over m³→hm³). For October with
    31 days this is ``744 * 3600 / 1e6 == 2.6784``.
    """
    return month_hours(year, month) * 3600.0 / 1e6


def stage_id(year: int, month: int, start_year: int, start_month: int) -> int:
    """Novomodelo 0-based stage index for a calendar month within the study horizon.

    Computes ``(year - start_year) * 12 + (month - start_month)`` so the study
    start (``start_year``/``start_month``) maps to stage ``0``. The result may be
    negative for dates before the study start; clamping is a caller concern
    handled by the filling converters, not here.
    """
    return (year - start_year) * 12 + (month - start_month)


def filling_schedule(
    start_year: int,
    start_month: int,
    duracao_months: int,
    horizon_start_year: int,
    horizon_start_month: int,
) -> tuple[int, int]:
    """Return ``(start_stage_id, entry_stage_id)`` for one ``exph`` filling row.

    Maps a plant's declared filling schedule — its start month
    (``start_year``/``start_month``, from ``data_inicio_enchimento``) and its
    duration in months (``duracao_months``, from ``duracao_enchimento``) — onto
    the half-open Novomodelo stage window ``[start_stage_id, entry_stage_id)``, with
    the edge clamps below applied.

    The raw start is :func:`stage_id` (unclamped, may be negative for a pre-study
    start). Two clamps then run:

    * **Pre-start clamp**: ``start_stage_id = max(0, raw_start)`` — a filling
      start at or before the study start clamps to stage 0 (no PreFilling phase).
    * **Entry formula**: ``entry = raw_start + duracao_months``, anchored on
      the *raw* (unclamped) start so the window always spans ``duracao_months``
      calendar months from the declared start even when ``raw_start < 0``. It is
      then floored at ``start_stage_id`` so ``entry_stage_id >= start_stage_id``
      always holds:
      ``entry_stage_id = max(start_stage_id, raw_start + duracao_months)``.

    When ``duracao_months == 0`` the result is the equal pair
    ``entry_stage_id == start_stage_id`` — the empty half-open window, i.e. the
    "no Filling phase" signal. This function does not branch on it; the caller
    decides to omit ``filling``.

    For JURUENA (Oct 2024 start, ``duracao_months == 1``, study start Sep 2024):
    ``raw_start == 1``, so the result is ``(1, 2)``.

    This helper trusts its inputs: the ``Timestamp`` guarantees a valid month, so
    no month-range validation is performed here.
    """
    raw_start = stage_id(
        start_year, start_month, horizon_start_year, horizon_start_month
    )
    start = max(0, raw_start)
    entry = max(start, raw_start + duracao_months)
    return start, entry


def filling_completion_date(
    start_year: int, start_month: int, duracao_months: int
) -> date:
    """First-of-month calendar date when a dead-volume filling plant enters service.

    A filling plant's operational start is the month it finishes filling and
    begins operating: ``duracao_months`` calendar months after its filling start
    (``data_inicio_enchimento`` + ``duracao_enchimento``). Computed straight from
    the raw ``exph`` schedule, so the date stays truthful even when filling
    completes before or after the study horizon — unlike the horizon-clamped stage
    index from :func:`filling_schedule`.

    For JURUENA (Oct 2024 start, ``duracao_months == 1``): November 2024 →
    ``date(2024, 11, 1)``.
    """
    total = start_year * 12 + (start_month - 1) + duracao_months
    return date(total // 12, total % 12 + 1, 1)


def filling_min_rate_m3s(
    min_storage_hm3: float,
    volume_morto_pct: float,
    start_stage_id: int,
    entry_stage_id: int,
    stage_year_month: Callable[[int], tuple[int, int]],
) -> float:
    """Constant per-stage minimum accumulation floor (m³/s) for the filling window.

    Fills the reservoir from its seeded ``filling_storage`` volume up to
    ``min_storage_hm3`` linearly across the half-open filling window
    ``[start_stage_id, entry_stage_id)``::

        rate = remaining / Σ_{t ∈ [start, entry)} ζ_t

    where ``remaining = min_storage_hm3 × (1 − volume_morto_pct / 100)`` and ζ_t
    comes from :func:`zeta`. ``stage_year_month`` maps a stage id to
    its ``(year, month)``; the converter supplies a closure over the horizon so
    this module stays horizon-agnostic.

    The window is **half-open**: the target reaches ``min_storage`` at
    ``entry_stage_id − 1``, so the last filling stage is
    ``entry_stage_id − 1``. For JURUENA (single-stage window, Oct 2024):
    ``2.93 / 2.6784 ≈ 1.09``.

    Returns ``0.0`` for an empty window (``start_stage_id >= entry_stage_id``)
    rather than raising ``ZeroDivisionError`` — this is the "no Filling phase"
    signal, and the caller omits ``filling`` in that case. This helper
    trusts its inputs: out-of-range ``volume_morto_pct`` is clamped upstream by
    the initial-conditions converter, not here.
    """
    remaining = min_storage_hm3 * (1.0 - volume_morto_pct / 100.0)
    denom = sum(
        zeta(*stage_year_month(t)) for t in range(start_stage_id, entry_stage_id)
    )
    if denom > 0.0:
        return remaining / denom
    return 0.0


def expansion_machine_counts(
    registry_counts: Mapping[int, int],
    start_counts: Mapping[int, int],
    unit_rows: Sequence[tuple[int, int]],
    query_stage_id: int,
) -> dict[int, int]:
    """Machines in service per ``conjunto`` at *query_stage_id* for an expanding plant.

    *registry_counts* is the final configuration (``hidr.dat``, entering machines
    included), *start_counts* the study-start one (``modif.dat`` ``NUMMAQ``
    applied), and *unit_rows* the ``(conjunto, entry_stage_id)`` of each entering
    machine. The count is ``min(registry, base + entered)`` with
    ``base = min(start, registry − all entering)``: the floor keeps a deck that
    declares no ``NUMMAQ`` (``start`` equal to the registry) from crediting the
    entering machines twice. Every conjunto of *registry_counts* is covered.
    """
    entering_total: Counter[int] = Counter(conjunto for conjunto, _ in unit_rows)
    entered = Counter(
        conjunto
        for conjunto, entry_stage_id in unit_rows
        if entry_stage_id <= query_stage_id
    )
    counts: dict[int, int] = {}
    for conjunto, registry in registry_counts.items():
        base = min(
            start_counts.get(conjunto, registry),
            registry - entering_total.get(conjunto, 0),
        )
        counts[conjunto] = min(registry, base + entered.get(conjunto, 0))
    return counts


def exph_unit_rows(
    exph_df: pd.DataFrame, code: int, start_year: int, start_month: int
) -> tuple[tuple[int, int], ...]:
    """``(conjunto, entry_stage_id)`` of every machine plant *code* brings online.

    inewave parses ``data_entrada_operacao`` and ``conjunto_maquina_entrada``
    independently, so a row missing either cannot place a machine and is skipped
    rather than cast (a ``NaT`` date would crash the stage arithmetic).
    """
    rows = exph_df.loc[
        (exph_df["codigo_usina"] == code) & exph_df["data_entrada_operacao"].notna()
    ]
    return tuple(
        (int(conjunto), stage_id(entry.year, entry.month, start_year, start_month))
        for entry, conjunto in zip(
            rows["data_entrada_operacao"], rows["conjunto_maquina_entrada"]
        )
        if not pd.isna(conjunto)
    )


def hreg_with_machines(hreg: pd.Series, counts: Mapping[int, int]) -> pd.Series:
    """A copy of cadastro row *hreg* whose machine counts are *counts*.

    ``numero_conjuntos_maquinas`` is widened to cover every conjunto in
    *counts*, since the capacity helpers loop ``1..numero_conjuntos_maquinas``.
    """
    hreg_copy = hreg.copy()
    for conjunto, count in counts.items():
        hreg_copy[f"maquinas_conjunto_{conjunto}"] = int(count)
    if counts:
        hreg_copy["numero_conjuntos_maquinas"] = max(counts)
    return hreg_copy


@dataclass(frozen=True)
class ExpansionConfig:
    """An ``EE`` plant's machine configuration across the study horizon.

    The three fields are the inputs of :func:`expansion_machine_counts`; a
    conjunto ``NUMCNJ`` drops at the study start has a start count of ``0``.
    """

    registry_counts: dict[int, int]
    start_counts: dict[int, int]
    unit_rows: tuple[tuple[int, int], ...]

    @property
    def full_online_stage(self) -> int:
        """First stage at which every entering machine is in service."""
        return max(stage for _conjunto, stage in self.unit_rows)

    def hreg_at(self, hreg: pd.Series, stage: int) -> pd.Series:
        """Cadastro row *hreg* with the machines in service at *stage*."""
        return hreg_with_machines(
            hreg,
            expansion_machine_counts(
                self.registry_counts, self.start_counts, self.unit_rows, stage
            ),
        )

    def declared_hreg(self, hreg: pd.Series) -> pd.Series:
        """Cadastro row *hreg* at the configuration ``hydros.json`` declares.

        The final one, never the study-start one: novomodelo forbids a per-stage
        bound above the declaration and the stages past the ramp carry no
        bound, so a start-configuration declaration would cap the plant there
        for the rest of the horizon.
        """
        return self.hreg_at(hreg, self.full_online_stage)


def online_machines(
    unit_rows: Sequence[tuple[int, int]],
    entry_stage_id: int,
    query_stage_id: int,
) -> dict[int, int]:
    """Count online machines per ``conjunto`` at the queried operating stage.

    ``unit_rows`` is a sequence of ``(conjunto, unit_entry_stage_id)`` pairs, one
    per entering machine (the converter pre-maps each ``data_entrada_operacao`` to
    a stage id via :func:`stage_id`). Each unit's effective online stage is
    ``max(unit_entry_stage_id, entry_stage_id)`` — the clamp ensures no unit is
    online before the reservoir is filled, since water cannot be turbined before
    the fill completes.

    The clamp is applied **before** the ``<= query_stage_id`` comparison. Returns
    ``{conjunto: count}`` counting units whose effective online stage is
    ``<= query_stage_id``; groups with zero online machines are omitted.
    """
    counts: Counter[int] = Counter(
        conjunto
        for conjunto, unit_entry_stage_id in unit_rows
        if max(unit_entry_stage_id, entry_stage_id) <= query_stage_id
    )
    return dict(counts)
