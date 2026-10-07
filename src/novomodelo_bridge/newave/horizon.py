"""Canonical the source model study-horizon arithmetic — the single source of truth.

Every per-stage table the pipeline emits (inflows, loads, bounds, penalties,
constraints) is sized against the study horizon, and the study/post-study
boundary drives post-study extrapolation. That arithmetic used to be hand-copied
at ~25 sites across the converters and comparators; this module owns it so all of
them agree by construction.

The source model conventions encoded here:

- The study runs from ``mes_inicio_estudo`` of ``ano_inicio_estudo`` through
  December of ``ano_inicio_estudo + num_anos_estudo - 1`` — i.e.
  ``study_months = (13 - start_month) + (num_anos - 1) * 12`` stages.
- ``num_anos_pos_estudo`` full calendar years follow, adding ``num_anos_pos * 12``
  post-study stages, for ``total_stages`` in all.
- Seasonal records for the post-study (static final) period are tagged with the
  sentinel year ``9999`` (:data:`POST_STUDY_YEAR`).
- ``99990`` and above is the source model's "big-M" sentinel meaning "no limit / restore
  default" in bound records (:data:`BIG_M`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date

from inewave.newave import Dger

from cobre_bridge.core.tolerances import BIG_M

# The source model tags post-study (static final period) seasonal data with year 9999.
POST_STUDY_YEAR = 9999


@dataclass(frozen=True)
class StudyHorizon:
    """Resolved study-horizon dimensions for a source-model case.

    Build with :func:`study_horizon`; every field is a plain count/index so the
    object is cheap to pass around and compare.
    """

    start_year: int
    start_month: int
    num_anos: int
    """Number of study years (``num_anos_estudo``)."""
    num_anos_pos: int
    """Number of post-study years (``num_anos_pos_estudo``)."""
    study_months: int
    """Number of study stages."""
    total_stages: int
    """Number of stages overall (study + post-study)."""

    @property
    def last_study_stage(self) -> int:
        """0-based index of the final study stage (the freeze point)."""
        return self.study_months - 1

    @property
    def first_year_stages(self) -> int:
        """Number of stages in the (partial) first calendar year."""
        return 13 - self.start_month

    @property
    def pos_months(self) -> int:
        """Number of post-study stages."""
        return self.num_anos_pos * 12

    def is_post_study(self, stage_id: int) -> bool:
        """True if ``stage_id`` (0-based) falls in the post-study tail."""
        return stage_id >= self.study_months


def study_horizon(dger: Dger) -> StudyHorizon:
    """Resolve the :class:`StudyHorizon` from a parsed ``dger.dat``.

    ``num_anos_estudo`` falls back to 1 and ``num_anos_pos_estudo`` to 0 when
    absent/zero (the pipeline rejects a genuinely empty study earlier).
    """
    start_year = int(dger.ano_inicio_estudo)
    start_month = int(dger.mes_inicio_estudo)
    num_anos = int(dger.num_anos_estudo or 1)
    num_anos_pos = int(dger.num_anos_pos_estudo or 0)
    study_months = (13 - start_month) + (num_anos - 1) * 12
    total_stages = study_months + num_anos_pos * 12
    return StudyHorizon(
        start_year=start_year,
        start_month=start_month,
        num_anos=num_anos,
        num_anos_pos=num_anos_pos,
        study_months=study_months,
        total_stages=total_stages,
    )


def historical_start_date(dger: Dger) -> str:
    """ISO ``YYYY-MM-DD`` operational-start date for always-in-service entities.

    Buses, transmission lines, thermals, non-controllable sources, and existing
    hydros carry no per-entity commissioning date in the source model — they are
    treated as in service since the start of the historical inflow record
    (``dger.ano_inicial_historico``), taken as January 1st of that year. Falls
    back to 1931 (the usual record start) when the field is absent, matching
    ``newave.converters.temporal``. The date is a canonical-ordering key in Cobre, not a
    commissioning gate (that is ``entry_stage_id``/``exit_stage_id``), so a shared
    value orders these entities by id.
    """
    year = int(dger.ano_inicial_historico or 1931)
    return f"{year:04d}-01-01"


def build_stage_dates(
    start_year: int, start_month: int, total_stages: int
) -> list[date]:
    """Return the first-of-month date of each stage, in order."""
    stages: list[date] = []
    year, month = start_year, start_month
    for _ in range(total_stages):
        stages.append(date(year, month, 1))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return stages


def stage_dates_for(horizon: StudyHorizon) -> list[date]:
    """Convenience: :func:`build_stage_dates` for a resolved horizon."""
    return build_stage_dates(
        horizon.start_year, horizon.start_month, horizon.total_stages
    )


def _repeating_year_steps(
    recs: Iterable[tuple[int, float]], transform: Callable[[float], float]
) -> dict[int, float | None]:
    """Value in force in each calendar month of a year that repeats, given its
    ``(month, raw_value)`` steps; months before the first step wrap around to
    the last. ``None`` marks a cleared month (raw value ``>= BIG_M``). Empty
    when *recs* is empty.
    """
    steps: dict[int, float | None] = {}
    for month, value in recs:
        steps[month] = None if value >= BIG_M else transform(value)
    if not steps:
        return {}
    in_force: dict[int, float | None] = {}
    current = steps[max(steps)]
    for month in range(1, 13):
        current = steps.get(month, current)
        in_force[month] = current
    return in_force


def seasonal_step_function(
    recs: Iterable[tuple[int, int, float]],
    transform: Callable[[float], float],
    *,
    seasonalize: bool,
    horizon: StudyHorizon,
    pre_recs: Iterable[tuple[int, float]] = (),
    pos_recs: Iterable[tuple[int, float]] = (),
) -> dict[int, float]:
    """Forward-fill dated override records into per-stage values.

    ``recs`` are ``(year, month, raw_value)`` change-points: each sets the value
    from its stage onward until the next overrides it. A raw value ``>= BIG_M``
    clears the fill ("restore default"). Returns ``{stage_id: transform(value)}``
    for every stage that has an active value.

    ``pre_recs`` / ``pos_recs`` are ``(month, raw_value)`` steps of the static
    pre-study and post-study periods, each read as one repeating year
    (:func:`_repeating_year_steps`). The ``PRE`` value in force in the month
    before the study starts holds from stage 0 until the first change-point.
    ``POS`` steps, when present, set every post-study stage by its calendar
    month and replace the extrapolation below.

    Otherwise the post-study tail follows the source model's rule, selected by
    *seasonalize*:

    - ``True`` (e.g. VMINT/VMAXT with their ``sazonaliza_*`` flag set): repeat the
      last study year's monthly pattern, so a genuinely seasonal constraint keeps
      cycling through the static final period.
    - ``False`` (e.g. VAZMINT / TURBMINT / TURBMAXT, which have no seasonalize
      flag): freeze the last study stage's value across the post-study tail.

    This is the single source of truth for the storage-bounds converter
    (``newave.converters.hydro.convert_storage_bounds``).
    """
    sm = horizon.start_month
    study_months = horizon.study_months
    total_stages = horizon.total_stages

    changepoints: list[tuple[int, float]] = []
    for year, month, value in recs:
        sid = (year - horizon.start_year) * 12 + (month - sm)
        changepoints.append((max(0, sid), value))
    changepoints.sort()
    pre = _repeating_year_steps(pre_recs, transform)
    pos = _repeating_year_steps(pos_recs, transform)

    result: dict[int, float] = {}
    cp_idx = 0
    current = pre.get((sm - 2) % 12 + 1)
    for stage_id in range(study_months):
        while cp_idx < len(changepoints) and changepoints[cp_idx][0] <= stage_id:
            raw = changepoints[cp_idx][1]
            current = None if raw >= BIG_M else transform(raw)
            cp_idx += 1
        if current is not None:
            result[stage_id] = current

    if total_stages <= study_months:
        return result

    if pos:
        for stage_id in range(study_months, total_stages):
            value = pos[((sm - 1 + stage_id) % 12) + 1]
            if value is not None:
                result[stage_id] = value
    elif seasonalize:
        seasonal: dict[int, float] = {}
        for stage_id in range(max(0, study_months - 12), study_months):
            if stage_id in result:
                cal = ((sm - 1 + stage_id) % 12) + 1
                seasonal[cal] = result[stage_id]
        for stage_id in range(study_months, total_stages):
            cal = ((sm - 1 + stage_id) % 12) + 1
            if cal in seasonal:
                result[stage_id] = seasonal[cal]
    else:
        last = study_months - 1
        if last in result:
            for stage_id in range(study_months, total_stages):
                result[stage_id] = result[last]

    return result
