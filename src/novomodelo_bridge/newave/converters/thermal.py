"""Thermal entity converter: maps the source model thermal plant data to Cobre thermal
JSON.

Also provides ``convert_thermal_bounds`` which builds a per-stage
``thermal_bounds.parquet`` from ``term.dat`` (monthly and remaining-years
minimum), ``expt.dat`` (temporal capacity/factor/TEIF/GTMIN/IPTER overrides) and
``manutt.dat`` (scheduled maintenance windows).
"""

from __future__ import annotations

import calendar
import logging
import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pyarrow as pa

from cobre_bridge.cobre import schemas as cobre_schemas
from cobre_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    emit,
    format_stage_ranges,
)
from cobre_bridge.newave.case import NewaveCase
from cobre_bridge.newave.converters.anticipated import read_anticipated_dispatch
from cobre_bridge.newave.horizon import build_stage_dates, historical_start_date
from cobre_bridge.newave.id_map import NewaveIdMap

_LOG = logging.getLogger(__name__)

# Parquet schema for per-stage thermal generation bounds.
_THERMAL_BOUNDS_SCHEMA = pa.schema(
    [
        pa.field("thermal_id", pa.int32()),
        pa.field("stage_id", pa.int32()),
        pa.field("min_generation_mw", pa.float64()),
        pa.field("max_generation_mw", pa.float64()),
        pa.field("cost_per_mwh", pa.float64()),
    ]
)


def thermal_generation_bounds(case: NewaveCase) -> dict[int, tuple[float, float]]:
    """Return ``{codigo_usina: (min_mw, max_mw)}`` static generation bounds.

    These are the plant-level (non-stage-varying) bounds written to
    ``thermals.json`` as ``generation.min_mw`` / ``generation.max_mw``, and the
    interval Cobre's semantic validator enforces on each
    ``past_anticipated_commitments.values_mw`` entry.

    The pair is the **envelope** of the per-stage bounds
    :func:`convert_thermal_bounds` writes: the smallest minimum and the largest
    maximum the plant reaches over the horizon, so it can never be tighter than
    the stage Cobre enforces it at. Reading the TERM.DAT registry pair directly
    instead would ignore EXPT.DAT, MANUTT.DAT and the maintenance-year IP rule,
    which enter only through those stages — and for a plant whose registry GTMIN
    exceeds its registry capacity product it would publish an inverted interval,
    which Cobre refuses to load (``max_mw`` must be >= ``min_mw``).

    A plant no capacity source describes is absent from the mapping; callers
    supply their own default.
    """
    model = _StageBoundsModel.build(case)
    bounds: dict[int, tuple[float, float]] = {}
    for code in sorted(model.codes):
        plant = model.bounds_for(code)
        bounds[code] = (min(plant.min_mw), max(plant.max_mw))
    return bounds


def convert_thermals(case: NewaveCase, id_map: NewaveIdMap) -> dict:
    """Convert the source model thermal plant data to a Cobre ``thermals.json`` dict.

    Reads ``conft.dat``, ``clast.dat``, and ``term.dat`` from *case*.
    Returns a dict with a ``"thermals"`` key containing a list of thermal
    entries sorted by Cobre 0-based ID.

    Parameters
    ----------
    case:
        Parsed the source model case.
    id_map:
        Pre-built ID mapping for bus cross-references.
    """
    conft_df = case.conft.usinas
    clast_df = case.clast.usinas

    # Anticipated dispatch (the source model GNL) — gated by
    # dger.despacho_antecipado_gnl. Returns an empty dict when the flag is off, so
    # non-GNL cases incur zero cost beyond the dger read inside the helper.
    anticipated_by_code = read_anticipated_dispatch(case)

    # Build cost lookup: codigo_usina -> cost for indice_ano_estudo == 1.
    cost_map: dict[int, float] = {}
    if clast_df is not None:
        first_year = clast_df[clast_df["indice_ano_estudo"] == 1]
        for _, row in first_year.iterrows():
            cost_map[int(row["codigo_usina"])] = float(row["valor"])

    # Static (min_mw, max_mw) per plant — single-sourced so the anticipated
    # commitment seeding clamps against the very bounds written here.
    gen_bounds = thermal_generation_bounds(case)

    # The source model carries no per-thermal commissioning date; treat every
    # thermal as in service since the historical record (Cobre uses the date only
    # as a canonical-ordering key, tiebroken by id).
    op_date = historical_start_date(case.dger)

    thermals: list[dict] = []
    for _, row in conft_df.iterrows():
        newave_code = int(row["codigo_usina"])
        name = str(row["nome_usina"]).strip()
        submercado = int(row["submercado"])

        bus_id = id_map.bus_id(submercado)

        gen_min, max_mw = gen_bounds.get(newave_code, (0.0, 0.0))
        cost = cost_map.get(newave_code, 0.0)

        anticipated = anticipated_by_code.get(newave_code)
        anticipated_config = (
            {"lead_stages": anticipated.lead_stages} if anticipated else None
        )

        thermal_entry: dict = {
            "id": id_map.thermal_id(newave_code),
            "name": name,
            "operational_start_date": op_date,
            "bus_id": bus_id,
            "cost_per_mwh": cost,
            "generation": {
                "min_mw": gen_min,
                "max_mw": max_mw,
            },
            "anticipated_config": anticipated_config,
            "entry_stage_id": None,
            "exit_stage_id": None,
        }
        thermals.append(thermal_entry)

    thermals.sort(key=lambda t: t["id"])

    return {
        "$schema": cobre_schemas.schema_url_for("system/thermals.json"),
        "thermals": thermals,
    }


def _apply_maint_to_capacity(
    base_capacity: float,
    maint_rows: pd.DataFrame,
    stage_dates: list[date],
) -> np.ndarray:
    """Compute monthly effective capacity after subtracting maintenance windows.

    For each stage (month), builds a daily-resolution view of the month and
    subtracts ``potencia`` (MW) for each maintenance unit whose window
    overlaps that month.  Multiple units (different ``codigo_unidade``) can be
    under maintenance simultaneously and are treated additively.

    Parameters
    ----------
    base_capacity:
        Installed capacity in MW (sum across all units).
    maint_rows:
        DataFrame slice for one thermal plant with columns
        ``data_inicio`` (datetime), ``duracao`` (int, days), ``potencia`` (float).
    stage_dates:
        First-of-month dates for every study stage.

    Returns
    -------
    np.ndarray
        Shape (total_stages,), dtype float64.  Each element is the monthly
        average effective capacity after maintenance.
    """
    total_stages = len(stage_dates)
    effective = np.full(total_stages, base_capacity, dtype=float)

    for _, row in maint_rows.iterrows():
        start_dt = pd.Timestamp(row["data_inicio"])
        duration_days = int(row["duracao"])
        unit_power = float(row["potencia"])
        end_dt = start_dt + timedelta(days=duration_days)

        for stage_idx, stage_start in enumerate(stage_dates):
            _, days_in_month = calendar.monthrange(stage_start.year, stage_start.month)
            # First day of the following month (exclusive upper bound).
            if stage_start.month == 12:
                stage_end = date(stage_start.year + 1, 1, 1)
            else:
                stage_end = date(stage_start.year, stage_start.month + 1, 1)

            maint_start_date = start_dt.date()
            maint_end_date = end_dt.date()

            # Overlap of [maint_start, maint_end) with [stage_start, stage_end).
            overlap_start = max(maint_start_date, stage_start)
            overlap_end = min(maint_end_date, stage_end)
            overlap_days = (overlap_end - overlap_start).days
            if overlap_days <= 0:
                continue

            # Fraction of the month under maintenance for this unit.
            fraction = overlap_days / days_in_month
            effective[stage_idx] -= unit_power * fraction

    return effective


def _stage_to_study_year(
    stage_idx: int,
    first_year_stages: int,
    num_anos: int,
) -> int:
    """Map a 0-based stage index to a 1-based ``indice_ano_estudo``.

    The first study year covers ``first_year_stages`` months (``13 -
    start_month``).  Subsequent years cover 12 months each.  Post-study
    stages are clamped to the last study year.
    """
    if stage_idx < first_year_stages:
        return 1
    year = (stage_idx - first_year_stages) // 12 + 2
    return min(year, num_anos)


@dataclass
class _StageInputs:
    """Per-stage thermal parameters each ``_step*`` helper transforms in place."""

    potencia: float
    fcmax: float
    teif: float
    ip: float
    gen_min: float


def _step1_zero_ip_before_maintenance(
    state: _StageInputs, stage_idx: int, maint_end_stage: int
) -> None:
    """Step 1: zero IP for ALL plants in stages before the maintenance end."""
    if stage_idx < maint_end_stage:
        state.ip = 0.0


def _step4_apply_expt_overrides(
    state: _StageInputs,
    overrides: list[dict],
    ref_date: date,
    is_post_study: bool,
    last_stage_date: date,
) -> None:
    """Step 4: apply EXPT overrides (POTEF/FCMAX/TEIFT/GTMIN/IPTER) in file order.

    Closed windows test against ``ref_date`` (frozen at the last study stage in
    the post-study tail); an open-ended override blankets the whole tail and,
    coming last in file order, wins over any per-month window for the stage.
    """
    for override in overrides:
        ov_start = pd.Timestamp(override["data_inicio"]).date()
        ov_end_raw = override["data_fim"]
        open_ended = pd.isna(ov_end_raw)
        ov_end = last_stage_date if open_ended else pd.Timestamp(ov_end_raw).date()
        if open_ended and is_post_study:
            applies = True
        else:
            applies = ov_start <= ref_date <= ov_end
        if not applies:
            continue

        tipo = override["tipo"]
        value = override["modificacao"]
        if tipo == "POTEF":
            state.potencia = value
        elif tipo == "FCMAX":
            state.fcmax = value
        elif tipo == "TEIFT":
            state.teif = value
        elif tipo == "GTMIN":
            state.gen_min = value
        elif tipo == "IPTER":
            state.ip = value


def _covers(windows: list[tuple[date, date]] | None, when: date) -> bool:
    """Whether some inclusive ``(start, end)`` window contains ``when``."""
    return any(start <= when <= end for start, end in windows or ())


def _step4b_apply_potef_availability(
    state: _StageInputs,
    windows: list[tuple[date, date]] | None,
    stage_date: date,
    *,
    nullified: bool,
) -> None:
    """Step 4b: a plant CONFT nulls is out of service outside every POTEF window.

    ``nullified`` is the CONFT.DAT ``EE``/``NE`` status, for which the source
    model discards the TERM.DAT capacity and minimum; an ``EX`` plant keeps them
    wherever EXPT declares nothing. ``stage_date`` is the post-study freeze date
    in the tail, not the stage's own date.
    """
    if nullified and not _covers(windows, stage_date):
        state.potencia = 0.0
        state.gen_min = 0.0


def _step4c_apply_gtmin_availability(
    state: _StageInputs,
    windows: list[tuple[date, date]] | None,
    stage_date: date,
    *,
    nullified: bool,
) -> None:
    """Step 4c: a plant CONFT nulls has no minimum outside every GTMIN window.

    Same rule as step 4b on the lower bound only, so a TERM.DAT GTMIN cannot leak
    in as a spurious must-run; capacity is step 4b's.
    """
    if nullified and not _covers(windows, stage_date):
        state.gen_min = 0.0


def _potef_online_at(
    windows: list[tuple[date, date]] | None,
    *,
    nullified: bool,
    when: date,
) -> bool:
    """Whether a plant has installed capacity at ``when`` per its POTEF schedule."""
    return not nullified or _covers(windows, when)


def _step5_apply_maint_reduction(
    state: _StageInputs,
    maint_reduction: np.ndarray | None,
    stage_idx: int,
    maint_end_stage: int,
) -> None:
    """Step 5: MANUTT subtracts its capacity reduction from ``potencia``.

    Applied only in stages before the maintenance end, matching sintetizador,
    which applies EXPT (step 4) before MANUTT.
    """
    if maint_reduction is not None and stage_idx < maint_end_stage:
        state.potencia -= float(maint_reduction[stage_idx])


#: Half the 0.01 MW resolution the deck writes GTMIN with: a GTMIN set to the
#: available capacity and rounded to that resolution can exceed it by up to this
#: much without being a data error.
_GTMIN_ROUNDING_MW = 0.005


def _step6_evaluate_bounds(state: _StageInputs) -> tuple[float, float, bool]:
    """Step 6: evaluate ``(min_mw, max_mw, gtmin_above_capacity)``.

    FCMAX bounds the maximum and GTMIN the minimum independently. A GTMIN above
    the available capacity is a data error the source model rejects; the
    inflexible minimum is honored and the maximum lifted to it to keep the LP
    feasible — clamping the minimum down instead would silently run the plant
    below its GTMIN. An excess within the deck's rounding of GTMIN
    (:data:`_GTMIN_ROUNDING_MW`) still lifts the maximum but is not flagged.
    """
    capacity_max = _capacity_max(state)
    min_mw = max(0.0, state.gen_min)
    gtmin_above_capacity = min_mw > capacity_max + _GTMIN_ROUNDING_MW
    max_mw = max(capacity_max, min_mw)
    return min_mw, max_mw, gtmin_above_capacity


def _capacity_max(state: _StageInputs) -> float:
    """Available maximum generation: ``potencia·(fcmax)·(100-ip)·(100-teif)`` (MW)."""
    potencia = max(0.0, state.potencia)
    return max(
        0.0,
        potencia
        * (state.fcmax / 100.0)
        * ((100.0 - state.ip) / 100.0)
        * ((100.0 - state.teif) / 100.0),
    )


@dataclass
class _GtminRecord:
    """One stage where a thermal plant's GTMIN exceeded its available capacity."""

    code: int
    stage_id: int
    gtmin_mw: float
    capacity_mw: float


@dataclass(frozen=True)
class _PlantStageBounds:
    """One plant's per-stage bounds in MW, plus what the caller may want to surface.

    ``ref_dates`` is the date each stage was evaluated at — the stage's own date
    in-study and the frozen freeze-reference date in the post-study tail. Every
    other date-dependent per-stage lookup (CLAST cost modifications) must test
    against it, not against the stage date, or the tail re-applies the last
    year's seasonal pattern.
    """

    min_mw: list[float]
    max_mw: list[float]
    ref_dates: list[date]
    gtmin_records: list[_GtminRecord]


@dataclass(frozen=True)
class _StageBoundsModel:
    """TERM/EXPT/MANUTT/CONFT parsed into the per-stage bound steps' inputs.

    Single source for both the per-stage ``thermal_bounds.parquet`` and the
    static envelope in ``thermals.json``; :meth:`bounds_for` emits no
    diagnostic, so the two callers cannot double-emit.
    """

    stage_dates: list[date]
    study_months: int
    first_year_stages: int
    maint_end_stage: int
    codes: frozenset[int]
    """Every plant the capacity sources describe (TERM, EXPT or MANUTT)."""
    nullified_codes: frozenset[int]
    codes_without_potef: frozenset[int]
    base_by_code_month: dict[tuple[int, int], dict[str, float]]
    base_default: dict[int, dict[str, float]]
    gen_min_other_years: dict[int, float]
    expt_by_code: dict[int, list[dict]]
    potef_windows: dict[int, list[tuple[date, date]]]
    gtmin_windows: dict[int, list[tuple[date, date]]]
    manutt_by_code: dict[int, pd.DataFrame]

    @classmethod
    def build(cls, case: NewaveCase) -> _StageBoundsModel:
        """Parse the capacity sources; reads no cost and emits no diagnostic."""
        horizon = case.horizon
        num_maint_years: int = case.dger.num_anos_manutencao_utes or 0
        # Maintenance years are counted as full calendar years from the study
        # start year.  For a March 2026 start with 1 maintenance year, the
        # period covers March-December 2026 (10 stages), not 12.
        maint_end_stage = num_maint_years * 12 + (1 - horizon.start_month)
        stage_dates = build_stage_dates(
            horizon.start_year, horizon.start_month, horizon.total_stages
        )

        term_df = case.term.usinas
        base_by_code_month: dict[tuple[int, int], dict[str, float]] = {}
        base_default: dict[int, dict[str, float]] = {}
        gen_min_other_years: dict[int, float] = {}
        if term_df is not None:
            for _, row in term_df.iterrows():
                code = int(row["codigo_usina"])
                mes = int(row["mes"])
                # ``mes == 13`` is TERM.DAT's single minimum for the years after
                # the first study year. ``inewave`` emits that row for every
                # plant and decodes a blank field as NaN; such a plant keeps its
                # calendar-month column instead.
                if mes == 13:
                    gen_min = float(row["geracao_minima"])
                    if not math.isnan(gen_min):
                        gen_min_other_years[code] = gen_min
                    continue
                values = {
                    "potencia": float(row["potencia_instalada"]),
                    "fcmax": float(row["fator_capacidade_maximo"]),
                    "teif": float(row.get("teif", 0.0)),
                    "ip": float(row.get("indisponibilidade_programada", 0.0)),
                    "gen_min": float(row["geracao_minima"]),
                }
                if 1 <= mes <= 12:
                    base_by_code_month[(code, mes)] = values
                base_default.setdefault(code, values)

        expt_by_code: dict[int, list[dict]] = {}
        if case.files.expt is not None:
            try:
                expt_df = case.expt.expansoes
                for _, row in expt_df.iterrows():
                    expt_by_code.setdefault(int(row["codigo_usina"]), []).append(
                        {
                            "tipo": str(row["tipo"]),
                            "modificacao": float(row["modificacao"]),
                            "data_inicio": row["data_inicio"],
                            "data_fim": row["data_fim"],
                        }
                    )
            except Exception:  # noqa: BLE001
                _LOG.warning("expt.dat could not be parsed; EXPT overrides skipped.")

        potef_windows: dict[int, list[tuple[date, date]]] = {}
        gtmin_windows: dict[int, list[tuple[date, date]]] = {}

        def _window(o: dict) -> tuple[date, date]:
            start = pd.Timestamp(o["data_inicio"]).date()
            end_raw = o["data_fim"]
            end = stage_dates[-1] if pd.isna(end_raw) else pd.Timestamp(end_raw).date()
            return start, end

        for code, overrides in expt_by_code.items():
            for o in overrides:
                if o["tipo"] == "POTEF":
                    potef_windows.setdefault(code, []).append(_window(o))
                elif o["tipo"] == "GTMIN":
                    gtmin_windows.setdefault(code, []).append(_window(o))

        # Only ``EE``/``NE`` have their registry capacity and minimum discarded;
        # ``EX``, or a status the deck leaves blank, keeps them operative.
        nullified_codes = {
            int(row["codigo_usina"])
            for _, row in case.conft.usinas.iterrows()
            if str(row["usina_existente"]).strip() in ("EE", "NE")
        }

        manutt_by_code: dict[int, pd.DataFrame] = {}
        manutt_df: pd.DataFrame | None = None
        if case.files.manutt is not None:
            try:
                manutt_df = case.manutt.manutencoes
            except Exception:  # noqa: BLE001
                _LOG.warning(
                    "%s could not be parsed; maintenance skipped.",
                    case.files.manutt.name,
                )
        # A file with no maintenance record reads as None: no maintenance.
        if manutt_df is not None:
            for code, grp in manutt_df.groupby("codigo_usina"):
                manutt_by_code[int(code)] = grp.reset_index(drop=True)

        return cls(
            stage_dates=stage_dates,
            study_months=horizon.study_months,
            first_year_stages=horizon.first_year_stages,
            maint_end_stage=maint_end_stage,
            codes=frozenset(
                set(expt_by_code) | set(manutt_by_code) | set(base_default)
            ),
            nullified_codes=frozenset(nullified_codes),
            codes_without_potef=frozenset(nullified_codes - potef_windows.keys()),
            base_by_code_month=base_by_code_month,
            base_default=base_default,
            gen_min_other_years=gen_min_other_years,
            expt_by_code=expt_by_code,
            potef_windows=potef_windows,
            gtmin_windows=gtmin_windows,
            manutt_by_code=manutt_by_code,
        )

    def _base(self, code: int, cal_month: int, stage_idx: int) -> dict[str, float]:
        row = self.base_by_code_month.get((code, cal_month)) or self.base_default.get(
            code
        )
        base = (
            dict(row)
            if row is not None
            else {
                "potencia": 0.0,
                "fcmax": 100.0,
                "teif": 0.0,
                "ip": 0.0,
                "gen_min": 0.0,
            }
        )
        # The monthly minimum columns cover the first study year, not the
        # maintenance years; the other four fields repeat across month rows.
        if stage_idx >= self.first_year_stages and code in self.gen_min_other_years:
            base["gen_min"] = self.gen_min_other_years[code]
        return base

    def bounds_for(self, code: int) -> _PlantStageBounds:
        """Run the per-stage bound steps for one plant over every stage."""
        overrides = self.expt_by_code.get(code, [])
        nullified = code in self.nullified_codes
        potef_windows = self.potef_windows.get(code)

        # MANUTT's reduction is a delta from the TERM.DAT capacity, applied to
        # the EXPT-modified ``potencia`` (step 5), matching sintetizador which
        # applies EXPT before MANUTT.
        maint_rows = self.manutt_by_code.get(code)
        maint_reduction: np.ndarray | None = None
        if maint_rows is not None and not maint_rows.empty:
            base_cap = self.base_default.get(code, {}).get("potencia", 0.0)
            effective = _apply_maint_to_capacity(base_cap, maint_rows, self.stage_dates)
            maint_reduction = np.maximum(0.0, base_cap - effective)

        # The source model's "período estático final" freezes the post-study tail
        # at one December snapshot (min, max and cost; maintenance ignored): the
        # last study stage, or the terminal stage for a plant whose POTEF brings
        # it online only in the tail. Evaluating each tail stage at its own date
        # instead would repeat the last year's seasonal pattern across the tail.
        last_study_idx = self.study_months - 1
        comes_online_in_post_study = not _potef_online_at(
            potef_windows, nullified=nullified, when=self.stage_dates[last_study_idx]
        ) and _potef_online_at(
            potef_windows, nullified=nullified, when=self.stage_dates[-1]
        )
        freeze_idx = (
            len(self.stage_dates) - 1 if comes_online_in_post_study else last_study_idx
        )

        min_mw: list[float] = []
        max_mw: list[float] = []
        ref_dates: list[date] = []
        gtmin_records: list[_GtminRecord] = []
        for stage_idx, stage_date in enumerate(self.stage_dates):
            is_post_study = stage_idx >= self.study_months
            ref_date = self.stage_dates[freeze_idx] if is_post_study else stage_date
            state = _StageInputs(**self._base(code, ref_date.month, stage_idx))

            _step1_zero_ip_before_maintenance(state, stage_idx, self.maint_end_stage)
            _step4_apply_expt_overrides(
                state, overrides, ref_date, is_post_study, self.stage_dates[-1]
            )
            _step4b_apply_potef_availability(
                state, potef_windows, ref_date, nullified=nullified
            )
            _step4c_apply_gtmin_availability(
                state, self.gtmin_windows.get(code), ref_date, nullified=nullified
            )
            _step5_apply_maint_reduction(
                state, maint_reduction, stage_idx, self.maint_end_stage
            )
            stage_min, stage_max, gtmin_above_capacity = _step6_evaluate_bounds(state)
            if gtmin_above_capacity:
                gtmin_records.append(
                    _GtminRecord(
                        code=code,
                        stage_id=stage_idx,
                        gtmin_mw=stage_min,
                        capacity_mw=_capacity_max(state),
                    )
                )
            min_mw.append(stage_min)
            max_mw.append(stage_max)
            ref_dates.append(ref_date)

        return _PlantStageBounds(
            min_mw=min_mw,
            max_mw=max_mw,
            ref_dates=ref_dates,
            gtmin_records=gtmin_records,
        )


def _thermal_names(case: NewaveCase) -> dict[int, str]:
    """Map thermal codes to plant names from conft.dat."""
    return {
        int(row["codigo_usina"]): str(row["nome_usina"]).strip()
        for _, row in case.conft.usinas.iterrows()
    }


def _emit_gtmin_above_capacity(records: list[_GtminRecord], case: NewaveCase) -> None:
    """Emit the per-plant GTMIN-above-capacity diagnostic from the captured records.

    One row per plant: the stages it occurred on (collapsed to ranges), the worst
    GTMIN vs the lowest available capacity across those stages, and the largest
    single-stage excess — so the user sees the plant, where, and by how much,
    instead of a bare code list. Values keep the deck's 0.01 MW resolution.
    """
    names = _thermal_names(case)
    by_code: dict[int, list[_GtminRecord]] = {}
    for record in records:
        by_code.setdefault(record.code, []).append(record)

    rows: list[list[object]] = []
    for code in sorted(by_code):
        recs = by_code[code]
        rows.append(
            [
                names.get(code, "?"),
                code,
                format_stage_ranges(r.stage_id for r in recs),
                round(max(r.gtmin_mw for r in recs), 2),
                round(min(r.capacity_mw for r in recs), 2),
                round(max(r.gtmin_mw - r.capacity_mw for r in recs), 2),
            ]
        )

    plant_count = len(by_code)
    emit(
        Diagnostic(
            code="thermal-gtmin-above-capacity",
            severity=Severity.WARNING,
            category="Thermal bounds",
            title=f"GTMIN exceeds available capacity ({plant_count} plant(s))",
            summary=(
                f"GTMIN exceeds the available capacity (after FCMAX, TEIF, IP and "
                f"maintenance) for {plant_count} thermal plant(s) in at least one "
                "stage; honoring GTMIN to keep the LP feasible."
            ),
            table=DiagnosticTable(
                columns=["Plant", "Code", "Stages", "GTMIN MW", "Cap MW", "Excess MW"],
                rows=rows,
                justify=["left", "right", "left", "right", "right", "right"],
                caption=(
                    "GTMIN = max, Cap = min available capacity, Excess = largest "
                    "single-stage excess across the listed stages"
                ),
            ),
            remediation="Check EXPT FCMAX/GTMIN and MANUTT for these plants.",
        ),
        logger=_LOG,
    )


def convert_thermal_bounds(
    case: NewaveCase,
    id_map: NewaveIdMap,
) -> pa.Table | None:
    """Build per-stage thermal generation bounds and ``clast.dat`` cost overrides.

    Returns ``None`` when every plant's bounds are stage-invariant and no cost
    varies, since ``thermals.json`` then already carries everything.
    """
    num_anos = case.horizon.num_anos
    first_year_stages = case.horizon.first_year_stages

    clast = case.clast
    clast_df = clast.usinas
    clast_modif_df = clast.modificacoes

    # cost_by_code_year: (newave_code, indice_ano_estudo) -> cost
    cost_by_code_year: dict[tuple[int, int], float] = {}
    # Track which thermals have costs that vary across years.
    cost_varies: set[int] = set()
    if clast_df is not None:
        for _, row in clast_df.iterrows():
            code = int(row["codigo_usina"])
            year_idx = int(row["indice_ano_estudo"])
            cost_by_code_year[(code, year_idx)] = float(row["valor"])
        # Detect thermals with non-uniform costs.
        codes_in_clast = {c for c, _ in cost_by_code_year}
        for code in codes_in_clast:
            year_costs = [
                cost_by_code_year[(code, y)]
                for y in range(1, num_anos + 1)
                if (code, y) in cost_by_code_year
            ]
            if len(set(year_costs)) > 1:
                cost_varies.add(code)

    # Date-range cost overrides from the modificacoes block at the end of
    # clast.dat. Each entry overrides the year-indexed cost for stages
    # whose first-of-month date falls within [data_inicio, data_fim].
    # A plant with at least one modification is treated as cost-varying
    # even when its year-indexed costs are uniform.
    modif_by_code: dict[int, list[dict]] = {}
    if clast_modif_df is not None and not clast_modif_df.empty:
        for _, row in clast_modif_df.iterrows():
            code = int(row["codigo_usina"])
            modif_by_code.setdefault(code, []).append(
                {
                    "data_inicio": row["data_inicio"],
                    "data_fim": row["data_fim"],
                    "custo": float(row["custo"]),
                }
            )
        cost_varies.update(modif_by_code.keys())

    model = _StageBoundsModel.build(case)

    if model.codes_without_potef:
        names = _thermal_names(case)
        emit(
            Diagnostic(
                code="thermal-expt-without-potef",
                severity=Severity.INFO,
                category="Thermal bounds",
                title=(
                    f"Plants without a POTEF entry ({len(model.codes_without_potef)})"
                ),
                summary=(
                    f"{len(model.codes_without_potef)} thermal plant(s) are marked "
                    "EE or NE in CONFT.DAT with no POTEF entry in EXPT.DAT; the "
                    "model discards their registry capacity and declares none, "
                    "so they are treated as not installed (max generation 0)."
                ),
                table=DiagnosticTable(
                    columns=["Plant", "Code"],
                    rows=[
                        [names.get(code, "?"), code]
                        for code in sorted(model.codes_without_potef)
                    ],
                    justify=["left", "right"],
                ),
            ),
            logger=_LOG,
        )

    plants: dict[int, _PlantStageBounds] = {}
    thermal_ids: dict[int, int] = {}
    for newave_code in sorted(model.codes | cost_varies):
        try:
            thermal_ids[newave_code] = id_map.thermal_id(newave_code)
        except KeyError:
            continue
        plants[newave_code] = model.bounds_for(newave_code)

    gtmin_records = [r for plant in plants.values() for r in plant.gtmin_records]
    if gtmin_records:
        _emit_gtmin_above_capacity(gtmin_records, case)

    # A stage-invariant plant's bounds equal the ``thermals.json`` envelope
    # (``thermal_generation_bounds``), so only a varying one needs the table.
    bounds_vary = any(
        len(set(plant.min_mw)) > 1 or len(set(plant.max_mw)) > 1
        for plant in plants.values()
    )
    if not bounds_vary and not cost_varies:
        _LOG.debug("Thermal bounds and costs are stage-invariant; no table emitted.")
        return None

    rows_thermal_id: list[int] = []
    rows_stage_id: list[int] = []
    rows_min: list[float] = []
    rows_max: list[float] = []
    rows_cost: list[float | None] = []

    for newave_code, plant in plants.items():
        for stage_idx, ref_date in enumerate(plant.ref_dates):
            stage_cost: float | None = None
            if newave_code in cost_varies:
                year_idx = _stage_to_study_year(stage_idx, first_year_stages, num_anos)
                stage_cost = cost_by_code_year.get((newave_code, year_idx))
                # Apply clast.modificacoes overrides in file order; later entries win
                # when windows overlap, matching the source model's sequential
                # application of the modification block. Tested against ``ref_date`` so
                # the post-study tail freezes at the last study stage's cost (December)
                # instead of letting a future-dated modification leak into the static
                # final period.
                for modif in modif_by_code.get(newave_code, []):
                    mod_start = pd.Timestamp(modif["data_inicio"]).date()
                    mod_end_raw = modif["data_fim"]
                    if pd.isna(mod_end_raw):
                        mod_end = model.stage_dates[-1]
                    else:
                        mod_end = pd.Timestamp(mod_end_raw).date()
                    if mod_start <= ref_date <= mod_end:
                        stage_cost = modif["custo"]

            rows_thermal_id.append(thermal_ids[newave_code])
            rows_stage_id.append(stage_idx)
            rows_min.append(plant.min_mw[stage_idx])
            rows_max.append(plant.max_mw[stage_idx])
            rows_cost.append(stage_cost)

    if not rows_thermal_id:
        return None

    return pa.table(
        {
            "thermal_id": pa.array(rows_thermal_id, type=pa.int32()),
            "stage_id": pa.array(rows_stage_id, type=pa.int32()),
            "min_generation_mw": pa.array(rows_min, type=pa.float64()),
            "max_generation_mw": pa.array(rows_max, type=pa.float64()),
            "cost_per_mwh": pa.array(rows_cost, type=pa.float64()),
        },
        schema=_THERMAL_BOUNDS_SCHEMA,
    )
