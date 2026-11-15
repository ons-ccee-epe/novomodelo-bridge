"""Hydro entity-record construction: the ``hydros.json`` per-plant builder
``convert_hydros`` and initial-storage conversion, the registry reader, and
the operated-set / cascade-walk helpers they build on.

Imports the capacity-cap helpers from ``.bounds`` and the
storage-envelope helpers from ``decomp.converters.cadastro``; within the package
only ``.turbine_limit`` imports from here, for the cascade walk.

The cascade walk (:func:`_downstream_operated`, shared by this module's own
``downstream_id`` entity field and ``scenarios.py``'s incremental-inflow
attribution) and ``scenarios.py``'s inflow-gauge attribution both read
:class:`~novomodelo_bridge.decomp.converters.cadastro.effective.EffectiveCadastro`'s
``downstream_plant``/``inflow_gauge`` accessors -- the post-``AC
NUMJUS``/``NUMPOS`` link/gauge -- rather than the base ``hidr`` columns
directly. ``AC JUSENA`` is a stored-energy link, not a water-routing one:
only the RHE stored energy reads it. ``AC NPOSNW`` (the boundary-cut coupling
gauge) is read only by the boundary FCF import, never by the operation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Protocol

import pandas as pd
from inewave.newave import Hidr

from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    emit,
    format_stage_ranges,
)
from novomodelo_bridge.core.hydro_units import build_mirror_unit_group
from novomodelo_bridge.core.pandas_utils import as_frame
from novomodelo_bridge.core.productivity import (
    forebay_curve_missing_diagnostic,
    fpha_efficiency,
    fpha_efficiency_capped_diagnostic,
    fpha_efficiency_capped_row,
    has_forebay_curve,
)
from novomodelo_bridge.decomp.converters.cadastro import (
    effective_storage_range,
    storage_envelope,
)
from novomodelo_bridge.decomp.converters.hydro.bounds import (
    _ITAIPU_CODE,
    _build_split_unit_groups,
    _rated_envelope,
    _split_plant_frequencies,
)
from novomodelo_bridge.novomodelo import schemas as novomodelo_schemas

if TYPE_CHECKING:
    from pathlib import Path

    from novomodelo_bridge.decomp.case import DecompCase
    from novomodelo_bridge.decomp.converters.cadastro import (
        EffectiveCadastro,
    )
    from novomodelo_bridge.decomp.converters.hydro.turbine_limit import TurbineLimits
    from novomodelo_bridge.decomp.id_map import DecompIdMap

_LOG = logging.getLogger(__name__)


def read_hidr(path: Path) -> pd.DataFrame:
    """Read the hydro registry file, indexed by plant code."""
    df = Hidr.read(str(path)).cadastro
    if df is None or df.empty:
        raise FileNotFoundError(f"{path} has no registry data")
    return df


class UhRegisters(Protocol):
    def uh(self, *, df: bool) -> object: ...


def _operated_uh(dadger: UhRegisters) -> pd.DataFrame:
    """The ``UH`` rows carrying an initial volume (the operated plants).

    Rows without one (coupling-only registrations) are excluded — their
    terminal-value treatment is the boundary importer's concern, not the
    registry's. :func:`convert_hydros` reports the excluded set (by name) once.
    """
    uh = as_frame(dadger.uh(df=True))
    if uh is None or uh.empty:
        raise ValueError("the deck has no UH records; cannot convert hydros")
    return uh[uh["volume_inicial"].notna()]


def initial_volume_percentages(dadger: UhRegisters) -> dict[int, float]:
    """``{code: UH volume_inicial}`` for every operated plant, in % of useful volume."""
    return {
        int(row["codigo_usina"]): float(row["volume_inicial"])
        for _, row in _operated_uh(dadger).iterrows()
    }


def _coupling_only_codes(dadger: UhRegisters) -> list[int]:
    """Operated-set-excluded ``UH`` codes — registrations with no initial volume."""
    uh = as_frame(dadger.uh(df=True))
    if uh is None or uh.empty:
        return []
    excluded = uh[uh["volume_inicial"].isna()]
    return sorted(int(c) for c in excluded["codigo_usina"])


def _registry_name(hidr: pd.DataFrame, code: int) -> str:
    """A plant's ``hidr.dat`` name, or ``""`` when the registry lacks it."""
    try:
        return str(hidr.loc[code, "nome_usina"]).strip()
    except KeyError:
        return ""


def _coupling_only_diagnostic(hidr: pd.DataFrame, codes: list[int]) -> Diagnostic:
    return Diagnostic(
        code="decomp-coupling-only-plants-excluded",
        severity=Severity.INFO,
        category="Entity exclusion",
        title=f"Plants DECOMP does not operate left out ({len(codes)})",
        summary=(
            "The listed plants have UH records in dadger.rvN with no initial "
            "volume: DECOMP does not operate them and keeps them only for the "
            "coupling with NEWAVE, so the case leaves them out."
        ),
        table=DiagnosticTable(
            columns=["Code", "Name"],
            rows=[[code, _registry_name(hidr, code)] for code in codes],
            justify=["right", "left"],
        ),
    )


class CascadeLinks(Protocol):
    @property
    def base(self) -> pd.DataFrame: ...

    def downstream_plant(self, code: int, stage_index: int) -> int: ...


def _cascade_walk(
    effective: CascadeLinks, code: int, operated: set[int], stage_index: int
) -> tuple[int | None, int | None]:
    """``(next operated plant, invalid link)`` below *code* at *stage_index*.

    Non-operated intermediates are skipped through and code 0 is the sink. A
    link that loops back or names a plant missing from ``hidr.dat`` ends the
    walk as a sink and is returned as the invalid link.
    """
    visited = {code}
    current = effective.downstream_plant(code, stage_index)
    while current != 0 and current not in operated:
        if current in visited or current not in effective.base.index:
            return None, current
        visited.add(current)
        current = effective.downstream_plant(current, stage_index)
    return (current if current != 0 else None), None


def _downstream_operated(
    effective: CascadeLinks,
    code: int,
    operated: set[int],
    *,
    stage_index: int = 0,
) -> int | None:
    """The next *operated* plant below *code* on the effective (post-``AC
    NUMJUS``) cascade at *stage_index*, or ``None`` at the sink.

    Every caller reads the first stage: the case has one cascade for the
    whole horizon, and the plants whose ``AC NUMJUS`` changes it on later
    stages are reported
    (:func:`novomodelo_bridge.decomp.converters.cadastro.topology_varies_diagnostic`).
    """
    return _cascade_walk(effective, code, operated, stage_index)[0]


def _cascade_link_invalid_diagnostic(rows: list[list[object]]) -> Diagnostic:
    return Diagnostic(
        code="decomp-cascade-link-invalid",
        severity=Severity.WARNING,
        category="Cadastro overrides",
        title=f"Cascade links that loop or leave hidr.dat ({len(rows)})",
        summary=(
            "Following the downstream links of the listed plants reaches a "
            "plant already passed, or one missing from hidr.dat, so the "
            "conversion ends their cascade at that link and their outflow "
            "leaves the system there."
        ),
        table=DiagnosticTable(
            columns=["Code", "Name", "Invalid link"],
            rows=rows,
            justify=["right", "left", "right"],
        ),
        remediation=(
            "Correct the downstream plant of each listed link in hidr.dat or "
            "its AC NUMJUS register in dadger.rvN."
        ),
    )


#: ``hidr.dat``'s 12 monthly reservoir-evaporation columns in calendar order
#: (Jan..Dec) — novomodelo's ``evaporation.coefficients_mm`` is index 0 = January.
_EVAPORATION_MONTH_COLUMNS = (
    "evaporacao_JAN",
    "evaporacao_FEV",
    "evaporacao_MAR",
    "evaporacao_ABR",
    "evaporacao_MAI",
    "evaporacao_JUN",
    "evaporacao_JUL",
    "evaporacao_AGO",
    "evaporacao_SET",
    "evaporacao_OUT",
    "evaporacao_NOV",
    "evaporacao_DEZ",
)


def _evaporation_flag_codes(dadger: UhRegisters) -> set[int]:
    """Plant codes whose ``UH`` register sets the reservoir-evaporation flag.

    DECOMP models a plant's evaporation only when its ``UH`` ``evaporacao``
    flag is set; a plant absent from this set is emitted with no evaporation
    regardless of the coefficients ``hidr.dat`` may carry.
    """
    uh = as_frame(dadger.uh(df=True))
    if uh is None or "evaporacao" not in uh.columns or "codigo_usina" not in uh.columns:
        return set()
    return set(uh.loc[uh["evaporacao"].fillna(0) != 0, "codigo_usina"].astype(int))


def _evaporation_coefficients_mm(hidr: pd.DataFrame, code: int) -> list[float] | None:
    """The plant's 12 monthly evaporation coefficients [mm/month], Jan..Dec,
    from ``hidr.dat`` — or ``None`` when the plant is absent or every month is
    zero (nothing to model).

    novomodelo (>= 0.14) computes the evaporation-outflow model internally from
    these monthly rates and the reservoir area geometry
    (``hydro_geometry.parquet``, emitted by the FPHA path), scaling each stage
    to its ``stage_hours / month_hours`` share of the calendar month. The
    bridge supplies only the monthly mm rates.
    """
    if code not in hidr.index or not all(
        column in hidr.columns for column in _EVAPORATION_MONTH_COLUMNS
    ):
        return None
    row = hidr.loc[code]
    coefficients = [float(row[column]) for column in _EVAPORATION_MONTH_COLUMNS]
    if not any(coefficients):
        return None
    return coefficients


@dataclass(frozen=True)
class _HydroContext:
    """The case-wide inputs every plant's ``hydros.json`` entry reads."""

    hidr: pd.DataFrame
    effective: EffectiveCadastro
    turbine_limits: TurbineLimits
    operated_codes: set[int]
    min_outflow_by_code: dict[int, float]
    mp: pd.DataFrame | None
    fd: pd.DataFrame | None
    op_date: str
    evaporation_codes: set[int]
    travel_time_hours: dict[int, float] | None
    fpha_codes: set[int] | None


def _hydro_context(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    effective: EffectiveCadastro,
    turbine_limits: TurbineLimits,
    travel_time_hours: dict[int, float] | None,
    fpha_codes: set[int] | None,
) -> _HydroContext:
    """Read the operated plants' ``UH`` minimum outflows and evaporation flags
    (and, for Itaipu, the ``MP``/``FD`` registers), noting every
    coupling-only ``UH`` registration."""
    dadger = case.dadger
    hidr = case.hidr
    start_date = case.start_date
    operated = _operated_uh(dadger)
    coupling_only = _coupling_only_codes(dadger)
    if coupling_only:
        emit(_coupling_only_diagnostic(hidr, coupling_only), logger=_LOG)
    operated_codes = set(id_map.hydro_codes)
    min_outflow_by_code: dict[int, float] = {}
    for _, row in operated.iterrows():
        value = row.get("vazao_defluente_minima")
        if not pd.isna(value):
            min_outflow_by_code[int(row["codigo_usina"])] = float(value)

    # Read the MP/FD registers only if the split plant is actually present —
    # keeps every other (hand-built or real, non-Itaipu) caller from needing
    # a Dadger stub for methods this function otherwise never touches.
    mp = fd = None
    if _ITAIPU_CODE in operated_codes:
        mp = as_frame(dadger.mp(df=True))
        fd = as_frame(dadger.fd(df=True))

    return _HydroContext(
        hidr=hidr,
        effective=effective,
        turbine_limits=turbine_limits,
        operated_codes=operated_codes,
        min_outflow_by_code=min_outflow_by_code,
        mp=mp,
        fd=fd,
        op_date=start_date.isoformat(),
        evaporation_codes=_evaporation_flag_codes(dadger),
        travel_time_hours=travel_time_hours,
        fpha_codes=fpha_codes,
    )


def _unit_groups(
    code: int,
    hreg: pd.Series,
    name: str,
    context: _HydroContext,
    id_map: DecompIdMap,
) -> tuple[list[dict], float, float]:
    """The plant's unit groups, with its entity ``max_turbined_m3s`` and
    ``max_generation_mw``: Itaipu's two per-frequency groups, or one mirror
    group for every other plant."""
    bus_id = id_map.bus_id(int(hreg["submercado"]))
    if code == _ITAIPU_CODE:
        frequencies = _split_plant_frequencies(
            hreg, code, context.effective, context.mp, context.fd
        )
        # Unconditional 60 Hz -> IV relocation: Itaipu's 60 Hz
        # unit group (group id 1, the higher of the two ascending
        # frequencies) is moved to the transshipment bus -- the AC
        # corridor into Ivaiporã -- so novomodelo's HydroGeneration{bus}
        # selector can separate the two groups; the 50 Hz group (group
        # id 0) stays on the plant's own SE bus (already computed above
        # as `bus_id`), where its HVDC Elo delivers directly and the
        # `carga_ande` load nets in.
        se_bus = bus_id
        iv_bus = id_map.transhipment_bus_id
        return _build_split_unit_groups(
            hreg,
            code,
            name,
            [se_bus, iv_bus],
            frequencies,
            context.effective,
            context.turbine_limits,
        )
    _, max_generation = _rated_envelope(hreg, code, context.effective)
    max_turbined = context.turbine_limits.envelope(code, 0)
    unit_groups = [
        build_mirror_unit_group(
            name=name,
            bus_id=bus_id,
            min_generation_mw=0.0,
            max_generation_mw=max_generation,
            min_turbined_m3s=0.0,
            max_turbined_m3s=max_turbined,
        )
    ]
    return unit_groups, max_turbined, max_generation


def _hydraulic_losses(effective: EffectiveCadastro, code: int) -> dict:
    """The plant's penstock hydraulic losses, explicitly lossless when it
    declares none."""
    # Penstock hydraulic losses — a computed FPHA requires the field
    # (novomodelo rejects an FPHA plant without it). tipo_perda 1 = a % of
    # gross head (factor); 2 = constant metres; anything else / no loss
    # emits an explicit lossless factor so the required field is present.
    perdas = effective.value(code, "perdas", 0)
    tipo_perda = int(effective.base.loc[code, "tipo_perda"])
    if tipo_perda == 1 and perdas > 0.0:
        return {"type": "factor", "value": perdas / 100.0}
    if tipo_perda == 2 and perdas > 0.0:
        return {"type": "constant", "value_m": perdas}
    return {"type": "factor", "value": 0.0}


def _fpha_fields(code: int, effective: EffectiveCadastro) -> dict:
    """The computed-FPHA entry fields: specific productivity, turbine
    efficiency, the inline constant tailrace, and hydraulic losses."""
    rho_esp = effective.value(code, "produtibilidade_especifica", 0)
    fields: dict = {
        "specific_productivity_mw_per_m3s_per_m": rho_esp,
        "efficiency": {"type": "constant", "value": fpha_efficiency(rho_esp)},
    }
    # Inline constant tailrace = mean canal de fuga: novomodelo's FPHA
    # fallback for a plant whose tailrace_curves.parquet family is
    # absent (all this deck's FPHA plants do carry a family, so it is
    # only a safety net).
    cf = effective.value(code, "canal_fuga_medio", 0)
    if cf > 0.0:
        fields["tailrace"] = {"type": "polynomial", "coefficients": [cf]}
    fields["hydraulic_losses"] = _hydraulic_losses(effective, code)
    return fields


class _PlantEntry(NamedTuple):
    """A plant's ``hydros.json`` entry, whether ``storage_envelope``
    collapsed its run-of-river registry band to the reference volume, and the
    rows it adds to :func:`convert_hydros`'s capped-FPHA-efficiency and
    invalid-cascade-link reports."""

    entry: dict
    runofriver_collapsed: bool
    efficiency_row: list[object] | None
    invalid_link_row: list[object] | None


def _hydro_entry(code: int, context: _HydroContext, id_map: DecompIdMap) -> _PlantEntry:
    """Plant *code*'s ``hydros.json`` entry."""
    hidr = context.hidr
    effective = context.effective
    if code not in hidr.index:
        raise ValueError(f"UH plant {code} is not in the hydro registry")
    hreg = hidr.loc[code]
    name = str(hreg["nome_usina"]).strip()
    is_fpha = context.fpha_codes is not None and code in context.fpha_codes
    downstream, invalid_link = _cascade_walk(effective, code, context.operated_codes, 0)
    min_storage_hm3, max_storage_hm3 = storage_envelope(effective, code)
    runofriver_collapsed = str(hreg.get("tipo_regulacao", "")).strip() == "D" and float(
        hreg["volume_minimo"]
    ) != float(hreg["volume_maximo"])
    unit_groups, max_turbined, max_generation = _unit_groups(
        code, hreg, name, context, id_map
    )
    entry: dict = {
        "id": id_map.hydro_id(code),
        "name": name,
        "operational_start_date": context.op_date,
        "downstream_id": (None if downstream is None else id_map.hydro_id(downstream)),
        "reservoir": {
            "min_storage_hm3": min_storage_hm3,
            "max_storage_hm3": max_storage_hm3,
        },
        "outflow": {
            "min_outflow_m3s": context.min_outflow_by_code.get(code, 0.0),
            "max_outflow_m3s": None,
        },
        "generation": {
            "model": "fpha" if is_fpha else "constant_productivity",
            "min_turbined_m3s": 0.0,
            "max_turbined_m3s": max_turbined,
            "min_generation_mw": 0.0,
            "max_generation_mw": max_generation,
        },
        "unit_groups": unit_groups,
    }
    efficiency_row: list[object] | None = None
    if is_fpha:
        entry.update(_fpha_fields(code, effective))
        efficiency_row = fpha_efficiency_capped_row(
            code, name, entry["specific_productivity_mw_per_m3s_per_m"]
        )
    travel_time_hours = context.travel_time_hours
    if travel_time_hours and code in travel_time_hours and downstream is not None:
        entry["travel_time_hours"] = travel_time_hours[code]
    if code in context.evaporation_codes:
        evaporation_mm = _evaporation_coefficients_mm(hidr, code)
        if evaporation_mm is not None:
            entry["evaporation"] = {"coefficients_mm": evaporation_mm}
    invalid_link_row: list[object] | None = (
        None if invalid_link is None else [code, name, invalid_link]
    )
    return _PlantEntry(entry, runofriver_collapsed, efficiency_row, invalid_link_row)


def _report_machine_set_overrides(
    effective: EffectiveCadastro, operated_codes: set[int]
) -> None:
    """Log how many operated plants carry an ``AC`` machine-configuration
    override."""
    affected_codes = set(effective.machine_conjunto_counts) | {
        code for code, _ in effective.machine_sets
    }
    n_ac_affected = len(affected_codes & operated_codes)
    if n_ac_affected:
        _LOG.info(
            "plants with AC NUMCON/NUMMAQ/POTEFE/VAZEFE/ALTEFE "
            "machine-configuration overrides applied: %d",
            n_ac_affected,
        )


def _report_runofriver_collapse(n_runofriver_collapsed: int) -> None:
    """Log how many run-of-river storage ranges collapsed to a single volume."""
    if n_runofriver_collapsed:
        _LOG.info(
            "run-of-river ('D') plants whose storage range collapsed to their "
            "reference volume (volume_referencia): %d; the hidr registry's "
            "(volume_minimo, volume_maximo) band would otherwise emit a "
            "phantom weekly storage buffer the plant does not have",
            n_runofriver_collapsed,
        )


def convert_hydros(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    effective: EffectiveCadastro,
    turbine_limits: TurbineLimits,
    travel_time_hours: dict[int, float] | None = None,
    fpha_codes: set[int] | None = None,
) -> dict:
    """Build ``hydros.json`` for the operated plants.

    *fpha_codes* (from :func:`novomodelo_bridge.decomp.converters.fpha.fpha_eligible_codes`)
    selects the plants emitted with novomodelo's computed-FPHA generation model:
    their ``generation.model`` is ``"fpha"`` and they carry the turbine
    ``efficiency`` (η = ρ_esp / K), the ``specific_productivity_mw_per_m3s_per_m``
    novomodelo derives ρ_eq from, and an inline constant ``tailrace`` (the
    ``canal_fuga_medio`` fallback used when ``tailrace_curves.parquet`` carries
    no family for the plant). Plants absent from it — or the whole set being
    ``None`` — keep the constant-productivity model whose ρ_eq rides in
    ``hydro_energy_productivity.parquet``.

    *travel_time_hours* (``{plant code: hours}``, from
    :func:`novomodelo_bridge.decomp.converters.travel_time.convert_travel_time`) stamps the
    ``VI`` water travel time onto each arc plant's entry; a plant absent from it
    — or the whole map being ``None`` — emits no ``travel_time_hours`` key
    (novomodelo defaults it to instantaneous). The key is emitted only when the plant
    also has a downstream arc for the delay to act on.

    ``max_generation_mw`` is the installed (un-derated, ``TEIF``/``IP``-free)
    max-over-stages envelope of the AC-adjusted rated unit-power sum,
    sourced per stage from *effective*'s machine-set view
    (:func:`_rated_envelope`, mirroring :func:`storage_envelope`'s
    outer-bound construction for the reservoir block) — some plants' *true*
    in-service machine configuration differs from ``hidr.dat``'s nameplate
    conjunto sum, and/or changes mid-horizon. ``max_turbined_m3s`` is
    instead the max-over-stages envelope of the plant's turbined-flow limit
    (*turbine_limits*, from
    :func:`~novomodelo_bridge.decomp.converters.hydro.turbine_limit.compute_turbine_limits`).
    A stage whose effective capacity (installed power or turbined-flow limit)
    drops below its own envelope gets a sparse per-stage overlay instead
    (:func:`convert_hydro_group_availability`), never a change to the
    declared envelope itself. The production model is constant
    productivity, with the value emitted separately
    (:func:`convert_hydro_energy_productivity`).

    Every plant declares one mirror unit group, except the per-frequency
    split plant (Itaipu, code 66), which declares two conjunto-backed groups
    instead (:func:`_build_split_unit_groups`), whose summed
    ``max_turbined_m3s``/``max_generation_mw`` *are* the entity envelope by
    construction — novomodelo rule 41 holds even though the two groups' own
    per-stage machine-set changes (if any) need not peak on the same stage,
    nor their turbined-flow limits peak on the same stage as their rated
    power. Itaipu's 60 Hz group (group id 1) is placed on
    ``id_map.transhipment_bus_id`` — the ``IV`` bus modeling the 765 kV
    corridor into Ivaiporã — rather than the plant's own submercado bus; the
    50 Hz group (group id 0) stays on the plant's own SE bus, where the HVDC
    Elo delivers directly and the ``carga_ande`` load nets in
    (:func:`~novomodelo_bridge.decomp.pipeline._convert_scenarios`). This
    relocation is unconditional whenever Itaipu is operated. The entity
    ``reservoir`` block is the plant's
    outer per-stage storage envelope (:func:`storage_envelope`), so
    per-stage bound overrides
    (:func:`novomodelo_bridge.decomp.converters.bounds.convert_storage_bounds`)
    always sit inside it. Per-family ``AC`` coverage is reported by
    ``check decomp`` (:mod:`novomodelo_bridge.decomp.preflight`), not logged here.
    """
    context = _hydro_context(
        case,
        id_map,
        effective=effective,
        turbine_limits=turbine_limits,
        travel_time_hours=travel_time_hours,
        fpha_codes=fpha_codes,
    )
    hydros: list[dict] = []
    n_runofriver_collapsed = 0
    efficiency_rows: list[list[object]] = []
    no_curve_rows: list[list[object]] = []
    invalid_link_rows: list[list[object]] = []
    for code in id_map.hydro_codes:
        plant = _hydro_entry(code, context, id_map)
        hydros.append(plant.entry)
        if plant.runofriver_collapsed:
            n_runofriver_collapsed += 1
        if plant.efficiency_row is not None:
            efficiency_rows.append(plant.efficiency_row)
        if plant.invalid_link_row is not None:
            invalid_link_rows.append(plant.invalid_link_row)
        no_curve = [
            stage.index
            for stage in case.calendar
            if not has_forebay_curve(effective.cota_polynomial(code, stage.index))
        ]
        if no_curve:
            no_curve_rows.append(
                [code, plant.entry["name"], format_stage_ranges(no_curve)]
            )

    _report_machine_set_overrides(effective, context.operated_codes)
    _report_runofriver_collapse(n_runofriver_collapsed)
    if efficiency_rows:
        emit(fpha_efficiency_capped_diagnostic(efficiency_rows), logger=_LOG)
    if no_curve_rows:
        emit(forebay_curve_missing_diagnostic(no_curve_rows), logger=_LOG)
    if invalid_link_rows:
        emit(_cascade_link_invalid_diagnostic(invalid_link_rows), logger=_LOG)

    return {
        "$schema": novomodelo_schemas.schema_url_for("system/hydros.json"),
        "hydros": hydros,
    }


def _initial_volume_hm3(effective: EffectiveCadastro, code: int, pct: float) -> float:
    """The ``UH`` ``volume_inicial`` percentage resolved to hm³ at stage 0.

    ``pct`` is a percentage of the *initial stage's* effective useful volume,
    not the plant's outer envelope, so the range is read from
    :func:`~novomodelo_bridge.decomp.converters.cadastro.effective.
    effective_storage_range` at stage ``0``. A run-of-river (``D``) plant's
    stage-0 range is already the single-point collapse
    ``(vol_ref, vol_ref)``, so its initial
    value is ``vol_ref`` regardless of *pct*. Shared by
    :func:`convert_initial_storage` (the initial condition) and
    :func:`_operated_initial_volumes` (the generation-productivity anchor) so
    both resolve a plant's initial volume identically.
    """
    v_min, v_max = effective_storage_range(effective, code, 0)
    return min(max(v_min + (pct / 100.0) * (v_max - v_min), v_min), v_max)


def _operated_initial_volumes(
    case: DecompCase, *, effective: EffectiveCadastro
) -> dict[int, float]:
    """``{code: initial reservoir volume hm³}`` for every operated ``UH`` plant.

    The volume :func:`convert_energy_productivity` anchors each plant's
    generation productivity on — the same value :func:`convert_initial_storage`
    emits as the initial condition (both via :func:`_initial_volume_hm3`).
    """
    return {
        code: _initial_volume_hm3(effective, code, pct)
        for code, pct in initial_volume_percentages(case.dadger).items()
    }


def convert_initial_storage(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    effective: EffectiveCadastro,
) -> list[dict]:
    """Initial reservoir volumes from ``UH`` (% of useful → hm³).

    The percentage is resolved by :func:`_initial_volume_hm3` against the
    stage-0 effective useful volume (run-of-river plants collapse to their
    reference volume).
    """
    operated = _operated_uh(case.dadger)
    storage: list[dict] = []
    filling: list[list[object]] = []
    for _, row in operated.iterrows():
        code = int(row["codigo_usina"])
        value = _initial_volume_hm3(effective, code, float(row["volume_inicial"]))
        storage.append({"hydro_id": id_map.hydro_id(code), "value_hm3": value})
        dead = row.get("volume_morto_inicial")
        if not pd.isna(dead):
            filling.append([code, _registry_name(case.hidr, code), float(dead)])
    storage.sort(key=lambda e: e["hydro_id"])
    if filling:
        emit(_dead_volume_filling_diagnostic(filling), logger=_LOG)
    return storage


def _dead_volume_filling_diagnostic(rows: list[list[object]]) -> Diagnostic:
    return Diagnostic(
        code="decomp-dead-volume-filling-unconverted",
        severity=Severity.WARNING,
        category="Initial conditions",
        title=f"Dead-volume filling not converted ({len(rows)})",
        summary=(
            "The listed plants declare an initial dead volume in UH, which "
            "DECOMP fills before the plant generates. The conversion does not "
            "model the filling and starts each plant with its dead volume "
            "full, so its storage and generation in the first stages differ "
            "from DECOMP's."
        ),
        table=DiagnosticTable(
            columns=["Code", "Name", "Initial dead volume (hm³)"],
            rows=rows,
            justify=["right", "left", "right"],
        ),
    )
