"""Hydro entity-record construction: the ``hydros.json`` per-plant builder
``convert_hydros`` and initial-storage conversion, the registry reader, and
the operated-set / cascade-walk helpers they build on.

Imports the capacity-cap helpers from ``.bounds`` and the
storage-envelope helpers from ``decomp.converters.cadastro``; nothing in the package
imports from here -- this is the DAG's top seam.

The cascade walk (:func:`_downstream_operated`, shared by this module's own
``downstream_id`` entity field and ``scenarios.py``'s incremental-inflow
attribution) and ``scenarios.py``'s inflow-gauge attribution both read
:class:`~cobre_bridge.decomp.converters.cadastro.effective.EffectiveCadastro`'s
``downstream_plant``/``inflow_gauge`` accessors -- the post-``AC
NUMJUS``/``NUMPOS`` link/gauge -- rather than the base ``hidr`` columns
directly. ``AC JUSENA`` (a downstream-energy coupling, not a water-routing
link) and ``AC NPOSNW`` (the other source family's own inflow gauge) are
deliberately not ingested -- no DECOMP consumer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
from idecomp.decomp.modelos.dadger import ACALTEFE
from inewave.newave import Hidr

from cobre_bridge.cobre import schemas as cobre_schemas
from cobre_bridge.core.hydro_units import build_mirror_unit_group
from cobre_bridge.core.productivity import fpha_efficiency
from cobre_bridge.decomp.converters.cadastro import (
    effective_storage_range,
    storage_envelope,
)
from cobre_bridge.decomp.converters.hydro.bounds import (
    _ITAIPU_CODE,
    _build_split_unit_groups,
    _head_corrected_envelope,
    _rated_envelope,
    _split_plant_frequencies,
)

if TYPE_CHECKING:
    from pathlib import Path

    from idecomp.decomp import Dadger

    from cobre_bridge.decomp.case import DecompCase
    from cobre_bridge.decomp.converters.cadastro import EffectiveCadastro
    from cobre_bridge.decomp.id_map import DecompIdMap

_LOG = logging.getLogger(__name__)


def read_hidr(path: Path) -> pd.DataFrame:
    """Read the hydro registry file, indexed by plant code."""
    df = Hidr.read(str(path)).cadastro
    if df is None or df.empty:
        raise FileNotFoundError(f"{path} has no registry data")
    return df


def _operated_uh(dadger: Dadger) -> pd.DataFrame:
    """The ``UH`` rows carrying an initial volume (the operated plants).

    Rows without one (coupling-only registrations) are excluded — their
    terminal-value treatment is the boundary importer's D3 territory, not the
    registry's. :func:`convert_hydros` reports the excluded set (by name) once.
    """
    uh = dadger.uh(df=True)
    if uh is None or uh.empty:
        raise ValueError("the deck has no UH records; cannot convert hydros")
    return uh[uh["volume_inicial"].notna()]


def _coupling_only_codes(dadger: Dadger) -> list[int]:
    """Operated-set-excluded ``UH`` codes — registrations with no initial volume."""
    uh = dadger.uh(df=True)
    if uh is None or uh.empty:
        return []
    excluded = uh[uh["volume_inicial"].isna()]
    return sorted(int(c) for c in excluded["codigo_usina"])


def _hydro_display_name(hidr: pd.DataFrame, code: int) -> str:
    """``'NAME (code)'`` for a plant, or just the code if the registry lacks it."""
    try:
        name = str(hidr.loc[code, "nome_usina"]).strip()
    except KeyError:
        return str(code)
    return f"{name} ({code})"


def _downstream_operated(
    effective: EffectiveCadastro,
    code: int,
    operated: set[int],
    *,
    stage_index: int = 0,
) -> int | None:
    """Walk the *effective* (post-``AC NUMJUS``) cascade to the next
    *operated* plant downstream, at *stage_index* (default the initial
    stage — the single stage-representative topology every caller uses).

    Non-operated intermediates are skipped through (their routing is
    instantaneous absence — the water continues down the declared chain);
    code 0 is the sink. If *code*'s own effective downstream link varies
    across stages (a temporal ``AC NUMJUS``), a tracked-gap warning is
    logged and the *stage_index* link is used for the whole horizon
    regardless — per-stage cascade topology is not modeled
    (:meth:`~cobre_bridge.decomp.converters.cadastro.effective.
    EffectiveCadastro.downstream_plant_varies`). A temporal override on a
    non-operated intermediate encountered mid-walk is not separately checked
    here (no rv3 row exercises it — see the module docstring); the
    pipeline's own relink diagnostic
    (:func:`cobre_bridge.decomp.pipeline._topology_relink_diagnostic`) still
    surfaces a resulting cascade change via a fallback warning rather than
    dropping it silently.
    """
    if effective.downstream_plant_varies(code):
        _LOG.warning(
            "plant %d's downstream link (AC NUMJUS) varies across stages; "
            "per-stage cascade topology is not modeled (deferred fidelity) "
            "-- using the stage-%d effective link for the whole horizon",
            code,
            stage_index,
        )
    visited = {code}
    current = effective.downstream_plant(code, stage_index)
    while current != 0 and current not in operated:
        if current in visited or current not in effective.base.index:
            _LOG.warning(
                "cascade walk from plant %d hit an invalid link at %d; "
                "treating as a sink",
                code,
                current,
            )
            return None
        visited.add(current)
        current = effective.downstream_plant(current, stage_index)
    return current if current != 0 else None


#: ``hidr.dat``'s 12 monthly reservoir-evaporation columns in calendar order
#: (Jan..Dec) — cobre's ``evaporation.coefficients_mm`` is index 0 = January.
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


def _evaporation_flag_codes(dadger: Dadger) -> set[int]:
    """Plant codes whose ``UH`` register sets the reservoir-evaporation flag.

    DECOMP models a plant's evaporation only when its ``UH`` ``evaporacao``
    flag is set; a plant absent from this set is emitted with no evaporation
    regardless of the coefficients ``hidr.dat`` may carry.
    """
    uh = dadger.uh(df=True)
    if uh is None or "evaporacao" not in uh.columns or "codigo_usina" not in uh.columns:
        return set()
    return set(uh.loc[uh["evaporacao"].fillna(0) != 0, "codigo_usina"].astype(int))


def _evaporation_coefficients_mm(hidr: pd.DataFrame, code: int) -> list[float] | None:
    """The plant's 12 monthly evaporation coefficients [mm/month], Jan..Dec,
    from ``hidr.dat`` — or ``None`` when the plant is absent or every month is
    zero (nothing to model).

    cobre (>= 0.14) computes the evaporation-outflow model internally from
    these monthly rates and the reservoir area geometry
    (``hydro_geometry.parquet``, emitted by the FPHA path), scaling each stage
    to its ``stage_hours / month_hours`` share of the calendar month — the C11
    fix (pre-0.14 deposited a whole month's evaporation on every stage). The
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


def convert_hydros(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    effective: EffectiveCadastro,
    travel_time_hours: dict[int, float] | None = None,
    fpha_codes: set[int] | None = None,
) -> dict:
    """Build ``hydros.json`` for the operated plants.

    *fpha_codes* (from :func:`cobre_bridge.decomp.converters.fpha.fpha_eligible_codes`)
    selects the plants emitted with cobre's computed-FPHA generation model:
    their ``generation.model`` is ``"fpha"`` and they carry the turbine
    ``efficiency`` (η = ρ_esp / K), the ``specific_productivity_mw_per_m3s_per_m``
    cobre derives ρ_eq from, and an inline constant ``tailrace`` (the
    ``canal_fuga_medio`` fallback used when ``tailrace_curves.parquet`` carries
    no family for the plant). Plants absent from it — or the whole set being
    ``None`` — keep the constant-productivity model whose ρ_eq rides in
    ``hydro_energy_productivity.parquet``.

    *travel_time_hours* (``{plant code: hours}``, from
    :func:`cobre_bridge.decomp.converters.travel_time.convert_travel_time`) stamps the
    ``VI`` water travel time onto each arc plant's entry; a plant absent from it
    — or the whole map being ``None`` — emits no ``travel_time_hours`` key
    (cobre defaults it to instantaneous). The key is emitted only when the plant
    also has a downstream arc for the delay to act on.

    ``max_generation_mw`` is the installed (un-derated, ``TEIF``/``IP``-free)
    max-over-stages envelope of the AC-adjusted rated unit-power sum,
    sourced per stage from *effective*'s machine-set view
    (:func:`_rated_envelope`, mirroring :func:`storage_envelope`'s
    outer-bound construction for the reservoir block) — some plants' *true*
    in-service machine configuration differs from ``hidr.dat``'s nameplate
    conjunto sum, and/or changes mid-horizon. ``max_turbined_m3s`` is
    instead the max-over-stages envelope of the **head-corrected** engolimento
    (:func:`_head_corrected_envelope` — see the module docstring
    for the validated formula and accuracy). A stage whose effective
    capacity (installed power or head-corrected flow) drops below its own
    envelope gets a sparse per-stage overlay instead
    (:func:`convert_hydro_group_availability`), never a change to the
    declared envelope itself. The production model is constant
    productivity, with the value emitted separately
    (:func:`convert_energy_productivity`).

    **``ACALTEFE`` tracked gap:** a plant declaring an ``AC ALTEFE``
    effective-head override is logged (not consumed — see the
    ``TRACKED COBRE-GAP WORKAROUND`` below) and conversion proceeds with the
    base ``hidr`` nominal head.

    Every plant declares one mirror unit group, except the per-frequency
    split plant (Itaipu, code 66), which declares two conjunto-backed groups
    instead (:func:`_build_split_unit_groups`), whose summed
    ``max_turbined_m3s``/``max_generation_mw`` *are* the entity envelope by
    construction — cobre rule 41 holds even though the two groups' own
    per-stage machine-set changes (if any) need not peak on the same stage,
    nor their head-corrected flows peak on the same stage as their rated
    power. Itaipu's 60 Hz group (group id 1) is placed on
    ``id_map.transhipment_bus_id`` — the ``IV`` bus modeling the 765 kV
    corridor into Ivaiporã — rather than the plant's own submercado bus; the
    50 Hz group (group id 0) stays on the plant's own SE bus, where the HVDC
    Elo delivers directly and the ``carga_ande`` load nets in
    (:func:`~cobre_bridge.decomp.pipeline._convert_scenarios`). This
    relocation is unconditional whenever Itaipu is operated. The entity
    ``reservoir`` block is the plant's
    outer per-stage storage envelope (:func:`storage_envelope`), so
    per-stage bound overrides
    (:func:`cobre_bridge.decomp.converters.bounds.convert_storage_bounds`)
    always sit inside it. Per-family ``AC`` coverage is reported by
    ``check decomp`` (:mod:`cobre_bridge.decomp.preflight`), not logged here.
    """
    dadger = case.dadger
    hidr = case.hidr
    start_date = case.start_date
    operated = _operated_uh(dadger)
    coupling_only = _coupling_only_codes(dadger)
    if coupling_only:
        _LOG.warning(
            "%d UH registration(s) without an initial volume excluded from "
            "the operated set (coupling-only): %s",
            len(coupling_only),
            ", ".join(_hydro_display_name(hidr, c) for c in coupling_only),
        )
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
        mp = dadger.mp(df=True)
        fd = dadger.fd(df=True)

    op_date = start_date.isoformat()
    hydros: list[dict] = []
    evaporation_codes = _evaporation_flag_codes(dadger)
    n_runofriver_collapsed = 0
    for code in id_map.hydro_codes:
        if code not in hidr.index:
            raise ValueError(f"UH plant {code} is not in the hydro registry")
        hreg = hidr.loc[code]
        name = str(hreg["nome_usina"]).strip()
        is_fpha = fpha_codes is not None and code in fpha_codes
        downstream = _downstream_operated(effective, code, operated_codes)
        min_storage_hm3, max_storage_hm3 = storage_envelope(effective, code)
        if str(hreg.get("tipo_regulacao", "")).strip() == "D" and float(
            hreg["volume_minimo"]
        ) != float(hreg["volume_maximo"]):
            n_runofriver_collapsed += 1
        bus_id = id_map.bus_id(int(hreg["submercado"]))
        if code == _ITAIPU_CODE:
            frequencies = _split_plant_frequencies(hreg, code, effective, mp, fd)
            # Unconditional 60 Hz -> IV relocation: Itaipu's 60 Hz
            # unit group (group id 1, the higher of the two ascending
            # frequencies) is moved to the transshipment bus -- the AC
            # corridor into Ivaiporã -- so cobre's HydroGeneration{bus}
            # selector can separate the two groups; the 50 Hz group (group
            # id 0) stays on the plant's own SE bus (already computed above
            # as `bus_id`), where its HVDC Elo delivers directly and the
            # `carga_ande` load nets in.
            se_bus = bus_id
            iv_bus = id_map.transhipment_bus_id
            unit_groups, max_turbined, max_generation = _build_split_unit_groups(
                hreg, code, name, [se_bus, iv_bus], frequencies, effective
            )
        else:
            _, max_generation = _rated_envelope(hreg, code, effective)
            max_turbined = _head_corrected_envelope(hreg, code, effective)
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
        entry: dict = {
            "id": id_map.hydro_id(code),
            "name": name,
            "operational_start_date": op_date,
            "downstream_id": (
                None if downstream is None else id_map.hydro_id(downstream)
            ),
            "reservoir": {
                "min_storage_hm3": min_storage_hm3,
                "max_storage_hm3": max_storage_hm3,
            },
            "outflow": {
                "min_outflow_m3s": min_outflow_by_code.get(code, 0.0),
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
        if is_fpha:
            rho_esp = effective.value(code, "produtibilidade_especifica", 0)
            entry["specific_productivity_mw_per_m3s_per_m"] = rho_esp
            entry["efficiency"] = {
                "type": "constant",
                "value": fpha_efficiency(rho_esp, name),
            }
            # Inline constant tailrace = mean canal de fuga: cobre's FPHA
            # fallback for a plant whose tailrace_curves.parquet family is
            # absent (all this deck's FPHA plants do carry a family, so it is
            # only a safety net).
            cf = effective.value(code, "canal_fuga_medio", 0)
            if cf > 0.0:
                entry["tailrace"] = {"type": "polynomial", "coefficients": [cf]}
            # Penstock hydraulic losses — a computed FPHA requires the field
            # (cobre rejects an FPHA plant without it). tipo_perda 1 = a % of
            # gross head (factor); 2 = constant metres; anything else / no loss
            # emits an explicit lossless factor so the required field is present.
            perdas = effective.value(code, "perdas", 0)
            tipo_perda = int(effective.base.loc[code, "tipo_perda"])
            if tipo_perda == 1 and perdas > 0.0:
                entry["hydraulic_losses"] = {"type": "factor", "value": perdas / 100.0}
            elif tipo_perda == 2 and perdas > 0.0:
                entry["hydraulic_losses"] = {"type": "constant", "value_m": perdas}
            else:
                entry["hydraulic_losses"] = {"type": "factor", "value": 0.0}
        if travel_time_hours and code in travel_time_hours and downstream is not None:
            entry["travel_time_hours"] = travel_time_hours[code]
        if code in evaporation_codes:
            evaporation_mm = _evaporation_coefficients_mm(hidr, code)
            if evaporation_mm is not None:
                entry["evaporation"] = {"coefficients_mm": evaporation_mm}
        hydros.append(entry)

    affected_codes = set(effective.machine_conjunto_counts) | {
        code for code, _ in effective.machine_sets
    }
    n_ac_affected = len(affected_codes & operated_codes)
    if n_ac_affected:
        _LOG.info(
            "applied AC NUMCON/NUMMAQ/POTEFE/VAZEFE machine-configuration "
            "overrides for %d plant(s)",
            n_ac_affected,
        )

    if n_runofriver_collapsed:
        _LOG.info(
            "collapsed %d run-of-river ('D') plant(s) storage range to their "
            "reference volume (volume_referencia): the hidr registry's "
            "(volume_minimo, volume_maximo) band would otherwise emit a "
            "phantom weekly storage buffer the plant does not have",
            n_runofriver_collapsed,
        )

    # TRACKED COBRE-GAP WORKAROUND (idecomp limitation, not cobre's): the
    # source model's AC ALTEFE register overrides a conjunto's effective
    # head, but idecomp's typed accessor exposes only the identifying/timing
    # columns for it, no value column — so it cannot be folded into the
    # head-corrected max_turbined above. Every plant keeps its base hidr
    # queda_nominal_conjunto_* nominal head regardless of an ALTEFE
    # declaration. Remove this warning (and wire the override into
    # _conjunto_h_nom) once idecomp exposes an ALTEFE value accessor.
    altefe = dadger.ac(codigo_usina=None, modificacao=ACALTEFE, df=True)
    if isinstance(altefe, pd.DataFrame) and not altefe.empty:
        n_altefe = altefe["codigo_usina"].nunique()
        _LOG.warning(
            "%d plant(s) declare an AC ALTEFE effective-head override that "
            "cannot be consumed (idecomp exposes no value accessor for "
            "ALTEFE); head-corrected max_turbined uses the base hidr "
            "nominal head for these plants",
            n_altefe,
        )

    return {
        "$schema": cobre_schemas.schema_url_for("system/hydros.json"),
        "hydros": hydros,
    }


def _initial_volume_hm3(effective: EffectiveCadastro, code: int, pct: float) -> float:
    """The ``UH`` ``volume_inicial`` percentage resolved to hm³ at stage 0.

    ``pct`` is a percentage of the *initial stage's* effective useful volume,
    not the plant's outer envelope, so the range is read from
    :func:`~cobre_bridge.decomp.converters.cadastro.effective.
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
        int(row["codigo_usina"]): _initial_volume_hm3(
            effective, int(row["codigo_usina"]), float(row["volume_inicial"])
        )
        for _, row in _operated_uh(case.dadger).iterrows()
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
    for _, row in operated.iterrows():
        code = int(row["codigo_usina"])
        value = _initial_volume_hm3(effective, code, float(row["volume_inicial"]))
        storage.append({"hydro_id": id_map.hydro_id(code), "value_hm3": value})
        dead = row.get("volume_morto_inicial")
        if not pd.isna(dead):
            _LOG.warning(
                "plant %d declares an initial dead volume (%s); "
                "dead-volume filling is not converted yet",
                code,
                dead,
            )
    storage.sort(key=lambda e: e["hydro_id"])
    return storage
