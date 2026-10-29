"""DECOMP hydro capacity caps: AC-adjusted rated capability, the
per-frequency split plant (Itaipu) unit groups, and the per-group per-stage
availability overlay.

Imports the equivalent-productivity helper from ``.productivity``; imports
nothing from ``.entity`` or ``.turbine_limit`` at runtime -- kept acyclic so
both can import from here, never the reverse.

``max_generation_mw`` is the rated envelope (:func:`_rated_envelope`) with the
available power (installed × ``MP`` × ``FD``) as its per-stage overlay;
``max_turbined_m3s`` is the envelope of the source model's per-stage turbined
limit (:class:`~novomodelo_bridge.decomp.converters.hydro.turbine_limit.
TurbineLimits`), with each stage's limit as its overlay.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from novomodelo_bridge.core.hydro_units import build_mirror_unit_group
from novomodelo_bridge.core.pandas_utils import as_frame
from novomodelo_bridge.core.tolerances import relative_tolerance
from novomodelo_bridge.decomp.group_bounds import GroupBoundEntry

if TYPE_CHECKING:
    from collections.abc import Sequence

    from novomodelo_bridge.decomp.case import DecompCase
    from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro
    from novomodelo_bridge.decomp.converters.hydro.turbine_limit import TurbineLimits
    from novomodelo_bridge.decomp.id_map import DecompIdMap


#: Itaipu's plant code — the one plant whose ``MP``/``FD`` maintenance and
#: availability registers are declared per generating-unit group (one row
#: per ``frequencia``, 50/60 Hz) rather than per plant. Both
#: :func:`convert_hydros` (two conjunto-backed unit groups instead of one
#: mirror group) and :func:`convert_hydro_group_availability` (a per-group,
#: not per-plant, availability overlay) special-case it; every other plant
#: takes the ordinary single-group path.
_ITAIPU_CODE = 66

#: Itaipu's unit-group id per frequency half (Hz). Groups follow the ascending
#: ``MP``/``FD`` frequencies (:func:`_split_plant_frequencies`), so 50 Hz is
#: group 0; the ``RI`` floors and the frequency-tagged ``RE`` bounds both key
#: on this map.
ITAIPU_GROUP_BY_FREQUENCY_HZ: dict[float, int] = {50.0: 0, 60.0: 1}


def split_plant_group_id(code: int, frequency_hz: float) -> int:
    """The unit-group id of plant *code*'s *frequency_hz* half.

    Raises
    ------
    ValueError
        When *code* is not Itaipu or *frequency_hz* is not one of its halves:
        Itaipu is the only plant split into per-frequency unit groups, so a
        frequency anywhere else names no group to bound.
    """
    group_id = (
        ITAIPU_GROUP_BY_FREQUENCY_HZ.get(frequency_hz) if code == _ITAIPU_CODE else None
    )
    if group_id is None:
        raise ValueError(
            f"plant {code}: a {frequency_hz:g} Hz generation term names no "
            f"frequency half; only plant {_ITAIPU_CODE} (Itaipu) is split "
            "into 50 Hz and 60 Hz unit groups"
        )
    return group_id


def _conjunto_machines(
    hreg: pd.Series,
    code: int,
    conjunto_index: int,
    effective: EffectiveCadastro,
    stage_index: int,
) -> tuple[int, float, float]:
    """``(units, rated flow m³/s, rated power MW)`` per unit of one 1-based
    *conjunto_index* at *stage_index*, after ``AC NUMMAQ``/``VAZEFE``/
    ``POTEFE``.

    Reads ``effective.machine_set(code, conjunto_index, stage_index)``: when
    present, its three fields already reflect
    :func:`~novomodelo_bridge.decomp.converters.cadastro.effective.
    build_effective_cadastro`'s independent per-field densification (a field
    with no override of its own is the ``hidr`` base, forward-filled flat);
    ``None`` means the pair carries no override at all, and every field falls
    back to the ``hidr.dat`` row directly.
    """
    machine_set = effective.machine_set(code, conjunto_index, stage_index)
    if machine_set is None:
        return (
            int(hreg[f"maquinas_conjunto_{conjunto_index}"]),
            float(hreg[f"vazao_nominal_conjunto_{conjunto_index}"]),
            float(hreg[f"potencia_nominal_conjunto_{conjunto_index}"]),
        )
    return machine_set.numero_maquinas, machine_set.vazao, machine_set.potencia


def _conjunto_count(
    hreg: pd.Series, code: int, effective: EffectiveCadastro, stage_index: int
) -> int:
    """The conjuntos effective at *stage_index* (``AC NUMCON``, else ``hidr``)."""
    n_sets = effective.machine_conjunto_count(code, stage_index)
    return int(hreg["numero_conjuntos_maquinas"]) if n_sets is None else n_sets


def _conjunto_rated_ac_adjusted(
    hreg: pd.Series,
    code: int,
    conjunto_index: int,
    effective: EffectiveCadastro,
    stage_index: int,
) -> tuple[float, float]:
    """Return ``(q, p)`` — the AC-adjusted rated flow/power of one 1-based
    ``conjunto_index`` at *stage_index* — the per-conjunto building block
    :func:`_compute_max_turbined_rated_ac_adjusted` sums over every
    conjunto, and :func:`_conjunto_rated_envelope`/
    :func:`convert_hydro_group_availability` call directly for a single
    conjunto to get one per-frequency group's own bounds.
    """
    n_machines, q_nom, p_nom = _conjunto_machines(
        hreg, code, conjunto_index, effective, stage_index
    )
    return q_nom * n_machines, p_nom * n_machines


def _compute_max_turbined_rated_ac_adjusted(
    hreg: pd.Series,
    code: int,
    effective: EffectiveCadastro,
    stage_index: int,
) -> tuple[float, float]:
    """Return ``(max_turbined, max_generation)`` — the AC-adjusted rated
    nameplate capacity at *stage_index* — the DECOMP-only counterpart to the
    shared, un-derated, stage-invariant
    ``core.hydro_units.rated_capacity``.

    Sums :func:`_conjunto_rated_ac_adjusted` over every conjunto effective
    at *stage_index* (``effective.machine_conjunto_count(code,
    stage_index)``, falling back to the ``hidr`` base
    ``numero_conjuntos_maquinas`` when that returns ``None``). Identical to
    the shared helper for a plant with no machine-set override at all —
    every lookup misses and falls back at every stage — so this changes
    nothing for the majority; it exists only because the override is
    per-conjunto, per-stage, and the shared helper exposes just the
    pre-summed, stage-invariant total. The shared helper itself is
    untouched. Callers needing the plant's *declared* (max-over-stages)
    capacity use :func:`_rated_envelope` instead of calling this per stage
    directly.
    """
    max_turbined = 0.0
    max_generation = 0.0
    for i in range(1, _conjunto_count(hreg, code, effective, stage_index) + 1):
        q, p = _conjunto_rated_ac_adjusted(hreg, code, i, effective, stage_index)
        max_turbined += q
        max_generation += p
    return max_turbined, max_generation


def _rated_envelope(
    hreg: pd.Series, code: int, effective: EffectiveCadastro
) -> tuple[float, float]:
    """Max-over-stages AC-adjusted rated ``(max_turbined, max_generation)``
    for the whole plant.

    The entity ``generation``/mirror-``unit_groups`` envelope for every
    non-split plant (:func:`convert_hydros`) and the per-plant comparison
    base for the per-group availability overlay
    (:func:`convert_hydro_group_availability`) — mirrors
    :func:`~novomodelo_bridge.decomp.converters.cadastro.effective.
    storage_envelope`'s outer-bound construction: the widest each of
    :func:`_compute_max_turbined_rated_ac_adjusted`'s two components ever
    reaches over the horizon, taken independently (the two maxima need not
    land on the same stage). A
    constant machine set collapses this to the stage-0 value, matching the
    date-blind value exactly. Not used for the split plant
    (Itaipu): its entity envelope is instead the *sum* of the two groups'
    own :func:`_conjunto_rated_envelope`, computed in
    :func:`convert_hydros`, so novomodelo rule 41 holds by construction rather
    than by coincidence of the two groups peaking on the same stage.
    """
    per_stage = [
        _compute_max_turbined_rated_ac_adjusted(hreg, code, effective, stage)
        for stage in range(effective.n_stages)
    ]
    return (
        max(turbined for turbined, _ in per_stage),
        max(generation for _, generation in per_stage),
    )


def _conjunto_rated_envelope(
    hreg: pd.Series, code: int, conjunto_index: int, effective: EffectiveCadastro
) -> tuple[float, float]:
    """Max-over-stages AC-adjusted rated ``(q, p)`` for one conjunto.

    The split plant's own per-group declared envelope
    (:func:`_build_split_unit_groups`) and the per-group comparison base
    for its own availability overlay
    (:func:`convert_hydro_group_availability`) — the per-conjunto mirror of
    :func:`_rated_envelope`.
    """
    per_stage = [
        _conjunto_rated_ac_adjusted(hreg, code, conjunto_index, effective, stage)
        for stage in range(effective.n_stages)
    ]
    return (
        max(q for q, _ in per_stage),
        max(p for _, p in per_stage),
    )


def _split_plant_frequencies(
    hreg: pd.Series,
    code: int,
    effective: EffectiveCadastro,
    mp: pd.DataFrame | None,
    fd: pd.DataFrame | None,
) -> list[float]:
    """Validate and return the split plant's ``frequencia`` values, sorted
    ascending — group id ``i`` (``i`` in ``0..len-1``) is then this list's
    ``i``-th entry, backed by hidr conjunto ``i + 1``
    (:func:`_build_split_unit_groups`,
    :func:`convert_hydro_group_availability`). The convention is
    oracle-invisible for a plant whose per-conjunto capacities are all
    identical (Itaipu's two conjuntos both being 7000 MW / 6620 m^3/s), so it
    is pinned for determinism, not derived from any frequency-to-conjunto
    label in the registry (``hidr.dat`` carries none).

    The conjunto-count/frequency-count agreement check below deliberately
    uses the plant's **stage-0** effective conjunto count
    (``effective.machine_conjunto_count(code, 0)``), never a per-stage
    value: no reference deck exercises a mid-horizon ``NUMCON`` change on
    the split plant specifically, and the ``MP``/``FD`` frequency rows
    themselves carry no per-stage cardinality to validate against. Tracked
    non-goal, not a silent branch (Itaipu carries no ``NUMCON`` override on
    rv3, so this is base-count = 2 there regardless).

    A register row without ``frequencia`` applies to the whole plant, so to
    every group (manual §3.4.6.13: the frequency field is optional).

    Raises
    ------
    ValueError
        Naming *code*, if the ``MP`` and ``FD`` registers disagree on the
        set of declared frequencies, either has no row for the plant,
        neither declares a frequency, or one mixes a plant-wide row with
        per-frequency rows — no silent single-group fallback for a
        mismatched registry — or if the stage-0 AC-adjusted conjunto count
        does not equal the number of declared frequencies.
    """

    mp_freqs, mp_plant_wide = _register_frequencies(mp, code)
    fd_freqs, fd_plant_wide = _register_frequencies(fd, code)
    if (mp_freqs and mp_plant_wide) or (fd_freqs and fd_plant_wide):
        raise ValueError(
            f"plant {code}: an MP or FD register mixes a row without "
            "frequencia with per-frequency rows"
        )
    if (
        not (mp_freqs or mp_plant_wide)
        or not (fd_freqs or fd_plant_wide)
        or not (mp_freqs or fd_freqs)
        or (mp_freqs and fd_freqs and mp_freqs != fd_freqs)
    ):
        raise ValueError(
            f"plant {code}: MP frequencia rows {sorted(mp_freqs)} and FD "
            f"frequencia rows {sorted(fd_freqs)} must agree, and at least "
            "one register must declare them, for the per-frequency split plant"
        )
    frequencies = sorted(mp_freqs or fd_freqs)

    n_sets = _conjunto_count(hreg, code, effective, 0)
    if n_sets != len(frequencies):
        raise ValueError(
            f"plant {code}: numero_conjuntos_maquinas ({n_sets}) does not "
            f"match the number of MP/FD frequencia rows ({len(frequencies)})"
        )
    return frequencies


def _register_frequencies(
    table: pd.DataFrame | None, code: int
) -> tuple[set[float], bool]:
    """Plant *code*'s declared ``frequencia`` values in an ``MP``/``FD``
    register, and whether it also has a row without one."""
    if table is None or table.empty:
        return set(), False
    column = table.loc[table["codigo_usina"] == code, "frequencia"]
    return {float(f) for f in column.dropna()}, bool(column.isna().any())


def _build_split_unit_groups(
    hreg: pd.Series,
    code: int,
    name: str,
    group_bus_ids: Sequence[int],
    frequencies: list[float],
    effective: EffectiveCadastro,
    turbine_limits: TurbineLimits,
) -> tuple[list[dict[str, object]], float, float]:
    """Conjunto-backed unit groups for the per-frequency split plant.

    Group id ``i`` (``i`` in ``0..len(frequencies)-1``, ascending frequency,
    so id 0 = the lowest — 50 Hz for Itaipu) is backed by hidr conjunto
    ``i + 1``: its ``max_generation_mw`` is *that* conjunto's own
    max-over-stages AC-adjusted rated envelope
    (:func:`_conjunto_rated_envelope`), and its ``max_turbined_m3s`` the
    envelope of its own turbined limit (*turbine_limits*). Group ``i`` sits on
    ``group_bus_ids[i]`` — a per-group bus, not necessarily the plant's own
    bus: :func:`convert_hydros` relocates Itaipu's 60 Hz group to the ``IV``
    transshipment bus while keeping its 50 Hz group on the plant's own bus.

    Raises
    ------
    ValueError
        Naming both counts, if *frequencies* and *group_bus_ids* disagree in
        length — a mis-wired split must fail loud, not silently colocate.

    Returns the groups plus their summed ``(max_turbined_m3s,
    max_generation_mw)``: :func:`convert_hydros` declares the plant's own
    entity envelope as exactly this sum, rather than recomputing it
    independently, so novomodelo rule 41 holds by construction even though the
    two groups need not peak on the same stage.
    """
    if len(frequencies) != len(group_bus_ids):
        raise ValueError(
            f"plant {code}: {len(frequencies)} split frequencies but "
            f"{len(group_bus_ids)} per-group buses were supplied"
        )
    groups: list[dict[str, object]] = []
    total_turbined = 0.0
    total_generation = 0.0
    for i in range(len(frequencies)):
        _, p = _conjunto_rated_envelope(hreg, code, i + 1, effective)
        q = turbine_limits.envelope(code, i)
        groups.append(
            build_mirror_unit_group(
                name=name,
                bus_id=group_bus_ids[i],
                min_generation_mw=0.0,
                max_generation_mw=p,
                min_turbined_m3s=0.0,
                max_turbined_m3s=q,
                group_id=i,
            )
        )
        total_turbined += q
        total_generation += p
    return groups, total_turbined, total_generation


def _frequency_row(
    table: pd.DataFrame | None, code: int, frequency: float
) -> pd.Series:
    """The single ``(codigo_usina, frequencia)``-matched register row, or the
    plant's row without ``frequencia`` when the register declares none.

    Unlike the ordinary single-group case (a missing register defaults its
    factor to ``1.0``), the split plant's two groups each require exactly
    one ``MP`` row and one ``FD`` row applying to their own frequency —
    validated by :func:`_split_plant_frequencies` before this is called, so a
    missing or duplicated row here means the registers disagree with that
    validation (defensive; should not happen in practice).
    """
    if table is None:
        raise ValueError(f"plant {code}: no register table for frequencia {frequency}")
    plant_rows = table.loc[table["codigo_usina"] == code]
    rows = plant_rows.loc[plant_rows["frequencia"] == frequency]
    if rows.empty:
        rows = plant_rows.loc[plant_rows["frequencia"].isna()]
    if len(rows) != 1:
        raise ValueError(
            f"plant {code}: expected exactly one register row at frequencia "
            f"{frequency}, found {len(rows)}"
        )
    return rows.iloc[0]


def _below_envelope(value: float, envelope: float) -> bool:
    """Whether *value* is below *envelope* past relative float noise."""
    return value < envelope - relative_tolerance(envelope)


def _single_group_factor_rows(records: pd.DataFrame | None) -> dict[int, pd.Series]:
    """``{code: row}`` for a maintenance/availability register (``MP``/``FD``),
    keeping only single-group plants — mirrors the shipped rule test's own
    ``drop_duplicates("codigo_usina", keep=False)`` idiom, which also drops
    Itaipu's two per-frequency rows (handled separately, per group, by
    :func:`convert_hydro_group_availability`'s own split-plant branch).
    """
    if records is None or records.empty:
        return {}
    single = records.drop_duplicates("codigo_usina", keep=False)
    return {int(row["codigo_usina"]): row for _, row in single.iterrows()}


def _stage_register_factor(
    row: pd.Series, prefix: str, code: int, stage_index: int
) -> float:
    """``row[f"{prefix}_{stage_index + 1}"]`` (1-based register columns).

    Raises loudly if *row* is present but the declared stage's column is
    missing or blank — a malformed register is never silently defaulted;
    only a plant with *no* row at all defaults to factor ``1.0`` (the
    caller's job, not this helper's).
    """
    column = f"{prefix}_{stage_index + 1}"
    if column not in row.index or pd.isna(row[column]):
        raise ValueError(
            f"plant {code}: {prefix} register has no value for stage "
            f"{stage_index} (column {column!r} missing or blank)"
        )
    return float(row[column])


def _availability_bound_entry(
    q_g: float,
    availability_mw: float,
    q_envelope: float,
    p_envelope: float,
) -> GroupBoundEntry | None:
    """One ``(hydro/group, stage)``'s availability-overlay entry, or ``None`` if
    neither bound column falls below that group's own declared envelope.

    ``max_generation_mw`` is *availability_mw* (installed × MP × FD — the
    source model's own reported available capacity, ``potencia_disponivel``)
    when it falls below *p_envelope*; ``max_turbined_m3s`` is the stage's
    turbined limit *q_g* when it falls below *q_envelope*. The turbined limit
    already holds the generator limit ``Pdisp / ρ_eq``, so no second
    ``ρ_eq · q`` generation cap is folded in: it would only duplicate it, and
    understate an FPHA plant whose production curve beats the linear
    ``ρ_eq``.
    """
    max_generation_mw = (
        availability_mw if _below_envelope(availability_mw, p_envelope) else None
    )
    max_turbined_m3s = q_g if _below_envelope(q_g, q_envelope) else None
    if max_generation_mw is None and max_turbined_m3s is None:
        return None
    return GroupBoundEntry(
        max_generation_mw=max_generation_mw, max_turbined_m3s=max_turbined_m3s
    )


def stage_availability_factors(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    effective: EffectiveCadastro,
) -> dict[tuple[int, int, int], float]:
    """``{(code, unit_group_id, stage_index): MP × FD}`` for every operated
    plant.

    A single-group plant (group 0) defaults a missing ``MP`` or ``FD``
    register to ``1.0``; a plant other than Itaipu whose register carries
    more than one row is also dropped to that default
    (:func:`_single_group_factor_rows`). Itaipu takes one factor per
    frequency group, each row frequency-matched via :func:`_frequency_row`
    with no default: the split plant declares both rows
    (:func:`_split_plant_frequencies`).
    """
    calendar = case.calendar
    mp = as_frame(case.dadger.mp(df=True))
    fd = as_frame(case.dadger.fd(df=True))
    mp_by_code = _single_group_factor_rows(mp)
    fd_by_code = _single_group_factor_rows(fd)

    factors: dict[tuple[int, int, int], float] = {}
    for code in id_map.hydro_codes:
        if code == _ITAIPU_CODE:
            frequencies = _split_plant_frequencies(
                case.hidr.loc[code], code, effective, mp, fd
            )
            rows = [
                (_frequency_row(mp, code, f), _frequency_row(fd, code, f))
                for f in frequencies
            ]
        else:
            rows = [(mp_by_code.get(code), fd_by_code.get(code))]
        for group_id, (mp_row, fd_row) in enumerate(rows):
            for stage in calendar:
                mp_factor = (
                    1.0
                    if mp_row is None
                    else _stage_register_factor(mp_row, "manutencao", code, stage.index)
                )
                fd_factor = (
                    1.0
                    if fd_row is None
                    else _stage_register_factor(fd_row, "fator", code, stage.index)
                )
                factors[(code, group_id, stage.index)] = mp_factor * fd_factor
    return factors


def convert_hydro_group_availability(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    effective: EffectiveCadastro,
    turbine_limits: TurbineLimits,
) -> dict[tuple[int, int, int], GroupBoundEntry]:
    """Per-group per-stage available capacity and turbined limit.

    ``max_generation_mw`` is the available power, the stage's AC-adjusted
    installed power times :func:`stage_availability_factors`' ``MP × FD``,
    against the group's rated envelope (:func:`_rated_envelope`, or Itaipu's
    per-conjunto :func:`_conjunto_rated_envelope`). ``max_turbined_m3s`` is
    the stage's turbined limit against its envelope over the horizon
    (*turbine_limits*). Itaipu's groups follow its ascending frequencies, as
    :func:`_build_split_unit_groups` numbers them.

    Returns the ``(hydro_id, hydro_unit_group_id, stage_id) ->
    GroupBoundEntry`` mapping ``convert_hydro_unit_group_bounds``
    consumes unchanged — populated *sparsely*, only where an emitted value
    falls below that group's own declared envelope, the same "only where it
    differs" convention every sibling emitter uses.
    """
    factors = stage_availability_factors(case, id_map, effective=effective)
    values: dict[tuple[int, int, int], GroupBoundEntry] = {}
    for code in id_map.hydro_codes:
        hreg = case.hidr.loc[code]
        hydro_id = id_map.hydro_id(code)
        for group_id in range(turbine_limits.group_count(code)):
            if code == _ITAIPU_CODE:
                _, p_envelope = _conjunto_rated_envelope(
                    hreg, code, group_id + 1, effective
                )
            else:
                _, p_envelope = _rated_envelope(hreg, code, effective)
            q_envelope = turbine_limits.envelope(code, group_id)
            for stage in case.calendar:
                if code == _ITAIPU_CODE:
                    _, installed_mw = _conjunto_rated_ac_adjusted(
                        hreg, code, group_id + 1, effective, stage.index
                    )
                else:
                    _, installed_mw = _compute_max_turbined_rated_ac_adjusted(
                        hreg, code, effective, stage.index
                    )
                entry = _availability_bound_entry(
                    turbine_limits.cap(code, group_id, stage.index),
                    installed_mw * factors[(code, group_id, stage.index)],
                    q_envelope,
                    p_envelope,
                )
                if entry is not None:
                    values[(hydro_id, group_id, stage.index)] = entry
    return values


#: The two Itaipu frequency-half minimum-generation floors on the ``RI``
#: register, keyed by the split-plant group id they bound. Each value is the
#: ``RI`` column prefix whose per-patamar slots (``…_1``.. ``…_5``) carry that
#: half's floor.
_ITAIPU_MIN_GENERATION_PREFIXES: tuple[tuple[int, str], ...] = (
    (ITAIPU_GROUP_BY_FREQUENCY_HZ[50.0], "geracao_minima_50_hz"),
    (ITAIPU_GROUP_BY_FREQUENCY_HZ[60.0], "geracao_minima_60_hz"),
)


def convert_itaipu_frequency_min_generation(
    case: DecompCase,
    id_map: DecompIdMap,
) -> dict[tuple[int, int, int], list[float]]:
    """Itaipu's per-frequency minimum-generation floors from the ``RI`` register.

    DECOMP's ``RI`` (restrição de Itaipu) register carries a must-run floor on
    each Itaipu frequency half: ``geracao_minima_50_hz`` on the 50 Hz group
    (on the plant's own SE bus, where it also serves the netted-in
    Paraguay/ANDE load) and ``geracao_minima_60_hz`` on the 60 Hz group (on
    the ``IV`` bus, the transshipment corridor into Ivaiporã). Each is a
    per-(estágio, patamar) list, forward-filled across the calendar the same
    way
    :func:`~novomodelo_bridge.decomp.converters.libs_electrical.read_carga_ande`
    fills the co-located ``carga_ande`` load. In DECOMP the 50 Hz floor binds
    (the 50 Hz
    half sits exactly at it), so dropping it lets the converted case
    under-run Itaipu's 50 Hz half and backfill the ANDE load from the rest of
    the system through SE's own lines.

    Returns ``{(hydro_id, hydro_unit_group_id, stage_index): [MW per block]}``
    for the two Itaipu groups, ready to merge into the
    ``hydro_unit_group_bounds`` overlay's ``min_generation_mw`` column
    (:func:`~novomodelo_bridge.decomp.group_bounds.convert_hydro_unit_group_bounds`).
    Returns ``{}`` when the deck operates no Itaipu or carries no ``RI``
    register (a deck with no Itaipu import).

    Raises
    ------
    ValueError
        When a declared row's ``estagio`` falls outside the calendar, when a
        frequency half's patamar-value count does not match the calendar's
        block count for that stage, or when the register declares a frequency
        half but no estágio-1 row for it (no forward-fill base) -- the same
        fail-loud contract as ``read_carga_ande``.
    """
    if _ITAIPU_CODE not in id_map.hydro_codes:
        return {}
    calendar = case.calendar
    ri = as_frame(case.dadger.ri(df=True))
    if ri is None or ri.empty:
        return {}

    hydro_id = id_map.hydro_id(_ITAIPU_CODE)
    n_stages = len(calendar)
    result: dict[tuple[int, int, int], list[float]] = {}
    for group_id, prefix in _ITAIPU_MIN_GENERATION_PREFIXES:
        patamar_columns = [
            column
            for _, column in sorted(
                (int(column[len(prefix) + 1 :]), column)
                for column in ri.columns
                if column.startswith(f"{prefix}_")
                and column[len(prefix) + 1 :].isdigit()
            )
        ]
        if not patamar_columns:
            continue

        declared: dict[int, list[float]] = {}
        for _, row in ri.iterrows():
            stage_index = int(row["estagio"]) - 1
            if not 0 <= stage_index < n_stages:
                raise ValueError(
                    f"RI {prefix} período {int(row['estagio'])} outside the "
                    f"calendar (1..{n_stages})"
                )
            values = [row[column] for column in patamar_columns]
            while values and pd.isna(values[-1]):
                values.pop()
            n_blocks = len(calendar[stage_index].block_hours)
            if len(values) != n_blocks:
                raise ValueError(
                    f"RI {prefix} stage {stage_index}: {len(values)} patamares "
                    f"declared vs {n_blocks} blocks in the calendar"
                )
            declared[stage_index] = [float(value) for value in values]

        if 0 not in declared:
            raise ValueError(
                f"RI {prefix}: no row declares estágio 1; sparse-stage "
                "inheritance has no base"
            )

        per_stage: list[list[float]] = []
        for stage in calendar:
            per_stage.append(
                declared.get(stage.index, per_stage[-1] if per_stage else declared[0])
            )
        for stage, values in zip(calendar, per_stage, strict=True):
            result[(hydro_id, group_id, stage.index)] = values

    return result
