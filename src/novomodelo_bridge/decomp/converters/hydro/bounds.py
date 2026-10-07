"""DECOMP hydro capacity caps: AC-adjusted rated capability, head-corrected
engolimento envelopes, the per-frequency split plant (Itaipu) unit groups,
and the B8 per-group per-stage availability overlay.

Imports the equivalent-productivity helper from ``.productivity``; imports
nothing from ``.entity`` -- kept acyclic so the entity
cluster can import from here, never the reverse.

The head-corrected ``max_turbined_m3s`` (:func:`_head_corrected_envelope`,
:func:`_conjunto_head_corrected_envelope`) mirrors the source-model side's
``newave.converters.hydro.bounds._compute_max_turbined_head_corrected``,
validated against ``relato.rv3``'s ``Qtur Maxima``: overall median error
~1.3%, run-of-river ~0.4%, reservoir plants ~4.8% residual undershoot (low
impact -- reservoirs are energy-, not flow-, limited). ``max_generation_mw``
(:func:`_rated_envelope`) stays on the un-derated rated envelope throughout;
only the turbined flow is head-aware, and the two compose orthogonally with
the per-stage machine-set envelope here (one scales the rated flow by head,
the other by which conjuntos are effective).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from cobre_bridge.core.hydro_units import build_mirror_unit_group
from cobre_bridge.core.productivity import KTURB_BY_TIPO_TURBINA
from cobre_bridge.core.tolerances import relative_tolerance
from cobre_bridge.decomp.converters.hydro.productivity import (
    _equivalent_productivity_mw_per_m3s,
)
from cobre_bridge.decomp.group_bounds import GroupBoundEntry

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cobre_bridge.decomp.case import DecompCase
    from cobre_bridge.decomp.converters.cadastro import EffectiveCadastro
    from cobre_bridge.decomp.id_map import DecompIdMap


#: Itaipu's plant code — the one plant whose ``MP``/``FD`` maintenance and
#: availability registers are declared per generating-unit group (one row
#: per ``frequencia``, 50/60 Hz) rather than per plant. Both
#: :func:`convert_hydros` (two conjunto-backed unit groups instead of one
#: mirror group) and :func:`convert_hydro_group_availability` (a per-group,
#: not per-plant, availability overlay) special-case it; every other plant
#: takes the ordinary single-group path.
_ITAIPU_CODE = 66


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
    conjunto, and :func:`_build_split_unit_groups`/
    :func:`_conjunto_rated_envelope`/:func:`convert_hydro_group_availability`
    call directly for a single conjunto to get one per-frequency group's own
    bounds. Reads ``effective.machine_set(code, conjunto_index,
    stage_index)``: when present, its three fields already reflect
    :func:`~cobre_bridge.decomp.converters.cadastro.effective.
    build_effective_cadastro`'s independent per-field densification (a
    field with no override of its own is the ``hidr`` base, forward-filled
    flat); ``None`` means the pair carries no override at all, and every
    field falls back to the ``hidr.dat`` row directly — the same
    ``.get(..., base)`` fallback the date-blind reader used.
    """
    machine_set = effective.machine_set(code, conjunto_index, stage_index)
    if machine_set is None:
        n_machines = int(hreg[f"maquinas_conjunto_{conjunto_index}"])
        q_nom = float(hreg[f"vazao_nominal_conjunto_{conjunto_index}"])
        p_nom = float(hreg[f"potencia_nominal_conjunto_{conjunto_index}"])
    else:
        n_machines = machine_set.numero_maquinas
        q_nom = machine_set.vazao
        p_nom = machine_set.potencia
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
    n_sets = effective.machine_conjunto_count(code, stage_index)
    if n_sets is None:
        n_sets = int(hreg["numero_conjuntos_maquinas"])
    max_turbined = 0.0
    max_generation = 0.0
    for i in range(1, n_sets + 1):
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
    base for the B8 availability overlay
    (:func:`convert_hydro_group_availability`) — mirrors
    :func:`~cobre_bridge.decomp.converters.cadastro.effective.
    storage_envelope`'s outer-bound construction: the widest each of
    :func:`_compute_max_turbined_rated_ac_adjusted`'s two components ever
    reaches over the horizon, taken independently (the two maxima need not
    land on the same stage). A
    constant machine set collapses this to the stage-0 value, matching the
    date-blind value exactly. Not used for the split plant
    (Itaipu): its entity envelope is instead the *sum* of the two groups'
    own :func:`_conjunto_rated_envelope`, computed in
    :func:`convert_hydros`, so cobre rule 41 holds by construction rather
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
    for its own B8 availability overlay
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


def _conjunto_h_nom(hreg: pd.Series, conjunto_index: int) -> float:
    """The base ``hidr`` nominal head (``queda_nominal_conjunto_{i}``) for one
    1-based *conjunto_index*, or ``0.0`` when the column is absent or blank.

    ``0.0`` is the same sentinel :func:`_conjunto_head_corrected_ac_adjusted`/
    :func:`_head_corrected_max_turbined_ac_adjusted` treat as "no nominal
    head" (falls back to the rated flow for that conjunto, Requirement 1) —
    a genuine ``hidr.dat`` row always carries this column (the registry is
    byte-identical across the two source families), so a missing column only
    ever happens in a hand-built test fixture that doesn't exercise the head
    correction. The source model's own effective-head override
    (``AC ALTEFE``) is never consulted here — see the ``TRACKED COBRE-GAP
    WORKAROUND`` in :func:`convert_hydros`.
    """
    raw = hreg.get(f"queda_nominal_conjunto_{conjunto_index}")
    if raw is None or pd.isna(raw):
        return 0.0
    return float(raw)


def _operating_head(
    effective: EffectiveCadastro, code: int, stage_index: int, rho_eq: float
) -> float | None:
    """The per-stage operating head ``h_op = ρ_eq / ρ_esp``, or ``None`` when
    *rho_eq*, ``produtibilidade_especifica``, or the resulting ratio is
    non-positive.

    The single defensive gate every head-corrected envelope/overlay caller
    checks before trusting *h_op* — a ``None`` return means "fall back to
    the rated flow for this stage", never a meaningless or negative bound
    (Requirement/Error Handling). *rho_eq* is passed in rather than
    recomputed so a caller that already needs it for its own hydraulic-cap
    calculation (:func:`convert_hydro_group_availability`) calls
    :func:`_equivalent_productivity_mw_per_m3s` exactly once per stage.
    """
    if rho_eq <= 0.0:
        return None
    rho_esp = effective.value(code, "produtibilidade_especifica", stage_index)
    if rho_esp <= 0.0:
        return None
    h_op = rho_eq / rho_esp
    return h_op if h_op > 0.0 else None


def _conjunto_affinity_flow(
    n_q: float, h_nom: float, h_op: float, kturb: float
) -> float:
    """One conjunto's turbine-affinity-corrected flow ``n·q_nom·(h_op / h_nom)
    ** k_turb``, or the uncorrected ``n·q_nom`` when *h_nom* ``<= 0`` (no
    nominal head — falls back to the rated flow for that conjunto).

    The single place the affinity term is computed, shared by
    :func:`_conjunto_head_corrected_ac_adjusted` (per-conjunto power cap) and
    :func:`_head_corrected_max_turbined_ac_adjusted` (plant-wide power cap) so
    the two read as true mirrors — only the *capping* differs between them,
    not the affinity law itself.
    """
    if h_nom <= 0.0:
        return n_q
    return n_q * (h_op / h_nom) ** kturb


def _conjunto_head_corrected_ac_adjusted(
    hreg: pd.Series,
    code: int,
    conjunto_index: int,
    effective: EffectiveCadastro,
    stage_index: int,
    h_op: float,
    rho_eq: float,
) -> float:
    """The per-conjunto head-corrected flow at *stage_index*: ``min(n·q_nom·
    (h_op / h_nom) ** k_turb, n·p_nom / ρ_eq)``.

    ``(n·q_nom, n·p_nom)`` come from :func:`_conjunto_rated_ac_adjusted` (the
    same AC-effective building block the rated pair uses); ``h_nom`` is the
    base ``hidr`` ``queda_nominal_conjunto_{conjunto_index}``
    (:func:`_conjunto_h_nom`) — **not** the ``AC ALTEFE`` override, which has
    no idecomp value accessor — and ``k_turb`` is looked up from
    ``KTURB_BY_TIPO_TURBINA`` by the plant's ``tipo_turbina`` (0.5 for
    Francis/Pelton, 0.2 for Kaplan). No TEIF/IP availability derating is
    applied (validated worse: 8.2% vs 1.3% median error vs ``Qtur Maxima`` —
    availability lives on the MP×FD/B8 path instead) and no
    ``tipo_regulacao`` head-branch (the full-range-mean *h_op* fits every
    class). ``h_nom ≤ 0`` skips the affinity correction for this conjunto
    (falls back to the rated ``n·q_nom``) but the power cap still applies;
    *h_op* and *rho_eq* are assumed already validated positive by the caller
    (:func:`_operating_head`). The per-conjunto (not plant-wide) power cap
    here is deliberate — used only for the split plant's own groups
    (:func:`_conjunto_head_corrected_envelope`,
    :func:`convert_hydro_group_availability`'s Itaipu branch), each of which
    only has its own conjunto's installed power to draw on.
    """
    n_q, n_p = _conjunto_rated_ac_adjusted(
        hreg, code, conjunto_index, effective, stage_index
    )
    h_nom = _conjunto_h_nom(hreg, conjunto_index)
    tipo_turbina = int(hreg.get("tipo_turbina", 0) or 0)
    kturb = KTURB_BY_TIPO_TURBINA.get(tipo_turbina, 0.5)
    affinity = _conjunto_affinity_flow(n_q, h_nom, h_op, kturb)
    return min(affinity, n_p / rho_eq)


def _head_corrected_max_turbined_ac_adjusted(
    hreg: pd.Series,
    code: int,
    effective: EffectiveCadastro,
    stage_index: int,
    h_op: float,
    rho_eq: float,
) -> float:
    """The plant-level head-corrected cap at *stage_index*: ``min(Σ_c
    n_c·q_eff_c, Σ_c n_c·p_nom_c / ρ_eq)`` — sum the per-conjunto affinity
    flow first, then apply **one** plant-wide power cap, over every conjunto
    effective at *stage_index* (``effective.machine_conjunto_count``,
    falling back to the ``hidr`` base ``numero_conjuntos_maquinas``).

    This is the non-split-plant counterpart of
    :func:`_conjunto_head_corrected_ac_adjusted`: it deliberately does
    **not** sum that function's own (per-conjunto-capped) return values —
    doing so would cap the power per conjunto, which only the split plant's
    own groups do. Instead it re-derives each conjunto's affinity flow the
    same way (``h_nom`` from :func:`_conjunto_h_nom`, ``k_turb`` from
    ``KTURB_BY_TIPO_TURBINA``, no TEIF/IP, no ``tipo_regulacao`` branch —
    see :func:`_conjunto_head_corrected_ac_adjusted`'s docstring for the
    validated rationale) and sums both the affinity flow and the installed
    power across conjuntos before taking the single min. *h_op* and *rho_eq*
    are assumed already validated positive by the caller
    (:func:`_operating_head`).
    """
    n_sets = effective.machine_conjunto_count(code, stage_index)
    if n_sets is None:
        n_sets = int(hreg["numero_conjuntos_maquinas"])
    tipo_turbina = int(hreg.get("tipo_turbina", 0) or 0)
    kturb = KTURB_BY_TIPO_TURBINA.get(tipo_turbina, 0.5)
    sum_affinity = 0.0
    sum_p = 0.0
    for i in range(1, n_sets + 1):
        n_q, n_p = _conjunto_rated_ac_adjusted(hreg, code, i, effective, stage_index)
        h_nom = _conjunto_h_nom(hreg, i)
        sum_affinity += _conjunto_affinity_flow(n_q, h_nom, h_op, kturb)
        sum_p += n_p
    return min(sum_affinity, sum_p / rho_eq)


def _head_corrected_envelope(
    hreg: pd.Series, code: int, effective: EffectiveCadastro
) -> float:
    """Max-over-stages head-corrected ``max_turbined_m3s`` for the whole
    plant — the head-aware mirror of :func:`_rated_envelope`'s flow
    component (``max_generation_mw`` stays on the rated envelope; this
    function returns only the flow).

    Per stage, computes ``h_op = ρ_eq(stage) / ρ_esp(stage)`` via
    :func:`_operating_head` (``ρ_eq(stage)`` from
    :func:`_equivalent_productivity_mw_per_m3s`, the same per-stage value
    :func:`convert_hydro_group_availability`'s hydraulic ceiling uses) and,
    when that's positive, the head-corrected flow
    (:func:`_head_corrected_max_turbined_ac_adjusted`); a stage where
    *h_op*, ``ρ_esp``, or ``ρ_eq`` is non-positive falls back defensively to
    that stage's rated flow (:func:`_compute_max_turbined_rated_ac_adjusted`)
    rather than emit a meaningless bound.
    """
    per_stage: list[float] = []
    for stage in range(effective.n_stages):
        rho_eq = _equivalent_productivity_mw_per_m3s(effective, code, stage)
        h_op = _operating_head(effective, code, stage, rho_eq)
        if h_op is None:
            value = _compute_max_turbined_rated_ac_adjusted(
                hreg, code, effective, stage
            )[0]
        else:
            value = _head_corrected_max_turbined_ac_adjusted(
                hreg, code, effective, stage, h_op, rho_eq
            )
        per_stage.append(value)
    return max(per_stage)


def _conjunto_head_corrected_envelope(
    hreg: pd.Series, code: int, conjunto_index: int, effective: EffectiveCadastro
) -> float:
    """Max-over-stages head-corrected flow for one conjunto — the per-group
    mirror of :func:`_head_corrected_envelope`, used for the split plant's
    own groups (:func:`_build_split_unit_groups`) and their own B8 overlay
    (:func:`convert_hydro_group_availability`'s Itaipu branch).

    Same per-stage defensive fallback as :func:`_head_corrected_envelope`:
    a stage where :func:`_operating_head` returns ``None`` uses that
    conjunto's rated flow (:func:`_conjunto_rated_ac_adjusted`) instead.
    """
    per_stage: list[float] = []
    for stage in range(effective.n_stages):
        rho_eq = _equivalent_productivity_mw_per_m3s(effective, code, stage)
        h_op = _operating_head(effective, code, stage, rho_eq)
        if h_op is None:
            value = _conjunto_rated_ac_adjusted(
                hreg, code, conjunto_index, effective, stage
            )[0]
        else:
            value = _conjunto_head_corrected_ac_adjusted(
                hreg, code, conjunto_index, effective, stage, h_op, rho_eq
            )
        per_stage.append(value)
    return max(per_stage)


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

    Raises
    ------
    ValueError
        Naming *code*, if the ``MP`` and ``FD`` registers disagree on the
        set of declared frequencies (or either is empty) — no silent
        single-group fallback for a mismatched registry — or if the
        stage-0 AC-adjusted conjunto count does not equal the number of
        declared frequencies.
    """

    def _freqs(table: pd.DataFrame | None) -> set[float]:
        if table is None or table.empty:
            return set()
        return {
            float(f) for f in table.loc[table["codigo_usina"] == code, "frequencia"]
        }

    mp_freqs, fd_freqs = _freqs(mp), _freqs(fd)
    if not mp_freqs or mp_freqs != fd_freqs:
        raise ValueError(
            f"plant {code}: MP frequencia rows {sorted(mp_freqs)} and FD "
            f"frequencia rows {sorted(fd_freqs)} must agree and be "
            "non-empty for the per-frequency split plant"
        )
    frequencies = sorted(mp_freqs)

    n_sets = effective.machine_conjunto_count(code, 0)
    if n_sets is None:
        n_sets = int(hreg["numero_conjuntos_maquinas"])
    if n_sets != len(frequencies):
        raise ValueError(
            f"plant {code}: numero_conjuntos_maquinas ({n_sets}) does not "
            f"match the number of MP/FD frequencia rows ({len(frequencies)})"
        )
    return frequencies


def _build_split_unit_groups(
    hreg: pd.Series,
    code: int,
    name: str,
    group_bus_ids: Sequence[int],
    frequencies: list[float],
    effective: EffectiveCadastro,
) -> tuple[list[dict[str, object]], float, float]:
    """Conjunto-backed unit groups for the per-frequency split plant.

    Group id ``i`` (``i`` in ``0..len(frequencies)-1``, ascending frequency,
    so id 0 = the lowest — 50 Hz for Itaipu) is backed by hidr conjunto
    ``i + 1``: its ``max_generation_mw`` is *that* conjunto's own
    max-over-stages AC-adjusted rated envelope
    (:func:`_conjunto_rated_envelope`); its ``max_turbined_m3s`` is instead
    that conjunto's own head-corrected envelope
    (:func:`_conjunto_head_corrected_envelope`) — each group has
    only its own conjunto's installed power to draw on, so the power cap
    inside that function is per-group, never plant-wide. Group ``i`` sits on
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
    independently, so cobre rule 41 holds by construction even though the
    two groups' own per-stage machine-set changes (if any) need not peak on
    the same stage.
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
        conjunto_index = i + 1
        _, p = _conjunto_rated_envelope(hreg, code, conjunto_index, effective)
        q = _conjunto_head_corrected_envelope(hreg, code, conjunto_index, effective)
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
    """The single ``(codigo_usina, frequencia)``-matched register row.

    Unlike the ordinary single-group case (a missing register defaults its
    factor to ``1.0``), the split plant's two groups each require exactly
    one ``MP`` row and one ``FD`` row at their own frequency — validated by
    :func:`_split_plant_frequencies` before this is called, so a missing or
    duplicated row here means the registers disagree with that validation
    (defensive; should not happen in practice).
    """
    if table is None:
        raise ValueError(f"plant {code}: no register table for frequencia {frequency}")
    rows = table.loc[
        (table["codigo_usina"] == code) & (table["frequencia"] == frequency)
    ]
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
    """One ``(hydro/group, stage)``'s B8 overlay entry, or ``None`` if
    neither bound column falls below that group's own declared envelope.

    ``max_generation_mw`` is *availability_mw* (installed × MP × FD — the
    source model's own reported available capacity, ``potencia_disponivel``)
    when it falls below *p_envelope*. The turbined-flow ceiling is *not*
    folded into this generation cap: ``max_turbined_m3s`` below already caps
    the flow at the per-stage head-corrected engolimento, and generation is
    ``productivity × turbined``, so the physical hydraulic limit binds through
    the flow cap — a separate ``ρ_eq·q_max`` generation cap only ever
    duplicated it (and understated any FPHA plant whose production curve beat
    the linear ``ρ_eq`` estimate).

    ``max_turbined_m3s`` is *q_g* — the per-stage **head-corrected** turbined
    flow (the head-aware engolimento supplied by the caller, not the rated
    unit-flow sum) — when it falls below *q_envelope* (a mid-horizon machine-set
    shrink or a per-stage operating-head change lowers the turbined-flow cap too,
    not just generation). Shared by the single-group and Itaipu
    per-conjunto-group emission loops in
    :func:`convert_hydro_group_availability`.
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


def convert_hydro_group_availability(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    effective: EffectiveCadastro,
) -> dict[tuple[int, int, int], GroupBoundEntry]:
    """B8 per-group per-stage available capacity.

    For the single-group majority (every plant but Itaipu):
    ``availability_mw(stage) = installed(stage) × MP(stage) × FD(stage)`` —
    ``installed(stage)`` is the per-stage AC-adjusted rated power
    (:func:`_compute_max_turbined_rated_ac_adjusted`, sourced from
    *effective*'s machine-set view), a missing ``MP``/``FD`` register
    defaults its factor to ``1.0``. This *is* the source model's own reported
    available capacity (``potencia_disponivel``), so it is emitted directly as
    the ``max_generation_mw`` overlay, sparse against the group's own rated
    envelope (:func:`_rated_envelope`). The per-stage hydraulic engolimento
    limit is *not* imposed as a second generation cap: the emitted
    ``max_turbined_m3s`` overlay is the per-stage **head-corrected** flow
    (:func:`_head_corrected_max_turbined_ac_adjusted`), sparse
    against the group's own head-corrected envelope
    (:func:`_head_corrected_envelope`), and generation is
    ``productivity × turbined`` — so the physical hydraulic limit binds
    through the flow cap, not through a redundant ``ρ_eq·q_max`` generation
    ceiling (:func:`_availability_bound_entry`). A mid-horizon machine-set
    shrink or head drop lowers both ceilings independently. Defensively, any
    *other* plant whose ``MP``/``FD`` register carries more than one row is
    also dropped from this path (today only Itaipu, code 66, does).

    For Itaipu specifically, the same B8 formula is computed **per
    conjunto-backed group** rather than per plant —
    ``availability_mw(g, stage) = installed_g(stage) × MP_g(stage) ×
    FD_g(stage)``, ``g`` in ``{0, 1}`` (frequencies sorted ascending,
    matching :func:`_build_split_unit_groups`'s group ids), each register
    row frequency-matched via :func:`_frequency_row` (no defaulting to
    ``1.0`` — the split plant declares both rows); its ``max_turbined_m3s``
    overlay is the per-conjunto head-corrected flow
    (:func:`_conjunto_head_corrected_ac_adjusted`), sparse against that
    group's own head-corrected envelope
    (:func:`_conjunto_head_corrected_envelope`).

    Returns the ``(hydro_id, hydro_unit_group_id, stage_id) ->
    GroupBoundEntry`` mapping ``convert_hydro_unit_group_bounds``
    consumes unchanged — populated *sparsely*, only where an emitted value
    falls below that group's own declared envelope, the same "only where it
    differs" convention every sibling emitter uses.
    """
    hidr = case.hidr
    calendar = case.calendar
    mp = case.dadger.mp(df=True)
    fd = case.dadger.fd(df=True)
    mp_by_code = _single_group_factor_rows(mp)
    fd_by_code = _single_group_factor_rows(fd)

    values: dict[tuple[int, int, int], GroupBoundEntry] = {}
    for code in id_map.hydro_codes:
        hreg = hidr.loc[code]
        hydro_id = id_map.hydro_id(code)
        if code == _ITAIPU_CODE:
            frequencies = _split_plant_frequencies(hreg, code, effective, mp, fd)
            for i, frequency in enumerate(frequencies):
                conjunto_index = i + 1
                _, p_envelope = _conjunto_rated_envelope(
                    hreg, code, conjunto_index, effective
                )
                q_envelope = _conjunto_head_corrected_envelope(
                    hreg, code, conjunto_index, effective
                )
                mp_row = _frequency_row(mp, code, frequency)
                fd_row = _frequency_row(fd, code, frequency)

                for stage in calendar:
                    rho_eq = _equivalent_productivity_mw_per_m3s(
                        effective, code, stage.index
                    )
                    q_g, p_g = _conjunto_rated_ac_adjusted(
                        hreg, code, conjunto_index, effective, stage.index
                    )
                    mp_factor = _stage_register_factor(
                        mp_row, "manutencao", code, stage.index
                    )
                    fd_factor = _stage_register_factor(
                        fd_row, "fator", code, stage.index
                    )
                    availability_mw = p_g * mp_factor * fd_factor
                    h_op = _operating_head(effective, code, stage.index, rho_eq)
                    q_head = (
                        q_g
                        if h_op is None
                        else _conjunto_head_corrected_ac_adjusted(
                            hreg,
                            code,
                            conjunto_index,
                            effective,
                            stage.index,
                            h_op,
                            rho_eq,
                        )
                    )
                    entry = _availability_bound_entry(
                        q_head, availability_mw, q_envelope, p_envelope
                    )
                    if entry is not None:
                        values[(hydro_id, i, stage.index)] = entry
            continue
        _, p_envelope = _rated_envelope(hreg, code, effective)
        q_envelope = _head_corrected_envelope(hreg, code, effective)
        mp_row = mp_by_code.get(code)
        fd_row = fd_by_code.get(code)

        for stage in calendar:
            rho_eq = _equivalent_productivity_mw_per_m3s(effective, code, stage.index)
            q_g, p_g = _compute_max_turbined_rated_ac_adjusted(
                hreg, code, effective, stage.index
            )
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
            availability_mw = p_g * mp_factor * fd_factor

            h_op = _operating_head(effective, code, stage.index, rho_eq)
            q_head = (
                q_g
                if h_op is None
                else _head_corrected_max_turbined_ac_adjusted(
                    hreg, code, effective, stage.index, h_op, rho_eq
                )
            )
            entry = _availability_bound_entry(
                q_head, availability_mw, q_envelope, p_envelope
            )
            if entry is not None:
                values[(hydro_id, 0, stage.index)] = entry

    return values


#: The two Itaipu frequency-half minimum-generation floors on the ``RI``
#: register, keyed by the split-plant group id they bound (0 = 50 Hz, 1 =
#: 60 Hz, matching :func:`_build_split_unit_groups`'s ascending-frequency
#: ordering). Each value is the ``RI`` column prefix whose per-patamar slots
#: (``…_1``.. ``…_5``) carry that half's floor.
_ITAIPU_MIN_GENERATION_PREFIXES: tuple[tuple[int, str], ...] = (
    (0, "geracao_minima_50_hz"),
    (1, "geracao_minima_60_hz"),
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
    :func:`~cobre_bridge.decomp.converters.libs_electrical.read_carga_ande`
    fills the co-located ``carga_ande`` load. In DECOMP the 50 Hz floor binds
    (the 50 Hz
    half sits exactly at it), so dropping it lets the converted case
    under-run Itaipu's 50 Hz half and backfill the ANDE load from the rest of
    the system through SE's own lines.

    Returns ``{(hydro_id, hydro_unit_group_id, stage_index): [MW per block]}``
    for the two Itaipu groups, ready to merge into the
    ``hydro_unit_group_bounds`` overlay's ``min_generation_mw`` column
    (:func:`~cobre_bridge.decomp.group_bounds.convert_hydro_unit_group_bounds`).
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
    ri = case.dadger.ri(df=True)
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
