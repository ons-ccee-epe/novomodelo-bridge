"""Per-stage-effective cadastro view: assembly, storage range, and resolution report."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd
from idecomp.decomp.modelos.dadger import ACCOTVOL, ACNUMJUS, ACNUMPOS

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from idecomp.decomp import Dadger

    from cobre_bridge.decomp.temporal import OperativeStage

from cobre_bridge.decomp.converters.cadastro.overrides import (
    DiversionChannel,
    MachineSet,
    OutOfHorizon,
    _forward_fill_series,
    _plant_code_key,
    _read_diversion_overrides,
    _read_keyed_overrides,
    _read_machine_set_overrides,
    _read_polynomial_overrides,
    _read_scalar_overrides,
)


@dataclass(frozen=True)
class EffectiveCadastro:
    """Per-stage-effective view of the cadastro.

    Holds the base cadastro (the ``hidr`` DataFrame, indexed by plant code)
    alongside a sparse map of the ``(code, param)`` pairs that carry at
    least one override, each already densified to a per-stage tuple by
    :func:`_forward_fill_series`. A ``(code, param)`` pair absent from
    *stage_varying* has no override and falls through to *base* for every
    stage.
    """

    base: pd.DataFrame
    n_stages: int
    stage_varying: Mapping[tuple[int, str], tuple[float, ...]]
    diversions: Mapping[int, tuple[DiversionChannel | None, ...]] = field(
        default_factory=dict
    )
    machine_conjunto_counts: Mapping[int, tuple[int, ...]] = field(default_factory=dict)
    machine_sets: Mapping[tuple[int, int], tuple[MachineSet, ...]] = field(
        default_factory=dict
    )
    cota_polynomials: Mapping[int, tuple[tuple[float, ...], ...]] = field(
        default_factory=dict
    )
    downstream_links: Mapping[int, tuple[int, ...]] = field(default_factory=dict)
    inflow_gauges: Mapping[int, tuple[int, ...]] = field(default_factory=dict)

    def value(self, code: int, param: str, stage_index: int) -> float:
        """Effective value of *param* for plant *code* at *stage_index*."""
        key = (code, param)
        if key in self.stage_varying:
            return self.stage_varying[key][stage_index]
        return float(self.base.loc[code, param])

    def series(self, code: int, param: str) -> list[float]:
        """Dense per-stage series of *param* for plant *code*."""
        key = (code, param)
        if key in self.stage_varying:
            return list(self.stage_varying[key])
        return [float(self.base.loc[code, param])] * self.n_stages

    def is_stage_varying(self, code: int, param: str) -> bool:
        """Whether *param* for plant *code* carries at least one override."""
        return (code, param) in self.stage_varying

    def has_diversion(self, code: int) -> bool:
        """Whether plant *code* has a base or ``AC DESVIO`` diversion channel.

        ``True`` iff *code* is a key of :attr:`diversions`, i.e. it carries a
        non-zero base ``desvio`` code or an in-horizon ``AC DESVIO`` record in
        at least one stage.
        """
        return code in self.diversions

    def diversion(self, code: int, stage_index: int) -> DiversionChannel | None:
        """The diversion channel active for plant *code* at *stage_index*.

        ``None`` when plant *code* has no diversion at all (absent from
        :attr:`diversions`) or when it has a diversion in some stages but not
        at *stage_index*.
        """
        if code not in self.diversions:
            return None
        return self.diversions[code][stage_index]

    def machine_conjunto_count(self, code: int, stage_index: int) -> int | None:
        """The overridden per-stage ``AC NUMCON`` conjunto count for plant *code*.

        ``None`` when *code* carries no ``NUMCON`` override at all (absent
        from :attr:`machine_conjunto_counts`) — the caller then falls through
        to the ``hidr`` base ``numero_conjuntos_maquinas``.
        """
        if code not in self.machine_conjunto_counts:
            return None
        return self.machine_conjunto_counts[code][stage_index]

    def machine_set(
        self, code: int, indice_conjunto: int, stage_index: int
    ) -> MachineSet | None:
        """The overridden per-stage machine configuration for one conjunto.

        ``None`` when ``(code, indice_conjunto)`` carries no ``NUMMAQ``/
        ``POTEFE``/``VAZEFE`` override at all (absent from
        :attr:`machine_sets`) — the caller then falls through to the
        ``hidr`` base per-conjunto columns.
        """
        key = (code, indice_conjunto)
        if key not in self.machine_sets:
            return None
        return self.machine_sets[key][stage_index]

    def cota_polynomial(self, code: int, stage_index: int) -> tuple[float, ...]:
        """The effective 5-coefficient forebay-cota polynomial for plant
        *code* at *stage_index*, ordered ``ordem 0..4``.

        Falls through to the base ``a{0..4}_volume_cota`` columns when *code*
        carries no ``AC COTVOL`` override at all (absent from
        :attr:`cota_polynomials`) — the same "absent means base" convention
        every other accessor here follows.
        """
        if code not in self.cota_polynomials:
            return tuple(
                float(self.base.loc[code, f"a{i}_volume_cota"]) for i in range(5)
            )
        return self.cota_polynomials[code][stage_index]

    def downstream_plant(self, code: int, stage_index: int) -> int:
        """The effective downstream-plant code for *code* at *stage_index*.

        Falls through to the base ``codigo_usina_jusante`` when *code*
        carries no ``AC NUMJUS`` override at all (absent from
        :attr:`downstream_links`) — the same "absent means base" convention
        every other accessor here follows. ``0`` is a valid return (the
        sink); the cascade walk (:func:`~cobre_bridge.decomp.converters.
        hydro.entity._downstream_operated`) treats it as such.
        """
        if code not in self.downstream_links:
            return int(self.base.loc[code, "codigo_usina_jusante"])
        return self.downstream_links[code][stage_index]

    def downstream_plant_varies(self, code: int) -> bool:
        """Whether *code*'s effective downstream link varies across stages
        (a temporal ``AC NUMJUS``) — the tracked-gap trigger:
        :func:`~cobre_bridge.decomp.converters.hydro.entity.
        _downstream_operated` reads one stage-representative link for the
        whole horizon (stage 0 by default), so a caller checks this to warn
        rather than silently picking a stage. ``False`` for a plant with no
        override at all.
        """
        if code not in self.downstream_links:
            return False
        series = self.downstream_links[code]
        return any(value != series[0] for value in series)

    def inflow_gauge(self, code: int, stage_index: int) -> int:
        """The effective inflow-gauge (``posto``) for *code* at *stage_index*.

        Falls through to the base ``posto`` when *code* carries no ``AC
        NUMPOS`` override at all (absent from :attr:`inflow_gauges`).
        """
        if code not in self.inflow_gauges:
            return int(self.base.loc[code, "posto"])
        return self.inflow_gauges[code][stage_index]

    def inflow_gauge_varies(self, code: int) -> bool:
        """Whether *code*'s effective inflow gauge varies across stages (a
        temporal ``AC NUMPOS``) — the gauge sibling of
        :meth:`downstream_plant_varies`.
        """
        if code not in self.inflow_gauges:
            return False
        series = self.inflow_gauges[code]
        return any(value != series[0] for value in series)


def effective_storage_range(
    effective: EffectiveCadastro, code: int, stage_index: int
) -> tuple[float, float]:
    """Per-stage effective storage range for plant *code*, in hm³.

    A run-of-river plant (``tipo_regulacao == "D"``) cannot accumulate water
    across stages — the source model's own precedent (``newave.converters.hydro``'s
    ``tipo_reg == "D"`` branch) freezes its operating range at the reference
    volume (``volume_referencia``) rather than the ``hidr`` registry's
    ``(volume_minimo, volume_maximo)`` band, so this collapses to
    ``(vol_ref, vol_ref)`` for a ``D`` plant. ``vol_ref`` falls back to the
    per-stage effective ``volume_minimo`` when ``volume_referencia`` is
    missing, ``NaN``, or ``<= 0`` — never a zero-width range at zero.
    ``tipo_regulacao`` is read defensively off the base row
    (``row.get(..., "")`` then ``str(...).strip()``, matching that same
    precedent), so a synthetic cadastro carrying no such column simply falls
    through to the per-stage read below.

    Every other regulation class — under the DECOMP predicate, a reservoir
    is ``tipo_regulacao in ("M", "S")`` — is unchanged: the per-stage
    ``(volume_minimo, volume_maximo)`` via :meth:`EffectiveCadastro.value`.
    This is the one place the ``D``-collapse predicate lives; every storage
    consumer (:func:`storage_envelope`, :func:`cobre_bridge.decomp.converters.bounds.
    convert_storage_bounds`, :func:`cobre_bridge.decomp.converters.hydro.entity.
    convert_initial_storage`) routes through it. Productivity does **not** —
    :func:`cobre_bridge.decomp.converters.hydro.productivity.
    _equivalent_productivity_mw_per_m3s` keeps reading the full
    ``(volume_minimo, volume_maximo)`` range directly,
    validated independently of this collapse.
    """
    row = effective.base.loc[code]
    tipo_reg = str(row.get("tipo_regulacao", "")).strip()
    if tipo_reg == "D":
        vol_ref_raw = row.get("volume_referencia")
        if vol_ref_raw is not None and not pd.isna(vol_ref_raw) and vol_ref_raw > 0:
            vol_ref = float(vol_ref_raw)
        else:
            vol_ref = effective.value(code, "volume_minimo", stage_index)
        return (vol_ref, vol_ref)
    return (
        effective.value(code, "volume_minimo", stage_index),
        effective.value(code, "volume_maximo", stage_index),
    )


def storage_envelope(effective: EffectiveCadastro, code: int) -> tuple[float, float]:
    """Outer per-stage operating range for plant *code*, in hm³.

    ``(min over stages of the effective floor, max over stages of the
    effective ceiling)`` per :func:`effective_storage_range` — the widest
    floor/ceiling the plant's dense per-stage range ever reaches. For a
    plant with no override and no run-of-river collapse both reduce to the
    base scalar. A run-of-river (``D``) plant's per-stage range is already
    the single-point collapse ``(vol_ref, vol_ref)`` at every stage, so its
    envelope collapses to that same point; every ``M``/``S`` plant is
    unchanged. This is the envelope the entity ``reservoir`` block
    declares as its default storage bounds;
    :func:`cobre_bridge.decomp.converters.bounds.convert_storage_bounds` emits a
    per-stage override wherever a stage's effective bounds differ from it.
    """
    ranges = [
        effective_storage_range(effective, code, stage_index)
        for stage_index in range(effective.n_stages)
    ]
    return (min(r[0] for r in ranges), max(r[1] for r in ranges))


def is_reservoir(effective: EffectiveCadastro, code: int) -> bool:
    """True iff plant *code* is a reservoir under the DECOMP predicate.

    DECOMP classifies both monthly-regulating (``"M"``) and weekly-regulating
    (``"S"``) plants as reservoirs; only ``"D"`` is run-of-river. This differs
    from the source model's monthly predicate (``"M"`` only).
    """
    if code not in effective.base.index:
        return False
    return str(effective.base.loc[code, "tipo_regulacao"]).strip() in ("M", "S")


def unregulated_runofriver_codes(
    effective: EffectiveCadastro,
    operated_codes: Iterable[int],
) -> set[int]:
    """Operated run-of-river codes with no reservoir upstream in the cascade.

    A plant *code* (from *operated_codes*) qualifies iff it is run-of-river
    (``tipo_regulacao == "D"``) and no reservoir (:func:`is_reservoir`) sits
    **upstream** of it — equivalently, it lies on the headwater side of the
    first reservoir in its cascade. These are the plants whose minimum-outflow
    restriction is released to avoid infeasibility on a stage/scenario with
    zero inflow.

    The upstream test walks *downstream* from every operated reservoir and
    marks each ``D`` plant reached as *not* unregulated; every operated ``D``
    plant not so marked lies above the first reservoir (or in a cascade with
    no reservoir at all) and qualifies. Topology is the *effective* (post-``AC
    NUMJUS``) chain at the initial stage — ``tipo_regulacao`` is a
    base-cadastro scalar and the cascade is read stage-representative, as
    every other cascade consumer does. ``0`` is the sink; a cycle or invalid
    link ends the walk.
    """
    operated = set(operated_codes)

    def _is_runofriver(code: int) -> bool:
        if code not in effective.base.index:
            return False
        return str(effective.base.loc[code, "tipo_regulacao"]).strip() == "D"

    runofriver = {code for code in operated if _is_runofriver(code)}

    # A reservoir bounds the D plants below it, not above it: walking
    # downstream from every operated reservoir, every D plant on that path is
    # downstream of a reservoir and must keep its floor.
    below_a_reservoir: set[int] = set()
    for code in operated:
        if not is_reservoir(effective, code):
            continue
        visited = {code}
        current = effective.downstream_plant(code, 0)
        while (
            current != 0 and current not in visited and current in effective.base.index
        ):
            visited.add(current)
            if current in runofriver:
                below_a_reservoir.add(current)
            current = effective.downstream_plant(current, 0)

    return runofriver - below_a_reservoir


@dataclass(frozen=True)
class CadastroResolutionReport:
    """Summary of the scalar ``AC`` overrides resolved by ``build_effective_cadastro``.

    ``applied`` maps each cadastro parameter to the count of distinct plant
    codes that received at least one in-horizon override; ``out_of_horizon``
    is the same tuple :func:`_read_scalar_overrides` returned, surfaced here
    for the caller instead of being silently dropped.
    """

    applied: Mapping[str, int]
    out_of_horizon: tuple[OutOfHorizon, ...]


def _require_cadastro_row(hidr: pd.DataFrame, code: int) -> None:
    """Raise ``ValueError`` if *code* has no row in the cadastro registry."""
    if code not in hidr.index:
        raise ValueError(
            f"AC override references plant code {code}, which is not in"
            " the cadastro registry"
        )


def build_effective_cadastro(
    dadger: Dadger,
    hidr: pd.DataFrame,
    calendar: Sequence[OperativeStage],
) -> tuple[EffectiveCadastro, CadastroResolutionReport]:
    """Assemble the per-stage-effective cadastro and its resolution report.

    Reads the scalar ``AC`` overrides from *dadger* via
    :func:`_read_scalar_overrides`, densifies each in-horizon ``(code,
    param)`` group against *hidr*'s base value with
    :func:`_forward_fill_series`, and reports how many distinct plants were
    touched per parameter. *hidr* is treated as an immutable base view and is
    never mutated; a ``(code, param)`` pair with no override is simply absent
    from the returned :class:`EffectiveCadastro`'s ``stage_varying`` map and
    falls through to *hidr* for every stage.

    Also builds the diversion-channel map from :func:`_read_diversion_overrides`
    for the union of {plants with a non-zero base ``desvio``} and {plants with
    an in-horizon ``AC DESVIO``}: the base ``desvio`` seeds the forward-fill
    (as ``DiversionChannel(downstream, limit=None)``, or ``None`` when
    ``desvio`` is zero), the resolved ``AC DESVIO`` records overlay it, and
    the result is stored densely per stage — sparsely, i.e. only for plants
    with a base or ``AC`` diversion. ``diversions`` is absent-by-default for
    every other plant, matching *stage_varying*'s sparsity discipline.

    Also builds the two machine-set maps from :func:`_read_machine_set_overrides`,
    each densified independently against *hidr*'s own per-conjunto columns:
    ``machine_conjunto_counts`` (sparse by plant code, seeded from
    ``numero_conjuntos_maquinas``) and ``machine_sets`` (sparse by
    ``(code, indice_conjunto)``, seeded from ``maquinas_conjunto_{k}``/
    ``potencia_nominal_conjunto_{k}``/``vazao_nominal_conjunto_{k}`` and
    zipped into a :class:`MachineSet` tuple). *hidr*'s per-conjunto columns
    are read only for the ``(code, k)`` pairs that actually carry an
    override — never scanned unconditionally — so a conjunto with, say, a
    ``NUMMAQ`` override but no ``POTEFE``/``VAZEFE`` override still gets its
    ``potencia``/``vazao`` from the ``hidr`` base at every stage.

    Also builds the ``cota_polynomials`` map from :func:`_read_polynomial_overrides`
    (the fifth, multi-row-per-plant shape): sparse by plant code, seeded from
    the base ``a{0..4}_volume_cota`` columns and densified the same way as
    every other per-stage series, one full 5-coefficient polynomial per
    stage rather than one scalar.

    Also builds ``downstream_links``/``inflow_gauges`` from
    :func:`_read_keyed_overrides`: sparse by plant
    code, seeded from the base ``codigo_usina_jusante``/``posto`` columns
    and densified the same way as every other per-stage series. Consumed by
    :meth:`EffectiveCadastro.downstream_plant`/``inflow_gauge`` — the
    cascade walk (:func:`~cobre_bridge.decomp.converters.hydro.entity.
    _downstream_operated`) and the incremental-inflow gauge attribution
    (:func:`~cobre_bridge.decomp.scenarios._incremental_context`) read one
    stage-representative (stage 0) value off these, never per-stage, so a
    temporal ``NUMJUS``/``NUMPOS`` is a tracked gap
    (:meth:`EffectiveCadastro.downstream_plant_varies`/``inflow_gauge_varies``),
    not a silent per-stage cascade. ``AC JUSENA``/``AC NPOSNW`` are
    deliberately **not** ingested here — no DECOMP consumer; see the
    deferred-fidelity warning in :mod:`cobre_bridge.decomp.converters.hydro`.

    Raises
    ------
    ValueError
        If an override references a plant *code* absent from ``hidr.index``
        — the registry has no cadastro row to override. Applies to the
        scalar overrides, the source plant of an ``AC DESVIO`` (the
        downstream plant it names is not validated here — that is M2.1's
        concern), and every machine-set override.
    """
    records, out_of_horizon = _read_scalar_overrides(dadger, calendar)
    diversion_records, diversion_out_of_horizon = _read_diversion_overrides(
        dadger, calendar
    )
    (
        numero_conjuntos_records,
        numero_maquinas_records,
        potencia_records,
        vazao_records,
        machine_out_of_horizon,
    ) = _read_machine_set_overrides(dadger, calendar)
    cota_records, cota_out_of_horizon = _read_polynomial_overrides(
        dadger, calendar, ACCOTVOL, "cota_volume"
    )
    topology_records, topology_out_of_horizon = _read_keyed_overrides(
        dadger,
        calendar,
        ACNUMJUS,
        "codigo_usina_jusante",
        "codigo_usina_jusante",
        int,
        _plant_code_key,
    )
    gauge_records, gauge_out_of_horizon = _read_keyed_overrides(
        dadger, calendar, ACNUMPOS, "codigo_posto", "posto", int, _plant_code_key
    )
    out_of_horizon = (
        out_of_horizon
        + diversion_out_of_horizon
        + machine_out_of_horizon
        + cota_out_of_horizon
        + topology_out_of_horizon
        + gauge_out_of_horizon
    )

    n_stages = len(calendar)
    stage_varying: dict[tuple[int, str], tuple[float, ...]] = {}
    applied: dict[str, int] = {}
    for (code, param), overrides in records.items():
        _require_cadastro_row(hidr, code)
        base_value = float(hidr.loc[code, param])
        stage_varying[(code, param)] = tuple(
            _forward_fill_series(base_value, overrides, n_stages)
        )
        applied[param] = applied.get(param, 0) + 1

    base_diversion_codes = {
        int(code)
        for code, desvio in zip(
            hidr.index.tolist(), hidr["desvio"].tolist(), strict=True
        )
        if float(desvio) != 0
    }
    diversions: dict[int, tuple[DiversionChannel | None, ...]] = {}
    for code in base_diversion_codes | set(diversion_records):
        _require_cadastro_row(hidr, code)
        base_desvio = float(hidr.loc[code, "desvio"])
        base_channel = (
            DiversionChannel(int(base_desvio), None) if base_desvio != 0 else None
        )
        diversion_overrides = diversion_records.get(code, [])
        diversions[code] = tuple(
            _forward_fill_series(base_channel, diversion_overrides, n_stages)
        )
    if diversion_records:
        applied["diversion"] = len(diversion_records)

    machine_conjunto_counts: dict[int, tuple[int, ...]] = {}
    for code, count_overrides in numero_conjuntos_records.items():
        _require_cadastro_row(hidr, code)
        base_count = int(hidr.loc[code, "numero_conjuntos_maquinas"])
        machine_conjunto_counts[code] = tuple(
            _forward_fill_series(base_count, count_overrides, n_stages)
        )
    if numero_conjuntos_records:
        applied["numero_conjuntos"] = len(numero_conjuntos_records)

    machine_sets: dict[tuple[int, int], tuple[MachineSet, ...]] = {}
    compound_keys = (
        set(numero_maquinas_records) | set(potencia_records) | set(vazao_records)
    )
    for code, conjunto in compound_keys:
        _require_cadastro_row(hidr, code)
        base_numero_maquinas = int(hidr.loc[code, f"maquinas_conjunto_{conjunto}"])
        base_potencia = float(hidr.loc[code, f"potencia_nominal_conjunto_{conjunto}"])
        base_vazao = float(hidr.loc[code, f"vazao_nominal_conjunto_{conjunto}"])
        numero_maquinas_series = _forward_fill_series(
            base_numero_maquinas,
            numero_maquinas_records.get((code, conjunto), []),
            n_stages,
        )
        potencia_series = _forward_fill_series(
            base_potencia, potencia_records.get((code, conjunto), []), n_stages
        )
        vazao_series = _forward_fill_series(
            base_vazao, vazao_records.get((code, conjunto), []), n_stages
        )
        machine_sets[(code, conjunto)] = tuple(
            MachineSet(numero_maquinas_series[s], potencia_series[s], vazao_series[s])
            for s in range(n_stages)
        )
    if numero_maquinas_records:
        applied["numero_maquinas"] = len(numero_maquinas_records)
    if potencia_records:
        applied["potencia"] = len(potencia_records)
    if vazao_records:
        applied["vazao"] = len(vazao_records)

    cota_polynomials: dict[int, tuple[tuple[float, ...], ...]] = {}
    for code, cota_overrides in cota_records.items():
        _require_cadastro_row(hidr, code)
        base_coefficients = tuple(
            float(hidr.loc[code, f"a{i}_volume_cota"]) for i in range(5)
        )
        cota_polynomials[code] = tuple(
            _forward_fill_series(base_coefficients, cota_overrides, n_stages)
        )
    if cota_records:
        applied["cota_volume"] = len(cota_records)

    downstream_links: dict[int, tuple[int, ...]] = {}
    for code, topology_overrides in topology_records.items():
        _require_cadastro_row(hidr, code)
        base_downstream = int(hidr.loc[code, "codigo_usina_jusante"])
        downstream_links[code] = tuple(
            _forward_fill_series(base_downstream, topology_overrides, n_stages)
        )
    if topology_records:
        applied["codigo_usina_jusante"] = len(topology_records)

    inflow_gauges: dict[int, tuple[int, ...]] = {}
    for code, gauge_overrides in gauge_records.items():
        _require_cadastro_row(hidr, code)
        base_gauge = int(hidr.loc[code, "posto"])
        inflow_gauges[code] = tuple(
            _forward_fill_series(base_gauge, gauge_overrides, n_stages)
        )
    if gauge_records:
        applied["posto"] = len(gauge_records)

    effective = EffectiveCadastro(
        base=hidr,
        n_stages=n_stages,
        stage_varying=stage_varying,
        diversions=diversions,
        machine_conjunto_counts=machine_conjunto_counts,
        machine_sets=machine_sets,
        cota_polynomials=cota_polynomials,
        downstream_links=downstream_links,
        inflow_gauges=inflow_gauges,
    )
    report = CadastroResolutionReport(
        applied=applied, out_of_horizon=tuple(out_of_horizon)
    )
    return effective, report
