"""AC-override ingestion: typed register readers, records, and class registries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd
from idecomp.decomp.modelos.dadger import (
    ACALTEFE,
    ACCOTVOL,
    ACDESVIO,
    ACJUSMED,
    ACNUMCON,
    ACNUMJUS,
    ACNUMMAQ,
    ACNUMPOS,
    ACPERHID,
    ACPOTEFE,
    ACPROESP,
    ACVAZEFE,
    ACVAZMIN,
    ACVMDESV,
    ACVOLMAX,
    ACVOLMIN,
    ACVSVERT,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from idecomp.decomp import Dadger

    from cobre_bridge.decomp.temporal import OperativeStage

from cobre_bridge.decomp.converters.cadastro.stage_resolution import (
    _parse_month,
    resolve_effective_stage,
)


@dataclass(frozen=True)
class _ScalarAcSpec:
    """One scalar single-value ``AC`` mnemonic to ingest.

    ``ac_class`` is the idecomp typed register class passed as
    ``modificacao=`` to ``Dadger.ac(...)``; ``value_column`` is the frame
    column carrying the new value; ``param`` is the target cadastro column
    the override applies to.
    """

    ac_class: type
    value_column: str
    param: str


#: The scalar single-value ``AC`` mnemonics ingested by
#: :func:`_read_scalar_overrides`. Module scope so every consumer of the
#: scalar-override machinery shares one registry. Membership is by shape:
#: the diversion volume thresholds (`VMDESV`/`VSVERT`) and the head/
#: productivity mnemonics (`PROESP`/`PERHID`/`JUSMED`) are scalar and belong
#: here; the diversion channel (`DESVIO`) and the `COTVOL` polynomial are
#: multi-row and get their own readers (:func:`_read_diversion_overrides`,
#: :func:`_read_polynomial_overrides`); the `int`-valued topology/gauge pair
#: (`NUMJUS`/`NUMPOS`) goes through :func:`_read_keyed_overrides` and does
#: not fit this `float`-typed tuple.
_SCALAR_AC_SPECS: tuple[_ScalarAcSpec, ...] = (
    _ScalarAcSpec(ACVOLMIN, "volume", "volume_minimo"),
    _ScalarAcSpec(ACVOLMAX, "volume", "volume_maximo"),
    _ScalarAcSpec(ACVAZMIN, "vazao", "vazao_minima_historica"),
    _ScalarAcSpec(ACVMDESV, "volume", "volume_desvio"),
    _ScalarAcSpec(ACVSVERT, "volume", "volume_vertedouro"),
    _ScalarAcSpec(ACPROESP, "produtibilidade", "produtibilidade_especifica"),
    _ScalarAcSpec(ACPERHID, "coeficiente", "perdas"),
    _ScalarAcSpec(ACJUSMED, "cota", "canal_fuga_medio"),
)


#: Every AC register the resolver ingests AND applies to a live consumer —
#: the single source of truth `check decomp` diffs the idecomp AC universe
#: against (see `cobre_bridge.decomp.preflight._ac_coverage`). The scalar
#: portion is derived from `_SCALAR_AC_SPECS`; any new NON-scalar reader
#: wired into `build_effective_cadastro` must add its class to this
#: frozenset too, or `check decomp` will misreport it as deferred.
APPLIED_AC_CLASSES: frozenset[type] = frozenset(
    spec.ac_class for spec in _SCALAR_AC_SPECS
) | frozenset(
    {
        ACDESVIO,
        ACNUMCON,
        ACNUMMAQ,
        ACPOTEFE,
        ACVAZEFE,
        ACCOTVOL,
        ACNUMJUS,
        ACNUMPOS,
    }
)

#: AC registers idecomp models but exposes NO value accessor for, so the
#: resolver cannot ingest them at all — reported distinctly from "deferred
#: (has a value but no consumer)". `ALTEFE` is the sole member (idecomp
#: 1.13.0); see the `TRACKED COBRE-GAP WORKAROUND` in
#: `cobre_bridge.decomp.converters.hydro.entity.convert_hydros`.
UNINGESTABLE_AC_CLASSES: frozenset[type] = frozenset({ACALTEFE})


@dataclass(frozen=True)
class OutOfHorizon:
    """An ``AC`` override whose effective date falls after the calendar horizon.

    Reported rather than dropped: the cadastro assembly surfaces these so a
    deck that overrides a plant's cadastro beyond the study horizon is never
    silently ignored.
    """

    code: int
    param: str
    mes: int
    ano: int


def _out_of_horizon_record(
    code: int,
    mes: str | int | float | None,
    ano: int | float | None,
    param: str,
    calendar: Sequence[OperativeStage],
) -> OutOfHorizon:
    """Build the :class:`OutOfHorizon` record for an override that resolved
    past the calendar horizon.

    Shared by :func:`_read_scalar_overrides` and
    :func:`_read_diversion_overrides`, which report an out-of-horizon override
    the same way, differing only in the *param* label.
    """
    resolved_ano = (
        calendar[0].start_date.year if ano is None or pd.isna(ano) else int(ano)
    )
    month = _parse_month(mes)
    assert month is not None  # eff is None only for a non-blank mes
    return OutOfHorizon(code, param, month, resolved_ano)


def _read_scalar_overrides(
    dadger: Dadger,
    calendar: Sequence[OperativeStage],
) -> tuple[dict[tuple[int, str], list[tuple[int, float]]], list[OutOfHorizon]]:
    """Ingest the scalar single-value ``AC`` overrides via idecomp's typed accessor.

    For each registered :class:`_ScalarAcSpec`, reads
    ``dadger.ac(codigo_usina=None, modificacao=spec.ac_class, df=True)`` and
    resolves every row's ``(mes, semana, ano)`` triple to an effective stage
    via :func:`resolve_effective_stage`. Rows that resolve within the horizon
    are grouped by ``(code, param)``; rows that resolve past the horizon are
    reported in the returned out-of-horizon list instead of being dropped.

    Raises
    ------
    KeyError
        If an accessor frame is missing an expected column (a malformed
        idecomp frame is a hard error, never a silent default).
    """
    records: dict[tuple[int, str], list[tuple[int, float]]] = {}
    out_of_horizon: list[OutOfHorizon] = []
    for spec in _SCALAR_AC_SPECS:
        table = dadger.ac(codigo_usina=None, modificacao=spec.ac_class, df=True)
        if not isinstance(table, pd.DataFrame) or table.empty:
            continue
        for _, row in table.iterrows():
            code = int(row["codigo_usina"])
            value = float(row[spec.value_column])
            mes = row["mes"]
            eff = resolve_effective_stage(mes, row["semana"], row["ano"], calendar)
            if eff is None:
                out_of_horizon.append(
                    _out_of_horizon_record(code, mes, row["ano"], spec.param, calendar)
                )
                continue
            records.setdefault((code, spec.param), []).append((eff, value))
    return records, out_of_horizon


@dataclass(frozen=True)
class DiversionChannel:
    """The diversion channel (canal de desvio) active for a plant at a stage.

    ``downstream`` is the ``codigo_usina`` of the plant that receives the
    diverted water; ``limit`` is the channel's flow limit in m³/s, or
    ``None`` for a base-declared diversion, which carries no explicit
    limit — only an ``AC DESVIO`` override supplies one.
    """

    downstream: int
    limit: float | None


def _read_diversion_overrides(
    dadger: Dadger,
    calendar: Sequence[OperativeStage],
) -> tuple[dict[int, list[tuple[int, DiversionChannel]]], list[OutOfHorizon]]:
    """Ingest the ``AC DESVIO`` diversion-channel override via idecomp's typed accessor.

    Mirrors :func:`_read_scalar_overrides` one-for-one: reads
    ``dadger.ac(codigo_usina=None, modificacao=ACDESVIO, df=True)`` and
    resolves every row's ``(mes, semana, ano)`` triple to an effective stage
    via :func:`resolve_effective_stage`. Rows that resolve within the horizon
    are grouped by the source plant *code* as ``(eff_stage, DiversionChannel(
    downstream, limit))``; rows that resolve past the horizon are reported
    (``param="diversion"``) in the returned out-of-horizon list instead of
    being dropped.

    Raises
    ------
    KeyError
        If the accessor frame is missing an expected column (a malformed
        idecomp frame is a hard error, never a silent default).
    """
    records: dict[int, list[tuple[int, DiversionChannel]]] = {}
    out_of_horizon: list[OutOfHorizon] = []
    table = dadger.ac(codigo_usina=None, modificacao=ACDESVIO, df=True)
    if not isinstance(table, pd.DataFrame) or table.empty:
        return records, out_of_horizon
    for _, row in table.iterrows():
        code = int(row["codigo_usina"])
        channel = DiversionChannel(
            int(row["codigo_usina_jusante"]), float(row["limite_vazao"])
        )
        mes = row["mes"]
        eff = resolve_effective_stage(mes, row["semana"], row["ano"], calendar)
        if eff is None:
            out_of_horizon.append(
                _out_of_horizon_record(code, mes, row["ano"], "diversion", calendar)
            )
            continue
        records.setdefault(code, []).append((eff, channel))
    return records, out_of_horizon


@dataclass(frozen=True)
class MachineSet:
    """One conjunto's effective machine configuration at a stage.

    ``numero_maquinas`` is the conjunto's machine-unit count; ``potencia``
    and ``vazao`` are its per-unit rated power (MW) and rated flow (m³/s) —
    the same three quantities the source model's ``NUMMAQ``/``POTEFE``/
    ``VAZEFE`` registers carry, densified per stage by
    :func:`build_effective_cadastro`.
    """

    numero_maquinas: int
    potencia: float
    vazao: float


def _plant_code_key(row: Any) -> int:
    """Group-key for the plant-keyed single-value ``AC`` mnemonics —
    ``codigo_usina`` alone (``NUMJUS``/``NUMPOS``)."""
    return int(row["codigo_usina"])


def _conjunto_key(row: Any) -> tuple[int, int]:
    """Group-key for the compound-keyed machine-set mnemonics —
    ``(codigo_usina, indice_conjunto)`` (``NUMMAQ``/``POTEFE``/``VAZEFE``)."""
    return int(row["codigo_usina"]), int(row["indice_conjunto"])


def _read_keyed_overrides[K, T](
    dadger: Dadger,
    calendar: Sequence[OperativeStage],
    ac_class: type,
    value_column: str,
    param: str,
    value_caster: Callable[[Any], T],
    key: Callable[[Any], K],
) -> tuple[dict[K, list[tuple[int, T]]], list[OutOfHorizon]]:
    """Ingest one single-value ``AC`` mnemonic, grouped by *key*.

    The shared body for both single-value override shapes that differ only in
    their grouping key: the compound-key machine-set mnemonics
    (``NUMMAQ``/``POTEFE``/``VAZEFE``, grouped by ``(codigo_usina,
    indice_conjunto)`` via :func:`_conjunto_key`) and the plant-keyed
    topology/gauge mnemonics (``NUMJUS``/``NUMPOS``, grouped by
    ``codigo_usina`` via :func:`_plant_code_key`). Sharing one loop keeps
    these from repeating near-identical bodies, the same way
    :func:`_forward_fill_series` is reused across the ``int``/``float``
    per-stage series it densifies. ``NUMCON`` keeps its own plant-keyed loop
    in :func:`_read_machine_set_overrides`, and the multi-row polynomial shape
    (:func:`_read_polynomial_overrides`) is genuinely different — neither is
    folded in here. Does not reuse :func:`_read_scalar_overrides`'s
    ``_SCALAR_AC_SPECS`` tuple: that tuple is uniformly ``float``-typed, while
    these mnemonics span ``int`` (machine counts, plant/gauge codes) and
    ``float`` (rated power/flow) via *value_caster*.

    Reads ``dadger.ac(codigo_usina=None, modificacao=ac_class, df=True)`` and
    resolves every row's ``(mes, semana, ano)`` triple to an effective stage
    via :func:`resolve_effective_stage`. In-horizon rows are grouped by
    ``key(row)`` as ``(eff_stage, value_caster(row[value_column]))``; rows
    that resolve past the horizon are reported (with *param* as the label) in
    the returned out-of-horizon list instead of being dropped.

    Raises
    ------
    KeyError
        If the accessor frame is missing an expected column (a malformed
        idecomp frame is a hard error, never a silent default).
    """
    records: dict[K, list[tuple[int, T]]] = {}
    out_of_horizon: list[OutOfHorizon] = []
    table = dadger.ac(codigo_usina=None, modificacao=ac_class, df=True)
    if not isinstance(table, pd.DataFrame) or table.empty:
        return records, out_of_horizon
    for _, row in table.iterrows():
        code = int(row["codigo_usina"])
        mes = row["mes"]
        eff = resolve_effective_stage(mes, row["semana"], row["ano"], calendar)
        if eff is None:
            out_of_horizon.append(
                _out_of_horizon_record(code, mes, row["ano"], param, calendar)
            )
            continue
        records.setdefault(key(row), []).append((eff, value_caster(row[value_column])))
    return records, out_of_horizon


def _read_machine_set_overrides(
    dadger: Dadger,
    calendar: Sequence[OperativeStage],
) -> tuple[
    dict[int, list[tuple[int, int]]],
    dict[tuple[int, int], list[tuple[int, int]]],
    dict[tuple[int, int], list[tuple[int, float]]],
    dict[tuple[int, int], list[tuple[int, float]]],
    list[OutOfHorizon],
]:
    """Ingest the machine-set ``AC`` overrides via idecomp's typed accessor.

    Covers the third distinct override shape (after the scalar single-value
    mnemonics and the non-scalar diversion channel): ``NUMCON`` is
    **plant-keyed** (one conjunto count per ``codigo_usina``), while
    ``NUMMAQ``/``POTEFE``/``VAZEFE`` are **compound-keyed** by
    ``(codigo_usina, indice_conjunto)`` — a conjunto's machine count, rated
    power, and rated flow. ``NUMCON`` is read by its own plant-keyed loop
    (its frame carries no ``indice_conjunto`` column); the other three are
    read through :func:`_read_keyed_overrides` with :func:`_conjunto_key`, one
    call per mnemonic.
    Every row's ``(mes, semana, ano)`` triple is resolved to an effective
    stage via :func:`resolve_effective_stage`; in-horizon rows are grouped by
    key, past-horizon rows are reported (param labels ``"numero_conjuntos"``,
    ``"numero_maquinas"``, ``"potencia"``, ``"vazao"``) in the returned
    out-of-horizon list instead of being dropped. Reads a frame only when
    ``isinstance(table, pd.DataFrame) and not table.empty``, mirroring
    :func:`_read_scalar_overrides`/:func:`_read_diversion_overrides`, so a
    ``None``/empty frame contributes nothing.

    A fifth machine-configuration mnemonic, ``ALTEFE`` (effective head), is
    deliberately **not** ingested here: the installed idecomp accessor
    exposes no value property on it, only the identifying/timing columns, so
    its ``df=True`` frame carries nothing to consume. Do not import or read
    it without a value accessor to back it.

    Raises
    ------
    KeyError
        If an accessor frame is missing an expected column (a malformed
        idecomp frame is a hard error, never a silent default).
    """
    numero_conjuntos: dict[int, list[tuple[int, int]]] = {}
    out_of_horizon: list[OutOfHorizon] = []
    table = dadger.ac(codigo_usina=None, modificacao=ACNUMCON, df=True)
    if isinstance(table, pd.DataFrame) and not table.empty:
        for _, row in table.iterrows():
            code = int(row["codigo_usina"])
            mes = row["mes"]
            eff = resolve_effective_stage(mes, row["semana"], row["ano"], calendar)
            if eff is None:
                out_of_horizon.append(
                    _out_of_horizon_record(
                        code, mes, row["ano"], "numero_conjuntos", calendar
                    )
                )
                continue
            numero_conjuntos.setdefault(code, []).append(
                (eff, int(row["numero_conjuntos"]))
            )

    numero_maquinas, maquinas_out_of_horizon = _read_keyed_overrides(
        dadger,
        calendar,
        ACNUMMAQ,
        "numero_maquinas",
        "numero_maquinas",
        int,
        _conjunto_key,
    )
    potencia, potencia_out_of_horizon = _read_keyed_overrides(
        dadger, calendar, ACPOTEFE, "potencia", "potencia", float, _conjunto_key
    )
    vazao, vazao_out_of_horizon = _read_keyed_overrides(
        dadger, calendar, ACVAZEFE, "vazao", "vazao", float, _conjunto_key
    )
    out_of_horizon.extend(maquinas_out_of_horizon)
    out_of_horizon.extend(potencia_out_of_horizon)
    out_of_horizon.extend(vazao_out_of_horizon)

    return numero_conjuntos, numero_maquinas, potencia, vazao, out_of_horizon


def _read_polynomial_overrides(
    dadger: Dadger,
    calendar: Sequence[OperativeStage],
    ac_class: type,
    param: str,
) -> tuple[dict[int, list[tuple[int, tuple[float, ...]]]], list[OutOfHorizon]]:
    """Ingest a multi-row-per-plant polynomial ``AC`` mnemonic (the fifth shape).

    Unlike the scalar, diversion, and compound-key shapes above, one
    *effective* override here is not a single row: ``AC COTVOL`` declares the
    plant's full 5-coefficient forebay-cota polynomial as up to five separate
    rows, one per **1-based** ``ordem`` (1..5, the coefficients a0..a4 — the
    reader normalises it to a 0-based tuple index), all sharing the same
    ``(codigo_usina, mes, semana, ano)`` triple. Reads
    ``dadger.ac(codigo_usina=None,
    modificacao=ac_class, df=True)`` and resolves **every row's own**
    ``(mes, semana, ano)`` triple to an effective stage via
    :func:`resolve_effective_stage`, then groups rows by ``(codigo_usina,
    effective_stage)`` rather than by the raw date fields themselves — a
    blank-date group's ``mes``/``semana``/``ano`` are typically distinct NaN
    objects that compare unequal to each other (``nan != nan``), which would
    silently fragment one group into several under a raw-field dict key;
    resolving first and grouping by the resolved stage sidesteps that
    entirely. Within a group, an ``ordem`` absent from the rows defaults its
    coefficient to ``0.0`` — the one documented default in this reader, never
    applied to an ``ordem`` that *is* present. Rows that resolve within the
    horizon are grouped by *codigo_usina* as ``(eff_stage, coefficients)``,
    *coefficients* ordered ``ordem 0..4``; rows whose date resolves past the
    horizon are reported (with *param* as the label) in the returned
    out-of-horizon list instead of being dropped.

    Raises
    ------
    KeyError
        If the accessor frame is missing an expected column (a malformed
        idecomp frame is a hard error, never a silent default).
    ValueError
        If a row's 1-based ``ordem`` is outside 1..5 — an out-of-range
        coefficient index is a malformed register, never silently dropped.
    """
    out_of_horizon: list[OutOfHorizon] = []
    table = dadger.ac(codigo_usina=None, modificacao=ac_class, df=True)
    if not isinstance(table, pd.DataFrame) or table.empty:
        return {}, out_of_horizon

    coeffs_by_group: dict[tuple[int, int], dict[int, float]] = {}
    for _, row in table.iterrows():
        code = int(row["codigo_usina"])
        # ``AC COTVOL`` numbers its coefficients 1..5 (a0..a4) and idecomp
        # surfaces that raw 1-based ``ordem`` verbatim. Normalise it to the
        # 0-based index the coefficient tuple below is read at: leaving the
        # 1-based value in place shifts every coefficient up one slot, which
        # silently zeroes the a0 constant term and drops a4 — for a
        # run-of-river plant whose override is a single constant forebay cota
        # (a0), that turns a fixed level of, say, 90 m into ``90·volume``.
        coeff_index = int(row["ordem"]) - 1
        if not 0 <= coeff_index < 5:
            raise ValueError(
                f"{param}: coefficient order {int(row['ordem'])} for plant "
                f"{code} is outside the expected 1..5 range"
            )
        coeficiente = float(row["coeficiente"])
        mes = row["mes"]
        eff = resolve_effective_stage(mes, row["semana"], row["ano"], calendar)
        if eff is None:
            out_of_horizon.append(
                _out_of_horizon_record(code, mes, row["ano"], param, calendar)
            )
            continue
        coeffs_by_group.setdefault((code, eff), {})[coeff_index] = coeficiente

    records: dict[int, list[tuple[int, tuple[float, ...]]]] = {}
    for (code, eff), coeffs_by_ordem in coeffs_by_group.items():
        coefficients = tuple(coeffs_by_ordem.get(i, 0.0) for i in range(5))
        records.setdefault(code, []).append((eff, coefficients))

    return records, out_of_horizon


def _forward_fill_series[T](
    base_value: T,
    records: Sequence[tuple[int, T]],
    n_stages: int,
) -> list[T]:
    """Densify a sparse set of effective-stage overrides into a dense series.

    *records* is a possibly-unordered, possibly-empty sequence of
    ``(effective_stage, value)`` pairs. The value at stage ``s`` is the value
    of the last record whose ``effective_stage <= s``, else *base_value*. A
    record at stage ``0`` therefore overwrites the base for every stage
    (permanent semantics); later records supersede earlier ones. *records*
    is not mutated. Generic over the value type ``T`` so both the scalar
    ``float`` overrides and the non-scalar ``DiversionChannel | None``
    overrides share this one densification helper.
    """
    ordered = sorted(records, key=lambda record: record[0])
    series: list[T] = []
    current = base_value
    next_index = 0
    for stage in range(n_stages):
        while next_index < len(ordered) and ordered[next_index][0] <= stage:
            current = ordered[next_index][1]
            next_index += 1
        series.append(current)
    return series
