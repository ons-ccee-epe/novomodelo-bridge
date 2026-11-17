"""The tailrace lateral flows of a DECOMP deck, resolved from its ``VL``, ``VU``
and ``VA`` records.

:func:`resolve_tailrace_flow` is pure: it returns each influenced plant's
declaration in DECOMP plant codes together with every finding, so the
converter and ``check decomp`` report the same diagnostics.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Protocol

from novomodelo_bridge.core.diagnostics import Diagnostic, DiagnosticTable, Severity
from novomodelo_bridge.core.errors import FieldParseError
from novomodelo_bridge.core.pandas_utils import as_float, as_frame, is_na

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping, Sequence

    from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro

_LOG = logging.getLogger(__name__)

_CATEGORY = "Hydro production model"
_REGISTER_ORDER = ("VL", "VU", "VA")
_VL_COEFFICIENTS = tuple(f"coeficiente_a{i}" for i in range(5))


class TailraceRegisters(Protocol):
    def vl(self, *, df: bool) -> object: ...

    def vu(self, *, df: bool) -> object: ...

    def va(self, *, df: bool) -> object: ...


@dataclass(frozen=True)
class ResolvedTailraceFlow:
    """One plant's tailrace lateral-flow declaration, in DECOMP plant codes.

    Influences are ``(plant code, factor)`` pairs sorted by code, one per
    plant. ``outflow_influences`` never holds the plant itself;
    ``incremental_inflow_influences`` may, when a gauge maps to the plant.
    ``own_outflow_factor`` includes the factors of VU records naming the plant
    itself. ``polynomial`` is the VL ``a0..a4`` when any is non-zero and no
    polinjus curve applies.
    """

    own_outflow_factor: float
    outflow_influences: tuple[tuple[int, float], ...]
    incremental_inflow_influences: tuple[tuple[int, float], ...]
    polynomial: tuple[float, ...] | None


@dataclass(frozen=True)
class TailraceFlowResolution:
    """The resolved declarations and findings of a deck.

    ``declarations`` holds only plants with at least one kept record.
    ``records`` lists every VL, VU and VA record read as ``(register,
    influenced plant code)``, VL first, each register in deck order.
    ``diagnostics`` holds at most one diagnostic per code, sorted by code.
    """

    declarations: Mapping[int, ResolvedTailraceFlow]
    records: tuple[tuple[str, int], ...]
    diagnostics: tuple[Diagnostic, ...]


class _Record(NamedTuple):
    """A VL, VU or VA record; ``source`` is the influencing plant (VU), the
    influencing gauge (VA), and 0 for VL."""

    register: str
    plant: int
    source: int
    factor: float
    coefficients: tuple[float, ...]


def resolve_tailrace_flow(
    dadger: TailraceRegisters,
    effective: EffectiveCadastro,
    case_codes: Collection[int],
    piecewise_codes: Collection[int],
) -> TailraceFlowResolution:
    """Resolve the ``VL``, ``VU`` and ``VA`` records of *dadger* per influenced plant.

    *case_codes* are the operated plant codes and *piecewise_codes* the plants
    whose tailrace comes from a polinjus curve. The result does not depend on
    the order of the register rows, except that the first ``VL`` record of a
    plant wins.

    Raises
    ------
    FieldParseError
        When a record carries a negative or non-finite factor.
    """
    all_records = _read_records(dadger)
    in_case = frozenset(case_codes)
    no_spill = _plants_outside_tailrace(
        effective, {r.plant for r in all_records if r.plant in in_case}
    )

    spill_registers: dict[int, set[str]] = defaultdict(set)
    kept: list[_Record] = []
    for record in all_records:
        if record.plant not in in_case:
            _LOG.debug(
                "%s record of plant %d dropped: the plant is not in the case",
                record.register,
                record.plant,
            )
        elif record.plant in no_spill:
            spill_registers[record.plant].add(record.register)
        else:
            kept.append(record)

    first_vl: dict[int, _Record] = {}
    dropped_vl: dict[int, int] = defaultdict(int)
    self_factors: dict[int, list[float]] = defaultdict(list)
    outflow: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    incremental: dict[int, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    survivors: set[int] = set()
    unmapped: set[tuple[int, int]] = set()
    ambiguous: set[tuple[int, int, tuple[int, ...]]] = set()
    varying: set[tuple[int, int]] = set()
    plants_of_gauge = (
        _plants_by_gauge(effective, in_case)
        if any(r.register == "VA" for r in kept)
        else {}
    )

    for record in kept:
        plant = record.plant
        if record.register == "VL":
            if plant in first_vl:
                dropped_vl[plant] += 1
                continue
            first_vl[plant] = record
        elif record.register == "VU":
            influencer = record.source
            if influencer == plant:
                self_factors[plant].append(record.factor)
            elif influencer in in_case:
                outflow[plant][influencer].append(record.factor)
            else:
                _LOG.debug(
                    "VU record of plant %d dropped: influencing plant %s is not "
                    "in the case",
                    plant,
                    influencer,
                )
                continue
        else:
            gauge = record.source
            candidates = plants_of_gauge.get(gauge, ())
            if not candidates:
                unmapped.add((plant, gauge))
                continue
            if len(candidates) > 1:
                ambiguous.add((plant, gauge, candidates))
                continue
            (mapped,) = candidates
            if len(set(effective.inflow_gauges.get(mapped, ()))) > 1:
                varying.add((mapped, gauge))
            incremental[plant][mapped].append(record.factor)
        survivors.add(plant)

    shadowed: list[int] = []
    declarations: dict[int, ResolvedTailraceFlow] = {}
    for plant in sorted(survivors):
        vl = first_vl.get(plant)
        polynomial = None
        if vl is not None and any(c != 0 for c in vl.coefficients):
            if plant in piecewise_codes:
                shadowed.append(plant)
            else:
                polynomial = vl.coefficients
        own = 1.0 if vl is None else vl.factor
        for factor in sorted(self_factors.get(plant, ())):
            own += factor
        declarations[plant] = ResolvedTailraceFlow(
            own_outflow_factor=own,
            outflow_influences=_summed(outflow.get(plant, {})),
            incremental_inflow_influences=_summed(incremental.get(plant, {})),
            polynomial=polynomial,
        )

    def name(code: int) -> str:
        return str(effective.base.loc[code, "nome_usina"]).strip()

    findings = (
        (
            _spill_skipped_diagnostic,
            [
                [
                    plant,
                    name(plant),
                    ", ".join(
                        r for r in _REGISTER_ORDER if r in spill_registers[plant]
                    ),
                ]
                for plant in sorted(spill_registers)
            ],
        ),
        (
            _duplicate_vl_diagnostic,
            [[plant, name(plant), n] for plant, n in sorted(dropped_vl.items())],
        ),
        (
            _gauge_not_in_case_diagnostic,
            [[plant, name(plant), gauge] for plant, gauge in sorted(unmapped)],
        ),
        (
            _gauge_ambiguous_diagnostic,
            [
                [plant, name(plant), gauge, ", ".join(map(str, plants))]
                for plant, gauge, plants in sorted(ambiguous)
            ],
        ),
        (
            _gauge_varies_diagnostic,
            [[plant, name(plant), gauge] for plant, gauge in sorted(varying)],
        ),
        (
            _self_outflow_diagnostic,
            [
                [plant, name(plant), _total(factors)]
                for plant, factors in sorted(self_factors.items())
            ],
        ),
        (
            _duplicate_influence_diagnostic,
            sorted(
                [plant, name(plant), register, influencer, _total(factors)]
                for register, table in (("VU", outflow), ("VA", incremental))
                for plant, influences in table.items()
                for influencer, factors in influences.items()
                if len(factors) > 1
            ),
        ),
        (
            _polynomial_shadowed_diagnostic,
            [[plant, name(plant)] for plant in shadowed],
        ),
    )
    diagnostics = sorted(
        (build(rows) for build, rows in findings if rows), key=lambda d: d.code
    )
    return TailraceFlowResolution(
        declarations=declarations,
        records=tuple((r.register, r.plant) for r in all_records),
        diagnostics=tuple(diagnostics),
    )


def _read_records(dadger: TailraceRegisters) -> list[_Record]:
    records: list[_Record] = []
    vl = as_frame(dadger.vl(df=True))
    if vl is not None:
        for row in vl.to_dict("records"):
            plant = int(row["codigo_usina_influenciada"])
            records.append(
                _Record(
                    "VL",
                    plant,
                    0,
                    _factor("VL", plant, row["fator_impacto_defluencia"]),
                    tuple(
                        0.0 if is_na(row[c]) else float(row[c])
                        for c in _VL_COEFFICIENTS
                    ),
                )
            )
    vu = as_frame(dadger.vu(df=True))
    if vu is not None:
        for row in vu.to_dict("records"):
            plant = int(row["codigo_usina_influenciada"])
            records.append(
                _Record(
                    "VU",
                    plant,
                    int(row["codigo_usina_influenciadora"]),
                    _factor("VU", plant, row["fator_impacto_defluencia"]),
                    (),
                )
            )
    va = as_frame(dadger.va(df=True))
    if va is not None:
        for row in va.to_dict("records"):
            plant = int(row["codigo_usina_influenciada"])
            records.append(
                _Record(
                    "VA",
                    plant,
                    int(row["codigo_posto_influenciador"]),
                    _factor("VA", plant, row["fator_impacto_incremental"]),
                    (),
                )
            )
    return records


def _factor(register: str, plant: int, value: object) -> float:
    factor = math.nan if is_na(value) else as_float(value)
    if not math.isfinite(factor) or factor < 0:
        raise FieldParseError(
            f"{register} record of plant {plant}: the factor {factor} is not a "
            "finite, non-negative number"
        )
    return factor


def _plants_outside_tailrace(
    effective: EffectiveCadastro, codes: Iterable[int]
) -> set[int]:
    return {
        code
        for code in codes
        if any(
            effective.value(code, "influencia_vertimento_canal_fuga", stage) == 0
            for stage in range(effective.n_stages)
        )
    }


def _plants_by_gauge(
    effective: EffectiveCadastro, codes: Iterable[int]
) -> dict[int, tuple[int, ...]]:
    by_gauge: dict[int, list[int]] = defaultdict(list)
    for code in sorted(codes):
        by_gauge[effective.inflow_gauge(code, 0)].append(code)
    return {gauge: tuple(plants) for gauge, plants in by_gauge.items()}


def _total(factors: Iterable[float]) -> float:
    # sorted: float addition is not associative, so deck order must not reach the sum
    return sum(sorted(factors))


def _summed(influences: Mapping[int, Sequence[float]]) -> tuple[tuple[int, float], ...]:
    return tuple(
        (code, _total(factors)) for code, factors in sorted(influences.items())
    )


def _spill_skipped_diagnostic(rows: Sequence[Sequence[object]]) -> Diagnostic:
    """WARNING: the skipped records carry a term DECOMP applies."""
    return Diagnostic(
        code="tailrace-spill-outside-tailrace-skipped",
        severity=Severity.WARNING,
        category=_CATEGORY,
        title=f"Lateral flows skipped for plants that spill outside the tailrace ({len(rows)})",
        summary=(
            "The listed plants keep their spilled flow out of the tailrace, "
            "with a VERTJU of 0 in hidr.dat or in an AC VERTJU record at one "
            "stage or more. Their VL, VU and VA records are not converted, so "
            "their tailrace lacks the lateral flows DECOMP applies and the "
            "results differ from DECOMP."
        ),
        table=DiagnosticTable(
            columns=["Code", "Name", "Records"],
            rows=rows,
            justify=["right", "left", "left"],
        ),
    )


def _duplicate_vl_diagnostic(rows: Sequence[Sequence[object]]) -> Diagnostic:
    """WARNING: a dropped VL record can carry another factor or polynomial."""
    return Diagnostic(
        code="tailrace-duplicate-vl-dropped",
        severity=Severity.WARNING,
        category=_CATEGORY,
        title=f"Repeated VL records dropped ({len(rows)})",
        summary=(
            "The listed plants have more than one VL record. Only the first "
            "one in the deck is converted and the others are dropped, so the "
            "results can differ from DECOMP."
        ),
        table=DiagnosticTable(
            columns=["Code", "Name", "Dropped"],
            rows=rows,
            justify=["right", "left", "right"],
        ),
        remediation="Keep one VL record per plant in dadger.rvN.",
    )


def _gauge_not_in_case_diagnostic(rows: Sequence[Sequence[object]]) -> Diagnostic:
    """WARNING: the dropped VA record carries a term DECOMP applies."""
    return Diagnostic(
        code="tailrace-gauge-not-in-case",
        severity=Severity.WARNING,
        category=_CATEGORY,
        title=f"VA records dropped: inflow gauge not in the case ({len(rows)})",
        summary=(
            "The listed VA records name an inflow gauge that no operated "
            "plant has, in hidr.dat or through AC NUMPOS. The records are "
            "dropped, so the tailrace of the influenced plants lacks the "
            "incremental inflow DECOMP applies and the results differ from "
            "DECOMP."
        ),
        table=DiagnosticTable(
            columns=["Code", "Name", "Gauge"],
            rows=rows,
            justify=["right", "left", "right"],
        ),
    )


def _gauge_ambiguous_diagnostic(rows: Sequence[Sequence[object]]) -> Diagnostic:
    """WARNING: the dropped VA record carries a term DECOMP applies."""
    return Diagnostic(
        code="tailrace-gauge-ambiguous",
        severity=Severity.WARNING,
        category=_CATEGORY,
        title=f"VA records dropped: inflow gauge shared by several plants ({len(rows)})",
        summary=(
            "The listed VA records name an inflow gauge that several operated "
            "plants have, so the plant whose incremental inflow applies "
            "cannot be told. The records are dropped, so the tailrace of the "
            "influenced plants lacks the incremental inflow DECOMP applies "
            "and the results differ from DECOMP."
        ),
        table=DiagnosticTable(
            columns=["Code", "Name", "Gauge", "Plants"],
            rows=rows,
            justify=["right", "left", "right", "left"],
        ),
    )


def _gauge_varies_diagnostic(rows: Sequence[Sequence[object]]) -> Diagnostic:
    """WARNING: the incremental inflow follows the plant, not the gauge."""
    return Diagnostic(
        code="tailrace-gauge-varies",
        severity=Severity.WARNING,
        category=_CATEGORY,
        title=f"VA gauge converted at the first stage for plants whose gauge changes ({len(rows)})",
        summary=(
            "The listed plants have an inflow gauge that changes between "
            "stages through AC NUMPOS records, and a VA record names their "
            "gauge at the first stage. The VA term follows the plant, not "
            "the gauge, so the results differ from DECOMP in the stages "
            "where the gauge has changed."
        ),
        table=DiagnosticTable(
            columns=["Code", "Name", "Gauge"],
            rows=rows,
            justify=["right", "left", "right"],
            caption="The gauge is the one at the first stage.",
        ),
    )


def _self_outflow_diagnostic(rows: Sequence[Sequence[object]]) -> Diagnostic:
    """INFO: the folded factor multiplies the same outflow as the VL factor."""
    return Diagnostic(
        code="tailrace-self-outflow-folded",
        severity=Severity.INFO,
        category=_CATEGORY,
        title=f"VU records naming their own plant folded into the VL factor ({len(rows)})",
        summary=(
            "The listed plants have a VU record whose influencing plant is "
            "the plant itself. Its factor is added to the VL factor of the "
            "plant, or to 1.0 when the plant has no VL record, which raises "
            "the tailrace as in DECOMP because both factors multiply the "
            "outflow of the plant."
        ),
        table=DiagnosticTable(
            columns=["Code", "Name", "Folded factor"],
            rows=rows,
            justify=["right", "left", "right"],
        ),
    )


def _duplicate_influence_diagnostic(rows: Sequence[Sequence[object]]) -> Diagnostic:
    """INFO: the summed factor is exact because each term is linear in the flow."""
    return Diagnostic(
        code="tailrace-duplicate-influence-summed",
        severity=Severity.INFO,
        category=_CATEGORY,
        title=f"Repeated VU and VA influences summed ({len(rows)})",
        summary=(
            "The listed plants repeat an influencing plant in their VU "
            "records, or have VA gauges that map to the same plant. The "
            "factors are added into one influence, which raises the tailrace "
            "as in DECOMP because each term is linear in the influencing "
            "flow."
        ),
        table=DiagnosticTable(
            columns=["Code", "Name", "Register", "Influencer", "Summed factor"],
            rows=rows,
            justify=["right", "left", "left", "right", "right"],
            caption="For VA records, the influencer is the plant that has the gauge.",
        ),
    )


def _polynomial_shadowed_diagnostic(rows: Sequence[Sequence[object]]) -> Diagnostic:
    """WARNING: the tailrace uses the polinjus curve, not the VL polynomial."""
    return Diagnostic(
        code="tailrace-vl-polynomial-shadowed",
        severity=Severity.WARNING,
        category=_CATEGORY,
        title=f"VL tailrace polynomial ignored for plants with a polinjus curve ({len(rows)})",
        summary=(
            "The listed plants have a VL record with tailrace polynomial "
            "coefficients and also a tailrace curve in polinjus. The polinjus "
            "curve is converted and the VL polynomial is not, so the "
            "tailrace of these plants can differ from the one DECOMP "
            "computes."
        ),
        table=DiagnosticTable(
            columns=["Code", "Name"], rows=rows, justify=["right", "left"]
        ),
    )
