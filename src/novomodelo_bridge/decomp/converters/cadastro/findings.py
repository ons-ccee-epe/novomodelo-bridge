"""The ``AC`` register findings ``check decomp`` and ``convert decomp`` both raise."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from idecomp.decomp.modelos import dadger as _dadger_models

from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    format_stage_ranges,
)
from novomodelo_bridge.decomp.converters.cadastro.overrides import (
    APPLIED_AC_CLASSES,
    AcRegisters,
    ac_frame,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from novomodelo_bridge.decomp.converters.cadastro.effective import (
        EffectiveCadastro,
    )

_CATEGORY = "Cadastro overrides"

#: Every idecomp ``AC`` register class, discovered by reflection so a register
#: a later idecomp release adds lands in :data:`DEFERRED_AC_CLASSES` with no
#: edit here.
ALL_AC_CLASSES: frozenset[type] = frozenset(
    obj
    for name, obj in vars(_dadger_models).items()
    if name.startswith("AC") and inspect.isclass(obj)
)

#: The ``AC`` registers the conversion does not read.
DEFERRED_AC_CLASSES: frozenset[type] = ALL_AC_CLASSES - APPLIED_AC_CLASSES


def ac_registers_present(dadger: AcRegisters, classes: Iterable[type]) -> list[type]:
    """The *classes* with at least one row in *dadger*, name-sorted."""
    present = [
        cls
        for cls in classes
        if (frame := ac_frame(dadger, cls)) is not None and not frame.empty
    ]
    return sorted(present, key=lambda cls: cls.__name__)


def ac_register_table(classes: Sequence[type]) -> DiagnosticTable:
    """One row per ``AC`` register, named as the manual writes it."""
    return DiagnosticTable(
        columns=["AC register"],
        rows=[[cls.__name__.removeprefix("AC")] for cls in classes],
    )


def ac_overrides_deferred_diagnostic(dadger: AcRegisters) -> Diagnostic | None:
    """The ``decomp-ac-overrides-deferred`` WARNING, or ``None`` when *dadger*
    declares no register of :data:`DEFERRED_AC_CLASSES`."""
    present = ac_registers_present(dadger, DEFERRED_AC_CLASSES)
    if not present:
        return None
    return Diagnostic(
        code="decomp-ac-overrides-deferred",
        severity=Severity.WARNING,
        category=_CATEGORY,
        title=f"AC registers not converted ({len(present)})",
        summary=(
            "The deck declares the listed AC registers, which the conversion "
            "does not read. The case keeps the hidr.dat value of each field "
            "they change, so its results can differ from DECOMP's."
        ),
        table=ac_register_table(present),
    )


def topology_varies_diagnostic(effective: EffectiveCadastro) -> Diagnostic | None:
    """The ``decomp-ac-topology-varies`` WARNING, or ``None`` when no ``AC
    NUMJUS``, ``NUMPOS`` or ``JUSENA`` register changes value between stages.

    The case carries one cascade for the whole horizon, so every cascade walk
    reads these registers at the first stage.
    """
    registers: tuple[tuple[str, Mapping[int, Sequence[int | None]]], ...] = (
        ("AC NUMJUS", effective.downstream_links),
        ("AC NUMPOS", effective.inflow_gauges),
        ("AC JUSENA", effective.energy_links),
    )
    rows: list[list[object]] = []
    for register, series_by_code in registers:
        for code in sorted(series_by_code):
            series = series_by_code[code]
            changed = [
                stage for stage, value in enumerate(series) if value != series[0]
            ]
            if not changed:
                continue
            name = (
                str(effective.base.loc[code, "nome_usina"]).strip()
                if code in effective.base.index
                else ""
            )
            used = "none" if series[0] is None else series[0]
            rows.append([code, name, register, used, format_stage_ranges(changed)])
    if not rows:
        return None
    return Diagnostic(
        code="decomp-ac-topology-varies",
        severity=Severity.WARNING,
        category=_CATEGORY,
        title=f"Cascade registers that change between stages ({len(rows)})",
        summary=(
            "The listed AC registers take a different value on later stages, "
            "but the converted case has one cascade for the whole horizon and "
            "uses the first stage's value throughout, so on the listed stages "
            "the water, inflow or stored-energy routing differs from DECOMP's."
        ),
        table=DiagnosticTable(
            columns=["Code", "Name", "Register", "Used", "Differs on stages"],
            rows=rows,
            justify=["right", "left", "left", "right", "left"],
            caption="Stage ids are 0-based.",
        ),
    )
