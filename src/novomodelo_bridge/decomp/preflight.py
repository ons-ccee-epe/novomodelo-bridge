"""``check decomp`` — validate a deck revision without converting it.

Answers one question cheaply: *would this deck convert, and what will the
conversion silently leave behind?* The checks are the subset of the
cross-input validation matrix that can be decided from the deck alone, so a
malformed calendar or an unusable scenario tree fails in milliseconds with
deck-side context instead of surfacing later as a solver diagnostic.

The deferred-feature inventory is deliberately part of the report. A deck
carrying anticipation records or a plant with per-group availability converts
fine today — but not *completely*, and the difference is invisible unless the
preflight names it.

The solver remains the authority on the converted case; this is a courtesy
mirror, never a substitute.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd
from idecomp.decomp.modelos import dadger as _dadger_models

from novomodelo_bridge.core.diagnostics import Diagnostic, DiagnosticTable, Severity
from novomodelo_bridge.core.errors import FieldParseError, diagnostic_from_exception
from novomodelo_bridge.core.preflight import (
    CheckItem,
    PreflightResult,
    PreflightVerdict,
    optional_input_advisory,
)
from novomodelo_bridge.decomp import constraint_registers
from novomodelo_bridge.decomp.converters.cadastro import (
    APPLIED_AC_CLASSES,
    UNINGESTABLE_AC_CLASSES,
)
from novomodelo_bridge.decomp.files import discover_decomp_files

if TYPE_CHECKING:
    from idecomp.decomp import Dadger, Vazoes

    from novomodelo_bridge.decomp.case import DecompCase
    from novomodelo_bridge.decomp.converters.cadastro import CadastroResolutionReport
    from novomodelo_bridge.decomp.files import DecompFiles
    from novomodelo_bridge.decomp.id_map import DecompIdMap
    from novomodelo_bridge.decomp.temporal import OperativeStage

_CONTEXT = "Preflight"

#: Every idecomp `AC` register class, discovered by reflection so a future
#: idecomp release adding one lands automatically in the deferred bucket
#: (the conservative default) with no edit here. idecomp is a hard
#: dependency (unlike the optional `novomodelo` wheel), so this module-scope
#: import/reflection is safe in every CI tier.
_ALL_AC_CLASSES: frozenset[type] = frozenset(
    obj
    for name, obj in vars(_dadger_models).items()
    if name.startswith("AC") and inspect.isclass(obj)
)

#: Per-stage scenario probabilities must sum to 1 within this absolute
#: tolerance — the deck writes them rounded to four decimals.
_PROBABILITY_ATOL = 1e-4

#: Relative tolerance on the block-factor identity ``Σ_b f_b·h_b = H``.
_FACTOR_RTOL = 1e-9


def _verdict(
    checks: list[CheckItem], diagnostics: list[Diagnostic]
) -> PreflightVerdict:
    if any(not check.passed for check in checks):
        return PreflightVerdict.WILL_NOT_CONVERT
    if any(diag.severity is Severity.WARNING for diag in diagnostics):
        return PreflightVerdict.WARNINGS
    return PreflightVerdict.OK


def _deferred(code: str, title: str, summary: str, remediation: str) -> Diagnostic:
    return Diagnostic(
        code=code,
        severity=Severity.WARNING,
        category=_CONTEXT,
        title=title,
        summary=summary,
        remediation=remediation,
    )


def _calendar_check(
    dadger: Dadger,
) -> tuple[CheckItem, list[OperativeStage], str | None]:
    """Build the operative calendar, turning its validation into one check."""
    from novomodelo_bridge.decomp.temporal import operative_calendar_from_dadger

    try:
        calendar = operative_calendar_from_dadger(dadger)
    except (ValueError, AttributeError, TypeError) as exc:
        return (
            CheckItem(
                label="Operative calendar (weekly walk, month-boundary close)",
                passed=False,
                detail=str(exc),
            ),
            [],
            str(exc),
        )
    span = f"{calendar[0].start_date} → {calendar[-1].end_date}"
    return (
        CheckItem(
            label="Operative calendar (weekly walk, month-boundary close)",
            passed=True,
            detail=f"{len(calendar)} stages, {span}",
        ),
        calendar,
        None,
    )


def _tree_checks(vazoes: Vazoes, calendar: list[OperativeStage]) -> list[CheckItem]:
    """Per-stage probability mass and the trunk-plus-terminal-fan shape gate."""
    from novomodelo_bridge.decomp.scenarios import convert_scenario_probabilities

    try:
        table = convert_scenario_probabilities(vazoes, calendar).to_pydict()
    except (ValueError, KeyError, AttributeError) as exc:
        return [
            CheckItem(
                label="Scenario tree probabilities",
                passed=False,
                detail=str(exc),
            )
        ]

    per_stage: dict[int, float] = {}
    counts: dict[int, int] = {}
    for stage_id, probability in zip(
        table["stage_id"], table["probability"], strict=True
    ):
        per_stage[stage_id] = per_stage.get(stage_id, 0.0) + probability
        counts[stage_id] = counts.get(stage_id, 0) + 1

    worst = max(
        (abs(total - 1.0) for total in per_stage.values()),
        default=0.0,
    )
    checks = [
        CheckItem(
            label="Scenario probabilities sum to 1 per stage",
            passed=worst <= _PROBABILITY_ATOL,
            detail=f"worst |Σp − 1| = {worst:.2e}",
        )
    ]

    # The training path enumerates a trunk that branches once, at the end. A
    # deck that fans earlier is a node-graph study and would be mis-modelled
    # silently as a wider terminal fan.
    ordered = [counts[stage] for stage in sorted(counts)]
    branching = [i for i, n in enumerate(ordered) if n > 1]
    fan_is_terminal = not branching or branching == [len(ordered) - 1]
    checks.append(
        CheckItem(
            label="Tree shape is a trunk with one terminal fan",
            passed=fan_is_terminal,
            detail=f"nodes per stage: {ordered}",
        )
    )
    return checks


def _load_factor_check(case: DecompCase, id_map: DecompIdMap) -> CheckItem:
    """The per-(bus, stage) identity ``Σ_b f_b·h_b = H`` (matrix row 17)."""
    from novomodelo_bridge.decomp.load import convert_load_factors

    try:
        document = convert_load_factors(case, id_map)
    except (ValueError, KeyError, AttributeError) as exc:
        return CheckItem(label="Load block factors", passed=False, detail=str(exc))

    calendar = case.calendar
    hours = {stage.index: list(stage.block_hours) for stage in calendar}
    worst = 0.0
    for entry in document.get("load_factors", []):
        stage_hours = hours.get(entry["stage_id"])
        if stage_hours is None:
            continue
        total = sum(
            factor["factor"] * hour
            for factor, hour in zip(entry["block_factors"], stage_hours, strict=False)
        )
        span = sum(stage_hours)
        worst = max(worst, abs(total - span) / span)
    return CheckItem(
        label="Load block factors reproduce the stage span",
        passed=worst <= _FACTOR_RTOL,
        detail=f"worst relative deviation {worst:.2e}",
    )


def _deferred_inventory(dadger: Dadger) -> list[Diagnostic]:
    """Name what the conversion will not carry, so it is never a surprise."""
    try:
        frame = dadger.ez(df=True)
    except (AttributeError, TypeError, ValueError):
        return []
    if frame is None or frame.empty:
        return []
    return [
        _deferred(
            "decomp-ez-ignored",
            "Coupling volume limit ignored",
            "The deck carries maximum-useful-volume records for the boundary "
            f"coupling; they are read and ignored by decision ({len(frame)} "
            "record(s)).",
            "Tracked in the conversion roadmap; no action needed to convert.",
        )
    ]


def _ac_present(dadger: Dadger, classes: frozenset[type]) -> list[type]:
    """The subset of *classes* present in *dadger*'s deck, name-sorted.

    A class is present iff its ``AC`` frame is a non-empty
    ``pd.DataFrame`` — mirrors the resolver's own guard
    (:func:`novomodelo_bridge.decomp.converters.cadastro.overrides.
    _read_scalar_overrides` and its siblings), so a ``None``/empty frame
    (an unregistered mnemonic, or an absent one) contributes nothing.
    """
    present = [
        cls
        for cls in classes
        # idecomp's stub declares codigo_usina: int, but None is its own
        # documented no-filter sentinel (scan every plant) -- passing a real
        # int here would narrow the scan to one usina.
        if isinstance(
            frame := dadger.ac(  # type: ignore[arg-type]
                codigo_usina=None, modificacao=cls, df=True
            ),
            pd.DataFrame,
        )
        and not frame.empty
    ]
    return sorted(present, key=lambda cls: cls.__name__)


def _ac_coverage(
    dadger: Dadger, report: CadastroResolutionReport
) -> tuple[list[CheckItem], list[Diagnostic]]:
    """Classify every idecomp ``AC`` class into applied/uningestable/deferred
    and report which of those buckets the deck actually exercises.

    A pure function of *dadger* (for the per-class deck-presence scan) and
    *report* (for ``out_of_horizon``) — no file I/O, no calendar, no
    ``hidr``. The three buckets are computed once by set arithmetic against
    the module-level :data:`_ALL_AC_CLASSES` reflection and the resolver's
    own :data:`~novomodelo_bridge.decomp.converters.cadastro.overrides.
    APPLIED_AC_CLASSES` / :data:`~novomodelo_bridge.decomp.converters.cadastro.
    overrides.UNINGESTABLE_AC_CLASSES` registries
    — enumerate-and-diff, never a hand-maintained list, so a newly-applied
    family automatically drops off the deferred bucket and a new idecomp
    class automatically lands in it.

    Always returns exactly one passing summary :class:`CheckItem` (coverage
    never blocks conversion) plus a diagnostic per bucket that has at least
    one class present in the deck: an ``INFO`` for the applied mnemonics
    (visibility only), a ``WARNING`` for the deferred mnemonics (has a
    value but no converter consumer), a distinct ``WARNING`` for ``ALTEFE``
    specifically (an idecomp accessor gap, not lumped with "deferred"), and
    a ``WARNING`` with a per-row detail table when *report* carries an
    out-of-horizon override.
    """
    applied = _ALL_AC_CLASSES & APPLIED_AC_CLASSES
    uningestable = _ALL_AC_CLASSES & UNINGESTABLE_AC_CLASSES
    deferred = _ALL_AC_CLASSES - APPLIED_AC_CLASSES - UNINGESTABLE_AC_CLASSES

    applied_present = _ac_present(dadger, applied)
    uningestable_present = _ac_present(dadger, uningestable)
    deferred_present = _ac_present(dadger, deferred)

    checks = [
        CheckItem(
            label="AC cadastro-override coverage",
            passed=True,
            detail=(
                f"applied {len(applied_present)}/{len(applied)}, "
                f"deferred {len(deferred_present)}/{len(deferred)}, "
                f"uningestable {len(uningestable_present)}/{len(uningestable)} "
                "family(ies) present in the deck"
            ),
        )
    ]
    diagnostics: list[Diagnostic] = []

    if applied_present:
        mnemonics = ", ".join(
            cls.__name__.removeprefix("AC") for cls in applied_present
        )
        diagnostics.append(
            Diagnostic(
                code="decomp-ac-overrides-applied",
                severity=Severity.INFO,
                category=_CONTEXT,
                title="AC cadastro overrides applied",
                summary=(
                    f"The deck declares AC cadastro overrides the conversion "
                    f"applies: {mnemonics}."
                ),
            )
        )

    if deferred_present:
        mnemonics = ", ".join(
            cls.__name__.removeprefix("AC") for cls in deferred_present
        )
        diagnostics.append(
            _deferred(
                "decomp-ac-overrides-deferred",
                "AC cadastro overrides not converted",
                f"The deck declares AC cadastro overrides with no converter "
                f"consumer: {mnemonics}.",
                "Tracked in the conversion roadmap; no action needed to convert.",
            )
        )

    if uningestable_present:
        diagnostics.append(
            _deferred(
                "decomp-ac-altefe-uningestable",
                "AC ALTEFE cannot be ingested",
                "The deck declares an AC ALTEFE effective-head override; "
                "the source-model reader cannot expose a value for it, so the "
                "conversion cannot read it at all (distinct from a deferred "
                "override, which does expose a value).",
                "This override cannot be read from the deck and is not applied; "
                "no action needed to convert.",
            )
        )

    if report.out_of_horizon:
        diagnostics.append(
            Diagnostic(
                code="decomp-ac-out-of-horizon",
                severity=Severity.WARNING,
                category=_CONTEXT,
                title="AC override effective date past the study horizon",
                summary=(
                    f"{len(report.out_of_horizon)} AC cadastro override(s) "
                    "resolve past the calendar horizon and are not applied."
                ),
                table=DiagnosticTable(
                    columns=["plant", "param", "month", "year"],
                    rows=[
                        [oh.code, oh.param, oh.mes, oh.ano]
                        for oh in report.out_of_horizon
                    ],
                ),
                remediation=(
                    "Check the override's (mes, semana, ano) triple against "
                    "the deck's study horizon."
                ),
            )
        )

    return checks, diagnostics


def _special_constraint_coverage(
    dadger: Dadger, files: DecompFiles
) -> tuple[list[CheckItem], list[Diagnostic]]:
    """Classify the deck's special constraints into converted (RE/HQ/HV/HE,
    split into bounds-lowered vs generic-emitted) and deferred (``FE``/RHA/
    LIBs-electrical), so an operator sees at a glance what the conversion
    will and will not carry — without running a conversion.

    A pure read of *dadger* (through the same census
    :func:`~novomodelo_bridge.decomp.constraint_registers.read_constraints` the
    pipeline consumes) and *files* (whose ``dadger`` path feeds the E1
    unreadable-electrical scan, and whose parent deck directory feeds the
    LIBs-electrical scan) — no conversion, no emitter.

    Always returns exactly one passing summary :class:`CheckItem` (coverage
    never blocks conversion, mirroring :func:`_ac_coverage`) plus a
    diagnostic per bucket that is non-empty: an ``INFO`` for the converted
    split — present whenever the deck declares at least one special
    constraint — reporting the bounds-lowered/generic-emitted counts as
    probes (per spec §10, never a hard-coded oracle), plus a ``WARNING`` per
    present deferred surface. The deferred diagnostics are produced verbatim
    by the E1 detection helpers
    (:func:`~novomodelo_bridge.decomp.constraint_registers.detect_unreadable_electrical`
    / :func:`~novomodelo_bridge.decomp.constraint_registers.detect_libs_electrical`)
    — never re-derived here.
    """
    census = constraint_registers.read_constraints(dadger)

    bounds_by_family: dict[str, int] = {}
    generic_by_family: dict[str, int] = {}
    for record in census.to_bounds:
        bounds_by_family[record.family] = bounds_by_family.get(record.family, 0) + 1
    for record in census.to_generic:
        generic_by_family[record.family] = generic_by_family.get(record.family, 0) + 1
    present_families = sorted(
        fam for fam, records in census.by_family.items() if records
    )

    n_bounds = len(census.to_bounds)
    n_generic = len(census.to_generic)

    checks = [
        CheckItem(
            label="Special-constraint coverage",
            passed=True,
            detail=(
                f"{n_bounds} record(s) lowered to bounds, {n_generic} "
                f"emitted as generic constraints, {len(present_families)} "
                "family(ies) present in the deck"
            ),
        )
    ]
    diagnostics: list[Diagnostic] = []

    if present_families:
        breakdown = "; ".join(
            f"{fam} {bounds_by_family.get(fam, 0)} bound(s)/"
            f"{generic_by_family.get(fam, 0)} generic"
            for fam in present_families
        )
        diagnostics.append(
            Diagnostic(
                code="decomp-special-constraints-converted",
                severity=Severity.INFO,
                category=_CONTEXT,
                title="Special constraints read and classified",
                summary=(
                    f"The deck declares special constraints in "
                    f"{len(present_families)} family(ies): {n_bounds} "
                    f"record(s) lower to an entity bound, {n_generic} emit "
                    f"as a generic constraint ({breakdown})."
                ),
            )
        )

    dadger_path = files.dadger
    diagnostics.extend(constraint_registers.detect_unreadable_electrical(dadger_path))
    libs_diagnostic = constraint_registers.detect_libs_electrical(dadger_path.parent)
    if libs_diagnostic is not None:
        diagnostics.append(libs_diagnostic)

    return checks, diagnostics


def run_decomp_preflight(src: Path) -> PreflightResult:
    """Validate the deck at *src*, writing nothing and raising nothing.

    A discovery failure is terminal and returns immediately; everything after
    it needs the resolved files. Later checks degrade independently — a
    malformed calendar blocks the tree and load checks (both need it) but
    still reports alongside the inventory.
    """
    from idecomp.decomp import Dadger, Vazoes

    from novomodelo_bridge.decomp.case import DecompCase
    from novomodelo_bridge.decomp.id_map import DecompIdMap

    try:
        files = discover_decomp_files(src)
    except Exception as exc:  # noqa: BLE001
        return PreflightResult(
            verdict=PreflightVerdict.WILL_NOT_CONVERT,
            diagnostics=[diagnostic_from_exception(exc, context=_CONTEXT)],
            checks=[
                CheckItem(
                    label="Deck discovery (caso.dat → revision index)",
                    passed=False,
                    detail=str(exc),
                )
            ],
        )

    checks: list[CheckItem] = [
        CheckItem(
            label="Deck discovery (caso.dat → revision index)",
            passed=True,
            detail=f"revision {files.revision}",
        )
    ]
    diagnostics: list[Diagnostic] = []

    optional_checks, optional_diags = optional_input_advisory(files)
    checks.extend(optional_checks)
    diagnostics.extend(optional_diags)

    try:
        # RegisterFile.read() is annotated to return the base RegisterFile,
        # not the calling subclass; narrow back to Dadger so downstream
        # helpers (typed Dadger, not the object escape hatch) type-check.
        dadger = cast(Dadger, Dadger.read(str(files.dadger)))
    except Exception as exc:  # noqa: BLE001
        checks.append(
            CheckItem(label="Deck registers readable", passed=False, detail=str(exc))
        )
        diagnostics.append(diagnostic_from_exception(exc, context=_CONTEXT))
        return PreflightResult(
            verdict=PreflightVerdict.WILL_NOT_CONVERT,
            diagnostics=diagnostics,
            checks=checks,
        )

    case = DecompCase(files=files)

    try:
        id_map = DecompIdMap.from_dadger(dadger)
    except (FieldParseError, ValueError) as exc:
        checks.append(CheckItem(label="Entity id map", passed=False, detail=str(exc)))
        diagnostics.append(diagnostic_from_exception(exc, context=_CONTEXT))
        return PreflightResult(
            verdict=PreflightVerdict.WILL_NOT_CONVERT,
            diagnostics=diagnostics,
            checks=checks,
        )

    checks.append(
        CheckItem(
            label="Entity id map",
            passed=True,
            detail=(
                f"{len(id_map.bus_codes)} subsystems, "
                f"{len(id_map.hydro_codes)} hydros, "
                f"{len(id_map.thermal_codes)} thermals"
            ),
        )
    )

    calendar_check, calendar, failure = _calendar_check(dadger)
    checks.append(calendar_check)

    if failure is None:
        checks.append(_load_factor_check(case, id_map))
        try:
            vazoes = cast(Vazoes, Vazoes.read(str(files.vazoes)))
        except Exception as exc:  # noqa: BLE001
            checks.append(
                CheckItem(label="Scenario tree readable", passed=False, detail=str(exc))
            )
        else:
            checks.extend(_tree_checks(vazoes, calendar))

        from novomodelo_bridge.decomp.converters.cadastro import (
            build_effective_cadastro,
        )
        from novomodelo_bridge.decomp.converters.hydro import read_hidr

        try:
            hidr = read_hidr(files.hidr)
            _effective, cadastro_report = build_effective_cadastro(
                dadger, hidr, calendar
            )
        except (ValueError, KeyError, OSError) as exc:
            checks.append(
                CheckItem(
                    label="AC cadastro overrides resolvable",
                    passed=False,
                    detail=str(exc),
                )
            )
        else:
            cov_checks, cov_diags = _ac_coverage(dadger, cadastro_report)
            checks.extend(cov_checks)
            diagnostics.extend(cov_diags)

        sc_checks, sc_diags = _special_constraint_coverage(dadger, files)
        checks.extend(sc_checks)
        diagnostics.extend(sc_diags)

    diagnostics.extend(_deferred_inventory(dadger))

    return PreflightResult(
        verdict=_verdict(checks, diagnostics),
        diagnostics=diagnostics,
        checks=checks,
    )
