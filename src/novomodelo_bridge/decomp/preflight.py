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

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    collect,
    counted,
    emit,
)
from novomodelo_bridge.core.errors import FieldParseError, diagnostic_from_exception
from novomodelo_bridge.core.libs_index import unread_lines_diagnostic
from novomodelo_bridge.core.preflight import (
    CheckItem,
    PreflightResult,
    PreflightVerdict,
    optional_input_advisory,
    tab_character_advisory,
    unindexed_libs_advisory,
)
from novomodelo_bridge.decomp import constraint_registers
from novomodelo_bridge.decomp.converters.cadastro import (
    AC_REGISTER_BY_PARAM,
    ALL_AC_CLASSES,
    APPLIED_AC_CLASSES,
    DEFERRED_AC_CLASSES,
    AcRegisters,
    ac_overrides_deferred_diagnostic,
    ac_register_table,
    ac_registers_present,
    topology_varies_diagnostic,
)
from novomodelo_bridge.decomp.files import (
    REGISTER_COMMENT_PREFIX,
    discover_decomp_files,
    unindexed_libs_diagnostic,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from idecomp.decomp import Dadger, Vazoes

    from novomodelo_bridge.decomp.case import DecompCase
    from novomodelo_bridge.decomp.converters.cadastro import (
        CadastroResolutionReport,
        EffectiveCadastro,
    )
    from novomodelo_bridge.decomp.id_map import DecompIdMap
    from novomodelo_bridge.decomp.temporal import OperativeStage

_CONTEXT = "Preflight"

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


_CALENDAR_LABEL = "Operative calendar (operative weeks, then whole months)"


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
                label=_CALENDAR_LABEL,
                passed=False,
                detail=str(exc),
            ),
            [],
            str(exc),
        )
    span = f"{calendar[0].start_date} → {calendar[-1].end_date}"
    return (
        CheckItem(
            label=_CALENDAR_LABEL,
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


class _SpecialConstraintRegisters(
    constraint_registers.ConstraintRegisters,
    constraint_registers.RhaRegisters,
    Protocol,
):
    """The deck accessors the special-constraint coverage check reads."""


def _ac_coverage(
    dadger: AcRegisters,
    effective: EffectiveCadastro,
    report: CadastroResolutionReport,
) -> tuple[list[CheckItem], list[Diagnostic]]:
    """Report which applied and which unconverted ``AC`` registers the deck
    declares, the ones taking effect after the horizon, and the cascade
    registers that change between stages.

    The two buckets diff the idecomp ``AC`` universe against the resolver's
    own :data:`~novomodelo_bridge.decomp.converters.cadastro.APPLIED_AC_CLASSES`,
    never a hand-maintained list. The unconverted-register and cascade
    WARNINGs are the ones ``convert decomp`` raises, from the same builders.
    Always returns exactly one passing summary :class:`CheckItem`: coverage
    never blocks conversion.
    """
    applied = ALL_AC_CLASSES & APPLIED_AC_CLASSES
    applied_present = ac_registers_present(dadger, applied)
    deferred_present = ac_registers_present(dadger, DEFERRED_AC_CLASSES)

    checks = [
        CheckItem(
            label="AC cadastro-override coverage",
            passed=True,
            detail=(
                f"{len(applied_present)} of {len(applied)} applied and "
                f"{len(deferred_present)} of {len(DEFERRED_AC_CLASSES)} "
                "unconverted AC registers present in the deck"
            ),
        )
    ]
    diagnostics: list[Diagnostic] = []

    if applied_present:
        diagnostics.append(
            Diagnostic(
                code="decomp-ac-overrides-applied",
                severity=Severity.INFO,
                category=_CONTEXT,
                title=f"AC registers applied ({len(applied_present)})",
                summary=(
                    "The deck declares the listed AC registers, which the "
                    "conversion applies from the stage each takes effect, as "
                    "in DECOMP."
                ),
                table=ac_register_table(applied_present),
            )
        )

    deferred_diagnostic = ac_overrides_deferred_diagnostic(dadger)
    if deferred_diagnostic is not None:
        diagnostics.append(deferred_diagnostic)

    if report.out_of_horizon:
        diagnostics.append(
            Diagnostic(
                code="decomp-ac-out-of-horizon",
                severity=Severity.INFO,
                category=_CONTEXT,
                title=(
                    "AC registers taking effect after the study horizon "
                    f"({len(report.out_of_horizon)})"
                ),
                summary=(
                    "The listed AC registers take effect after the last stage "
                    "of the study, so, as in DECOMP, they change nothing."
                ),
                table=DiagnosticTable(
                    columns=["Code", "AC register", "Month", "Year"],
                    rows=[
                        [
                            oh.code,
                            AC_REGISTER_BY_PARAM.get(oh.param, oh.param),
                            oh.mes,
                            oh.ano,
                        ]
                        for oh in report.out_of_horizon
                    ],
                    justify=["right", "left", "right", "right"],
                ),
                remediation=(
                    "Check the month, week and year of each listed AC register "
                    "in dadger.rvN against the study horizon."
                ),
            )
        )

    varies_diagnostic = topology_varies_diagnostic(effective)
    if varies_diagnostic is not None:
        diagnostics.append(varies_diagnostic)

    return checks, diagnostics


def _special_constraint_coverage(
    dadger: _SpecialConstraintRegisters,
    libs_file: Path | None,
    id_map: DecompIdMap,
    pumping_ids: Mapping[int, int],
    *,
    libs_long_form: bool | None = False,
    plant_names: Mapping[int, str] | None = None,
) -> tuple[list[CheckItem], list[Diagnostic]]:
    """Classify the deck's special constraints into converted (RE/HQ/HV/HE,
    split into bounds-lowered vs generic-emitted), left out (a term on an
    entity the case lacks) and deferred (RHA/LIBs-electrical), without
    running a conversion.

    Counts the census ``convert decomp`` consumes,
    :func:`~novomodelo_bridge.decomp.constraint_registers.read_screened_constraints`
    against *id_map* and the ``UE`` codes of *pumping_ids*. Always returns
    exactly one passing summary :class:`CheckItem` (coverage never blocks
    conversion), an ``INFO`` for the converted split when any constraint
    converts, and, verbatim as ``convert decomp`` raises them, the screen's
    WARNINGs (conflicting restrictions under *plant_names*), the one of
    :func:`~novomodelo_bridge.decomp.constraint_registers.detect_unconverted_rha`
    and, when *libs_long_form* is ``False`` (the LIBS file has no long-form
    card ``convert decomp`` reads; ``None`` when the read failed, which its own
    failed check reports), the one of
    :func:`~novomodelo_bridge.decomp.constraint_registers.detect_libs_electrical`.
    :func:`_libs_electrical_read` raises the LIBS file's own WARNINGs.
    """
    census, screened = constraint_registers.read_screened_constraints(
        dadger,
        hydro_codes=id_map.hydro_codes,
        thermal_codes=id_map.thermal_codes,
        pumping_codes=pumping_ids.keys(),
        plant_names={} if plant_names is None else plant_names,
    )

    bounds_by_family: dict[str, int] = {}
    generic_by_family: dict[str, int] = {}
    for record in census.to_bounds:
        bounds_by_family[record.family] = bounds_by_family.get(record.family, 0) + 1
    for record in census.to_generic:
        generic_by_family[record.family] = generic_by_family.get(record.family, 0) + 1
    present_families = sorted(
        fam for fam, records in census.by_family.items() if records
    )
    converted_families = sorted(bounds_by_family.keys() | generic_by_family.keys())

    n_bounds = len(census.to_bounds)
    n_generic = len(census.to_generic)

    checks = [
        CheckItem(
            label="Special-constraint coverage",
            passed=True,
            detail=(
                f"{counted(n_bounds, 'record')} lowered to bounds, {n_generic} "
                "emitted as generic constraints, "
                f"{counted(len(present_families), 'family', 'families')} present "
                "in the deck"
            ),
        )
    ]
    diagnostics: list[Diagnostic] = []

    if converted_families:
        diagnostics.append(
            Diagnostic(
                code="decomp-special-constraints-converted",
                severity=Severity.INFO,
                category=_CONTEXT,
                title=f"Special constraints converted ({n_bounds + n_generic})",
                summary=(
                    "The table counts how the deck's special constraints "
                    "convert: each restriction on a single variable becomes a "
                    "bound on it, and the others become generic constraints. "
                    "Both bind as DECOMP's restrictions do."
                ),
                table=DiagnosticTable(
                    columns=["Register", "As bounds", "As generic constraints"],
                    rows=[
                        [
                            fam,
                            bounds_by_family.get(fam, 0),
                            generic_by_family.get(fam, 0),
                        ]
                        for fam in converted_families
                    ],
                    justify=["left", "right", "right"],
                ),
            )
        )
    diagnostics.extend(screened)

    rha_diagnostic = constraint_registers.detect_unconverted_rha(dadger)
    if rha_diagnostic is not None:
        diagnostics.append(rha_diagnostic)
    libs_diagnostic = constraint_registers.detect_libs_electrical(libs_file)
    if libs_diagnostic is not None and libs_long_form is False:
        diagnostics.append(libs_diagnostic)

    return checks, diagnostics


def _libs_electrical_read(
    case: DecompCase, calendar: list[OperativeStage]
) -> tuple[bool | None, list[CheckItem], list[Diagnostic]]:
    """Read the LIBS ``RESTRICAO-ELETRICA-ESPECIAL`` file as ``convert decomp`` does.

    Returns whether it has a long-form card ``convert decomp`` reads, with the
    WARNINGs the read raises, the lines it does not read among them. A file it
    cannot read gives ``None`` and a failed check, since ``convert decomp``
    stops on it.
    """
    from novomodelo_bridge.decomp.converters import libs_electrical

    path = case.files.libs_restricao_eletrica
    found: list[Diagnostic] = []
    try:
        with collect() as found:
            restricoes = case.libs_restricao_eletrica
            model = None
            if path is not None and restricoes is not None:
                unread = unread_lines_diagnostic(path, restricoes.data)
                if unread is not None:
                    emit(unread)
                model = libs_electrical.read_libs_electrical(restricoes, calendar)
    except Exception as exc:
        failed = CheckItem(
            label="LIBS electrical restrictions readable",
            passed=False,
            detail=str(exc),
        )
        return (
            None,
            [failed],
            [*found, diagnostic_from_exception(exc, context=_CONTEXT)],
        )
    return model is not None, [], found


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
    except Exception as exc:
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

    tab_checks, tab_diags = tab_character_advisory(
        files.fixed_column_files, comment_prefix=REGISTER_COMMENT_PREFIX
    )
    checks.extend(tab_checks)
    diagnostics.extend(tab_diags)

    libs_checks, libs_diags = unindexed_libs_advisory(unindexed_libs_diagnostic(files))
    checks.extend(libs_checks)
    diagnostics.extend(libs_diags)

    try:
        dadger = Dadger.read(str(files.dadger))
    except Exception as exc:
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
            vazoes = Vazoes.read(str(files.vazoes))
        except Exception as exc:
            checks.append(
                CheckItem(label="Scenario tree readable", passed=False, detail=str(exc))
            )
        else:
            checks.extend(_tree_checks(vazoes, calendar))

        from novomodelo_bridge.decomp.converters.cadastro import (
            build_effective_cadastro,
        )
        from novomodelo_bridge.decomp.converters.hydro import read_hidr
        from novomodelo_bridge.decomp.converters.network import (
            pumping_station_id_map,
        )

        plant_names: dict[int, str] = {}
        try:
            hidr = read_hidr(files.hidr)
            effective, cadastro_report = build_effective_cadastro(
                dadger, hidr, calendar
            )
        except (ValueError, KeyError, OSError, FieldParseError) as exc:
            checks.append(
                CheckItem(
                    label="AC cadastro overrides resolvable",
                    passed=False,
                    detail=str(exc),
                )
            )
        else:
            plant_names = {
                int(code): str(name).strip()
                for code, name in effective.base["nome_usina"].items()
            }
            cov_checks, cov_diags = _ac_coverage(dadger, effective, cadastro_report)
            checks.extend(cov_checks)
            diagnostics.extend(cov_diags)

        libs_long_form, electrical_checks, electrical_diags = _libs_electrical_read(
            case, calendar
        )
        checks.extend(electrical_checks)
        diagnostics.extend(electrical_diags)

        sc_checks, sc_diags = _special_constraint_coverage(
            dadger,
            files.libs_restricao_eletrica,
            id_map,
            pumping_station_id_map(dadger),
            libs_long_form=libs_long_form,
            plant_names=plant_names,
        )
        checks.extend(sc_checks)
        diagnostics.extend(sc_diags)

    return PreflightResult(
        verdict=_verdict(checks, diagnostics),
        diagnostics=diagnostics,
        checks=checks,
    )
