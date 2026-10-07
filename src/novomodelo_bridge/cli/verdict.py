"""The unified ``--json`` verdict envelope shared by every command.

Each command's machine-readable ``--json`` output is one self-describing
document with EXACTLY five top-level keys, in this fixed insertion order::

    {
        "schema_version": <int>,   # always first; discriminator version
        "command": <str>,          # which command produced this document
        "status": <str>,           # command-specific outcome ("ok", "error", …)
        "summary": <object>,       # ALL command-specific payload lives here
        "diagnostics": [<object>]  # zero or more Diagnostic.to_dict() entries
    }

``schema_version`` is ALWAYS the first key so a consumer can read it before
deciding how to interpret the rest.

The nesting rule (the single design rule every command obeys): the ONLY
top-level keys that ever appear are the five envelope keys above. Every
command-specific payload — convert's counts, check's checklist, compare's
headline, dashboard's artifact — lives UNDER ``summary``. A new command adds
its payload by supplying a ``summary`` dict, never a new top-level key.

The envelope itself is deterministic: it carries no timestamp, no git SHA, no
absolute path. Determinism of the ``summary`` contents is the caller's
responsibility; provenance (if ever needed) lives in
:mod:`cobre_bridge.cli.conversion_manifest`, never here.

Version bump policy for :data:`SCHEMA_VERSION`: increment it when a key is
renamed or removed, or when an existing key's meaning changes — a breaking
change for consumers. Adding a NEW optional key under ``summary`` is
backward-compatible and does NOT bump the version.

This is a pure leaf: besides :class:`~cobre_bridge.core.diagnostics.Severity` (the
runtime dependency of ``_convert_status``), it imports stdlib only — the
cross-module ``Diagnostic`` / ``ConversionReport`` / ``CompareVerdict`` types
are referenced under ``TYPE_CHECKING``. It reads its inputs, allocates fresh
dicts/lists, mutates nothing, and does no I/O. The CLI layer serializes
the returned dict to stdout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cobre_bridge.core.diagnostics import Severity

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cobre_bridge.comparators.dataset import ComparisonDataset
    from cobre_bridge.comparators.verdict import CompareVerdict
    from cobre_bridge.core.conversion import ConversionReport
    from cobre_bridge.core.diagnostics import Diagnostic

# Bump only on a breaking change: a key rename/removal or a meaning change.
# Adding a NEW optional key under ``summary`` is backward-compatible — do NOT bump.
SCHEMA_VERSION: int = 1


def build_verdict(
    command: str,
    status: str,
    summary: Mapping[str, object],
    diagnostics: Sequence[Diagnostic] = (),
) -> dict[str, object]:
    """Build the unified verdict envelope for one command invocation.

    Returns a JSON-serializable dict with the five fixed top-level keys in fixed
    insertion order, so ``json.dump(..., sort_keys=False)`` is byte-stable:
    ``schema_version`` (always first), ``command``, ``status``, ``summary``,
    ``diagnostics``.

    Args:
        command: The command discriminator (e.g. ``"convert newave"``). Passed
            through verbatim — NOT validated against an allow-list.
        status: The command-specific outcome (e.g. ``"ok"`` / ``"error"`` /
            ``"dry-run"``). Passed through verbatim — NOT validated.
        summary: The command-specific payload. Copied via ``dict(summary)`` so the
            caller's key insertion order is preserved and the caller's mapping is
            never aliased or mutated by the returned document.
        diagnostics: The findings to serialize, each via :meth:`Diagnostic.to_dict`.
            Defaults to the empty tuple (no mutable default); empty → ``[]``.

    Returns:
        A new dict with exactly ``{schema_version, command, status, summary,
        diagnostics}``. Adds NO timestamp, provenance, or absolute path.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "command": command,
        "status": status,
        "summary": dict(summary),
        "diagnostics": [d.to_dict() for d in diagnostics],
    }


def convert_summary(
    hydros: int,
    thermals: int,
    buses: int,
    lines: int,
    stages: int,
) -> dict[str, object]:
    """The ``convert`` command's ``summary`` block — entity counts in fixed order.

    Returns ``{"hydros", "thermals", "buses", "lines", "stages"}`` in that
    insertion order. The ``--dry-run`` / ``validation`` sub-keys are appended
    inline by the convert retrofit; they are not constructed here.
    """
    return {
        "hydros": hydros,
        "thermals": thermals,
        "buses": buses,
        "lines": lines,
        "stages": stages,
    }


def check_summary(checks: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """The ``check`` command's ``summary`` block — the checklist under ``checks``.

    Each row is copied via ``dict(c)`` so the returned document never aliases the
    caller's check mappings. Returns ``{"checks": [...]}``.
    """
    return {"checks": [dict(c) for c in checks]}


def compare_summary(verdict: CompareVerdict) -> dict[str, object]:
    """The ``compare`` command's ``summary`` block — the headline verdict.

    Returns ``{"within_tol", "total", "worst_variable", "worst_smape",
    "all_within_tol"}`` in that order. ``worst_smape`` is the raw sMAPE ratio
    (NOT a percentage), matching :attr:`CompareVerdict.worst_smape`.

    Mirrors the console ``all_within_tol`` guard in
    :func:`cobre_bridge.ui.console.render_compare_verdict`: on a perfect match the
    "worst" clause is meaningless, so ``worst_variable`` is nulled to ``None`` and
    ``worst_smape`` to ``0.0``. Otherwise both pass through verbatim.
    """
    all_within_tol = verdict.all_within_tol
    return {
        "within_tol": verdict.within_tol,
        "total": verdict.total,
        "worst_variable": None if all_within_tol else verdict.worst_variable,
        "worst_smape": 0.0 if all_within_tol else verdict.worst_smape,
        "all_within_tol": all_within_tol,
    }


def _coerce_unmapped_code(code: object) -> int | list[int]:
    """JSON-friendly coercion for one ``unmapped`` entry.

    Most levels list scalar entity codes (coerced to ``int``); the network
    (``line``) level lists ``[submarket_de, submarket_para]`` corridor **pairs**
    — lists, not scalar codes — so those are coerced element-wise and kept as
    lists of ``int`` rather than forced through ``int(...)`` (which raised
    ``TypeError`` on the pair).
    """
    if isinstance(code, list | tuple):
        return [int(c) for c in code]
    return int(code)


def decomp_dataset_summary(
    dataset: ComparisonDataset, tolerance: float
) -> dict[str, object]:
    """The ``compare decomp`` command's dataset-sourced ``summary`` block.

    Returns the shared headline fields from :func:`compare_summary`
    (``within_tol``, ``total``, ``worst_variable``, ``worst_smape``,
    ``all_within_tol`` — sourced via
    :func:`~cobre_bridge.comparators.verdict.build_compare_verdict`, so the
    headline is computed once from ``dataset.summary`` and matches
    ``compare newave``'s) PLUS the three DECOMP-specific keys read straight
    from *dataset*, in this fixed order: ``stages`` (the count of distinct
    ``dataset.tidy["stage"]`` values), ``variables``
    (``dataset.summary.to_dicts()`` verbatim), and ``unmapped``
    (``dataset.metadata["unmapped"]``, with every entity id coerced to
    ``int``).

    *tolerance* is accepted for call-site symmetry with the other compare
    commands' summary builders — it is not consumed here because
    ``dataset.summary``'s ``within_tol_rate`` (which
    :func:`~cobre_bridge.comparators.verdict.build_compare_verdict` reads) was
    already computed against a tolerance when the caller built *dataset*
    (e.g. via ``build_decomp_dataset(..., tolerance=...)``).
    """
    from cobre_bridge.comparators.verdict import build_compare_verdict

    summary = compare_summary(build_compare_verdict(dataset))
    summary["stages"] = int(dataset.tidy["stage"].n_unique())
    summary["variables"] = dataset.summary.to_dicts()
    summary["unmapped"] = {
        level: [_coerce_unmapped_code(code) for code in codes]
        for level, codes in dataset.metadata["unmapped"].items()
    }
    return summary


def dashboard_summary(output: str, size_kb: float) -> dict[str, object]:
    """The ``dashboard`` command's ``summary`` block — the written artifact.

    ``output`` is the artifact path as a string (the caller passes
    ``str(path)``); ``size_kb`` is its size in kibibytes as a float. Returns
    ``{"output", "size_kb"}``.
    """
    return {"output": output, "size_kb": size_kb}


def _convert_verdict_summary(report: ConversionReport | None) -> dict[str, object]:
    """The convert ``summary`` block — entity counts, zeroed when *report* is None.

    A thin wrapper over :func:`convert_summary` that supplies the five counts
    from a :class:`ConversionReport` (or all zeros on the failure path, where
    ``report`` is ``None``). Keeping the count plumbing here lets the
    real-run, failure, and dry-run call sites share one source of truth while
    the key order itself stays owned by :func:`convert_summary`.
    """
    if report is None:
        return convert_summary(0, 0, 0, 0, 0)
    return convert_summary(
        report.hydro_count,
        report.thermal_count,
        report.bus_count,
        report.line_count,
        report.stage_count,
    )


def _convert_status(diagnostics: Sequence[Diagnostic], *, success: str) -> str:
    """Derive the convert verdict ``status`` from diagnostic severity ONLY.

    Returns ``"error"`` when any diagnostic has ``ERROR`` severity, otherwise the
    caller's *success* token (``"ok"`` for a real run, ``"dry-run"`` for a dry
    run). Validation outcome never enters here — it lands in ``summary.validation``
    and the exit code, keeping ``status`` a pure diagnostics signal.
    """
    if any(d.severity is Severity.ERROR for d in diagnostics):
        return "error"
    return success
