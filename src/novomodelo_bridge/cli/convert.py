"""Convert-command handlers for ``convert newave`` / ``convert decomp``."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import typer

from cobre_bridge.cli.args import ConvertArgs
from cobre_bridge.cli.conversion_manifest import _write_conversion_manifest
from cobre_bridge.cli.failure import _emit_convert_json, _fail
from cobre_bridge.cli.validate import _run_cobre_validation
from cobre_bridge.cli.verdict import (
    _convert_status,
    _convert_verdict_summary,
    build_verdict,
)
from cobre_bridge.core.errors import (
    BridgeError,
    SourceFileError,
    diagnostic_from_exception,
)
from cobre_bridge.decomp.files import discover_decomp_files
from cobre_bridge.ui.console import (
    conversion_progress,
    get_console,
    make_table,
    print_status,
    render_conversion_summary,
    render_diagnostics,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rich.console import Console

    from cobre_bridge.core.conversion import ConversionReport
    from cobre_bridge.core.diagnostics import Diagnostic


def _handle_conversion_pipeline_failure(
    exc: Exception, args: ConvertArgs, *, command: str, err_console: Console
) -> NoReturn:
    """Render/emit a pipeline failure verdict and exit 1 (shared by both tracks).

    ``report`` is None on this path, so counts are zeroed and the dry-run
    would-write listing is empty; ``status`` is "error" because
    ``diagnostic_from_exception`` yields an ERROR-severity diagnostic.
    """
    diag = diagnostic_from_exception(exc, context="Conversion")
    if args.json_output:
        diagnostics = [diag]
        summary = _convert_verdict_summary(None)
        if args.dry_run:
            summary["would_write"] = []
            status = _convert_status(diagnostics, success="dry-run")
        else:
            status = _convert_status(diagnostics, success="ok")
        _emit_convert_json(build_verdict(command, status, summary, diagnostics))
    else:
        render_diagnostics(
            [diag], heading="Conversion notes", console=err_console, quiet=args.quiet
        )
    raise typer.Exit(code=1)


def _write_diagnostics_json(
    report: ConversionReport,
    path: Path,
    *,
    diagnostics: Sequence[Diagnostic] | None = None,
    console: Console,
) -> None:
    """Write the conversion counts + diagnostics to *path* as JSON.

    ``diagnostics`` defaults to ``report.diagnostics`` (the plain
    ``convert newave``/``convert decomp`` contract); a caller with additional
    findings not yet folded into ``report`` — e.g. ``convert decomp
    --boundary-fcf``'s importer diagnostics — passes the combined list
    explicitly so the sidecar matches the ``--json`` verdict's merge.

    A write failure is reported but does not change the exit code — the conversion
    itself already succeeded.
    """
    resolved_diagnostics = report.diagnostics if diagnostics is None else diagnostics
    payload = {
        "summary": {
            "hydros": report.hydro_count,
            "thermals": report.thermal_count,
            "buses": report.bus_count,
            "lines": report.line_count,
            "stages": report.stage_count,
        },
        "diagnostics": [d.to_dict() for d in resolved_diagnostics],
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        print_status(
            f"Warning: failed to write diagnostics JSON: {exc}",
            console=console,
            style="#F5A623",
        )
    else:
        print_status(f"Diagnostics written to {path}", console=console)


def _run_newave_conversion(args: ConvertArgs) -> None:
    """Execute the convert newave subcommand."""
    from cobre_bridge.core.conversion import clear_dst_contents
    from cobre_bridge.newave.files import NewaveFiles
    from cobre_bridge.newave.pipeline import (
        CONVERSION_PHASE_LABELS,
        NEWAVE_CLEARED_ARTIFACTS,
        convert_newave_case,
    )

    src: Path = args.src
    dst: Path = args.dst

    out_console = args.out_console()
    err_console = args.err_console()

    if not src.exists() or not src.is_dir():
        _fail(
            "convert newave",
            args,
            SourceFileError(f"source directory '{src}' does not exist"),
            1,
            summary=_convert_verdict_summary(None),
        )

    if dst.exists() and any(dst.iterdir()):
        if not args.force:
            _fail(
                "convert newave",
                args,
                BridgeError(
                    f"destination directory '{dst}' is not empty."
                    " Use --force to overwrite."
                ),
                1,
                summary=_convert_verdict_summary(None),
            )
        # --force: remove previous pipeline outputs before converting. A dry run
        # never mutates the destination, so the clear is skipped even with --force.
        if not args.dry_run:
            clear_dst_contents(dst, NEWAVE_CLEARED_ARTIFACTS)

    # A dry run creates no destination directory; the pipeline writes nothing.
    if not args.dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    try:
        with conversion_progress(
            len(CONVERSION_PHASE_LABELS),
            verbose=args.verbose > 0,
            quiet=args.quiet,
            no_color=args.no_color,
        ) as step:
            report: ConversionReport = convert_newave_case(
                src, dst, on_phase=step, dry_run=args.dry_run
            )
    except Exception as exc:  # noqa: BLE001
        _handle_conversion_pipeline_failure(
            exc, args, command="convert newave", err_console=err_console
        )

    if args.dry_run:
        # Dry run: report the would-write listing only; touch nothing on disk
        # (no diagnostics-json sidecar, no manifest, no validation).
        if args.json_output:
            # The would-write listing moves UNDER ``summary`` (dst-relative,
            # forward-slash, sorted) so the only top-level keys are the five
            # envelope keys.
            summary = _convert_verdict_summary(report)
            summary["would_write"] = sorted(
                Path(p).relative_to(dst).as_posix() for p in report.would_write_paths
            )
            status = _convert_status(report.diagnostics, success="dry-run")
            _emit_convert_json(
                build_verdict("convert newave", status, summary, report.diagnostics)
            )
        else:
            if not args.quiet:
                _render_dry_run_summary(report, console=out_console)
            render_diagnostics(
                report.diagnostics,
                heading="Conversion notes",
                console=err_console,
                quiet=args.quiet,
            )
        if args.validate:
            print_status(
                "Note: --validate is ignored under --dry-run"
                " (nothing was written to validate).",
                console=err_console,
                style="#F5A623",
            )
        return

    # Build the convert ``summary`` + ``status`` up front; ``--validate`` may
    # later append a ``summary["validation"]`` sub-object (under --json), and the
    # verdict is emitted to stdout only after validation has run so that block is
    # populated. ``status`` is diagnostics-only and is NOT touched by validation.
    summary = _convert_verdict_summary(report)
    status = _convert_status(report.diagnostics, success="ok")

    if not args.json_output:
        if not args.quiet:
            render_conversion_summary(report, console=out_console)
        render_diagnostics(
            report.diagnostics,
            heading="Conversion notes",
            console=err_console,
            quiet=args.quiet,
        )

    if args.diagnostics_json is not None:
        # The --diagnostics-json sidecar coexists with --json (both can be set).
        _write_diagnostics_json(report, args.diagnostics_json, console=err_console)

    # Provenance manifest, always written on a successful conversion. Notes go
    # to err_console (stderr) so the --json stdout verdict stays byte-deterministic.
    _write_conversion_manifest(
        report,
        src,
        dst,
        command="convert newave",
        discover=NewaveFiles.from_directory,
        console=err_console,
    )

    # ``--validate`` failure flips the exit code, never the status (verdict
    # exit-code contract). The verdict is emitted only AFTER this block so the
    # ``validation`` sub-object it folds under ``summary`` is populated first.
    validation_failed = False
    if args.validate:
        validation_failed = _run_cobre_validation(
            dst,
            command="convert newave",
            summary=summary,
            json_output=args.json_output,
            err_console=err_console,
            whitelist_substrings=(),
        )

    # Emit the --json verdict now (after validation has populated ``summary``).
    if args.json_output:
        _emit_convert_json(
            build_verdict("convert newave", status, summary, report.diagnostics)
        )

    _gate_convert_exit(status, validation_failed=validation_failed)


def _gate_convert_exit(status: str, *, validation_failed: bool) -> None:
    """Convert exit-code gate: validation failure (2) over error status (1)."""
    if validation_failed:
        raise typer.Exit(code=2)
    if status == "error":
        raise typer.Exit(code=1)


def _render_dry_run_summary(
    report: ConversionReport, *, console: Console | None = None
) -> None:
    """Render the dry-run would-write listing to stdout as a primary result.

    Prints a clear "Dry run — no files written" banner, the entity summary, and a
    table of every path the conversion would have written. Diagnostics are
    rendered separately on stderr.
    """
    target = console or get_console()
    print_status(
        "Dry run — no files written.",
        console=target,
        style="bold #F5A623",
    )
    render_conversion_summary(report, console=target)

    # Human output lists the ABSOLUTE paths (easy to copy-paste / inspect on
    # disk); the ``--dry-run --json`` document instead emits dst-relative sorted
    # paths for byte-stable, location-independent automation. The divergence is
    # intentional — keep the two representations separate.
    rows: list[list[object]] = [[path] for path in report.would_write_paths]
    table = make_table(
        ["Would write"],
        rows,
        title=f"{len(rows)} output files",
    )
    target.print(table)


#: Warning substring that marks the lag-blind stage shape
#: (``state_variables.inflow_lags = false`` on every stage, alongside the
#: positive inflow-lag depth cobre infers from the imported boundary policy) as
#: deliberate external-solver interoperability, not a misconfiguration. Only
#: fires once a boundary FCF is imported. Matched against cobre's stable
#: substring, never the volatile message prefix, which would drift.
_DECOMP_VALIDATION_WHITELIST: tuple[str, ...] = ("external-solver interoperability",)


def _run_decomp_conversion(args: ConvertArgs) -> None:
    """Execute the convert decomp subcommand.

    Structurally mirrors ``_run_newave_conversion``: a TTY phase bar over the
    DECOMP phases, the ``✓ Converted …`` summary, grouped diagnostic panels, a
    ``--dry-run`` branch, a provenance-manifest write, and a unified ``--json``
    verdict.

    A broad ``except Exception`` (rather than a fixed exception tuple) also
    covers an ERROR-severity post-emission self-check finding (cobre rules
    43/41/45/38/36 + the block_id-range rule) — ``convert_decomp_case`` raises a
    ``ValueError`` naming the failing rule(s) and entities, mapped to exit 1
    like every other conversion failure.

    Unless ``--no-fcf`` is set, the boundary-FCF importer runs after the
    manifest write and BEFORE ``--validate`` (so validation sees the patched
    ``config.json``) whenever the deck declares its cut files: the capability
    probe, then ``import_boundary_fcf``. A deck with no cut files simply
    converts without a boundary FCF (an INFO note, not an error); a capability
    or importer failure exits 1 via the same ``diagnostic_from_exception``
    mapping as a conversion failure. A successful import confirms the boundary
    FCF and records a ``summary["boundary_fcf"]`` sub-object; the boundary
    loads automatically on a plain ``cobre run <case>``.

    ``--validate`` runs after a successful conversion (and boundary-FCF
    import) via the shared ``_run_cobre_validation`` helper (mirroring
    ``convert newave``), with the DECOMP external-solver-interop whitelist so
    the deliberate ``inflow_lags=false`` shape never surfaces as a scary
    warning; a failed validation exits 2, giving ``convert decomp`` the same
    0/1/2 exit-code set as ``convert newave``.
    """
    from cobre_bridge.decomp.case import DecompCase
    from cobre_bridge.decomp.pipeline import (
        DECOMP_CONVERSION_PHASE_LABELS,
        FcfInputs,
        convert_decomp_case,
    )

    out_console = args.out_console()
    err_console = args.err_console()
    # Threaded to `import_boundary_fcf` below, so it reuses the pipeline's own
    # in-memory `config`/`initial_conditions` dicts instead of re-reading
    # either file off disk.
    fcf_inputs = FcfInputs()

    try:
        with conversion_progress(
            len(DECOMP_CONVERSION_PHASE_LABELS),
            verbose=args.verbose > 0,
            quiet=args.quiet,
            no_color=args.no_color,
        ) as step:
            report: ConversionReport = convert_decomp_case(
                args.src,
                args.dst,
                force=args.force,
                on_phase=step,
                dry_run=args.dry_run,
                fcf_inputs_out=fcf_inputs,
            )
    except Exception as exc:  # noqa: BLE001
        _handle_conversion_pipeline_failure(
            exc, args, command="convert decomp", err_console=err_console
        )

    if args.dry_run:
        # Dry run: report the would-write listing only; touch nothing on disk
        # (no diagnostics-json sidecar, no validation).
        if args.json_output:
            summary = _convert_verdict_summary(report)
            summary["would_write"] = sorted(
                Path(p).relative_to(args.dst).as_posix()
                for p in report.would_write_paths
            )
            status = _convert_status(report.diagnostics, success="dry-run")
            _emit_convert_json(
                build_verdict("convert decomp", status, summary, report.diagnostics)
            )
        else:
            if not args.quiet:
                _render_dry_run_summary(report, console=out_console)
            render_diagnostics(
                report.diagnostics,
                heading="Conversion notes",
                console=err_console,
                quiet=args.quiet,
            )
        if args.validate:
            print_status(
                "Note: --validate is ignored under --dry-run"
                " (nothing was written to validate).",
                console=err_console,
                style="#F5A623",
            )
        if not args.no_fcf:
            print_status(
                "Note: boundary FCF import is skipped under --dry-run"
                " (nothing was written to import into).",
                console=err_console,
                style="#F5A623",
            )
        return

    # Build the convert ``summary`` up front; ``--validate`` may later append a
    # ``summary["validation"]`` sub-object (under --json), and the verdict is
    # emitted to stdout only after validation has run so that block is
    # populated. ``status`` is computed at emission time (below) from the
    # merged converter + boundary-FCF diagnostics, since the latter are not
    # known until the boundary-FCF block below has run.
    summary = _convert_verdict_summary(report)

    if not args.json_output:
        if not args.quiet:
            render_conversion_summary(report, console=out_console)
        render_diagnostics(
            report.diagnostics,
            heading="Conversion notes",
            console=err_console,
            quiet=args.quiet,
        )

    # The --diagnostics-json sidecar write is deferred until AFTER the
    # boundary-FCF block below (both on its success and its failure path) so
    # it can serialize the combined converter + boundary-FCF diagnostic set —
    # mirroring the --json verdict's merge — instead of the converter-only
    # snapshot a write at this point would capture. When --boundary-fcf is
    # off (or never reaches the sink), ``boundary_diagnostics`` stays empty
    # and the sidecar content is unchanged from a converter-only write.

    # Provenance manifest, always written on a successful conversion. Notes go
    # to err_console (stderr) so the --json stdout verdict stays byte-deterministic.
    _write_conversion_manifest(
        report,
        args.src,
        args.dst,
        command="convert decomp",
        discover=discover_decomp_files,
        console=err_console,
    )

    # Runs after the manifest write and BEFORE ``--validate`` so validation
    # sees the patched ``config.json`` (``policy.boundary``); cobre infers the
    # inflow-lag depth from the boundary, so no ``state_space`` is written.
    # Cut-files-absent, the capability probe, and the importer all funnel
    # through this one broad ``except`` mapped to exit 1 (a conversion-step
    # failure), not the ``--validate`` exit-2 idiom — this is a conversion step,
    # not a validation gate. The call runs inside a ``dx.collect()`` sink so its
    # ``Diagnostic``s reach the Rich panels and the ``--json`` verdict instead
    # of degrading to invisible log records.
    boundary_diagnostics: list[Diagnostic] = []
    # The boundary FCF is imported by default; ``--no-fcf`` skips the whole
    # step (and its deck discovery). With it on, the deck's own FC records
    # (or the cortes* glob) locate the cut files; a deck that declares none
    # simply converts without a boundary FCF (an INFO note, not an error).
    # The single `DecompCase` built here is handed to the importer below, so
    # the deck is discovered (and, on first attribute access, parsed) once
    # for the FCF step rather than the importer re-discovering it itself.
    case = DecompCase.from_directory(args.src) if not args.no_fcf else None
    fcf_cut_files_present = (
        case is not None
        and case.files.cortesh is not None
        and case.files.cortes is not None
    )
    if not args.no_fcf and not fcf_cut_files_present:
        print_status(
            "Note: the deck declares no cortes/cortesh files; converting "
            "without a boundary FCF.",
            console=err_console,
            style="#F5A623",
        )
    if fcf_cut_files_present:
        assert case is not None  # narrowed by fcf_cut_files_present
        from cobre_bridge.core import diagnostics as dx

        fcf_diags: list[Diagnostic] = []
        try:
            from cobre_bridge.decomp.fcf.capability import (
                ensure_boundary_fcf_capability,
            )
            from cobre_bridge.decomp.fcf.importer import import_boundary_fcf

            ensure_boundary_fcf_capability()

            with dx.collect() as fcf_diags, tempfile.TemporaryDirectory() as work_dir:
                import_boundary_fcf(
                    args.dst,
                    case,
                    work_dir=Path(work_dir),
                    # Never None: a None cost_scale_factor triggers cobre's
                    # legacy 1e6 scaling — the source cuts are authored in
                    # cobre's native scale already.
                    cost_scale_factor=1.0,
                    config=fcf_inputs.config,
                    initial_conditions=fcf_inputs.initial_conditions,
                )
        except Exception as exc:  # noqa: BLE001
            diag = diagnostic_from_exception(exc, context="Boundary FCF import")
            failure_diagnostics = [*report.diagnostics, *fcf_diags, diag]
            if args.diagnostics_json is not None:
                # The sidecar carries the same merged set as the failure
                # verdict below: the converter's own findings, whatever the
                # importer's sink captured before it raised, and the failure
                # diagnostic itself.
                _write_diagnostics_json(
                    report,
                    args.diagnostics_json,
                    diagnostics=failure_diagnostics,
                    console=err_console,
                )
            if args.json_output:
                _emit_convert_json(
                    build_verdict(
                        "convert decomp",
                        _convert_status(failure_diagnostics, success="ok"),
                        summary,
                        failure_diagnostics,
                    )
                )
            else:
                render_diagnostics(
                    [*fcf_diags, diag],
                    heading="Conversion notes",
                    console=err_console,
                    quiet=args.quiet,
                )
            raise typer.Exit(code=1)
        else:
            boundary_diagnostics = list(fcf_diags)
            print_status(
                f"Boundary FCF imported. Run this case with: cobre run {args.dst}",
                console=err_console,
            )
            summary["boundary_fcf"] = {
                "imported": True,
                "path": "boundary",
            }
            if not args.json_output:
                # boundary_diagnostics only: ``report.diagnostics`` was
                # already rendered above (the converter's own panel), so
                # this renders solely the importer's captured diagnostics —
                # never a double-render of the same findings.
                render_diagnostics(
                    boundary_diagnostics,
                    heading="Conversion notes",
                    console=err_console,
                    quiet=args.quiet,
                )

    # The merge of the converter's own findings and the boundary-FCF
    # importer's (empty when ``--boundary-fcf`` was not requested, or the
    # sink never captured anything). Computed once here so both the
    # ``--diagnostics-json`` sidecar and the ``--json`` verdict below emit the
    # identical merged set.
    combined_diagnostics = [*report.diagnostics, *boundary_diagnostics]

    if args.diagnostics_json is not None:
        # The --diagnostics-json sidecar coexists with --json (both can be
        # set); it always carries the same merged diagnostics as the
        # eventual verdict, on the success path handled here.
        _write_diagnostics_json(
            report,
            args.diagnostics_json,
            diagnostics=combined_diagnostics,
            console=err_console,
        )

    validation_failed = False
    if args.validate:
        validation_failed = _run_cobre_validation(
            args.dst,
            command="convert decomp",
            summary=summary,
            json_output=args.json_output,
            err_console=err_console,
            whitelist_substrings=_DECOMP_VALIDATION_WHITELIST,
        )

    # ``status`` is derived from ``combined_diagnostics`` (importer diagnostics
    # are INFO-only, so this stays "ok" whenever the converter's own
    # diagnostics allow it) and feeds both the --json verdict below and the
    # exit-code gate, so it is computed unconditionally.
    status = _convert_status(combined_diagnostics, success="ok")

    # Emit the --json verdict now (after validation has populated ``summary``).
    if args.json_output:
        _emit_convert_json(
            build_verdict(
                "convert decomp",
                status,
                summary,
                combined_diagnostics,
            )
        )

    _gate_convert_exit(status, validation_failed=validation_failed)
