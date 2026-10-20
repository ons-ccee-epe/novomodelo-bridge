"""Pipeline orchestration: full export from Novomodelo policy to DECOMP binaries."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from novomodelo_bridge.core.diagnostics import Diagnostic, Severity, emit
from novomodelo_bridge.dessem.exporters.fcf.content.audit_binary_mirror import (
    content_csv_path,
    write_cortdeco_content_csv,
    write_mapcut_content_csv,
)
from novomodelo_bridge.dessem.exporters.fcf.core.cortdeco_writer import (
    CutInput,
    write_cortdeco,
    zeroed_gnl_block,
)
from novomodelo_bridge.dessem.exporters.fcf.core.cross_check import (
    cross_check_pair,
    reject_pair,
)
from novomodelo_bridge.dessem.exporters.fcf.core.inputs_reader import (
    CaseInputs,
    anticipated_thermals,
    read_case_inputs,
)
from novomodelo_bridge.dessem.exporters.fcf.core.layout import (
    LayoutError,
    assert_no_travel_time,
    assert_uniform_blocks,
    cortdeco_block_offsets,
    cut_head_indices,
)
from novomodelo_bridge.dessem.exporters.fcf.core.mapcut_writer import (
    MapcutHeader,
    write_mapcut,
)
from novomodelo_bridge.dessem.exporters.fcf.core.mapper import (
    MappingError,
    cut_building_pools,
    discount_factors,
    downstream_positions,
    first_node_per_stage,
    gnl_lag_months,
    inflow_lag_drop_audit,
    match_source_hydros,
    plant_codes,
    select_submarket_buses,
    storage_plant_ids,
    submarket_for_bus,
    tree_indices,
)
from novomodelo_bridge.dessem.exporters.fcf.core.reader import (
    PolicyFormatError,
    nodes_by_pool,
    read_policy,
    select_pools,
)
from novomodelo_bridge.dessem.exporters.fcf.eco.audit_raw_cuts import emit_eco_csvs
from novomodelo_bridge.dessem.exporters.fcf.manifest import (
    build_export_manifest,
    write_export_manifest,
)

if TYPE_CHECKING:
    from novomodelo_bridge.dessem.exporters.fcf.source import SourceDeck

_LOG = logging.getLogger(__name__)


def _emit_export_warnings() -> None:
    """Emit diagnostics for known export limitations (P3 and P14).

    Called once per export run to inform the user that certain coefficients
    are not yet supported and are emitted as zeros.
    """
    emit(
        Diagnostic(
            code="dessem-fcf-travel-time-unsupported",
            severity=Severity.WARNING,
            category="FCF export limitations",
            title="Travel-time coefficients not supported",
            summary=(
                "Water travel-time (tempo de viagem) coefficients are not yet "
                "supported: the export writes no travel-time block."
            ),
            notes=[
                "Premise P3: n_utv = 0, so the cuts carry no pi_qdefp coefficients.",
            ],
            remediation=(
                "A policy whose cuts carry travel-time state is refused rather "
                "than exported without it."
            ),
        )
    )
    emit(
        Diagnostic(
            code="dessem-fcf-gnl-coefficients-zeroed",
            severity=Severity.WARNING,
            category="FCF export limitations",
            title="GNL coefficients emitted as zeros",
            summary=(
                "Anticipated thermal (GNL) coefficients are intentionally "
                "emitted as zeros; GNL sensitivity is not captured."
            ),
            notes=[
                "Premise P14: Mapping Novomodelo's anticipated-thermal ring positions "
                "to DECOMP's (submarket, stage, block) address is unresolved.",
                "The exported cuts model hydro storage value but not GNL "
                "commitment value.",
            ],
            remediation=(
                "The pi_gnl block is structurally valid but zero-filled. "
                "DESSEM will read the file without errors."
            ),
        )
    )


# Phase labels for progress reporting (order matches run_export steps)
EXPORT_PHASE_LABELS: tuple[str, ...] = (
    "Reading inputs",
    "Reading policy checkpoint",
    "Selecting pools",
    "Validating premises",
    "Emitting ECO CSVs",
    "Building mappings",
    "Assembling mapcut header",
    "Writing mapcut",
    "Assembling cut inputs",
    "Writing cortdeco",
    "Cross-checking pair",
    "Writing content CSVs",
    "Writing manifest",
)


@dataclass(frozen=True)
class ExportSettings:
    """Export pipeline settings."""

    eco_subdirectory: str = "eco"
    content_subdirectory: str = "content"
    include_terminal_pool: bool = False
    emit_audit_csv: bool = False


@dataclass(frozen=True)
class OutputPaths:
    """Paths to output artifacts."""

    mapcut: Path
    cortdeco: Path
    manifest: Path
    mapcut_content: Path
    cortdeco_content: Path


def _resolve_output_paths(
    output_dir: Path, revision: str, settings: ExportSettings
) -> OutputPaths:
    """Resolve all output paths."""
    return OutputPaths(
        mapcut=output_dir / f"mapcut.{revision}",
        cortdeco=output_dir / f"cortdeco.{revision}",
        manifest=output_dir / "export_manifest.json",
        mapcut_content=content_csv_path(
            output_dir, settings.content_subdirectory, "mapcut_content.csv"
        ),
        cortdeco_content=content_csv_path(
            output_dir, settings.content_subdirectory, "cortdeco_content.csv"
        ),
    )


@contextmanager
def _log_step(step_name: str) -> Generator[None, None, None]:
    """Context manager that logs step start/finish with timing."""
    _LOG.info("starting: %s", step_name)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        _LOG.info("finished: %s (%.3fs)", step_name, elapsed)


def _write_failure_manifest(
    case_dir: Path,
    source: SourceDeck,
    settings: ExportSettings,
    paths: OutputPaths,
    failed_step: str,
) -> None:
    """Write a failure manifest, suppressing errors."""
    try:
        manifest = build_export_manifest(
            case_path=case_dir,
            source_path=source.directory,
            revision=source.revision,
            settings_snapshot=asdict(settings),
            outputs={
                "mapcut": paths.mapcut,
                "cortdeco": paths.cortdeco,
            },
            status="failed",
            failed_step=failed_step,
        )
        write_export_manifest(manifest, paths.manifest)
    except OSError as exc:
        _LOG.warning("failed to write failure manifest: %s", exc)


def _assemble_mapcut_header(
    manifest_data: Any,
    pools: dict,
    trunk_pool_ids: tuple[int, ...],
    inputs: CaseInputs,
    plant_ids: tuple[int, ...],
    hydro_codes: tuple[int, ...],
    gnl_thermals: tuple,
    gnl_submarket_codes: tuple[int, ...],
    gnl_lags: tuple[int, ...],
    n_patamares: int,
) -> MapcutHeader:
    """Assemble the mapcut header from all sources."""
    from novomodelo_bridge.dessem.exporters.fcf.core.reader import PolicyManifest

    manifest: PolicyManifest = manifest_data

    # Calculate derived quantities
    n_nodes = len(trunk_pool_ids)
    numero_cortes = sum(
        pools[pid].populated_count - pools[pid].warm_start_count
        for pid in trunk_pool_ids
    )
    numero_cenarios = len(manifest.nodes)
    n_estagios = manifest.num_stages

    patamares_gnl = tuple(n_patamares for _ in gnl_thermals)

    # Discount factors
    factors = discount_factors(inputs.stages, inputs.annual_discount_rate)

    # Head indices
    heads = cut_head_indices(numero_cortes, n_nodes, numero_cenarios)

    # Get date from first stage
    start_date = inputs.stages[0].start_date
    dia = int(start_date[8:10])
    mes = int(start_date[5:7])
    ano = int(start_date[0:4])

    return MapcutHeader(
        numero_iteracoes=manifest.completed_iterations,
        numero_cortes=numero_cortes,
        numero_submercados=len(select_submarket_buses(inputs.buses)),
        numero_uhes=len(hydro_codes),
        numero_cenarios=numero_cenarios,
        numero_estagios=n_estagios,
        numero_semanas=n_nodes,
        n_utv=0,  # P3
        dia=dia,
        mes=mes,
        ano=ano,
        codigos_uhes=hydro_codes,
        codigos_uhes_jusante=downstream_positions(inputs.hydros, plant_ids),
        indice_no_arvore=tree_indices(manifest),
        indice_primeiro_no_estagio=first_node_per_stage(manifest),
        patamares_por_estagio=tuple(len(s.blocks) for s in inputs.stages),
        registro_ultimo_corte_no=heads,
        codigos_submercados_gnl=gnl_submarket_codes,
        lag_meses_gnl=gnl_lags,
        patamares_gnl=patamares_gnl,
        taxa_desconto=factors,
        max_lag=0,
    )


def _assemble_cuts(
    pools: dict,
    trunk_pool_ids: tuple[int, ...],
    plant_ids: tuple[int, ...],
    offsets: Any,
) -> list[list[CutInput]]:
    """Assemble cut inputs from pools."""
    from novomodelo_bridge.dessem.exporters.fcf.core.cortdeco_writer import (
        storage_coefficients,
    )

    cuts: list[list[CutInput]] = []
    for pool_id in trunk_pool_ids:
        pool = pools[pool_id]
        node_cuts: list[CutInput] = []
        for piece in pool.pieces:
            if piece.slot_index < pool.warm_start_count:
                continue
            pi_varm = storage_coefficients(piece, pool.slots, plant_ids)
            pi_gnl = zeroed_gnl_block(offsets)
            node_cuts.append(
                CutInput(
                    intercept=piece.intercept,
                    pi_varm=pi_varm,
                    pi_gnl=pi_gnl,
                )
            )
        cuts.append(node_cuts)
    return cuts


def run_export(
    case_dir: Path,
    source: SourceDeck,
    output_dir: Path,
    settings: ExportSettings,
    *,
    on_phase: Callable[[str], None] | None = None,
) -> int:
    """Run full export pipeline.

    *on_phase* is an optional callback invoked at the start of each phase with the
    phase label (see EXPORT_PHASE_LABELS). Used by the CLI to drive a progress bar.

    *source* is the deck the case was converted from; it supplies the output
    revision and the plant codes.

    Returns exit code: 0 success, non-zero failure.
    """
    revision = source.revision
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _resolve_output_paths(output_dir, revision, settings)

    def _step(label: str) -> None:
        if on_phase is not None:
            on_phase(label)

    current_step = "initialize"
    unvalidated_pair_published = False

    # Emit warnings about known limitations upfront.
    _emit_export_warnings()

    try:
        current_step = "read settings-derived inputs"
        _step(EXPORT_PHASE_LABELS[0])
        with _log_step(current_step):
            inputs = read_case_inputs(case_dir)
            match_source_hydros(inputs.hydros, source.hydros)

        current_step = "read the policy checkpoint"
        _step(EXPORT_PHASE_LABELS[1])
        with _log_step(current_step):
            checkpoint = read_policy(case_dir)
            manifest = checkpoint.manifest

        current_step = "select the trunk pools"
        _step(EXPORT_PHASE_LABELS[2])
        with _log_step(current_step):
            pool_ids = sorted(nodes_by_pool(manifest))
            pools = select_pools(
                checkpoint, pool_ids, include_terminal=settings.include_terminal_pool
            )
            trunk_pool_ids = cut_building_pools(pools, len(pool_ids) - 1)
            terminal_pool_id = pool_ids[-1]

        current_step = "validate premises"
        _step(EXPORT_PHASE_LABELS[3])
        with _log_step(current_step):
            # P3: No travel time
            first_pool = pools[trunk_pool_ids[0]]
            assert_no_travel_time(first_pool.slots)

            # P8: Audit inflow lag drops
            inflow_lag_drop_audit(pools)

            gnl_thermals = anticipated_thermals(inputs)
            gnl_lags = gnl_lag_months(gnl_thermals, source.gnl_thermals)

        current_step = "emit ECO CSVs"
        _step(EXPORT_PHASE_LABELS[4])
        with _log_step(current_step):
            if settings.emit_audit_csv:
                emit_eco_csvs(
                    pools,
                    pool_ids,
                    output_dir,
                    settings.eco_subdirectory,
                    terminal_pool_id,
                    settings.include_terminal_pool,
                )

        current_step = "build mapping quantities"
        _step(EXPORT_PHASE_LABELS[5])
        with _log_step(current_step):
            plant_ids = storage_plant_ids(pools[trunk_pool_ids[0]].slots)
            hydro_codes = plant_codes(plant_ids, source.hydros)

            # P6: GNL submarket codes
            gnl_submarket_codes = tuple(
                submarket_for_bus(t.bus_id) for t in gnl_thermals
            )

            # Uniform blocks for P3/NCOEF
            n_patamares = assert_uniform_blocks(
                tuple(len(s.blocks) for s in inputs.stages)
            )

            offsets = cortdeco_block_offsets(
                n_uhes=len(hydro_codes),
                n_utv=0,  # P3
                max_lag=0,
                n_sbm_gnl=len(gnl_submarket_codes),
                n_estagios=manifest.num_stages,
                n_patamares=n_patamares,
            )

        current_step = "assemble mapcut header"
        _step(EXPORT_PHASE_LABELS[6])
        with _log_step(current_step):
            header = _assemble_mapcut_header(
                manifest,
                pools,
                trunk_pool_ids,
                inputs,
                plant_ids,
                hydro_codes,
                gnl_thermals,
                gnl_submarket_codes,
                gnl_lags,
                n_patamares,
            )

        current_step = "write mapcut"
        _step(EXPORT_PHASE_LABELS[7])
        unvalidated_pair_published = True
        with _log_step(current_step):
            write_mapcut(header, paths.mapcut)

        current_step = "assemble cut inputs"
        _step(EXPORT_PHASE_LABELS[8])
        with _log_step(current_step):
            cuts = _assemble_cuts(pools, trunk_pool_ids, plant_ids, offsets)

        current_step = "write cortdeco"
        _step(EXPORT_PHASE_LABELS[9])
        with _log_step(current_step):
            write_cortdeco(
                cuts,
                paths.cortdeco,
                numero_cortes=header.numero_cortes,
                offsets=offsets,
            )

        current_step = "cross-check the pair"
        _step(EXPORT_PHASE_LABELS[10])
        with _log_step(current_step):
            mapcut_contents, cortdeco_contents = cross_check_pair(
                paths.mapcut, paths.cortdeco
            )
            unvalidated_pair_published = False  # Validated

        current_step = "write content CSVs"
        _step(EXPORT_PHASE_LABELS[11])
        with _log_step(current_step):
            if settings.emit_audit_csv:
                write_mapcut_content_csv(mapcut_contents, paths.mapcut_content)
                write_cortdeco_content_csv(
                    cortdeco_contents,
                    paths.cortdeco_content,
                    hydro_codes=header.codigos_uhes,
                    submarket_codes=header.codigos_submercados_gnl,
                    n_estagios=header.numero_estagios,
                    n_patamares=n_patamares,
                )

        current_step = "write export manifest"
        _step(EXPORT_PHASE_LABELS[12])
        with _log_step(current_step):
            outputs: dict[str, Path] = {
                "mapcut": paths.mapcut,
                "cortdeco": paths.cortdeco,
            }
            if settings.emit_audit_csv:
                outputs["mapcut_content"] = paths.mapcut_content
                outputs["cortdeco_content"] = paths.cortdeco_content
            manifest_obj = build_export_manifest(
                case_path=case_dir,
                source_path=source.directory,
                revision=revision,
                settings_snapshot=asdict(settings),
                outputs=outputs,
            )
            write_export_manifest(manifest_obj, paths.manifest)

        _LOG.info("export completed successfully")
        return 0

    except (LayoutError, MappingError, PolicyFormatError) as exc:
        _LOG.error("export failed at step '%s': %s", current_step, exc)
        emit(
            Diagnostic(
                code="dessem-fcf-export-failed",
                severity=Severity.ERROR,
                category="Export failure",
                title="Export failed",
                summary=str(exc),
                notes=[f"Failed step: {current_step}"],
            )
        )
        if unvalidated_pair_published:
            reject_pair(paths.mapcut, paths.cortdeco)
        _write_failure_manifest(case_dir, source, settings, paths, current_step)
        return 1

    except BaseException as exc:
        _LOG.error("export failed at step '%s': %s", current_step, exc)
        if unvalidated_pair_published:
            reject_pair(paths.mapcut, paths.cortdeco)
        _write_failure_manifest(case_dir, source, settings, paths, current_step)
        raise
