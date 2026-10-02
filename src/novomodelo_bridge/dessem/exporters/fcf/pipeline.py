"""Pipeline orchestration: full export from Novomodelo policy to DECOMP binaries.

The pipeline orchestrates the full export: read inputs, apply mappings, write
binaries, cross-check, emit CSVs, and write manifest. It handles failure paths
and ensures manifest is written even on errors.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from novomodelo_bridge.core.diagnostics import Diagnostic, Severity, emit
from novomodelo_bridge.dessem.exporters.fcf.content_csv import (
    content_csv_path,
    write_cortdeco_content_csv,
    write_mapcut_content_csv,
)
from novomodelo_bridge.dessem.exporters.fcf.cortdeco_writer import (
    CutInput,
    write_cortdeco,
    zeroed_gnl_block,
)
from novomodelo_bridge.dessem.exporters.fcf.cross_check import (
    cross_check_pair,
    reject_pair,
)
from novomodelo_bridge.dessem.exporters.fcf.eco_csv import emit_eco_csvs
from novomodelo_bridge.dessem.exporters.fcf.inputs_reader import (
    CaseInputs,
    anticipated_thermals,
    read_case_inputs,
)
from novomodelo_bridge.dessem.exporters.fcf.layout import (
    LayoutError,
    assert_no_travel_time,
    assert_uniform_blocks,
    cortdeco_block_offsets,
    cut_head_indices,
)
from novomodelo_bridge.dessem.exporters.fcf.manifest import (
    build_export_manifest,
    write_export_manifest,
)
from novomodelo_bridge.dessem.exporters.fcf.mapcut_writer import (
    MapcutHeader,
    write_mapcut,
)
from novomodelo_bridge.dessem.exporters.fcf.mapper import (
    EVIDENCED_GNL_LAG_MESES,
    MappingError,
    assert_gnl_lead_time_is_evidenced,
    cut_building_pools,
    discount_factors,
    first_node_per_stage,
    inflow_lag_drop_audit,
    load_hydro_codes,
    select_submarket_buses,
    submarket_for_bus,
    tree_indices,
)
from novomodelo_bridge.dessem.exporters.fcf.reader import (
    POLICY_MANIFEST_RELATIVE,
    PolicyFormatError,
    load_pools,
    nodes_by_pool,
    read_policy_manifest,
)

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
                "supported and are omitted from the export."
            ),
            notes=[
                "Premise P3: Novomodelo does not model water travel time between "
                "cascade stages, so n_utv = 0 and pi_qdefp coefficients are absent.",
            ],
            remediation=(
                "If the source policy includes travel-time state variables, "
                "their sensitivity is not captured in the exported FCF."
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
    "Reading policy manifest",
    "Loading pools",
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
    hydro_codes_path: str = ""
    include_terminal_pool: bool = False


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
    revision: str,
    settings: ExportSettings,
    paths: OutputPaths,
    failed_step: str,
) -> None:
    """Write a failure manifest, suppressing errors."""
    try:
        manifest = build_export_manifest(
            case_path=case_dir,
            revision=revision,
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
    hydro_codes: tuple[int, ...],
    gnl_thermals: tuple,
    gnl_submarket_codes: tuple[int, ...],
    n_patamares: int,
) -> MapcutHeader:
    """Assemble the mapcut header from all sources."""
    from novomodelo_bridge.dessem.exporters.fcf.reader import PolicyManifest

    manifest: PolicyManifest = manifest_data

    # Calculate derived quantities
    n_nodes = len(trunk_pool_ids)
    numero_cortes = sum(
        pools[pid].populated_count - pools[pid].warm_start_count
        for pid in trunk_pool_ids
    )
    numero_cenarios = len(manifest.nodes)
    n_estagios = manifest.num_stages

    # Downstream mapping (simplified: use 0 for terminal)
    downstream = []
    for hydro in inputs.hydros:
        if hydro.downstream_id is None:
            downstream.append(0)
        else:
            # Find the downstream hydro's code
            try:
                downstream_idx = next(
                    i
                    for i, h in enumerate(inputs.hydros)
                    if h.id == hydro.downstream_id
                )
                downstream.append(hydro_codes[downstream_idx])
            except StopIteration:
                downstream.append(0)

    # GNL lag months (P15: all 2 for evidenced plants)
    lag_meses = tuple(EVIDENCED_GNL_LAG_MESES for _ in gnl_thermals)
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
        codigos_uhes_jusante=tuple(downstream),
        indice_no_arvore=tree_indices(manifest),
        indice_primeiro_no_estagio=first_node_per_stage(manifest),
        patamares_por_estagio=tuple(len(s.blocks) for s in inputs.stages),
        registro_ultimo_corte_no=heads,
        codigos_submercados_gnl=gnl_submarket_codes,
        lag_meses_gnl=lag_meses,
        patamares_gnl=patamares_gnl,
        taxa_desconto=factors,
        max_lag=0,
    )


def _assemble_cuts(
    pools: dict,
    trunk_pool_ids: tuple[int, ...],
    hydro_codes: tuple[int, ...],
    offsets: Any,
) -> list[list[CutInput]]:
    """Assemble cut inputs from pools."""
    from novomodelo_bridge.dessem.exporters.fcf.cortdeco_writer import (
        storage_coefficients,
    )

    cuts: list[list[CutInput]] = []
    for pool_id in trunk_pool_ids:
        pool = pools[pool_id]
        node_cuts: list[CutInput] = []
        for piece in pool.pieces:
            if piece.slot_index >= pool.warm_start_count:
                # Not a warm-start cut, include it
                pass
            pi_varm = storage_coefficients(piece, pool.slots, hydro_codes)
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
    revision: str,
    output_dir: Path,
    settings: ExportSettings,
    *,
    on_phase: Callable[[str], None] | None = None,
) -> int:
    """Run full export pipeline.

    *on_phase* is an optional callback invoked at the start of each phase with the
    phase label (see EXPORT_PHASE_LABELS). Used by the CLI to drive a progress bar.

    Returns exit code: 0 success, non-zero failure.
    """
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
            hydro_codes = load_hydro_codes(Path(settings.hydro_codes_path))
            inputs = read_case_inputs(case_dir)

        current_step = "read the policy manifest"
        _step(EXPORT_PHASE_LABELS[1])
        with _log_step(current_step):
            manifest = read_policy_manifest(case_dir / POLICY_MANIFEST_RELATIVE)

        current_step = "read the trunk pools"
        _step(EXPORT_PHASE_LABELS[2])
        with _log_step(current_step):
            pool_ids = sorted(nodes_by_pool(manifest))
            pools = load_pools(
                case_dir, pool_ids, include_terminal=settings.include_terminal_pool
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

            # P15: Validate GNL lead times
            gnl_thermals = anticipated_thermals(inputs)
            assert_gnl_lead_time_is_evidenced(gnl_thermals)

        current_step = "emit ECO CSVs"
        _step(EXPORT_PHASE_LABELS[4])
        with _log_step(current_step):
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
                hydro_codes,
                gnl_thermals,
                gnl_submarket_codes,
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
            cuts = _assemble_cuts(pools, trunk_pool_ids, hydro_codes, offsets)

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
            manifest_obj = build_export_manifest(
                case_path=case_dir,
                revision=revision,
                settings_snapshot=asdict(settings),
                outputs={
                    "mapcut": paths.mapcut,
                    "cortdeco": paths.cortdeco,
                    "mapcut_content": paths.mapcut_content,
                    "cortdeco_content": paths.cortdeco_content,
                },
            )
            write_export_manifest(manifest_obj, paths.manifest)

        _LOG.info("export completed successfully")
        return 0

    except (LayoutError, MappingError, PolicyFormatError) as exc:
        _LOG.error("export failed at step '%s': %s", current_step, exc)
        if unvalidated_pair_published:
            reject_pair(paths.mapcut, paths.cortdeco)
        _write_failure_manifest(case_dir, revision, settings, paths, current_step)
        return 1

    except BaseException as exc:
        _LOG.error("export failed at step '%s': %s", current_step, exc)
        if unvalidated_pair_published:
            reject_pair(paths.mapcut, paths.cortdeco)
        _write_failure_manifest(case_dir, revision, settings, paths, current_step)
        raise
