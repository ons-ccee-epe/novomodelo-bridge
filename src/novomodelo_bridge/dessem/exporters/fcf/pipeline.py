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
    CortdecoContents,
    MapcutContents,
    cross_check_pair,
    reject_pair,
)
from novomodelo_bridge.dessem.exporters.fcf.core.inputs_reader import (
    CaseInputs,
    StageInfo,
    anticipated_thermals,
    read_case_inputs,
)
from novomodelo_bridge.dessem.exporters.fcf.core.layout import (
    CutBlockOffsets,
    LayoutError,
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
    StageCutPool,
    nodes_by_pool,
    read_policy,
    select_pools,
)
from novomodelo_bridge.dessem.exporters.fcf.core.travel_time import (
    TravelTimePlant,
    max_lag,
    release_lag_weights,
    travel_time_plants,
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
    """Emit the diagnostic for the known export limitation P14, once per run."""
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


def _emit_travel_time_info(plants: tuple[TravelTimePlant, ...]) -> None:
    """Report the plants whose in-transit water the cuts price (premise P3)."""
    if not plants:
        return
    emit(
        Diagnostic(
            code="dessem-fcf-travel-time-exported",
            severity=Severity.INFO,
            category="FCF export",
            title="Travel-time coefficients exported",
            summary=(
                f"The cuts value the water in transit from {len(plants)} "
                "plant(s), by the stage it was released in, up to "
                f"{max_lag(plants)} stage(s) back."
            ),
            notes=[
                f"Plant {plant.code}: its water reaches the plant downstream "
                f"{plant.hours} h after release."
                for plant in plants
            ]
            + [
                "Premise P3: a release spread evenly over its stage arrives "
                "spread evenly over an equally long span; release periods "
                "before the study are as long as its first stage."
            ],
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
    travel_time: tuple[TravelTimePlant, ...],
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
        tempo_viagem=travel_time,
    )


def _assemble_cuts(
    pools: dict,
    trunk_pool_ids: tuple[int, ...],
    plant_ids: tuple[int, ...],
    offsets: Any,
    travel_time: tuple[TravelTimePlant, ...],
    stages: tuple[StageInfo, ...],
) -> list[list[CutInput]]:
    """Assemble cut inputs from pools."""
    from novomodelo_bridge.dessem.exporters.fcf.core.cortdeco_writer import (
        storage_coefficients,
    )

    cuts: list[list[CutInput]] = []
    for pool_id in trunk_pool_ids:
        pool = pools[pool_id]
        release_weights = release_lag_weights(
            pool.slots, travel_time, stages, pool.graph_stage_id, pool_id
        )
        node_cuts: list[CutInput] = []
        for piece in pool.pieces:
            if piece.slot_index < pool.warm_start_count:
                continue
            node_cuts.append(
                CutInput(
                    intercept=piece.intercept,
                    pi_varm=storage_coefficients(piece, pool.slots, plant_ids),
                    pi_qdefp=release_weights @ piece.coefficients,
                    pi_gnl=zeroed_gnl_block(offsets),
                )
            )
        cuts.append(node_cuts)
    return cuts


@dataclass
class _ExportProgress:
    """The step an export run is in, which ``run_export``'s failure handlers read."""

    on_phase: Callable[[str], None] | None
    current_step: str = "initialize"
    unvalidated_pair_published: bool = False

    @contextmanager
    def step(
        self, step_name: str, phase: int, *, publishes_pair: bool = False
    ) -> Generator[None, None, None]:
        """Enter *step_name*, report ``EXPORT_PHASE_LABELS[phase]``, log timing."""
        self.current_step = step_name
        if self.on_phase is not None:
            self.on_phase(EXPORT_PHASE_LABELS[phase])
        if publishes_pair:
            self.unvalidated_pair_published = True
        with _log_step(step_name):
            yield


@dataclass(frozen=True)
class _ExportPlan:
    """Everything the binary pair and its content CSVs are written from."""

    pools: dict[int, StageCutPool]
    trunk_pool_ids: tuple[int, ...]
    plant_ids: tuple[int, ...]
    travel_time: tuple[TravelTimePlant, ...]
    stages: tuple[StageInfo, ...]
    n_patamares: int
    offsets: CutBlockOffsets
    header: MapcutHeader


def _plan_export(
    case_dir: Path,
    source: SourceDeck,
    output_dir: Path,
    settings: ExportSettings,
    progress: _ExportProgress,
) -> _ExportPlan:
    """Read the case and its policy, check the premises, and lay out the pair."""
    with progress.step("read settings-derived inputs", 0):
        inputs = read_case_inputs(case_dir)
        match_source_hydros(inputs.hydros, source.hydros)

    with progress.step("read the policy checkpoint", 1):
        checkpoint = read_policy(case_dir)
        manifest = checkpoint.manifest

    with progress.step("select the trunk pools", 2):
        pool_ids = sorted(nodes_by_pool(manifest))
        pools = select_pools(
            checkpoint, pool_ids, include_terminal=settings.include_terminal_pool
        )
        trunk_pool_ids = cut_building_pools(pools, len(pool_ids) - 1)
        terminal_pool_id = pool_ids[-1]

    with progress.step("validate premises", 3):
        travel_time = travel_time_plants(
            inputs.hydros, source.hydros, source.travel_time_codes, inputs.stages
        )
        _emit_travel_time_info(travel_time)

        # P8: Audit inflow lag drops
        inflow_lag_drop_audit(pools)

        gnl_thermals = anticipated_thermals(inputs)
        gnl_lags = gnl_lag_months(gnl_thermals, source.gnl_thermals)

    with progress.step("emit ECO CSVs", 4):
        if settings.emit_audit_csv:
            emit_eco_csvs(
                pools,
                pool_ids,
                output_dir,
                settings.eco_subdirectory,
                terminal_pool_id,
                settings.include_terminal_pool,
            )

    with progress.step("build mapping quantities", 5):
        plant_ids = storage_plant_ids(pools[trunk_pool_ids[0]].slots)
        hydro_codes = plant_codes(plant_ids, source.hydros)

        # P6: GNL submarket codes
        gnl_submarket_codes = tuple(submarket_for_bus(t.bus_id) for t in gnl_thermals)

        # Uniform blocks for P3/NCOEF
        n_patamares = assert_uniform_blocks(tuple(len(s.blocks) for s in inputs.stages))

        offsets = cortdeco_block_offsets(
            n_uhes=len(hydro_codes),
            n_utv=len(travel_time),
            max_lag=max_lag(travel_time),
            n_sbm_gnl=len(gnl_submarket_codes),
            n_estagios=manifest.num_stages,
            n_patamares=n_patamares,
        )

    with progress.step("assemble mapcut header", 6):
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
            travel_time,
        )

    return _ExportPlan(
        pools=pools,
        trunk_pool_ids=trunk_pool_ids,
        plant_ids=plant_ids,
        travel_time=travel_time,
        stages=inputs.stages,
        n_patamares=n_patamares,
        offsets=offsets,
        header=header,
    )


def _write_validated_pair(
    plan: _ExportPlan, paths: OutputPaths, progress: _ExportProgress
) -> tuple[MapcutContents, CortdecoContents]:
    """Write mapcut and cortdeco, then cross-check the pair as read back."""
    with progress.step("write mapcut", 7, publishes_pair=True):
        write_mapcut(plan.header, paths.mapcut)

    with progress.step("assemble cut inputs", 8):
        cuts = _assemble_cuts(
            plan.pools,
            plan.trunk_pool_ids,
            plan.plant_ids,
            plan.offsets,
            plan.travel_time,
            plan.stages,
        )

    with progress.step("write cortdeco", 9):
        write_cortdeco(
            cuts,
            paths.cortdeco,
            numero_cortes=plan.header.numero_cortes,
            offsets=plan.offsets,
        )

    with progress.step("cross-check the pair", 10):
        contents = cross_check_pair(paths.mapcut, paths.cortdeco)
        progress.unvalidated_pair_published = False
    return contents


def _write_content_csvs(
    plan: _ExportPlan,
    mapcut_contents: MapcutContents,
    cortdeco_contents: CortdecoContents,
    paths: OutputPaths,
) -> None:
    """Write the audit CSV mirrors of the read-back pair."""
    write_mapcut_content_csv(mapcut_contents, paths.mapcut_content)
    write_cortdeco_content_csv(
        cortdeco_contents,
        paths.cortdeco_content,
        hydro_codes=plan.header.codigos_uhes,
        submarket_codes=plan.header.codigos_submercados_gnl,
        n_estagios=plan.header.numero_estagios,
        n_patamares=plan.n_patamares,
        travel_time_codes=tuple(plant.code for plant in plan.travel_time),
        max_lag=plan.header.max_lag,
    )


def _write_success_manifest(
    case_dir: Path,
    source: SourceDeck,
    settings: ExportSettings,
    paths: OutputPaths,
) -> None:
    """Write the manifest of a completed export."""
    outputs: dict[str, Path] = {
        "mapcut": paths.mapcut,
        "cortdeco": paths.cortdeco,
    }
    if settings.emit_audit_csv:
        outputs["mapcut_content"] = paths.mapcut_content
        outputs["cortdeco_content"] = paths.cortdeco_content
    manifest = build_export_manifest(
        case_path=case_dir,
        source_path=source.directory,
        revision=source.revision,
        settings_snapshot=asdict(settings),
        outputs=outputs,
    )
    write_export_manifest(manifest, paths.manifest)


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
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _resolve_output_paths(output_dir, source.revision, settings)
    progress = _ExportProgress(on_phase)

    _emit_export_warnings()

    try:
        plan = _plan_export(case_dir, source, output_dir, settings, progress)
        mapcut_contents, cortdeco_contents = _write_validated_pair(
            plan, paths, progress
        )

        with progress.step("write content CSVs", 11):
            if settings.emit_audit_csv:
                _write_content_csvs(plan, mapcut_contents, cortdeco_contents, paths)

        with progress.step("write export manifest", 12):
            _write_success_manifest(case_dir, source, settings, paths)

        _LOG.info("export completed successfully")
        return 0

    except (LayoutError, MappingError, PolicyFormatError) as exc:
        _LOG.error("export failed at step '%s': %s", progress.current_step, exc)
        emit(
            Diagnostic(
                code="dessem-fcf-export-failed",
                severity=Severity.ERROR,
                category="Export failure",
                title="Export failed",
                summary=str(exc),
                notes=[f"Failed step: {progress.current_step}"],
            )
        )
        if progress.unvalidated_pair_published:
            reject_pair(paths.mapcut, paths.cortdeco)
        _write_failure_manifest(
            case_dir, source, settings, paths, progress.current_step
        )
        return 1

    except BaseException as exc:
        _LOG.error("export failed at step '%s': %s", progress.current_step, exc)
        if progress.unvalidated_pair_published:
            reject_pair(paths.mapcut, paths.cortdeco)
        _write_failure_manifest(
            case_dir, source, settings, paths, progress.current_step
        )
        raise
