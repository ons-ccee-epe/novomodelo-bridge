"""Pipeline orchestration: full export from Novomodelo policy to DECOMP binaries."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    emit,
)
from novomodelo_bridge.dessem.exporters.fcf.content.audit_binary_mirror import (
    content_csv_path,
    write_cortdeco_content_csv,
    write_mapcut_content_csv,
)
from novomodelo_bridge.dessem.exporters.fcf.core.cortdeco_writer import (
    CutInput,
    node_and_iteration,
    write_cortdeco,
)
from novomodelo_bridge.dessem.exporters.fcf.core.cross_check import (
    CortdecoContents,
    MapcutContents,
    cross_check_pair,
    reject_pair,
)
from novomodelo_bridge.dessem.exporters.fcf.core.inputs_reader import (
    CaseInputs,
    InputReadError,
    StageInfo,
    ThermalInfo,
    anticipated_thermals,
    read_case_inputs,
)
from novomodelo_bridge.dessem.exporters.fcf.core.layout import (
    COST_FIELDS,
    GNL_EXPORT_MIDDLE_OFFSET,
    CutBlockOffsets,
    LayoutError,
    assert_uniform_blocks,
    cortdeco_block_offsets,
    cortdeco_gnl_offset,
    cut_head_indices,
)
from novomodelo_bridge.dessem.exporters.fcf.core.mapcut_writer import (
    MapcutHeader,
    write_mapcut,
)
from novomodelo_bridge.dessem.exporters.fcf.core.mapper import (
    GnlCommittedCost,
    GnlDeliveries,
    MappingError,
    SourceGnlThermal,
    cut_building_pools,
    discount_factors,
    downstream_positions,
    first_node_per_stage,
    gnl_block,
    gnl_committed_costs,
    gnl_drop_audit,
    gnl_lag_months,
    inflow_lag_drop_audit,
    match_source_hydros,
    plant_codes,
    select_gnl_deliveries,
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
    UndatedBucketAudit,
    max_lag,
    release_lag_weights,
    travel_time_plants,
    undated_bucket_audit,
)
from novomodelo_bridge.dessem.exporters.fcf.eco.audit_raw_cuts import emit_eco_csvs
from novomodelo_bridge.dessem.exporters.fcf.manifest import (
    build_export_manifest,
    write_export_manifest,
)

if TYPE_CHECKING:
    from novomodelo_bridge.dessem.exporters.fcf.source import SourceDeck

_LOG = logging.getLogger(__name__)


def _emit_travel_time_info(
    plants: tuple[TravelTimePlant, ...], undated: UndatedBucketAudit
) -> None:
    """Report the plants whose in-transit water the cuts price (premise P3)."""
    if not plants:
        return
    notes = [
        f"Plant {plant.code}: its water reaches the plant downstream "
        f"{plant.hours} h after release."
        for plant in plants
    ] + [
        "Premise P3: a release spread evenly over its stage arrives "
        "spread evenly over an equally long span; release periods "
        "before the study are as long as its first stage."
    ]
    if undated.total_buckets:
        notes.append(
            f"{undated.total_buckets} transit bucket(s) carry no arrival window, "
            "Novomodelo's placeholders for water that would arrive after the "
            "study ends; they take no part in pi_qdefp."
        )
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
            notes=notes,
        )
    )


def _emit_undated_bucket_warning(undated: UndatedBucketAudit) -> None:
    """Warn when cuts price transit buckets premise P3 gives no weight."""
    if not undated.total_priced:
        return
    emit(
        Diagnostic(
            code="dessem-fcf-undated-buckets-priced",
            severity=Severity.WARNING,
            category="FCF export",
            title="Coefficients on undated transit buckets dropped",
            summary=(
                f"{undated.total_priced} cut coefficient(s) sit on transit buckets "
                "with no arrival window and are left out of pi_qdefp."
            ),
            notes=[
                f"Pool {count.pool_id}: {count.priced} coefficient(s) on "
                f"{count.buckets} undated bucket(s)."
                for count in undated.per_pool
                if count.priced
            ]
            + [
                "Premise P3: a release is priced through the buckets its water "
                "arrives in; a bucket with no arrival window has no release to charge."
            ],
            remediation=(
                "The exported cuts carry no value for the water those "
                "coefficients priced, so they can overstate the future cost "
                "and are no longer a guaranteed lower bound on it. Check "
                "whether the case's boundary future-cost function prices "
                "water that arrives after the study ends."
            ),
        )
    )


_GNL_MISSING_SHORT_LEAD = "lead time shorter than the delivery stage"
_GNL_MISSING_NO_SLOT = "no slot in the policy"
_GNL_MISSING_NOTES = {
    _GNL_MISSING_SHORT_LEAD: (
        "A plant whose lead time is shorter than the delivery stage carries no "
        "commitment for that delivery in the policy, so 0.0 is its exact "
        "coefficient."
    ),
    _GNL_MISSING_NO_SLOT: (
        "The policy should hold a slot for the plant's delivery and does not; "
        "0.0 is written, so the cuts do not price that plant's commitment. "
        "Check that the policy was produced from this case."
    ),
}


def _emit_gnl_delivery_missing(
    gnl_thermals: Sequence[ThermalInfo],
    source_plants: Sequence[SourceGnlThermal],
    deliveries: GnlDeliveries,
) -> None:
    """Report the GNL plants whose cuts carry 0.0 for want of a delivery slot."""
    stage = deliveries.stage
    if stage is None or None not in deliveries.positions:
        return
    rows: list[list[object]] = []
    causes: set[str] = set()
    for thermal, plant, position in zip(
        gnl_thermals, source_plants, deliveries.positions, strict=True
    ):
        if position is not None:
            continue
        lead = thermal.lead_time_hours
        assert lead is not None  # anticipated_thermals keeps only thermals with a lead
        cause = (
            _GNL_MISSING_SHORT_LEAD if lead < deliveries.hours else _GNL_MISSING_NO_SLOT
        )
        causes.add(cause)
        rows.append([plant.code, thermal.name, stage.start_date, lead, cause])
    emit(
        Diagnostic(
            code="dessem-fcf-gnl-delivery-missing",
            severity=Severity.WARNING,
            category="FCF export",
            title="GNL delivery missing from the policy",
            summary=(
                f"{len(rows)} GNL plant(s) have no coefficient for the delivery "
                "of the case's second stage; their cuts carry 0.0 for GNL."
            ),
            table=DiagnosticTable(
                columns=[
                    "GNL plant",
                    "Name",
                    "Delivery start",
                    "Lead time (h)",
                    "Cause",
                ],
                rows=rows,
                justify=["right", "left", "left", "right", "left"],
            ),
            notes=[
                note for cause, note in _GNL_MISSING_NOTES.items() if cause in causes
            ],
        )
    )


def _first_node_gnl_values(
    cortdeco: CortdecoContents,
    n_nodes: int,
    offsets: CutBlockOffsets,
    deliveries: GnlDeliveries,
) -> dict[int, tuple[float, ...]]:
    """Each delivered GNL entry's R$/MWh values over the first node's written cuts.

    Read from the validated pair, never from the in-memory `CutInput.pi_gnl`,
    which holds novomodelo's sign: `write_cortdeco` owns the on-disk sign and scale.
    """
    first_node = [
        cut
        for cut in cortdeco.cuts
        if not cut.is_extra_record
        and node_and_iteration(cut.record_index, n_nodes)[0] == 0
    ]
    values: dict[int, tuple[float, ...]] = {}
    for entry, position in enumerate(deliveries.positions):
        if position is None:
            continue
        index = (
            cortdeco_gnl_offset(offsets, entry, GNL_EXPORT_MIDDLE_OFFSET, 0)
            - offsets.pi_gnl
        )
        values[entry] = tuple(float(cut.pi_gnl[index]) for cut in first_node)
    return values


def _emit_gnl_coefficients_exported(
    cortdeco: CortdecoContents,
    n_nodes: int,
    offsets: CutBlockOffsets,
    deliveries: GnlDeliveries,
    gnl_thermals: Sequence[ThermalInfo],
    source_plants: Sequence[SourceGnlThermal],
    submarket_codes: Sequence[int],
    dropped: int,
) -> None:
    """Report the GNL coefficients the file holds and those it leaves out."""
    stage = deliveries.stage
    values = _first_node_gnl_values(cortdeco, n_nodes, offsets, deliveries)
    if stage is None or not values:
        return
    rows: list[list[object]] = [
        [
            plant.code,
            thermal.name,
            submarket,
            f"{stage.start_date} to {stage.end_date}",
            deliveries.hours,
            min(values[entry]),
            max(values[entry]),
        ]
        for entry, (thermal, plant, submarket) in enumerate(
            zip(gnl_thermals, source_plants, submarket_codes, strict=True)
        )
        if entry in values
    ]
    notes = [
        "Only the first cut-building node's cuts carry GNL coefficients.",
        "The policy holds one coefficient per delivery; the file writes it, "
        "unchanged, on every load block of the delivery.",
    ]
    if dropped:
        notes.append(
            f"{dropped} other GNL coefficients of the exported cuts are not "
            "written: the first cut-building node's other deliveries and every "
            "GNL coefficient of the later nodes' cuts. They are not folded into "
            "the cut intercept, so the cuts price those commitments as zero."
        )
    emit(
        Diagnostic(
            code="dessem-fcf-gnl-coefficients-exported",
            severity=Severity.INFO,
            category="FCF export",
            title="GNL coefficients exported",
            summary=(
                f"The cuts of the first cut-building node carry GNL coefficients "
                f"for {len(rows)} plant(s), at the delivery of the case's second "
                "stage."
            ),
            table=DiagnosticTable(
                columns=[
                    "GNL plant",
                    "Name",
                    "Submarket",
                    "Delivery",
                    "Hours",
                    "Min R$/MWh",
                    "Max R$/MWh",
                ],
                rows=rows,
                justify=["right", "left", "right", "left", "right", "right", "right"],
            ),
            notes=notes,
        )
    )


def _emit_gnl_coefficient_zero(
    cortdeco: CortdecoContents,
    n_nodes: int,
    offsets: CutBlockOffsets,
    deliveries: GnlDeliveries,
    gnl_thermals: Sequence[ThermalInfo],
    source_plants: Sequence[SourceGnlThermal],
) -> None:
    """Report the plants with a delivery whose first-node cuts hold an exact 0.0."""
    values = _first_node_gnl_values(cortdeco, n_nodes, offsets, deliveries)
    rows: list[list[object]] = [
        [plant.code, thermal.name, zeros, len(values[entry])]
        for entry, (thermal, plant) in enumerate(
            zip(gnl_thermals, source_plants, strict=True)
        )
        if entry in values and (zeros := values[entry].count(0.0))
    ]
    if not rows:
        return
    emit(
        Diagnostic(
            code="dessem-fcf-gnl-coefficient-zero",
            severity=Severity.WARNING,
            category="FCF export",
            title="GNL coefficient is zero in some cuts",
            summary=(
                f"{len(rows)} GNL plant(s) have cuts whose GNL coefficient is "
                "exactly 0.0 although the policy holds their delivery."
            ),
            table=DiagnosticTable(
                columns=["GNL plant", "Name", "Zero cuts", "Cuts"],
                rows=rows,
                justify=["right", "left", "right", "right"],
            ),
            notes=[
                "The zero is the policy's own value, written unchanged. A "
                "commitment at the plant's capacity is a known cause."
            ],
        )
    )


def _emit_gnl_committed_cost(
    mapcut: MapcutContents,
    costs: Sequence[GnlCommittedCost],
    source_plants: Sequence[SourceGnlThermal],
) -> None:
    """Report each stage's committed GNL cost as read back from `mapcut` record 10."""
    if not source_plants:
        return
    value_field = COST_FIELDS.index("geracao_termica_minima_gerada_gnl")
    rows: list[list[object]] = []
    for index, cost in enumerate(costs):
        entries = [
            f"{plant.code} {plant.name}: " + ("-" if mw is None else f"{mw:g} MW")
            for plant, mw in zip(source_plants, cost.committed_mw, strict=True)
        ]
        rows.append(
            [
                cost.stage.id,
                f"{cost.stage.start_date} to {cost.stage.end_date}",
                cost.hours,
                ", ".join(entries),
                mapcut.cost_records[index][value_field],
            ]
        )
    emit(
        Diagnostic(
            code="dessem-fcf-gnl-committed-cost-exported",
            severity=Severity.INFO,
            category="FCF export",
            title="GNL committed cost exported",
            summary=(
                "mapcut record 10 holds the cost of the GNL generation committed "
                f"before the study for each of the {len(rows)} stages."
            ),
            table=DiagnosticTable(
                columns=["Stage", "Period", "Hours", "GNL plants", "10³ R$"],
                rows=rows,
                justify=["right", "left", "right", "left", "right"],
            ),
            notes=[
                "Each stage's value is the sum, over the GNL plants, of the "
                "committed MW times the stage's load-block hours times the "
                "plant's unit cost in the case, in 10³ R$ and not discounted.",
                "A stage no commitment window covers is written as 0.0.",
            ],
        )
    )


def _emit_gnl_submarket_repeated(
    submarket_codes: Sequence[int], source_plants: Sequence[SourceGnlThermal]
) -> None:
    """Report GNL plants that share a submarket code."""
    plants_by_submarket: dict[int, list[int]] = {}
    for submarket, plant in zip(submarket_codes, source_plants, strict=True):
        plants_by_submarket.setdefault(submarket, []).append(plant.code)
    rows: list[list[object]] = [
        [submarket, ", ".join(str(code) for code in codes)]
        for submarket, codes in sorted(plants_by_submarket.items())
        if len(codes) > 1
    ]
    if not rows:
        return
    emit(
        Diagnostic(
            code="dessem-fcf-gnl-submarket-repeated",
            severity=Severity.WARNING,
            category="FCF export",
            title="GNL plants share a submarket",
            summary=f"{len(rows)} submarket(s) hold more than one GNL plant.",
            table=DiagnosticTable(
                columns=["Submarket", "GNL plants"],
                rows=rows,
                justify=["right", "left"],
            ),
            notes=[
                "Each plant keeps its own entry in the file. Readers that key the "
                "GNL block by submarket code merge the plants of a submarket."
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
    gnl_patamar_hours: tuple[float, ...],
    gnl_committed_cost: tuple[float, ...],
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
        gnl_patamar_hours=gnl_patamar_hours,
        taxa_desconto=factors,
        geracao_termica_minima_gerada_gnl=gnl_committed_cost,
        tempo_viagem=travel_time,
    )


def _assemble_cuts(
    pools: dict,
    trunk_pool_ids: tuple[int, ...],
    plant_ids: tuple[int, ...],
    offsets: Any,
    travel_time: tuple[TravelTimePlant, ...],
    stages: tuple[StageInfo, ...],
    gnl_deliveries: GnlDeliveries,
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
            # Only the first cut-building node prices the second-stage delivery.
            pi_gnl = (
                gnl_block(gnl_deliveries, piece.coefficients, offsets)
                if pool_id == trunk_pool_ids[0]
                else np.zeros(offsets.ncoef - offsets.pi_gnl)
            )
            node_cuts.append(
                CutInput(
                    intercept=piece.intercept,
                    pi_varm=storage_coefficients(piece, pool.slots, plant_ids),
                    pi_qdefp=release_weights @ piece.coefficients,
                    pi_gnl=pi_gnl,
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
    """Everything the binary pair, its GNL reports and its content CSVs come from."""

    pools: dict[int, StageCutPool]
    trunk_pool_ids: tuple[int, ...]
    plant_ids: tuple[int, ...]
    travel_time: tuple[TravelTimePlant, ...]
    stages: tuple[StageInfo, ...]
    n_patamares: int
    offsets: CutBlockOffsets
    header: MapcutHeader
    gnl_thermals: tuple[ThermalInfo, ...]
    gnl_deliveries: GnlDeliveries
    gnl_dropped: int
    committed_costs: tuple[GnlCommittedCost, ...]


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
        undated = undated_bucket_audit(
            {pool_id: pools[pool_id] for pool_id in trunk_pool_ids}
        )
        _emit_travel_time_info(travel_time, undated)
        _emit_undated_bucket_warning(undated)

        # P8: Audit inflow lag drops
        inflow_lag_drop_audit(pools)

        gnl_thermals = anticipated_thermals(inputs)
        gnl_lags = gnl_lag_months(gnl_thermals, source.gnl_thermals)
        gnl_deliveries = select_gnl_deliveries(
            pools[trunk_pool_ids[0]], gnl_thermals, inputs.stages
        )
        _emit_gnl_delivery_missing(gnl_thermals, source.gnl_thermals, gnl_deliveries)
        gnl_drop = gnl_drop_audit(pools, trunk_pool_ids, gnl_deliveries)
        committed_costs = gnl_committed_costs(
            inputs.stages, gnl_thermals, inputs.past_anticipated_commitments
        )

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
        _emit_gnl_submarket_repeated(gnl_submarket_codes, source.gnl_thermals)

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
            gnl_deliveries.block_hours,
            tuple(cost.value for cost in committed_costs),
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
        gnl_thermals=gnl_thermals,
        gnl_deliveries=gnl_deliveries,
        gnl_dropped=gnl_drop.total,
        committed_costs=committed_costs,
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
            plan.gnl_deliveries,
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


def _emit_gnl_reports(
    plan: _ExportPlan,
    source_plants: Sequence[SourceGnlThermal],
    mapcut_contents: MapcutContents,
    cortdeco_contents: CortdecoContents,
) -> None:
    """Report the GNL values the validated pair holds."""
    n_nodes = len(plan.trunk_pool_ids)
    _emit_gnl_coefficients_exported(
        cortdeco_contents,
        n_nodes,
        plan.offsets,
        plan.gnl_deliveries,
        plan.gnl_thermals,
        source_plants,
        plan.header.codigos_submercados_gnl,
        plan.gnl_dropped,
    )
    _emit_gnl_coefficient_zero(
        cortdeco_contents,
        n_nodes,
        plan.offsets,
        plan.gnl_deliveries,
        plan.gnl_thermals,
        source_plants,
    )
    _emit_gnl_committed_cost(mapcut_contents, plan.committed_costs, source_plants)


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

    try:
        plan = _plan_export(case_dir, source, output_dir, settings, progress)
        mapcut_contents, cortdeco_contents = _write_validated_pair(
            plan, paths, progress
        )
        _emit_gnl_reports(plan, source.gnl_thermals, mapcut_contents, cortdeco_contents)

        with progress.step("write content CSVs", 11):
            if settings.emit_audit_csv:
                _write_content_csvs(plan, mapcut_contents, cortdeco_contents, paths)

        with progress.step("write export manifest", 12):
            _write_success_manifest(case_dir, source, settings, paths)

        _LOG.info("export completed successfully")
        return 0

    except (InputReadError, LayoutError, MappingError, PolicyFormatError) as exc:
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
