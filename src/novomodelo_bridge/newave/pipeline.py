"""Conversion pipeline: orchestrates entity and temporal/stochastic converters.

Reads a source-model case directory and writes a complete Cobre case directory.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa

from cobre_bridge.cobre import scalar_parameters as scalar_params_conv
from cobre_bridge.cobre import schemas as cobre_schemas
from cobre_bridge.cobre.case_writer import CaseWriter
from cobre_bridge.core import diagnostics as dx
from cobre_bridge.core import emission_checks
from cobre_bridge.core.bound_merge import merge_bound_tables
from cobre_bridge.core.conversion import (
    ClearedArtifacts,
    ConversionReport,
    clear_dst_contents,
)
from cobre_bridge.core.generic_constraint_builder import ConstraintIdAllocator
from cobre_bridge.newave.case import NewaveCase
from cobre_bridge.newave.converters import constraints as constraints_conv
from cobre_bridge.newave.converters import hydro as hydro_conv
from cobre_bridge.newave.converters import inflow_windows
from cobre_bridge.newave.converters import initial_conditions as ic_conv
from cobre_bridge.newave.converters import network as network_conv
from cobre_bridge.newave.converters import stochastic as stochastic_conv
from cobre_bridge.newave.converters import tailrace as tailrace_conv
from cobre_bridge.newave.converters import temporal as temporal_conv
from cobre_bridge.newave.converters import thermal as thermal_conv

logger = logging.getLogger(__name__)

#: Coarse conversion phases reported to an optional ``on_phase`` callback (drives the
#: CLI progress bar). The order matches the boundaries in
#: :func:`_convert_newave_case_impl`; the count is the progress-bar total.
CONVERSION_PHASE_LABELS: tuple[str, ...] = (
    "Discovering files",
    "Converting entities",
    "Converting temporal & stochastic",
    "Converting constraints",
    "Writing JSON",
    "Writing Parquet",
)


def _compute_prod_media_sin_safe(case: NewaveCase) -> float | None:
    """Return ``PROD_MEDIA_SIN`` (mean PRODT), or ``None`` on failure.

    The source model converts the PENALID R$/MWh penalties (VAZMIN, TURBMN/TURBMX, …)
    with the mean **PRODT** — the equivalent productivity from vol_min to vol_max — over
    all existing plants including zeros. See :func:`hydro.compute_prodt_sin_mean`;
    validated against pmo.dat's applied penalty (0.6299 ↔ VAZMIN 821.78). Falls back to
    ``None`` (caller uses its legacy point-reference mean) when the source model files
    cannot be read — e.g. in unit tests that mock the pipeline.
    """
    try:
        return hydro_conv.compute_prodt_sin_mean(case)
    except (OSError, ValueError, AttributeError, TypeError, KeyError):
        return None


def _compute_per_stage_sin_productivities(
    case: NewaveCase,
) -> tuple[list[float], list[float]] | None:
    """Return ``(PROD_MEDIA_SIN[s], MAX_PRODTACUM_SIN[s])`` per stage, or None.

    These SIN-aggregate productivity constants are what the source model uses to convert
    its flow-domain hydro penalties (manual §3.24 p.87).

    - ``PROD_MEDIA_SIN[s]`` = mean **PRODT** (equivalent ρ vol_min→vol_max) over
      the existing plants. PRODT is **structural** — it does *not* track the
      seasonal reference volume — so this is essentially **flat** across the
      horizon. pmo.dat confirms it: the applied VAZMIN/TURBMN/TURBMX penalties
      vary only 0.15% over the whole study (= the per-config PRODT-mean spread), not the
      ~2% that the old seasonal point-reference ρ implied. We therefore emit a constant
      ``compute_prodt_sin_mean`` for every stage; the per-stage penalty override then
      collapses to the base for the ρ_avg-scaled columns (correct — the source model
      does not seasonalise these penalties). The genuinely seasonal per-plant ρ still
      ships in ``hydro_energy_productivity.parquet`` for the *generation* model — that
      is a separate concern from the penalty.
    - ``MAX_PRODTACUM_SIN`` = max accumulated cascade ρ at **altura máxima**,
      **constant** over the horizon. It governs the DESVIO ("outros usos") and
      evaporation penalties, which pmo.dat reports as *fixed* — the max is
      ITAIPU's cascade, which carries no CFUGA/CMONT override, so it never moves.
      (See :func:`constraints.compute_max_prodtacum_sin`.)

    Both lists span the full horizon (``_total_study_stages``, incl. post-study). Falls
    back to ``None`` when the source model files can't be read (mocked tests).
    """
    try:
        rho_avg = hydro_conv.compute_per_stage_prodt_sin_mean(case)
        if not rho_avg:
            return None
        max_prodtacum = constraints_conv.compute_max_prodtacum_sin(case)
        if max_prodtacum is None:
            return None
        rho_max_acum = [max_prodtacum] * len(rho_avg)
        return rho_avg, rho_max_acum
    except (
        OSError,
        ValueError,
        AttributeError,
        TypeError,
        KeyError,
        StopIteration,
        ZeroDivisionError,
    ):
        return None


def _merge_hydro_bounds(
    withdrawal: pa.Table | None,
    storage: pa.Table | None,
) -> pa.Table | None:
    """Merge water withdrawal and storage bounds into one table.

    Both inputs are optional.  When both are present, they are outer-joined
    on ``(hydro_id, stage_id)`` so that rows from either source appear in the
    result with nulls for missing columns.
    """
    if withdrawal is None and storage is None:
        return None
    if withdrawal is None:
        return storage
    if storage is None:
        return withdrawal

    import polars as pl

    w = pl.from_arrow(withdrawal)
    s = pl.from_arrow(storage)
    merged = merge_bound_tables(w, s, on=["hydro_id", "stage_id"], precedence="base")
    merged = merged.sort("hydro_id", "stage_id")
    return merged.to_arrow()


def _fold_head_turbined_bounds(
    base: pa.Table | None,
    head_turbined: pa.Table | None,
) -> pa.Table | None:
    """Fold per-stage head-corrected ``max_turbined_m3s`` into the hydro bounds.

    ``head_turbined`` carries the per-(hydro, stage) engolimento cap derived from
    the per-stage operating head. An explicit TURBMAXT override already present in
    ``base`` (from ``convert_storage_bounds``) takes precedence — it is a deliberate
    operational limit — so the head-derived value only fills cells TURBMAXT leaves
    empty (the common case: plants with CFUGA/CMONT but no TURBMAXT).
    """
    if head_turbined is None:
        return base

    import polars as pl

    h = pl.from_arrow(head_turbined)
    if base is None:
        return h.sort("hydro_id", "stage_id").to_arrow()

    b = pl.from_arrow(base)
    merged = merge_bound_tables(b, h, on=["hydro_id", "stage_id"], precedence="base")
    merged = merged.sort("hydro_id", "stage_id")
    return merged.to_arrow()


def convert_newave_case(
    src: Path,
    dst: Path,
    *,
    on_phase: Callable[[str], None] | None = None,
    dry_run: bool = False,
) -> ConversionReport:
    """Convert a source-model case directory to a Cobre case directory.

    Parameters
    ----------
    src:
        Path to the source model case directory.  Must exist and contain all required
        the source model input files.
    dst:
        Path to the output Cobre case directory.  Must not exist or must be
        empty (call site is responsible for enforcing the --force contract
        before calling this function).
    dry_run:
        When ``True``, run the full in-memory conversion but write nothing to
        *dst* (no files, no subdirectories). The would-write paths are still
        recorded in :attr:`ConversionReport.would_write_paths`.

    Returns
    -------
    ConversionReport
        Summary of what was converted, including a ``warnings`` list of every
        degraded-input substitution that occurred (empty when the conversion
        ran clean).

    Raises
    ------
    FileNotFoundError
        If *src* does not exist, is not a directory, or a required the source model file
        is missing.
    """
    collector = dx.WarningCollector()
    pkg_logger = logging.getLogger("cobre_bridge")
    pkg_logger.addHandler(collector)
    try:
        # Structured diagnostics emitted by converters land in ``collected``; any
        # remaining ``logger.warning`` strings (sites not yet migrated) are picked
        # up by ``collector`` and bridged below, so every warning still surfaces.
        with dx.collect() as collected:
            report = _convert_newave_case_impl(src, dst, on_phase, dry_run=dry_run)
    except BaseException:
        # The write phase is a sequence of independent file writes with no
        # rollback, so a failure partway through (disk full, a converter
        # raising mid-write, an interrupt) can leave a subset of the output
        # files behind. Remove the known pipeline outputs so a half-written
        # case is never mistaken for a complete one and a plain (no --force)
        # re-run is not refused as "destination not empty". Re-raise so the
        # CLI still reports the original failure. Skipped under a dry run:
        # nothing was written, and dst may be a pre-existing populated
        # directory the user never asked to clear.
        if not dry_run:
            clear_dst_contents(dst, NEWAVE_CLEARED_ARTIFACTS)
        raise
    finally:
        pkg_logger.removeHandler(collector)
    report.diagnostics = dx.finalize_diagnostics(collected, collector.messages)
    report.warnings = [
        d.summary for d in report.diagnostics if d.severity is dx.Severity.WARNING
    ]
    return report


def _convert_newave_case_impl(
    src: Path,
    dst: Path,
    on_phase: Callable[[str], None] | None = None,
    *,
    dry_run: bool = False,
) -> ConversionReport:
    """Run the conversion pipeline (warning capture handled by the wrapper).

    ``on_phase`` (when given) is called once at each boundary in
    :data:`CONVERSION_PHASE_LABELS`, so the CLI can advance a progress bar without
    the pipeline knowing anything about rendering.

    When ``dry_run`` is ``True``, the build phases run exactly as for a real run,
    but the write phase creates no directories and writes no files; the paths that
    would have been written are still accumulated and recorded on the report.
    """
    report = ConversionReport()
    step = on_phase if on_phase is not None else (lambda _label: None)

    step("Discovering files")
    logger.debug("Discovering NEWAVE files from %s", src)
    # Build the parsed-case object once; every converter reads its parsed inputs from
    # ``case`` (each source-model file parsed once and cached).
    case = NewaveCase.from_directory(src)

    logger.debug("Building NewaveIdMap from %s", src)
    id_map = case.id_map

    step("Converting entities")
    logger.debug("Converting hydros")
    hydros_dict = hydro_conv.convert_hydros(case, id_map)

    logger.debug("Computing base hydro productivities")
    base_productivities = hydro_conv.compute_base_productivities(case, id_map)

    logger.debug("Generating hydro geometry")
    cadastro = hydro_conv.read_cadastro(case)
    geometry_table = hydro_conv.generate_hydro_geometry(cadastro, id_map)

    logger.debug("Converting thermals")
    thermals_dict = thermal_conv.convert_thermals(case, id_map)

    logger.debug("Converting buses")
    buses_dict = network_conv.convert_buses(case, id_map)

    logger.debug("Converting lines")
    lines_dict = network_conv.convert_lines(case, id_map)

    logger.debug("Converting penalties")
    # DESVIO ("outros usos") + evaporation use MAX_PRODTACUM_SIN (max accumulated ρ at
    # altura máxima, constant); VAZMIN/TURBMN/TURBMX use PROD_MEDIA_SIN (mean PRODT,
    # drifting per CFUGA/CMONT config). Both come from the source model inputs and
    # return None when files are absent (mocked-pipeline tests), so convert_penalties
    # falls back to its own legacy approximation.
    max_prodtacum_sin = constraints_conv.compute_max_prodtacum_sin(case)
    prod_media_sin = _compute_prod_media_sin_safe(case)
    penalties_dict = network_conv.convert_penalties(
        case,
        hydros_dict,
        productivities=base_productivities,
        max_accumulated_productivity=max_prodtacum_sin,
        prod_media_sin=prod_media_sin,
    )

    step("Converting temporal & stochastic")
    logger.debug("Converting stages")
    stages_dict = temporal_conv.convert_stages(case, id_map)

    logger.debug("Converting config")
    config_dict = temporal_conv.convert_config(case)

    logger.debug("Converting initial conditions")
    ic_dict = ic_conv.convert_initial_conditions(case, id_map)

    logger.debug("Converting the hydrological tendency to conditioning windows")
    observation_windows = inflow_windows.convert_recent_observation_windows(
        case, id_map
    )
    if observation_windows:
        ic_dict["recent_observations"] = observation_windows

    logger.debug("Converting inflow stats")
    inflow_table = stochastic_conv.convert_inflow_stats(case, id_map)

    logger.debug("Converting load stats")
    load_table = stochastic_conv.convert_load_stats(case, id_map)

    logger.debug("Converting inflow history from vazoes.dat")
    inflow_history_table = inflow_windows.convert_inflow_history_windows(case, id_map)

    logger.debug("Converting water withdrawal")
    withdrawal_table = hydro_conv.convert_water_withdrawal(case, id_map)

    logger.debug("Converting storage bounds from VMAXT/VMINT")
    storage_bounds_table = hydro_conv.convert_storage_bounds(case, id_map)

    logger.debug("Converting per-stage head-corrected turbined caps")
    head_turbined_table = hydro_conv.convert_turbined_bounds_head_corrected(
        case, id_map
    )

    allocator = ConstraintIdAllocator()

    step("Converting constraints")
    logger.debug("Converting VminOP constraints")
    vminop_result = constraints_conv.convert_vminop_constraints(
        case, id_map, allocator=allocator
    )
    vminop_referenced_ids: list[int] = []
    rho_acum_overrides: dict[int, list[float]] = {}
    if vminop_result is not None:
        vminop_referenced_ids = list(vminop_result.referenced_hydro_ids)
        rho_acum_overrides = vminop_result.rho_acum_overrides

    logger.debug("Converting electric constraints")
    electric_result = constraints_conv.convert_electric_constraints(
        case, id_map, allocator=allocator
    )

    logger.debug("Converting AGRINT group constraints")
    agrint_result = constraints_conv.convert_agrint_constraints(
        case, id_map, allocator=allocator
    )

    logger.debug("Converting load factors")
    load_factors_dict = stochastic_conv.convert_load_factors(case, id_map)

    logger.debug("Converting line bounds (folds in per-block exchange factors)")
    line_bounds_table = network_conv.convert_line_bounds(case, id_map)

    logger.debug("Converting non-controllable sources")
    ncs_dict = network_conv.convert_non_controllable_sources(case, id_map)

    logger.debug("Converting NCS block factors")
    ncs_factors_dict = network_conv.convert_ncs_factors(case, id_map)

    logger.debug("Converting NCS stats")
    ncs_stats_table = network_conv.convert_ncs_stats(case, id_map)

    logger.debug("Converting production models")
    production_models_dict = hydro_conv.convert_production_models(case, id_map)

    logger.debug("Converting hydro energy productivity overrides")
    hydro_energy_productivity_table = hydro_conv.convert_hydro_energy_productivity(
        case, id_map
    )

    logger.debug("Converting tailrace curves from polinjus")
    tailrace_table = tailrace_conv.convert_tailrace_curves(case, id_map)

    logger.debug("Converting thermal bounds from expt.dat and manutt.dat")
    thermal_bounds_table = thermal_conv.convert_thermal_bounds(case, id_map)

    # Merge hydro_bounds before the write phase so the post-emission
    # self-checks see the exact table that will be written.
    hydro_bounds_table = _merge_hydro_bounds(withdrawal_table, storage_bounds_table)
    hydro_bounds_table = _fold_head_turbined_bounds(
        hydro_bounds_table, head_turbined_table
    )

    # Clamp any per-stage MAX bound above the plant's declared envelope back to
    # the declared value (WARNING, not silent), so the rule-43 self-check below
    # passes on the result.
    hydro_bounds_table = emission_checks.clamp_hydro_bounds_to_declared(
        hydros_dict, hydro_bounds_table
    )

    # Post-emission self-checks: mirror cheap cobre load invariants
    # (rules 43, 41, 36, and the block_id-range rule) over the in-memory
    # artifacts before anything is written. See cobre_bridge.core.emission_checks
    # for the rule scope.
    bound_families = [
        emission_checks.BoundFamily("Hydro", "hydro_id", hydro_bounds_table),
        emission_checks.BoundFamily("Thermal", "thermal_id", thermal_bounds_table),
        emission_checks.BoundFamily("Line", "line_id", line_bounds_table),
    ]

    def _run_checks() -> None:
        emission_checks.check_hydro_bounds_no_raising(hydros_dict, hydro_bounds_table)
        emission_checks.check_unit_group_envelope(hydros_dict)
        emission_checks.check_bound_row_uniqueness(bound_families)
        emission_checks.check_bound_block_id_range(stages_dict, bound_families)

    emission_checks.run_and_gate(_run_checks)

    # Every output path is routed through one ``CaseWriter`` so the
    # would-write listing, the byte format, and the dry-run gate live in
    # exactly one place, rather than guarding ~30 individual write sites.
    writer = CaseWriter(dst, dry_run=dry_run)

    step("Writing JSON")
    writer.write_json("config.json", config_dict)
    writer.write_json("stages.json", stages_dict)
    writer.write_json("penalties.json", penalties_dict)
    writer.write_json("initial_conditions.json", ic_dict)
    writer.write_json("system/hydros.json", hydros_dict)
    writer.write_json("system/thermals.json", thermals_dict)
    writer.write_json("system/buses.json", buses_dict)
    writer.write_json("system/lines.json", lines_dict)
    writer.write_json("system/non_controllable_sources.json", ncs_dict)
    writer.write_json("scenarios/load_factors.json", load_factors_dict)
    writer.write_json("scenarios/non_controllable_factors.json", ncs_factors_dict)

    writer.write_json("system/hydro_production_models.json", production_models_dict)

    # Declare per-hydro @rho_eq_h{id} / @rho_acum_h{id} computed parameters for
    # every hydro in the case. cobre rejects any @name token that has not been
    # declared, so we always emit the file so handwritten or generated
    # constraint expressions can reference any per-hydro productivity. The
    # ids set is the union of the VminOP-referenced ids (covers the values we
    # currently emit) and every hydro that has a production-model entry
    # (covers future references).
    all_hydro_ids = sorted(
        set(vminop_referenced_ids)
        | {
            int(m["hydro_id"])
            for m in production_models_dict.get("production_models", [])
        }
    )
    scalar_parameters_dict = scalar_params_conv.build_scalar_parameters(
        all_hydro_ids,
        rho_acum_per_stage_overrides=rho_acum_overrides or None,
    )
    writer.write_json("constraints/generic_parameters.json", scalar_parameters_dict)

    step("Writing Parquet")
    writer.write_parquet("system/hydro_geometry.parquet", geometry_table)

    if hydro_energy_productivity_table.num_rows > 0:
        writer.write_parquet(
            "system/hydro_energy_productivity.parquet",
            hydro_energy_productivity_table,
        )

    # Optional tailrace curves (polinjus) — only written when the case ships them.
    if tailrace_table is not None and tailrace_table.num_rows > 0:
        writer.write_parquet("system/tailrace_curves.parquet", tailrace_table)

    writer.write_parquet("scenarios/inflow_seasonal_stats.parquet", inflow_table)

    writer.write_parquet("scenarios/load_seasonal_stats.parquet", load_table)

    writer.write_parquet("scenarios/inflow_history.parquet", inflow_history_table)

    writer.write_parquet("constraints/line_bounds.parquet", line_bounds_table)

    writer.write_parquet("scenarios/non_controllable_stats.parquet", ncs_stats_table)

    if hydro_bounds_table is not None:
        writer.write_parquet("constraints/hydro_bounds.parquet", hydro_bounds_table)

    if thermal_bounds_table is not None:
        writer.write_parquet("constraints/thermal_bounds.parquet", thermal_bounds_table)

    # Per-bus excess-cost override: forbid energy excess at fictitious
    # submarkets (pure transshipment nodes) by pricing it prohibitively.
    bus_penalty_table = network_conv.convert_bus_penalty_overrides(case, id_map)
    if bus_penalty_table is not None:
        writer.write_parquet(
            "constraints/penalty_overrides_bus.parquet", bus_penalty_table
        )

    # Per-stage hydro penalty override: The source model's PROD_MEDIA_SIN /
    # MAX_PRODTACUM_SIN shift with seasonal (VOLREF_SAZ) and temporal (CFUGA/CMONT)
    # productivity changes, so the ρ-scaled flow-domain hydro penalties are
    # stage-varying. Emitted sparsely (only stages/columns that differ from
    # penalties.json), keeping the penalty conversion coherent with the per-stage ρ
    # already shipped in system/hydro_energy_productivity.parquet.
    per_stage_sin = _compute_per_stage_sin_productivities(case)
    if per_stage_sin is not None:
        per_stage_rho_avg, per_stage_rho_max_acum = per_stage_sin
        hydro_ids = [int(h["id"]) for h in hydros_dict.get("hydros", []) if "id" in h]
        hydro_penalty_table = network_conv.convert_hydro_penalty_overrides(
            case,
            hydro_ids,
            penalties_dict["hydro"],
            per_stage_rho_avg,
            per_stage_rho_max_acum,
        )
        if hydro_penalty_table is not None:
            writer.write_parquet(
                "constraints/penalty_overrides_hydro.parquet", hydro_penalty_table
            )

    all_constraints: list[dict] = []
    bounds_tables: list[pa.Table] = []

    if vminop_result is not None:
        all_constraints.extend(vminop_result.constraints_dict.get("constraints", []))
        bounds_tables.append(vminop_result.bounds)

    if electric_result is not None:
        all_constraints.extend(electric_result.constraints)
        bounds_tables.append(electric_result.bounds)

    if agrint_result is not None:
        all_constraints.extend(agrint_result.constraints)
        bounds_tables.append(agrint_result.bounds)

    if all_constraints:
        merged_dict = {
            "$schema": cobre_schemas.schema_url_for(
                "constraints/generic_constraints.json"
            ),
            "constraints": all_constraints,
        }
        writer.write_json("constraints/generic_constraints.json", merged_dict)

        if bounds_tables:
            merged_bounds = pa.concat_tables(bounds_tables)
            writer.write_parquet(
                "constraints/generic_constraint_bounds.parquet", merged_bounds
            )

    report.hydro_count = len(hydros_dict.get("hydros", []))
    report.thermal_count = len(thermals_dict.get("thermals", []))
    report.bus_count = len(buses_dict.get("buses", []))
    report.line_count = len(lines_dict.get("lines", []))
    report.stage_count = len(stages_dict.get("stages", []))
    report.would_write_paths = [str(p) for p in writer.would_write]

    return report


NEWAVE_CLEARED_ARTIFACTS = ClearedArtifacts(
    subdirs=("system", "scenarios", "constraints"),
    files=(
        "config.json",
        "stages.json",
        "penalties.json",
        "initial_conditions.json",
        "conversion_manifest.json",
    ),
)
