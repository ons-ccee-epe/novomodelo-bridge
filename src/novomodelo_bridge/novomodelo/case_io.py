"""Shared low-level readers for Cobre case/output files.

Both the dashboard (``dashboard/data.py``) and the comparator
(``cobre/readers.py``) read the same on-disk Cobre artifacts. This
module holds the pieces they genuinely share, so a fix lands in one place
instead of drifting between two near-copies.

Currently it owns the hydro-productivity resolution (the parquet override table
plus the legacy production-models JSON fallback). It is the seed of the wider
shared Cobre-I/O foundation; more low-level readers can move here over time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

_LOG = logging.getLogger(__name__)


def case_dir_for(output_dir: Path) -> Path:
    """Return the Cobre *case* directory that owns *output_dir*.

    A Cobre case lays its inputs (``system/``, ``constraints/``,
    ``penalties.json``, …) in the case directory and its solver outputs
    (``simulation/``, ``training/``, ``policy/``) one level below, in
    ``<case_dir>/output``. The comparator is handed the ``output/`` directory,
    so the case directory is its parent.

    This is the single home for that "``output`` sits under the case" contract,
    which used to be hard-coded as ``cobre_output_dir.parent`` at ~13 sites — and
    the one place a future custom ``--output`` layout would override.
    """
    return output_dir.parent


def productivity_from_energy_parquet(case_dir: Path) -> dict[int, float]:
    """Return ``{hydro_id: ρ_eq}`` from ``hydro_energy_productivity.parquet``.

    Cobre's productivity-resolution-rules contract puts per-(hydro, stage) ρ_eq
    in the parquet override table. Prefer the per-hydro NULL-stage_id "default"
    row; fall back to the productivity at the smallest stage_id when no default
    row exists. Returns ``{}`` when the file is absent or unreadable.
    """
    parquet_path = case_dir / "system" / "hydro_energy_productivity.parquet"
    if not parquet_path.exists():
        return {}
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(parquet_path)
    except (OSError, ImportError):
        _LOG.warning("Failed to read %s", parquet_path)
        return {}

    cols = table.column_names
    if "hydro_id" not in cols or "equivalent_productivity_mw_per_m3s" not in cols:
        return {}
    hydro_ids = table["hydro_id"].to_pylist()
    stage_ids = (
        table["stage_id"].to_pylist() if "stage_id" in cols else [None] * len(hydro_ids)
    )
    values = table["equivalent_productivity_mw_per_m3s"].to_pylist()

    # First pass: take NULL-stage_id rows (per-hydro defaults).
    result: dict[int, float] = {}
    for hid, sid, v in zip(hydro_ids, stage_ids, values, strict=True):
        if sid is None and v is not None and hid not in result:
            result[int(hid)] = float(v)

    # Second pass: for hydros without a default row, fall back to the smallest
    # stage_id row's value (best-effort representative).
    by_hydro_min_stage: dict[int, tuple[int, float]] = {}
    for hid, sid, v in zip(hydro_ids, stage_ids, values, strict=True):
        if sid is None or v is None:
            continue
        hid_i = int(hid)
        if hid_i in result:
            continue
        cur = by_hydro_min_stage.get(hid_i)
        if cur is None or sid < cur[0]:
            by_hydro_min_stage[hid_i] = (sid, float(v))
    for hid, (_sid, v) in by_hydro_min_stage.items():
        result[hid] = v
    return result


def productivity_from_production_models(case_dir: Path) -> dict[int, float]:
    """Legacy fallback: read productivity from ``hydro_production_models.json``.

    Pre-modernization cobre-bridge cases (before the parquet-only contract) embed
    ``productivity_mw_per_m3s`` in the JSON stage_range entries; this preserves
    dashboard/compare support for those older outputs. New cases use
    :func:`productivity_from_energy_parquet` instead. Returns ``{}`` when the file
    is absent, unreadable, or carries only model selection.
    """
    pm_path = case_dir / "system" / "hydro_production_models.json"
    if not pm_path.exists():
        return {}
    try:
        with pm_path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        _LOG.warning("Failed to parse %s", pm_path)
        return {}
    result: dict[int, float] = {}
    for model in data.get("production_models", []):
        hydro_id = int(model.get("hydro_id"))
        ranges = model.get("stage_ranges") or []
        for entry in ranges:
            prod = entry.get("productivity_mw_per_m3s")
            if prod is not None:
                result[hydro_id] = float(prod)
                break
    return result


def resolve_hydro_productivities(
    case_dir: Path, hydros: list[dict]
) -> dict[int, float | None]:
    """Resolve per-hydro ρ_eq using the single canonical fallback cascade.

    For each hydro in ``hydros`` (the ``hydros.json`` ``"hydros"`` list), try in
    order:

    1. ``hydro_energy_productivity.parquet`` — the modern per-(hydro, stage)
       contract (:func:`productivity_from_energy_parquet`);
    2. ``hydro_production_models.json`` — legacy embedded productivity
       (:func:`productivity_from_production_models`);
    3. ``hydros.json`` ``generation.productivity_mw_per_m3s`` — deprecated
       embedded field;
    4. ``hydros.json`` top-level ``productivity_mw_per_m3s`` — oldest layout.

    The cascade is **per hydro**, not whole-dict: a hydro absent from the
    parquet still falls through to the next source even when other hydros do
    have parquet rows. Returns ``{hydro_id: ρ or None}`` (``None`` when no
    source has a value).

    This is the one place the dashboard and the results comparator agree on
    productivity for a converted case — both must call it so they never report
    a different ρ for the same plant.
    """
    energy = productivity_from_energy_parquet(case_dir)
    pm = productivity_from_production_models(case_dir)
    result: dict[int, float | None] = {}
    for hydro in hydros:
        hid = int(hydro["id"])
        prod = energy.get(hid)
        if prod is None:
            prod = pm.get(hid)
        if prod is None:
            gen = hydro.get("generation")
            if isinstance(gen, dict):
                prod = gen.get("productivity_mw_per_m3s")
        if prod is None:
            prod = hydro.get("productivity_mw_per_m3s")
        result[hid] = float(prod) if prod is not None else None
    return result
