"""Tests for the shared Cobre-I/O seam (``cobre_bridge.cobre.case_io``).

Pins the contract that the dashboard and the results comparator resolve hydro
productivity through the SAME canonical cascade, so the two products never
report a different ρ for the same converted case (regression for the divergent
hydro-reader layers: the comparator did a per-hydro fallback while the
dashboard only consulted production_models when the WHOLE parquet dict was
empty).
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from cobre_bridge.cobre.case_io import resolve_hydro_productivities
from cobre_bridge.cobre.readers import read_cobre_hydro_metadata
from cobre_bridge.dashboard.data import load_hydro_metadata
from tests.conftest import hydro_with_group


def _build_case(case_dir: Path) -> None:
    """Lay out a case where hydro 0 is in the parquet and hydro 1 is not.

    hydro 1 is absent from hydro_energy_productivity.parquet but present in
    hydro_production_models.json — the exact "parquet present overall but id
    missing" case that made the two readers disagree.
    """
    system = case_dir / "system"
    system.mkdir(parents=True, exist_ok=True)
    (case_dir / "output").mkdir(parents=True, exist_ok=True)

    # 0.13-shaped hydros.json (unit_groups[].bus_id, no top-level bus_id).
    # Feeds load_hydro_metadata (dashboard/data.py) below.
    # read_cobre_hydro_metadata (cobre/readers.py) no longer reads
    # any bus id from hydros.json at all, so it is
    # indifferent to this shape.
    hydros = {
        "hydros": [
            hydro_with_group(
                0,
                0,
                name="H0",
                generation={
                    "productivity_mw_per_m3s": 0.9,  # embedded (lowest priority)
                    "max_turbined_m3s": 100.0,
                },
                reservoir={},
            ),
            hydro_with_group(
                1,
                0,
                name="H1",
                generation={
                    "productivity_mw_per_m3s": 0.9,  # embedded (must NOT win)
                    "max_turbined_m3s": 100.0,
                },
                reservoir={},
            ),
        ]
    }
    (system / "hydros.json").write_text(json.dumps(hydros))

    # Parquet has a row for hydro 0 only (NULL stage_id default).
    table = pa.table(
        {
            "hydro_id": pa.array([0], pa.int32()),
            "stage_id": pa.array([None], pa.int32()),
            "equivalent_productivity_mw_per_m3s": pa.array([0.5], pa.float64()),
        }
    )
    pq.write_table(table, system / "hydro_energy_productivity.parquet")

    # production_models supplies hydro 1's productivity.
    pm = {
        "production_models": [
            {"hydro_id": 1, "stage_ranges": [{"productivity_mw_per_m3s": 0.7}]}
        ]
    }
    (system / "hydro_production_models.json").write_text(json.dumps(pm))


def test_resolve_cascade_is_per_hydro(tmp_path: Path) -> None:
    """hydro 0 -> parquet (0.5); hydro 1 -> production_models (0.7), NOT the
    embedded 0.9 (proving the fallback is per-hydro, not whole-dict)."""
    case_dir = tmp_path / "case"
    _build_case(case_dir)
    with (case_dir / "system" / "hydros.json").open() as f:
        hydros = json.load(f)["hydros"]

    prods = resolve_hydro_productivities(case_dir, hydros)
    assert prods[0] == 0.5
    assert prods[1] == 0.7


def test_dashboard_and_comparator_agree_on_productivity(tmp_path: Path) -> None:
    """The two reader layers report identical ρ for every hydro."""
    case_dir = tmp_path / "case"
    _build_case(case_dir)

    comparator = read_cobre_hydro_metadata(case_dir / "output")
    dashboard = load_hydro_metadata(case_dir)

    for hid in (0, 1):
        assert (
            comparator[hid]["productivity_mw_per_m3s"] == dashboard[hid]["productivity"]
        )
    # And specifically the canonical values, not the embedded 0.9.
    assert dashboard[1]["productivity"] == 0.7
    assert comparator[1]["productivity_mw_per_m3s"] == 0.7
