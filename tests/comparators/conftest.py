"""Decomp results-compare builders used only by tests/comparators/."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import pytest

from novomodelo_bridge.comparators.decomp.results import _AlignedDecompFrames
from novomodelo_bridge.decomp.id_map import DecompIdMap
from tests.conftest import _FakeDadger, make_decomp_case


def _patch_shared_case(
    monkeypatch: pytest.MonkeyPatch,
    *,
    id_map: DecompIdMap,
    dadger: object | None = None,
) -> None:
    """Patch the shared ``DecompCase.from_directory`` build so
    ``build_decomp_dataset``'s ``case.id_map``/``case.dadger`` resolve to
    *id_map*/*dadger* without touching the filesystem -- the case is now
    built unconditionally, before ``_read_aligned_frames`` runs, so every
    fixture exercising ``build_decomp_dataset`` against a bare ``tmp_path``
    needs this (mirrors ``_patch_aligned_frames``'s own "patch at the seam"
    convention; monkeypatch's last ``setattr`` wins, so a test needing a
    specific id map calls this again after ``_patch_aligned_frames``).
    """
    fake_dadger = _FakeDadger() if dadger is None else dadger
    monkeypatch.setattr(
        "novomodelo_bridge.decomp.case.DecompCase.from_directory",
        lambda directory: make_decomp_case(
            directory, dadger=fake_dadger, id_map=id_map
        ),
    )


def _patch_aligned_frames(
    monkeypatch: pytest.MonkeyPatch, aligned: _AlignedDecompFrames
) -> None:
    monkeypatch.setattr(
        "novomodelo_bridge.comparators.decomp.results._read_aligned_frames",
        lambda *_args, **_kwargs: aligned,
    )
    # The shared ``DecompCase`` build now runs unconditionally at
    # the top of ``build_decomp_dataset`` (before ``_read_aligned_frames``,
    # which this stub bypasses) -- degenerate but valid, so a bare
    # ``tmp_path`` keeps working; tests needing a specific id map call
    # ``_patch_shared_case`` again afterwards (monkeypatch's last ``setattr``
    # wins).
    _patch_shared_case(monkeypatch, id_map=DecompIdMap(bus_codes=(), bus_names=()))
    # ``build_decomp_dataset`` also calls
    # ``read_novomodelo_bus_aggregates`` directly (outside ``_read_aligned_frames``).
    # Unlike the other novomodelo readers it does NOT degrade to empty on a missing
    # case -- it raises ``NovomodeloPartitionMissingError`` for the pre-0.13
    # ``hydro_bus_generation`` partition, which a bare ``tmp_path`` always
    # trips. Stub it here too, so every fixture that does not care about
    # the Energy Balance metadata (the vast majority) keeps working
    # against a bare ``tmp_path``; tests that DO care override this again
    # afterwards (monkeypatch's last ``setattr`` wins).
    monkeypatch.setattr(
        "novomodelo_bridge.comparators.decomp.results.novomodelo_readers."
        "read_novomodelo_bus_aggregates",
        lambda *_args, **_kwargs: pl.DataFrame(),
    )
    # ``build_decomp_dataset`` also calls ``_cost_frames`` directly
    # (outside ``_read_aligned_frames``), which reads ``read_relato_costs`` --
    # unlike every other reader here, it RAISES on a missing/empty parse
    # (the "no silent-empty" reader contract), which a bare
    # ``tmp_path`` always trips. Stub it here too, so every fixture that does
    # not care about the cost metadata keeps working against a bare
    # ``tmp_path``; tests that DO care override this again afterwards
    # (monkeypatch's last ``setattr`` wins).
    monkeypatch.setattr(
        "novomodelo_bridge.comparators.decomp.results._cost_frames",
        lambda *_args, **_kwargs: ({}, pl.DataFrame()),
    )
    # ``build_decomp_dataset`` also calls
    # ``read_novomodelo_hydro_bus_labels`` directly (outside ``_read_aligned_frames``).
    # Like ``read_novomodelo_bus_aggregates`` above, it reads the
    # ``simulation/hydro_bus_generation/`` partition and RAISES
    # ``NovomodeloPartitionMissingError`` on a bare ``tmp_path`` instead of
    # degrading to empty. Stub it here too, so every fixture that does not
    # care about the hydro metadata keeps working against a bare
    # ``tmp_path``; tests that DO care override this again afterwards
    # (monkeypatch's last ``setattr`` wins).
    monkeypatch.setattr(
        "novomodelo_bridge.comparators.decomp.results.novomodelo_readers."
        "read_novomodelo_hydro_bus_labels",
        lambda *_args, **_kwargs: {},
    )


def _aligned_fixture() -> _AlignedDecompFrames:
    """One hydro plant, one thermal plant, one bus -- already aligned to Novomodelo
    ids/stages, matching the shape :func:`_read_aligned_frames` returns."""
    source_hydro = pl.DataFrame(
        {
            "entity_id": [0, 1],
            "newave_code": [10, 20],
            "stage_id": [0, 0],
            "geracao_MW": [120.0, 60.0],
            "vazao_turbinada_m3s": [80.0, 40.0],
            "vazao_vertida_m3s": [0.0, 0.0],
            "vazao_defluente_m3s": [80.0, 40.0],
            "volume_util_final_hm3": [500.0, 300.0],
        }
    )
    novomodelo_hydro = pl.DataFrame(
        {
            "entity_id": [0, 1],
            "stage_id": [0, 0],
            "generation_mw": [110.0, 60.0],
            "turbined_m3s": [78.0, 40.0],
            "spillage_m3s": [0.0, 0.0],
            "outflow_m3s": [78.0, 40.0],
            "useful_storage_hm3": [480.0, 300.0],
        }
    )
    source_thermal = pl.DataFrame(
        {
            "entity_id": [0],
            "newave_code": [5],
            "stage_id": [0],
            "geracao_MW": [30.0],
        }
    )
    novomodelo_thermal = pl.DataFrame(
        {"entity_id": [0], "stage_id": [0], "generation_mw": [28.0]}
    )
    source_bus = pl.DataFrame(
        {
            "entity_id": [0],
            "newave_code": [1],
            "stage_id": [0],
            "deficit_MW": [0.0],
            "cmo": [45.0],
        }
    )
    novomodelo_bus = pl.DataFrame(
        {"entity_id": [0], "stage_id": [0], "deficit_mw": [0.0], "spot_price": [44.0]}
    )
    return _AlignedDecompFrames(
        source_hydro=source_hydro,
        source_thermal=source_thermal,
        source_bus=source_bus,
        novomodelo_hydro=novomodelo_hydro,
        novomodelo_thermal=novomodelo_thermal,
        novomodelo_bus=novomodelo_bus,
        hydro_names={0: "A", 1: "B"},
        thermal_names={0: "T"},
        bus_names={0: "SE"},
        unmapped={"hydro": [], "thermal": [86, 224], "bus": []},
    )


def _balance_fixture() -> _AlignedDecompFrames:
    """``_aligned_fixture`` extended with the Energy Balance
    reference frames, keyed to the same bus (novomodelo id 0, name "SE")."""
    nw_market = pl.DataFrame(
        {
            "newave_code": [0, 0, 0],
            "stage": [1, 1, 1],
            "variable": ["GHTOT", "GTERM", "DEFT"],
            "value": [600.0, 250.0, 0.0],
        }
    )
    nw_net_load = pl.DataFrame(
        {
            "newave_code": [0],
            "stage": [1],
            "variable": ["NET_LOAD"],
            "value": [950.0],
        }
    )
    nw_sin = pl.DataFrame(
        {
            "newave_code": [0, 0],
            "stage": [1, 1],
            "variable": ["EARMF", "ENA"],
            "value": [7000.0, 1600.0],
        }
    )
    return dataclasses.replace(
        _aligned_fixture(), nw_market=nw_market, nw_net_load=nw_net_load, nw_sin=nw_sin
    )


def _ree_id_map() -> DecompIdMap:
    """Two hydro plants (codes 10, 20 -> novomodelo ids 0, 1) -- matches
    ``_aligned_fixture``'s own hydro codes/ids so the same
    ``_patch_aligned_frames`` fixture can back both the E1 result rows and
    the REE rollup in the same ``build_decomp_dataset`` test."""
    return DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(10, 20))


def _ree_membership_fixture() -> pl.DataFrame:
    """Both fixture plants (codes 10, 20) belong to REE 100 ('SUDESTE')."""
    return pl.DataFrame(
        {
            "codigo_usina": [10, 20],
            "nome_usina": ["A", "B"],
            "codigo_ree": [100, 100],
            "nome_ree": ["SUDESTE", "SUDESTE"],
            "codigo_submercado": [1, 1],
            "nome_submercado": ["SE", "SE"],
            "nome_submercado_newave": ["SUDESTE", "SUDESTE"],
        }
    )


def _ree_dec_oper_ree_fixture() -> pl.DataFrame:
    """One REE (100), stage 1 (1-based), two nodes -- scenario-mean
    ``ena_MWmes=145.0``, ``earm_final_MWmes=1010.0``, deliberately offset from
    the Novomodelo-side fixture's ``150.0`` / ``1000.0`` (see
    :func:`_ree_novomodelo_hydro_fixture`) so the per-variable diff is
    hand-checkable rather than trivially zero."""
    return pl.DataFrame(
        {
            "estagio": [1, 1],
            "no": [1, 2],
            "cenario": [1, 1],
            "codigo_ree": [100, 100],
            "nome_ree": ["SUDESTE", "SUDESTE"],
            "codigo_submercado": [1, 1],
            "nome_submercado": ["SE", "SE"],
            "ena_MWmes": [140.0, 150.0],
            "earm_inicial_MWmes": [900.0, 900.0],
            "earm_inicial_percentual": [70.0, 70.0],
            "earm_final_MWmes": [1000.0, 1020.0],
            "earm_final_percentual": [72.0, 74.0],
            "earm_maximo_MWmes": [2000.0, 2000.0],
        }
    )


def _ree_aligned_fixture() -> _AlignedDecompFrames:
    """``_aligned_fixture()`` with its ``novomodelo_hydro`` extended to carry the
    ENA/EARM columns :func:`_novomodelo_ree_sums` reads -- the base fixture is
    trimmed to only the columns E1's ``_HYDRO_VARIABLES`` needs."""
    base = _aligned_fixture()
    return dataclasses.replace(
        base,
        novomodelo_hydro=base.novomodelo_hydro.with_columns(
            pl.Series("incremental_inflow_energy_mw", [90.0, 60.0]),
            pl.Series("stored_energy_final_mwh", [400000.0, 330000.0]),
        ),
    )


def _patch_ree_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wire ``read_relato_membership``/``read_dec_oper_ree`` -- outside
    ``_read_aligned_frames`` -- to the fixtures above."""
    monkeypatch.setattr(
        "novomodelo_bridge.comparators.decomp.results.read_relato_membership",
        lambda *_a, **_k: _ree_membership_fixture(),
    )
    monkeypatch.setattr(
        "novomodelo_bridge.comparators.decomp.results.read_dec_oper_ree",
        lambda *_a, **_k: _ree_dec_oper_ree_fixture(),
    )


def _extract_tab_content(html: str, tab_id: str) -> str:
    """Slice one ``id="tab-..."`` tab's content, up to the next ``id="tab-`` marker.

    Distinct from the same-named ``test_chart_helpers._extract_tab_content``,
    which is a ``<section>``-bounded matcher returning the inner group -- a
    different contract despite the shared name.
    """
    import re

    match = re.search(rf'id="{tab_id}".*?(?=id="tab-|\Z)', html, re.S)
    return match.group(0) if match else ""


def _write_generic_constraints_case(
    case_dir: Path,
    constraints: list[dict[str, Any]],
    bound_rows: list[dict[str, Any]],
) -> Path:
    """Write ``constraints/generic_constraints.json`` +
    ``constraints/generic_constraint_bounds.parquet`` under *case_dir* and
    return the Novomodelo output dir (``case_dir/output``) `case_dir_for`
    resolves back to *case_dir* from -- mirrors `_write_lines_json`."""
    constraints_dir = case_dir / "constraints"
    constraints_dir.mkdir(parents=True, exist_ok=True)
    (constraints_dir / "generic_constraints.json").write_text(
        json.dumps({"constraints": constraints})
    )
    pd.DataFrame(
        bound_rows,
        columns=[
            "constraint_id",
            "stage_id",
            "block_id",
            "bound_lower",
            "bound_upper",
        ],
    ).to_parquet(constraints_dir / "generic_constraint_bounds.parquet")
    output_dir = case_dir / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def _usih_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    """A ``dec_oper_usih``-shaped frame: one stage-aggregate
    (``patamar=None``) row per (code, stage) -- the shape `_stage_rows`
    keeps."""
    base = {
        "no": 1,
        "cenario": 1,
        "patamar": None,
        "duracao": None,
        "vazao_defluente_m3s": 0.0,
        "vazao_turbinada_m3s": 0.0,
        "vazao_desviada_m3s": 0.0,
        "vazao_vertida_m3s": 0.0,
        "volume_util_final_hm3": 0.0,
        "geracao_MW": 0.0,
    }
    return pl.DataFrame([{**base, **row} for row in rows])


def _no_dec_oper(*_args: object, **_kwargs: object) -> pl.DataFrame:
    """A ``read_dec_oper_usih``/``read_dec_oper_usit`` stub for "this deck has
    no such table": raises ``FileNotFoundError`` like the real reader would,
    so `_dec_oper_hydro_stage_frame`/`_dec_oper_thermal_stage_frame`'s own
    degrade-to-empty ``except`` path is exercised -- a bare ``pl.DataFrame()``
    (no columns at all) is not a shape the real reader ever returns (it
    raises on an empty parse) and trips `_scenario_mean`'s ``group_by``."""
    raise FileNotFoundError("dec_oper_*.csv not found")
