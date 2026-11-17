"""End-to-end schema-load gate for the DECOMP conversion track.

Tier-2 (needs the ``novomodelo-python`` wheel, no solver binary): runs the real
``convert_decomp_case`` against the committed ``tests/decks/decomp_mini/``
deck and then the real ``novomodelo.io.validate`` on the produced case, so a
schema-shape regression that makes converted output unloadable fails CI
instead of passing silently.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from idecomp.decomp import Dadger, Vazoes

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.converters.cadastro import build_effective_cadastro
from novomodelo_bridge.decomp.converters.penalties import FICTITIOUS_EXCHANGE_FACTOR
from novomodelo_bridge.decomp.fcf.bootstrap import bootstrap_terminal_manifest
from novomodelo_bridge.decomp.fcf.coupling import resolve_couplings
from novomodelo_bridge.decomp.fcf.importer import (
    _complexo_map,
    _patch_policy_boundary,
    _seed_recent_observations,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.pipeline import convert_decomp_case
from novomodelo_bridge.decomp.scenarios import convert_recent_observation_windows
from novomodelo_bridge.novomodelo.case_writer import CaseWriter
from novomodelo_bridge.novomodelo.readers import read_novomodelo_hydro_block_operation
from tests._fcf_fixtures import (
    make_boundary_cuts,
    make_cut_record,
    synthetic_roundtrip,
)
from tests.conftest import (
    FloatDigitsKwargs,
    case_json_floats,
    requires_block_thermal_cost,
    requires_novomodelo_python,
    requires_writer_binding,
)

#: A ``CI`` import contract 7 in submarket 1, and an ``FE`` line putting it into
#: ``RE 21`` with coefficient -0.5 (fixed-width columns per the DECOMP manual).
_CONTRACT_LINES = (
    "CI    7  1 IMPTEST       1       0  300     10.00"
    "    0  300     10.00    0  300     10.00",
    "FE   21   1     7   1       -0.500",
)

#: Stage-2 ``CT`` records: ANGRA 1 per-block bounds, ANGRA 2 uniform bounds, both differing CVUs.
_DIFFERING_CVU_ANCHOR = "CT   13   1   ANGRA 2    1   1350.1350.     20.121350.1350.     20.121350.1350.     20.12"
_DIFFERING_CVU_LINES = (
    "CT    1   1   ANGRA 1    2   500.0500.0     30.00480.0480.0     31.00460.0460.0     32.00",
    "CT   13   1   ANGRA 2    2   1350.1350.     20.001350.1350.     21.001350.1350.     22.00",
)

#: After ``HV 117``: ``HV 118`` floors plant 18's (A. VERMELHA) stage outflow
#: volume plus half its stored volume, ``HV 119`` caps pumping station 1's
#: stage pumped volume (fixed-width columns per the DECOMP manual).
_RHV_VOLUME_ANCHOR = "CV  117   1    43  -1.5866762     VARM"
_RHV_VOLUME_LINES = (
    "HV  118   1    4",
    "LV  118   1         10.0",
    "CV  118   1    18         1.0     VDEF",
    "CV  118   1    18         0.5     VARM",
    "HV  119   1    4",
    "LV  119   1                  500.0",
    "CV  119   1     1         1.0     VBOM",
)


def _deck_with_lines(
    deck: Path, tmp_path: Path, anchor: str, lines: tuple[str, ...]
) -> Path:
    """A copy of *deck* with *lines* inserted after the dadger line *anchor*."""
    copy = tmp_path / "deck"
    shutil.copytree(deck, copy)
    dadger = copy / "dadger.rv2"
    text = dadger.read_text(encoding="latin-1")
    assert text.count(f"{anchor}\n") == 1
    dadger.write_text(
        text.replace(
            f"{anchor}\n", f"{anchor}\n" + "".join(f"{line}\n" for line in lines)
        ),
        encoding="latin-1",
    )
    return copy


def _differing_cvu_deck(deck: Path, tmp_path: Path) -> Path:
    copy = _deck_with_lines(deck, tmp_path, _DIFFERING_CVU_ANCHOR, _DIFFERING_CVU_LINES)
    ct = Dadger.read(str(copy / "dadger.rv2")).ct(df=True)
    stage_2 = ct[ct["estagio"] == 2].set_index("codigo_usina")
    expected = {
        1: ([30.0, 31.0, 32.0], [500.0, 480.0, 460.0]),
        13: ([20.0, 21.0, 22.0], [1350.0, 1350.0, 1350.0]),
    }
    for code, (cvu, bounds) in expected.items():
        row = stage_2.loc[code]
        assert [row[f"cvu_{b}"] for b in (1, 2, 3)] == cvu
        assert [row[f"disponibilidade_{b}"] for b in (1, 2, 3)] == bounds
        assert [row[f"inflexibilidade_{b}"] for b in (1, 2, 3)] == bounds
    return copy


@requires_novomodelo_python
def test_decomp_mini_deck_converts_and_validates(
    decomp_mini_deck: Path, tmp_path: Path
) -> None:
    import novomodelo.io  # tier-2: guarded import inside the body, never module scope

    dst = tmp_path / "converted"
    convert_decomp_case(decomp_mini_deck, dst, force=True)

    result = novomodelo.io.validate(str(dst))
    assert result["valid"] is True
    assert result["errors"] == []


@requires_novomodelo_python
@requires_block_thermal_cost
def test_differing_ct_cvus_case_validates(
    decomp_mini_deck: Path, tmp_path: Path
) -> None:
    import novomodelo.io  # tier-2: guarded import inside the body, never module scope

    deck = _differing_cvu_deck(decomp_mini_deck, tmp_path)
    dst = tmp_path / "converted"
    convert_decomp_case(deck, dst, force=True)

    result = novomodelo.io.validate(str(dst))
    assert result["valid"] is True
    assert result["errors"] == []


@requires_novomodelo_python
def test_fe_contract_term_reaches_the_re_generic_constraint(
    decomp_mini_deck: Path, tmp_path: Path
) -> None:
    import novomodelo.io  # tier-2: guarded import inside the body, never module scope

    deck = _deck_with_lines(
        decomp_mini_deck, tmp_path, "FU   21   1    21         1.0", _CONTRACT_LINES
    )
    dst = tmp_path / "converted"
    convert_decomp_case(deck, dst, force=True)

    result = novomodelo.io.validate(str(dst))
    assert result["valid"] is True
    assert result["errors"] == []
    document = json.loads(
        (dst / "constraints" / "generic_constraints.json").read_text()
    )
    (re_21,) = [c for c in document["constraints"] if c["name"] == "RE_21"]
    assert re_21["expression"].endswith("- 0.5 * contract_import(0)")


@requires_novomodelo_python
def test_rhv_volume_tipo_case_validates(decomp_mini_deck: Path, tmp_path: Path) -> None:
    import novomodelo.io  # tier-2: guarded import inside the body, never module scope

    deck = _deck_with_lines(
        decomp_mini_deck, tmp_path, _RHV_VOLUME_ANCHOR, _RHV_VOLUME_LINES
    )
    dst = tmp_path / "converted"
    convert_decomp_case(deck, dst, force=True)

    result = novomodelo.io.validate(str(dst))
    assert result["valid"] is True
    assert result["errors"] == []


def test_rhv_volume_tipos_write_stage_level_generic_rows(
    decomp_mini_deck: Path, tmp_path: Path
) -> None:
    deck = _deck_with_lines(
        decomp_mini_deck, tmp_path, _RHV_VOLUME_ANCHOR, _RHV_VOLUME_LINES
    )
    dst = tmp_path / "converted"
    convert_decomp_case(deck, dst, force=True)

    hydros = json.loads((dst / "system" / "hydros.json").read_text())["hydros"]
    (vermelha,) = [h for h in hydros if h["name"] == "A. VERMELHA"]
    document = json.loads(
        (dst / "constraints" / "generic_constraints.json").read_text()
    )
    by_name = {c["name"]: c for c in document["constraints"]}
    hv_118, hv_119 = by_name["HV_118"], by_name["HV_119"]
    hydro_id = vermelha["id"]
    assert hv_118["expression"] == (
        f"hydro_outflow_volume({hydro_id}) + 0.5 * hydro_storage({hydro_id})"
    )
    assert hv_119["expression"] == "pumping_volume(0)"

    rows = pq.read_table(
        dst / "constraints" / "generic_constraint_bounds.parquet"
    ).to_pylist()
    floor = 10.0 + 0.5 * vermelha["reservoir"]["min_storage_hm3"]
    expected = {
        hv_118["id"]: (floor, None),
        hv_119["id"]: (None, 500.0),
    }
    for constraint_id, sides in expected.items():
        own = [r for r in rows if r["constraint_id"] == constraint_id]
        assert [r["stage_id"] for r in own] == [0, 1, 2, 3]
        assert all(r["block_id"] is None for r in own)
        assert {(r["bound_lower"], r["bound_upper"]) for r in own} == {sides}


@pytest.mark.parametrize(
    ("kwargs", "digits"), [({"float_digits": 4}, 4), ({}, 15)], ids=["4", "default"]
)
def test_decomp_mini_deck_writes_json_floats_at_requested_digits(
    decomp_mini_deck: Path, tmp_path: Path, kwargs: FloatDigitsKwargs, digits: int
) -> None:
    dst = tmp_path / "converted"
    convert_decomp_case(decomp_mini_deck, dst, force=True, **kwargs)

    floats = case_json_floats(dst)
    assert floats
    assert all(value == float(f"{value:.{digits}g}") for value in floats)


def test_differing_ct_cvus_write_exact_block_cost_rows(
    decomp_mini_deck: Path, tmp_path: Path
) -> None:
    deck = _differing_cvu_deck(decomp_mini_deck, tmp_path)
    base_dst = tmp_path / "base"
    dst = tmp_path / "converted"
    convert_decomp_case(decomp_mini_deck, base_dst, force=True)
    convert_decomp_case(deck, dst, force=True)

    thermals = json.loads((dst / "system" / "thermals.json").read_text())["thermals"]
    ids = {t["name"]: t["id"] for t in thermals}
    angra_1, angra_2 = ids["ANGRA 1"], ids["ANGRA 2"]

    keyed = []
    for case in (base_dst, dst):
        table = pq.read_table(case / "constraints" / "thermal_bounds.parquet")
        rows = table.to_pylist()
        by_key = {(r["thermal_id"], r["stage_id"], r["block_id"]): r for r in rows}
        assert len(by_key) == len(rows)
        keyed.append(by_key)
    base_rows, rows = keyed

    expected = {
        (angra_1, 1, 0): (500.0, 500.0, 30.0),
        (angra_1, 1, 1): (480.0, 480.0, 31.0),
        (angra_1, 1, 2): (460.0, 460.0, 32.0),
        (angra_1, 1, None): (None, None, pytest.approx(31.202380952380953)),
    }
    stage_means = {
        1: 21.202380952380953,
        2: 21.31547619047619,
        3: 21.26388888888889,
    }
    for stage_id, mean in stage_means.items():
        expected[angra_2, stage_id, None] = (1350.0, 1350.0, pytest.approx(mean))
        for block_id, cvu in enumerate((20.0, 21.0, 22.0)):
            expected[angra_2, stage_id, block_id] = (None, None, cvu)

    touched = {(angra_1, 1), *((angra_2, stage_id) for stage_id in stage_means)}
    assert {
        key: (r["min_generation_mw"], r["max_generation_mw"], r["cost_per_mwh"])
        for key, r in rows.items()
        if key[:2] in touched
    } == expected
    assert {k: r for k, r in rows.items() if k[:2] not in touched} == {
        k: r for k, r in base_rows.items() if k[:2] not in touched
    }

    assert (dst / "system" / "thermals.json").read_bytes() == (
        base_dst / "system" / "thermals.json"
    ).read_bytes()


def test_decomp_mini_micro_tier_keeps_its_order(
    decomp_mini_deck: Path, tmp_path: Path
) -> None:
    dst = tmp_path / "converted"
    convert_decomp_case(decomp_mini_deck, dst, force=True)

    penalties = json.loads((dst / "penalties.json").read_text())
    lines = json.loads((dst / "system" / "lines.json").read_text())["lines"]
    h = penalties["hydro"]
    assert h["turbined_cost"] > h["spillage_cost"] > 0
    assert h["diversion_cost"] == h["spillage_cost"] > 0

    carrying = [line for line in lines if "exchange_cost" in line]
    (fictitious,) = {line["exchange_cost"] for line in carrying}
    assert penalties["line"]["exchange_cost"] > fictitious > 0
    assert all(
        line["exchange_cost"]
        == penalties["line"]["exchange_cost"] * FICTITIOUS_EXCHANGE_FACTOR
        for line in carrying
    )

    violation_floor = min(
        h[k]
        for k in (
            "turbined_violation_below_cost",
            "outflow_violation_below_cost",
            "outflow_violation_above_cost",
            "evaporation_violation_cost",
            "water_withdrawal_violation_cost",
        )
    )
    assert max(h["spillage_cost"], h["diversion_cost"]) < violation_floor


def test_decomp_mini_halves_exchange_only_on_lines_touching_marketless_buses(
    decomp_mini_deck: Path, tmp_path: Path
) -> None:
    dst = tmp_path / "converted"
    convert_decomp_case(decomp_mini_deck, dst, force=True)

    buses = json.loads((dst / "system" / "buses.json").read_text())["buses"]
    lines = json.loads((dst / "system" / "lines.json").read_text())["lines"]
    marketless = {b["id"] for b in buses if "deficit_segments" not in b}
    assert all(
        ("exchange_cost" in line)
        == (line["source_bus_id"] in marketless or line["target_bus_id"] in marketless)
        for line in lines
    )
    assert any("exchange_cost" in line for line in lines)
    assert any("exchange_cost" not in line for line in lines)


#: decomp_mini's one ``NW`` plant: no initial volume, run-of-river, draining
#: into SALTO.
_ESTRELA = 260
_NW_CASE_FILES = ("initial_conditions.json",)
_NW_CASE_DIRS = ("system", "scenarios", "constraints")


def _hydros(case_dir: Path) -> list[dict]:
    return json.loads((case_dir / "system" / "hydros.json").read_text())["hydros"]


def _hydro_named(hydros: list[dict], name: str) -> dict:
    (hydro,) = [h for h in hydros if h["name"] == name]
    return hydro


def test_decomp_mini_converts_estrela_as_a_dormant_hydro(
    decomp_mini_deck: Path, decomp_mini_case: DecompCase, tmp_path: Path
) -> None:
    dst = tmp_path / "converted"
    convert_decomp_case(decomp_mini_deck, dst, force=True)

    hydros = _hydros(dst)
    estrela = _hydro_named(hydros, "ESTRELA")
    salto = _hydro_named(hydros, "SALTO")
    id_map = DecompIdMap.from_dadger(decomp_mini_case.dadger)
    assert id_map.dormant_hydro_codes == (_ESTRELA,)
    assert [h["id"] for h in hydros] == list(range(len(id_map.hydro_codes)))
    assert estrela["id"] == id_map.hydro_id(_ESTRELA)
    assert estrela["entry_stage_id"] == len(decomp_mini_case.calendar)
    assert estrela["operational_start_date"] == "2026-05-01"
    assert estrela["downstream_id"] == salto["id"]
    assert "entry_stage_id" not in salto

    initial = json.loads((dst / "initial_conditions.json").read_text())
    storage = {e["hydro_id"]: e["value_hm3"] for e in initial["storage"]}
    reference = float(decomp_mini_case.hidr.loc[_ESTRELA, "volume_referencia"])
    assert storage[estrela["id"]] == pytest.approx(reference)


def test_decomp_mini_writes_no_operation_bounds_for_estrela(
    decomp_mini_deck: Path, tmp_path: Path
) -> None:
    dst = tmp_path / "converted"
    with dx.collect() as collected:
        convert_decomp_case(decomp_mini_deck, dst, force=True)

    estrela = _hydro_named(_hydros(dst), "ESTRELA")["id"]
    for name in ("hydro_bounds", "hydro_unit_group_bounds"):
        table = pq.read_table(dst / "constraints" / f"{name}.parquet")
        assert estrela not in table.column("hydro_id").to_pylist()

    models = json.loads((dst / "system" / "hydro_production_models.json").read_text())[
        "production_models"
    ]
    (model,) = [m for m in models if m["hydro_id"] == estrela]
    assert [r["model"] for r in model["stage_ranges"]] == ["constant_productivity"]
    productivity = pq.read_table(
        dst / "system" / "hydro_energy_productivity.parquet"
    ).to_pylist()
    assert estrela in {row["hydro_id"] for row in productivity}

    listing = {
        "decomp-coupling-only-plants-excluded",
        "boundary-fcf-source-only-plants-dropped",
        "decomp-runofriver-outflow-relaxed",
    }
    for diagnostic in collected:
        if diagnostic.code in listing and diagnostic.table is not None:
            assert _ESTRELA not in [row[0] for row in diagnostic.table.rows]


def _with_reversed_uh_lines(deck: Path, tmp_path: Path) -> Path:
    copy = tmp_path / "reversed"
    shutil.copytree(deck, copy)
    dadger = copy / "dadger.rv2"
    lines = dadger.read_text(encoding="latin-1").splitlines(keepends=True)
    slots = [i for i, line in enumerate(lines) if line.startswith("UH ")]
    uh_lines = [lines[i] for i in slots]
    for slot, line in zip(slots, reversed(uh_lines), strict=True):
        lines[slot] = line
    assert uh_lines != [lines[i] for i in slots]
    dadger.write_text("".join(lines), encoding="latin-1")
    return copy


def test_uh_row_order_does_not_change_the_converted_case(
    decomp_mini_deck: Path, tmp_path: Path
) -> None:
    reversed_deck = _with_reversed_uh_lines(decomp_mini_deck, tmp_path)
    base_dst = tmp_path / "base"
    reversed_dst = tmp_path / "reversed_case"
    convert_decomp_case(decomp_mini_deck, base_dst, force=True)
    convert_decomp_case(reversed_deck, reversed_dst, force=True)

    files = [Path(name) for name in _NW_CASE_FILES] + [
        path.relative_to(base_dst)
        for directory in _NW_CASE_DIRS
        for path in sorted((base_dst / directory).rglob("*"))
        if path.is_file()
    ]
    assert len(files) > len(_NW_CASE_FILES)
    for relative in files:
        assert (reversed_dst / relative).read_bytes() == (
            base_dst / relative
        ).read_bytes(), relative


_HYDRO_INFLOW_LAG = 1
#: The writer numbers the inflow-lag slots it reserves by lag order from 1.
_LAG_DEPTH_1_SUBINDEX = 1
_SALTO = 262


@requires_novomodelo_python
@requires_writer_binding
def test_boundary_cuts_bind_the_dormant_nw_plant_and_the_case_runs(
    decomp_mini_deck: Path, tmp_path: Path
) -> None:
    import novomodelo.io  # tier-2: guarded import inside the body, never module scope

    case_dir = tmp_path / "case"
    convert_decomp_case(decomp_mini_deck, case_dir, force=True)
    case = DecompCase.from_directory(decomp_mini_deck)
    assert _ESTRELA in case.id_map.dormant_hydro_codes
    estrela = case.id_map.hydro_id(_ESTRELA)

    manifest = bootstrap_terminal_manifest(case_dir, work_dir=tmp_path / "work")

    lag_estrela = (-0.5,) + (0.0,) * 11
    cuts = make_boundary_cuts(
        (_ESTRELA, _SALTO),
        (
            make_cut_record(
                pi_varm=(0.0, 0.0), pi_qafl=(lag_estrela, (0.0,) * 12), rhs=1.0e6
            ),
        ),
    )

    effective, _ = build_effective_cadastro(case.dadger, case.hidr, case.calendar)
    couplings = resolve_couplings(
        (_ESTRELA, _SALTO), case.id_map, effective, _complexo_map(case.dadger)
    ).couplings
    assert [c.plant_index for c in couplings if c.hydro_id == estrela] == [0]

    policy = synthetic_roundtrip(
        case_dir / "boundary",
        cuts,
        manifest,
        case.id_map,
        stage_id=cuts.boundary_stage,
        couplings=couplings,
        inflow_lag_depth=1,
    )
    (terminal,) = policy["stage_cuts"]
    slots = [
        (slot["entity_type"], slot["entity_id"], slot["subindex"])
        for slot in terminal["entity_manifest"]
    ]
    position = slots.index((_HYDRO_INFLOW_LAG, estrela, _LAG_DEPTH_1_SUBINDEX))
    (cut,) = terminal["cuts"]
    assert cut["coefficients"][position] != 0.0

    config = json.loads((case_dir / "config.json").read_text())
    initial_conditions = json.loads((case_dir / "initial_conditions.json").read_text())
    writer = CaseWriter(case_dir)
    _patch_policy_boundary(writer, config)
    _seed_recent_observations(
        writer,
        convert_recent_observation_windows(
            Vazoes.read(str(case.files.vazoes)), effective, case.id_map, case.calendar
        ),
        initial_conditions,
    )

    result = novomodelo.io.validate(str(case_dir))
    assert result["valid"] is True
    assert result["errors"] == []

    output_dir = case_dir / "output"
    novomodelo.run.run(
        str(case_dir),
        output_dir=str(output_dir),
        config_overrides={
            "training.stopping_rules": [{"type": "iteration_limit", "limit": 1}],
            "simulation.enabled": True,
        },
    )

    operation = read_novomodelo_hydro_block_operation(output_dir)
    rows = operation.filter(operation["hydro_id"] == estrela).to_dicts()
    assert rows
    stored = {e["hydro_id"]: e["value_hm3"] for e in initial_conditions["storage"]}
    for row in rows:
        assert row["turbined_m3s"] == 0.0
        assert row["generation_mw"] == 0.0
        assert row["storage_initial_hm3"] == pytest.approx(stored[estrela], abs=1e-6)
        assert row["storage_final_hm3"] == pytest.approx(stored[estrela], abs=1e-6)
