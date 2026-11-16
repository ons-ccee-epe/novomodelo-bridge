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

from novomodelo_bridge.core.diagnostics import Severity
from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.constraint_registers import read_constraints
from novomodelo_bridge.decomp.converters.network import pumping_station_id_map
from novomodelo_bridge.decomp.pipeline import convert_decomp_case
from tests.conftest import (
    FloatDigitsKwargs,
    case_json_floats,
    requires_novomodelo_python,
    requires_stage_volume_tokens,
)

#: A ``CI`` import contract 7 in submarket 1, and an ``FE`` line putting it into
#: ``RE 21`` with coefficient -0.5 (fixed-width columns per the DECOMP manual).
_CONTRACT_LINES = (
    "CI    7  1 IMPTEST       1       0  300     10.00"
    "    0  300     10.00    0  300     10.00",
    "FE   21   1     7   1       -0.500",
)

#: Two single-term ``VDES`` ``HV`` registers over stages 1 to 4, each with lower
#: limit 0.0 (fixed-width columns as in the ``HV`` 117 block): ``HV`` 911 on plant
#: 43, a base cadastro ``desvio`` the conversion models no channel for, and ``HV``
#: 912 on plant 118, which carries a converted channel.
_VDES_HV_LINES = (
    "HV  911   1    4",
    "LV  911   1          0.0",
    "CV  911   1    43         1.0     VDES",
    "HV  912   1    4",
    "LV  912   1          0.0",
    "CV  912   1   118         1.0     VDES",
)

#: ``HV`` 901-904 over stages 1 to 4: ``VDEF``, ``VDES``, ``VBOM``, ``VARM`` + ``VDEF``.
_RHV_VOLUME_LINES = (
    "HV  901   1    4",
    "LV  901   1        100.0",
    "CV  901   1   118         1.0     VDEF",
    "HV  902   1    4",
    "LV  902   1          0.0      50.0",
    "CV  902   1   118         1.0     VDES",
    "HV  903   1    4",
    "LV  903   1                   20.0",
    "CV  903   1     4         1.0     VBOM",
    "HV  904   1    4",
    "LV  904   1        100.0",
    "CV  904   1    34         1.0     VARM",
    "CV  904   1   118         0.5     VDEF",
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


def test_decomp_mini_deck_hv_slack_matches_storage_violation_cost(
    decomp_mini_deck: Path, tmp_path: Path
) -> None:
    dst = tmp_path / "converted"
    convert_decomp_case(decomp_mini_deck, dst, force=True)

    constraints = json.loads(
        (dst / "constraints" / "generic_constraints.json").read_text()
    )["constraints"]
    penalties = json.loads((dst / "penalties.json").read_text())
    hv_slacks = [c["slack"] for c in constraints if c["name"].startswith("HV_")]

    assert hv_slacks
    for slack in hv_slacks:
        assert slack["penalty_basis"] == "per_unit"
        assert slack["penalty"] == penalties["hydro"]["storage_violation_below_cost"]


def test_rhv_vdes_is_refused_without_a_converted_diversion_channel(
    decomp_mini_deck: Path, decomp_mini_case: DecompCase, tmp_path: Path
) -> None:
    deck = _deck_with_lines(
        decomp_mini_deck,
        tmp_path,
        "CV  117   1    43  -1.5866762     VARM",
        _VDES_HV_LINES,
    )
    injected = {
        record.constraint_id: [(term.code, term.variable) for term in record.terms]
        for record in read_constraints(
            DecompCase.from_directory(deck).dadger
        ).by_family["HV"]
        if record.constraint_id in (911, 912)
    }
    assert injected == {911: [(43, "VDES")], 912: [(118, "VDES")]}

    dst = tmp_path / "converted"
    report = convert_decomp_case(deck, dst, force=True)

    constraints = json.loads(
        (dst / "constraints" / "generic_constraints.json").read_text()
    )["constraints"]
    names = {c["name"] for c in constraints}
    assert not names & {"HV_911", "HV_912"}
    hydro_id = decomp_mini_case.id_map.hydro_id(118)
    plant_118_stage_rows = [
        row
        for row in pq.read_table(
            dst / "constraints" / "hydro_bounds.parquet"
        ).to_pylist()
        if row["hydro_id"] == hydro_id and row["block_id"] is None
    ]
    assert len(plant_118_stage_rows) == 4
    assert {row["min_diversion_hm3"] for row in plant_118_stage_rows} == {0.0}
    (refusal,) = [
        d for d in report.diagnostics if d.code == "decomp-rhv-vdes-no-diversion"
    ]
    assert refusal.severity is Severity.WARNING
    assert "HV constraint 911" in refusal.summary
    assert "plant code 43" in refusal.summary


def test_single_term_hvs_land_in_their_entity_bound_columns(
    decomp_mini_deck: Path, decomp_mini_case: DecompCase, tmp_path: Path
) -> None:
    deck = _deck_with_lines(
        decomp_mini_deck,
        tmp_path,
        "CV  117   1    43  -1.5866762     VARM",
        _RHV_VOLUME_LINES,
    )
    injected = {
        record.constraint_id: [(term.code, term.variable) for term in record.terms]
        for record in read_constraints(
            DecompCase.from_directory(deck).dadger
        ).by_family["HV"]
        if record.constraint_id in range(901, 905)
    }
    assert injected == {
        901: [(118, "VDEF")],
        902: [(118, "VDES")],
        903: [(4, "VBOM")],
        904: [(34, "VARM"), (118, "VDEF")],
    }

    dst = tmp_path / "converted"
    convert_decomp_case(deck, dst, force=True)
    constraints = json.loads(
        (dst / "constraints" / "generic_constraints.json").read_text()
    )["constraints"]
    penalties = json.loads((dst / "penalties.json").read_text())
    bounds = pq.read_table(
        dst / "constraints" / "generic_constraint_bounds.parquet"
    ).to_pylist()
    hydro_bounds = pq.read_table(
        dst / "constraints" / "hydro_bounds.parquet"
    ).to_pylist()
    pumping_bounds = pq.read_table(
        dst / "constraints" / "pumping_bounds.parquet"
    ).to_pylist()
    reservoirs = {
        hydro["id"]: hydro["reservoir"]
        for hydro in json.loads((dst / "system" / "hydros.json").read_text())["hydros"]
    }
    hydro_id = decomp_mini_case.id_map.hydro_id
    hv = {c["name"]: c for c in constraints if c["name"].startswith("HV_")}

    assert set(hv) == {"HV_117", "HV_904"}
    assert hv["HV_904"]["expression"] == (
        f"hydro_useful_volume_final({hydro_id(34)})"
        f" + 0.5 * hydro_outflow_volume({hydro_id(118)})"
    )

    hv_ids = {c["id"] for c in hv.values()}
    hv_rows = [row for row in bounds if row["constraint_id"] in hv_ids]
    assert {row["constraint_id"] for row in hv_rows} == hv_ids
    assert all(row["block_id"] is None for row in hv_rows)
    endpoints = [
        row[side]
        for row in hv_rows
        for side in ("bound_lower", "bound_upper")
        if row[side] is not None
    ]
    assert endpoints
    assert all(abs(endpoint) < 1e20 for endpoint in endpoints)
    assert all("block" not in c["expression"] for c in hv.values())
    for c in hv.values():
        assert c["slack"]["penalty_basis"] == "per_unit"
        assert (
            c["slack"]["penalty"] == penalties["hydro"]["storage_violation_below_cost"]
        )

    plant_118 = [
        row
        for row in hydro_bounds
        if row["hydro_id"] == hydro_id(118) and row["block_id"] is None
    ]
    assert sorted(row["stage_id"] for row in plant_118) == [0, 1, 2, 3]
    for row in plant_118:
        assert row["min_outflow_hm3"] == 100.0
        assert row["min_diversion_hm3"] == 0.0
        assert row["max_diversion_hm3"] == 50.0

    station_id = pumping_station_id_map(decomp_mini_case.dadger)[4]
    station = [row for row in pumping_bounds if row["pumping_station_id"] == station_id]
    assert sorted(row["stage_id"] for row in station) == [0, 1, 2, 3]
    for row in station:
        assert row["block_id"] is None
        assert row["max_hm3"] == 20.0

    # hydros.json is written at 15 significant digits, the bounds at full precision.
    tolerance = 1e-9
    for code in (121, 120, 124, 117, 73, 115, 275, 227):
        reservoir = reservoirs[hydro_id(code)]
        for stage in range(4):
            (row,) = [
                row
                for row in hydro_bounds
                if row["hydro_id"] == hydro_id(code)
                and row["stage_id"] == stage
                and row["min_storage_hm3"] is not None
            ]
            assert row["max_storage_hm3"] is not None
            assert row["min_storage_hm3"] <= row["max_storage_hm3"]
            assert row["min_storage_hm3"] >= reservoir["min_storage_hm3"] - tolerance
            assert row["max_storage_hm3"] <= reservoir["max_storage_hm3"] + tolerance


@requires_novomodelo_python
@requires_stage_volume_tokens
def test_rhv_volume_tipos_pass_novomodelo_validation(
    decomp_mini_deck: Path, tmp_path: Path
) -> None:
    import novomodelo.io  # tier-2: guarded import inside the body, never module scope

    deck = _deck_with_lines(
        decomp_mini_deck,
        tmp_path,
        "CV  117   1    43  -1.5866762     VARM",
        _RHV_VOLUME_LINES,
    )
    dst = tmp_path / "converted"
    convert_decomp_case(deck, dst, force=True)

    result = novomodelo.io.validate(str(dst))
    assert result["valid"] is True
    assert result["errors"] == []


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
