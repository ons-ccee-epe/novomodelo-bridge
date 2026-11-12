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

import pytest

from novomodelo_bridge.decomp.converters.penalties import FICTITIOUS_EXCHANGE_FACTOR
from novomodelo_bridge.decomp.pipeline import convert_decomp_case
from tests.conftest import (
    FloatDigitsKwargs,
    case_json_floats,
    requires_novomodelo_python,
)

#: A ``CI`` import contract 7 in submarket 1, and an ``FE`` line putting it into
#: ``RE 21`` with coefficient -0.5 (fixed-width columns per the DECOMP manual).
_CONTRACT_LINES = (
    "CI    7  1 IMPTEST       1       0  300     10.00"
    "    0  300     10.00    0  300     10.00",
    "FE   21   1     7   1       -0.500",
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
