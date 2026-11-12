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
from idecomp.decomp import Dadger

from novomodelo_bridge.decomp.pipeline import convert_decomp_case
from tests.conftest import (
    FloatDigitsKwargs,
    case_json_floats,
    requires_block_thermal_cost,
    requires_novomodelo_python,
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
