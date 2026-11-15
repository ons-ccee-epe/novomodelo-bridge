"""Tests for the DECOMP ``UH`` admission predicates (``decomp/plants.py``).

Tier 1 -- pure Python over synthetic frames and the source text.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from novomodelo_bridge.decomp import plants
from novomodelo_bridge.decomp.plants import (
    coupling_only_codes,
    dormant_codes,
    operated_uh,
)

_DECOMP_SRC = (
    Path(__file__).resolve().parents[2] / "src" / "novomodelo_bridge" / "decomp"
)
_INLINE_PREDICATE = re.compile(r"\[\"volume_inicial\"\]\.(notna|isna)\(\)")
_NW_COLUMN = re.compile(r"configuracao_newave")


def _uh(volumes: list[float | None]) -> pd.DataFrame:
    return pd.DataFrame({"codigo_usina": [3, 1, 2], "volume_inicial": volumes})


def test_operated_uh_keeps_the_rows_with_an_initial_volume() -> None:
    uh = _uh([94.7, 96.1, None])

    pd.testing.assert_frame_equal(operated_uh(uh), uh.iloc[:2])


def test_coupling_only_codes_are_the_rows_without_an_initial_volume_in_ascending_order() -> (
    None
):
    assert coupling_only_codes(_uh([94.7, 96.1, None])) == (2,)
    assert coupling_only_codes(_uh([None, 96.1, None])) == (2, 3)


def test_dormant_codes_are_the_nw_rows_in_ascending_order() -> None:
    uh = pd.DataFrame(
        {
            "codigo_usina": [9, 4, 7, 2],
            "configuracao_newave": ["NW", None, " NW ", ""],
            "volume_inicial": [94.7, None, None, 10.0],
        }
    )

    assert dormant_codes(uh) == (7, 9)


def test_dormant_codes_are_empty_without_the_nw_column() -> None:
    assert dormant_codes(_uh([94.7, 96.1, None])) == ()


def _nw_uh() -> pd.DataFrame:
    """Row 5 is operated, row 3 is ``NW`` with no volume, row 4 is
    coupling-only, and row 2 is ``NW`` with a volume of 0.0."""
    return pd.DataFrame(
        {
            "codigo_usina": [5, 3, 4, 2],
            "configuracao_newave": ["", "NW", "", "NW"],
            "volume_inicial": [94.7, None, None, 0.0],
        }
    )


def test_operated_uh_leaves_out_nw_rows() -> None:
    uh = _nw_uh()

    pd.testing.assert_frame_equal(operated_uh(uh), uh.iloc[:1])


def test_converted_codes_are_the_operated_and_nw_codes_in_ascending_order() -> None:
    uh = _nw_uh()

    assert plants.converted_codes(uh) == (2, 3, 5)
    assert plants.converted_uh(uh)["codigo_usina"].tolist() == [5, 3, 2]


def test_coupling_only_codes_leave_out_nw_rows() -> None:
    assert coupling_only_codes(_nw_uh()) == (4,)


def test_no_other_decomp_module_compares_configuracao_newave() -> None:
    sites = [
        f"{path.relative_to(_DECOMP_SRC)}:{number}"
        for path in sorted(_DECOMP_SRC.rglob("*.py"))
        if path != _DECOMP_SRC / "plants.py"
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if _NW_COLUMN.search(line)
    ]

    assert sites == []


def test_no_other_decomp_module_filters_uh_rows_on_the_initial_volume() -> None:
    sites = [
        f"{path.relative_to(_DECOMP_SRC)}:{number}"
        for path in sorted(_DECOMP_SRC.rglob("*.py"))
        if path != _DECOMP_SRC / "plants.py"
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if _INLINE_PREDICATE.search(line)
    ]

    assert sites == []
