"""A valid DECOMP deck the conversion cannot carry fails as ``source-unsupported``,
with the same diagnostic from ``convert decomp`` and ``check decomp``.

Tier-1: the refusal fires before anything is written, so nothing needs
``novomodelo``.
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, Any

import pytest
from idecomp.decomp import Dadger
from typer.testing import CliRunner

from novomodelo_bridge.cli import app

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _replace(text: str, old: str, new: str, *, count: int = 1) -> str:
    assert text.count(old) == count
    return text.replace(old, new)


def _partial_deficit_depth(text: str) -> str:
    """Subsystem 1's deficit curve covers 95 % of the load on its first block."""
    return _replace(
        text,
        "CD   1    1   1PDEF      1     100   8291.25",
        "CD   1    1   1PDEF      1      95   8291.25",
    )


def _reserved_subsystem_name(text: str) -> str:
    """The fictitious subsystem 11 takes the transhipment node's name."""
    return _replace(text, "SB  11   FC", "SB  11   IV")


def _summer_time_week(text: str) -> str:
    """The first operative week loses an hour of its second block, as a
    summer-time (VR) week does."""
    lines = text.split("\n")
    first_stage = [i for i, line in enumerate(lines) if line.startswith("DP   1 ")]
    assert len(first_stage) == 5
    for i in first_stage:
        lines[i] = _replace(lines[i], "      74.0", "      73.0")
    return "\n".join(lines)


_EDIT_BY_FIELD: dict[str, Callable[[str], str]] = {
    "CD limite_superior": _partial_deficit_depth,
    "SB nome_submercado": _reserved_subsystem_name,
    "DP duracao": _summer_time_week,
}


def _edited_deck(deck: Path, tmp_path: Path, edit: Callable[[str], str]) -> Path:
    copy = tmp_path / "deck"
    shutil.copytree(deck, copy)
    dadger = copy / "dadger.rv2"
    dadger.write_text(edit(dadger.read_text(encoding="latin-1")), encoding="latin-1")
    return copy


def test_the_edits_parse_as_intended(decomp_mini_deck: Path, tmp_path: Path) -> None:
    def parsed(edit: Callable[[str], str], name: str) -> Dadger:
        deck = _edited_deck(decomp_mini_deck, tmp_path / name, edit)
        return Dadger.read(str(deck / "dadger.rv2"))

    cd = parsed(_partial_deficit_depth, "cd").cd(df=True)
    assert cd["limite_superior_1"].tolist() == [95.0, 100.0, 100.0, 100.0]
    sb = parsed(_reserved_subsystem_name, "sb").sb(df=True)
    assert sb["nome_submercado"].str.strip().tolist() == ["SE", "S", "NE", "N", "IV"]
    dp = parsed(_summer_time_week, "dp").dp(df=True)
    first = dp[dp["estagio"] == 1]
    assert first["duracao_2"].tolist() == [73.0] * 5


def _unsupported(result: Any) -> list[dict[str, Any]]:
    document = json.loads(result.stdout)
    return [d for d in document["diagnostics"] if d["code"] == "source-unsupported"]


@pytest.mark.parametrize("field", list(_EDIT_BY_FIELD))
def test_convert_and_check_decomp_refuse_the_deck_alike(
    field: str, decomp_mini_deck: Path, tmp_path: Path
) -> None:
    deck = _edited_deck(decomp_mini_deck, tmp_path, _EDIT_BY_FIELD[field])
    runner = CliRunner()

    convert = runner.invoke(
        app, ["convert", "decomp", str(deck), str(tmp_path / "out"), "--json"]
    )
    check = runner.invoke(app, ["check", "decomp", str(deck), "--json"])

    assert (convert.exit_code, check.exit_code) == (1, 2)
    [converted] = _unsupported(convert)
    [checked] = _unsupported(check)
    assert converted["notes"] == ["file: dadger.rvN", f"field: {field}"]
    assert checked["notes"] == converted["notes"]
    assert checked["summary"] == converted["summary"]
