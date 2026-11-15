"""A valid DECOMP deck the conversion cannot carry fails as ``source-unsupported``.

Tier-1: the refusal fires before anything is written, so nothing needs
``novomodelo``.
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, Any

from idecomp.decomp import Dadger
from typer.testing import CliRunner

from novomodelo_bridge.cli import app

if TYPE_CHECKING:
    from pathlib import Path

_CD_FULL_DEPTH = "CD   1    1   1PDEF      1     100   8291.25"
_CD_PARTIAL_DEPTH = "CD   1    1   1PDEF      1      95   8291.25"


def _partial_depth_deck(deck: Path, tmp_path: Path) -> Path:
    """A copy of *deck* whose subsystem 1 deficit curve covers 95 % of the load."""
    copy = tmp_path / "deck"
    shutil.copytree(deck, copy)
    dadger = copy / "dadger.rv2"
    text = dadger.read_text(encoding="latin-1")
    assert text.count(_CD_FULL_DEPTH) == 1
    dadger.write_text(
        text.replace(_CD_FULL_DEPTH, _CD_PARTIAL_DEPTH), encoding="latin-1"
    )
    cd = Dadger.read(str(dadger)).cd(df=True)
    assert cd["limite_superior_1"].tolist() == [95.0, 100.0, 100.0, 100.0]
    return copy


def _errors(result: Any) -> list[dict[str, Any]]:
    document = json.loads(result.stdout)
    return [d for d in document["diagnostics"] if d["severity"] == "error"]


def test_convert_decomp_reports_a_partial_deficit_depth_as_unsupported(
    decomp_mini_deck: Path, tmp_path: Path
) -> None:
    deck = _partial_depth_deck(decomp_mini_deck, tmp_path)

    result = CliRunner().invoke(
        app, ["convert", "decomp", str(deck), str(tmp_path / "out"), "--json"]
    )

    assert result.exit_code == 1
    [error] = _errors(result)
    assert error["code"] == "source-unsupported"
    assert error["notes"] == ["file: dadger.rvN", "field: CD limite_superior"]
    assert "95 % of the load" in error["summary"]
