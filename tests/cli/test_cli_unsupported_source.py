"""A valid DECOMP deck the conversion cannot carry fails as ``source-unsupported``.

Each case edits a copy of the DECOMP mini-deck. ``convert decomp`` stops on it
with exit 1, and ``check decomp`` never crashes on it and, where it reaches the
refusal, fails with the same diagnostic. ``compare decomp`` stops with exit 1
on a refusal it meets before reading the Novomodelo output.

Tier-1: the refusal fires before anything is written, so nothing needs
``novomodelo``.
"""

from __future__ import annotations

import json
import shutil
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

import pytest
from idecomp.decomp import Dadger, Dadgnl
from typer.testing import CliRunner

from novomodelo_bridge.cli import app
from tests.conftest import branching_vazoes

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _replace(text: str, old: str, new: str, *, count: int = 1) -> str:
    assert text.count(old) == count
    return text.replace(old, new)


def _edit_dadger(deck: Path, edit: Callable[[str], str]) -> None:
    dadger = deck / "dadger.rv2"
    dadger.write_text(edit(dadger.read_text(encoding="latin-1")), encoding="latin-1")


def _half_deficit_depth(deck: Path) -> AbstractContextManager[None]:
    """Subsystem 1's deficit curve covers 50 % of the load on its first block."""
    _edit_dadger(
        deck,
        lambda text: _replace(
            text,
            "CD   1    1   1PDEF      1     100   8291.25",
            "CD   1    1   1PDEF      1      50   8291.25",
        ),
    )
    return nullcontext()


def _block_varying_deficit_cost(deck: Path) -> AbstractContextManager[None]:
    """Subsystem 1's deficit costs more on its first block."""
    _edit_dadger(
        deck,
        lambda text: _replace(
            text,
            "CD   1    1   1PDEF      1     100   8291.25",
            "CD   1    1   1PDEF      1     100   9000.00",
        ),
    )
    return nullcontext()


def _reserved_subsystem_name(deck: Path) -> AbstractContextManager[None]:
    """The fictitious subsystem 11 takes the transhipment node's name."""
    _edit_dadger(deck, lambda text: _replace(text, "SB  11   FC", "SB  11   IV"))
    return nullcontext()


def _summer_time_week(deck: Path) -> AbstractContextManager[None]:
    """The first operative week loses an hour of its second block, as a
    summer-time (VR) week does."""

    def edit(text: str) -> str:
        lines = text.split("\n")
        first_stage = [i for i, line in enumerate(lines) if line.startswith("DP   1 ")]
        assert len(first_stage) == 5
        for i in first_stage:
            lines[i] = _replace(lines[i], "      74.0", "      73.0")
        return "\n".join(lines)

    _edit_dadger(deck, edit)
    return nullcontext()


def _branching_trunk(_deck: Path) -> AbstractContextManager[None]:
    """The inflow scenario tree branches at stage 3, before the last stage."""
    return branching_vazoes()


_GNL_PLANTS = ((86, 1, "SANTA CRUZ"), (224, 3, "PSERGIPE I"))
_GL_WEEKS = 9
_FIRST_GL_WEEK = date(2026, 3, 14)


def _dadgnl_text(*, weeks: dict[int, int]) -> str:
    """Two GNL plants with GL weeks from the study start, the last two of them
    past the horizon; *weeks* gives each plant's GL week count."""
    lines = [
        f"TG {code:4d}   {bus}   {name:<10} 1"
        + "     0.0500.0    199.22  0.0500.0    199.22  0.0500.0    199.22 "
        for code, bus, name in _GNL_PLANTS
    ]
    lines += ["GS   1   3", "GS   2   4", "GS   3   4"]
    lines += [f"NL {code:4d}   {bus}   2" for code, bus, _ in _GNL_PLANTS]
    for code, bus, _ in _GNL_PLANTS:
        for week in range(1, weeks[code] + 1):
            start = _FIRST_GL_WEEK + timedelta(weeks=week - 1)
            lines.append(
                f"GL {code:4d}   {bus}   {week:2d}          0.0  24.       0.0  67."
                f"       0.0  77. {start:%d%m%Y}"
            )
    return "\n".join(lines) + "\n"


def _gnl_commitments_ending_apart(deck: Path) -> AbstractContextManager[None]:
    """The second GNL plant's GL weeks past the horizon end a week before the
    first plant's."""
    text = _dadgnl_text(weeks={86: _GL_WEEKS, 224: _GL_WEEKS - 1})
    (deck / "dadgnl.rv2").write_text(text, encoding="latin-1")
    return nullcontext()


@dataclass(frozen=True)
class _Refusal:
    edit: Callable[[Path], AbstractContextManager[None]]
    file: str
    field: str
    #: A bridge feature not built yet, rather than a novomodelo limit: the
    #: message says so.
    not_converted_yet: bool
    check_refuses: bool


_REFUSALS: dict[str, _Refusal] = {
    "deficit depth 50 %": _Refusal(
        _half_deficit_depth,
        "dadger.rvN",
        "CD limite_superior",
        not_converted_yet=True,
        check_refuses=True,
    ),
    "deficit cost varying by block": _Refusal(
        _block_varying_deficit_cost,
        "dadger.rvN",
        "CD custo",
        not_converted_yet=False,
        check_refuses=True,
    ),
    "subsystem named IV": _Refusal(
        _reserved_subsystem_name,
        "dadger.rvN",
        "SB nome_submercado",
        not_converted_yet=False,
        check_refuses=True,
    ),
    "summer-time week": _Refusal(
        _summer_time_week,
        "dadger.rvN",
        "DP duracao",
        not_converted_yet=False,
        check_refuses=True,
    ),
    "branching trunk": _Refusal(
        _branching_trunk,
        "vazoes.rvN",
        "probabilidades cenario",
        not_converted_yet=True,
        check_refuses=True,
    ),
    "GNL commitments ending apart": _Refusal(
        _gnl_commitments_ending_apart,
        "dadgnl.rvN",
        "GL data_inicio",
        not_converted_yet=True,
        check_refuses=False,
    ),
}


def _copied_deck(deck: Path, tmp_path: Path) -> Path:
    copy = tmp_path / "deck"
    shutil.copytree(deck, copy)
    return copy


def test_the_edits_parse_as_intended(decomp_mini_deck: Path, tmp_path: Path) -> None:
    def edited(edit: Callable[[Path], AbstractContextManager[None]], name: str) -> Path:
        deck = _copied_deck(decomp_mini_deck, tmp_path / name)
        edit(deck)
        return deck

    def dadger(
        edit: Callable[[Path], AbstractContextManager[None]], name: str
    ) -> Dadger:
        return Dadger.read(str(edited(edit, name) / "dadger.rv2"))

    cd = dadger(_half_deficit_depth, "depth").cd(df=True)
    assert cd["limite_superior_1"].tolist() == [50.0, 100.0, 100.0, 100.0]
    cd = dadger(_block_varying_deficit_cost, "cost").cd(df=True)
    assert cd["custo_1"].tolist() == [9000.0, 8291.25, 8291.25, 8291.25]
    sb = dadger(_reserved_subsystem_name, "sb").sb(df=True)
    assert sb["nome_submercado"].str.strip().tolist() == ["SE", "S", "NE", "N", "IV"]
    dp = dadger(_summer_time_week, "dp").dp(df=True)
    assert dp[dp["estagio"] == 1]["duracao_2"].tolist() == [73.0] * 5

    deck = edited(_gnl_commitments_ending_apart, "gnl")
    gl = Dadgnl.read(str(deck / "dadgnl.rv2")).gl(df=True)
    last = gl.groupby("codigo_usina")["data_inicio"].last().to_dict()
    assert last == {86: "09052026", 224: "02052026"}


def test_the_gnl_file_converts_when_the_commitments_end_together(
    decomp_mini_deck: Path, tmp_path: Path
) -> None:
    deck = _copied_deck(decomp_mini_deck, tmp_path)
    (deck / "dadgnl.rv2").write_text(
        _dadgnl_text(weeks={86: _GL_WEEKS, 224: _GL_WEEKS}), encoding="latin-1"
    )

    result = CliRunner().invoke(
        app, ["convert", "decomp", str(deck), str(tmp_path / "out"), "--json"]
    )

    assert result.exit_code == 0, result.stdout


def _refusals(result: Any) -> list[dict[str, Any]]:
    document = json.loads(result.stdout)
    return [d for d in document["diagnostics"] if d["severity"] == "error"]


@pytest.mark.parametrize("case", list(_REFUSALS))
def test_convert_and_check_decomp_refuse_the_deck(
    case: str, decomp_mini_deck: Path, tmp_path: Path
) -> None:
    refusal = _REFUSALS[case]
    deck = _copied_deck(decomp_mini_deck, tmp_path)
    runner = CliRunner()

    with refusal.edit(deck):
        convert = runner.invoke(
            app, ["convert", "decomp", str(deck), str(tmp_path / "out"), "--json"]
        )
        check = runner.invoke(app, ["check", "decomp", str(deck), "--json"])

    assert convert.exit_code == 1
    [converted] = _refusals(convert)
    assert converted["code"] == "source-unsupported"
    assert converted["notes"] == [f"file: {refusal.file}", f"field: {refusal.field}"]
    assert ("not converted yet" in converted["summary"]) is refusal.not_converted_yet

    checked = _refusals(check)
    assert all(d["code"] != "unexpected-error" for d in checked)
    if refusal.check_refuses:
        assert check.exit_code == 2
        [unsupported] = [d for d in checked if d["code"] == "source-unsupported"]
        assert unsupported["notes"] == converted["notes"]
        assert unsupported["summary"] == converted["summary"]
    else:
        assert check.exit_code in (0, 1)


def test_compare_decomp_exits_1_on_a_deck_it_refuses(
    decomp_mini_deck: Path, tmp_path: Path
) -> None:
    deck = _copied_deck(decomp_mini_deck, tmp_path)
    _reserved_subsystem_name(deck)

    result = CliRunner().invoke(
        app, ["compare", "decomp", str(deck), str(tmp_path), "--json"]
    )

    assert result.exit_code == 1
    [refused] = _refusals(result)
    assert (refused["code"], refused["notes"]) == (
        "source-unsupported",
        ["file: dadger.rvN", "field: SB nome_submercado"],
    )
