"""End-to-end schema-load gate for the NEWAVE conversion track.

Tier-2 (needs the ``novomodelo-python`` wheel, no solver binary): runs the real
``convert_newave_case`` against the committed ``tests/decks/newave_mini/``
deck and then the real ``novomodelo.io.validate`` on the produced case, so a
schema-shape regression that makes converted output unloadable fails CI
instead of passing silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from novomodelo_bridge.newave.pipeline import convert_newave_case
from tests.conftest import case_json_floats, requires_novomodelo_python


@requires_novomodelo_python
def test_newave_mini_deck_converts_and_validates(
    newave_mini_deck: Path, tmp_path: Path
) -> None:
    import novomodelo.io  # tier-2: guarded import inside the body, never module scope

    dst = tmp_path / "converted"
    convert_newave_case(newave_mini_deck, dst)

    result = novomodelo.io.validate(str(dst))
    assert result["valid"] is True
    assert result["errors"] == []

    # Pins the CaseWriter Option-A byte format (a single trailing newline),
    # the one on-disk change from routing NEWAVE writes through the writer.
    assert (dst / "config.json").read_text().endswith("\n")


@pytest.mark.parametrize(
    ("kwargs", "digits"), [({"float_digits": 4}, 4), ({}, 15)], ids=["4", "default"]
)
def test_newave_mini_deck_writes_json_floats_at_requested_digits(
    newave_mini_deck: Path, tmp_path: Path, kwargs: dict[str, int], digits: int
) -> None:
    dst = tmp_path / "converted"
    convert_newave_case(newave_mini_deck, dst, **kwargs)

    floats = case_json_floats(dst)
    assert floats
    assert all(value == float(f"{value:.{digits}g}") for value in floats)
