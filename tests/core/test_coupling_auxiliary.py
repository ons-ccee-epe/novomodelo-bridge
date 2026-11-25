"""Tests for the auxiliary coupling plants' ids and names
(``core/coupling_auxiliary.py``)."""

from __future__ import annotations

import pytest

from novomodelo_bridge.core.coupling_auxiliary import (
    coupling_auxiliary_id,
    coupling_auxiliary_name,
    is_coupling_auxiliary,
)
from novomodelo_bridge.core.errors import UnsupportedSourceError


def test_coupling_auxiliary_id_offsets_the_plant_code() -> None:
    assert coupling_auxiliary_id(1) == 1001
    assert coupling_auxiliary_id(999) == 1999


@pytest.mark.parametrize("code", [0, 1000, 1200])
def test_coupling_auxiliary_id_refuses_a_code_the_offset_cannot_hold(
    code: int,
) -> None:
    with pytest.raises(UnsupportedSourceError, match="not converted yet"):
        coupling_auxiliary_id(code)


def test_is_coupling_auxiliary_starts_at_the_offset() -> None:
    assert is_coupling_auxiliary(1001)
    assert not is_coupling_auxiliary(999)


def test_coupling_auxiliary_name_prefixes_the_plant_code() -> None:
    assert coupling_auxiliary_name(288, " BELO MONTE  ") == "ACOPL-288 BELO MONTE"
    assert coupling_auxiliary_name(901, "") == "ACOPL-901"
