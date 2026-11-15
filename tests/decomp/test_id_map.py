"""Tests for the deterministic entity-id map (``decomp/id_map.py``).

Tier 1 — pure Python; a duck-typed stub deck exercises the ``from_dadger``
parse boundary with no ``idecomp``/``novomodelo`` import.
"""

from __future__ import annotations

import pandas as pd
import pytest

from novomodelo_bridge.core.errors import FieldParseError, diagnostic_from_exception
from novomodelo_bridge.decomp.id_map import DecompIdMap


class _NoSbDadger:
    """A stub deck whose ``SB`` register is absent — the only accessor
    ``from_dadger`` reaches before the raise."""

    def sb(self, df: bool = True) -> None:
        return None

    def uh(self, df: bool = True) -> None:
        return None

    def ct(self, df: bool = True) -> None:
        return None


class TestFromDadgerSbBoundary:
    def test_no_sb_records_raises_field_parse_error(self) -> None:
        with pytest.raises(FieldParseError) as exc_info:
            DecompIdMap.from_dadger(_NoSbDadger())

        exc = exc_info.value
        assert exc.field == "SB register"
        assert exc.path is None
        assert exc.row is None
        assert "SB records" in str(exc)
        assert not isinstance(exc, ValueError)

    def test_diagnostic_from_the_raised_error_is_source_field_parse(self) -> None:
        with pytest.raises(FieldParseError) as exc_info:
            DecompIdMap.from_dadger(_NoSbDadger())

        diag = diagnostic_from_exception(exc_info.value, context="Conversion")
        assert diag.code == "source-field-parse"
        assert "field: SB register" in diag.notes
        assert diag.remediation is not None


class _UhDadger:
    """A stub deck with one subsystem and operated and coupling-only ``UH``
    rows: plants 1 and 156 carry an initial volume, 46 and 134 do not."""

    def sb(self, df: bool = True) -> pd.DataFrame:
        return pd.DataFrame({"codigo_submercado": [1], "nome_submercado": ["SE"]})

    def uh(self, df: bool = True) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "codigo_usina": [156, 46, 1, 134],
                "volume_inicial": [94.71, None, 96.13, None],
            }
        )

    def ct(self, df: bool = True) -> None:
        return None


def test_from_dadger_keeps_the_uh_rows_without_initial_volume_apart() -> None:
    id_map = DecompIdMap.from_dadger(_UhDadger())

    assert id_map.hydro_codes == (1, 156)
    assert id_map.coupling_only_codes == (46, 134)
    with pytest.raises(KeyError):
        id_map.hydro_id(46)


class _NwDadger:
    """A stub deck whose ``UH`` rows are: 5 operated, 3 ``NW`` with no volume,
    4 coupling-only, and 2 ``NW`` with a volume of 0.0."""

    def sb(self, df: bool = True) -> pd.DataFrame:
        return pd.DataFrame({"codigo_submercado": [1], "nome_submercado": ["SE"]})

    def uh(self, df: bool = True) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "codigo_usina": [5, 3, 4, 2],
                "configuracao_newave": ["", "NW", "", "NW"],
                "volume_inicial": [94.7, None, None, 0.0],
            }
        )

    def ct(self, df: bool = True) -> None:
        return None


class TestNwPlants:
    def test_nw_rows_take_dense_ids_in_ascending_code_order(self) -> None:
        id_map = DecompIdMap.from_dadger(_NwDadger())

        assert id_map.hydro_codes == (2, 3, 5)
        assert id_map.hydro_id(5) == 2
        assert id_map.hydro_id(3) == 1
        assert id_map.dormant_hydro_codes == (2, 3)
        assert id_map.operated_hydro_codes == (5,)
        assert id_map.coupling_only_codes == (4,)

    def test_dormant_codes_outside_the_hydro_codes_raise(self) -> None:
        with pytest.raises(ValueError, match="dormant"):
            DecompIdMap(
                bus_codes=(1,),
                bus_names=("SE",),
                hydro_codes=(1, 2),
                dormant_hydro_codes=(3,),
            )

    def test_dormant_codes_that_are_not_ascending_raise(self) -> None:
        with pytest.raises(ValueError, match="dormant"):
            DecompIdMap(
                bus_codes=(1,),
                bus_names=("SE",),
                hydro_codes=(1, 2),
                dormant_hydro_codes=(2, 1),
            )

    def test_dormant_codes_that_are_coupling_only_raise(self) -> None:
        with pytest.raises(ValueError, match="dormant"):
            DecompIdMap(
                bus_codes=(1,),
                bus_names=("SE",),
                hydro_codes=(1, 2),
                coupling_only_codes=(2,),
                dormant_hydro_codes=(2,),
            )

    def test_idle_hydro_codes_are_the_coupling_only_and_nw_codes(self) -> None:
        id_map = DecompIdMap.from_dadger(_NwDadger())

        assert id_map.idle_hydro_codes == (2, 3, 4)
