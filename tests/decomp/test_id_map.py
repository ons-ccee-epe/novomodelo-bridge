"""Tests for the deterministic entity-id map (``decomp/id_map.py``).

Tier 1 — pure Python; a duck-typed stub deck exercises the ``from_dadger``
parse boundary with no ``idecomp``/``novomodelo`` import.
"""

from __future__ import annotations

import pytest

from novomodelo_bridge.core.errors import FieldParseError, diagnostic_from_exception
from novomodelo_bridge.decomp.id_map import DecompIdMap


class _NoSbDadger:
    """A stub deck whose ``SB`` register is absent — the only accessor
    ``from_dadger`` reaches before the raise."""

    def sb(self, df: bool = True) -> None:
        return None


class TestFromDadgerSbBoundary:
    def test_no_sb_records_raises_field_parse_error(self) -> None:
        with pytest.raises(FieldParseError) as exc_info:
            DecompIdMap.from_dadger(_NoSbDadger())  # type: ignore[arg-type]

        exc = exc_info.value
        assert exc.field == "SB register"
        assert exc.path is None
        assert exc.row is None
        assert "SB records" in str(exc)
        assert not isinstance(exc, ValueError)

    def test_diagnostic_from_the_raised_error_is_source_field_parse(self) -> None:
        with pytest.raises(FieldParseError) as exc_info:
            DecompIdMap.from_dadger(_NoSbDadger())  # type: ignore[arg-type]

        diag = diagnostic_from_exception(exc_info.value, context="Conversion")
        assert diag.code == "source-field-parse"
        assert "field: SB register" in diag.notes
        assert diag.remediation is not None
