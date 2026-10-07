"""XSS / HTML-injection regression tests for the dashboard and compare report.

The source-model-derived strings (plant/line/case names) flow into the generated HTML
and into ``<script>`` data blobs. These tests pin the escaping primitives and the
chokepoints that consume untrusted strings, so a crafted name cannot inject markup or
break out of a script context.
"""

from __future__ import annotations

import json

from novomodelo_bridge.ui.html.document import (
    build_html,
    escape_attr,
    escape_text,
    json_for_script,
)
from novomodelo_bridge.ui.html.plotly import plotly_div

_ATTACK = "</script><img src=x onerror=alert(1)>"


class TestEscapePrimitives:
    def test_escape_text_neutralises_markup(self) -> None:
        assert escape_text("<b>&'\"") == "&lt;b&gt;&amp;'\""

    def test_escape_attr_also_escapes_quotes(self) -> None:
        # quote=True so the value cannot terminate a quoted attribute.
        assert escape_attr('" onmouseover=x') == "&quot; onmouseover=x"
        assert escape_attr("' onmouseover=x") == "&#x27; onmouseover=x"

    def test_escape_handles_non_str(self) -> None:
        assert escape_text(42) == "42"


class TestJsonForScript:
    def test_neutralises_script_break(self) -> None:
        out = json_for_script({"name": _ATTACK})
        # No raw angle brackets survive to break out of <script>.
        assert "<" not in out and ">" not in out
        assert "</script>" not in out

    def test_round_trips_via_json_loads(self) -> None:
        payload = {"name": _ATTACK, "nested": [1, "a&b", "c<d>e"]}
        # \uXXXX escapes are valid JSON, so the data is preserved exactly.
        assert json.loads(json_for_script(payload)) == payload

    def test_preserves_ordinary_spaces(self) -> None:
        # The U+2028/U+2029 escaping must not touch ordinary ASCII spaces.
        assert json_for_script({"k": "a b c"}) == '{"k":"a b c"}'

    def test_escapes_js_line_separators(self) -> None:
        # U+2028 / U+2029 are valid in JSON strings but break a <script> block.
        raw = "a" + chr(0x2028) + "b" + chr(0x2029) + "c"
        out = json_for_script({"k": raw})
        assert chr(0x2028) not in out and chr(0x2029) not in out
        assert "\\u2028" in out and "\\u2029" in out
        assert json.loads(out) == {"k": raw}


class TestBuildHtmlChokepoint:
    def _doc(self, title: str, label: str = "Overview") -> str:
        return build_html(
            title=title,
            tab_defs=[("overview", label)],
            tab_contents={"overview": "<p>ok</p>"},
            css="",
            js="",
        )

    def test_escapes_untrusted_title(self) -> None:
        # title carries the case-directory name.
        out = self._doc("case</title><script>alert(1)</script>")
        assert "<script>alert(1)</script>" not in out
        assert "case&lt;/title&gt;&lt;script&gt;" in out

    def test_escapes_tab_label(self) -> None:
        out = self._doc("Dashboard", label="Over<view>")
        assert "Over&lt;view&gt;" in out
        assert "<view>" not in out


class TestPlotlyDivScriptEmbed:
    def test_trace_name_cannot_break_out_of_script(self) -> None:
        div = plotly_div([{"type": "bar", "name": _ATTACK}], {"title": "t"})
        # The malicious trace name must not appear as raw markup.
        assert "</script><img" not in div
        # ...and must appear in its escaped form inside the data blob.
        assert "\\u003c/script\\u003e" in div
