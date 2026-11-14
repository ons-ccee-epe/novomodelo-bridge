"""Tests for the shared Plotly rendering helpers."""

from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots

from novomodelo_bridge.ui.html.document import json_for_script
from novomodelo_bridge.ui.html.js import PLANT_EXPLORER_JS
from novomodelo_bridge.ui.html.plotly import (
    LEGEND_DEFAULTS,
    MARGIN_DEFAULTS,
    MAX_STAGE_TICKS,
    _normalize_plotly_titles,
    add_mean_p50_band,
    apply_stage_date_axis,
    apply_standard_layout,
    fig_to_html,
    plotly_div,
    render_figure,
    stage_axis_ticks,
    stage_date_xaxis,
)
from novomodelo_bridge.ui.theme import BAND_FILL, BAND_LINE


def _normalise_ids(s: str) -> str:
    """Replace Plotly's random div id (hex + hyphens) so structural HTML compares."""
    return re.sub(r"[0-9a-f]{4,}(-[0-9a-f]{2,})*", "ID", s)


def _bar_fig() -> go.Figure:
    return go.Figure([go.Bar(x=[1, 2], y=[3, 4], name="t")])


def test_render_figure_matches_manual_update_layout_pattern() -> None:
    """render_figure(...) must produce the exact HTML of the boilerplate it replaces.

    This is the contract the performance_charts conversion relied on:
    ``render_figure(fig, **kw)`` == ``fig.update_layout(legend=LEGEND_DEFAULTS,
    margin=MARGIN_DEFAULTS, **kw); fig_to_html(fig)``.
    """
    manual = _bar_fig()
    manual.update_layout(
        title="T",
        xaxis_title="X",
        yaxis_title="Y",
        barmode="stack",
        height=420,
        legend=LEGEND_DEFAULTS,
        margin=MARGIN_DEFAULTS,
        template="plotly_white",
    )
    manual_html = fig_to_html(manual)

    factory_html = render_figure(
        _bar_fig(),
        title="T",
        xaxis_title="X",
        yaxis_title="Y",
        barmode="stack",
        height=420,
    )

    assert _normalise_ids(manual_html) == _normalise_ids(factory_html)


def test_render_figure_forwards_unified_hover() -> None:
    manual = _bar_fig()
    manual.update_layout(
        title="A",
        legend=LEGEND_DEFAULTS,
        margin=MARGIN_DEFAULTS,
        template="plotly_white",
    )
    manual_html = fig_to_html(manual, unified_hover=False)

    factory_html = render_figure(_bar_fig(), title="A", unified_hover=False)
    assert _normalise_ids(manual_html) == _normalise_ids(factory_html)


def test_apply_standard_layout_matches_manual_and_returns_fig() -> None:
    """apply_standard_layout(fig, **kw) mutates fig like the manual update_layout
    and returns it (for make_chart_card / return-fig call sites)."""
    manual = _bar_fig()
    manual.update_layout(
        title="T",
        xaxis_title="X",
        legend=LEGEND_DEFAULTS,
        margin=MARGIN_DEFAULTS,
        template="plotly_white",
    )

    fig = _bar_fig()
    returned = apply_standard_layout(fig, title="T", xaxis_title="X")
    assert returned is fig  # mutates in place AND returns the same figure
    assert fig.to_dict()["layout"] == manual.to_dict()["layout"]


def test_render_figure_lets_caller_override_legend_and_margin() -> None:
    custom_margin = {"l": 10, "r": 10, "t": 10, "b": 10}
    fig = _bar_fig()
    render_figure(fig, title="C", margin=custom_margin)
    layout = fig.to_dict()["layout"]
    # caller's margin wins over the default
    assert layout["margin"] == custom_margin
    # legend defaulted to the shared constant
    assert layout["legend"]["orientation"] == LEGEND_DEFAULTS["orientation"]


def test_apply_standard_layout_defaults_to_plotly_white_template() -> None:
    """apply_standard_layout must default the template, agreeing with plotly_div's
    ``plotly_white`` default so both render paths share the house style."""
    fig = apply_standard_layout(_bar_fig())
    assert fig.layout.template is not None


def test_apply_standard_layout_lets_caller_override_template() -> None:
    """An explicit template= in **layout wins over the plotly_white default."""
    fig = apply_standard_layout(_bar_fig(), template="none")
    assert (
        fig.layout.template.to_plotly_json()
        == go.Figure(layout={"template": "none"}).layout.template.to_plotly_json()
    )


def _percentile_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stage_id": [1, 2],
            "mean": [1.0, 2.0],
            "p10": [0.8, 1.6],
            "p50": [0.95, 1.9],
            "p90": [1.2, 2.4],
        }
    )


def test_add_mean_p50_band_builds_traces_from_promoted_home() -> None:
    """The band helper, promoted from dashboard.chart_helpers, still builds the
    mean/p50/band trace triple from its new ``ui.html.plotly`` home."""
    fig = go.Figure()
    result = add_mean_p50_band(fig, _percentile_df(), "stage_id", "Hydro", "#3B82F6")

    assert result is fig
    assert len(result.data) == 4  # mean + p50 + p10 (invisible) + p90 (fill)
    assert result.data[0].name == "Hydro"
    assert result.data[0].line.width == 2
    assert result.data[3].fill == "tonexty"


def test_add_mean_p50_band_empty_df_is_noop() -> None:
    empty = pd.DataFrame(columns=["stage_id", "mean", "p10", "p50", "p90"])
    fig = go.Figure()
    result = add_mean_p50_band(fig, empty, "stage_id", "Hydro", "#3B82F6")
    assert len(result.data) == 0


def test_band_fill_and_line_match_legacy_literals() -> None:
    """BAND_FILL/BAND_LINE equal the pre-unification band colour literals that
    charts/_shared.py and add_mean_p50_band's fillcolor computation produced."""
    assert BAND_FILL == "rgba(74,144,184,0.15)"
    assert BAND_LINE == "rgba(255,255,255,0)"


# ---------------------------------------------------------------------------
# _normalize_plotly_titles / plotly_div title normalization
#
# plotly.js 3.x dropped bare-string `title` support; a raw layout dict handed
# to plotly_div (unlike a go.Figure, which coerces on assignment) reaches the
# browser unvalidated, so a bare string is silently dropped at render. These
# golden-blind tests catch that class of regression directly on the JSON
# payload, since the title text itself is still present either way.
# ---------------------------------------------------------------------------


def test_normalize_plotly_titles_rewrites_bare_string_title() -> None:
    layout = {"title": "Chart Title"}
    result = _normalize_plotly_titles(layout)
    assert result is layout  # mutates in place and returns it
    assert layout["title"] == {"text": "Chart Title"}


def test_normalize_plotly_titles_rewrites_nested_axis_titles() -> None:
    layout = {
        "xaxis": {"title": "Stage"},
        "xaxis2": {"title": "Stage 2"},
        "yaxis": {"title": "MW"},
    }
    _normalize_plotly_titles(layout)
    assert layout["xaxis"]["title"] == {"text": "Stage"}
    assert layout["xaxis2"]["title"] == {"text": "Stage 2"}
    assert layout["yaxis"]["title"] == {"text": "MW"}


def test_normalize_plotly_titles_rewrites_scene_axis_titles() -> None:
    layout = {
        "scene": {
            "xaxis": {"title": "Turbined (m3/s)"},
            "yaxis": {"title": "Volume (hm3)"},
            "zaxis": {"title": "GH (MW)", "autorange": True},
        }
    }
    _normalize_plotly_titles(layout)
    scene = layout["scene"]
    assert scene["xaxis"]["title"] == {"text": "Turbined (m3/s)"}
    assert scene["yaxis"]["title"] == {"text": "Volume (hm3)"}
    assert scene["zaxis"]["title"] == {"text": "GH (MW)"}
    assert scene["zaxis"]["autorange"] is True  # untouched sibling key


def test_normalize_plotly_titles_rewrites_colorbar_title() -> None:
    """A layout-level colorbar (e.g. coloraxis.colorbar) is walked too."""
    layout = {"coloraxis": {"colorbar": {"title": "Corr"}}}
    _normalize_plotly_titles(layout)
    assert layout["coloraxis"]["colorbar"]["title"] == {"text": "Corr"}


def test_normalize_plotly_titles_is_idempotent_on_object_form_title() -> None:
    """An already-object-form title (e.g. from a prior normalize pass) is untouched."""
    layout = {"title": {"text": "Already Object"}, "xaxis": {"title": {"text": "X"}}}
    _normalize_plotly_titles(layout)
    assert layout["title"] == {"text": "Already Object"}
    assert layout["xaxis"]["title"] == {"text": "X"}


def test_normalize_plotly_titles_noop_when_title_absent() -> None:
    layout = {"xaxis": {"type": "date"}, "height": 400}
    before = {"xaxis": {"type": "date"}, "height": 400}
    _normalize_plotly_titles(layout)
    assert layout == before


def test_plotly_div_bare_title_layout_emits_object_form_no_bare_titles() -> None:
    """plotly_div's output for a bare-string-title layout carries object-form
    titles and no bare ``"title": "<nonempty>"`` anywhere in the emitted JSON."""
    traces = [{"type": "scatter", "x": [1, 2], "y": [3, 4]}]
    layout = {
        "title": "X",
        "xaxis": {"title": "Y"},
        "yaxis": {"title": "Z"},
    }
    result = plotly_div(traces, layout)

    assert '"title":{"text":"X"}' in result
    assert '"title":{"text":"Y"}' in result
    assert '"title":{"text":"Z"}' in result
    assert not re.search(r'"title":"[^{"][^"]*"', result)


def _stage_axis(n: int) -> tuple[list[str], list[str]]:
    dates = [f"{2026 + i // 12}-{i % 12 + 1:02d}-01" for i in range(n)]
    return dates, [f"S{i}" for i in range(n)]


@pytest.mark.parametrize(
    ("n_stages", "stride"),
    [(1, 1), (8, 1), (9, 2), (17, 3), (30, 6), (72, 12), (144, 24), (200, 25)],
)
def test_stage_axis_ticks_label_every_stride_th_stage_from_the_first(
    n_stages: int, stride: int
) -> None:
    """The smallest of the 1/2/3/6/12/24 strides that fits the tick cap, or a
    plain fraction of the axis beyond them."""
    dates, labels = _stage_axis(n_stages)

    tickvals, ticktext = stage_axis_ticks(dates, labels)

    assert tickvals == dates[::stride]
    assert ticktext == labels[::stride]
    assert len(tickvals) <= MAX_STAGE_TICKS


def test_stage_date_xaxis_is_a_date_axis_with_the_thinned_ticks() -> None:
    dates, labels = _stage_axis(72)
    tickvals, ticktext = stage_axis_ticks(dates, labels)

    assert stage_date_xaxis(dates, labels) == {
        "type": "date",
        "tickmode": "array",
        "tickvals": tickvals,
        "ticktext": ticktext,
        "automargin": True,
    }


def test_apply_stage_date_axis_sets_every_subplot_x_axis() -> None:
    dates, labels = _stage_axis(72)
    fig = make_subplots(rows=2, cols=1)

    apply_stage_date_axis(fig, dates, labels)

    expected = stage_date_xaxis(dates, labels)["tickvals"]
    assert list(fig.layout.xaxis.tickvals) == expected
    assert list(fig.layout.xaxis2.tickvals) == expected


def test_plant_explorer_layout_embeds_the_shared_legend() -> None:
    """The JavaScript plant-explorer charts place their legend like the
    Python-built ones, from the same default."""
    assert f"legend: {json_for_script(LEGEND_DEFAULTS)}," in PLANT_EXPLORER_JS
    assert "__LEGEND_DEFAULTS__" not in PLANT_EXPLORER_JS


def test_plotly_div_gives_the_x_axis_automargin_and_keeps_its_own_keys() -> None:
    """plotly.js ignores the template name, so the layout itself must let the
    x-axis labels clear the bottom legend."""
    html = plotly_div([], {"xaxis": {"title": "Stage"}})

    assert '"xaxis":{"automargin":true,"title":{"text":"Stage"}}' in html
