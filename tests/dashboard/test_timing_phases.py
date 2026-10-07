"""Unit tests for the shared top-level timing-phase module.

Tier-1: pure Python + pandas/plotly, imports no novomodelo and reads no example/
deck, so it collects and runs even in a novomodelo-free environment.
"""

from __future__ import annotations

import re

import pandas as pd

from novomodelo_bridge.dashboard.tabs import performance_charts, training
from novomodelo_bridge.dashboard.tabs.timing_phases import (
    TOP_LEVEL_PHASE_COLUMNS,
    TOP_LEVEL_PHASE_CONFIG,
    active_top_level_phases,
    build_timing_stacked_figure,
)
from novomodelo_bridge.ui.html.plotly import fig_to_html
from novomodelo_bridge.ui.theme import PERFORMANCE_PHASE_COLORS

_EXPECTED_CONFIG: tuple[tuple[str, str, str], ...] = (
    ("forward_wall_ms", "Forward", PERFORMANCE_PHASE_COLORS["forward"]),
    ("backward_wall_ms", "Backward", PERFORMANCE_PHASE_COLORS["backward"]),
    ("cut_selection_ms", "Cut Selection", PERFORMANCE_PHASE_COLORS["lp_solve"]),
    ("lower_bound_ms", "Lower Bound Eval", "#14B8A6"),
    ("mpi_allreduce_ms", "MPI AllReduce", "#8B5CF6"),
    ("overhead_ms", "Other Overhead", PERFORMANCE_PHASE_COLORS["overhead"]),
)


def _make_timing(columns: list[str], n: int = 3) -> pd.DataFrame:
    """Build a timing frame with ``iteration`` plus the requested phase columns."""
    data: dict[str, list[float]] = {"iteration": list(range(1, n + 1))}
    for i, col in enumerate(columns):
        data[col] = [100.0 * (i + 1) + j for j in range(n)]
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# TOP_LEVEL_PHASE_CONFIG / TOP_LEVEL_PHASE_COLUMNS
# ---------------------------------------------------------------------------


def test_top_level_phase_config_matches_expected_triples() -> None:
    assert TOP_LEVEL_PHASE_CONFIG == _EXPECTED_CONFIG


def test_top_level_phase_columns_is_derived_tuple_in_order() -> None:
    assert TOP_LEVEL_PHASE_COLUMNS == tuple(c for c, _, _ in _EXPECTED_CONFIG)


# ---------------------------------------------------------------------------
# active_top_level_phases
# ---------------------------------------------------------------------------


def test_active_top_level_phases_returns_present_subset_in_config_order() -> None:
    # Columns supplied out of config order to prove the detector re-sorts them.
    timing = _make_timing(["overhead_ms", "forward_wall_ms"])
    assert active_top_level_phases(timing) == [
        ("forward_wall_ms", "Forward", PERFORMANCE_PHASE_COLORS["forward"]),
        ("overhead_ms", "Other Overhead", PERFORMANCE_PHASE_COLORS["overhead"]),
    ]


def test_active_top_level_phases_empty_frame_returns_empty_list() -> None:
    assert active_top_level_phases(pd.DataFrame()) == []


def test_active_top_level_phases_no_recognised_columns_returns_empty_list() -> None:
    timing = pd.DataFrame({"iteration": [1, 2], "unrelated_ms": [1.0, 2.0]})
    assert active_top_level_phases(timing) == []


# ---------------------------------------------------------------------------
# build_timing_stacked_figure
# ---------------------------------------------------------------------------


def test_build_timing_stacked_figure_two_phase_traces() -> None:
    timing = _make_timing(["forward_wall_ms", "backward_wall_ms"])
    fig = build_timing_stacked_figure(timing)
    assert fig is not None
    assert len(fig.data) == 2

    by_name = {trace.name: trace for trace in fig.data}
    assert set(by_name) == {"Forward", "Backward"}

    fwd = by_name["Forward"]
    assert fwd.marker.color == PERFORMANCE_PHASE_COLORS["forward"]
    assert list(fwd.x) == [1, 2, 3]
    assert list(fwd.y) == list(timing["forward_wall_ms"])

    bwd = by_name["Backward"]
    assert bwd.marker.color == PERFORMANCE_PHASE_COLORS["backward"]
    assert list(bwd.x) == [1, 2, 3]
    assert list(bwd.y) == list(timing["backward_wall_ms"])


def test_build_timing_stacked_figure_empty_frame_returns_none() -> None:
    assert build_timing_stacked_figure(pd.DataFrame()) is None


def test_build_timing_stacked_figure_no_recognised_columns_returns_none() -> None:
    timing = pd.DataFrame({"iteration": [1, 2], "unrelated_ms": [1.0, 2.0]})
    assert build_timing_stacked_figure(timing) is None


def test_build_timing_stacked_figure_no_iteration_column_falls_back_to_range() -> None:
    # The no-`iteration`-column path: untested by either pre-dedup tab, adopted
    # from training.py as the shared superset (see the module docstring).
    timing = pd.DataFrame({"forward_wall_ms": [1.0, 2.0, 3.0]})
    fig = build_timing_stacked_figure(timing)
    assert fig is not None
    assert list(fig.data[0].x) == [0, 1, 2]


# ---------------------------------------------------------------------------
# Rendering-stability pin
# ---------------------------------------------------------------------------
#
# These two constants are the normalized HTML strings rendered by
# performance_charts.chart_iteration_timing_breakdown and
# training._chart_timing_stacked (then fig_to_html) on _timing_fixture(),
# on the current intended output: apply_standard_layout/render_figure default
# to the "plotly_white" template (DASH-08), so both baselines carry that
# template's resolved layout, not the earlier grey "plotly" default. The pin
# proves neither tab's rendered output drifts going forward; the dedup that
# moved TOP_LEVEL_PHASE_COLUMNS/_TOP_LEVEL_PHASE_LABELS/_detect_top_level_columns
# (performance_charts) and _TOP_LEVEL_PHASE_CONFIG/_active_top_level_phases
# (training) into timing_phases is independently verified above by the trace
# count/name/marker_color/x/y assertions, which are template-independent.

_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _normalize_div_id(html: str) -> str:
    """Replace Plotly's per-render random div-id UUID with a fixed placeholder.

    ``go.Figure.to_html`` mints a fresh ``uuid.uuid4()`` on every call, so two
    renders of an otherwise-identical figure are never byte-equal without
    this — isolating that one nondeterministic substring is what makes a
    byte-equality pin on Plotly HTML meaningful at all.
    """
    return _UUID_RE.sub("<uuid>", html)


def _timing_fixture() -> pd.DataFrame:
    """Frame shared by both tabs' pin: 4 of the 6 top-level phases present."""
    return pd.DataFrame(
        {
            "iteration": [1, 2, 3],
            "forward_wall_ms": [100.0, 110.0, 120.0],
            "backward_wall_ms": [200.0, 210.0, 220.0],
            "lower_bound_ms": [5.0, 6.0, 7.0],
            "overhead_ms": [10.0, 11.0, 12.0],
        }
    )


_PERFORMANCE_BASELINE_HTML = r"""<div style="height:420px; width:100%;">                            <div id="<uuid>" class="plotly-graph-div" style="height:100%; width:100%;"></div>            <script>                window.PLOTLYENV=window.PLOTLYENV || {};                                if (document.getElementById("<uuid>")) {                    Plotly.newPlot(                        "<uuid>",                        [{"hovertemplate":"Forward: %{y:.0f} ms\u003cextra\u003e\u003c\u002fextra\u003e","marker":{"color":"#3B82F6"},"name":"Forward","x":[1,2,3],"y":[100.0,110.0,120.0],"type":"bar"},{"hovertemplate":"Backward: %{y:.0f} ms\u003cextra\u003e\u003c\u002fextra\u003e","marker":{"color":"#14B8A6"},"name":"Backward","x":[1,2,3],"y":[200.0,210.0,220.0],"type":"bar"},{"hovertemplate":"Lower Bound Eval: %{y:.0f} ms\u003cextra\u003e\u003c\u002fextra\u003e","marker":{"color":"#14B8A6"},"name":"Lower Bound Eval","x":[1,2,3],"y":[5.0,6.0,7.0],"type":"bar"},{"hovertemplate":"Other Overhead: %{y:.0f} ms\u003cextra\u003e\u003c\u002fextra\u003e","marker":{"color":"#6B7280"},"name":"Other Overhead","x":[1,2,3],"y":[10.0,11.0,12.0],"type":"bar"}],                        {"template":{"data":{"barpolar":[{"marker":{"line":{"color":"white","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"white","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"#C8D4E3","linecolor":"#C8D4E3","minorgridcolor":"#C8D4E3","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"#C8D4E3","linecolor":"#C8D4E3","minorgridcolor":"#C8D4E3","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermap":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermap"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"white","showlakes":true,"showland":true,"subunitcolor":"#C8D4E3"},"hoverlabel":{"align":"left"},"hovermode":"closest","paper_bgcolor":"white","plot_bgcolor":"white","polar":{"angularaxis":{"gridcolor":"#EBF0F8","linecolor":"#EBF0F8","ticks":""},"bgcolor":"white","radialaxis":{"gridcolor":"#EBF0F8","linecolor":"#EBF0F8","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"white","gridcolor":"#DFE8F3","gridwidth":2,"linecolor":"#EBF0F8","showbackground":true,"ticks":"","zerolinecolor":"#EBF0F8"},"yaxis":{"backgroundcolor":"white","gridcolor":"#DFE8F3","gridwidth":2,"linecolor":"#EBF0F8","showbackground":true,"ticks":"","zerolinecolor":"#EBF0F8"},"zaxis":{"backgroundcolor":"white","gridcolor":"#DFE8F3","gridwidth":2,"linecolor":"#EBF0F8","showbackground":true,"ticks":"","zerolinecolor":"#EBF0F8"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"#DFE8F3","linecolor":"#A2B1C6","ticks":""},"baxis":{"gridcolor":"#DFE8F3","linecolor":"#A2B1C6","ticks":""},"bgcolor":"white","caxis":{"gridcolor":"#DFE8F3","linecolor":"#A2B1C6","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"#EBF0F8","linecolor":"#EBF0F8","ticks":"","title":{"standoff":15},"zerolinecolor":"#EBF0F8","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"#EBF0F8","linecolor":"#EBF0F8","ticks":"","title":{"standoff":15},"zerolinecolor":"#EBF0F8","zerolinewidth":2}}},"legend":{"font":{"size":11},"orientation":"h","yanchor":"bottom","y":1.02,"xanchor":"center","x":0.5},"margin":{"l":60,"r":30,"t":60,"b":50},"title":{"text":"Iteration Timing \u2014 Top-Level Phases (non-overlapping, sums to total)"},"xaxis":{"title":{"text":"Iteration"}},"yaxis":{"title":{"text":"Time (ms)"}},"barmode":"stack","height":420,"hovermode":"x unified","autosize":true},                        {"responsive": true}                    )                };            </script>        </div>"""

_TRAINING_BASELINE_HTML = r"""<div style="height:100%; width:100%;">                            <div id="<uuid>" class="plotly-graph-div" style="height:100%; width:100%;"></div>            <script>                window.PLOTLYENV=window.PLOTLYENV || {};                                if (document.getElementById("<uuid>")) {                    Plotly.newPlot(                        "<uuid>",                        [{"hovertemplate":"Forward: %{y:.0f} ms\u003cextra\u003e\u003c\u002fextra\u003e","marker":{"color":"#3B82F6"},"name":"Forward","x":[1,2,3],"y":[100.0,110.0,120.0],"type":"bar"},{"hovertemplate":"Backward: %{y:.0f} ms\u003cextra\u003e\u003c\u002fextra\u003e","marker":{"color":"#14B8A6"},"name":"Backward","x":[1,2,3],"y":[200.0,210.0,220.0],"type":"bar"},{"hovertemplate":"Lower Bound Eval: %{y:.0f} ms\u003cextra\u003e\u003c\u002fextra\u003e","marker":{"color":"#14B8A6"},"name":"Lower Bound Eval","x":[1,2,3],"y":[5.0,6.0,7.0],"type":"bar"},{"hovertemplate":"Other Overhead: %{y:.0f} ms\u003cextra\u003e\u003c\u002fextra\u003e","marker":{"color":"#6B7280"},"name":"Other Overhead","x":[1,2,3],"y":[10.0,11.0,12.0],"type":"bar"}],                        {"template":{"data":{"barpolar":[{"marker":{"line":{"color":"white","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"white","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"#C8D4E3","linecolor":"#C8D4E3","minorgridcolor":"#C8D4E3","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"#C8D4E3","linecolor":"#C8D4E3","minorgridcolor":"#C8D4E3","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermap":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermap"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"white","showlakes":true,"showland":true,"subunitcolor":"#C8D4E3"},"hoverlabel":{"align":"left"},"hovermode":"closest","paper_bgcolor":"white","plot_bgcolor":"white","polar":{"angularaxis":{"gridcolor":"#EBF0F8","linecolor":"#EBF0F8","ticks":""},"bgcolor":"white","radialaxis":{"gridcolor":"#EBF0F8","linecolor":"#EBF0F8","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"white","gridcolor":"#DFE8F3","gridwidth":2,"linecolor":"#EBF0F8","showbackground":true,"ticks":"","zerolinecolor":"#EBF0F8"},"yaxis":{"backgroundcolor":"white","gridcolor":"#DFE8F3","gridwidth":2,"linecolor":"#EBF0F8","showbackground":true,"ticks":"","zerolinecolor":"#EBF0F8"},"zaxis":{"backgroundcolor":"white","gridcolor":"#DFE8F3","gridwidth":2,"linecolor":"#EBF0F8","showbackground":true,"ticks":"","zerolinecolor":"#EBF0F8"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"#DFE8F3","linecolor":"#A2B1C6","ticks":""},"baxis":{"gridcolor":"#DFE8F3","linecolor":"#A2B1C6","ticks":""},"bgcolor":"white","caxis":{"gridcolor":"#DFE8F3","linecolor":"#A2B1C6","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"#EBF0F8","linecolor":"#EBF0F8","ticks":"","title":{"standoff":15},"zerolinecolor":"#EBF0F8","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"#EBF0F8","linecolor":"#EBF0F8","ticks":"","title":{"standoff":15},"zerolinecolor":"#EBF0F8","zerolinewidth":2}}},"legend":{"font":{"size":11},"orientation":"h","yanchor":"bottom","y":1.02,"xanchor":"center","x":0.5},"margin":{"l":60,"r":30,"t":60,"b":50},"barmode":"stack","xaxis":{"title":{"text":"Iteration"}},"yaxis":{"title":{"text":"Time (ms)"}},"hovermode":"x unified","autosize":true},                        {"responsive": true}                    )                };            </script>        </div>"""


def test_chart_iteration_timing_breakdown_matches_rendering_pin() -> None:
    html = performance_charts.chart_iteration_timing_breakdown(_timing_fixture())
    assert _normalize_div_id(html) == _PERFORMANCE_BASELINE_HTML
    # plotly_white (not the grey "plotly" default), per the DASH-08 template default.
    assert '"plot_bgcolor":"white"' in html


def test_training_chart_timing_stacked_matches_rendering_pin() -> None:
    fig = training._chart_timing_stacked(_timing_fixture())
    assert fig is not None
    html = fig_to_html(fig)
    assert _normalize_div_id(html) == _TRAINING_BASELINE_HTML
    assert '"plot_bgcolor":"white"' in html
