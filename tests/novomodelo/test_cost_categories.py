"""Drift guards for the canonical Novomodelo cost taxonomy.

The dashboard cost charts (``COST_GROUPS``), the comparator cost breakdown
(``_COST_MAP`` + ``read_novomodelo_cost_breakdown``) and the theme colours must
all classify and name exactly the same Novomodelo cost-component columns. These
tests fail if any of them drifts from ``COST_CATEGORIES``, which is how the
comparator once dropped ``contract_cost`` and the dashboard once lumped
``excess_cost`` into "Other".
"""

from __future__ import annotations

import math

import pytest

from novomodelo_bridge.novomodelo.cost_categories import (
    AGGREGATE_COST_COLUMNS,
    COST_CATEGORIES,
    COST_COMPONENT_BY_LABEL,
    COST_KEY_COLUMNS,
    NOVOMODELO_COST_COMPONENT_COLUMNS,
)
from novomodelo_bridge.ui.theme import COST_CATEGORY_COLORS, COST_COMPONENT_COLORS


def test_component_and_aggregate_columns_are_disjoint() -> None:
    assert not (set(NOVOMODELO_COST_COMPONENT_COLUMNS) & AGGREGATE_COST_COLUMNS)
    assert not (set(NOVOMODELO_COST_COMPONENT_COLUMNS) & COST_KEY_COLUMNS)
    assert len(NOVOMODELO_COST_COMPONENT_COLUMNS) == len(
        set(NOVOMODELO_COST_COMPONENT_COLUMNS)
    )


def test_labels_are_unique_across_categories_and_components() -> None:
    """A label names one category or one component, never two things."""
    categories = [cat.label for cat in COST_CATEGORIES]
    components = [comp.label for cat in COST_CATEGORIES for comp in cat.components]

    assert len(set(categories)) == len(categories)
    assert len(set(components)) == len(components)
    assert len(COST_COMPONENT_BY_LABEL) == len(components)


def test_theme_colours_cover_exactly_the_taxonomy() -> None:
    """Every category and component has a colour, and no colour is orphaned."""
    assert set(COST_CATEGORY_COLORS) == {cat.label for cat in COST_CATEGORIES}
    assert set(COST_COMPONENT_COLORS) == set(COST_COMPONENT_BY_LABEL)


def _oklch(hex_color: str) -> tuple[float, float, float]:
    """OKLCH lightness, chroma and hue (degrees) of an sRGB hex colour."""

    def linear(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (linear(int(hex_color[i : i + 2], 16) / 255) for i in (1, 3, 5))
    lms = (
        0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b,
        0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b,
        0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b,
    )
    l_, m_, s_ = (v ** (1 / 3) for v in lms)
    lightness = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b_ = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return lightness, math.hypot(a, b_), math.degrees(math.atan2(b_, a)) % 360


@pytest.mark.parametrize("category", COST_CATEGORIES, ids=lambda c: c.label)
def test_component_colours_are_a_light_to_dark_ramp_of_their_category(
    category,
) -> None:
    """A stack of components reads as its category: one hue, each step
    visibly darker than the one before it, in taxonomy order."""
    _, _, hue = _oklch(COST_CATEGORY_COLORS[category.label])
    shades = [_oklch(COST_COMPONENT_COLORS[c.label]) for c in category.components]

    for _, _, h in shades:
        assert abs((h - hue + 180) % 360 - 180) <= 10
    for (l_lighter, _, _), (l_darker, _, _) in zip(shades, shades[1:]):
        assert l_lighter - l_darker >= 0.06


def test_dashboard_cost_groups_cover_exactly_the_canonical_columns() -> None:
    from novomodelo_bridge.dashboard.chart_helpers import COST_GROUPS

    for level, groups in COST_GROUPS.items():
        mapped = [col for cols in groups.values() for col in cols]
        assert len(mapped) == len(set(mapped)), f"{level} maps a column twice"
        assert set(mapped) == set(NOVOMODELO_COST_COMPONENT_COLUMNS), (
            f"dashboard COST_GROUPS[{level!r}] drifted from the canonical "
            f"cost-component set: "
            f"missing={set(NOVOMODELO_COST_COMPONENT_COLUMNS) - set(mapped)}, "
            f"extra={set(mapped) - set(NOVOMODELO_COST_COMPONENT_COLUMNS)}"
        )


def test_dashboard_non_cost_cols_use_shared_sets() -> None:
    from novomodelo_bridge.dashboard.chart_helpers import _NON_COST_COLS

    assert _NON_COST_COLS == COST_KEY_COLUMNS | AGGREGATE_COST_COLUMNS


def test_comparator_cost_map_covers_exactly_the_canonical_columns() -> None:
    from novomodelo_bridge.comparators.charts import _COST_MAP

    mapped = {col for _, _, novomodelo_cols, _ in _COST_MAP for col in novomodelo_cols}
    assert mapped == set(NOVOMODELO_COST_COMPONENT_COLUMNS), (
        "comparator _COST_MAP drifted from the canonical cost-component set: "
        f"missing={set(NOVOMODELO_COST_COMPONENT_COLUMNS) - mapped}, "
        f"extra={mapped - set(NOVOMODELO_COST_COMPONENT_COLUMNS)}"
    )


def test_comparator_cost_map_uses_the_dashboard_names_and_colours() -> None:
    """The compare breakdown names and colours each Novomodelo component as
    the dashboard does."""
    from novomodelo_bridge.comparators.charts import _COST_MAP

    novomodelo_rows = {
        label: (tuple(cols), color) for label, _, cols, color in _COST_MAP if cols
    }
    assert novomodelo_rows == {
        label: (comp.columns, COST_COMPONENT_COLORS[label])
        for label, comp in COST_COMPONENT_BY_LABEL.items()
    }


def test_comparator_exclusion_set_is_the_shared_aggregate_set() -> None:
    from novomodelo_bridge.comparators.charts import _NOVOMODELO_NON_COST_KEYS

    assert _NOVOMODELO_NON_COST_KEYS == AGGREGATE_COST_COLUMNS
