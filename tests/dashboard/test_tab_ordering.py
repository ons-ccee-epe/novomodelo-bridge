"""Tests for tab ordering matching wireframe specification.

The single source of truth for tab order is each tab module's ``TAB_ORDER``
constant — ``get_renderable_tabs`` sorts ``TAB_MODULES`` by it. These tests
assert on those constants directly (no parallel ordering table to drift).
"""

from __future__ import annotations

from novomodelo_bridge.dashboard.tabs import TAB_MODULES, stochastic, training


def test_stochastic_before_training() -> None:
    assert stochastic.TAB_ORDER < training.TAB_ORDER


def test_stochastic_tab_order_value() -> None:
    assert stochastic.TAB_ORDER == 10


def test_training_tab_order_value() -> None:
    assert training.TAB_ORDER == 20


def test_tab_orders_are_unique() -> None:
    """Distinct TAB_ORDER values keep the render sort deterministic."""
    orders = [m.TAB_ORDER for m in TAB_MODULES]
    assert len(orders) == len(set(orders))


def test_tab_ids_are_unique() -> None:
    ids = [m.TAB_ID for m in TAB_MODULES]
    assert len(ids) == len(set(ids))
