"""Tab registry for the dashboard."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from cobre_bridge.dashboard.data import DashboardData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class TabModule(Protocol):
    """Structural interface that every dashboard tab module must satisfy."""

    TAB_ID: str
    TAB_LABEL: str
    TAB_ORDER: int
    REQUIRED_JS: list[str]

    def can_render(self, data: DashboardData) -> bool:
        """Return True when this tab has sufficient data to be shown."""
        ...

    def render(self, data: DashboardData) -> str:
        """Return the full HTML string for the tab content area."""
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from cobre_bridge.dashboard.tabs import (  # noqa: E402
    constraints,
    costs,
    energy_balance,
    network,
    overview,
    performance,
    plants,
    stochastic,
    training,
)

TAB_MODULES: list[TabModule] = [
    overview,
    training,
    stochastic,
    energy_balance,
    costs,
    plants,
    network,
    constraints,
    performance,
]

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _render_error_placeholder(tab_label: str, exc: Exception) -> str:
    """HTML shown in place of a tab whose ``render()`` raised.

    Keeps the tab visible (with an escaped error message) instead of silently
    dropping it, so a single failing tab can't make the dashboard look as if
    that section simply does not exist.
    """
    import html as _html

    return (
        '<div style="padding:24px;color:#8a1f1f;background:#fdecec;'
        'border:1px solid #f3b5b5;border-radius:6px;">'
        f"<strong>{_html.escape(tab_label)} failed to render.</strong>"
        '<pre style="white-space:pre-wrap;margin-top:8px;">'
        f"{_html.escape(str(exc))}</pre>"
        "</div>"
    )


def get_renderable_tabs(data: DashboardData) -> list[tuple[str, str, str]]:
    """Return ordered, renderable tabs for *data*.

    Returns:
        List of ``(tab_id, tab_label, rendered_html)`` tuples.
    """
    result: list[tuple[str, str, str]] = []
    ordered = sorted(TAB_MODULES, key=lambda m: m.TAB_ORDER)
    for module in ordered:
        if not module.can_render(data):
            continue
        try:
            html = module.render(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Tab %r raised an exception during render; showing an error "
                "placeholder instead of dropping it: %s",
                module.TAB_ID,
                exc,
            )
            html = _render_error_placeholder(module.TAB_LABEL, exc)
        result.append((module.TAB_ID, module.TAB_LABEL, html))
    return result


def collect_required_js(data: DashboardData) -> str:
    """Union the ``REQUIRED_JS`` blocks of every renderable tab for *data*.

    Each tab declares the shared JS function definitions it needs instead of
    relying on another tab having emitted them — a tab's shared JS dependency
    stays satisfied regardless of which other tabs render for this case shape
    (see ``build_html``'s ``required_js`` parameter). Modules that declare
    nothing are tolerated via ``getattr(..., [])``.

    Returns:
        The deduped blocks (first-seen order, deduped by string value) joined
        with ``"\\n"``.
    """
    ordered = sorted(TAB_MODULES, key=lambda m: m.TAB_ORDER)
    blocks: list[str] = []
    for module in ordered:
        if not module.can_render(data):
            continue
        for block in getattr(module, "REQUIRED_JS", []):
            if block not in blocks:
                blocks.append(block)
    return "\n".join(blocks)
