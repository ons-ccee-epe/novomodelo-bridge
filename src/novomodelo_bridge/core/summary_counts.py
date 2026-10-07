"""Results-footer count extraction shared by the compare report and summary."""

from __future__ import annotations

from collections.abc import Mapping


def footer_counts(metadata: Mapping[str, object]) -> tuple[int, dict[str, int]]:
    """Return the results footer ``(total, by_entity_type)`` from metadata.

    Reads ``metadata["footer_counts"]``; missing/ill-typed metadata yields
    ``(0, {})``.
    """
    raw = metadata.get("footer_counts")
    if not isinstance(raw, dict):
        return 0, {}
    total = raw.get("total", 0)
    return (
        int(total) if isinstance(total, int) else 0,
        _as_int_counts(raw.get("by_entity_type")),
    )


def _as_int_counts(value: object) -> dict[str, int]:
    """Coerce a metadata mapping into a ``dict[str, int]`` (empty on mismatch)."""
    if not isinstance(value, dict):
        return {}
    return {
        key: count
        for key, count in value.items()
        if isinstance(key, str) and isinstance(count, int)
    }
