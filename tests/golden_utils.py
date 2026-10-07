"""Shared golden-file helpers for the ``tests/golden/`` snapshots.

Regenerate goldens via ``scripts/regen-goldens.sh``
(``NOVOMODELO_BRIDGE_UPDATE_GOLDENS=1``) -- never hand-edit a file under
``tests/golden/``.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

_GOLDEN_DIR = Path(__file__).parent / "golden"


def update_goldens() -> bool:
    """Return True when goldens should be (re)written rather than checked."""
    return os.environ.get("NOVOMODELO_BRIDGE_UPDATE_GOLDENS") == "1"


def _strip_chart_id(html: str) -> str:
    """Normalise the random ``chart-<hex>`` div id and the plotly.js CDN version."""
    html = re.sub(r"chart-[0-9a-f]+", "chart-XXXX", html)
    return re.sub(r"plotly-[0-9.]+\.min\.js", "plotly-XXXX.min.js", html)


def assert_html_golden(actual: str, name: str) -> None:
    """Snapshot ``actual`` HTML against the golden file ``name``.

    Comparison is normalised through :func:`_strip_chart_id`, since
    ``plotly_div`` mints a fresh ``uuid4`` div id on every render. On update,
    the golden is rewritten only when its normalised content actually
    changed -- an unconditional overwrite would churn the committed file's
    random id on every regen even with no real change. Never silently
    overwritten on a mismatch when not updating.
    """
    path = _GOLDEN_DIR / name
    if update_goldens():
        if not path.exists() or _strip_chart_id(actual) != _strip_chart_id(
            path.read_text(encoding="utf-8")
        ):
            path.write_text(actual, encoding="utf-8")
        return
    golden = path.read_text(encoding="utf-8")
    assert _strip_chart_id(actual) == _strip_chart_id(golden)


def assert_frame_golden(actual: pl.DataFrame, name: str) -> None:
    """Snapshot ``actual`` against the golden file ``name``.

    ``pl.read_json`` infers wider/narrower dtypes than ``actual``'s schema, so
    the golden is cast to that schema before comparing.
    """
    path = _GOLDEN_DIR / name
    if update_goldens():
        actual.write_json(path)
        return
    golden = pl.read_json(path).cast(actual.schema)
    assert_frame_equal(actual, golden, check_exact=True, check_dtypes=True)


def assert_json_golden(records: list[dict[str, object]], name: str) -> None:
    """Snapshot ``records`` against the golden JSON file ``name``."""
    path = _GOLDEN_DIR / name
    if update_goldens():
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        return
    golden = json.loads(path.read_text(encoding="utf-8"))
    assert records == golden
