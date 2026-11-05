"""The post-study water calendar the DECOMP conversion emits for a travel-time deck.

The helper tests need no ``novomodelo-python``; the end-to-end test validates the
converted ``decomp_mini`` deck through the real ``novomodelo.io.validate``.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from novomodelo_bridge.decomp.pipeline import _water_bucket_stages, convert_decomp_case
from tests.conftest import requires_novomodelo_python


def test_first_of_month_horizon_gives_the_whole_month() -> None:
    assert _water_bucket_stages(date(2026, 5, 1), {66: 360.0}) == [
        {"start_date": "2026-05-01", "duration_hours": 744.0}
    ]


def test_mid_month_horizon_gives_the_days_left_including_its_own() -> None:
    assert _water_bucket_stages(date(2026, 2, 14), {66: 360.0}) == [
        {"start_date": "2026-02-14", "duration_hours": 360.0}
    ]


def test_deck_without_travel_time_gets_no_water_stage() -> None:
    assert _water_bucket_stages(date(2026, 5, 1), {}) is None


def test_travel_time_longer_than_the_stage_names_the_plant() -> None:
    with pytest.raises(ValueError, match="plant 66 .*400 h.*360 h"):
        _water_bucket_stages(date(2026, 2, 14), {18: 120.0, 66: 400.0})


@requires_novomodelo_python
def test_decomp_mini_deck_water_only_post_study_file_validates(
    decomp_mini_deck: Path, tmp_path: Path
) -> None:
    import novomodelo.io  # tier-2: guarded import inside the body, never module scope

    dst = tmp_path / "converted"
    convert_decomp_case(decomp_mini_deck, dst, force=True)

    horizon_end = json.loads((dst / "stages.json").read_text())["stages"][-1][
        "end_date"
    ]
    post_study = json.loads((dst / "post_study_stages.json").read_text())
    assert set(post_study) == {"water_bucket_stages"}
    assert [stage["start_date"] for stage in post_study["water_bucket_stages"]] == [
        horizon_end
    ]

    result = novomodelo.io.validate(str(dst))
    assert result["valid"] is True
    assert result["errors"] == []
    assert not [w for w in result["warnings"] if "is deprecated" in w["message"]]
