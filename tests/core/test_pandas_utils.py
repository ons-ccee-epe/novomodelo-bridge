"""Tests for the shared pandas helpers."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from novomodelo_bridge.core.pandas_utils import (
    as_float,
    as_frame,
    as_frame_or_empty,
    as_int,
    as_timestamp,
    is_na,
)


@pytest.mark.parametrize(
    "value",
    [float("nan"), np.nan, pd.NA, pd.NaT, None],
)
def test_is_na_true_for_na_sentinels(value: object) -> None:
    assert is_na(value) is True


@pytest.mark.parametrize(
    "value",
    [0.0, 1, "", "FICT.", -5.0, [], np.float64(3.0)],
)
def test_is_na_false_for_real_values(value: object) -> None:
    assert is_na(value) is False


def test_is_na_does_not_raise_on_array() -> None:
    # pd.isna(array) returns an array; is_na must collapse to a single bool.
    assert is_na(np.array([1.0, np.nan])) is False


def test_as_frame_passes_a_frame_through() -> None:
    frame = pd.DataFrame({"a": [1]})
    assert as_frame(frame) is frame


@pytest.mark.parametrize("value", [None, [1], "records"])
def test_as_frame_is_none_for_anything_else(value: object) -> None:
    assert as_frame(value) is None


def test_as_frame_or_empty_passes_a_frame_through_and_defaults_to_empty() -> None:
    frame = pd.DataFrame({"a": [1]})
    assert as_frame_or_empty(frame) is frame
    assert as_frame_or_empty(None).empty


def test_as_int_and_as_float_convert_cells() -> None:
    assert as_int(np.int64(3)) == 3
    assert as_int("7") == 7
    assert as_float(np.float32(1.5)) == 1.5
    assert as_float("2.5") == 2.5


def test_as_int_raises_like_int_on_a_missing_cell() -> None:
    with pytest.raises(TypeError):
        as_int(None)
    with pytest.raises(ValueError):
        as_int(float("nan"))


def test_as_timestamp_converts_datetime_cells() -> None:
    assert as_timestamp(np.datetime64("2026-01-05")) == pd.Timestamp("2026-01-05")
    assert as_timestamp(date(2026, 1, 5)).date() == date(2026, 1, 5)


def test_as_timestamp_rejects_nat() -> None:
    with pytest.raises(TypeError):
        as_timestamp(pd.NaT)
