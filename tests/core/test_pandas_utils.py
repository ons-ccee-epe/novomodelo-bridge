"""Tests for the shared pandas helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from novomodelo_bridge.core.pandas_utils import is_na


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
