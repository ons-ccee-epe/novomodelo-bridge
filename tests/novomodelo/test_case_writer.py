"""Tests for the ``CaseWriter`` write funnel (``case_writer.py``)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from novomodelo_bridge.novomodelo.case_writer import CaseWriter, round_floats


def test_write_json_appends_trailing_newline(tmp_path: Path) -> None:
    CaseWriter(tmp_path).write_json("config.json", {"a": 1})

    written = (tmp_path / "config.json").read_text(encoding="utf-8")
    assert written == json.dumps({"a": 1}, indent=2, ensure_ascii=False) + "\n"


def test_write_json_does_not_escape_non_ascii(tmp_path: Path) -> None:
    CaseWriter(tmp_path).write_json("config.json", {"nome": "cenário"})

    written = (tmp_path / "config.json").read_text(encoding="utf-8")
    assert "cenário" in written
    assert "\\u" not in written


def test_dry_run_writes_nothing_but_records_would_write(tmp_path: Path) -> None:
    writer = CaseWriter(tmp_path, dry_run=True)
    table = pa.table({"x": [1, 2, 3]})

    writer.write_json("system/hydros.json", {"hydros": []})
    writer.write_parquet("scenarios/x.parquet", table)

    assert not (tmp_path / "system").exists()
    assert not (tmp_path / "scenarios").exists()
    assert writer.would_write == [
        tmp_path / "system" / "hydros.json",
        tmp_path / "scenarios" / "x.parquet",
    ]


def test_write_parquet_uses_zstd_and_round_trips(tmp_path: Path) -> None:
    table = pa.table({"x": [1, 2, 3]})

    CaseWriter(tmp_path).write_parquet("constraints/b.parquet", table)

    path = tmp_path / "constraints" / "b.parquet"
    parquet_file = pq.ParquetFile(path)
    compression = parquet_file.metadata.row_group(0).column(0).compression
    assert compression == "ZSTD"
    assert parquet_file.read().equals(table)


def test_write_json_creates_parent_directory(tmp_path: Path) -> None:
    CaseWriter(tmp_path).write_json("system/hydros.json", {})

    assert (tmp_path / "system").is_dir()
    assert (tmp_path / "system" / "hydros.json").is_file()


def test_write_json_rounds_to_15_significant_digits_by_default(tmp_path: Path) -> None:
    CaseWriter(tmp_path).write_json("config.json", {"x": 208.19733988000294})

    written = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert written["x"] == 208.197339880003


def test_write_json_keeps_full_float_precision_when_digits_is_none(
    tmp_path: Path,
) -> None:
    CaseWriter(tmp_path, float_digits=None).write_json(
        "config.json", {"x": 208.19733988000294}
    )

    written = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert written["x"] == 208.19733988000294


def test_write_json_rounds_nested_floats_to_significant_digits(tmp_path: Path) -> None:
    data = {
        "a": 0.0949999988079071,
        "b": [208.19733988000294, {"c": 147327107.29284817}],
    }

    CaseWriter(tmp_path, float_digits=7).write_json("config.json", data)

    written = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert written == {"a": 0.095, "b": [208.1973, {"c": 147327100.0}]}


def test_write_json_rounding_does_not_mutate_the_caller_dict(tmp_path: Path) -> None:
    data = {"a": [0.0949999988079071]}

    CaseWriter(tmp_path, float_digits=7).write_json("config.json", data)

    assert data == {"a": [0.0949999988079071]}


@pytest.mark.parametrize("digits", [0, 18])
def test_case_writer_rejects_float_digits_out_of_range(
    tmp_path: Path, digits: int
) -> None:
    with pytest.raises(ValueError, match="float_digits"):
        CaseWriter(tmp_path, float_digits=digits)


def test_round_floats_keeps_relative_precision_of_small_values() -> None:
    assert round_floats(0.00018622117020507253, 4) == 0.0001862


def test_round_floats_leaves_non_float_values_untouched() -> None:
    data = {"i": 3, "t": True, "n": None, "s": "1.23456789", "z": 0.0}

    assert round_floats(data, 3) == data


def test_round_floats_passes_non_finite_values_through() -> None:
    rounded = round_floats([math.inf, -math.inf, math.nan], 3)

    assert rounded[0] == math.inf
    assert rounded[1] == -math.inf
    assert math.isnan(rounded[2])
