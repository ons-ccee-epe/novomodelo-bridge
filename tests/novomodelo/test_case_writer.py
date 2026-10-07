"""Tests for the ``CaseWriter`` write funnel (``case_writer.py``)."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from cobre_bridge.cobre.case_writer import CaseWriter


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
