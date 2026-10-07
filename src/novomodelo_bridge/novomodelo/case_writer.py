"""Single write funnel for Cobre case-file output (JSON + Parquet)."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_LOG = logging.getLogger(__name__)


class CaseWriter:
    """Writes case files under ``dst``, tracking ``dry_run``/``would_write``.

    JSON is written with a trailing newline (``indent=2, ensure_ascii=False``)
    so both conversion tracks emit one byte format instead of two.
    """

    def __init__(self, dst: Path, *, dry_run: bool = False) -> None:
        self.dst = dst
        self.dry_run = dry_run
        self.would_write: list[Path] = []

    def write_json(self, relpath: str, data: Mapping[str, object]) -> None:
        path = self.dst / relpath
        self.would_write.append(path)
        if self.dry_run:
            _LOG.debug("would write %s", path)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    def write_parquet(self, relpath: str, table: pa.Table) -> None:
        path = self.dst / relpath
        self.would_write.append(path)
        if self.dry_run:
            _LOG.debug("would write %s", path)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        # TRACKED COBRE-GAP WORKAROUND (C3): the solver's parquet reader is
        # built without snappy; zstd until the compression contract is settled.
        pq.write_table(table, path, compression="zstd")
