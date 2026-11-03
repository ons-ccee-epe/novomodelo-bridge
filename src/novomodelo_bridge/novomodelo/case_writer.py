"""Single write funnel for Novomodelo case-file output (JSON + Parquet)."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_LOG = logging.getLogger(__name__)

# float64 round-trips exactly at 17 significant digits.
MAX_FLOAT_DIGITS = 17
DEFAULT_FLOAT_DIGITS = 15


def round_floats(value: object, digits: int) -> object:
    """Return a copy of *value* with each finite nonzero float rounded to
    *digits* significant digits."""
    if isinstance(value, float):
        if value == 0.0 or not math.isfinite(value):
            return value
        return float(f"{value:.{digits}g}")
    if isinstance(value, Mapping):
        return {key: round_floats(item, digits) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [round_floats(item, digits) for item in value]
    return value


class CaseWriter:
    """Writes case files under ``dst``, tracking ``dry_run``/``would_write``.

    JSON is written with a trailing newline (``indent=2, ensure_ascii=False``)
    so both conversion tracks emit one byte format instead of two.
    """

    def __init__(
        self,
        dst: Path,
        *,
        dry_run: bool = False,
        float_digits: int | None = DEFAULT_FLOAT_DIGITS,
    ) -> None:
        if float_digits is not None and not 1 <= float_digits <= MAX_FLOAT_DIGITS:
            raise ValueError(
                f"float_digits must be between 1 and {MAX_FLOAT_DIGITS}, "
                f"got {float_digits}"
            )
        self.dst = dst
        self.dry_run = dry_run
        self.float_digits = float_digits
        self.would_write: list[Path] = []

    def write_json(self, relpath: str, data: Mapping[str, object]) -> None:
        path = self.dst / relpath
        self.would_write.append(path)
        if self.dry_run:
            _LOG.debug("would write %s", path)
            return
        payload = (
            data if self.float_digits is None else round_floats(data, self.float_digits)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    def write_parquet(self, relpath: str, table: pa.Table) -> None:
        path = self.dst / relpath
        self.would_write.append(path)
        if self.dry_run:
            _LOG.debug("would write %s", path)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        # TRACKED NOVOMODELO-GAP WORKAROUND (C3): the solver's parquet reader is
        # built without snappy; zstd until the compression contract is settled.
        pq.write_table(table, path, compression="zstd")
