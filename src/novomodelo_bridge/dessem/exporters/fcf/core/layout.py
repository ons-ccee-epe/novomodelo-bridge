"""DECOMP binary layout constants and the invariants that check them.

Record map for mapcut.rv0:
| Records | Content | Count |
|---------|---------|-------|
| 0-3 | regs 1-4 | 4 |
| 4-17 | undocumented per-plant physical records | 14 |
| 18 | reg 5, parent-node indices | 1 |
| 19 | reg 6, stage data | 1 |
| 20.. | regs 7/8 travel time | `n_utv * n_estagios * (max_lag+1)` |
| .. | reg 9 per node and reg 10 per stage | `n_cenarios + n_estagios` |
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from novomodelo_bridge.dessem.exporters.fcf.core.reader import EntitySlotRecord

RECORD_SIZE = 48020
TAMANHO_CORTE = 26976

PHYSICAL_RECORD_COUNT = 14
PHYSICAL_RECORD_FIRST = 4
PHYSICAL_RECORD_LAST = PHYSICAL_RECORD_FIRST + PHYSICAL_RECORD_COUNT - 1

_MAX_POINTER = 2**31 - 1
_HEADER_RECORD_COUNT = 4
_TREE_RECORD_COUNT = 1
_STAGE_RECORD_COUNT = 1

ENTITY_TYPE_HYDRO_TRANSIT_BUCKET = 3


class LayoutError(Exception):
    """Raised when a DECOMP binary violates its layout invariant."""


class MapcutDimensions(Protocol):
    """Protocol for objects carrying mapcut dimension fields."""

    @property
    def n_utv(self) -> int: ...
    @property
    def numero_estagios(self) -> int: ...
    @property
    def max_lag(self) -> int: ...
    @property
    def numero_cenarios(self) -> int: ...


def mapcut_record_count(
    n_utv: int, n_estagios: int, max_lag: int, n_cenarios: int
) -> int:
    """Calculate expected record count for a mapcut file."""
    for name, value in (
        ("n_utv", n_utv),
        ("n_estagios", n_estagios),
        ("max_lag", max_lag),
        ("n_cenarios", n_cenarios),
    ):
        if value < 0:
            raise LayoutError(f"{name} must be non-negative, got {value}")
    return (
        _HEADER_RECORD_COUNT
        + PHYSICAL_RECORD_COUNT
        + _TREE_RECORD_COUNT
        + _STAGE_RECORD_COUNT
        + n_utv * n_estagios * (max_lag + 1)
        + n_cenarios
        + n_estagios
    )


def derive_n_utv(slots: Sequence[EntitySlotRecord]) -> int:
    """Count travel-time axes from entity manifest."""
    return sum(
        1 for slot in slots if slot.entity_type == ENTITY_TYPE_HYDRO_TRANSIT_BUCKET
    )


def assert_no_travel_time(slots: Sequence[EntitySlotRecord]) -> None:
    """Refuse conversion if travel-time state exists (premise P3)."""
    count = derive_n_utv(slots)
    if count:
        raise LayoutError(
            f"the case carries {count} HydroTransitBucket slot(s), so premise P3 "
            f"does not hold: travel-time coefficients are not supported"
        )


def assert_record_multiple(size_bytes: int, path_label: str) -> int:
    """Verify file size is a whole number of records."""
    records, remainder = divmod(size_bytes, RECORD_SIZE)
    if remainder:
        raise LayoutError(
            f"{path_label} is {size_bytes} bytes, not a whole number of "
            f"{RECORD_SIZE}-byte records: {records} records plus {remainder} "
            f"trailing bytes"
        )
    return records


def assert_mapcut_layout(path: Path, header: MapcutDimensions) -> None:
    """Verify written mapcut matches its header's declared layout."""
    expected = mapcut_record_count(
        header.n_utv, header.numero_estagios, header.max_lag, header.numero_cenarios
    )
    observed = assert_record_multiple(path.stat().st_size, str(path))
    if observed != expected:
        raise LayoutError(
            f"{path} holds {observed} records but its header declares {expected} "
            f"(n_utv={header.n_utv}, n_estagios={header.numero_estagios}, "
            f"max_lag={header.max_lag}, n_cenarios={header.numero_cenarios})"
        )

    with path.open("rb") as handle:
        handle.seek(PHYSICAL_RECORD_FIRST * RECORD_SIZE)
        for index in range(PHYSICAL_RECORD_FIRST, PHYSICAL_RECORD_LAST + 1):
            body = handle.read(RECORD_SIZE)
            if body.strip(b"\x00"):
                offset = next(position for position, byte in enumerate(body) if byte)
                raise LayoutError(
                    f"record {index} is inside the zero-filled physical span "
                    f"(records {PHYSICAL_RECORD_FIRST}-{PHYSICAL_RECORD_LAST}) "
                    f"but byte {offset} is {body[offset]}"
                )


def cortdeco_ncoef(
    n_uhes: int,
    n_utv: int,
    max_lag: int,
    n_sbm_gnl: int,
    n_estagios: int,
    n_patamares: int,
) -> int:
    """Calculate NCOEF for cortdeco: 1 + n_uhes + n_utv*max_lag + gnl_block."""
    for name, value in (
        ("n_uhes", n_uhes),
        ("n_utv", n_utv),
        ("max_lag", max_lag),
        ("n_sbm_gnl", n_sbm_gnl),
        ("n_estagios", n_estagios),
        ("n_patamares", n_patamares),
    ):
        if value < 0:
            raise LayoutError(f"{name} must be non-negative, got {value}")
    return 1 + n_uhes + n_utv * max_lag + n_sbm_gnl * n_estagios * n_patamares


@dataclass(frozen=True)
class CutBlockOffsets:
    """Block start positions inside one cortdeco cut's NCOEF-length vector."""

    rhs: int
    pi_varm: int
    pi_gnl: int
    ncoef: int


def cortdeco_block_offsets(
    n_uhes: int,
    n_utv: int,
    max_lag: int,
    n_sbm_gnl: int,
    n_estagios: int,
    n_patamares: int,
) -> CutBlockOffsets:
    """Calculate NCOEF and block start positions."""
    ncoef = cortdeco_ncoef(n_uhes, n_utv, max_lag, n_sbm_gnl, n_estagios, n_patamares)
    return CutBlockOffsets(
        rhs=0,
        pi_varm=1,
        pi_gnl=1 + n_uhes + n_utv * max_lag,
        ncoef=ncoef,
    )


def cortdeco_gnl_offset(
    offsets: CutBlockOffsets,
    sbm: int,
    stage: int,
    block: int,
    n_sbm_gnl: int,
    n_estagios: int,
    n_patamares: int,
) -> int:
    """Calculate submarket-major pi_gnl address for (sbm, stage, block)."""
    for name, value, bound in (
        ("sbm", sbm, n_sbm_gnl),
        ("stage", stage, n_estagios),
        ("block", block, n_patamares),
    ):
        if not 0 <= value < bound:
            raise LayoutError(f"{name} must be in 0..{bound - 1}, got {value}")
    return (
        offsets.pi_gnl + sbm * (n_estagios * n_patamares) + stage * n_patamares + block
    )


def cut_head_indices(
    numero_cortes: int, n_nodes: int, n_nodes_total: int
) -> tuple[int, ...]:
    """Return 1-based index of each node's last cut, zero for non-cut-building."""
    for name, value in (
        ("numero_cortes", numero_cortes),
        ("n_nodes", n_nodes),
        ("n_nodes_total", n_nodes_total),
    ):
        if value < 0:
            raise LayoutError(f"{name} must be non-negative, got {value}")
    if n_nodes > n_nodes_total:
        raise LayoutError(
            f"n_nodes is {n_nodes} but n_nodes_total is {n_nodes_total}: "
            f"more cut-building nodes than nodes"
        )
    if n_nodes and numero_cortes % n_nodes:
        raise LayoutError(
            f"numero_cortes {numero_cortes} not divisible by n_nodes {n_nodes}"
        )
    return tuple(
        numero_cortes - node if node < n_nodes else 0 for node in range(n_nodes_total)
    )


def cortdeco_record_count(numero_cortes: int) -> int:
    """Return numero_cortes + 1: CEPEL writes one cut beyond the head table."""
    if numero_cortes < 0:
        raise LayoutError(f"numero_cortes must be non-negative, got {numero_cortes}")
    return numero_cortes + 1


def chain_pointer(record_index: int, n_nodes: int) -> int:
    """Calculate 1-based pointer to same node's previous cut, or 0 to terminate."""
    if record_index < 0:
        raise LayoutError(f"record_index must be non-negative, got {record_index}")
    if n_nodes <= 0:
        raise LayoutError(f"n_nodes must be positive, got {n_nodes}")
    if record_index < n_nodes:
        return 0
    return record_index - n_nodes + 1


def assert_pointer_in_range(pointer: int) -> None:
    """Refuse a pointer int32 cannot hold."""
    if not 0 <= pointer <= _MAX_POINTER:
        raise LayoutError(
            f"chain pointer {pointer} is outside int32 range 0..{_MAX_POINTER}"
        )


def _read_pointers(path: Path, records: int) -> list[int]:
    """Read every record's chain pointer."""
    pointers: list[int] = []
    with path.open("rb") as handle:
        for _ in range(records):
            pointers.append(int(np.frombuffer(handle.read(4), dtype="<i4")[0]))
            handle.seek(TAMANHO_CORTE - 4, os.SEEK_CUR)
    return pointers


def assert_cortdeco_layout(path: Path, numero_cortes: int, n_nodes: int) -> None:
    """Verify written cortdeco by walking its own bytes."""
    if n_nodes <= 0:
        raise LayoutError(f"n_nodes must be positive, got {n_nodes}")
    expected = cortdeco_record_count(numero_cortes)
    size = path.stat().st_size
    records, remainder = divmod(size, TAMANHO_CORTE)
    if remainder:
        raise LayoutError(
            f"{path} is {size} bytes, not a whole number of "
            f"{TAMANHO_CORTE}-byte records"
        )
    if records != expected:
        raise LayoutError(
            f"{path} holds {records} records but numero_cortes {numero_cortes} "
            f"implies {expected}"
        )

    pointers = _read_pointers(path, records)
    heads = cut_head_indices(numero_cortes, n_nodes, n_nodes)
    per_node = numero_cortes // n_nodes
    visited: set[int] = set()

    for node, head in enumerate(heads):
        index = head - 1
        walked = 0
        while True:
            if not 0 <= index < numero_cortes:
                raise LayoutError(
                    f"node {node}'s chain reached record {index}, "
                    f"outside 0..{numero_cortes - 1}"
                )
            if index in visited:
                raise LayoutError(f"record {index} reached by more than one chain")
            visited.add(index)
            walked += 1
            pointer = pointers[index]
            if pointer == 0:
                break
            index = pointer - 1
        if walked != per_node:
            raise LayoutError(
                f"node {node}'s chain holds {walked} cuts, expected {per_node}"
            )

    extra = pointers[numero_cortes]
    last_head = heads[n_nodes - 1]
    if extra != last_head:
        raise LayoutError(f"extra record's pointer is {extra}, expected {last_head}")


def write_durably(path: Path, payload: bytes) -> None:
    """Write payload and force to stable storage."""
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def sync_directory(path: Path) -> None:
    """Force a completed rename in path to stable storage."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def assert_gnl_span_is_zero(path: Path, records: int, offsets: CutBlockOffsets) -> None:
    """Verify every record's pi_gnl span is zero (premise P14)."""
    width = offsets.ncoef - offsets.pi_gnl
    span = 8 * width
    with path.open("rb") as handle:
        for index in range(records):
            handle.seek(index * TAMANHO_CORTE + 4 + 8 * offsets.pi_gnl)
            block = handle.read(span)
            if block.strip(b"\x00"):
                position = next(offset for offset, byte in enumerate(block) if byte)
                slot = position // 8
                raise LayoutError(
                    f"record {index}'s pi_gnl slot {slot} is not zero (premise P14)"
                )


def assert_uniform_blocks(patamares_por_estagio: Sequence[int]) -> int:
    """Return the single n_patamares or raise if varying."""
    distinct = sorted(set(patamares_por_estagio))
    if len(distinct) != 1:
        raise LayoutError(
            f"patamares_por_estagio is not uniform: {distinct}; "
            f"cortdeco's pi_gnl block has no defined layout"
        )
    return distinct[0]
