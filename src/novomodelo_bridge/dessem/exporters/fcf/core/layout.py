"""DECOMP binary layout constants and the invariants that check them.

Record map for mapcut.rv0:
| Records | Content | Count |
|---------|---------|-------|
| 0-3 | regs 1-4 | 4 |
| 4-17 | undocumented per-plant physical records | 14 |
| 18 | reg 5, parent-node indices | 1 |
| 19 | reg 6, stage data | 1 |
| 20.. | regs 7/8 travel time | `sum(lag + 1)` over reg 6's per-plant stage lags |
| .. | reg 9 per node and reg 10 per stage | `n_cenarios + n_estagios` |
"""

from __future__ import annotations

import itertools
import os
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

RECORD_SIZE = 48020
TAMANHO_CORTE = 26976

PHYSICAL_RECORD_COUNT = 14
PHYSICAL_RECORD_FIRST = 4
PHYSICAL_RECORD_LAST = PHYSICAL_RECORD_FIRST + PHYSICAL_RECORD_COUNT - 1
COST_FIELDS = (
    "taxa_desconto",
    "parcela_custo_geracao_termica_minima",
    "parcela_custo_contrato_importacao_minimo",
    "parcela_custo_contrato_exportacao_minimo",
    "geracao_termica_minima_sinalizada_gnl",
    "geracao_termica_minima_gerada_gnl",
)

_MAX_POINTER = 2**31 - 1
_HEADER_RECORD_COUNT = 4
_TREE_RECORD_COUNT = 1
_STAGE_RECORD_COUNT = 1


class LayoutError(Exception):
    """Raised when a DECOMP binary violates its layout invariant."""


class MapcutDimensions(Protocol):
    """Protocol for objects carrying mapcut dimension fields."""

    @property
    def travel_time_lags(self) -> tuple[int, ...]: ...
    @property
    def numero_estagios(self) -> int: ...
    @property
    def numero_cenarios(self) -> int: ...


def travel_time_record_count(travel_time_lags: Sequence[int]) -> int:
    """Regs 7/8 records: one per plant, stage and lag `0..lag` of reg 6."""
    if any(lag < 0 for lag in travel_time_lags):
        raise LayoutError(
            f"travel-time lags must be non-negative, got {list(travel_time_lags)}"
        )
    return sum(lag + 1 for lag in travel_time_lags)


def travel_time_lag_starts(
    travel_time_lags: Sequence[int], n_utv: int, n_estagios: int
) -> tuple[int, ...]:
    """Reg 6's second travel-time array, laid out like *travel_time_lags*.

    Both are stage-major with plants inner. Each start is 1 plus the lags of
    every (plant, stage) before it in plant-major order, as DECOMP writes it.
    """
    if len(travel_time_lags) != n_utv * n_estagios:
        raise LayoutError(
            f"{len(travel_time_lags)} travel-time lags for {n_utv} plants and "
            f"{n_estagios} stages"
        )
    starts = [0] * len(travel_time_lags)
    position = 1
    for plant in range(n_utv):
        for stage in range(n_estagios):
            index = stage * n_utv + plant
            starts[index] = position
            position += travel_time_lags[index]
    return tuple(starts)


def mapcut_record_count(
    travel_time_lags: Sequence[int], n_estagios: int, n_cenarios: int
) -> int:
    """Calculate expected record count for a mapcut file."""
    for name, value in (("n_estagios", n_estagios), ("n_cenarios", n_cenarios)):
        if value < 0:
            raise LayoutError(f"{name} must be non-negative, got {value}")
    return (
        _HEADER_RECORD_COUNT
        + PHYSICAL_RECORD_COUNT
        + _TREE_RECORD_COUNT
        + _STAGE_RECORD_COUNT
        + travel_time_record_count(travel_time_lags)
        + n_cenarios
        + n_estagios
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
        header.travel_time_lags, header.numero_estagios, header.numero_cenarios
    )
    observed = assert_record_multiple(path.stat().st_size, str(path))
    if observed != expected:
        raise LayoutError(
            f"{path} holds {observed} records but its header declares {expected} "
            f"(travel-time lags={list(header.travel_time_lags)}, "
            f"n_estagios={header.numero_estagios}, "
            f"n_cenarios={header.numero_cenarios})"
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
    """Block starts in one cortdeco cut's NCOEF-length vector, and the pi_gnl axis lengths."""

    rhs: int
    pi_varm: int
    pi_qdefp: int
    pi_gnl: int
    ncoef: int
    n_sbm_gnl: int
    n_estagios: int
    n_patamares: int

    def __post_init__(self) -> None:
        lengths = (self.n_sbm_gnl, self.n_estagios, self.n_patamares)
        width = self.ncoef - self.pi_gnl
        if (
            min(lengths) < 0
            or width != self.n_sbm_gnl * self.n_estagios * self.n_patamares
        ):
            raise LayoutError(
                f"pi_gnl block is {width} wide but n_sbm_gnl={self.n_sbm_gnl}, "
                f"n_estagios={self.n_estagios} and n_patamares={self.n_patamares} "
                f"must be non-negative and multiply to that width"
            )


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
        pi_qdefp=1 + n_uhes,
        pi_gnl=1 + n_uhes + n_utv * max_lag,
        ncoef=ncoef,
        n_sbm_gnl=n_sbm_gnl,
        n_estagios=n_estagios,
        n_patamares=n_patamares,
    )


def cortdeco_gnl_offset(
    offsets: CutBlockOffsets, entry: int, middle: int, patamar: int
) -> int:
    """Return the cut-vector position of pi_gnl (entry, middle, patamar).

    Entry-major, then middle, then patamar; the middle axis has
    `offsets.n_estagios` positions.
    """
    for name, value, bound in (
        ("entry", entry, offsets.n_sbm_gnl),
        ("middle", middle, offsets.n_estagios),
        ("patamar", patamar, offsets.n_patamares),
    ):
        if not 0 <= value < bound:
            raise LayoutError(f"{name} must be in 0..{bound - 1}, got {value}")
    return (
        offsets.pi_gnl
        + (entry * offsets.n_estagios + middle) * offsets.n_patamares
        + patamar
    )


#: Middle-axis offset that holds the exported coefficient; idecomp labels it `lag1`.
GNL_EXPORT_MIDDLE_OFFSET = 0


def gnl_positions(
    n_sbm_gnl: int, n_estagios: int, n_patamares: int
) -> Iterator[tuple[int, int, int]]:
    """Yield (entry, middle, patamar) in on-disk order."""
    return itertools.product(range(n_sbm_gnl), range(n_estagios), range(n_patamares))


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
    """Force a completed rename in path to stable storage.

    A no-op on Windows, which cannot open a directory as a file.
    """
    if sys.platform == "win32":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def assert_gnl_support(
    path: Path, records: int, offsets: CutBlockOffsets, n_nodes: int
) -> None:
    """Verify pi_gnl is nonzero only at middle offset 0 of the first node's cuts.

    The first cut-building node's chain, plus the extra record when it is the
    only node, carries finite, bit-identical values across patamares at middle
    offset 0. Every other pi_gnl byte is zero, so `-0.0` and NaN are refused.
    """
    if offsets.n_sbm_gnl == 0:
        return
    numero_cortes = records - 1
    pointers = _read_pointers(path, records)
    fed: set[int] = set()
    index = cut_head_indices(numero_cortes, n_nodes, n_nodes)[0] - 1
    while True:
        if not 0 <= index < numero_cortes:
            raise LayoutError(
                f"node 0's chain reached record {index}, outside 0..{numero_cortes - 1}"
            )
        if index in fed:
            raise LayoutError(f"record {index} reached twice by node 0's chain")
        fed.add(index)
        if pointers[index] == 0:
            break
        index = pointers[index] - 1
    if n_nodes == 1:
        fed.add(numero_cortes)

    slots = {
        position: cortdeco_gnl_offset(offsets, *position) - offsets.pi_gnl
        for position in gnl_positions(
            offsets.n_sbm_gnl, offsets.n_estagios, offsets.n_patamares
        )
    }
    position_at = {slot: position for position, slot in slots.items()}
    support = np.array(
        [s for (_, m, _), s in slots.items() if m == GNL_EXPORT_MIDDLE_OFFSET],
        dtype=np.intp,
    ).reshape(offsets.n_sbm_gnl, -1)
    off_support = np.array(
        [s for (_, m, _), s in slots.items() if m != GNL_EXPORT_MIDDLE_OFFSET],
        dtype=np.intp,
    )

    span = 8 * (offsets.ncoef - offsets.pi_gnl)
    with path.open("rb") as handle:
        for index in range(records):
            handle.seek(index * TAMANHO_CORTE + 4 + 8 * offsets.pi_gnl)
            bits = np.frombuffer(handle.read(span), dtype="<u8")
            values = bits.view("<f8")
            where = f"record {index}'s pi_gnl entry"

            if index not in fed:
                nonzero = np.flatnonzero(bits)
                if nonzero.size:
                    slot = int(nonzero[0])
                    entry, middle, patamar = position_at[slot]
                    raise LayoutError(
                        f"{where} {entry}, middle {middle}, patamar {patamar} is "
                        f"{float(values[slot])!r}: only the first cut-building "
                        f"node's cuts carry GNL coefficients"
                    )
                continue

            stray = off_support[bits[off_support] != 0]
            if stray.size:
                slot = int(stray[0])
                entry, middle, patamar = position_at[slot]
                raise LayoutError(
                    f"{where} {entry}, middle {middle}, patamar {patamar} is "
                    f"{float(values[slot])!r}: only middle offset "
                    f"{GNL_EXPORT_MIDDLE_OFFSET} carries a GNL coefficient"
                )
            exported = values[support]
            not_finite = np.argwhere(~np.isfinite(exported))
            if not_finite.size:
                entry, patamar = (int(i) for i in not_finite[0])
                raise LayoutError(
                    f"{where} {entry}, middle {GNL_EXPORT_MIDDLE_OFFSET}, "
                    f"patamar {patamar} is {float(exported[entry, patamar])!r}: "
                    f"a GNL coefficient must be finite"
                )
            exported_bits = bits[support]
            # TRACKED NOVOMODELO-GAP (C24): novomodelo holds one value per delivery, not per load block.
            unequal = np.argwhere(exported_bits != exported_bits[:, :1])
            if unequal.size:
                entry, patamar = (int(i) for i in unequal[0])
                raise LayoutError(
                    f"{where} {entry}, middle {GNL_EXPORT_MIDDLE_OFFSET}: patamar 0 "
                    f"is {float(exported[entry, 0])!r} but patamar {patamar} is "
                    f"{float(exported[entry, patamar])!r}; a GNL coefficient is "
                    f"one value per entry, repeated on every patamar"
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
