"""Native serializer for the DECOMP `cortdeco` binary.

One record is: an int32 chain pointer at offset 0, then `NCOEF` float64 in
block order `rhs | pi_varm | pi_gnl`, then zero padding to `TAMANHO_CORTE`.

`pi_gnl` carries zeros under premise P14. Reducing Novomodelo's anticipated-thermal
ring positions to `pi_gnl`'s `(submarket, stage, block)` address is unresolved.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from novomodelo_bridge.dessem.exporters.fcf.core.layout import (
    TAMANHO_CORTE,
    CutBlockOffsets,
    LayoutError,
    assert_cortdeco_layout,
    assert_gnl_span_is_zero,
    assert_pointer_in_range,
    chain_pointer,
    cortdeco_record_count,
    cut_head_indices,
    sync_directory,
    write_durably,
)
from novomodelo_bridge.dessem.exporters.fcf.core.mapper import (
    MappingError,
    hydro_code_for_position,
    index_slots,
    negate_gnl_array,
    to_decomp_cost,
    to_decomp_costs,
)

if TYPE_CHECKING:
    from novomodelo_bridge.dessem.exporters.fcf.core.reader import (
        AffinePieceRecord,
        EntitySlotRecord,
    )

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class CutInput:
    """One `cortdeco` cut, in Novomodelo's raw sign and units, in DECOMP block order.

    `pi_varm[i]` is hydro position `i`'s storage coefficient, ordered like
    `codigos_uhes`. Under premise P14 `pi_gnl` is `zeroed_gnl_block(offsets)`.
    """

    intercept: float
    pi_varm: NDArray[np.float64]
    pi_gnl: NDArray[np.float64]


def zeroed_gnl_block(offsets: CutBlockOffsets) -> NDArray[np.float64]:
    """The `pi_gnl` block every cut carries under premise P14: zeros."""
    return np.zeros(offsets.ncoef - offsets.pi_gnl, dtype=np.float64)


def storage_coefficients(
    piece: AffinePieceRecord,
    slots: Sequence[EntitySlotRecord],
    hydro_codes: Sequence[int],
) -> NDArray[np.float64]:
    """One piece's `pi_varm` sub-array, extracted from its pool's raw coefficients.

    Storage positions are ascending in the entity manifest and positionally
    identical to `codigos_uhes`, so extraction is a direct slice.
    """
    positions = index_slots(slots).storage
    if tuple(positions) != tuple(range(len(hydro_codes))):
        raise MappingError(
            f"storage occupies manifest positions {positions[:4]}... "
            f"({len(positions)} slots) but the code map declares "
            f"{len(hydro_codes)} plants at positions 0..{len(hydro_codes) - 1}"
        )
    for position in positions:
        hydro_code_for_position(hydro_codes, position)
    coefficients = np.asarray(piece.coefficients, dtype=np.float64)
    return coefficients[np.asarray(positions, dtype=np.intp)]


def _validate_lengths(cut: CutInput, offsets: CutBlockOffsets) -> None:
    """Validate coefficient array lengths."""
    capacity = (TAMANHO_CORTE - 4) // 8
    if offsets.ncoef > capacity:
        raise LayoutError(
            f"NCOEF is {offsets.ncoef} but a {TAMANHO_CORTE}-byte record "
            f"holds at most {capacity} float64 coefficients"
        )
    expected_varm = offsets.pi_gnl - offsets.pi_varm
    if len(cut.pi_varm) != expected_varm:
        raise LayoutError(
            f"pi_varm has {len(cut.pi_varm)} entries but layout expects {expected_varm}"
        )
    expected_gnl = offsets.ncoef - offsets.pi_gnl
    if len(cut.pi_gnl) != expected_gnl:
        raise LayoutError(
            f"pi_gnl has {len(cut.pi_gnl)} entries but layout expects {expected_gnl}"
        )


def _convert(cut: CutInput) -> tuple[float, NDArray[np.float64]]:
    """Apply premise P2 (÷1000) and P4 (GNL sign) exactly once."""
    rhs = to_decomp_cost(cut.intercept)
    coefficients = to_decomp_costs(np.concatenate((cut.pi_varm, cut.pi_gnl)))
    gnl_start = len(cut.pi_varm)
    coefficients[gnl_start:] = negate_gnl_array(coefficients[gnl_start:])
    return rhs, coefficients


def _normalize_signed_zero(
    rhs: float, coefficients: NDArray[np.float64]
) -> tuple[float, NDArray[np.float64]]:
    """Turn `-0.0` into `+0.0`, and change nothing else."""
    return rhs + 0.0, coefficients + 0.0


def _assert_gnl_input_is_zero(cuts: Sequence[Sequence[CutInput]]) -> None:
    """Refuse a non-zero `pi_gnl` while premise P14 holds."""
    for node, node_cuts in enumerate(cuts):
        for iteration, cut in enumerate(node_cuts):
            nonzero = np.flatnonzero(np.asarray(cut.pi_gnl, dtype=np.float64) != 0.0)
            if nonzero.size:
                slot = int(nonzero[0])
                raise LayoutError(
                    f"node {node} iteration {iteration} has non-zero pi_gnl "
                    f"at slot {slot} ({cut.pi_gnl[slot]!r}), but P14 requires "
                    f"zeros"
                )


def _assert_finite(
    rhs: float, coefficients: NDArray[np.float64], pi_varm_width: int
) -> None:
    """Reject non-finite values."""
    if not np.isfinite(rhs):
        raise LayoutError(f"rhs is not finite: {rhs!r}")
    bad = np.flatnonzero(~np.isfinite(coefficients))
    if bad.size:
        position = int(bad[0])
        if position < pi_varm_width:
            raise LayoutError(
                f"pi_varm position {position} is not finite: {coefficients[position]!r}"
            )
        raise LayoutError(
            f"pi_gnl position {position - pi_varm_width} is not finite: "
            f"{coefficients[position]!r}"
        )


def node_and_iteration(record_index: int, n_nodes: int) -> tuple[int, int]:
    """Which node and iteration own 0-based record `record_index`."""
    if record_index < 0:
        raise LayoutError(f"record_index must be non-negative, got {record_index}")
    if n_nodes <= 0:
        raise LayoutError(f"n_nodes must be positive, got {n_nodes}")
    return (n_nodes - 1) - (record_index % n_nodes), record_index // n_nodes


def serialize_cut(cut: CutInput, pointer: int, offsets: CutBlockOffsets) -> bytes:
    """One fixed `cortdeco` record: pointer, then `NCOEF` float64, zero-padded."""
    _validate_lengths(cut, offsets)
    rhs, coefficients = _convert(cut)
    _assert_finite(rhs, coefficients, len(cut.pi_varm))
    rhs, coefficients = _normalize_signed_zero(rhs, coefficients)

    body = bytearray(TAMANHO_CORTE)
    pointer_view = np.frombuffer(body, dtype="<i4", count=1)
    pointer_view[0] = pointer
    coefficient_view = np.frombuffer(body, dtype="<f8", count=offsets.ncoef, offset=4)
    coefficient_view[offsets.rhs] = rhs
    coefficient_view[offsets.pi_varm :] = coefficients
    return bytes(body)


def write_cortdeco(
    cuts: Sequence[Sequence[CutInput]],
    path: Path,
    *,
    numero_cortes: int,
    offsets: CutBlockOffsets,
) -> int:
    """Write a complete `cortdeco`, returning the record count.

    `cuts[node][iteration]` is one cut-building node's cut sequence.
    Atomic write: builds at .partial, validates, renames into place.
    """
    n_nodes = len(cuts)
    if n_nodes == 0:
        raise LayoutError("no cut-building nodes were supplied")
    supplied = sum(len(node_cuts) for node_cuts in cuts)
    if supplied != numero_cortes:
        raise LayoutError(
            f"{supplied} cuts were supplied across {n_nodes} nodes but "
            f"numero_cortes is {numero_cortes}"
        )
    heads = cut_head_indices(numero_cortes, n_nodes, n_nodes)
    per_node = numero_cortes // n_nodes
    if per_node == 0:
        raise LayoutError(
            f"numero_cortes is {numero_cortes} over {n_nodes} nodes, "
            f"so chains would be empty (P13 requires a last cut)"
        )
    for node, node_cuts in enumerate(cuts):
        if len(node_cuts) != per_node:
            raise LayoutError(
                f"node {node} carries {len(node_cuts)} cuts but expected {per_node}"
            )
    _assert_gnl_input_is_zero(cuts)

    expected = cortdeco_record_count(numero_cortes)
    records: list[bytes] = []
    for index in range(numero_cortes):
        node, iteration = node_and_iteration(index, n_nodes)
        pointer = chain_pointer(index, n_nodes)
        assert_pointer_in_range(pointer)
        records.append(serialize_cut(cuts[node][iteration], pointer, offsets))

    # P13: extra record duplicates the last node's last cut
    extra_pointer = chain_pointer(numero_cortes, n_nodes)
    assert_pointer_in_range(extra_pointer)
    records.append(
        serialize_cut(cuts[n_nodes - 1][per_node - 1], extra_pointer, offsets)
    )

    if len(records) != expected:
        raise LayoutError(
            f"assembled {len(records)} records but layout expects {expected}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    try:
        write_durably(temporary, b"".join(records))
        assert_cortdeco_layout(temporary, numero_cortes, n_nodes)
        assert_gnl_span_is_zero(temporary, expected, offsets)
        os.replace(temporary, path)
        sync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    _LOG.warning(
        "P14: GNL (anticipated thermal) coefficients emitted as zeros — "
        "mapping Novomodelo ring positions to DECOMP (submarket, stage, block) "
        "address is unresolved"
    )
    _LOG.info(
        "wrote cortdeco %s: records=%d bytes=%d numero_cortes=%d nodes=%d "
        "ncoef=%d heads=%s",
        path,
        expected,
        expected * TAMANHO_CORTE,
        numero_cortes,
        n_nodes,
        offsets.ncoef,
        list(heads),
    )
    return expected
