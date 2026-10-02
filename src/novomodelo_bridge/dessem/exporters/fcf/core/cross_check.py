"""Cross-check validation: verify written mapcut/cortdeco pair agreement.

After writing both binaries, the pipeline must validate that they agree on key
structural values. This catches wiring bugs between the two writers before
publishing the pair.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from novomodelo_bridge.dessem.exporters.fcf.core.layout import (
    PHYSICAL_RECORD_FIRST,
    PHYSICAL_RECORD_LAST,
    RECORD_SIZE,
    TAMANHO_CORTE,
    CutBlockOffsets,
    LayoutError,
    assert_gnl_span_is_zero,
    assert_uniform_blocks,
    cortdeco_block_offsets,
    cortdeco_record_count,
    cut_head_indices,
    sync_directory,
)

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class MapcutScalars:
    """Scalar fields from mapcut reg 1 and reg 2."""

    numero_iteracoes: int
    numero_cortes: int
    numero_submercados: int
    numero_uhes: int
    numero_cenarios: int
    tamanho_corte: int
    dia: int
    mes: int
    ano: int
    numero_estagios: int
    numero_semanas: int
    n_utv: int
    max_lag: int


@dataclass(frozen=True)
class MapcutContents:
    """All values read from a mapcut binary."""

    scalars: MapcutScalars
    codigos_uhes: tuple[int, ...]
    codigos_uhes_jusante: tuple[int, ...]
    indice_no_arvore: tuple[int, ...]
    indice_primeiro_no_estagio: tuple[int, ...]
    patamares_por_estagio: tuple[int, ...]
    cut_heads: tuple[int, ...]
    codigos_submercados_gnl: tuple[int, ...]
    lag_meses_gnl: tuple[int, ...]
    patamares_gnl: tuple[int, ...]
    gnl_trailing_block: tuple[float, ...]
    taxa_desconto: tuple[float, ...]
    gnl_record_indices: tuple[int, ...]
    cost_records: tuple[tuple[float, ...], ...]
    physical_span_nonzero_records: tuple[int, ...]
    travel_time_lags: tuple[int, ...]


@dataclass(frozen=True)
class CortdecoCut:
    """One cut record from cortdeco."""

    record_index: int
    next_index: int
    is_extra_record: bool
    rhs: float
    pi_varm: NDArray[np.float64]
    pi_gnl: NDArray[np.float64]


@dataclass(frozen=True)
class CortdecoContents:
    """All values read from a cortdeco binary."""

    numero_cortes: int
    record_count: int
    cuts: tuple[CortdecoCut, ...]


def read_mapcut(path: Path) -> MapcutContents:
    """Read a mapcut binary file and return its contents."""
    with path.open("rb") as f:
        data = f.read()

    def read_int32_array(record: int, count: int, offset: int = 0) -> tuple[int, ...]:
        start = record * RECORD_SIZE + offset
        arr = np.frombuffer(data[start : start + 4 * count], dtype="<i4")
        return tuple(int(x) for x in arr)

    def read_float64_array(
        record: int, count: int, offset: int = 0
    ) -> tuple[float, ...]:
        start = record * RECORD_SIZE + offset
        arr = np.frombuffer(data[start : start + 8 * count], dtype="<f8")
        return tuple(float(x) for x in arr)

    # Reg 1: general header
    reg1 = read_int32_array(0, 5)
    numero_iteracoes = reg1[0]
    numero_cortes = reg1[1]
    numero_submercados = reg1[2]
    numero_uhes = reg1[3]
    numero_cenarios = reg1[4]
    cut_heads = read_int32_array(0, numero_cenarios, offset=20)

    # Reg 2: case metadata
    reg2 = read_int32_array(1, 4)
    tamanho_corte = reg2[0]
    dia, mes, ano = reg2[1], reg2[2], reg2[3]

    # Reg 3-4: hydro codes
    codigos_uhes = read_int32_array(2, numero_uhes)
    codigos_uhes_jusante = read_int32_array(3, numero_uhes)

    # Physical records P1 check
    physical_nonzero = []
    for idx in range(PHYSICAL_RECORD_FIRST, PHYSICAL_RECORD_LAST + 1):
        start = idx * RECORD_SIZE
        if data[start : start + RECORD_SIZE].strip(b"\x00"):
            physical_nonzero.append(idx)

    # Reg 5: tree indices (record 18)
    indice_no_arvore = read_int32_array(18, numero_cenarios)

    # Reg 6: stage data (record 19)
    reg6_scalars = read_int32_array(19, 5)
    numero_estagios = reg6_scalars[1]
    numero_semanas = reg6_scalars[2]
    n_utv = reg6_scalars[3]
    max_lag = reg6_scalars[4]

    indice_primeiro_no_estagio = read_int32_array(19, numero_estagios, offset=20)
    patamares_por_estagio = read_int32_array(
        19, numero_estagios, offset=20 + 4 * numero_estagios
    )
    travel_time_count = n_utv * numero_estagios
    travel_time_lags = (
        read_int32_array(19, travel_time_count, offset=20 + 8 * numero_estagios)
        if travel_time_count
        else ()
    )

    # Reg 9 and 10 records
    gnl_indices = []
    cost_records = []
    record_idx = 20

    # First pass: identify record types
    for stage in range(numero_estagios):
        gnl_indices.append(record_idx)
        record_idx += 1
        cost_record = read_float64_array(record_idx, 6)
        cost_records.append(cost_record)
        record_idx += 1
    for _ in range(numero_cenarios - numero_estagios):
        gnl_indices.append(record_idx)
        record_idx += 1

    # Read first GNL record for configuration
    gnl_rec = gnl_indices[0]
    ngnl = read_int32_array(gnl_rec, 1)[0]
    codigos_submercados_gnl = read_int32_array(gnl_rec, ngnl, offset=4)
    lag_meses_gnl = read_int32_array(gnl_rec, ngnl, offset=4 + 4 * ngnl)
    patamares_gnl = read_int32_array(gnl_rec, ngnl, offset=4 + 8 * ngnl)
    gnl_trailing_block = read_float64_array(
        gnl_rec, numero_estagios * ngnl, offset=4 + 12 * ngnl
    )

    taxa_desconto = tuple(cr[0] for cr in cost_records)

    scalars = MapcutScalars(
        numero_iteracoes=numero_iteracoes,
        numero_cortes=numero_cortes,
        numero_submercados=numero_submercados,
        numero_uhes=numero_uhes,
        numero_cenarios=numero_cenarios,
        tamanho_corte=tamanho_corte,
        dia=dia,
        mes=mes,
        ano=ano,
        numero_estagios=numero_estagios,
        numero_semanas=numero_semanas,
        n_utv=n_utv,
        max_lag=max_lag,
    )

    return MapcutContents(
        scalars=scalars,
        codigos_uhes=codigos_uhes,
        codigos_uhes_jusante=codigos_uhes_jusante,
        indice_no_arvore=indice_no_arvore,
        indice_primeiro_no_estagio=indice_primeiro_no_estagio,
        patamares_por_estagio=patamares_por_estagio,
        cut_heads=cut_heads,
        codigos_submercados_gnl=codigos_submercados_gnl,
        lag_meses_gnl=lag_meses_gnl,
        patamares_gnl=patamares_gnl,
        gnl_trailing_block=gnl_trailing_block,
        taxa_desconto=taxa_desconto,
        gnl_record_indices=tuple(gnl_indices),
        cost_records=tuple(cost_records),
        physical_span_nonzero_records=tuple(physical_nonzero),
        travel_time_lags=travel_time_lags,
    )


def read_cortdeco(
    path: Path, offsets: CutBlockOffsets, n_nodes: int
) -> CortdecoContents:
    """Read a cortdeco binary file and return its contents."""
    with path.open("rb") as f:
        data = f.read()

    record_count = len(data) // TAMANHO_CORTE
    numero_cortes = record_count - 1  # P13: extra record

    cuts = []
    for idx in range(record_count):
        start = idx * TAMANHO_CORTE
        pointer = int(np.frombuffer(data[start : start + 4], dtype="<i4")[0])
        coefficients = np.frombuffer(
            data[start + 4 : start + 4 + 8 * offsets.ncoef], dtype="<f8"
        )

        rhs = float(coefficients[offsets.rhs])
        pi_varm = coefficients[offsets.pi_varm : offsets.pi_gnl].copy()
        pi_gnl = coefficients[offsets.pi_gnl :].copy()

        cuts.append(
            CortdecoCut(
                record_index=idx,
                next_index=pointer,
                is_extra_record=(idx == numero_cortes),
                rhs=rhs,
                pi_varm=pi_varm,
                pi_gnl=pi_gnl,
            )
        )

    return CortdecoContents(
        numero_cortes=numero_cortes, record_count=record_count, cuts=tuple(cuts)
    )


def cross_check_pair(
    mapcut_path: Path, cortdeco_path: Path
) -> tuple[MapcutContents, CortdecoContents]:
    """Read both files back and validate agreement.

    Returns the read-back contents for content CSV generation.
    """
    mapcut_contents = read_mapcut(mapcut_path)
    scalars = mapcut_contents.scalars

    # Re-derive offsets from mapcut (not from writer's offsets)
    n_patamares = assert_uniform_blocks(mapcut_contents.patamares_por_estagio)
    offsets = cortdeco_block_offsets(
        n_uhes=scalars.numero_uhes,
        n_utv=scalars.n_utv,
        max_lag=scalars.max_lag,
        n_sbm_gnl=len(mapcut_contents.codigos_submercados_gnl),
        n_estagios=scalars.numero_estagios,
        n_patamares=n_patamares,
    )

    n_nodes = scalars.numero_semanas
    cortdeco_contents = read_cortdeco(cortdeco_path, offsets, n_nodes)

    # Check numero_cortes
    if scalars.numero_cortes != cortdeco_contents.numero_cortes:
        raise LayoutError(
            f"mapcut declares numero_cortes={scalars.numero_cortes} but "
            f"cortdeco holds {cortdeco_contents.numero_cortes}"
        )

    # Check head table
    expected_heads = cut_head_indices(
        scalars.numero_cortes, n_nodes, scalars.numero_cenarios
    )
    if mapcut_contents.cut_heads != expected_heads:
        raise LayoutError(
            f"mapcut head table {mapcut_contents.cut_heads[:5]}... "
            f"does not match expected {expected_heads[:5]}..."
        )

    # P14: Assert GNL span is zero
    expected_records = cortdeco_record_count(scalars.numero_cortes)
    assert_gnl_span_is_zero(cortdeco_path, expected_records, offsets)

    _LOG.info(
        "cross-check passed: numero_cortes=%d heads=%s offsets.ncoef=%d",
        scalars.numero_cortes,
        expected_heads[:5],
        offsets.ncoef,
    )

    return mapcut_contents, cortdeco_contents


def reject_pair(mapcut_path: Path, cortdeco_path: Path) -> None:
    """Rename both files to .rejected suffix."""
    for path in (mapcut_path, cortdeco_path):
        rejected = path.with_name(path.name + ".rejected")
        try:
            os.replace(path, rejected)
            sync_directory(path.parent)
            _LOG.warning("rejected %s -> %s", path, rejected)
        except FileNotFoundError:
            pass  # File not yet written
