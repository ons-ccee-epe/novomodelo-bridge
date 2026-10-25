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
    travel_time_lag_starts,
    travel_time_record_count,
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
class TravelTimeRecord:
    """One reg 7/8 record: the share of a plant's lagged release arriving in a stage.

    `plant` is the plant's 0-based travel-time position. Only reg 7, which
    opens each plant's run, carries `codigo_usina` and `numero_horas`.
    """

    record_index: int
    plant: int
    stage: int
    lag: int
    share: float
    codigo_usina: int | None
    numero_horas: int | None


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
    travel_time_lag_starts: tuple[int, ...]
    travel_time_records: tuple[TravelTimeRecord, ...]

    @property
    def codigos_uhes_tempo_viagem(self) -> tuple[int, ...]:
        """The travel-time plants' codes, in reg 7 order."""
        return tuple(
            record.codigo_usina
            for record in self.travel_time_records
            if record.codigo_usina is not None
        )


@dataclass(frozen=True)
class CortdecoCut:
    """One cut record from cortdeco."""

    record_index: int
    next_index: int
    is_extra_record: bool
    rhs: float
    pi_varm: NDArray[np.float64]
    pi_qdefp: NDArray[np.float64]
    pi_gnl: NDArray[np.float64]


@dataclass(frozen=True)
class CortdecoContents:
    """All values read from a cortdeco binary."""

    numero_cortes: int
    record_count: int
    cuts: tuple[CortdecoCut, ...]


def _int32s(data: bytes, record: int, count: int, offset: int = 0) -> tuple[int, ...]:
    start = record * RECORD_SIZE + offset
    return tuple(
        int(x) for x in np.frombuffer(data[start : start + 4 * count], dtype="<i4")
    )


def _float64s(
    data: bytes, record: int, count: int, offset: int = 0
) -> tuple[float, ...]:
    start = record * RECORD_SIZE + offset
    return tuple(
        float(x) for x in np.frombuffer(data[start : start + 8 * count], dtype="<f8")
    )


def read_mapcut(path: Path) -> MapcutContents:
    """Read a mapcut binary file and return its contents."""
    with path.open("rb") as f:
        data = f.read()

    # Reg 1: general header
    reg1 = _int32s(data, 0, 5)
    numero_iteracoes = reg1[0]
    numero_cortes = reg1[1]
    numero_submercados = reg1[2]
    numero_uhes = reg1[3]
    numero_cenarios = reg1[4]
    cut_heads = _int32s(data, 0, numero_cenarios, offset=20)

    # Reg 2: case metadata
    reg2 = _int32s(data, 1, 4)
    tamanho_corte = reg2[0]
    dia, mes, ano = reg2[1], reg2[2], reg2[3]

    # Reg 3-4: hydro codes
    codigos_uhes = _int32s(data, 2, numero_uhes)
    codigos_uhes_jusante = _int32s(data, 3, numero_uhes)

    # Physical records P1 check
    physical_nonzero = []
    for idx in range(PHYSICAL_RECORD_FIRST, PHYSICAL_RECORD_LAST + 1):
        start = idx * RECORD_SIZE
        if data[start : start + RECORD_SIZE].strip(b"\x00"):
            physical_nonzero.append(idx)

    # Reg 5: tree indices (record 18)
    indice_no_arvore = _int32s(data, 18, numero_cenarios)

    # Reg 6: stage data (record 19)
    reg6_scalars = _int32s(data, 19, 5)
    numero_estagios = reg6_scalars[1]
    numero_semanas = reg6_scalars[2]
    n_utv = reg6_scalars[3]
    max_lag = reg6_scalars[4]

    indice_primeiro_no_estagio = _int32s(data, 19, numero_estagios, offset=20)
    patamares_por_estagio = _int32s(
        data, 19, numero_estagios, offset=20 + 4 * numero_estagios
    )
    travel_time_count = n_utv * numero_estagios
    lags_offset = 20 + 8 * numero_estagios
    travel_time_lags = _int32s(data, 19, travel_time_count, offset=lags_offset)
    travel_time_lag_starts = _int32s(
        data, 19, travel_time_count, offset=lags_offset + 4 * travel_time_count
    )

    travel_time_records = _read_travel_time_records(
        data, travel_time_lags, n_utv, numero_estagios
    )

    # Reg 9 and 10 records start after travel-time records
    gnl_indices = []
    cost_records = []
    record_idx = 20 + travel_time_record_count(travel_time_lags)

    # First pass: identify record types
    for stage in range(numero_estagios):
        gnl_indices.append(record_idx)
        record_idx += 1
        cost_record = _float64s(data, record_idx, 6)
        cost_records.append(cost_record)
        record_idx += 1
    for _ in range(numero_cenarios - numero_estagios):
        gnl_indices.append(record_idx)
        record_idx += 1

    # Read first GNL record for configuration
    gnl_rec = gnl_indices[0]
    ngnl = _int32s(data, gnl_rec, 1)[0]
    codigos_submercados_gnl = _int32s(data, gnl_rec, ngnl, offset=4)
    lag_meses_gnl = _int32s(data, gnl_rec, ngnl, offset=4 + 4 * ngnl)
    patamares_gnl = _int32s(data, gnl_rec, ngnl, offset=4 + 8 * ngnl)
    gnl_trailing_block = _float64s(
        data, gnl_rec, numero_estagios * ngnl, offset=4 + 12 * ngnl
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
        travel_time_lag_starts=travel_time_lag_starts,
        travel_time_records=travel_time_records,
    )


def _read_travel_time_records(
    data: bytes, travel_time_lags: tuple[int, ...], n_utv: int, numero_estagios: int
) -> tuple[TravelTimeRecord, ...]:
    """Regs 7/8 from record 20 on: plant-major, though reg 6's lags are stage-major.

    Raises:
        LayoutError: a record's lag field is not its position in the run.
    """
    codigo: int | None
    horas: int | None
    records: list[TravelTimeRecord] = []
    index = 20
    for plant in range(n_utv):
        for stage in range(numero_estagios):
            for lag in range(travel_time_lags[stage * n_utv + plant] + 1):
                if stage == 0 and lag == 0:
                    codigo, horas, lag_field = _int32s(data, index, 3)
                    (share,) = _float64s(data, index, 1, offset=12)
                else:
                    codigo = horas = None
                    (lag_field,) = _int32s(data, index, 1)
                    (share,) = _float64s(data, index, 1, offset=4)
                if lag_field != lag:
                    raise LayoutError(
                        f"mapcut record {index} carries lag {lag_field} but is lag "
                        f"{lag} of travel-time plant {plant + 1}, stage {stage + 1}"
                    )
                records.append(
                    TravelTimeRecord(
                        record_index=index,
                        plant=plant,
                        stage=stage,
                        lag=lag,
                        share=share,
                        codigo_usina=codigo,
                        numero_horas=horas,
                    )
                )
                index += 1
    return tuple(records)


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
        pi_varm = coefficients[offsets.pi_varm : offsets.pi_qdefp].copy()
        pi_qdefp = coefficients[offsets.pi_qdefp : offsets.pi_gnl].copy()
        pi_gnl = coefficients[offsets.pi_gnl :].copy()

        cuts.append(
            CortdecoCut(
                record_index=idx,
                next_index=pointer,
                is_extra_record=(idx == numero_cortes),
                rhs=rhs,
                pi_varm=pi_varm,
                pi_qdefp=pi_qdefp,
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

    expected_starts = travel_time_lag_starts(
        mapcut_contents.travel_time_lags, scalars.n_utv, scalars.numero_estagios
    )
    if mapcut_contents.travel_time_lag_starts != expected_starts:
        raise LayoutError(
            f"mapcut reg 6 holds travel-time lag starts "
            f"{mapcut_contents.travel_time_lag_starts} but its lags imply "
            f"{expected_starts}"
        )
    declared_lag = max(mapcut_contents.travel_time_lags, default=0)
    if scalars.max_lag != declared_lag:
        raise LayoutError(
            f"mapcut declares max_lag={scalars.max_lag} but its largest "
            f"travel-time lag is {declared_lag}"
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
