"""Native serializer for the DECOMP `mapcut` binary.

`idecomp.Mapcut.write` raises `NotImplementedError`, so every byte here is
written natively.

Field order within each record follows `idecomp`'s own section definitions.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from novomodelo_bridge.dessem.exporters.fcf.core.layout import (
    PHYSICAL_RECORD_COUNT,
    PHYSICAL_RECORD_FIRST,
    PHYSICAL_RECORD_LAST,
    RECORD_SIZE,
    TAMANHO_CORTE,
    LayoutError,
    assert_mapcut_layout,
    mapcut_record_count,
    sync_directory,
    write_durably,
)

_LOG = logging.getLogger(__name__)

STAGE_RECORD_LEADING_FLAG = 1
_COST_FIELDS_PER_STAGE = 6

_Segment = tuple[Literal["<i4", "<f8"], Sequence[int] | Sequence[float]]


@dataclass(frozen=True)
class MapcutHeader:
    """Every value the `mapcut` records need, so the writer is a pure function."""

    numero_iteracoes: int
    numero_cortes: int
    numero_submercados: int
    numero_uhes: int
    numero_cenarios: int
    numero_estagios: int
    numero_semanas: int
    n_utv: int
    dia: int
    mes: int
    ano: int
    codigos_uhes: tuple[int, ...]
    codigos_uhes_jusante: tuple[int, ...]
    indice_no_arvore: tuple[int, ...]
    indice_primeiro_no_estagio: tuple[int, ...]
    patamares_por_estagio: tuple[int, ...]
    registro_ultimo_corte_no: tuple[int, ...]
    codigos_submercados_gnl: tuple[int, ...]
    lag_meses_gnl: tuple[int, ...]
    patamares_gnl: tuple[int, ...]
    taxa_desconto: tuple[float, ...]
    max_lag: int = 0


def _validate(header: MapcutHeader) -> None:
    """Validate header internal consistency."""
    checks = (
        ("codigos_uhes", len(header.codigos_uhes), "numero_uhes", header.numero_uhes),
        (
            "codigos_uhes_jusante",
            len(header.codigos_uhes_jusante),
            "numero_uhes",
            header.numero_uhes,
        ),
        (
            "indice_no_arvore",
            len(header.indice_no_arvore),
            "numero_cenarios",
            header.numero_cenarios,
        ),
        (
            "registro_ultimo_corte_no",
            len(header.registro_ultimo_corte_no),
            "numero_cenarios",
            header.numero_cenarios,
        ),
        (
            "indice_primeiro_no_estagio",
            len(header.indice_primeiro_no_estagio),
            "numero_estagios",
            header.numero_estagios,
        ),
        (
            "patamares_por_estagio",
            len(header.patamares_por_estagio),
            "numero_estagios",
            header.numero_estagios,
        ),
        (
            "taxa_desconto",
            len(header.taxa_desconto),
            "numero_estagios",
            header.numero_estagios,
        ),
        (
            "lag_meses_gnl",
            len(header.lag_meses_gnl),
            "codigos_submercados_gnl",
            len(header.codigos_submercados_gnl),
        ),
        (
            "patamares_gnl",
            len(header.patamares_gnl),
            "codigos_submercados_gnl",
            len(header.codigos_submercados_gnl),
        ),
    )
    for field, actual, against, expected in checks:
        if actual != expected:
            raise LayoutError(
                f"{field} has {actual} entries but {against} is {expected}; "
                f"the header is internally inconsistent"
            )
    if header.n_utv:
        raise LayoutError(
            f"n_utv is {header.n_utv}; mapcut regs 7/8 are unimplemented (P3)"
        )
    if header.max_lag < 0:
        raise LayoutError(f"max_lag is {header.max_lag}; cannot be negative")


def _record(index: int, *segments: _Segment) -> bytes:
    """One fixed-size record, zero-padded."""
    body = b"".join(
        np.asarray(values, dtype=dtype).tobytes() for dtype, values in segments
    )
    if len(body) > RECORD_SIZE:
        raise LayoutError(
            f"record {index} packs {len(body)} bytes, "
            f"more than {RECORD_SIZE}-byte record"
        )
    return body.ljust(RECORD_SIZE, b"\x00")


def _general_record(header: MapcutHeader) -> bytes:
    """Reg 1: general header."""
    return _record(
        0,
        (
            "<i4",
            (
                header.numero_iteracoes,
                header.numero_cortes,
                header.numero_submercados,
                header.numero_uhes,
                header.numero_cenarios,
                *header.registro_ultimo_corte_no,
            ),
        ),
    )


def _case_record(header: MapcutHeader) -> bytes:
    """Reg 2: case metadata."""
    return _record(1, ("<i4", (TAMANHO_CORTE, header.dia, header.mes, header.ano)))


def _stage_record(header: MapcutHeader) -> bytes:
    """Reg 6: stage data."""
    return _record(
        19,
        (
            "<i4",
            (
                STAGE_RECORD_LEADING_FLAG,
                header.numero_estagios,
                header.numero_semanas,
                header.n_utv,
                header.max_lag,
                *header.indice_primeiro_no_estagio,
                *header.patamares_por_estagio,
            ),
        ),
    )


def _gnl_record(header: MapcutHeader, index: int) -> bytes:
    """Reg 9: GNL configuration. P12: trailing block is zeros."""
    ngnl = len(header.codigos_submercados_gnl)
    return _record(
        index,
        (
            "<i4",
            (
                ngnl,
                *header.codigos_submercados_gnl,
                *header.lag_meses_gnl,
                *header.patamares_gnl,
            ),
        ),
        ("<f8", [0.0] * (header.numero_estagios * ngnl)),
    )


def _cost_record(header: MapcutHeader, stage: int, index: int) -> bytes:
    """Reg 10: cost fields. P11: only taxa_desconto is filled."""
    fields = [header.taxa_desconto[stage]] + [0.0] * (_COST_FIELDS_PER_STAGE - 1)
    return _record(index, ("<f8", fields))


def write_mapcut(header: MapcutHeader, path: Path) -> int:
    """Write a complete `mapcut`, returning the record count.

    Atomic write: builds at .partial, validates, renames into place.
    """
    _validate(header)
    expected = mapcut_record_count(
        header.n_utv,
        header.numero_estagios,
        header.max_lag,
        header.numero_cenarios,
    )

    records: list[bytes] = [
        _general_record(header),
        _case_record(header),
        _record(2, ("<i4", header.codigos_uhes)),
        _record(3, ("<i4", header.codigos_uhes_jusante)),
    ]
    records.extend(b"\x00" * RECORD_SIZE for _ in range(PHYSICAL_RECORD_COUNT))
    records.append(_record(18, ("<i4", header.indice_no_arvore)))
    records.append(_stage_record(header))

    for stage in range(header.numero_estagios):
        records.append(_gnl_record(header, len(records)))
        records.append(_cost_record(header, stage, len(records)))
    for _ in range(header.numero_cenarios - header.numero_estagios):
        records.append(_gnl_record(header, len(records)))

    if len(records) != expected:
        raise LayoutError(
            f"assembled {len(records)} records but layout expects {expected}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    try:
        write_durably(temporary, b"".join(records))
        assert_mapcut_layout(temporary, header)
        os.replace(temporary, path)
        sync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    _LOG.warning(
        "P1: mapcut records %d-%d emitted as zeros — physical hydro data deferred",
        PHYSICAL_RECORD_FIRST,
        PHYSICAL_RECORD_LAST,
    )
    _LOG.warning(
        "P3: travel-time (tempo de viagem) coefficients not yet supported — "
        "n_utv=0, pi_qdefp block empty"
    )
    _LOG.warning("P11: reg 10 cost fields emitted as zeros except taxa_desconto")
    _LOG.warning("P12: reg 9 trailing GNL block emitted as zeros")
    _LOG.info(
        "wrote mapcut %s: records=%d bytes=%d numero_cortes=%d numero_cenarios=%d",
        path,
        expected,
        expected * RECORD_SIZE,
        header.numero_cortes,
        header.numero_cenarios,
    )
    return expected
