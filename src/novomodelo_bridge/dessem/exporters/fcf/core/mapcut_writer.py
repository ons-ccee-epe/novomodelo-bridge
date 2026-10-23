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
from typing import TYPE_CHECKING, Literal

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
    travel_time_lag_starts,
    write_durably,
)

if TYPE_CHECKING:
    from novomodelo_bridge.dessem.exporters.fcf.core.travel_time import TravelTimePlant

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
    tempo_viagem: tuple[TravelTimePlant, ...] = ()

    @property
    def n_utv(self) -> int:
        """Plants with a travel time, each with its own regs 7/8."""
        return len(self.tempo_viagem)

    @property
    def travel_time_lags(self) -> tuple[int, ...]:
        """Reg 6's last lag per stage and plant, stage-major with plants inner.

        Regs 7/8 run plant-major instead: reading this array plant-major
        swaps plants' lags wherever two plants' travel times differ.
        """
        return tuple(
            plant.stage_lags[stage]
            for stage in range(self.numero_estagios)
            for plant in self.tempo_viagem
        )

    @property
    def max_lag(self) -> int:
        """The largest of `travel_time_lags`, 0 without travel time."""
        return max(self.travel_time_lags, default=0)


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
    for plant in header.tempo_viagem:
        if len(plant.arrival_shares) != header.numero_estagios:
            raise LayoutError(
                f"travel-time plant {plant.code} has arrival shares for "
                f"{len(plant.arrival_shares)} stages but numero_estagios is "
                f"{header.numero_estagios}"
            )
        if plant.hours <= 0:
            raise LayoutError(
                f"travel-time plant {plant.code} has {plant.hours} h of travel time"
            )


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
                *header.travel_time_lags,
                *travel_time_lag_starts(
                    header.travel_time_lags, header.n_utv, header.numero_estagios
                ),
            ),
        ),
    )


def _travel_time_records(header: MapcutHeader, first_index: int) -> list[bytes]:
    """Regs 7/8: per plant, stage and lag, the share of that release arriving.

    Reg 7 opens each plant's run with its code and travel hours; reg 8 carries
    every later lag.
    """
    records: list[bytes] = []
    for plant in header.tempo_viagem:
        for stage, shares in enumerate(plant.arrival_shares):
            for lag, share in enumerate(shares):
                index = first_index + len(records)
                if stage == 0 and lag == 0:
                    records.append(
                        _record(
                            index,
                            ("<i4", (plant.code, plant.hours, lag)),
                            ("<f8", (share,)),
                        )
                    )
                else:
                    records.append(_record(index, ("<i4", (lag,)), ("<f8", (share,))))
    return records


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
        header.travel_time_lags, header.numero_estagios, header.numero_cenarios
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
    records.extend(_travel_time_records(header, len(records)))

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
    _LOG.warning("P11: reg 10 cost fields emitted as zeros except taxa_desconto")
    _LOG.warning("P12: reg 9 trailing GNL block emitted as zeros")
    _LOG.info(
        "wrote mapcut %s: records=%d bytes=%d numero_cortes=%d numero_cenarios=%d "
        "n_utv=%d max_lag=%d",
        path,
        expected,
        expected * RECORD_SIZE,
        header.numero_cortes,
        header.numero_cenarios,
        header.n_utv,
        header.max_lag,
    )
    return expected
