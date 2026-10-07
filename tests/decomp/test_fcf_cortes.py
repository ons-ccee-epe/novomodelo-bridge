"""Tests for the source model's ``CORTESH``/``cortes`` readers (``fcf/cortes.py``).

Per H5 (deck-independent unit coverage), every *unit* test in this module
reads no real deck: the header-parse (``read_cortesh``), record-assembly
(``read_cortes``), and trailer-detection (``_read_trailer``) paths are
exercised against synthetic ``_FakeCortesh``/``_FakeCortes`` stand-ins and
tiny in-code ``struct.pack`` blobs.
"""

from __future__ import annotations

import struct
from pathlib import Path
from unittest.mock import patch

import pandas as pd  # type: ignore[import-untyped]  # pandas-stubs not installed
import pytest

from novomodelo_bridge.decomp.fcf.cortes import (
    BoundaryCuts,
    CortesHeader,
    CutFamilySummary,
    StageCutRecord,
    _build_header,
    _read_trailer,
    read_cortes,
    read_cortesh,
    required_inflow_lag_depth,
    summarize_cut_families,
)


def test_build_header_rejects_boundary_stage_outside_individualized_band() -> None:
    """The per-stage gate names the individualized band it rejected against.

    Only the band attributes are needed on the fake: the gate check runs
    before any ``dados_uhes``/plant access, so the raise fires without ever
    touching the rest of the (deliberately absent) ``Cortesh`` surface.
    """

    class _FakeCortesh:
        estagio_individualizado_inicial = 3
        estagio_individualizado_final = 14

    with pytest.raises(ValueError, match=r"\[3, 14\]"):
        _build_header(_FakeCortesh(), boundary_stage=2)  # type: ignore[arg-type]


def _hybrid_fake_cortesh_169_plants_10_ficticia() -> object:
    """A synthetic hybrid ``Cortesh`` (mirrors production ``mar-26-rv2`` shape).

    169 plants total, 10 of them fictitious (indices 5, 12, 20, 33, 47, 58,
    69, 81, 95, 110 — deliberately including 5 and 12 so the non-fictitious
    subset has holes at both), leaving 159 non-fictitious. ``tipo_agregacao_caso
    == 0`` (hybrid) with an individualized band of ``[3, 14]``.
    """
    all_indices = list(range(1, 170))
    ficticia_indices = {5, 12, 20, 33, 47, 58, 69, 81, 95, 110}
    assert len(ficticia_indices) == 10

    class _FakeCortesh:
        tipo_agregacao_caso = 0
        numero_maximo_uhes = 169
        dados_uhes = pd.DataFrame(
            {
                "indice_usina": all_indices,
                "codigo_usina": [index * 1000 for index in all_indices],
                "ficticia": [
                    1 if index in ficticia_indices else 0 for index in all_indices
                ],
            }
        )
        dados_submercados = pd.DataFrame({"codigo_submercado": [1, 2]})
        numero_patamares = 3
        lag_maximo_gnl = 2
        numero_submercados = 2
        tamanho_registro_individualizado = 1500
        estagio_individualizado_inicial = 3
        estagio_individualizado_final = 14
        ultimo_registro_cortes_estagio = pd.DataFrame(
            {
                "tipo_estagio": ["estudo"],
                "estagio": [4],
                "indice_ultimo_corte": [1],
            }
        )

    return _FakeCortesh()


def test_build_header_hybrid_excludes_fictitious_plants() -> None:
    header = _build_header(
        _hybrid_fake_cortesh_169_plants_10_ficticia(),  # type: ignore[arg-type]
        boundary_stage=4,
    )

    assert header.n_plants == 159
    assert len(header.plant_codes) == 159
    assert header.individualized is True


def test_build_header_hybrid_tolerates_indice_usina_holes() -> None:
    # Fictitious plants sit at indice_usina 5 and 12 (among others); the
    # non-fictitious subset skips them without raising the old
    # "indice_usina is not a contiguous 1..N range" error.
    header = _build_header(
        _hybrid_fake_cortesh_169_plants_10_ficticia(),  # type: ignore[arg-type]
        boundary_stage=4,
    )

    # codigo_usina == indice_usina * 1000, so the excluded fictitious slots'
    # codes (5000, 12000) must be absent from the retained plant_codes, and
    # the retained codes stay in ascending indice_usina slot order.
    assert 5000 not in header.plant_codes
    assert 12000 not in header.plant_codes
    assert list(header.plant_codes) == sorted(header.plant_codes)


def test_read_cortesh_synthetic_preserves_slot_order() -> None:
    class _FakeCortesh:
        tipo_agregacao_caso = 1
        numero_maximo_uhes = 3
        dados_uhes = pd.DataFrame(
            {
                "indice_usina": [1, 2, 3],
                "codigo_usina": [30, 10, 20],
                "ficticia": [0, 0, 0],
            }
        )
        dados_submercados = pd.DataFrame({"codigo_submercado": [1]})
        numero_patamares = 3
        lag_maximo_gnl = 2
        numero_submercados = 1
        tamanho_registro_individualizado = 112
        estagio_individualizado_inicial = 1
        estagio_individualizado_final = 12
        ultimo_registro_cortes_estagio = pd.DataFrame(
            {
                "tipo_estagio": ["estudo"],
                "estagio": [10],
                "indice_ultimo_corte": [1],
            }
        )

    with patch(
        "novomodelo_bridge.decomp.fcf.cortes.Cortesh.read",
        return_value=_FakeCortesh(),
    ):
        header = read_cortesh(Path("unused-cortesh.dat"))

    assert isinstance(header, CortesHeader)
    # Slot order (indice_usina 1,2,3), not sorted by codigo_usina (30,10,20).
    assert header.plant_codes == (30, 10, 20)
    # No fictitious plants in this deck: n_plants/plant_codes equal the full
    # plant set — the fully-individualized, no-fictitious case is
    # regression-safe under the fictitious-exclusion + band-gate rework.
    assert header.n_plants == 3
    assert header.lag_maximo_gnl == 2
    assert header.individualized is True


def test_read_cortes_rejects_nonzero_sar(tmp_path: Path) -> None:
    class _FakeCortesh:
        tipo_agregacao_caso = 1
        numero_maximo_uhes = 1
        dados_uhes = pd.DataFrame(
            {"indice_usina": [1], "codigo_usina": [4], "ficticia": [0]}
        )
        dados_submercados = pd.DataFrame({"codigo_submercado": [1]})
        numero_patamares = 3
        lag_maximo_gnl = 0
        numero_submercados = 1
        tamanho_registro_individualizado = 32
        estagio_individualizado_inicial = 1
        estagio_individualizado_final = 12
        ultimo_registro_cortes_estagio = pd.DataFrame(
            {
                "tipo_estagio": ["estudo"],
                "estagio": [5],
                "indice_ultimo_corte": [1],
            }
        )

    record_size = _FakeCortesh.tamanho_registro_individualizado
    record = bytearray(record_size)
    # Nonzero rhs (bytes 16:24) on the last physical record marks this as
    # the consolidated (no-sentinel) shape, so `boundary_stage` is required
    # and no trailer is derived.
    record[16:24] = struct.pack("<d", 1.0)
    cortes_path = tmp_path / "cortes.dat"
    cortes_path.write_bytes(bytes(record) * 2)

    bad_frame = pd.DataFrame(
        {
            "rhs": [123.0],
            "pi_varm_uhe4": [0.0],
            **{f"pi_qafl_uhe4_lag{lag}": [0.0] for lag in range(1, 13)},
            "pi_mx_sar_uhe4": [7.0],
        }
    )

    class _FakeCortes:
        cortes = bad_frame

    with (
        patch(
            "novomodelo_bridge.decomp.fcf.cortes.Cortes.from_cortesh",
            return_value=_FakeCortes(),
        ),
        pytest.raises(ValueError, match="SAR"),
    ):
        read_cortes(cortes_path, _FakeCortesh(), boundary_stage=5)


def test_read_cortes_synthetic_records_and_trailer_stage(tmp_path: Path) -> None:
    class _FakeCortesh:
        tipo_agregacao_caso = 1
        numero_maximo_uhes = 1
        dados_uhes = pd.DataFrame(
            {"indice_usina": [1], "codigo_usina": [4], "ficticia": [0]}
        )
        dados_submercados = pd.DataFrame({"codigo_submercado": [1]})
        numero_patamares = 3
        lag_maximo_gnl = 0
        numero_submercados = 1
        tamanho_registro_individualizado = 32
        estagio_individualizado_inicial = 1
        estagio_individualizado_final = 12
        ultimo_registro_cortes_estagio = pd.DataFrame(
            {
                "tipo_estagio": ["estudo"],
                "estagio": [10],
                "indice_ultimo_corte": [1],
            }
        )
        ano_inicio_estudo = 2024

    record_size = _FakeCortesh.tamanho_registro_individualizado
    # The single physical record is the sentinel: leading int32[4] trailer
    # (study_start_month, study_start_year, cut_stage_month, cut_stage_year),
    # then rhs left at 0.0 (bytes [16:24] default-zeroed) so `_read_trailer`
    # recognizes it as the sentinel, not a real cut.
    sentinel = bytearray(record_size)
    sentinel[0:16] = struct.pack("<4i", 9, 2024, 10, 2024)
    cortes_path = tmp_path / "cortes-010.dat"
    cortes_path.write_bytes(bytes(sentinel))

    qafl_lags = tuple(float(lag) for lag in range(1, 13))
    frame = pd.DataFrame(
        {
            "rhs": [100.0],
            "indice_corte": [7],
            "iteracao_construcao": [3],
            "indice_forward": [2],
            "iteracao_desativacao": [0],
            "pi_varm_uhe4": [0.5],
            **{
                f"pi_qafl_uhe4_lag{lag}": [value]
                for lag, value in enumerate(qafl_lags, start=1)
            },
            "pi_mx_sar_uhe4": [0.0],
        }
    )

    class _FakeCortes:
        cortes = frame

    with patch(
        "novomodelo_bridge.decomp.fcf.cortes.Cortes.from_cortesh",
        return_value=_FakeCortes(),
    ):
        boundary = read_cortes(cortes_path, _FakeCortesh(), boundary_stage=None)

    # Trailer-derived: (cut_stage_year=2024 - ano_inicio_estudo=2024) * 12
    # + cut_stage_month=10 == 10.
    assert boundary.boundary_stage == 10
    assert len(boundary.records) == 1
    record = boundary.records[0]
    assert record.pi_varm == (0.5,)
    assert record.pi_qafl == (qafl_lags,)
    assert record.cut_id == 7
    assert record.iteration == 3
    assert record.forward_pass_index == 2
    assert record.is_active is True


def test_read_trailer_sentinel_vs_nonzero_rhs(tmp_path: Path) -> None:
    # A tiny record: int32[4] (16 bytes) + rhs float64 (8 bytes), no padding.
    record_size = 24

    sentinel_path = tmp_path / "sentinel.dat"
    sentinel_path.write_bytes(struct.pack("<4i", 1, 2, 3, 4) + struct.pack("<d", 0.0))

    nonzero_path = tmp_path / "nonzero.dat"
    nonzero_path.write_bytes(struct.pack("<4i", 1, 2, 3, 4) + struct.pack("<d", 5.0))

    assert _read_trailer(sentinel_path, record_size) == (1, 2, 3, 4)
    assert _read_trailer(nonzero_path, record_size) is None


def test_summarize_families_empty_records_raises() -> None:
    header = CortesHeader(
        plant_codes=(4,),
        submercado_codes=(1,),
        n_patamares=3,
        lag_maximo_gnl=0,
        n_plants=1,
        individualized=True,
        record_size=32,
        last_cut_record_by_stage=(1,),
    )
    empty = BoundaryCuts(header=header, boundary_stage=5, records=())

    with pytest.raises(ValueError, match="no active cuts"):
        summarize_cut_families(empty)


def test_summarize_families_counts_plant_once_across_records() -> None:
    header = CortesHeader(
        plant_codes=(4, 5),
        submercado_codes=(1,),
        n_patamares=3,
        lag_maximo_gnl=1,
        n_plants=2,
        individualized=True,
        record_size=32,
        last_cut_record_by_stage=(1,),
    )
    records = (
        StageCutRecord(
            cut_id=1,
            iteration=1,
            forward_pass_index=1,
            is_active=True,
            rhs=10.0,
            pi_varm=(1.0, 0.0),
            pi_qafl=((1.0,) + (0.0,) * 11, (0.0,) * 12),
            pi_gnl=(1.0,),
        ),
        StageCutRecord(
            cut_id=2,
            iteration=1,
            forward_pass_index=2,
            is_active=True,
            rhs=20.0,
            pi_varm=(1.0, 0.0),
            pi_qafl=((1.0,) + (0.0,) * 11, (0.0,) * 12),
            pi_gnl=(1.0,),
        ),
    )
    cuts = BoundaryCuts(header=header, boundary_stage=5, records=records)

    summary = summarize_cut_families(cuts)

    # Plant 0 is nonzero at lag 1 in both records; counted once, not twice.
    assert summary.storage_nonzero_plants == 1
    assert summary.lag_nonzero_by_depth == (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    assert summary.gnl_nonzero_slots == 1
    assert summary.rhs_min == 10.0
    assert summary.rhs_max == 20.0


def _summary_with_lag_depths(lag_nonzero_by_depth: tuple[int, ...]) -> CutFamilySummary:
    """A ``CutFamilySummary`` whose only varying field is the lag-depth census."""
    return CutFamilySummary(
        n_active_cuts=1,
        storage_nonzero_plants=0,
        lag_nonzero_by_depth=lag_nonzero_by_depth,
        gnl_nonzero_slots=0,
        rhs_min=0.0,
        rhs_max=0.0,
    )


def test_required_inflow_lag_depth_returns_deepest_nonzero_depth() -> None:
    # Nonzero at calendar-month depths 1 and 3 (0-based indices 0 and 2): the
    # deepest referenced depth is 3, so 3 lag slots must be reserved.
    summary = _summary_with_lag_depths((2, 0, 1) + (0,) * 9)

    assert required_inflow_lag_depth(summary) == 3


def test_required_inflow_lag_depth_full_annual_depth() -> None:
    summary = _summary_with_lag_depths((1,) * 12)

    assert required_inflow_lag_depth(summary) == 12


def test_required_inflow_lag_depth_zero_when_no_lag_terms() -> None:
    # No cut carries a lag coefficient: no slots to reserve.
    summary = _summary_with_lag_depths((0,) * 12)

    assert required_inflow_lag_depth(summary) == 0
