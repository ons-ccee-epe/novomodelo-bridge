"""Tests for ``newave.converters.tailrace.convert_tailrace_curves`` (polinjus → parquet)."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import pytest

from novomodelo_bridge.newave.converters.tailrace import convert_tailrace_curves
from novomodelo_bridge.newave.id_map import NewaveIdMap


class _FakePolinjus:
    """Minimal stand-in for ``inewave.libs.UsinasHidreletricas``.

    The converter only calls the two family/segment accessors with ``df=True``.
    """

    def __init__(self, families: pd.DataFrame, segments: pd.DataFrame) -> None:
        self._families = families
        self._segments = segments

    def hidreletrica_curvajusante(self, df: bool = True) -> pd.DataFrame:
        return self._families

    def hidreletrica_curvajusante_polinomio_segmento(
        self, df: bool = True
    ) -> pd.DataFrame:
        return self._segments


class _FakeCase:
    def __init__(self, polinjus: Any) -> None:
        self.polinjus = polinjus


def _id_map(hydro_codes: list[int]) -> NewaveIdMap:
    """ID map mapping ``hydro_codes`` to dense 0-based ids in list order."""
    return NewaveIdMap(
        subsystem_ids=[1],
        hydro_codes=hydro_codes,
        thermal_codes=[],
    )


def _segments(rows: list[dict[str, float | int]]) -> pd.DataFrame:
    cols = [
        "codigo_usina",
        "coeficiente_a0",
        "coeficiente_a1",
        "coeficiente_a2",
        "coeficiente_a3",
        "coeficiente_a4",
        "indice_familia",
        "indice_polinomio",
        "limite_inferior_vazao_jusante",
        "limite_superior_vazao_jusante",
    ]
    return pd.DataFrame(rows, columns=cols)


def _seg_row(
    code: int,
    family: int,
    segment: int,
    q_inf: float,
    q_sup: float,
    a0: float = 0.0,
) -> dict[str, float | int]:
    return {
        "codigo_usina": code,
        "coeficiente_a0": a0,
        "coeficiente_a1": 0.0,
        "coeficiente_a2": 0.0,
        "coeficiente_a3": 0.0,
        "coeficiente_a4": 0.0,
        "indice_familia": family,
        "indice_polinomio": segment,
        "limite_inferior_vazao_jusante": q_inf,
        "limite_superior_vazao_jusante": q_sup,
    }


def test_returns_none_when_no_polinjus() -> None:
    case = _FakeCase(polinjus=None)
    assert convert_tailrace_curves(case, _id_map([10])) is None  # type: ignore[arg-type]


def test_returns_none_when_empty() -> None:
    case = _FakeCase(
        _FakePolinjus(
            families=pd.DataFrame(
                columns=["codigo_usina", "indice_familia", "nivel_montante_referencia"]
            ),
            segments=_segments([]),
        )
    )
    assert convert_tailrace_curves(case, _id_map([10])) is None  # type: ignore[arg-type]


def test_schema_matches_novomodelo_contract() -> None:
    families = pd.DataFrame(
        [{"codigo_usina": 10, "indice_familia": 1, "nivel_montante_referencia": 500.0}]
    )
    segments = _segments([_seg_row(10, 1, 1, 0.0, 1000.0)])
    case = _FakeCase(_FakePolinjus(families, segments))

    table = convert_tailrace_curves(case, _id_map([10]))  # type: ignore[arg-type]
    assert table is not None

    fields = {f.name: (str(f.type), f.nullable) for f in table.schema}
    assert fields == {
        "hydro_id": ("int32", False),
        "family_id": ("int32", False),
        "downstream_reference_level_m": ("double", True),
        "segment_id": ("int32", False),
        "outflow_min_m3s": ("double", False),
        "outflow_max_m3s": ("double", False),
        "coefficient_0": ("double", False),
        "coefficient_1": ("double", False),
        "coefficient_2": ("double", False),
        "coefficient_3": ("double", False),
        "coefficient_4": ("double", False),
    }


def test_remaps_codes_joins_downstream_level_and_sorts() -> None:
    # Two plants, families with distinct downstream reference levels; segments
    # supplied scrambled.
    families = pd.DataFrame(
        [
            {
                "codigo_usina": 10,
                "indice_familia": 1,
                "nivel_montante_referencia": 500.0,
            },
            {
                "codigo_usina": 20,
                "indice_familia": 1,
                "nivel_montante_referencia": 700.0,
            },
            {
                "codigo_usina": 20,
                "indice_familia": 2,
                "nivel_montante_referencia": 705.0,
            },
        ]
    )
    segments = _segments(
        [
            _seg_row(20, 2, 1, 0.0, 50.0, a0=705.0),
            _seg_row(10, 1, 2, 100.0, 200.0, a0=510.0),
            _seg_row(10, 1, 1, 0.0, 100.0, a0=500.0),
            _seg_row(20, 1, 1, 0.0, 80.0, a0=700.0),
        ]
    )
    case = _FakeCase(_FakePolinjus(families, segments))

    table = convert_tailrace_curves(case, _id_map([10, 20]))  # type: ignore[arg-type]
    assert table is not None
    out = table.to_pydict()

    # hydro_id 10->0, 20->1; sorted by (hydro_id, family_id, segment_id).
    assert out["hydro_id"] == [0, 0, 1, 1]
    assert out["family_id"] == [1, 1, 1, 2]
    assert out["segment_id"] == [1, 2, 1, 1]
    assert out["downstream_reference_level_m"] == [500.0, 500.0, 700.0, 705.0]
    assert out["coefficient_0"] == [500.0, 510.0, 700.0, 705.0]


def test_skips_codes_absent_from_id_map() -> None:
    families = pd.DataFrame(
        [
            {
                "codigo_usina": 10,
                "indice_familia": 1,
                "nivel_montante_referencia": 500.0,
            },
            {
                "codigo_usina": 99,
                "indice_familia": 1,
                "nivel_montante_referencia": 900.0,
            },
        ]
    )
    segments = _segments(
        [
            _seg_row(10, 1, 1, 0.0, 100.0),
            _seg_row(99, 1, 1, 0.0, 100.0),  # 99 not in id map -> dropped
        ]
    )
    case = _FakeCase(_FakePolinjus(families, segments))

    table = convert_tailrace_curves(case, _id_map([10]))  # type: ignore[arg-type]
    assert table is not None
    assert table.to_pydict()["hydro_id"] == [0]


def test_returns_none_when_all_codes_unmapped() -> None:
    families = pd.DataFrame(
        [{"codigo_usina": 99, "indice_familia": 1, "nivel_montante_referencia": 900.0}]
    )
    segments = _segments([_seg_row(99, 1, 1, 0.0, 100.0)])
    case = _FakeCase(_FakePolinjus(families, segments))

    assert convert_tailrace_curves(case, _id_map([10])) is None  # type: ignore[arg-type]


def test_nan_href_coerced_to_null() -> None:
    families = pd.DataFrame(
        [
            {
                "codigo_usina": 10,
                "indice_familia": 1,
                "nivel_montante_referencia": math.nan,
            }
        ]
    )
    segments = _segments([_seg_row(10, 1, 1, 0.0, 100.0)])
    case = _FakeCase(_FakePolinjus(families, segments))

    table = convert_tailrace_curves(case, _id_map([10]))  # type: ignore[arg-type]
    assert table is not None
    assert table.to_pydict()["downstream_reference_level_m"] == [None]


@pytest.mark.parametrize("missing", ["families", "segments"])
def test_returns_none_when_a_frame_is_empty(missing: str) -> None:
    families = pd.DataFrame(
        [{"codigo_usina": 10, "indice_familia": 1, "nivel_montante_referencia": 500.0}]
    )
    segments = _segments([_seg_row(10, 1, 1, 0.0, 100.0)])
    if missing == "families":
        families = families.iloc[0:0]
    else:
        segments = segments.iloc[0:0]
    case = _FakeCase(_FakePolinjus(families, segments))
    assert convert_tailrace_curves(case, _id_map([10])) is None  # type: ignore[arg-type]
