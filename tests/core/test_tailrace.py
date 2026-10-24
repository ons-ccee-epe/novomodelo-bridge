"""Tests for the tailrace-curve evaluators in ``novomodelo_bridge.core.tailrace``."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from novomodelo_bridge.core.tailrace import (
    TailraceFamily,
    TailraceSegment,
    tailrace_families_from_hidr,
    tailrace_families_from_polinjus,
    tailrace_level_m,
)


def _flat(level_m: float, outflow_max_m3s: float = math.inf) -> TailraceSegment:
    return TailraceSegment(outflow_max_m3s, (level_m, 0.0, 0.0, 0.0, 0.0))


def test_family_level_takes_the_first_segment_reaching_the_outflow() -> None:
    family = TailraceFamily(math.nan, (_flat(10.0, 100.0), _flat(12.0, 300.0)))
    assert family.level_m(100.0) == 10.0
    assert family.level_m(150.0) == 12.0


def test_family_level_past_every_limit_takes_the_last_segment() -> None:
    family = TailraceFamily(math.nan, (_flat(10.0, 100.0), _flat(12.0, 300.0)))
    assert family.level_m(500.0) == 12.0


def test_family_level_evaluates_the_quartic_in_the_outflow() -> None:
    family = TailraceFamily(
        math.nan, (TailraceSegment(math.inf, (1.0, 0.5, 0.01, 0.0, 0.0)),)
    )
    assert family.level_m(10.0) == pytest.approx(1.0 + 5.0 + 1.0)


_FAMILIES = (
    TailraceFamily(10.0, (_flat(15.0),)),
    TailraceFamily(30.0, (_flat(35.0),)),
)


def test_tailrace_level_interpolates_between_the_bracketing_families() -> None:
    assert tailrace_level_m(_FAMILIES, 100.0, 20.0) == pytest.approx(25.0)


@pytest.mark.parametrize(
    ("downstream_level_m", "expected"),
    [
        pytest.param(5.0, 15.0, id="below-the-lowest-reference"),
        pytest.param(40.0, 35.0, id="above-the-highest-reference"),
        pytest.param(None, 15.0, id="no-plant-downstream"),
    ],
)
def test_tailrace_level_outside_the_references_takes_the_nearest_family(
    downstream_level_m: float | None, expected: float
) -> None:
    assert tailrace_level_m(_FAMILIES, 100.0, downstream_level_m) == expected


def test_tailrace_level_of_a_single_family_ignores_the_downstream_level() -> None:
    families = (TailraceFamily(math.nan, (_flat(15.0),)),)
    assert tailrace_level_m(families, 100.0, 50.0) == 15.0


def _hidr_row(**polynomials: float) -> pd.Series:
    return pd.Series(polynomials, dtype=float)


def test_hidr_families_sort_by_reference_level() -> None:
    hreg = _hidr_row(
        numero_polinomios_jusante=2,
        referencia_jusante_1=30.0,
        a0_jusante_1=35.0,
        a1_jusante_1=0.0,
        a2_jusante_1=0.0,
        a3_jusante_1=0.0,
        a4_jusante_1=0.0,
        referencia_jusante_2=10.0,
        a0_jusante_2=15.0,
        a1_jusante_2=0.01,
        a2_jusante_2=0.0,
        a3_jusante_2=0.0,
        a4_jusante_2=0.0,
    )
    families = tailrace_families_from_hidr(hreg, 7)
    assert [f.downstream_reference_level_m for f in families] == [10.0, 30.0]
    assert families[0].segments == (
        TailraceSegment(math.inf, (15.0, 0.01, 0.0, 0.0, 0.0)),
    )


def test_hidr_families_without_a_polynomial_raise() -> None:
    with pytest.raises(ValueError, match="plant 7: hidr.dat declares 0"):
        tailrace_families_from_hidr(_hidr_row(numero_polinomios_jusante=0), 7)


def _polinjus_frames(
    references: list[float | None],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    families = pd.DataFrame(
        [
            {
                "codigo_usina": 7,
                "indice_familia": index,
                "nivel_montante_referencia": reference,
            }
            for index, reference in enumerate(references, start=1)
        ]
    )
    segments = pd.DataFrame(
        [
            {
                "codigo_usina": 7,
                "indice_familia": family,
                "indice_polinomio": segment,
                "limite_superior_vazao_jusante": 100.0 * segment,
                "coeficiente_a0": float(10 * family + segment),
                "coeficiente_a1": 0.0,
                "coeficiente_a2": 0.0,
                "coeficiente_a3": 0.0,
                "coeficiente_a4": 0.0,
            }
            for family in range(1, len(references) + 1)
            for segment in (2, 1)
        ]
    )
    return families, segments


def test_polinjus_families_order_segments_and_reference_levels() -> None:
    families = tailrace_families_from_polinjus(*_polinjus_frames([30.0, 10.0]))[7]
    assert [f.downstream_reference_level_m for f in families] == [10.0, 30.0]
    assert [s.coefficients[0] for s in families[0].segments] == [21.0, 22.0]


def test_polinjus_families_with_a_blank_reference_raise() -> None:
    with pytest.raises(ValueError, match="plant 7: a tailrace family has no"):
        tailrace_families_from_polinjus(*_polinjus_frames([30.0, None]))
