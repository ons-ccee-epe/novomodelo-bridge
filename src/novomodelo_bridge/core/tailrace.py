"""Source-model-agnostic tailrace curves shared by both conversion tracks.

A plant's downstream (tailrace) level is a family of piecewise quartics in the
plant's outflow, one family per reference level of the plant downstream
(segments keyed by ``(codigo_usina, indice_familia, indice_polinomio)``).
``idecomp.libs.UsinasHidreletricas`` exposes the identical column layout as the
source model's ``polinjus``, and ``hidr.dat`` carries the same shape as up to
six unsegmented polynomials, so one core serves both tracks:

- :func:`build_tailrace_table` writes the families as the ``tailrace_curves``
  Arrow table, called by both tracks' ``convert_tailrace_curves``
  (:mod:`novomodelo_bridge.newave.converters.tailrace`,
  :mod:`novomodelo_bridge.decomp.converters.fpha`);
- :class:`TailraceFamily` and :func:`tailrace_level_m` evaluate them, built by
  :func:`tailrace_families_from_polinjus` or
  :func:`tailrace_families_from_hidr`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa

from novomodelo_bridge.core.productivity import evaluate_cota

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_LOG = logging.getLogger(__name__)

# Arrow schema required by novomodelo's `parse_tailrace_curves`: only
# `downstream_reference_level_m` is nullable; every other column is non-nullable
# Int32/Float64.
_TAILRACE_SCHEMA = pa.schema(
    [
        pa.field("hydro_id", pa.int32(), nullable=False),
        pa.field("family_id", pa.int32(), nullable=False),
        pa.field("downstream_reference_level_m", pa.float64(), nullable=True),
        pa.field("segment_id", pa.int32(), nullable=False),
        pa.field("outflow_min_m3s", pa.float64(), nullable=False),
        pa.field("outflow_max_m3s", pa.float64(), nullable=False),
        pa.field("coefficient_0", pa.float64(), nullable=False),
        pa.field("coefficient_1", pa.float64(), nullable=False),
        pa.field("coefficient_2", pa.float64(), nullable=False),
        pa.field("coefficient_3", pa.float64(), nullable=False),
        pa.field("coefficient_4", pa.float64(), nullable=False),
    ]
)


def build_tailrace_table(
    families: pd.DataFrame | None,
    segments: pd.DataFrame | None,
    hydro_id_of: Callable[[int], int],
) -> pa.Table | None:
    """Build the ``tailrace_curves`` table from polinjus ``families``/``segments``.

    The source-model-agnostic core shared by :func:`convert_tailrace_curves`
    (which extracts the two frames from ``case.polinjus``) and the DECOMP
    pipeline (whose ``polinjus.csv`` reads to the identical column layout via
    ``idecomp.libs.UsinasHidreletricas``). *hydro_id_of* maps a source plant
    code to its dense 0-based novomodelo id and raises ``KeyError`` for a code absent
    from the id map (filtered fictitious plants, etc.) — those segments are
    dropped. Returns ``None`` when there are no families/segments or none map to
    a converted hydro (meaning "do not write the optional file"; novomodelo's FPHA
    then falls back to the entity-level tailrace from ``hydros.json``).
    """
    if families is None or segments is None or families.empty or segments.empty:
        return None

    # Attach each segment's family downstream reference level, then remap
    # plant codes to dense 0-based hydro ids, dropping plants absent from the map
    # (filtered fictitious plants, etc.). A left merge keeps every segment row.
    merged = segments.merge(
        families[["codigo_usina", "indice_familia", "nivel_montante_referencia"]],
        on=["codigo_usina", "indice_familia"],
        how="left",
    )

    code_to_hydro: dict[int, int] = {}
    skipped: set[int] = set()
    for raw_code in merged["codigo_usina"].unique():
        code = int(raw_code)
        try:
            code_to_hydro[code] = hydro_id_of(code)
        except KeyError:
            skipped.add(code)
    if skipped:
        _LOG.debug(
            "Skipped %d polinjus plant code(s) absent from the id map: %s",
            len(skipped),
            sorted(skipped),
        )

    merged = merged[merged["codigo_usina"].isin(list(code_to_hydro))]
    if merged.empty:
        return None

    # Deterministic output order; novomodelo re-sorts by the same integer keys.
    hydro_ids = [code_to_hydro[int(code)] for code in merged["codigo_usina"].tolist()]
    merged = merged.assign(hydro_id=hydro_ids).sort_values(
        ["hydro_id", "indice_familia", "indice_polinomio"], kind="stable"
    )

    # The downstream reference level is nullable; coerce any NaN to None so novomodelo's
    # finite-check passes.
    downstream_level = [
        None if pd.isna(v) else float(v)
        for v in merged["nivel_montante_referencia"].tolist()
    ]

    def _f64(column: str) -> pa.Array:
        return pa.array(merged[column].astype(float).tolist(), type=pa.float64())

    def _i32(column: str) -> pa.Array:
        return pa.array(merged[column].astype(int).tolist(), type=pa.int32())

    return pa.table(
        {
            "hydro_id": _i32("hydro_id"),
            "family_id": _i32("indice_familia"),
            "downstream_reference_level_m": pa.array(
                downstream_level, type=pa.float64()
            ),
            "segment_id": _i32("indice_polinomio"),
            "outflow_min_m3s": _f64("limite_inferior_vazao_jusante"),
            "outflow_max_m3s": _f64("limite_superior_vazao_jusante"),
            "coefficient_0": _f64("coeficiente_a0"),
            "coefficient_1": _f64("coeficiente_a1"),
            "coefficient_2": _f64("coeficiente_a2"),
            "coefficient_3": _f64("coeficiente_a3"),
            "coefficient_4": _f64("coeficiente_a4"),
        },
        schema=_TAILRACE_SCHEMA,
    )


#: ``hidr.dat`` holds at most this many downstream polynomials per plant
#: (``a0_jusante_k``..``a4_jusante_k``, ``referencia_jusante_k``).
_HIDR_TAILRACE_POLYNOMIALS = 6


@dataclass(frozen=True)
class TailraceSegment:
    """One piece of a tailrace family, valid up to *outflow_max_m3s*."""

    outflow_max_m3s: float
    coefficients: tuple[float, float, float, float, float]


@dataclass(frozen=True)
class TailraceFamily:
    """A tailrace curve for one reference level of the plant downstream (m)."""

    downstream_reference_level_m: float
    segments: tuple[TailraceSegment, ...]

    def level_m(self, outflow_m3s: float) -> float:
        """The downstream level (m) at *outflow_m3s*.

        Evaluates the first segment whose upper limit reaches the outflow, in
        segment order, and the last segment past every limit.
        """
        segment = next(
            (s for s in self.segments if outflow_m3s <= s.outflow_max_m3s),
            self.segments[-1],
        )
        return evaluate_cota(segment.coefficients, outflow_m3s)


def tailrace_level_m(
    families: Sequence[TailraceFamily],
    outflow_m3s: float,
    downstream_level_m: float | None,
) -> float:
    """A plant's downstream level (m) at *outflow_m3s*.

    *families* must be sorted by ascending reference level. With more than one
    family, the level is interpolated linearly between the two families whose
    reference levels bracket *downstream_level_m*, the level of the plant
    downstream; outside the references it takes the nearest family. ``None``
    (no plant downstream) takes the lowest family.
    """
    if len(families) == 1 or downstream_level_m is None:
        return families[0].level_m(outflow_m3s)
    if downstream_level_m <= families[0].downstream_reference_level_m:
        return families[0].level_m(outflow_m3s)
    for lower, upper in zip(families, families[1:], strict=False):
        if downstream_level_m <= upper.downstream_reference_level_m:
            span = (
                upper.downstream_reference_level_m - lower.downstream_reference_level_m
            )
            weight = (downstream_level_m - lower.downstream_reference_level_m) / span
            return (1.0 - weight) * lower.level_m(outflow_m3s) + weight * upper.level_m(
                outflow_m3s
            )
    return families[-1].level_m(outflow_m3s)


def tailrace_families_from_polinjus(
    families: pd.DataFrame, segments: pd.DataFrame
) -> dict[int, tuple[TailraceFamily, ...]]:
    """``{plant code: families}`` from the ``polinjus`` frames, each plant's
    families sorted by ascending reference level and segments by
    ``indice_polinomio``.

    Raises
    ------
    ValueError
        When a plant with more than one family leaves a reference level blank:
        the families cannot be ordered for interpolation.
    """
    reference = {
        (int(row.codigo_usina), int(row.indice_familia)): row.nivel_montante_referencia
        for row in families.itertuples()
    }
    by_code: dict[int, list[TailraceFamily]] = {}
    ordered = segments.sort_values(
        ["codigo_usina", "indice_familia", "indice_polinomio"], kind="stable"
    )
    for (code, family), rows in ordered.groupby(
        ["codigo_usina", "indice_familia"], sort=False
    ):
        level = reference.get((int(code), int(family)))
        by_code.setdefault(int(code), []).append(
            TailraceFamily(
                downstream_reference_level_m=(
                    math.nan if level is None or pd.isna(level) else float(level)
                ),
                segments=tuple(
                    TailraceSegment(
                        outflow_max_m3s=float(row.limite_superior_vazao_jusante),
                        coefficients=(
                            float(row.coeficiente_a0),
                            float(row.coeficiente_a1),
                            float(row.coeficiente_a2),
                            float(row.coeficiente_a3),
                            float(row.coeficiente_a4),
                        ),
                    )
                    for row in rows.itertuples()
                ),
            )
        )
    return {code: _sorted_families(code, found) for code, found in by_code.items()}


def tailrace_families_from_hidr(
    hreg: pd.Series, code: int
) -> tuple[TailraceFamily, ...]:
    """A plant's ``hidr.dat`` downstream polynomials as unsegmented families,
    sorted by ascending reference level.

    Raises
    ------
    ValueError
        When the row declares no polynomial, or more than one with a blank
        reference level.
    """
    count = hreg.get("numero_polinomios_jusante")
    n_polynomials = 0 if count is None or pd.isna(count) else int(count)
    if not 1 <= n_polynomials <= _HIDR_TAILRACE_POLYNOMIALS:
        raise ValueError(
            f"plant {code}: hidr.dat declares {n_polynomials} downstream "
            f"polynomials; expected 1 to {_HIDR_TAILRACE_POLYNOMIALS}"
        )
    found = []
    for k in range(1, n_polynomials + 1):
        level = hreg.get(f"referencia_jusante_{k}")
        found.append(
            TailraceFamily(
                downstream_reference_level_m=(
                    math.nan if level is None or pd.isna(level) else float(level)
                ),
                segments=(
                    TailraceSegment(
                        outflow_max_m3s=math.inf,
                        coefficients=(
                            float(hreg[f"a0_jusante_{k}"]),
                            float(hreg[f"a1_jusante_{k}"]),
                            float(hreg[f"a2_jusante_{k}"]),
                            float(hreg[f"a3_jusante_{k}"]),
                            float(hreg[f"a4_jusante_{k}"]),
                        ),
                    ),
                ),
            )
        )
    return _sorted_families(code, found)


def _sorted_families(
    code: int, families: list[TailraceFamily]
) -> tuple[TailraceFamily, ...]:
    if len(families) > 1 and any(
        math.isnan(f.downstream_reference_level_m) for f in families
    ):
        raise ValueError(
            f"plant {code}: a tailrace family has no downstream reference "
            "level, so its families cannot be interpolated"
        )
    return tuple(sorted(families, key=lambda f: f.downstream_reference_level_m))
