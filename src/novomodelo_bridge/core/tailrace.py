"""Source-model-agnostic tailrace-curve table builder shared by both
conversion tracks.

Builds the ``tailrace_curves`` Arrow table from a plant's downstream-level
curve families (segments keyed by ``(codigo_usina, indice_familia,
indice_polinomio)``) — the shared core the source-model track's
``convert_tailrace_curves`` (:mod:`cobre_bridge.newave.converters.tailrace`) and the
DECOMP track's ``convert_tailrace_curves`` (:mod:`cobre_bridge.decomp.converters.fpha`)
both call, since ``idecomp.libs.UsinasHidreletricas`` exposes the identical
column layout as the source model's ``polinjus``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa

if TYPE_CHECKING:
    from collections.abc import Callable

_LOG = logging.getLogger(__name__)

# Arrow schema required by cobre's `parse_tailrace_curves`: only
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
    code to its dense 0-based cobre id and raises ``KeyError`` for a code absent
    from the id map (filtered fictitious plants, etc.) — those segments are
    dropped. Returns ``None`` when there are no families/segments or none map to
    a converted hydro (meaning "do not write the optional file"; cobre's FPHA
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

    # Deterministic output order; cobre re-sorts by the same integer keys.
    hydro_ids = [code_to_hydro[int(code)] for code in merged["codigo_usina"].tolist()]
    merged = merged.assign(hydro_id=hydro_ids).sort_values(
        ["hydro_id", "indice_familia", "indice_polinomio"], kind="stable"
    )

    # The downstream reference level is nullable; coerce any NaN to None so cobre's
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
