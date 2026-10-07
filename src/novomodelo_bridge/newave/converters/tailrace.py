"""Convert the source model downstream-level curve families to
``tailrace_curves.parquet``.

The source model's ``polinjus`` cadastro stores, per hydro plant, one or more *families*
of downstream (tailrace) level curves — the downstream level as a piecewise polynomial
in the total outflow (turbined + spilled + lateral). Each family is keyed by the
downstream plant's reference level. cobre's FPHA production model consumes these as
``system/tailrace_curves.parquet`` — piecewise-quartic backwater families — to evaluate
the exact net head.

The mapping reads the ``inewave`` source columns (left) and emits descriptive cobre
output columns (right):

==========================================  ================================
``polinjus`` (UsinasHidreletricas) column   ``tailrace_curves`` column
==========================================  ================================
``codigo_usina``                            ``hydro_id`` (0-based, via id map)
``indice_familia``                          ``family_id``
``nivel_montante_referencia``               ``downstream_reference_level_m``
``indice_polinomio``                        ``segment_id``
``limite_inferior_vazao_jusante``           ``outflow_min_m3s``
``limite_superior_vazao_jusante``           ``outflow_max_m3s``
``coeficiente_a0`` .. ``coeficiente_a4``    ``coefficient_0`` .. ``coefficient_4``
==========================================  ================================

The family/segment indices are the source model's 1-based sequential keys and are passed
through unchanged — cobre treats them as opaque per-plant grouping keys (the d31
reference case likewise uses 1-based ``family_id``/``segment_id``). Only the plant code
is remapped to cobre's dense 0-based id; plants absent from the id map (filtered
fictitious plants, etc.) are skipped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pandas as pd
import pyarrow as pa

from cobre_bridge.core.tailrace import build_tailrace_table

if TYPE_CHECKING:
    from cobre_bridge.newave.case import NewaveCase
    from cobre_bridge.newave.id_map import NewaveIdMap


def convert_tailrace_curves(case: NewaveCase, id_map: NewaveIdMap) -> pa.Table | None:
    """Build ``system/tailrace_curves.parquet`` from the source model's ``polinjus``
    families.

    Parameters
    ----------
    case:
        Parsed the source model case. ``case.polinjus`` supplies the downstream-level
        curve families (``None`` when the case ships no ``polinjus`` file).
    id_map:
        Entity ID map translating the source model plant codes to 0-based Cobre hydro
        IDs.

    Returns
    -------
    pyarrow.Table | None
        A table with the eleven ``tailrace_curves`` columns sorted by
        ``(hydro_id, family_id, segment_id)``, or ``None`` when the case has no
        ``polinjus`` file or no segment maps to a converted hydro. ``None`` means
        "do not write the optional file"; cobre's FPHA then falls back to the
        entity-level tailrace from ``hydros.json``.
    """
    uh = case.polinjus
    if uh is None:
        return None

    families = cast(pd.DataFrame, uh.hidreletrica_curvajusante(df=True))
    segments = cast(
        pd.DataFrame, uh.hidreletrica_curvajusante_polinomio_segmento(df=True)
    )
    return build_tailrace_table(families, segments, id_map.hydro_id)
