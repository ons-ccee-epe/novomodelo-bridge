"""Canonical definition of "which hydro plants are in the Cobre case".

The source model's ``confhd.dat`` lists every hydro the study knows about, tagged by
``usina_existente`` (``EX`` and ``EE`` = in operation, ``NE``/``NC`` = not yet
built) and including *fictitious* accounting plants. Only the in-service,
non-fictitious plants become LP variables in Cobre.

A plant is identified as **fictitious structurally**, not by the ``FICT.`` name
prefix: a fictitious accounting twin shares its inflow gauge (``posto``) with a
real generating plant and itself has zero productivity (``produtibilidade_especifica ==
0``). In the source model's full decks each ``FICT.*`` plant pairs with its real twin on
the same posto, so the structural rule flags exactly the same plants as the prefix — but
it is robust to the naming convention and, crucially, a plant orphaned by a
REE/subsystem reduction (no generating posto-mate left) is **not** fictitious and stays
in the LP as a ρ=0 routing/regulation reservoir. This unifies "is it fictitious" with
"should we keep an orphaned twin" into one rule.

These helpers are the one home for that filter (historically copy-pasted at ~13
sites). They are pure functions over an already-read ``Confhd.usinas`` DataFrame
plus the ``Hidr.cadastro`` (for the productivity test); they do **not** read
files.
"""

from __future__ import annotations

import pandas as pd

# Productivity at or below this is treated as "no generation" (the source model FICT
# accounting plants carry produtibilidade_especifica == 0 exactly).
_RHO_COL = "produtibilidade_especifica"

#: ``usina_existente`` values that put a plant in the LP from stage 0. ``EE``
#: ("existente em expansão") is already in operation and only gains machines
#: later, so narrowing this to ``EX`` drops a real plant and silently rewires
#: the cascade around it (:func:`existing_hydros`).
IN_SERVICE_STATUSES = ("EX", "EE")


def _is_in_service(confhd_df: pd.DataFrame) -> pd.Series:
    """Boolean mask over *confhd_df* of the rows whose status is in service."""
    return (
        confhd_df["usina_existente"].astype(str).str.strip().isin(IN_SERVICE_STATUSES)
    )


def existing_hydros(confhd_df: pd.DataFrame) -> pd.DataFrame:
    """Return the in-service rows (fictitious ones included)."""
    return confhd_df[_is_in_service(confhd_df)]


def fictitious_codes(confhd_df: pd.DataFrame, cadastro: pd.DataFrame) -> set[int]:
    """Return codes of existing plants that are fictitious accounting nodes.

    Structural rule: a plant is fictitious iff it generates nothing
    (``produtibilidade_especifica == 0``) **and** shares its ``posto`` with
    another existing plant that does generate (ρ > 0). The generating twin owns
    the inflow; the ρ=0 twin is the accounting node.

    A ρ=0 plant whose posto carries no generator is **not** fictitious — it is a real
    regulation reservoir (the source model has these, e.g. GUARAPIRANGA/BILLINGS) or a
    twin orphaned by a reduction (e.g. ``FICT.MAUA`` once ``MAUA`` is filtered out), and
    it stays in the LP.

    Returns an empty set when the productivity (``cadastro``) or ``posto`` data
    needed for the test is unavailable.
    """
    if (
        "usina_existente" not in confhd_df.columns
        or "posto" not in confhd_df.columns
        or cadastro.empty
        or _RHO_COL not in cadastro.columns
    ):
        return set()  # data needed for the structural test is unavailable
    existing = existing_hydros(confhd_df)
    rho: dict[int, float] = {}
    posto: dict[int, int] = {}
    for _, row in existing.iterrows():
        code = int(row["codigo_usina"])
        if code not in cadastro.index:
            continue  # no geometry → can't classify; treat as real
        rho[code] = float(cadastro.loc[code, _RHO_COL])
        posto[code] = int(row["posto"])
    generating_postos = {posto[c] for c, r in rho.items() if r > 0.0}
    return {c for c, r in rho.items() if r == 0.0 and posto[c] in generating_postos}


def filling_hydro_codes(
    confhd_df: pd.DataFrame, exph_df: pd.DataFrame | None
) -> set[int]:
    """Return codes of ``NE`` plants that carry an ``exph`` dead-volume filling row.

    A future (``usina_existente == "NE"``) plant is *admitted* into the Cobre LP iff
    its ``codigo_usina`` appears in *exph_df* with a non-null
    ``data_inicio_enchimento`` — the filling schedule lives on the first ``exph`` row
    per plant, while unit rows carry ``NaT`` there. ``NC`` plants and bare ``NE``
    plants (no filling row) are **not** admitted.

    Returns an empty set when *exph_df* is ``None``/empty or when the columns needed
    for the test (``usina_existente`` on confhd, ``codigo_usina`` /
    ``data_inicio_enchimento`` on exph) are unavailable — never raises, mirroring
    :func:`fictitious_codes`'s defensive empty-set behaviour.
    """
    if (
        exph_df is None
        or exph_df.empty
        or "usina_existente" not in confhd_df.columns
        or "codigo_usina" not in exph_df.columns
        or "data_inicio_enchimento" not in exph_df.columns
    ):
        return set()
    ne_codes = {
        int(code)
        for code in confhd_df.loc[confhd_df["usina_existente"] == "NE", "codigo_usina"]
    }
    filling_rows = exph_df.loc[
        exph_df["data_inicio_enchimento"].notna(), "codigo_usina"
    ]
    filling = {int(code) for code in filling_rows}
    return ne_codes & filling


def active_hydros(
    confhd_df: pd.DataFrame,
    cadastro: pd.DataFrame,
    exph_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the existing, non-fictitious hydro rows that enter the Cobre LP.

    The in-service rows (:data:`IN_SERVICE_STATUSES`) minus
    :func:`fictitious_codes`, in confhd
    declaration order (the id-map relies on this order to assign 0-based Cobre
    hydro IDs). *cadastro* (``Hidr.cadastro``) is required for the structural
    fictitious test; with the data unavailable, no plant is classified
    fictitious and all existing plants are returned.

    When *exph_df* (``Exph().expansoes`` or ``None``) is supplied, the ``NE`` plants
    that carry a dead-volume filling row (:func:`filling_hydro_codes`) are also
    admitted, each kept at its **confhd declaration position** (a single boolean mask
    over *confhd_df*, never a concat/append, so id assignment stays deterministic).
    With ``exph_df is None`` the result is the in-service set alone — no ``NE``
    plant is admitted.
    """
    fict = fictitious_codes(confhd_df, cadastro)
    in_service = _is_in_service(confhd_df)
    if fict:
        in_service = in_service & ~confhd_df["codigo_usina"].isin(fict)
    filling = filling_hydro_codes(confhd_df, exph_df)
    keep = in_service | confhd_df["codigo_usina"].isin(filling)
    return confhd_df[keep]


def active_hydro_codes(
    confhd_df: pd.DataFrame,
    cadastro: pd.DataFrame,
    exph_df: pd.DataFrame | None = None,
) -> list[int]:
    """Return ``codigo_usina`` of the active hydros in declaration order."""
    return [
        int(code)
        for code in active_hydros(confhd_df, cadastro, exph_df)["codigo_usina"]
    ]
