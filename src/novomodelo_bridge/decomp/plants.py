"""Canonical definition of "which ``UH`` rows become hydros in the Novomodelo case".

Every ``UH`` row falls in one of four sets:

- *dormant*: the row is marked ``NW``, with nulls read as blank and blanks
  stripped. DECOMP does not operate the plant but NEWAVE does, so the boundary
  cuts carry its terms; the case converts it as a hydro that enters after the
  horizon. The initial volume is not read: a dormant row may carry one.
- *operated*: the row carries an initial volume and is not dormant. DECOMP
  operates it, and the case converts it as a hydro.
- *converted*: the operated and the dormant rows. Identity, topology, state
  and inflow data are written for every converted plant; operation data only
  for the operated ones.
- *coupling-only*: the row has no initial volume and is not dormant. DECOMP
  keeps it only for the coupling with NEWAVE, so the case leaves it out.

These helpers are the one home for that rule. They are pure functions over an
already-read, non-empty ``UH`` DataFrame (``as_frame(dadger.uh(df=True))``); they
do **not** read files, and handling a missing or empty frame stays with each
caller.
"""

from __future__ import annotations

import pandas as pd


def _is_dormant(uh: pd.DataFrame) -> pd.Series:
    if "configuracao_newave" not in uh.columns:
        return pd.Series(False, index=uh.index)
    return uh["configuracao_newave"].astype(str).str.strip() == "NW"


def operated_uh(uh: pd.DataFrame) -> pd.DataFrame:
    """The non-dormant rows carrying an initial volume, in the frame's order."""
    return uh[uh["volume_inicial"].notna() & ~_is_dormant(uh)]


def converted_uh(uh: pd.DataFrame) -> pd.DataFrame:
    """The operated and the dormant rows, in the frame's order."""
    return uh[uh["volume_inicial"].notna() | _is_dormant(uh)]


def converted_codes(uh: pd.DataFrame) -> tuple[int, ...]:
    """The ascending codes of the operated and the dormant rows."""
    return tuple(sorted(int(c) for c in converted_uh(uh)["codigo_usina"]))


def coupling_only_codes(uh: pd.DataFrame) -> tuple[int, ...]:
    """The ascending codes of the non-dormant rows without an initial volume."""
    rows = uh[uh["volume_inicial"].isna() & ~_is_dormant(uh)]
    return tuple(sorted(int(c) for c in rows["codigo_usina"]))


def dormant_codes(uh: pd.DataFrame) -> tuple[int, ...]:
    """The ascending codes of the rows marked ``NW``.

    A null or blank marker is not ``NW``, and a frame without that column has no
    dormant rows. The volume is not read: a dormant row may carry one.
    """
    if "configuracao_newave" not in uh.columns:
        return ()
    return tuple(sorted(int(c) for c in uh[_is_dormant(uh)]["codigo_usina"]))
