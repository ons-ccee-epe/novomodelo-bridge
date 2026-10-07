"""The source model anticipated thermal dispatch (``adterm.dat``) reader.

Maps the source model's per-(thermal, lag, patamar) MW dispatch table into the
single-MW-per-delivery-stage form Cobre's anticipated thermal model expects.

The source model side
-----------
``adterm.dat`` records dispatch already committed for the first few
study stages on GNL / fuel-constrained thermals.  The :class:`Adterm`
``despachos`` DataFrame has columns:

* ``codigo_usina`` — the source model thermal code
* ``nome_usina`` — plant name (informational)
* ``lag`` — 1-based stage offset.  ``lag=1`` delivers at the first study
  stage (Cobre stage 0), ``lag=2`` at stage 1, etc.
* ``patamar`` — 1-based block index within the delivery stage
* ``valor`` — committed MW for that block

Activation is gated by ``dger.despacho_antecipado_gnl``: when the flag is 0 (or absent)
the source model ignores adterm.dat entirely; cobre-bridge mirrors that by treating
every thermal as non-anticipated.

Cobre side
----------
For each anticipated thermal the LP needs a single MW value per delivery stage
(``InitialConditions.past_anticipated_commitments``).  The MW is held
*uniformly across blocks* of the delivery stage by the fishing row, so the
right aggregation from the source model's per-block values is the
block-duration-weighted mean:

    MW_eq = Σ_b f_b · MW_b

where ``f_b`` is the block fraction at the delivery stage (from
``patamar.dat``).  This preserves the total committed MWh exactly while
respecting the cobre LP's constant-MW-per-stage convention.

**Cobre seeding (>= 0.7.0).**  Cobre honours non-zero pre-horizon
seeds: the always-active anticipated "fishing" equality pins
generation to the committed MW at each delivery stage (``slot 0`` may hold a non-zero
seed at stage 0), so passing the true committed MW through reproduces the source model's
pre-commitment.  The semantic validator only rejects a committed value outside the
plant's static generation bounds ``[min_mw, max_mw]``.  ``convert_initial_conditions``
therefore writes these block-weighted MWs directly, clamping into the bounds (with a
warning) on the rare out-of-range value; this module just returns the *true*
block-weighted MWs — the bounds policy lives at the conversion site, not here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from inewave.newave import Dger, Patamar

if TYPE_CHECKING:
    from cobre_bridge.newave.case import NewaveCase

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnticipatedDispatch:
    """Resolved anticipated-dispatch payload for one thermal plant.

    Attributes
    ----------
    lead_stages:
        Number of future study stages with a pre-committed dispatch.
        Equals the maximum ``lag`` observed for this plant in adterm.dat.
    values_mw:
        Block-duration-weighted mean MW per delivery stage.  Index ``k`` (0-based)
        corresponds to delivery stage ``k`` (Cobre 0-based), i.e. The source model ``lag
        = k + 1``.  Length equals ``lead_stages``.
    """

    lead_stages: int
    values_mw: list[float]


def is_anticipated_dispatch_enabled(dger: Dger) -> bool:
    """Return True when ``dger.despacho_antecipado_gnl`` activates adterm.

    The source model treats the field as a boolean flag (``0`` = ignore adterm.dat, any
    non-zero value = honour it).  When the attribute is missing or ``None`` we default
    to ``False`` (safer; matches the historical cobre-bridge behaviour of producing no
    ``anticipated_config``).
    """
    raw = getattr(dger, "despacho_antecipado_gnl", None)
    if raw is None:
        return False
    try:
        return int(raw) != 0
    except (TypeError, ValueError):
        return False


def _block_fractions_by_stage(
    patamar: Patamar,
    start_year: int,
    start_month: int,
    n_stages: int,
) -> list[dict[int, float]]:
    """Return per-stage ``{patamar_1based: fraction}`` lookups.

    Falls back to a uniform 1/N split when patamar.dat lacks a row for
    the (year, month, patamar) the delivery stage needs — same
    convention used by the temporal converter.
    """
    num_pat: int = patamar.numero_patamares or 1
    df = patamar.duracao_mensal_patamares
    lookup: dict[tuple[int, int, int], float] = {}
    last_year: int | None = None
    if df is not None and not df.empty:
        for _, row in df.iterrows():
            y = int(row["data"].year)
            m = int(row["data"].month)
            p = int(row["patamar"])
            lookup[(y, m, p)] = float(row["valor"])
            if last_year is None or y > last_year:
                last_year = y

    stages: list[dict[int, float]] = []
    y, m = start_year, start_month
    for _ in range(n_stages):
        frac_by_pat: dict[int, float] = {}
        for p in range(1, num_pat + 1):
            f = lookup.get((y, m, p))
            if f is None and last_year is not None:
                f = lookup.get((last_year, m, p))
            if f is None:
                f = 1.0 / num_pat
            frac_by_pat[p] = f
        stages.append(frac_by_pat)
        m += 1
        if m > 12:
            m = 1
            y += 1
    return stages


def read_anticipated_dispatch(
    case: NewaveCase,
) -> dict[int, AnticipatedDispatch]:
    """Return ``{codigo_usina: AnticipatedDispatch}`` from adterm.dat.

    Returns an empty dict when:

    * ``dger.despacho_antecipado_gnl`` is ``0`` / absent / unparseable
      (the source model ignores adterm in this case);
    * ``adterm.dat`` is not present in the case directory;
    * adterm.dat exists but contains no rows.

    Each thermal's ``lead_stages`` equals the maximum ``lag`` seen for that plant;
    entries with ``valor`` all zero are still included (the source model may commit a
    literal zero MW), so the caller cannot use "has an adterm row" as a proxy for "has
    non-trivial dispatch".

    The MW per delivery stage is block-duration-weighted (see module
    docstring).  Plants whose maximum lag exceeds the study horizon are
    truncated to the horizon length and a warning is emitted — Cobre's
    semantic validator otherwise rejects ``entry_stage_id +
    lead_stages > n_stages``. This in-study-only ceiling matches the source
    model's own data availability; the sibling conversion track's
    post-horizon anticipated-delivery surface
    (:mod:`cobre_bridge.decomp.converters.anticipated`) has no analog here — a
    registered asymmetry, not an unported feature.
    """
    dger = case.dger
    if not is_anticipated_dispatch_enabled(dger):
        _LOG.debug("despacho_antecipado_gnl=0 (or absent); ignoring adterm.dat.")
        return {}

    adterm_path: Path | None = case.files.adterm
    if adterm_path is None or not adterm_path.exists():
        _LOG.info(
            "despacho_antecipado_gnl is on but adterm.dat is missing; "
            "no anticipated thermals will be emitted."
        )
        return {}

    despachos: pd.DataFrame | None = case.adterm.despachos
    if despachos is None or despachos.empty:
        return {}

    patamar = case.patamar
    # Need the case's calendar start to weight by the right month's blocks.
    horizon = case.horizon
    start_month = horizon.start_month
    start_year = horizon.start_year
    # Mirror the temporal converter's horizon calculation.
    study_months = horizon.study_months
    pos_months = horizon.pos_months
    n_stages_total = study_months + pos_months

    block_fractions = _block_fractions_by_stage(
        patamar, start_year, start_month, n_stages_total
    )

    result: dict[int, AnticipatedDispatch] = {}
    for code, group in despachos.groupby("codigo_usina"):
        code_int = int(code)
        max_lag = int(group["lag"].max())
        if max_lag <= 0:
            continue
        if max_lag > n_stages_total:
            _LOG.warning(
                "adterm.dat plant code=%d has lag=%d exceeding the %d-stage "
                "horizon; truncating to %d (cobre would reject otherwise).",
                code_int,
                max_lag,
                n_stages_total,
                n_stages_total,
            )
            max_lag = n_stages_total
        values_mw: list[float] = []
        for lag in range(1, max_lag + 1):
            sub = group[group["lag"] == lag]
            stage_idx = lag - 1
            if stage_idx < len(block_fractions):
                fracs = block_fractions[stage_idx]
            else:
                fracs = {}
            mw_eq = 0.0
            total_w = 0.0
            for _, row in sub.iterrows():
                p = int(row["patamar"])
                mw = float(row["valor"])
                w = fracs.get(p, 0.0)
                mw_eq += w * mw
                total_w += w
            # Renormalise if patamar weights don't sum to 1 (defensive —
            # patamar.dat fractions should already sum to 1 per month).
            if total_w > 0.0 and abs(total_w - 1.0) > 1e-6:
                mw_eq /= total_w
            values_mw.append(mw_eq)
        result[code_int] = AnticipatedDispatch(lead_stages=max_lag, values_mw=values_mw)

    return result
