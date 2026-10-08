"""Resolve the source model fictitious-plant cascade chains for real plants.

The source model's ``confhd.dat`` encodes the physical water-balance cascade via
``codigo_usina_jusante``.  For some real plants this link is 0 (terminal) or points to a
fictitious plant whose role is purely topological — an accounting entity used only for
energy-cascade routing.

Fictitious plants are identified **structurally** (see
:func:`plants.fictitious_codes`): zero productivity sharing a generating
plant's posto.  Novomodelo-bridge filters them out of the LP, but it must preserve
the cascade connectivity they provide so that:

- Real-plant ``downstream_id`` in ``hydros.json`` points to the next
  real plant in the cascade.
- Accumulated cascade productivities (used by penalty conversion and
  ``EARM = (V − V_min) · ρ_acum`` accounting) correctly include any
  fictitious plant's own productivity along the chain (zero in practice).

This module is the single source of truth for that resolution.  Both
``newave.converters.hydro.convert_hydros`` and
``newave.converters.constraints._build_hydro_downstream_map`` consume it so the
LP topology and the cascade accounting agree.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

from novomodelo_bridge.core.productivity import compute_productivity
from novomodelo_bridge.newave.plants import IN_SERVICE_STATUSES

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FictCascadeResolution:
    """Resolved cascade for a single real plant.

    Attributes
    ----------
    downstream_code:
        The source model code of the next real plant in the cascade, or ``None`` when
        the chain terminates at the sea.
    fict_chain:
        Ordered list of fictitious-plant the source model codes traversed between the
        real plant and ``downstream_code``.  Empty when no fictitious plants lie on the
        path.
    fict_rho_sum:
        Cumulative ``ρ_eq`` of the fictitious plants in ``fict_chain``.  This must be
        folded into the upstream real plant's effective ``ρ_eq`` so that novomodelo's cascade
        sum reproduces the source model's ``produtibilidade_acumulada_calculo_earm``.
        Fictitious plants have ρ = 0 by definition, so this is zero in practice — kept
        for generality.
    """

    downstream_code: int | None
    fict_chain: tuple[int, ...]
    fict_rho_sum: float


def resolve_cascade(
    confhd_df: pd.DataFrame,
    cadastro: pd.DataFrame,
    fictitious: set[int] | None = None,
    filling_codes: set[int] | None = None,
) -> dict[int, FictCascadeResolution]:
    """Resolve the effective cascade downstream of every real (existing) plant.

    Resolution rules, applied per real plant ``P``:

    1. If ``P.codigo_usina_jusante`` is non-zero and points to a real
       plant ``R``, use ``R`` as the downstream.  No fictitious contribution.
    2. If ``P.codigo_usina_jusante`` is non-zero and points to a fictitious
       (or NE/NC) plant, walk its ``codigo_usina_jusante`` chain through any
       further such plants until a real plant or terminal (0) is reached.
       Accumulate the ``ρ_eq`` of every fictitious plant traversed.
    3. If ``P.codigo_usina_jusante`` is 0 (physically terminal) but a
       fictitious plant shares ``P``'s ``posto``, treat that fictitious plant as the
       implicit downstream and walk its chain as in rule 2.  This is the source model's
       convention for energy-cascade accounting on plants that do not physically flow to
       a downstream reservoir (e.g. ``MAUA`` → ``FICT.MAUA`` → ``CAPIVARA``, all on
       posto 57).
    4. Otherwise the cascade terminates: ``downstream_code = None``.

    Parameters
    ----------
    confhd_df:
        ``Confhd.usinas`` — must contain at least ``codigo_usina``,
        ``codigo_usina_jusante``, ``usina_existente`` (and ``posto`` for
        rule 3).
    cadastro:
        ``Hidr.cadastro`` with MODIF.DAT overrides already applied.  Used to
        compute the fictitious plants' ``ρ_eq`` and (when *fictitious* is not
        supplied) to identify them.
    fictitious:
        Pre-computed set of fictitious plant codes (the single source of truth
        is :func:`plants.fictitious_codes`).  Defaults to computing it from
        *confhd_df* + *cadastro* — pass it explicitly to decouple the
        classification (e.g. in tests).
    filling_codes:
        Pre-computed set of ``NE``-with-filling plant codes (the single source
        of truth is :attr:`plants.filling_hydro_codes`).  Such plants are
        admitted as real novomodelo nodes: the walker stops at them and upstream
        plants resolve their ``downstream_code`` to them.  Defaults to ``None``
        (treated as the empty set), which preserves byte-identical output for
        cases with no filling plant.

    Returns
    -------
    dict[int, FictCascadeResolution]
        One entry per real existing plant in ``confhd_df``.
    """
    if fictitious is None:
        from novomodelo_bridge.newave.plants import fictitious_codes

        fictitious = fictitious_codes(confhd_df, cadastro)

    if filling_codes is None:
        filling_codes = set()

    # Classify every row. ``NE``/``NC`` ("não existirá"/"não considerada")
    # plants are out-of-LP topological pass-throughs; fictitious plants are
    # out-of-LP accounting nodes that DO contribute ρ_eq (zero in practice).
    # A plant that is neither is a real LP node — including a ρ=0 reservoir
    # orphaned by a reduction, which is (correctly) not in ``fictitious``.
    row_by_code: dict[int, pd.Series] = {}
    real_codes: set[int] = set()
    absent_codes: set[int] = set()
    fict_by_posto: dict[int, int] = {}
    has_posto = "posto" in confhd_df.columns
    for _, row in confhd_df.iterrows():
        code = int(row["codigo_usina"])
        status = str(row["usina_existente"]).strip()
        row_by_code[code] = row
        if code in filling_codes:
            # ``NE``-with-filling plants are admitted as real novomodelo nodes: the
            # walker must stop at them and upstream plants resolve to them.
            # They are neither absent (topological pass-through) nor fictitious.
            real_codes.add(code)
            continue
        if status not in IN_SERVICE_STATUSES:
            absent_codes.add(code)
            continue
        if code in fictitious:
            if has_posto and not pd.isna(row["posto"]):
                fict_by_posto.setdefault(int(row["posto"]), code)
        else:
            real_codes.add(code)

    fict_rho_eq: dict[int, float] = {}
    for code in fictitious:
        if code in absent_codes:
            continue
        fict_rho_eq[code] = (
            float(compute_productivity(cadastro.loc[code]))
            if code in cadastro.index
            else 0.0
        )

    def _walk_chain(
        start_code: int,
    ) -> tuple[int | None, list[int], float]:
        """Walk transparently through fictitious and absent (NE/NC) plants.

        Returns ``(real_downstream, fict_codes_traversed, rho_sum)``. Stops when
        reaching a real plant (returned), terminal (None), an unknown code, or when a
        cycle is detected (defensive — should not occur in well-formed the source model
        cases).  Fictitious plants accumulate their ``ρ_eq`` into ``rho_sum`` and appear
        in ``fict_codes_traversed``; NE/NC plants are skipped silently as topological
        pass-throughs.
        """
        fict_codes: list[int] = []
        rho_sum = 0.0
        cur_code: int | None = start_code
        visited: set[int] = set()
        while cur_code is not None and cur_code != 0:
            if cur_code in visited:
                return None, fict_codes, rho_sum
            visited.add(cur_code)
            row = row_by_code.get(cur_code)
            if row is None:
                return None, fict_codes, rho_sum
            if cur_code in real_codes:
                return cur_code, fict_codes, rho_sum
            if cur_code not in absent_codes:
                # Fictitious plant — pick up its ρ_eq contribution.
                fict_codes.append(cur_code)
                rho_sum += fict_rho_eq.get(cur_code, 0.0)
            # NE/NC and fictitious plants alike are walked through; only their
            # ρ_eq handling differs (NE/NC contributes nothing).
            ds_raw = row.get("codigo_usina_jusante")
            cur_code = int(ds_raw) if ds_raw is not None and not pd.isna(ds_raw) else 0
        return None, fict_codes, rho_sum

    result: dict[int, FictCascadeResolution] = {}
    for code in real_codes:
        row = row_by_code[code]
        ds_raw = row.get("codigo_usina_jusante")
        ds_code = int(ds_raw) if ds_raw is not None and not pd.isna(ds_raw) else 0

        if ds_code != 0:
            if ds_code in real_codes:
                # Rule 1 — direct real downstream.
                result[code] = FictCascadeResolution(
                    downstream_code=ds_code,
                    fict_chain=(),
                    fict_rho_sum=0.0,
                )
                continue
            # Rule 2 — confhd points to a fictitious or NE/NC plant.  Walk the
            # chain to find the next real plant downstream.
            real_ds, fict_codes, rho_sum = _walk_chain(ds_code)
            result[code] = FictCascadeResolution(
                downstream_code=real_ds,
                fict_chain=tuple(fict_codes),
                fict_rho_sum=rho_sum,
            )
            continue

        # Rule 3 — physically terminal.  A fictitious plant sharing this
        # plant's posto carries the energy-cascade link (its real twin).
        posto_raw = row.get("posto")
        fict_code = (
            fict_by_posto.get(int(posto_raw))
            if posto_raw is not None and not pd.isna(posto_raw)
            else None
        )
        if fict_code is None:
            result[code] = FictCascadeResolution(
                downstream_code=None,
                fict_chain=(),
                fict_rho_sum=0.0,
            )
            continue
        real_ds, fict_codes, rho_sum = _walk_chain(fict_code)
        result[code] = FictCascadeResolution(
            downstream_code=real_ds,
            fict_chain=tuple(fict_codes),
            fict_rho_sum=rho_sum,
        )

    # Log NE/NC plants that bridged a real-to-real cascade so the operator
    # can audit the bypassed mid-cascade links.  Top-of-cascade NE/NC
    # leaves are intentionally not reported — they have no upstream to
    # rewire and would only add noise.
    rewired_absent: dict[int, list[int]] = defaultdict(list)
    for upstream_code in result:
        ds_raw = row_by_code[upstream_code].get("codigo_usina_jusante")
        ds_seed = int(ds_raw) if ds_raw is not None and not pd.isna(ds_raw) else 0
        if ds_seed == 0 or ds_seed not in absent_codes:
            continue
        # The original confhd link landed on an NE/NC plant; the walker
        # found the next in-service downstream (possibly through more NE/NC or
        # fictitious hops).  Attribute every NE/NC along that path to this
        # upstream so a single log entry summarizes who got rewired.
        cur: int = ds_seed
        seen: set[int] = set()
        while cur not in (0,) and cur not in seen and cur not in real_codes:
            seen.add(cur)
            if cur in absent_codes:
                rewired_absent[cur].append(upstream_code)
            row = row_by_code.get(cur)
            if row is None:
                break
            ds_inner = row.get("codigo_usina_jusante")
            cur = int(ds_inner) if ds_inner is not None and not pd.isna(ds_inner) else 0
    for absent_code, upstreams in rewired_absent.items():
        status = str(row_by_code[absent_code]["usina_existente"]).strip()
        absent_name = str(row_by_code[absent_code]["nome_usina"]).strip()
        upstream_names = [str(row_by_code[u]["nome_usina"]).strip() for u in upstreams]
        logger.info(
            "Bypassing %s plant '%s' (code %d): rewired %d upstream(s) %s "
            "to its downstream in the cascade.",
            status,
            absent_name,
            absent_code,
            len(upstreams),
            upstream_names,
        )

    return result
