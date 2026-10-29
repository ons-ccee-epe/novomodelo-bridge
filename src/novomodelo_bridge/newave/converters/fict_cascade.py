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
``newave.converters.hydro.convert_hydros`` and the accumulated-productivity
helpers in ``newave.converters.constraints`` consume it so the LP topology and
the cascade accounting agree.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

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
    truncated:
        ``True`` when the source model ends the plant's cascade (``codigo_usina_jusante
        == 0``) and the link to ``downstream_code`` comes from the fictitious plant
        sharing its posto (rule 3) — the source model's coupling between two REEs,
        where the stored energy below this plant is accounted in the fictitious
        plant's REE, not the plant's own.
    counterpart_fict:
        The fictitious plant sharing this plant's posto (its accounting twin in
        another REE), or ``None``.
    counterpart_downstream:
        The first real plant on ``counterpart_fict``'s cascade, or ``None``.
    """

    downstream_code: int | None
    fict_chain: tuple[int, ...]
    fict_rho_sum: float
    truncated: bool = False
    counterpart_fict: int | None = None
    counterpart_downstream: int | None = None


def _jusante_code(row: pd.Series) -> int:
    """The row's ``codigo_usina_jusante``, with a missing link read as 0 (terminal)."""
    ds_raw = row.get("codigo_usina_jusante")
    return int(ds_raw) if ds_raw is not None and not pd.isna(ds_raw) else 0


@dataclass(frozen=True)
class _CascadeTopology:
    """The ``confhd.dat`` plants classified for the cascade walk."""

    row_by_code: dict[int, pd.Series]
    real_codes: set[int]
    absent_codes: set[int]
    fict_by_posto: dict[int, int]
    has_posto: bool
    fict_rho_eq: dict[int, float]

    @classmethod
    def classify(
        cls,
        confhd_df: pd.DataFrame,
        cadastro: pd.DataFrame,
        fictitious: AbstractSet[int],
        filling_codes: set[int],
    ) -> _CascadeTopology:
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

        return cls(
            row_by_code=row_by_code,
            real_codes=real_codes,
            absent_codes=absent_codes,
            fict_by_posto=fict_by_posto,
            has_posto=has_posto,
            fict_rho_eq=fict_rho_eq,
        )

    def walk_chain(self, start_code: int) -> tuple[int | None, list[int], float]:
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
            row = self.row_by_code.get(cur_code)
            if row is None:
                return None, fict_codes, rho_sum
            if cur_code in self.real_codes:
                return cur_code, fict_codes, rho_sum
            if cur_code not in self.absent_codes:
                # Fictitious plant — pick up its ρ_eq contribution.
                fict_codes.append(cur_code)
                rho_sum += self.fict_rho_eq.get(cur_code, 0.0)
            # NE/NC and fictitious plants alike are walked through; only their
            # ρ_eq handling differs (NE/NC contributes nothing).
            cur_code = _jusante_code(row)
        return None, fict_codes, rho_sum

    def resolve(self, code: int) -> FictCascadeResolution:
        """Apply the :func:`resolve_cascade` rules to real plant *code*."""
        row = self.row_by_code[code]
        ds_code = _jusante_code(row)
        posto_raw = row.get("posto")
        twin = (
            self.fict_by_posto.get(int(posto_raw))
            if self.has_posto and posto_raw is not None and not pd.isna(posto_raw)
            else None
        )
        twin_downstream = self.walk_chain(twin)[0] if twin is not None else None

        if ds_code != 0:
            if ds_code in self.real_codes:
                # Rule 1 — direct real downstream.
                real_ds, fict_codes, rho_sum = ds_code, [], 0.0
            else:
                # Rule 2 — confhd points to a fictitious or NE/NC plant.  Walk the
                # chain to find the next real plant downstream.
                real_ds, fict_codes, rho_sum = self.walk_chain(ds_code)
            truncated = False
        elif twin is None:
            # Rule 4 — the cascade terminates.
            real_ds, fict_codes, rho_sum = None, [], 0.0
            truncated = False
        else:
            # Rule 3 — physically terminal.  A fictitious plant sharing this
            # plant's posto carries the energy-cascade link (its real twin).
            real_ds, fict_codes, rho_sum = self.walk_chain(twin)
            truncated = True
        return FictCascadeResolution(
            downstream_code=real_ds,
            fict_chain=tuple(fict_codes),
            fict_rho_sum=rho_sum,
            truncated=truncated,
            counterpart_fict=twin,
            counterpart_downstream=twin_downstream,
        )


def _log_bypassed_absent_plants(
    topology: _CascadeTopology, result: dict[int, FictCascadeResolution]
) -> None:
    """Log NE/NC plants that bridged a real-to-real cascade so the operator
    can audit the bypassed mid-cascade links.

    Top-of-cascade NE/NC leaves are intentionally not reported — they have no
    upstream to rewire and would only add noise.
    """
    row_by_code = topology.row_by_code
    rewired_absent: dict[int, list[int]] = defaultdict(list)
    for upstream_code in result:
        ds_seed = _jusante_code(row_by_code[upstream_code])
        if ds_seed == 0 or ds_seed not in topology.absent_codes:
            continue
        # The original confhd link landed on an NE/NC plant; the walker
        # found the next in-service downstream (possibly through more NE/NC or
        # fictitious hops).  Attribute every NE/NC along that path to this
        # upstream so a single log entry summarizes who got rewired.
        cur: int = ds_seed
        seen: set[int] = set()
        while cur not in (0,) and cur not in seen and cur not in topology.real_codes:
            seen.add(cur)
            if cur in topology.absent_codes:
                rewired_absent[cur].append(upstream_code)
            row = row_by_code.get(cur)
            if row is None:
                break
            cur = _jusante_code(row)
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


def resolve_cascade(
    confhd_df: pd.DataFrame,
    cadastro: pd.DataFrame,
    *,
    fictitious: AbstractSet[int],
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
        compute the fictitious plants' ``ρ_eq``.
    fictitious:
        Pre-computed set of fictitious plant codes. The single source of truth
        is ``NewaveCase.fictitious_codes`` (which applies the pumping exemption).
        Required parameter to ensure callers propagate the correct set.
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
    if filling_codes is None:
        filling_codes = set()

    topology = _CascadeTopology.classify(confhd_df, cadastro, fictitious, filling_codes)
    result: dict[int, FictCascadeResolution] = {
        code: topology.resolve(code) for code in topology.real_codes
    }
    _log_bypassed_absent_plants(topology, result)
    return result
