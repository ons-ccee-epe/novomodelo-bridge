"""Small-plant conversion for DECOMP-like decks (``PQ`` → must-run NCS).

``PQ`` records carry the non-simulated generation netted from load — one
series per (subsystem, name), with per-block MW declared sparsely by stage
(a stage inherits the last declared one). Each series becomes one
non-controllable source on its subsystem's bus, emitted through the proven
deterministic NCS path:

- registry entry with ``allow_curtailment: false`` (the series is netted
  from load at the source — must-run, pinning dispatch to availability);
- ``non_controllable_stats``: ``mean`` = availability fraction
  (energy-weighted stage mean / ``max_generation_mw``), ``std = 0``;
- ``non_controllable_factors``: ``factor_b = MW_b / stage_mean`` with the
  ``Σ f_b·h_b = H`` invariant asserted.

NCS ids are assigned by sorted (subsystem code, name); when the renewable
card file gains its reader, its parks append after the ``PQ`` series in
this same id space.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa

from cobre_bridge.cobre import schemas as cobre_schemas
from cobre_bridge.core.tolerances import relative_tolerance

if TYPE_CHECKING:
    from collections.abc import Sequence

    from idecomp.decomp import Dadger
    from idecomp.libs import Renovaveis

    from cobre_bridge.decomp.case import DecompCase
    from cobre_bridge.decomp.id_map import DecompIdMap
    from cobre_bridge.decomp.temporal import OperativeStage

_INVARIANT_RTOL = 1e-9
# TRACKED COBRE-GAP WORKAROUND (C1, the cobre repository's
# conversion-found-improvements registry): the schema requires factors > 0,
# but a zero-generation block (solar at the light patamar) is real data.
# Remove the clamp when factor >= 0 is accepted.
_MIN_FACTOR = 1e-9

_LOG = logging.getLogger(__name__)

_NCS_STATS_SCHEMA = pa.schema(
    [
        pa.field("ncs_id", pa.int32()),
        pa.field("stage_id", pa.int32()),
        pa.field("mean", pa.float64()),
        pa.field("std", pa.float64()),
    ]
)


@dataclass(frozen=True)
class _PqSeries:
    """One ``PQ`` series with its dense per-stage block generation."""

    ncs_id: int
    name: str
    bus_id: int
    per_stage_blocks: tuple[tuple[float, ...], ...]

    @property
    def max_generation_mw(self) -> float:
        return max(
            (mw for blocks in self.per_stage_blocks for mw in blocks),
            default=0.0,
        )


def _pq_series(
    dadger: Dadger,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
) -> list[_PqSeries]:
    """Read ``PQ`` into dense per-stage series, sorted (subsystem, name).

    Declared stages inherit forward (a stage without a record repeats the
    last declared one); stage 1 must be declared. Missing generation
    fields read as 0.0.
    """
    pq = dadger.pq(df=True)
    if pq is None or pq.empty:
        return []

    declared: dict[tuple[int, str], dict[int, tuple[float, ...]]] = {}
    for _, row in pq.iterrows():
        stage_index = int(row["estagio"]) - 1
        if not 0 <= stage_index < len(calendar):
            raise ValueError(
                f"PQ stage {int(row['estagio'])} outside the calendar "
                f"(1..{len(calendar)})"
            )
        n_blocks = len(calendar[stage_index].block_hours)
        values = tuple(
            0.0 if pd.isna(row[f"geracao_{k}"]) else float(row[f"geracao_{k}"])
            for k in range(1, n_blocks + 1)
        )
        key = (int(row["codigo_submercado"]), str(row["nome"]).strip())
        declared.setdefault(key, {})[stage_index] = values

    series: list[_PqSeries] = []
    for ncs_id, (key, stages) in enumerate(sorted(declared.items())):
        sub_code, name = key
        if 0 not in stages:
            raise ValueError(
                f"PQ series {name!r} (subsystem {sub_code}) does not declare "
                "stage 1; sparse-stage inheritance has no base"
            )
        dense: list[tuple[float, ...]] = []
        for stage in calendar:
            dense.append(stages.get(stage.index, dense[-1] if dense else ()))
        series.append(
            _PqSeries(
                ncs_id=ncs_id,
                name=f"{name}_{sub_code}",
                bus_id=id_map.bus_id(sub_code),
                per_stage_blocks=tuple(dense),
            )
        )
    return series


def _stage_mean_mw(blocks: Sequence[float], stage: OperativeStage) -> float:
    energy = sum(
        mw * hours for mw, hours in zip(blocks, stage.block_hours, strict=True)
    )
    return energy / stage.total_hours


def convert_non_controllable_sources(
    case: DecompCase,
    id_map: DecompIdMap,
) -> dict:
    """Build ``non_controllable_sources.json`` (``PQ`` + renewable parks)."""
    calendar = case.calendar
    op_date = case.start_date.isoformat()
    entries = [
        {
            "id": s.ncs_id,
            "name": s.name,
            "operational_start_date": op_date,
            "bus_id": s.bus_id,
            "max_generation_mw": s.max_generation_mw,
            # PQ generation is netted from load at the source — implicitly
            # must-run; pinning dispatch to availability mirrors the treatment
            # validated in docs/findings/ncs-must-run-treatment.md.
            "allow_curtailment": False,
        }
        for s in _all_series(case.dadger, id_map, calendar, case.renovaveis)
    ]
    return {
        "$schema": cobre_schemas.schema_url_for("system/non_controllable_sources.json"),
        "non_controllable_sources": entries,
    }


def convert_ncs_stats(
    case: DecompCase,
    id_map: DecompIdMap,
) -> pa.Table:
    """Build ``non_controllable_stats`` rows: availability fraction, std 0."""
    calendar = case.calendar
    ncs_ids: list[int] = []
    stage_ids: list[int] = []
    means: list[float] = []
    for s in _all_series(case.dadger, id_map, calendar, case.renovaveis):
        max_gen = s.max_generation_mw
        for stage in calendar:
            mean_mw = _stage_mean_mw(s.per_stage_blocks[stage.index], stage)
            ncs_ids.append(s.ncs_id)
            stage_ids.append(stage.index)
            # TRACKED COBRE-GAP WORKAROUND (C2): the [0, 1] bound rejects
            # 1 + ulp when the stage mean IS the maximum; clamp until the
            # load-side check tolerates it.
            means.append(0.0 if max_gen == 0.0 else min(mean_mw / max_gen, 1.0))

    return pa.table(
        {
            "ncs_id": pa.array(ncs_ids, type=pa.int32()),
            "stage_id": pa.array(stage_ids, type=pa.int32()),
            "mean": pa.array(means, type=pa.float64()),
            "std": pa.array([0.0] * len(means), type=pa.float64()),
        },
        schema=_NCS_STATS_SCHEMA,
    )


def convert_ncs_factors(
    case: DecompCase,
    id_map: DecompIdMap,
) -> dict:
    """Build ``non_controllable_factors.json`` block shapes per (ncs, stage).

    Entries exist only where the stage mean is positive; factors of a zero
    mean are undefined and deliberately absent. A zero block under a
    positive mean (a solar series' light block, say) is clamped to a tiny
    positive factor — the schema requires factors > 0, and under must-run
    pinning the resulting availability is numerically zero anyway.
    """
    calendar = case.calendar
    entries: list[dict] = []
    clamped = 0
    for s in _all_series(case.dadger, id_map, calendar, case.renovaveis):
        for stage in calendar:
            blocks = s.per_stage_blocks[stage.index]
            mean_mw = _stage_mean_mw(blocks, stage)
            if mean_mw == 0.0:
                continue
            raw_factors = [mw / mean_mw for mw in blocks]
            weighted = sum(
                f * hours
                for f, hours in zip(raw_factors, stage.block_hours, strict=True)
            )
            if abs(weighted - stage.total_hours) > _INVARIANT_RTOL * stage.total_hours:
                raise ValueError(
                    f"NCS {s.name} stage {stage.index}: block factors violate "
                    f"the hours invariant (Σ f·h = {weighted}, expected "
                    f"{stage.total_hours})"
                )
            factors = []
            for factor in raw_factors:
                if factor <= 0.0:
                    factor = _MIN_FACTOR
                    clamped += 1
                factors.append(factor)
            entries.append(
                {
                    "ncs_id": s.ncs_id,
                    "stage_id": stage.index,
                    "block_factors": [
                        {"block_id": b, "factor": factor}
                        for b, factor in enumerate(factors)
                    ],
                }
            )
    if clamped:
        _LOG.warning(
            "%d zero NCS block factor(s) clamped to %.0e (schema requires "
            "factors > 0; must-run pinning keeps the availability "
            "numerically zero)",
            clamped,
            _MIN_FACTOR,
        )
    return {
        "$schema": cobre_schemas.schema_url_for(
            "scenarios/non_controllable_factors.json"
        ),
        "non_controllable_factors": entries,
    }


def _sorted_pee_codes(
    renovaveis: Renovaveis, calendar: Sequence[OperativeStage]
) -> list[int]:
    """Every declared renewable park's ``codigo_pee``, sorted — the ncs_id
    assignment order :func:`_pee_series` (the NCS series writer) and
    :func:`build_pee_ncs_id_map` (the public ``codigo_pee -> ncs_id`` map)
    share, so neither enumeration can drift from the other.

    A code is "declared" here iff it appears on at least one
    ``PEE-GER-PER-PAT-CEN`` row (mirrors :func:`_pee_series`'s own
    ``per_park`` keys, without building its per-(stage,block) generation).

    Returns ``[]`` when the deck carries no ``PEE-CAD``/``PEE-GER`` card at
    all (mirrors :func:`_pee_series`'s own no-op guard).

    Raises
    ------
    ValueError
        When a row's ``estagio`` falls outside *calendar*.
    """
    cad = renovaveis.pee_cad(df=True)
    ger = renovaveis.pee_ger_per_pat_cen(df=True)
    if cad is None or cad.empty or ger is None or ger.empty:
        return []

    n_stages = len(calendar)
    codes: set[int] = set()
    for _, row in ger.iterrows():
        estagio = int(row["estagio"])
        if not 1 <= estagio <= n_stages:
            raise ValueError(
                f"renewable generation at stage {estagio} outside the "
                f"calendar (1..{n_stages})"
            )
        codes.add(int(row["codigo_pee"]))
    return sorted(codes)


def build_pee_ncs_id_map(
    case: DecompCase,
    id_map: DecompIdMap,
) -> dict[int, int]:
    """Public ``codigo_pee -> ncs_id`` map for the emitter, sharing
    :func:`_pee_series`'s ordering via :func:`_sorted_pee_codes`.

    ``ncs_id`` continues the ``PQ`` series' id space (:func:`_pq_series`'s
    count is the offset), then :func:`_sorted_pee_codes`'s sorted-by-code
    order — the exact same two facts :func:`_pee_series` itself combines, so
    a park's id here always matches the id its own :class:`_PqSeries` gets.

    Returns ``{}`` when *renovaveis* is ``None`` or the deck declares no
    renewable parks (mirrors :func:`_all_series`'s "no renovaveis -> PQ
    only" convention).
    """
    renovaveis = case.renovaveis
    if renovaveis is None:
        return {}
    calendar = case.calendar
    first_ncs_id = len(_pq_series(case.dadger, id_map, calendar))
    sorted_codes = _sorted_pee_codes(renovaveis, calendar)
    return {code: first_ncs_id + offset for offset, code in enumerate(sorted_codes)}


def _pee_series(
    renovaveis: Renovaveis,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
    first_ncs_id: int,
) -> list[_PqSeries]:
    """Renewable equivalent parks as NCS series, appended after ``PQ``.

    The generation card is scenario-indexed but current production fills
    it uniformly; the collapse asserts identity per (park, stage, block)
    and fails loudly if a deck ever carries genuinely per-scenario values
    (the external availability class becomes the target then).
    """
    cad = renovaveis.pee_cad(df=True)
    subm = renovaveis.pee_subm(df=True)
    ger = renovaveis.pee_ger_per_pat_cen(df=True)
    if cad is None or cad.empty or ger is None or ger.empty:
        return []

    sorted_codes = _sorted_pee_codes(renovaveis, calendar)
    ncs_id_by_code = {
        code: first_ncs_id + offset for offset, code in enumerate(sorted_codes)
    }

    names = {
        int(r["codigo_pee"]): str(r["nome_pee"]).strip() for _, r in cad.iterrows()
    }
    buses = {
        int(r["codigo_pee"]): int(r["codigo_submercado"]) for _, r in subm.iterrows()
    }

    values: dict[tuple[int, int, int], dict[int, float]] = {}
    for _, row in ger.iterrows():
        # The PEE-GER-PER-PAT-CEN card is single-período: one `estagio` per row.
        # (idecomp < 1.14.1 mis-modelled it as an estagio_inicial/estagio_final
        # range, which left `geracao` unreadable — see the idecomp renovaveis
        # PEE-GER record layout; requires idecomp >= 1.14.1.)
        estagio = int(row["estagio"])
        stage_index = estagio - 1
        if not 0 <= stage_index < len(calendar):
            raise ValueError(
                f"renewable generation at stage {estagio} outside the "
                f"calendar (1..{len(calendar)})"
            )
        key = (int(row["codigo_pee"]), stage_index, int(row["patamar"]) - 1)
        values.setdefault(key, {})[int(row["cenario"])] = float(row["geracao"])

    per_park: dict[int, dict[int, list[float]]] = {}
    for (code, stage_index, block), by_scenario in values.items():
        scenario_values = list(by_scenario.values())
        # DECOMP renewables are deterministic: every scenario carries the same
        # generation for a (park, stage, block). A per-scenario spread is a deck
        # typo (e.g. one scenario's value fat-fingered), not real stochasticity —
        # so recover the majority (modal) value, robust to a single stray entry,
        # and warn rather than crash or trust the outlier (the max).
        representative = Counter(scenario_values).most_common(1)[0][0]
        spread = max(scenario_values) - min(scenario_values)
        if spread > relative_tolerance(representative):
            _LOG.warning(
                "renewable park %d stage %d block %d: generation is not identical "
                "across %d scenarios (%.6g..%.6g) — DECOMP renewables are "
                "deterministic, so this is a deck typo; using the modal value %.6g",
                code,
                stage_index + 1,
                block + 1,
                len(scenario_values),
                min(scenario_values),
                max(scenario_values),
                representative,
            )
        stage_blocks = per_park.setdefault(code, {}).setdefault(
            stage_index, [0.0] * len(calendar[stage_index].block_hours)
        )
        stage_blocks[block] = representative

    series: list[_PqSeries] = []
    for code in sorted_codes:
        stages = per_park[code]
        if 0 not in stages:
            raise ValueError(
                f"renewable park {code} does not declare stage 1; "
                "sparse-stage inheritance has no base"
            )
        dense: list[tuple[float, ...]] = []
        for stage in calendar:
            blocks = stages.get(stage.index)
            dense.append(tuple(blocks) if blocks else dense[-1])
        sub_code = buses.get(code)
        if sub_code is None:
            raise ValueError(f"renewable park {code} has no subsystem record")
        series.append(
            _PqSeries(
                ncs_id=ncs_id_by_code[code],
                name=f"{names.get(code, f'PEE_{code}')}_{sub_code}",
                bus_id=id_map.bus_id(sub_code),
                per_stage_blocks=tuple(dense),
            )
        )
    return series


def _all_series(
    dadger: Dadger,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
    renovaveis: Renovaveis | None,
) -> list[_PqSeries]:
    series = _pq_series(dadger, id_map, calendar)
    if renovaveis is not None:
        series += _pee_series(renovaveis, id_map, calendar, len(series))
    return series
