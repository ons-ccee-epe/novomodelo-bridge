"""Load conversion for DECOMP-like decks (``DP`` → stats + block factors).

The ``DP`` records carry the deterministic per-block demand per
(subsystem, stage). Emission follows the deterministic load path:

- ``load_seasonal_stats.parquet``: ``mean_mw`` = the energy-weighted stage
  mean ``Σ_b MW_b·h_b / H`` with ``std_mw = 0.0``;
- ``load_factors.json``: ``factor_b = MW_b / mean_mw`` per block, emitted
  only for buses with load — a zero-mean bus (the fictitious subsystem,
  the transhipment bus) gets its stats row and no factors entry.

The emission invariant ``Σ_b factor_b·h_b = H`` (per bus, stage) holds by
construction and is asserted anyway: a violation means the block data and
the calendar disagree.

When Itaipu's ``RI`` register declares ``carga_ande``, the caller supplies it
via the optional *extra_bus_loads* parameter on both public entry points
below, keyed by Itaipu's own SE bus -- where it nets ADDITIVELY onto that
bus's real ``DP`` demand rather than replacing it
(:func:`_per_stage_block_loads`). The transhipment (``IV``) bus itself is
absent from ``DP`` by construction and carries no load of its own; it hosts
only the 60 Hz Itaipu group's generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa

from cobre_bridge.cobre import schemas as cobre_schemas

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from idecomp.decomp import Dadger

    from cobre_bridge.decomp.case import DecompCase
    from cobre_bridge.decomp.id_map import DecompIdMap
    from cobre_bridge.decomp.temporal import OperativeStage

_INVARIANT_RTOL = 1e-9


def _per_stage_block_loads(
    dadger: Dadger,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
    *,
    extra_bus_loads: Mapping[tuple[int, int], list[float]] | None = None,
) -> dict[tuple[int, int], list[float]]:
    """Read ``DP`` into ``{(bus_id, stage_index): [MW per block]}``.

    Missing load values (the fictitious subsystem's blank fields) read as
    0.0. Block counts must match the calendar's per-stage block structure.

    *extra_bus_loads* supplies additional per-(bus, stage) block-load rows --
    Itaipu's ``carga_ande``, keyed by its own SE bus -- merged in ADDITIVELY
    after the ``DP`` read: a ``(bus, stage)`` key already present from ``DP``
    is summed element-wise with the extra values rather than replaced, so
    SE's real demand is never clobbered by the ANDE addition; a key absent
    from ``DP`` is inserted as-is.
    """
    dp = dadger.dp(df=True)
    if dp is None or dp.empty:
        raise ValueError("the deck has no DP records; cannot convert load")

    loads: dict[tuple[int, int], list[float]] = {}
    for _, row in dp.iterrows():
        stage_index = int(row["estagio"]) - 1
        if not 0 <= stage_index < len(calendar):
            raise ValueError(
                f"DP stage {int(row['estagio'])} outside the calendar "
                f"(1..{len(calendar)})"
            )
        n_blocks = int(row["numero_patamares"])
        stage_blocks = len(calendar[stage_index].block_hours)
        if n_blocks != stage_blocks:
            raise ValueError(
                f"DP stage {int(row['estagio'])}: {n_blocks} blocks vs "
                f"{stage_blocks} in the calendar"
            )
        bus = id_map.bus_id(int(row["codigo_submercado"]))
        values = [
            0.0 if pd.isna(row[f"carga_{k}"]) else float(row[f"carga_{k}"])
            for k in range(1, n_blocks + 1)
        ]
        loads[(bus, stage_index)] = values
    if extra_bus_loads:
        for key, extra_values in extra_bus_loads.items():
            existing_values = loads.get(key)
            if existing_values is None:
                loads[key] = extra_values
                continue
            if len(existing_values) != len(extra_values):
                raise ValueError(
                    f"bus {key[0]} stage {key[1]}: extra_bus_loads block "
                    f"count ({len(extra_values)}) does not match DP's "
                    f"({len(existing_values)})"
                )
            loads[key] = [
                a + b for a, b in zip(existing_values, extra_values, strict=True)
            ]
    return loads


def _stage_mean_mw(block_loads: Sequence[float], stage: OperativeStage) -> float:
    """Energy-weighted stage mean: ``Σ_b MW_b·h_b / H``."""
    energy = sum(
        mw * hours for mw, hours in zip(block_loads, stage.block_hours, strict=True)
    )
    return energy / stage.total_hours


def convert_load_stats(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    extra_bus_loads: Mapping[tuple[int, int], list[float]] | None = None,
) -> pa.Table:
    """Build ``load_seasonal_stats`` rows: one per (bus, stage), ``std = 0``.

    Every bus in the id map gets a row for every stage; a bus absent from
    ``DP`` entirely (the transhipment bus, in practice) carries zero load.
    *extra_bus_loads* (see :func:`_per_stage_block_loads`) nets its values
    additively onto whichever bus it keys -- Itaipu's SE bus, for the
    ``carga_ande`` load -- so that bus's own ``DP`` demand is preserved.
    """
    calendar = case.calendar
    loads = _per_stage_block_loads(
        case.dadger, id_map, calendar, extra_bus_loads=extra_bus_loads
    )

    bus_ids: list[int] = []
    stage_ids: list[int] = []
    means: list[float] = []
    for bus in range(id_map.n_buses):
        for stage in calendar:
            block_loads = loads.get((bus, stage.index))
            mean = 0.0 if block_loads is None else _stage_mean_mw(block_loads, stage)
            bus_ids.append(bus)
            stage_ids.append(stage.index)
            means.append(mean)

    return pa.table(
        {
            "bus_id": pa.array(bus_ids, type=pa.int32()),
            "stage_id": pa.array(stage_ids, type=pa.int32()),
            "mean_mw": pa.array(means, type=pa.float64()),
            "std_mw": pa.array([0.0] * len(means), type=pa.float64()),
        }
    )


def convert_load_factors(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    extra_bus_loads: Mapping[tuple[int, int], list[float]] | None = None,
) -> dict:
    """Build the ``load_factors.json`` dict from the ``DP`` block loads.

    Entries exist only for (bus, stage) pairs with load; factors of a zero
    mean are undefined and deliberately absent. *extra_bus_loads*
    (see :func:`_per_stage_block_loads`) feeds the same
    ``Σ_b factor_b·h_b = H`` invariant check below as every ``DP``-sourced
    row.
    """
    calendar = case.calendar
    loads = _per_stage_block_loads(
        case.dadger, id_map, calendar, extra_bus_loads=extra_bus_loads
    )

    entries: list[dict] = []
    for (bus, stage_index), block_loads in sorted(loads.items()):
        stage = calendar[stage_index]
        mean = _stage_mean_mw(block_loads, stage)
        if mean == 0.0:
            continue
        factors = [mw / mean for mw in block_loads]
        weighted = sum(
            f * hours for f, hours in zip(factors, stage.block_hours, strict=True)
        )
        if abs(weighted - stage.total_hours) > _INVARIANT_RTOL * stage.total_hours:
            raise ValueError(
                f"bus {bus} stage {stage_index}: block factors violate "
                f"the hours invariant (Σ f·h = {weighted}, expected "
                f"{stage.total_hours})"
            )
        entries.append(
            {
                "bus_id": bus,
                "stage_id": stage_index,
                "block_factors": [
                    {"block_id": b, "factor": factor}
                    for b, factor in enumerate(factors)
                ],
            }
        )

    return {
        "$schema": cobre_schemas.schema_url_for("scenarios/load_factors.json"),
        "load_factors": entries,
    }
