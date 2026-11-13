"""Thermal conversion for DECOMP-like decks (``CT`` records).

``CT`` declares, per (plant, stage), the per-block incremental cost
(``cvu``), availability (``disponibilidade``) and inflexibility
(``inflexibilidade``), sparsely by stage (later stages inherit the last
declared record). ``convert_thermal_bounds`` returns a pair: the
``min``/``max_generation_mw`` bound contributions (:class:`~novomodelo_bridge.
decomp.bounds_accumulator.BoundContribution`) the accumulator later resolves,
and a ``cost_per_mwh`` side-table — cost is not a registered bound axis (it
has no column in ``bounds_accumulator.THERMAL_BOUNDS_SCHEMA``), so it never
travels as a contribution and rides alongside for the pipeline to rejoin
after ``build_bound_tables``.

The side-table is decided apart from the generation split below. Every
``(thermal, stage)`` has one ``block_id = None`` row carrying the
hours-weighted CVU and, when the stage's CVUs are not all exactly equal, one
row per block carrying that block's exact CVU. novomodelo resolves
``thermal_bounds`` block rows column by column (block, then stage, then
registry), so a block row may carry only a cost or only bounds, and the stage
mean is shadowed wherever every block has its own cost.

Per ``(thermal, stage)``, the generation bound contributes **either** one
stage-level (``block_id = None``) contribution carrying the hours-weighted
``min``/``max`` — when the stage's per-block ``disponibilidade``/
``inflexibilidade`` values are block-uniform — **or** one contribution per
block (``block_id = 0..n-1``, no base) carrying each block's own exact
``min``/``max`` — when they are not. Never both: the accumulator's
``resolve()`` does not replicate novomodelo's replace-not-merge column semantics,
so a base contribution left alongside per-block ones would be folded into
every block's intersection instead of being shadowed by them.

GNL plants live in the anticipated-dispatch file and are converted by the
anticipation track, not here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import pandas as pd
import pyarrow as pa

from novomodelo_bridge.core.pandas_utils import as_frame
from novomodelo_bridge.decomp.bounds_accumulator import BoundContribution
from novomodelo_bridge.decomp.temporal import hours_weighted
from novomodelo_bridge.novomodelo import schemas as novomodelo_schemas

if TYPE_CHECKING:
    from collections.abc import Sequence

    from idecomp.decomp import Dadger

    from novomodelo_bridge.decomp.case import DecompCase
    from novomodelo_bridge.decomp.id_map import DecompIdMap
    from novomodelo_bridge.decomp.temporal import OperativeStage


def dense_stage_records(
    frame: pd.DataFrame | None,
    calendar: Sequence[OperativeStage],
    *,
    register: str,
    name_column: str,
) -> dict[int, dict]:
    """Densify a per-stage thermal register (``CT``, ``TG``) over *calendar*.

    Both registers declare, per ``(plant, estagio)``, the per-block ``cvu``,
    ``disponibilidade`` and ``inflexibilidade``, sparsely: stage 1 is
    mandatory and a later stage inherits the last declared record. Returns
    ``{code: {"name", "bus_code", "stages": [{cvu, disp, inflex} per block,
    one entry per calendar stage]}}``.
    """
    if frame is None or frame.empty:
        return {}

    declared: dict[int, dict] = {}
    for _, row in frame.iterrows():
        stage_index = int(row["estagio"]) - 1
        if not 0 <= stage_index < len(calendar):
            raise ValueError(
                f"{register} stage {int(row['estagio'])} outside the calendar "
                f"(1..{len(calendar)})"
            )
        n_blocks = len(calendar[stage_index].block_hours)

        def _blocks(
            prefix: str, row: pd.Series = row, n: int = n_blocks
        ) -> list[float]:
            return [
                0.0 if pd.isna(row[f"{prefix}_{k}"]) else float(row[f"{prefix}_{k}"])
                for k in range(1, n + 1)
            ]

        code = int(row["codigo_usina"])
        plant = declared.setdefault(
            code,
            {
                "name": str(row[name_column]).strip(),
                "bus_code": int(row["codigo_submercado"]),
                "declared": {},
            },
        )
        plant["declared"][stage_index] = {
            "cvu": _blocks("cvu"),
            "disp": _blocks("disponibilidade"),
            "inflex": _blocks("inflexibilidade"),
        }

    for code, plant in declared.items():
        if 0 not in plant["declared"]:
            raise ValueError(
                f"{register} plant {code} ({plant['name']}) does not declare "
                "stage 1; sparse-stage inheritance has no base"
            )
        dense: list[dict] = []
        for stage in calendar:
            dense.append(plant["declared"].get(stage.index, dense[-1] if dense else {}))
        plant["stages"] = dense
        del plant["declared"]
    return declared


def _ct_dense(
    dadger: Dadger,
    calendar: Sequence[OperativeStage],
) -> dict[int, dict]:
    return dense_stage_records(
        as_frame(dadger.ct(df=True)), calendar, register="CT", name_column="nome_usina"
    )


def convert_thermals(
    case: DecompCase,
    id_map: DecompIdMap,
) -> dict:
    """Build ``thermals.json`` from the ``CT`` records (stage-1 base values)."""
    calendar = case.calendar
    plants = _ct_dense(case.dadger, calendar)
    op_date = case.start_date.isoformat()
    first = calendar[0]

    thermals: list[dict] = []
    for code in id_map.thermal_codes:
        plant = plants.get(code)
        if plant is None:
            raise ValueError(f"thermal code {code} has no CT records")
        base = plant["stages"][0]
        thermals.append(
            {
                "id": id_map.thermal_id(code),
                "name": plant["name"],
                "operational_start_date": op_date,
                "bus_id": id_map.bus_id(plant["bus_code"]),
                "cost_per_mwh": hours_weighted(base["cvu"], first),
                "generation": {
                    "min_mw": hours_weighted(base["inflex"], first),
                    "max_mw": hours_weighted(base["disp"], first),
                },
            }
        )
    return {
        "$schema": novomodelo_schemas.schema_url_for("system/thermals.json"),
        "thermals": thermals,
    }


#: The ``cost_per_mwh`` side-table schema — cost rides alongside the
#: generation-bound contributions (see the module docstring) rather than
#: through them, so it needs its own schema instead of
#: ``bounds_accumulator.THERMAL_BOUNDS_SCHEMA``.
_THERMAL_COST_SCHEMA = pa.schema(
    [
        pa.field("thermal_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("block_id", pa.int32(), nullable=True),
        pa.field("cost_per_mwh", pa.float64(), nullable=True),
    ]
)


class ThermalBounds(NamedTuple):
    """:func:`convert_thermal_bounds`'s return shape.

    ``generation`` is the ``min``/``max_generation_mw`` contribution list the
    pipeline feeds into ``bounds_accumulator.resolve`` alongside every other
    family's contributions; ``cost`` is the ``cost_per_mwh`` side-table
    (schema :data:`_THERMAL_COST_SCHEMA`) the pipeline rejoins onto the
    resolved ``thermal_bounds`` table afterwards: the stage row plus the block
    rows where CVUs differ (see the module docstring).
    """

    generation: list[BoundContribution]
    cost: pa.Table


def convert_thermal_bounds(
    case: DecompCase,
    id_map: DecompIdMap,
) -> ThermalBounds:
    """Thermal generation-bound contributions plus the ``cost_per_mwh`` side-table.

    Every ``(thermal, stage)`` contributes one ``cost`` row (``block_id =
    None``) carrying the hours-weighted ``cost_per_mwh``, plus one row per
    block when the stage's CVUs differ (see the module docstring). The
    ``generation`` bound contributes, per ``(thermal, stage)``,
    **either** one stage-level (``block_id = None``) contribution carrying
    the hours-weighted ``min``/``max_generation_mw`` — when the stage's
    ``disponibilidade``/``inflexibilidade`` are block-uniform — **or** one
    contribution per block (``block_id = 0..n-1``, no base) carrying each
    block's own exact ``min``/``max`` — when they are not (see the module
    docstring's replace-vs-intersect note). This mirrors ``convert_lines``'
    sparse base-vs-override convention (``decomp/converters/network.py``), except the
    two never coexist here.
    """
    calendar = case.calendar
    plants = _ct_dense(case.dadger, calendar)

    contributions: list[BoundContribution] = []
    cost_thermal_ids: list[int] = []
    cost_stage_ids: list[int] = []
    cost_block_ids: list[int | None] = []
    cost_values: list[float] = []
    for code in id_map.thermal_codes:
        plant = plants.get(code)
        if plant is None:
            raise ValueError(f"thermal code {code} has no CT records")
        thermal_id = id_map.thermal_id(code)
        for stage in calendar:
            values = plant["stages"][stage.index]
            disp = values["disp"]
            inflex = values["inflex"]
            cvu = values["cvu"]

            cost_thermal_ids.append(thermal_id)
            cost_stage_ids.append(stage.index)
            cost_block_ids.append(None)
            cost_values.append(hours_weighted(cvu, stage))

            if not all(c == cvu[0] for c in cvu):
                for b, c in enumerate(cvu):
                    cost_thermal_ids.append(thermal_id)
                    cost_stage_ids.append(stage.index)
                    cost_block_ids.append(b)
                    cost_values.append(c)

            uniform = all(d == disp[0] for d in disp) and all(
                m == inflex[0] for m in inflex
            )
            if uniform:
                contributions.append(
                    BoundContribution(
                        family="thermal",
                        entity_id=thermal_id,
                        stage_id=stage.index,
                        block_id=None,
                        axis="generation",
                        lower=hours_weighted(inflex, stage),
                        upper=hours_weighted(disp, stage),
                        contributor="CT",
                    )
                )
            else:
                for b in range(len(disp)):
                    contributions.append(
                        BoundContribution(
                            family="thermal",
                            entity_id=thermal_id,
                            stage_id=stage.index,
                            block_id=b,
                            axis="generation",
                            lower=inflex[b],
                            upper=disp[b],
                            contributor="CT",
                        )
                    )

    cost_table = pa.table(
        {
            "thermal_id": pa.array(cost_thermal_ids, type=pa.int32()),
            "stage_id": pa.array(cost_stage_ids, type=pa.int32()),
            "block_id": pa.array(cost_block_ids, type=pa.int32()),
            "cost_per_mwh": pa.array(cost_values, type=pa.float64()),
        },
        schema=_THERMAL_COST_SCHEMA,
    )
    return ThermalBounds(generation=contributions, cost=cost_table)
