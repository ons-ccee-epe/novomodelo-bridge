"""Network conversion for DECOMP-like decks (``SB`` + ``CD`` → ``buses.json``,
``IA`` → ``lines.json`` and the line bounds).

Declared subsystems become buses with their deficit curves; the implicit
transhipment node becomes a converter-created zero-load bus with no
deficit curve of its own (it carries no demand, so no deficit is ever
priced there).

The deficit emitters are deliberately gated: the decks in hand carry a
single 100 %-depth segment with one cost, uniform across blocks and
stages. Anything richer fails loudly rather than being silently
approximated (per-block deficit costs and stage-varying deficit costs
have no Cobre encoding today).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa

from cobre_bridge.cobre import schemas as cobre_schemas

if TYPE_CHECKING:
    from collections.abc import Sequence

    from idecomp.decomp import Dadger

    from cobre_bridge.decomp.case import DecompCase
    from cobre_bridge.decomp.id_map import DecompIdMap
    from cobre_bridge.decomp.temporal import OperativeStage

_COST_COLUMN = re.compile(r"^custo_(\d+)$")
_LIMIT_COLUMN = re.compile(r"^limite_superior_(\d+)$")
_FULL_DEPTH_PERCENT = 100.0


def _bus_deficit_costs(dadger: Dadger) -> dict[int, float]:
    """Extract one deficit cost per subsystem code from the ``CD`` records.

    Enforces the single-segment shape the Cobre bus model encodes: one
    curve per subsystem, 100 % depth, one cost uniform across blocks and
    across every declared stage.
    """
    cd = dadger.cd(df=True)
    if cd is None or cd.empty:
        return {}

    cost_columns = [c for c in cd.columns if _COST_COLUMN.match(c)]
    limit_columns = [c for c in cd.columns if _LIMIT_COLUMN.match(c)]

    costs: dict[int, float] = {}
    for code, group in cd.groupby("codigo_submercado"):
        code = int(code)
        if group["codigo_curva"].nunique() != 1:
            raise ValueError(
                f"subsystem {code}: multi-segment deficit curves are not "
                "supported yet (multiple codigo_curva values)"
            )
        values: set[float] = set()
        for _, row in group.iterrows():
            for column in cost_columns:
                value = row[column]
                if not pd.isna(value):
                    values.add(float(value))
            for column in limit_columns:
                limit = row[column]
                if not pd.isna(limit) and float(limit) != _FULL_DEPTH_PERCENT:
                    raise ValueError(
                        f"subsystem {code}: deficit segment depth "
                        f"{float(limit)} % is not supported yet (expected "
                        f"{_FULL_DEPTH_PERCENT} %)"
                    )
        if len(values) != 1:
            raise ValueError(
                f"subsystem {code}: deficit cost must be one value uniform "
                f"across blocks and stages; got {sorted(values)}"
            )
        costs[code] = values.pop()
    return costs


def convert_buses(
    case: DecompCase,
    id_map: DecompIdMap,
) -> dict:
    """Build the ``buses.json`` dict from ``SB`` + ``CD`` records.

    Buses model subsystems, which have no commissioning date; the study
    start date serves as the canonical-ordering key. Buses without a
    ``CD`` record (the fictitious subsystem, the transhipment bus) omit
    ``deficit_segments`` and defer to the global default — they carry no
    load, so the value is never priced.
    """
    costs = _bus_deficit_costs(case.dadger)
    op_date = case.start_date.isoformat()

    buses: list[dict] = []
    for code in id_map.bus_codes:
        bus_id = id_map.bus_id(code)
        entry: dict = {
            "id": bus_id,
            "name": id_map.bus_name(bus_id),
            "operational_start_date": op_date,
        }
        if code in costs:
            entry["deficit_segments"] = [{"depth_mw": None, "cost": costs[code]}]
        buses.append(entry)

    buses.append(
        {
            "id": id_map.transhipment_bus_id,
            "name": id_map.bus_name(id_map.transhipment_bus_id),
            "operational_start_date": op_date,
        }
    )

    return {
        "$schema": cobre_schemas.schema_url_for("system/buses.json"),
        "buses": buses,
    }


def _ia_dense(
    dadger: Dadger,
    calendar: Sequence[OperativeStage],
) -> dict[tuple[str, str], list[dict]]:
    """Read ``IA`` into dense per-stage per-block limits per exchange pair.

    Returns ``{(name_de, name_para): [{"de_para": [...], "para_de": [...]}
    per stage]}`` with declared stages forward-filled (stage 1 mandatory).
    """
    ia = dadger.ia(df=True)
    if ia is None or ia.empty:
        return {}

    declared: dict[tuple[str, str], dict[int, dict]] = {}
    for _, row in ia.iterrows():
        stage_index = int(row["estagio"]) - 1
        if not 0 <= stage_index < len(calendar):
            raise ValueError(
                f"IA stage {int(row['estagio'])} outside the calendar "
                f"(1..{len(calendar)})"
            )
        n_blocks = len(calendar[stage_index].block_hours)

        def _limits(
            prefix: str, row: pd.Series = row, n: int = n_blocks
        ) -> list[float]:
            values = []
            for k in range(1, n + 1):
                value = row[f"{prefix}_{k}"]
                if pd.isna(value):
                    raise ValueError(
                        f"IA {row['nome_submercado_de']}-"
                        f"{row['nome_submercado_para']} stage "
                        f"{int(row['estagio'])}: missing {prefix}_{k}"
                    )
                values.append(float(value))
            return values

        pair = (
            str(row["nome_submercado_de"]).strip(),
            str(row["nome_submercado_para"]).strip(),
        )
        declared.setdefault(pair, {})[stage_index] = {
            "de_para": _limits("limite_de_para"),
            "para_de": _limits("limite_para_de"),
        }

    dense: dict[tuple[str, str], list[dict]] = {}
    for pair, stages in declared.items():
        if 0 not in stages:
            raise ValueError(
                f"IA pair {pair[0]}-{pair[1]} does not declare stage 1; "
                "sparse-stage inheritance has no base"
            )
        per_stage: list[dict] = []
        for stage in calendar:
            per_stage.append(
                stages.get(stage.index, per_stage[-1] if per_stage else {})
            )
        dense[pair] = per_stage
    return dense


_LINE_BOUNDS_SCHEMA = pa.schema(
    [
        pa.field("line_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("block_id", pa.int32(), nullable=True),
        pa.field("direct_mw", pa.float64(), nullable=False),
        pa.field("reverse_mw", pa.float64(), nullable=False),
    ]
)


def convert_lines(
    case: DecompCase,
    id_map: DecompIdMap,
) -> tuple[dict, pa.Table]:
    """Convert the ``IA`` exchange network.

    Returns ``(lines.json dict, line_bounds table)``: one line per declared
    pair, plus a ``line_bounds`` table carrying a stage-level base row
    (``block_id = None``) whose direct/reverse capacity is the maximum block
    limit of each direction, and — only for (line, stage) entries whose
    blocks actually differ from that base — one per-block override row
    (``block_id = 0..n-1``) per declared block. Base and block rows coexist
    (cobre rule 36). Block rows carry the ``IA`` record's own per-block
    limit as an absolute MW value, read directly with no factor round-trip.
    The unbounded sentinel (99999) passes through as a plain large capacity.

    TRACKED COBRE-GAP WORKAROUND (C4, the cobre repository's
    conversion-found-improvements registry): a deck that declares no ``IA``
    pairs still emits ``system/lines.json`` with an empty ``lines`` list,
    because cobre requires the file structurally even for a lineless study.
    Remove this note when the registry's C4 entry closes.
    """
    calendar = case.calendar
    dense = _ia_dense(case.dadger, calendar)
    op_date = case.start_date.isoformat()

    pairs = sorted(
        dense,
        key=lambda p: (id_map.bus_id_by_name(p[0]), id_map.bus_id_by_name(p[1])),
    )

    lines: list[dict] = []
    bounds_line_ids: list[int] = []
    bounds_stage_ids: list[int] = []
    bounds_direct: list[float] = []
    bounds_reverse: list[float] = []
    bounds_block_ids: list[int | None] = []

    for line_id, pair in enumerate(pairs):
        per_stage = dense[pair]
        base_first = per_stage[0]
        lines.append(
            {
                "id": line_id,
                "name": f"{pair[0]}-{pair[1]}",
                "operational_start_date": op_date,
                "source_bus_id": id_map.bus_id_by_name(pair[0]),
                "target_bus_id": id_map.bus_id_by_name(pair[1]),
                "capacity": {
                    "direct_mw": max(base_first["de_para"]),
                    "reverse_mw": max(base_first["para_de"]),
                },
            }
        )
        for stage in calendar:
            limits = per_stage[stage.index]
            direct_base = max(limits["de_para"])
            reverse_base = max(limits["para_de"])
            bounds_line_ids.append(line_id)
            bounds_stage_ids.append(stage.index)
            bounds_direct.append(direct_base)
            bounds_reverse.append(reverse_base)
            bounds_block_ids.append(None)

            # Per-block rows carry the IA record's own de_para/para_de
            # values directly, as absolute MW — no factor round-trip, so
            # no division and no reconstruction error. A block whose limit
            # is 0.0 is an ordinary bound (direct_mw = 0.0 / reverse_mw = 0.0),
            # not a sentinel to skip.
            blocks = list(zip(limits["de_para"], limits["para_de"], strict=True))
            uniform = all(d == direct_base and r == reverse_base for d, r in blocks)
            if not uniform:
                for b, (d, r) in enumerate(blocks):
                    bounds_line_ids.append(line_id)
                    bounds_stage_ids.append(stage.index)
                    bounds_direct.append(d)
                    bounds_reverse.append(r)
                    bounds_block_ids.append(b)

    bounds = pa.table(
        {
            "line_id": pa.array(bounds_line_ids, type=pa.int32()),
            "stage_id": pa.array(bounds_stage_ids, type=pa.int32()),
            "block_id": pa.array(bounds_block_ids, type=pa.int32()),
            "direct_mw": pa.array(bounds_direct, type=pa.float64()),
            "reverse_mw": pa.array(bounds_reverse, type=pa.float64()),
        },
        schema=_LINE_BOUNDS_SCHEMA,
    )
    return (
        {"$schema": cobre_schemas.schema_url_for("system/lines.json"), "lines": lines},
        bounds,
    )


#: The converter-created SE<->IV line's fixed name -- the
#: transshipment link that carries Itaipu's 60 Hz output into SE, mirroring
#: the ``"{de}-{para}"`` naming convention IA lines get above.
_IV_SE_LINE_NAME = "IV-SE"

#: idecomp's ``df=True`` accessor expands the ``RI`` register's per-patamar
#: ``geracao_maxima_60_hz`` list the same way it expands ``carga_ande``
#: (mirrors ``libs_electrical._CARGA_ANDE_COLUMN``): one
#: ``geracao_maxima_60_hz_{k}`` column per patamar slot, a row's own slots
#: beyond its declared patamar count reading back as ``NaN``.
_GERACAO_MAXIMA_60HZ_COLUMN = re.compile(r"^geracao_maxima_60_hz_(\d+)$")

#: The source model's own "no limit" numeric idiom for an exchange pair --
#: the exact value :func:`convert_lines` already passes through as ordinary
#: ``IA`` data (see its own docstring). Reused verbatim, never invented, as
#: the converter-created SE<->IV line's capacity when the deck carries no
#: ``RI`` register to size the 60 Hz half.
_UNBOUNDED_LINE_CAPACITY_MW = 99999.0


def _itaipu_60hz_capacity_mw(dadger: Dadger) -> float:
    """The SE<->IV line's capacity: the largest declared ``RI``
    ``geracao_maxima_60_hz`` value across every stage and patamar, or
    :data:`_UNBOUNDED_LINE_CAPACITY_MW` when the deck carries no ``RI``
    register at all.

    An absent register means "no limit", not zero -- the line must never
    bind the 60 Hz output.
    """
    ri = dadger.ri(df=True)
    if ri is None or ri.empty:
        return _UNBOUNDED_LINE_CAPACITY_MW
    columns = [c for c in ri.columns if _GERACAO_MAXIMA_60HZ_COLUMN.match(c)]
    values = [
        float(row[column])
        for _, row in ri.iterrows()
        for column in columns
        if not pd.isna(row[column])
    ]
    return max(values) if values else _UNBOUNDED_LINE_CAPACITY_MW


def append_iv_se_line(
    case: DecompCase,
    *,
    lines_doc: dict,
    line_bounds: pa.Table,
    source_bus_id: int,
    target_bus_id: int,
    capacity_mw: float,
) -> tuple[dict, pa.Table]:
    """Append the converter-created ``IV-SE`` line to a :func:`convert_lines` result.

    Unconditional on Itaipu detection alone: the caller decides whether
    Itaipu (code 66) is operated and only calls this when it is, so a deck
    that never operates Itaipu stays byte-identical to today. But before
    synthesizing, this checks whether *lines_doc*
    already carries a line connecting *source_bus_id* and *target_bus_id*
    (as an unordered pair -- the deck's own ``IA`` line may run either
    direction). When one exists, the ``IV`` bus is not islanded: the deck's
    own line already wires it, and carries the source model's own capacity
    rather than the unbounded sentinel, so this is a no-op --
    *lines_doc*/*line_bounds* are returned unchanged.

    Otherwise (the genuinely-islanded case) it synthesizes exactly as
    before: the new line's id is the next free id after every ``IA`` line
    (``len(lines_doc["lines"])``), so the existing lines are never
    renumbered and ``constraints.build_fi_line_map`` stays stable.

    Mirrors :func:`convert_lines`'s own ``lines.json`` entry and stage-level
    ``line_bounds`` base-row (``block_id = None``) shape -- one base row per
    calendar stage, no per-block override rows (this link carries no
    per-block data of its own). Both directions share *capacity_mw* since
    the source model declares no separate reverse limit for this
    converter-created link.
    """
    calendar = case.calendar
    pair = frozenset({source_bus_id, target_bus_id})
    existing_line = next(
        (
            line
            for line in lines_doc["lines"]
            if frozenset({line["source_bus_id"], line["target_bus_id"]}) == pair
        ),
        None,
    )
    if existing_line is not None:
        # The deck's own IA register already wires this pair (with its real
        # capacity), so the converter-created IV-SE line is a no-op here.
        return lines_doc, line_bounds

    line_id = len(lines_doc["lines"])
    new_line = {
        "id": line_id,
        "name": _IV_SE_LINE_NAME,
        "operational_start_date": case.start_date.isoformat(),
        "source_bus_id": source_bus_id,
        "target_bus_id": target_bus_id,
        "capacity": {"direct_mw": capacity_mw, "reverse_mw": capacity_mw},
    }
    extended_lines_doc = {**lines_doc, "lines": [*lines_doc["lines"], new_line]}

    n_stages = len(calendar)
    new_bounds = pa.table(
        {
            "line_id": pa.array([line_id] * n_stages, type=pa.int32()),
            "stage_id": pa.array([s.index for s in calendar], type=pa.int32()),
            "block_id": pa.array([None] * n_stages, type=pa.int32()),
            "direct_mw": pa.array([capacity_mw] * n_stages, type=pa.float64()),
            "reverse_mw": pa.array([capacity_mw] * n_stages, type=pa.float64()),
        },
        schema=_LINE_BOUNDS_SCHEMA,
    )
    return extended_lines_doc, pa.concat_tables([line_bounds, new_bounds])


def pumping_station_id_map(dadger: Dadger) -> dict[int, int]:
    """``{codigo_usina: 0-based pumping-station id}`` — the single authority
    for ``UE`` id assignment.

    The id is the row's position in ``codigo_usina``-sorted order — exactly
    what :func:`convert_pumping_stations` assigns via ``len(stations)``.
    ``single_term_bounds.single_term_bound_contributions``'s ``QBOM`` path
    resolves its pumping-station entity id through this same map (its
    ``pumping_station_ids`` argument), since the QBOM term's ``code`` is a
    pumping-station ``codigo_usina``, not a hydro one.
    """
    ue = dadger.ue(df=True)
    if ue is None or ue.empty:
        return {}
    codes = sorted(int(c) for c in ue["codigo_usina"])
    return {code: station_id for station_id, code in enumerate(codes)}


def convert_pumping_stations(
    case: DecompCase,
    id_map: DecompIdMap,
) -> dict:
    """Convert the ``UE`` pumping stations (1:1).

    Water is lifted from the downstream plant to the upstream one; both
    must be operated plants. Station ids come from
    :func:`pumping_station_id_map`, the single authority for ``UE`` id
    assignment.
    """
    dadger = case.dadger
    ue = dadger.ue(df=True)
    if ue is None or ue.empty:
        return {
            "$schema": cobre_schemas.schema_url_for("system/pumping_stations.json"),
            "pumping_stations": [],
        }

    station_ids = pumping_station_id_map(dadger)
    op_date = case.start_date.isoformat()
    stations: list[dict] = []
    for _, row in ue.sort_values("codigo_usina").iterrows():
        code = int(row["codigo_usina"])
        name = str(row["nome_usina"]).strip()
        source = int(row["codigo_usina_jusante"])
        destination = int(row["codigo_usina_montante"])
        stations.append(
            {
                "id": station_ids[code],
                "name": name,
                "operational_start_date": op_date,
                "bus_id": id_map.bus_id(int(row["codigo_submercado"])),
                "source_hydro_id": id_map.hydro_id(source),
                "destination_hydro_id": id_map.hydro_id(destination),
                "consumption_mw_per_m3s": float(row["taxa_consumo"]),
                "flow": {
                    "min_m3s": float(row["vazao_minima_bombeavel"]),
                    "max_m3s": float(row["vazao_maxima_bombeavel"]),
                },
            }
        )
    return {
        "$schema": cobre_schemas.schema_url_for("system/pumping_stations.json"),
        "pumping_stations": stations,
    }
