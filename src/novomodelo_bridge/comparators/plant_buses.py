"""Buses that share a hydro plant, judged as one group.

A hydro plant with unit groups on several buses generates on any of them, and
the source model and Novomodelo can each place its output differently. Each
bus's own net import, and the flow on a line between two of them, follow that
placement; the net import of the buses taken together does not. That group
net import is the flow of the lines crossing the group's boundary, so it is
built from the line rows the same way on both tracks: each source model's
bus net import is the net flow of the bus's lines.

The group's row takes ``net_import_mw``. Each member's own row moves to
:data:`~novomodelo_bridge.core.agreement.SHARED_PLANT_NET_IMPORT` and the
rows of a line between members to
:data:`~novomodelo_bridge.core.agreement.SHARED_PLANT_NET_FLOW`, which the
console shows but the headline does not count.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from novomodelo_bridge.comparators.model import ResultComparison, result_diff
from novomodelo_bridge.core.agreement import (
    SHARED_PLANT_NET_FLOW,
    SHARED_PLANT_NET_IMPORT,
    TIED_NET_FLOW,
)
from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    emit,
)
from novomodelo_bridge.novomodelo import readers as novomodelo_readers

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

_LOG = logging.getLogger(__name__)

_NET_IMPORT = "net_import_mw"
_LINE_FLOWS = frozenset({"net_flow_mw", TIED_NET_FLOW})


@dataclass(frozen=True)
class SharedPlantGroup:
    """Buses joined by hydro plants whose unit groups sit on more than one
    of them.

    Attributes:
        bus_ids: The group's bus ids, ascending.
        plants: The names of the plants that join them.
    """

    bus_ids: tuple[int, ...]
    plants: tuple[str, ...]


def find_shared_plant_groups(hydros: Sequence[dict]) -> tuple[SharedPlantGroup, ...]:
    """The groups of buses that *hydros* (``system/hydros.json`` entries)
    join, by their smallest bus id.

    A plant on one bus joins nothing; two plants that share a bus put all
    their buses in one group.
    """
    parent: dict[int, int] = {}

    def root(bus: int) -> int:
        while parent[bus] != bus:
            parent[bus] = parent[parent[bus]]
            bus = parent[bus]
        return bus

    joining: list[tuple[str, list[int]]] = []
    for hydro in hydros:
        buses = sorted(
            {int(group["bus_id"]) for group in hydro.get("unit_groups") or []}
        )
        if len(buses) < 2:
            continue
        joining.append((str(hydro.get("name") or f"hydro {hydro['id']}"), buses))
        for bus in buses:
            parent.setdefault(bus, bus)
        first = root(buses[0])
        for bus in buses[1:]:
            parent[root(bus)] = first

    members: dict[int, set[int]] = defaultdict(set)
    plants: dict[int, list[str]] = defaultdict(list)
    for bus in parent:
        members[root(bus)].add(bus)
    for name, buses in joining:
        plants[root(buses[0])].append(name)
    return tuple(
        sorted(
            (
                SharedPlantGroup(
                    bus_ids=tuple(sorted(members[key])),
                    plants=tuple(sorted(plants[key])),
                )
                for key in members
            ),
            key=lambda group: group.bus_ids,
        )
    )


def merge_shared_plant_buses(
    results: Sequence[ResultComparison], novomodelo_output_dir: Path
) -> list[ResultComparison]:
    """*results* with the buses that share a hydro plant judged as one group.

    Run on the full result list, after the line rows have been compared. For
    each group of :func:`find_shared_plant_groups` over the converted case's
    ``system/hydros.json``, a ``net_import_mw`` row named after its buses
    (``SE+IV``), keyed by its smallest bus id, carries each model's net flow
    into the group at every stage where each line crossing its boundary has
    a row. One warning names every group with a moved or added row.
    """
    if not any(
        (r.entity_type == "bus" and r.variable == _NET_IMPORT)
        or (r.entity_type == "line" and r.variable in _LINE_FLOWS)
        for r in results
    ):
        return list(results)
    groups = find_shared_plant_groups(
        novomodelo_readers.read_novomodelo_hydros(novomodelo_output_dir)
    )
    if not groups:
        return list(results)

    group_of = {
        bus: index for index, group in enumerate(groups) for bus in group.bus_ids
    }
    internal: dict[int, int] = {}
    boundary: dict[int, dict[int, float]] = defaultdict(dict)
    line_names: dict[int, str] = {}
    for line in novomodelo_readers.read_novomodelo_lines(novomodelo_output_dir):
        line_id = int(line["id"])
        source = group_of.get(int(line["source_bus_id"]))
        target = group_of.get(int(line["target_bus_id"]))
        line_names[line_id] = str(line.get("name") or f"line {line_id}")
        if source is not None and source == target:
            internal[line_id] = source
            continue
        if target is not None:
            boundary[target][line_id] = 1.0
        if source is not None:
            boundary[source][line_id] = -1.0

    marked: list[ResultComparison] = []
    touched: set[int] = set()
    lead_codes: dict[int, int] = {}
    flows: dict[tuple[int, int], tuple[float, float]] = {}
    for r in results:
        if r.entity_type == "bus" and r.variable == _NET_IMPORT:
            index = group_of.get(r.novomodelo_id)
            if index is not None:
                if r.novomodelo_id == groups[index].bus_ids[0]:
                    lead_codes[index] = r.newave_code
                touched.add(index)
                r = replace(r, variable=SHARED_PLANT_NET_IMPORT)
        elif r.entity_type == "line" and r.variable in _LINE_FLOWS:
            flows[(r.novomodelo_id, r.stage)] = (r.newave_value, r.novomodelo_value)
            index = internal.get(r.novomodelo_id)
            if index is not None and r.variable == "net_flow_mw":
                touched.add(index)
                r = replace(r, variable=SHARED_PLANT_NET_FLOW)
        marked.append(r)

    bus_names = {
        int(bus["id"]): str(bus["name"])
        for bus in novomodelo_readers.read_novomodelo_buses(novomodelo_output_dir)
        if bus.get("name")
    }
    for index, group in enumerate(groups):
        rows = _group_rows(
            group,
            boundary.get(index, {}),
            flows,
            name="+".join(bus_names.get(b, f"bus {b}") for b in group.bus_ids),
            newave_code=lead_codes.get(index, 0),
        )
        if rows:
            touched.add(index)
            marked.extend(rows)

    if touched:
        _emit_shared_plant_groups(
            [(groups[index], index) for index in sorted(touched)],
            internal,
            bus_names,
            line_names,
        )
    return marked


def _group_rows(
    group: SharedPlantGroup,
    signs: Mapping[int, float],
    flows: Mapping[tuple[int, int], tuple[float, float]],
    *,
    name: str,
    newave_code: int,
) -> list[ResultComparison]:
    """The group's net import row at each stage where every line in *signs*
    (``+1`` into the group, ``-1`` out of it) has a row in *flows*."""
    if not signs:
        return []
    stages = sorted({stage for line_id, stage in flows if line_id in signs})
    rows: list[ResultComparison] = []
    for stage in stages:
        legs = [(signs[line_id], flows.get((line_id, stage))) for line_id in signs]
        if any(flow is None for _, flow in legs):
            continue
        reference = sum(sign * flow[0] for sign, flow in legs if flow is not None)
        novomodelo = sum(sign * flow[1] for sign, flow in legs if flow is not None)
        abs_diff, rel_diff = result_diff(reference, novomodelo)
        rows.append(
            ResultComparison(
                entity_type="bus",
                entity_name=name,
                newave_code=newave_code,
                novomodelo_id=group.bus_ids[0],
                stage=stage,
                variable=_NET_IMPORT,
                newave_value=reference,
                novomodelo_value=novomodelo,
                abs_diff=abs_diff,
                rel_diff=rel_diff,
            )
        )
    return rows


def _emit_shared_plant_groups(
    reported: Sequence[tuple[SharedPlantGroup, int]],
    internal: Mapping[int, int],
    bus_names: Mapping[int, str],
    line_names: Mapping[int, str],
) -> None:
    rows: list[list[str]] = []
    for group, index in reported:
        inside = sorted(
            line_names[line_id] for line_id, owner in internal.items() if owner == index
        )
        rows.append(
            [
                ", ".join(str(b) for b in group.bus_ids),
                "+".join(bus_names.get(b, f"bus {b}") for b in group.bus_ids),
                ", ".join(group.plants),
                ", ".join(inside) or "none",
            ]
        )
    emit(
        Diagnostic(
            code="buses-share-plant",
            severity=Severity.WARNING,
            category="Compare data",
            title=f"Buses sharing a hydro plant judged as one ({len(reported)})",
            summary=(
                "The listed plants have unit groups on more than one bus, so "
                "the source model and Novomodelo can each place their output "
                "on any of those buses. Each bus's own net import, and the "
                "flow on the lines between them, follow where each model "
                "places it. The comparison judges the net import of each "
                "group of buses as one, under the group's name, shows each "
                "bus's own as Net import of a bus sharing a plant and the "
                "flow between them as Net flow between shared-plant buses, "
                "and leaves those two out of the headline and the --json "
                "status."
            ),
            table=DiagnosticTable(
                columns=["Bus IDs", "Group", "Hydro plants", "Lines between them"],
                rows=rows,
                justify=["left", "left", "left", "left"],
            ),
        ),
        logger=_LOG,
    )
