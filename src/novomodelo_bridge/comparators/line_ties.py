"""Lines whose flow the two models can route differently at the same cost.

A loop of lines carries a circulation at no cost when its exchange costs
split into two halves of equal sum: the flow added on the lines of one half
is the flow removed from the other. The LP is then degenerate, and each
model stops at a different split of the same bus exchanges between the
loop's lines. The bus net imports still agree; the line flows do not.

Both source models halve the exchange penalty on a line that touches a
fictitious subsystem, and the conversion keeps that, so a route through a
fictitious bus costs exactly the direct line and every such triangle ties.
The rows of a line on a tied loop move from ``net_flow_mw`` to
:data:`~novomodelo_bridge.core.agreement.TIED_NET_FLOW`, which the console
shows but the headline does not count.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from novomodelo_bridge.core.agreement import TIED_NET_FLOW
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

    from novomodelo_bridge.comparators.model import ResultComparison

_LOG = logging.getLogger(__name__)

#: Two halves tie when their costs differ by at most this share of the loop's.
_COST_TOLERANCE: float = 1e-9


@dataclass(frozen=True)
class _Line:
    id: int
    name: str
    ends: tuple[int, int]
    cost: float


@dataclass(frozen=True)
class TiedLoop:
    """A loop of lines whose exchange costs split into two equal halves.

    Attributes:
        buses: The bus ids around the loop, starting at the smallest.
        line_ids: The loop's line ids.
        halves: The line names of each half, the one with fewer lines first.
        cost: The exchange cost of each half (R$/MWh).
    """

    buses: tuple[int, ...]
    line_ids: frozenset[int]
    halves: tuple[tuple[str, ...], tuple[str, ...]]
    cost: float


def _simple_loops(
    lines: Sequence[_Line],
) -> list[tuple[tuple[int, ...], tuple[_Line, ...]]]:
    """Each simple loop of *lines* once, as its buses and lines in order.

    Two lines between the same buses are a loop of two.
    """
    incident: dict[int, list[_Line]] = {}
    for line in lines:
        first, second = line.ends
        if first != second:
            incident.setdefault(first, []).append(line)
            incident.setdefault(second, []).append(line)

    found: set[frozenset[int]] = set()
    loops: list[tuple[tuple[int, ...], tuple[_Line, ...]]] = []

    def walk(start: int, buses: list[int], path: list[_Line]) -> None:
        for line in incident[buses[-1]]:
            if line in path:
                continue
            first, second = line.ends
            other = second if first == buses[-1] else first
            if other == start and path:
                key = frozenset(p.id for p in path) | {line.id}
                if key not in found:
                    found.add(key)
                    loops.append((tuple(buses), (*path, line)))
            elif other > start and other not in buses:
                walk(start, [*buses, other], [*path, line])

    for start in sorted(incident):
        walk(start, [start], [])
    return loops


def _equal_halves(
    loop: Sequence[_Line],
) -> tuple[tuple[_Line, ...], tuple[_Line, ...]] | None:
    """Two halves of *loop* with the same total cost, the shorter first, or
    ``None`` when no split ties."""
    total = sum(line.cost for line in loop)
    for size in range(1, len(loop) // 2 + 1):
        for half in itertools.combinations(loop, size):
            if abs(2.0 * sum(line.cost for line in half) - total) <= (
                _COST_TOLERANCE * total
            ):
                rest = tuple(line for line in loop if line not in half)
                return half, rest
    return None


def find_tied_loops(
    lines: Sequence[dict], default_cost: float | None
) -> tuple[TiedLoop, ...]:
    """The loops of *lines* (``system/lines.json`` entries) whose exchange
    costs tie, fewest lines first.

    A line without its own ``exchange_cost`` costs *default_cost*
    (``penalties.json``); when that is ``None`` too, its cost is unknown and
    no loop through it is judged. A tied loop whose lines all lie on shorter
    tied loops, such as the outline of two tied triangles, adds nothing and
    is left out.
    """
    known: list[_Line] = []
    for entry in lines:
        cost = entry.get("exchange_cost", default_cost)
        if cost is None:
            continue
        known.append(
            _Line(
                id=int(entry["id"]),
                name=str(entry.get("name") or f"line {entry['id']}"),
                ends=(
                    int(entry["source_bus_id"]),
                    int(entry["target_bus_id"]),
                ),
                cost=float(cost),
            )
        )

    tied: list[TiedLoop] = []
    covered: set[int] = set()
    for buses, loop in sorted(_simple_loops(known), key=lambda item: len(item[1])):
        halves = _equal_halves(loop)
        if halves is None:
            continue
        line_ids = frozenset(line.id for line in loop)
        if line_ids <= covered:
            continue
        covered |= line_ids
        shorter, longer = halves
        tied.append(
            TiedLoop(
                buses=buses,
                line_ids=line_ids,
                halves=(
                    tuple(line.name for line in shorter),
                    tuple(line.name for line in longer),
                ),
                cost=sum(line.cost for line in shorter),
            )
        )
    return tuple(tied)


def mark_tied_lines(
    results: Sequence[ResultComparison], novomodelo_output_dir: Path
) -> list[ResultComparison]:
    """*results* with each tied line's net flow moved to
    :data:`~novomodelo_bridge.core.agreement.TIED_NET_FLOW`.

    The loops come from the converted case's ``system/lines.json`` and
    ``penalties.json``. One warning names every loop with a moved row.
    """
    if not any(
        r.entity_type == "line" and r.variable == "net_flow_mw" for r in results
    ):
        return list(results)
    lines = novomodelo_readers.read_novomodelo_lines(novomodelo_output_dir)
    default_cost = (
        novomodelo_readers.read_novomodelo_default_exchange_cost(novomodelo_output_dir)
        if any("exchange_cost" not in line for line in lines)
        else None
    )
    loops = find_tied_loops(lines, default_cost)
    tied_ids = frozenset().union(*(loop.line_ids for loop in loops))

    marked: list[ResultComparison] = []
    moved: set[int] = set()
    for r in results:
        if (
            r.entity_type == "line"
            and r.variable == "net_flow_mw"
            and r.novomodelo_id in tied_ids
        ):
            moved.add(r.novomodelo_id)
            r = replace(r, variable=TIED_NET_FLOW)
        marked.append(r)

    reported = [loop for loop in loops if loop.line_ids & moved]
    if reported:
        bus_names = {
            int(bus["id"]): str(bus["name"])
            for bus in novomodelo_readers.read_novomodelo_buses(novomodelo_output_dir)
            if bus.get("name")
        }
        _emit_tied_loops(reported, bus_names)
    return marked


def _emit_tied_loops(loops: Sequence[TiedLoop], bus_names: Mapping[int, str]) -> None:
    emit(
        Diagnostic(
            code="line-routes-tied",
            severity=Severity.WARNING,
            category="Compare data",
            title=f"Line flows on equal-cost loops not judged ({len(loops)})",
            summary=(
                "The exchange costs of the lines on each listed loop split "
                "into two halves of equal cost, so the source model and "
                "Novomodelo can carry the same exchanges between buses along "
                "either half at the same cost, and each settles on its own "
                "split. The flows on those lines differ even where the buses' "
                "net imports agree. The comparison shows them as Net flow on "
                "tied routes and leaves them out of the headline and the "
                "--json status; compare Net import on the buses instead."
            ),
            table=DiagnosticTable(
                columns=["Loop", "Equal-cost halves", "Cost per half (R$/MWh)"],
                rows=[
                    [
                        " – ".join(bus_names.get(b, f"bus {b}") for b in loop.buses),
                        " = ".join(" + ".join(half) for half in loop.halves),
                        f"{loop.cost:.6g}",
                    ]
                    for loop in loops
                ],
                justify=["left", "left", "right"],
            ),
            notes=[
                "A route through a fictitious bus costs the same as the "
                "direct line because the source model halves the exchange "
                "penalty on each line that touches a fictitious bus."
            ],
        ),
        logger=_LOG,
    )
