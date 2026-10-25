"""Builders used only by tests/decomp/."""

from __future__ import annotations

import math
from collections.abc import Mapping

from novomodelo_bridge.core.tailrace import TailraceFamily, TailraceSegment
from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.converters.cadastro import (
    EffectiveCadastro,
    effective_storage_range,
)
from novomodelo_bridge.decomp.converters.hydro import (
    TurbineLimits,
    compute_turbine_limits,
    stage_availability_factors,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap


def flat_turbine_limits(
    id_map: DecompIdMap,
    n_stages: int,
    cap_m3s: float = 100.0,
    *,
    group_counts: Mapping[int, int] | None = None,
) -> TurbineLimits:
    """Every operated plant's turbined-flow cap at *cap_m3s* on every stage.

    A plant has one unit group unless *group_counts* gives its count by code.
    """
    counts = group_counts or {}
    return TurbineLimits(
        {
            (code, group_id): (cap_m3s,) * n_stages
            for code in id_map.hydro_codes
            for group_id in range(counts.get(code, 1))
        }
    )


def flat_tailrace_turbine_limits(
    case: DecompCase,
    id_map: DecompIdMap,
    effective: EffectiveCadastro,
    *,
    tailrace_m: float,
) -> TurbineLimits:
    """*case*'s computed turbined-flow limits with every plant's tailrace flat
    at *tailrace_m*, its storage at its stage-0 minimum, and zero inflow."""
    tailrace = (
        TailraceFamily(
            math.nan, (TailraceSegment(math.inf, (tailrace_m, 0.0, 0.0, 0.0, 0.0)),)
        ),
    )
    return compute_turbine_limits(
        effective,
        id_map,
        case.calendar,
        initial_volumes={
            code: effective_storage_range(effective, code, 0)[0]
            for code in id_map.hydro_codes
        },
        incremental_inflows={
            (code, stage.index): 0.0
            for code in id_map.hydro_codes
            for stage in case.calendar
        },
        tailrace=dict.fromkeys(id_map.hydro_codes, tailrace),
        availability_factors=stage_availability_factors(
            case, id_map, effective=effective
        ),
    )
