"""Hydro unit-group construction shared by both conversion tracks.

``build_mirror_unit_group`` is the single builder for novomodelo's mandatory
``unit_groups`` array entries — reused verbatim by the source-model track
(:mod:`novomodelo_bridge.newave.converters.hydro.entity`) and the DECOMP track
(:mod:`novomodelo_bridge.decomp.converters.hydro`). :func:`rated_capacity` is the
nameplate sum over a ``hidr`` row's machine sets, and
:func:`fpha_zero_capacity_diagnostic` reports the plants either track keeps
off novomodelo's computed FPHA because that capacity is zero.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from novomodelo_bridge.core.diagnostics import Diagnostic, DiagnosticTable, Severity

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd


def rated_capacity(hreg: pd.Series) -> tuple[float, float]:
    """Return ``(max_turbined, max_generation)`` as the rated nameplate capacity:
    ``Σ_c (n_c · q_nom_c)`` for flow and ``Σ_c (n_c · p_nom_c)`` for power over
    the row's ``numero_conjuntos_maquinas`` machine sets, with **no** TEIF/IP
    availability derating and no head correction.

    The source model emits the power value ``[1]`` as every plant's
    ``max_generation`` (independent of the production function): it equals the
    source model's installed-capacity ceiling / FPHA ``GHmax`` exactly
    (verified TUCURUI 7445, QUEBRA QUEIX 120). The flow value ``[0]``
    (``Σ n·q_nom``) is the source model's fitting-grid ``Qmax``, **not** the
    operational turbined cap, which is head-corrected.
    """
    n_sets = int(hreg["numero_conjuntos_maquinas"])
    max_turbined = 0.0
    max_generation = 0.0
    for i in range(1, n_sets + 1):
        n_machines = int(hreg[f"maquinas_conjunto_{i}"])
        max_turbined += float(hreg[f"vazao_nominal_conjunto_{i}"]) * n_machines
        max_generation += float(hreg[f"potencia_nominal_conjunto_{i}"]) * n_machines
    return max_turbined, max_generation


def fpha_zero_capacity_diagnostic(
    plants: Sequence[tuple[str, int, float, float]],
) -> Diagnostic:
    """INFO diagnostic for plants kept on ``constant_productivity`` because
    their rated turbined flow or rated power is zero.

    Each entry of *plants* is ``(name, code, max_turbined_m3s,
    max_generation_mw)``. novomodelo's computed FPHA samples ``[0, max_turbined]``
    and clamps at ``max_generation``; a zero on either side collapses the
    production cloud and aborts the fit, so such a plant cannot be ``fpha``.
    """
    return Diagnostic(
        code="fpha-zero-capacity",
        severity=Severity.INFO,
        category="Hydro production model",
        title=f"FPHA skipped for zero-capacity plants ({len(plants)} plant(s))",
        summary=(
            f"{len(plants)} plant(s) eligible for the computed FPHA have zero "
            "rated turbined flow or rated power after overrides and keep "
            "constant productivity."
        ),
        table=DiagnosticTable(
            columns=["Plant", "Code", "Turbined (m3/s)", "Generation (MW)"],
            rows=[[name, code, q, p] for name, code, q, p in plants],
            justify=["left", "right", "right", "right"],
        ),
    )


def build_mirror_unit_group(
    *,
    name: str,
    bus_id: int,
    min_generation_mw: float,
    max_generation_mw: float,
    min_turbined_m3s: float,
    max_turbined_m3s: float,
    group_id: int = 0,
) -> dict[str, object]:
    """Build one "mirror" unit group for a hydro plant.

    novomodelo requires every hydro to declare a non-empty ``unit_groups`` array
    (``RawUnitGroup``, all seven fields present). For the ordinary,
    single-group plant every caller in the bridge emits, the group's bounds
    *mirror* the plant's own generation envelope verbatim — no clamping, no
    zeroing of minima, no recomputation — and with a single group,
    ``sum(group maxima) == plant maximum`` holds trivially, which is exactly
    what novomodelo rule 41 checks, so the rule is satisfied by construction.

    A plant whose halves are maintained independently (e.g. a two-frequency
    split) instead calls this twice, once per physically separate group,
    passing each group's own conjunto-backed bounds (not the plant's) and a
    distinct ``group_id`` — the caller is responsible for the group maxima
    still summing to the plant envelope (rule 41) and for the ids being
    unique within the plant (novomodelo rule 39; the overlay is id-addressed, so
    array order is never load-bearing — see ``decomp/group_bounds.py``).

    Parameters
    ----------
    name:
        The plant name, used verbatim as the group name.
    bus_id:
        The group's bus id (the plant's own bus for every group, when every
        group sits on the same bus).
    min_generation_mw, max_generation_mw:
        The group's generation bounds in MW.
    min_turbined_m3s, max_turbined_m3s:
        The group's turbined-flow bounds in m^3/s.
    group_id:
        The group's ``id``. Defaults to ``0`` — unit-group ids are dense and
        0-based within a plant, and a single mirror group is the plant's
        only (hence first) group. Pass a distinct value for each group of a
        multi-group plant.

    Returns
    -------
    dict[str, object]
        Exactly the seven ``RawUnitGroup`` keys required by
        ``hydros.schema.json`` — no extras.
    """
    return {
        "id": group_id,
        "name": name,
        "bus_id": bus_id,
        "min_generation_mw": min_generation_mw,
        "max_generation_mw": max_generation_mw,
        "min_turbined_m3s": min_turbined_m3s,
        "max_turbined_m3s": max_turbined_m3s,
    }
