"""FPHA (hydro production function) conversion for DECOMP-like decks.

novomodelo evaluates reservoir generation with a *computed* FPHA: given the plant
geometry (a volume→height→area VHA curve), the tailrace curve families, the
specific productivity ``ρ_esp`` and turbine efficiency, and a volume fitting
window, novomodelo fits the production-function hyperplanes itself. This module
builds the DECOMP-side inputs, mirroring :mod:`novomodelo_bridge.newave.converters.hydro`
and :mod:`novomodelo_bridge.newave.converters.tailrace` on the source-model side and reusing
their shared cores (:func:`~novomodelo_bridge.core.tailrace.build_tailrace_table`,
:func:`~novomodelo_bridge.core.productivity.fpha_efficiency`).

Eligibility mirrors the source-model rule: a plant with a non-degenerate
volume→cota polynomial (the forebay curve), a positive ``ρ_esp``, and a positive
``AC``-adjusted rated turbined flow and power. Ineligible plants keep the
constant-productivity path.

**Fitting window.** Each plant is fit over its whole storage envelope
(:func:`~novomodelo_bridge.decomp.converters.cadastro.storage_envelope`, the
range the entity ``reservoir`` block declares) — a single volume for a
run-of-river plant — and :func:`convert_hydro_geometry` samples the same range,
since novomodelo clamps the window to the geometry table. Never a band around
the initial volume: novomodelo's planes are an outer approximation only inside
the window, and below it they turn negative at zero flow, forcing a minimum
turbined flow that makes a drawn-down stage infeasible.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pyarrow as pa

from novomodelo_bridge.core.diagnostics import Diagnostic, Severity, emit
from novomodelo_bridge.core.hydro_units import fpha_zero_capacity_diagnostic
from novomodelo_bridge.core.productivity import (
    evaluate_cota,
    fpha_efficiency,
    fpha_spillage_fit_max_m3s,
)
from novomodelo_bridge.core.tailrace import build_tailrace_table
from novomodelo_bridge.decomp.converters.cadastro import storage_envelope
from novomodelo_bridge.decomp.converters.hydro.bounds import _rated_envelope

if TYPE_CHECKING:
    from pathlib import Path

    from idecomp.decomp import Mlt

    from novomodelo_bridge.decomp.case import DecompCase
    from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro
    from novomodelo_bridge.decomp.id_map import DecompIdMap

_LOG = logging.getLogger(__name__)

#: Number of volume samples per plant in the VHA geometry table — matches
#: :func:`novomodelo_bridge.newave.converters.hydro.generate_hydro_geometry`.
_GEOMETRY_N_POINTS = 100

_GEOMETRY_SCHEMA = pa.schema(
    [
        pa.field("hydro_id", pa.int32(), nullable=False),
        pa.field("volume_hm3", pa.float64(), nullable=False),
        pa.field("height_m", pa.float64(), nullable=False),
        pa.field("area_km2", pa.float64(), nullable=False),
    ]
)


def read_polinjus(path: Path):  # noqa: ANN201 (idecomp.libs type is optional)
    """Read the ``polinjus`` tailrace-curve-family cadastro, or ``None``.

    Returns an ``idecomp.libs.UsinasHidreletricas`` (the same accessor surface
    the source model's ``Polinjus`` exposes) or ``None`` when the deck ships no
    ``polinjus`` file.
    """
    if not path.exists():
        return None
    from idecomp.libs import UsinasHidreletricas

    return UsinasHidreletricas.read(str(path))


def convert_tailrace_curves(case: DecompCase, id_map: DecompIdMap) -> pa.Table | None:
    """Build ``system/tailrace_curves.parquet`` from the ``polinjus`` families.

    Thin DECOMP wrapper over the shared
    :func:`~novomodelo_bridge.core.tailrace.build_tailrace_table` core — the
    ``polinjus`` object exposes the identical ``hidreletrica_curvajusante`` /
    ``…_polinomio_segmento`` frames the source-model side reads. ``None`` when
    the deck has no ``polinjus`` or no segment maps to a converted hydro.
    """
    polinjus = case.polinjus
    if polinjus is None:
        return None
    families = polinjus.hidreletrica_curvajusante(df=True)
    segments = polinjus.hidreletrica_curvajusante_polinomio_segmento(df=True)
    return build_tailrace_table(families, segments, id_map.hydro_id)


def is_fpha_eligible(effective: EffectiveCadastro, code: int) -> bool:
    """Whether plant *code* can be fit by novomodelo's computed FPHA.

    Requires a non-degenerate (post-``AC COTVOL``) volume→cota polynomial, a
    positive ``produtibilidade_especifica`` at the initial stage, and a
    positive rated turbined flow **and** rated power
    (:func:`~novomodelo_bridge.decomp.converters.hydro.bounds._rated_envelope`,
    the ``AC NUMCON``/``NUMMAQ``/``POTEFE``/``VAZEFE``-adjusted envelope
    ``hydros.json`` emits). novomodelo samples the fit on ``[0, max_turbined]`` and
    clamps it at ``max_generation``, so a zero on either side collapses the
    production cloud and aborts the fit. Storage swing is not required — a
    run-of-river plant fits through the single-volume path.
    """
    return _has_fpha_curve_inputs(effective, code) and _has_rated_capacity(
        effective, code
    )


def _has_fpha_curve_inputs(effective: EffectiveCadastro, code: int) -> bool:
    coeffs = effective.cota_polynomial(code, 0)
    if all(c == 0.0 for c in coeffs):
        return False
    return effective.value(code, "produtibilidade_especifica", 0) > 0.0


def _has_rated_capacity(effective: EffectiveCadastro, code: int) -> bool:
    max_turbined, max_generation = _rated_envelope(
        effective.base.loc[code], code, effective
    )
    return max_turbined > 0.0 and max_generation > 0.0


def fpha_eligible_codes(effective: EffectiveCadastro, id_map: DecompIdMap) -> set[int]:
    """The operated plant codes emitted as ``model: "fpha"``.

    Single source of truth shared by the ``hydros.json`` generation model, the
    production-model doc, and the energy-productivity parquet exclusion, so the
    three agree on which plants are FPHA. A plant passing every check but
    rated capacity is reported through
    :func:`~novomodelo_bridge.core.hydro_units.fpha_zero_capacity_diagnostic`.
    """
    eligible: set[int] = set()
    zero_capacity: list[tuple[str, int, float, float]] = []
    for code in id_map.hydro_codes:
        if code not in effective.base.index:
            continue
        if not _has_fpha_curve_inputs(effective, code):
            continue
        if _has_rated_capacity(effective, code):
            eligible.add(code)
        else:
            hreg = effective.base.loc[code]
            zero_capacity.append(
                (
                    str(hreg["nome_usina"]).strip(),
                    code,
                    *_rated_envelope(hreg, code, effective),
                )
            )
    if zero_capacity:
        emit(fpha_zero_capacity_diagnostic(zero_capacity), logger=_LOG)
    return eligible


def fitting_window(effective: EffectiveCadastro, code: int) -> dict[str, float]:
    """The FPHA volume ``fitting_window`` for plant *code*, in hm³: its storage
    envelope (see the module docstring)."""
    vmin, vmax = storage_envelope(effective, code)
    return {"volume_min_hm3": vmin, "volume_max_hm3": vmax}


def fpha_config(
    effective: EffectiveCadastro, mlt: Mlt | None, code: int, stage_index: int
) -> dict:
    """Plant *code*'s computed ``fpha_config`` at *stage_index*: its
    :func:`fitting_window` and, when :func:`spillage_fit_max` has one, its
    ``spillage_fit_max_m3s``."""
    config: dict = {
        "source": "computed",
        "fitting_window": fitting_window(effective, code),
    }
    spillage_max = spillage_fit_max(effective, mlt, code, stage_index)
    if spillage_max is not None:
        config["spillage_fit_max_m3s"] = spillage_max
    return config


def spillage_fit_max(
    effective: EffectiveCadastro, mlt: Mlt | None, code: int, stage_index: int
) -> float | None:
    """The FPHA spillage fit range (m³/s) of plant *code* at *stage_index*, from
    the ``mlt.dat`` means of the inflow gauge in force (``AC NUMPOS``, else the
    ``hidr.dat`` posto; see
    :func:`~novomodelo_bridge.core.productivity.fpha_spillage_fit_max_m3s`).

    ``None`` without ``mlt.dat``, for a gauge it does not list, or for a gauge
    without a positive mean.
    """
    table = mlt.valores if mlt is not None else None
    posto = effective.inflow_gauge(code, stage_index)
    if table is None or str(posto) not in table.columns:
        return None
    return fpha_spillage_fit_max_m3s(posto, table[str(posto)].tolist())


def mlt_absent_diagnostic(plant_count: int) -> Diagnostic:
    """WARNING for a deck with FPHA plants and no ``mlt.dat``."""
    return Diagnostic(
        code="fpha-mlt-absent",
        severity=Severity.WARNING,
        category="Hydro production model",
        title="FPHA spillage range without mlt.dat",
        summary=(
            f"The deck has no mlt.dat, so the spillage coefficient of {plant_count} "
            "FPHA plant(s) is fit over twice the plant's maximum turbined flow. "
            "DECOMP fits it over twice the largest monthly long-term mean inflow "
            "of the plant's gauge. A plant whose floods exceed the narrower range "
            "can have its spillage held down by its own production function."
        ),
        remediation="Add the deck's mlt.dat to the deck directory.",
    )


def turbine_efficiency(effective: EffectiveCadastro, code: int, name: str) -> float:
    """Dimensionless turbine efficiency ``η = ρ_esp / K`` for plant *code*.

    Reuses the source-model helper (clamps to 1.0 with a warning if the input
    ``ρ_esp`` implies an unphysical ``η > 1``).
    """
    rho_esp = effective.value(code, "produtibilidade_especifica", 0)
    return fpha_efficiency(rho_esp, name)


def convert_hydro_geometry(
    effective: EffectiveCadastro, id_map: DecompIdMap
) -> pa.Table:
    """Build ``system/hydro_geometry.parquet`` (volume→height→area VHA curves).

    Mirrors :func:`novomodelo_bridge.newave.converters.hydro.generate_hydro_geometry` but
    reads the **effective** (post-``AC COTVOL``) volume→cota polynomial and the
    base ``a*_cota_area`` height→area polynomial, sampling
    :data:`_GEOMETRY_N_POINTS` uniform volume points over the plant's storage
    envelope (the :func:`fitting_window` range). Emitted for every operated plant with a
    non-degenerate cota polynomial (so any plant is a valid backwater reference
    for its upstream FPHA neighbour); a run-of-river plant (``vmin == vmax``)
    contributes a single point; heights/areas are clamped to ``>= 0``.
    """
    hydro_ids: list[int] = []
    volumes: list[float] = []
    heights: list[float] = []
    areas: list[float] = []

    for code in id_map.hydro_codes:
        if code not in effective.base.index:
            continue
        vc_coeffs = effective.cota_polynomial(code, 0)
        if all(c == 0.0 for c in vc_coeffs):
            continue
        base = effective.base.loc[code]
        ca_coeffs = [float(base[f"a{i}_cota_area"]) for i in range(5)]
        vmin, vmax = storage_envelope(effective, code)
        hydro_id = id_map.hydro_id(code)

        if vmax <= vmin:
            grid = [vmin]
        else:
            step = (vmax - vmin) / (_GEOMETRY_N_POINTS - 1)
            grid = [vmin + step * i for i in range(_GEOMETRY_N_POINTS)]

        for v in grid:
            h = max(evaluate_cota(vc_coeffs, v), 0.0)
            a = max(evaluate_cota(ca_coeffs, h), 0.0)
            hydro_ids.append(hydro_id)
            volumes.append(v)
            heights.append(h)
            areas.append(a)

    return pa.table(
        {
            "hydro_id": pa.array(hydro_ids, type=pa.int32()),
            "volume_hm3": pa.array(volumes, type=pa.float64()),
            "height_m": pa.array(heights, type=pa.float64()),
            "area_km2": pa.array(areas, type=pa.float64()),
        },
        schema=_GEOMETRY_SCHEMA,
    )
