"""Hydro geometry: the VHA volume->height->area table, the storage range a
plant operates over, the machine configuration an expanding plant declares, and FPHA
eligibility (the volume->cota polynomial, the specific productivity, and the
rated capacity of that declared configuration).

Depends only on :mod:`.overrides` within the package.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa

from novomodelo_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    emit,
)
from novomodelo_bridge.core.hydro_units import (
    fpha_zero_capacity_diagnostic,
    rated_capacity,
)
from novomodelo_bridge.newave.case import NewaveCase
from novomodelo_bridge.newave.converters.hydro.overrides import (
    _apply_permanent_overrides,
)
from novomodelo_bridge.newave.filling import ExpansionConfig, exph_unit_rows
from novomodelo_bridge.newave.id_map import NewaveIdMap

_LOG = logging.getLogger(__name__)


def _is_fpha_eligible(hreg: pd.Series) -> bool:
    """Whether a hydro plant can be fit by novomodelo's *computed* FPHA.

    Requires a non-degenerate volume→cota polynomial (the forebay curve), a
    positive specific productivity ``rho_esp`` (needed to derive the
    dimensionless turbine efficiency), and a positive rated turbined flow
    **and** rated power (:func:`~novomodelo_bridge.core.hydro_units.rated_capacity`
    over the declared machine sets). novomodelo samples the fit on
    ``[0, max_turbined]`` and clamps it at ``max_generation``, so a zero on
    either side collapses the production cloud and aborts the fit. Storage
    swing is **not** required: run-of-river / zero-storage plants
    (``vmax == vmin``) emit a single VHA geometry row and novomodelo fits them
    through the single-volume FPHA path (γ_V = 0), matching the source model,
    which fits these plants with ``Npt_V = 1``.
    """
    return _has_fpha_curve_inputs(hreg) and _has_rated_capacity(hreg)


def _has_fpha_curve_inputs(hreg: pd.Series) -> bool:
    coeffs = [float(hreg[f"a{i}_volume_cota"]) for i in range(5)]
    if all(c == 0.0 for c in coeffs):
        return False
    rho_esp_raw = hreg.get("produtibilidade_especifica")
    if rho_esp_raw is None:
        return False
    rho_esp = float(rho_esp_raw)
    return not math.isnan(rho_esp) and rho_esp > 0.0


def _has_rated_capacity(hreg: pd.Series) -> bool:
    max_turbined, max_generation = rated_capacity(hreg)
    return max_turbined > 0.0 and max_generation > 0.0


def held_volume(hreg: pd.Series) -> float | None:
    """The ``volume_referencia`` (hm³) the source model holds a daily ('D') or
    run-of-river ('S') plant at every stage, which may lie outside
    ``[volume_minimo, volume_maximo]``; ``None`` for any other plant and for a
    'D'/'S' plant without one."""
    if str(hreg.get("tipo_regulacao", "")).strip() not in ("D", "S"):
        return None
    vol_ref_raw = hreg.get("volume_referencia")
    if vol_ref_raw is None or pd.isna(vol_ref_raw):
        return None
    return float(vol_ref_raw)


def operating_storage_range(hreg: pd.Series) -> tuple[float, float]:
    """``(min, max)`` storage the plant operates over, in hm³: the
    :func:`held_volume` when there is one, else the hidr range. The reservoir
    bounds, the VHA geometry and the FPHA volume window all read this one range.
    """
    held = held_volume(hreg)
    if held is not None:
        return held, held
    return float(hreg["volume_minimo"]), float(hreg["volume_maximo"])


def _expansion_configs(
    case: NewaveCase, cadastro: pd.DataFrame
) -> dict[int, ExpansionConfig]:
    """``{code: config}`` for every active ``EE`` plant with a machine still to enter.

    *cadastro* is the permanent-override-corrected registry (the study-start
    configuration); the uncorrected ``case.hidr.cadastro`` holds the final one.
    """
    exph_df = case.exph.expansoes if case.exph is not None else None
    if exph_df is None or exph_df.empty:
        return {}
    registry = case.hidr.cadastro
    horizon = case.horizon
    configs: dict[int, ExpansionConfig] = {}
    for _, row in case.active_hydros.iterrows():
        code = int(row["codigo_usina"])
        if str(row["usina_existente"]).strip() != "EE" or code not in cadastro.index:
            continue
        unit_rows = exph_unit_rows(
            exph_df, code, horizon.start_year, horizon.start_month
        )
        if not unit_rows:
            continue
        final = registry.loc[code]
        start = cadastro.loc[code]
        n_final = int(final["numero_conjuntos_maquinas"])
        n_start = int(start["numero_conjuntos_maquinas"])
        configs[code] = ExpansionConfig(
            registry_counts={
                c: int(final[f"maquinas_conjunto_{c}"]) for c in range(1, n_final + 1)
            },
            start_counts={
                c: int(start[f"maquinas_conjunto_{c}"]) if c <= n_start else 0
                for c in range(1, n_final + 1)
            },
            unit_rows=unit_rows,
        )
    return configs


def fpha_eligible_codes(case: NewaveCase) -> set[int]:
    """The source model plant codes emitted as ``model: "fpha"`` for this case.

    Empty unless :attr:`NewaveCase.fpha_enabled`. The single source of truth for
    FPHA eligibility, shared by :func:`convert_hydros`,
    :func:`convert_production_models`, and :func:`convert_hydro_energy_productivity`
    so the three files agree on which plants are FPHA. Rated capacity is read on
    the machine configuration :func:`convert_hydros` declares, so an expanding
    plant with no machine in service at the study start still qualifies. A
    plant passing every check but rated capacity is reported through
    :func:`~novomodelo_bridge.core.hydro_units.fpha_zero_capacity_diagnostic`.
    """
    if not case.fpha_enabled:
        return set()
    cadastro = _apply_permanent_overrides(case.hidr.cadastro, case)
    expansion = _expansion_configs(case, cadastro)
    eligible: set[int] = set()
    zero_capacity: list[tuple[str, int, float, float]] = []
    for _, row in case.active_hydros.iterrows():
        code = int(row["codigo_usina"])
        if code not in cadastro.index:
            continue
        hreg = cadastro.loc[code]
        if not _has_fpha_curve_inputs(hreg):
            continue
        config = expansion.get(code)
        declared = hreg if config is None else config.declared_hreg(hreg)
        if _has_rated_capacity(declared):
            eligible.add(code)
        else:
            zero_capacity.append(
                (str(hreg["nome_usina"]).strip(), code, *rated_capacity(declared))
            )
    if zero_capacity:
        emit(fpha_zero_capacity_diagnostic(zero_capacity), logger=_LOG)
    return eligible


_EVAP_MONTHS = [
    "JAN",
    "FEV",
    "MAR",
    "ABR",
    "MAI",
    "JUN",
    "JUL",
    "AGO",
    "SET",
    "OUT",
    "NOV",
    "DEZ",
]


_GEOMETRY_N_POINTS = 100

_GEOMETRY_SCHEMA = pa.schema(
    [
        pa.field("hydro_id", pa.int32()),
        pa.field("volume_hm3", pa.float64()),
        pa.field("height_m", pa.float64()),
        pa.field("area_km2", pa.float64()),
    ]
)


def _eval_poly(coeffs: list[float], x: np.ndarray) -> np.ndarray:
    """Evaluate a 4th-degree polynomial: c0 + c1*x + ... + c4*x^4."""
    return (
        coeffs[0]
        + coeffs[1] * x
        + coeffs[2] * x**2
        + coeffs[3] * x**3
        + coeffs[4] * x**4
    )


def _vha_points(
    vc_coeffs: list[float], ca_coeffs: list[float], vol_min: float, vol_max: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(volume, height, area)`` samples of the VHA curve: one point for a
    single-volume range, else :data:`_GEOMETRY_N_POINTS` uniformly spaced ones,
    with negative heights and areas clamped to 0.0."""
    vol_grid: np.ndarray = (
        np.array([vol_min])
        if vol_min == vol_max
        else np.linspace(vol_min, vol_max, _GEOMETRY_N_POINTS)
    )
    height_arr: np.ndarray = np.maximum(_eval_poly(vc_coeffs, vol_grid), 0.0)
    area_arr: np.ndarray = np.maximum(_eval_poly(ca_coeffs, height_arr), 0.0)
    return vol_grid, height_arr, area_arr


def _report_geometry_skips(skips: list[tuple[str, int, str]]) -> None:
    if not skips:
        return
    emit(
        Diagnostic(
            code="hydro-geometry-skipped",
            severity=Severity.WARNING,
            category="Hydro geometry",
            title=f"Hydro geometry skipped ({len(skips)} plant(s))",
            summary=(
                f"{len(skips)} plant(s) had no geometry generated: "
                "not present in the cadastro, or an all-zero "
                "volume-to-height polynomial."
            ),
            table=DiagnosticTable(
                columns=["Plant", "Code", "Reason"],
                rows=[[name, code, reason] for name, code, reason in skips],
                justify=["left", "right", "left"],
            ),
        ),
        logger=_LOG,
    )


def _report_geometry_extensions(rows: list[list[object]]) -> None:
    if not rows:
        return
    emit(
        Diagnostic(
            code="hydro-geometry-extended",
            severity=Severity.INFO,
            category="Hydro geometry",
            title=(
                f"Hydro geometry extended past the storage range ({len(rows)} plant(s))"
            ),
            summary=(
                f"{len(rows)} plant(s) have an FPHA reference volume "
                "outside [volume_minimo, volume_maximo], where the source "
                "model reads the downstream level of the plant above. Their "
                "geometry samples the volume-to-height polynomial out to "
                "those volumes."
            ),
            table=DiagnosticTable(
                columns=[
                    "Plant",
                    "Code",
                    "volume_minimo (hm³)",
                    "volume_maximo (hm³)",
                    "Sampled from (hm³)",
                    "Sampled to (hm³)",
                ],
                rows=rows,
                justify=["left", "right", "right", "right", "right", "right"],
            ),
        ),
        logger=_LOG,
    )


def generate_hydro_geometry(
    cadastro: pd.DataFrame,
    id_map: NewaveIdMap,
    reference_volumes: Mapping[int, Sequence[float]] | None = None,
) -> pa.Table:
    """Generate a VHA curve table for all hydro plants in *id_map*.

    For each plant code in ``id_map.all_hydro_codes``, samples
    :data:`_GEOMETRY_N_POINTS` uniformly spaced volume points on the plant's
    :func:`operating_storage_range`, evaluates the volume-to-height polynomial
    (``a0_volume_cota`` through ``a4_volume_cota``), then evaluates the
    height-to-area polynomial (``a0_cota_area`` through ``a4_cota_area``), and
    collects the results into a PyArrow Table.

    A plant whose range is a single volume emits one point there, so novomodelo
    reads the forebay level at the volume the reservoir is held at, for the
    plant's own head and as the downstream level of the plant above.  Any
    other plant's range extends to cover its ``reference_volumes``, because
    novomodelo reads the downstream level there and clamps a volume outside the
    geometry to its range; the source model evaluates the polynomial outside
    ``[volume_minimo, volume_maximo]`` for it too.  Plants
    whose volume_cota polynomial coefficients are all zero are skipped and
    reported via a diagnostic.  Negative height or area values produced by the
    polynomials are clamped to 0.0.

    Parameters
    ----------
    cadastro:
        The ``Hidr.cadastro`` DataFrame (indexed by ``codigo_usina``) with
        permanent MODIF.DAT overrides already applied.
    id_map:
        Pre-built ID mapping; ``id_map.all_hydro_codes`` determines which
        plants are processed.
    reference_volumes:
        :func:`~novomodelo_bridge.newave.converters.hydro.reference_volumes.fpha_reference_volumes`
        of the case, keyed by plant code.

    Returns
    -------
    pa.Table
        Schema: ``(hydro_id: INT32, volume_hm3: DOUBLE, height_m: DOUBLE,
        area_km2: DOUBLE)``.  One row per sampled volume point across all
        eligible plants, ordered by plant then by volume.
    """
    hydro_ids: list[int] = []
    volumes: list[float] = []
    heights: list[float] = []
    areas: list[float] = []

    # Loop-accumulate-then-emit-once: one record per skipped plant, emitted
    # after the loop so both skip causes survive finalize_diagnostics' de-dup.
    geometry_skips: list[tuple[str, int, str]] = []
    extended: list[list[object]] = []

    for newave_code in id_map.all_hydro_codes:
        if newave_code not in cadastro.index:
            geometry_skips.append(("?", newave_code, "plant not in cadastro"))
            continue

        hreg = cadastro.loc[newave_code]
        vol_min, vol_max = operating_storage_range(hreg)

        # Polynomial coefficients for volume -> height (hm3 -> m).
        vc_coeffs = [float(hreg[f"a{i}_volume_cota"]) for i in range(5)]
        if all(c == 0.0 for c in vc_coeffs):
            name = str(hreg.get("nome_usina", "?"))
            geometry_skips.append(
                (name, newave_code, "all volume-to-height polynomial coefficients zero")
            )
            continue

        # Polynomial coefficients for height -> area (m -> km2).
        ca_coeffs = [float(hreg[f"a{i}_cota_area"]) for i in range(5)]

        plant_references = (reference_volumes or {}).get(newave_code)
        if plant_references and vol_min < vol_max:
            sampled = (
                min(vol_min, *plant_references),
                max(vol_max, *plant_references),
            )
            if sampled != (vol_min, vol_max):
                extended.append(
                    [
                        str(hreg.get("nome_usina", "?")).strip(),
                        newave_code,
                        *(round(v, 2) for v in (vol_min, vol_max, *sampled)),
                    ]
                )
                vol_min, vol_max = sampled

        novomodelo_id = id_map.hydro_id(newave_code)
        vol_grid, height_arr, area_arr = _vha_points(
            vc_coeffs, ca_coeffs, vol_min, vol_max
        )
        hydro_ids.extend([novomodelo_id] * len(vol_grid))
        volumes.extend(vol_grid.tolist())
        heights.extend(height_arr.tolist())
        areas.extend(area_arr.tolist())

    _report_geometry_skips(geometry_skips)
    _report_geometry_extensions(extended)

    return pa.table(
        {
            "hydro_id": pa.array(hydro_ids, type=pa.int32()),
            "volume_hm3": pa.array(volumes, type=pa.float64()),
            "height_m": pa.array(heights, type=pa.float64()),
            "area_km2": pa.array(areas, type=pa.float64()),
        },
        schema=_GEOMETRY_SCHEMA,
    )
