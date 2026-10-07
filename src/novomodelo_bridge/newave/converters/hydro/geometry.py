"""Hydro geometry: the VHA volume->height->area table, seasonal reference-volume
reads, the machine configuration an expanding plant declares, and FPHA
eligibility (the volume->cota polynomial, the specific productivity, and the
rated capacity of that declared configuration).

Depends only on :mod:`.overrides` within the package.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import pyarrow as pa

from cobre_bridge.core.diagnostics import Diagnostic, DiagnosticTable, Severity, emit
from cobre_bridge.core.hydro_units import fpha_zero_capacity_diagnostic, rated_capacity
from cobre_bridge.newave.case import NewaveCase
from cobre_bridge.newave.converters.hydro.overrides import _apply_permanent_overrides
from cobre_bridge.newave.filling import ExpansionConfig, exph_unit_rows
from cobre_bridge.newave.id_map import NewaveIdMap

_LOG = logging.getLogger(__name__)


def _is_fpha_eligible(hreg: pd.Series) -> bool:
    """Whether a hydro plant can be fit by cobre's *computed* FPHA.

    Requires a non-degenerate volume→cota polynomial (the forebay curve), a
    positive specific productivity ``rho_esp`` (needed to derive the
    dimensionless turbine efficiency), and a positive rated turbined flow
    **and** rated power (:func:`~cobre_bridge.core.hydro_units.rated_capacity`
    over the declared machine sets). cobre samples the fit on
    ``[0, max_turbined]`` and clamps it at ``max_generation``, so a zero on
    either side collapses the production cloud and aborts the fit. Storage
    swing is **not** required: run-of-river / zero-storage plants
    (``vmax == vmin``) emit a single VHA geometry row and cobre fits them
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
    :func:`~cobre_bridge.core.hydro_units.fpha_zero_capacity_diagnostic`.
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


def _seasonal_reference_volume(
    monthly_useful: dict[int, float],
    vmin: float,
    vmax: float,
    *,
    is_fpha: bool,
    fpha_config: dict | None = None,
) -> list[dict]:
    """Build 12 seasonal ``reference_volume`` entries from a volref_saz row.

    ``season_id = calendar_month - 1`` (matching the ``stages.json`` season map). The
    source model stores the seasonal reference as *useful* storage above ``vmin``, so
    the absolute reference volume is ``clamp(vmin + useful, vmin, vmax)`` (a missing or
    zero month means "operate at vmin"). FPHA seasons additionally carry ``fpha_config``
    (see :func:`_fpha_computed_config`); non-FPHA seasons declare only the reference
    volume (their productivity comes from ``hydro_energy_productivity.parquet``), so the
    value serves purely as the backwater reference for the upstream FPHA plant.
    """
    seasons: list[dict] = []
    for month in range(1, 13):
        useful = monthly_useful.get(month, 0.0)
        abs_vol = max(vmin, min(vmax, vmin + useful))
        season: dict = {
            "season_id": month - 1,
            "model": "fpha" if is_fpha else "constant_productivity",
            "reference_volume": {"volume_hm3": abs_vol},
        }
        if is_fpha and fpha_config is not None:
            season["fpha_config"] = fpha_config
        seasons.append(season)
    return seasons


def _read_volref_saz(case: NewaveCase) -> dict[int, dict[int, float]]:
    """Read ``volref_saz.dat`` into ``{plant_code: {calendar_month: useful_vol_hm3}}``.

    The source model uses two distinct conventions inside this file:

    - **Row of all-zeros** — sentinel meaning "no seasonal reference for this
      plant"; the source model falls back to its altura_65 / volume_referencia default.
      We mirror this by *excluding* the plant from the returned mapping.
    - **Row with at least one non-zero value** — real seasonal reference.
      Individual zero months in such a row mean "operate at exactly
      ``volume_minimo``" (useful = 0 above the dead-storage minimum), so we
      keep all twelve monthly entries including explicit zeros.

    Returns an empty dict when ``volref_saz.dat`` is absent.
    """
    vs = case.volref_saz
    if vs is None:
        _LOG.debug("volref_saz.dat not found; seasonal productivity disabled.")
        return {}

    df = vs.volumes
    if df is None or df.empty:
        return {}

    by_plant: dict[int, dict[int, float]] = {}
    for _, row in df.iterrows():
        code = int(row["codigo_usina"])
        month = int(row["mes"])
        by_plant.setdefault(code, {})[month] = float(row["valor"])

    return {
        code: months
        for code, months in by_plant.items()
        if any(v > 0.0 for v in months.values())
    }


def generate_hydro_geometry(cadastro: pd.DataFrame, id_map: NewaveIdMap) -> pa.Table:
    """Generate a VHA curve table for all hydro plants in *id_map*.

    For each plant code in ``id_map.all_hydro_codes``, samples 100 uniformly
    spaced volume points on ``[volume_minimo, volume_maximo]``, evaluates the
    volume-to-height polynomial (``a0_volume_cota`` through
    ``a4_volume_cota``), then evaluates the height-to-area polynomial
    (``a0_cota_area`` through ``a4_cota_area``), and collects the results
    into a PyArrow Table.

    Plants where ``volume_minimo == volume_maximo`` (run-of-river with no
    reservoir) are skipped.  Plants whose volume_cota polynomial coefficients
    are all zero are skipped and reported via a diagnostic.  Negative height
    or area values produced by the polynomials are clamped to 0.0.

    Parameters
    ----------
    cadastro:
        The ``Hidr.cadastro`` DataFrame (indexed by ``codigo_usina``) with
        permanent MODIF.DAT overrides already applied.
    id_map:
        Pre-built ID mapping; ``id_map.all_hydro_codes`` determines which
        plants are processed.

    Returns
    -------
    pa.Table
        Schema: ``(hydro_id: INT32, volume_hm3: DOUBLE, height_m: DOUBLE,
        area_km2: DOUBLE)``.  One row per sampled volume point across all
        eligible plants, ordered by plant then by volume.
    """
    _N_POINTS = 100

    hydro_ids: list[int] = []
    volumes: list[float] = []
    heights: list[float] = []
    areas: list[float] = []

    # Loop-accumulate-then-emit-once: one record per skipped plant, emitted
    # after the loop so both skip causes survive finalize_diagnostics' de-dup.
    geometry_skips: list[tuple[str, int, str]] = []

    def _eval_poly(coeffs: list[float], x: np.ndarray) -> np.ndarray:
        """Evaluate a 4th-degree polynomial: c0 + c1*x + ... + c4*x^4."""
        return (
            coeffs[0]
            + coeffs[1] * x
            + coeffs[2] * x**2
            + coeffs[3] * x**3
            + coeffs[4] * x**4
        )

    for newave_code in id_map.all_hydro_codes:
        if newave_code not in cadastro.index:
            geometry_skips.append(("?", newave_code, "plant not in cadastro"))
            continue

        hreg = cadastro.loc[newave_code]
        vol_min = float(hreg["volume_minimo"])
        vol_max = float(hreg["volume_maximo"])

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

        cobre_id = id_map.hydro_id(newave_code)

        if vol_min == vol_max:
            # Run-of-river or fixed-level: emit a single geometry point
            # so evaporation can still use the surface area.
            v = np.array([vol_min])
            h = _eval_poly(vc_coeffs, v)
            h = np.maximum(h, 0.0)
            a = _eval_poly(ca_coeffs, h)
            a = np.maximum(a, 0.0)
            hydro_ids.append(cobre_id)
            volumes.append(float(v[0]))
            heights.append(float(h[0]))
            areas.append(float(a[0]))
            continue

        vol_grid: np.ndarray = np.linspace(vol_min, vol_max, _N_POINTS)
        height_arr: np.ndarray = _eval_poly(vc_coeffs, vol_grid)
        height_arr = np.maximum(height_arr, 0.0)

        area_arr: np.ndarray = _eval_poly(ca_coeffs, height_arr)
        area_arr = np.maximum(area_arr, 0.0)

        hydro_ids.extend([cobre_id] * _N_POINTS)
        volumes.extend(vol_grid.tolist())
        heights.extend(height_arr.tolist())
        areas.extend(area_arr.tolist())

    if geometry_skips:
        emit(
            Diagnostic(
                code="hydro-geometry-skipped",
                severity=Severity.WARNING,
                category="Hydro geometry",
                title=f"Hydro geometry skipped ({len(geometry_skips)} plant(s))",
                summary=(
                    f"{len(geometry_skips)} plant(s) had no geometry generated: "
                    "not present in the cadastro, or an all-zero "
                    "volume-to-height polynomial."
                ),
                table=DiagnosticTable(
                    columns=["Plant", "Code", "Reason"],
                    rows=[
                        [name, code, reason] for name, code, reason in geometry_skips
                    ],
                    justify=["left", "right", "left"],
                ),
            ),
            logger=_LOG,
        )

    schema = pa.schema(
        [
            pa.field("hydro_id", pa.int32()),
            pa.field("volume_hm3", pa.float64()),
            pa.field("height_m", pa.float64()),
            pa.field("area_km2", pa.float64()),
        ]
    )

    return pa.table(
        {
            "hydro_id": pa.array(hydro_ids, type=pa.int32()),
            "volume_hm3": pa.array(volumes, type=pa.float64()),
            "height_m": pa.array(heights, type=pa.float64()),
            "area_km2": pa.array(areas, type=pa.float64()),
        },
        schema=schema,
    )
