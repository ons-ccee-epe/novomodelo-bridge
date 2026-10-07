"""Hydro productivity: per-plant/per-stage equivalent-productivity (rho_eq)
machinery and the production-model JSON.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from cobre_bridge.cobre import schemas as cobre_schemas
from cobre_bridge.core.productivity import equivalent_productivity_from_coeffs

if TYPE_CHECKING:
    from cobre_bridge.decomp.converters.cadastro import EffectiveCadastro
    from cobre_bridge.decomp.id_map import DecompIdMap


def _equivalent_productivity_mw_per_m3s(
    effective: EffectiveCadastro,
    code: int,
    stage_index: int,
    reference_volume_hm3: float | None = None,
) -> float:
    """``ρ_eq = ρ_esp · h_net`` for one plant at one stage.

    The gross head is an upstream cota minus the mean tailrace level, with the
    hydraulic-loss model applied. The cota anchor depends on
    *reference_volume_hm3*:

    * ``None`` (the engolimento anchor) — the volume-averaged cota over the
      **full** operating range (``volume_minimo``..``volume_maximo``). This is
      the value the head-aware ``max_turbined`` envelope and
      :func:`convert_hydro_group_availability`'s B8 hydraulic cap consume,
      validated against the source model's ``Qtur Maxima``; it stays on the
      full-range mean so that path is unchanged.
    * a volume (the **generation** anchor) — the cota at that single volume.
      :func:`convert_energy_productivity` passes the plant's initial reservoir
      volume because the source model anchors each plant's hydro production
      function (FPHA) at the initial volume ± a fit window (manual §3.4.6.4,
      ``FP`` fields 10-11), **not** the full-range mean. For a reservoir
      starting far from mid-range the two heads differ materially — validated
      against ``dec_oper_usih`` effective productivity, the initial-volume
      anchor cuts the reservoir median error 4.3% -> 2.6% and the worst case
      (BARRA BONITA, 87% full) 22.7% -> 5.9%.

    Every other input is read at *stage_index* through *effective*, so an
    ``AC PROESP``/``PERHID``/``JUSMED``/``COTVOL``/``VOLMIN``/``VOLMAX``
    override shifts this stage's ρ_eq; ``tipo_perda`` carries no ``AC`` register
    of its own and stays a base read.
    """
    v_min = effective.value(code, "volume_minimo", stage_index)
    v_max = effective.value(code, "volume_maximo", stage_index)
    rho_esp = effective.value(code, "produtibilidade_especifica", stage_index)
    cf = effective.value(code, "canal_fuga_medio", stage_index)
    perdas = effective.value(code, "perdas", stage_index)
    tipo_perda = int(effective.base.loc[code, "tipo_perda"])
    coeffs = effective.cota_polynomial(code, stage_index)
    return equivalent_productivity_from_coeffs(
        coeffs,
        volume_min_hm3=v_min,
        volume_max_hm3=v_max,
        rho_esp=rho_esp,
        canal_fuga_m=cf,
        tipo_perda=tipo_perda,
        perdas=perdas,
        reference_volume_hm3=reference_volume_hm3,
        plant_name=str(effective.base.loc[code, "nome_usina"]).strip(),
    )


def convert_energy_productivity(
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
    initial_volumes: dict[int, float] | None = None,
    exclude_codes: set[int] | None = None,
) -> pa.Table:
    """Per-plant equivalent generation productivity, anchored at initial volume.

    ``ρ_eq = ρ_esp · h_net`` with the head taken at the plant's **initial
    reservoir volume** — the source model's FPHA fit anchor (see
    :func:`_equivalent_productivity_mw_per_m3s`). *initial_volumes*
    (``{code: hm³}``, from :func:`_operated_initial_volumes`) supplies that
    anchor; a plant absent from it — or the whole map being ``None`` — falls
    back to the full-range mean anchor (``reference_volume_hm3=None``), the
    pre-FPHA-fix behaviour.

    *exclude_codes* omits those plants from the emitted table — used for the
    ``hydro_energy_productivity.parquet`` write to drop the FPHA plants, which
    carry ``model: "fpha"`` and would be a double-supply (cobre rejects a plant
    that has both a computed FPHA and a parquet ρ_eq). Left ``None`` (every
    plant emitted) the full list still feeds :func:`convert_penalties`' system
    ρ_avg/ρ_max, so the penalty scale is unchanged by the FPHA split. The
    per-stage head variation instead feeds
    :func:`convert_hydro_group_availability`'s own hydraulic ceiling.
    """
    vols = initial_volumes or {}
    excluded = exclude_codes or set()
    hydro_ids: list[int] = []
    values: list[float] = []
    for code in id_map.hydro_codes:
        if code in excluded:
            continue
        hydro_ids.append(id_map.hydro_id(code))
        values.append(
            _equivalent_productivity_mw_per_m3s(
                effective, code, 0, reference_volume_hm3=vols.get(code)
            )
        )

    return pa.table(
        {
            "hydro_id": pa.array(hydro_ids, type=pa.int32()),
            "stage_id": pa.array([None] * len(hydro_ids), type=pa.int32()),
            "equivalent_productivity_mw_per_m3s": pa.array(values, type=pa.float64()),
            # Present-but-null: the reader requires the full column set.
            "reference_outflow_m3s": pa.array(
                [None] * len(hydro_ids), type=pa.float64()
            ),
            "specific_productivity_mw_per_m3s_per_m": pa.array(
                [None] * len(hydro_ids), type=pa.float64()
            ),
        }
    )


def convert_production_models(
    id_map: DecompIdMap,
    fpha_configs: dict[int, dict] | None = None,
    reference_volumes: dict[int, dict] | None = None,
) -> dict:
    """Per-plant production-model selection.

    A plant in *fpha_configs* (``{code: fpha_config}``, from the pipeline via
    :func:`cobre_bridge.decomp.converters.fpha.fitting_window`) is emitted as ``model:
    "fpha"`` — cobre fits the production function from the plant geometry
    (``hydro_geometry.parquet``) + tailrace families (``tailrace_curves.parquet``)
    over the config's ``fitting_window`` — with its ``reference_volume`` (from
    *reference_volumes*, the initial-volume anchor) setting the FPHA reference /
    backwater level. Every other operated plant keeps ``constant_productivity``,
    its ρ_eq riding in ``hydro_energy_productivity.parquet``. *fpha_configs* and
    *reference_volumes* are pre-built by the pipeline so this module needs no
    import from :mod:`cobre_bridge.decomp.converters.fpha` (which imports it).
    """
    fpha_configs = fpha_configs or {}
    reference_volumes = reference_volumes or {}
    models: list[dict] = []
    for code in id_map.hydro_codes:
        stage_range: dict = {"start_stage_id": 0, "end_stage_id": None}
        if code in fpha_configs:
            stage_range["model"] = "fpha"
            stage_range["fpha_config"] = fpha_configs[code]
            reference_volume = reference_volumes.get(code)
            if reference_volume is not None:
                stage_range["reference_volume"] = reference_volume
        else:
            stage_range["model"] = "constant_productivity"
        models.append(
            {
                "hydro_id": id_map.hydro_id(code),
                "selection_mode": "stage_ranges",
                "stage_ranges": [stage_range],
            }
        )
    return {
        "$schema": cobre_schemas.schema_url_for("system/hydro_production_models.json"),
        "production_models": models,
    }
