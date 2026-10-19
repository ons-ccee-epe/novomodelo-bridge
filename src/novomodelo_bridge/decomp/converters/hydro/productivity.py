"""Hydro productivity: per-plant/per-stage equivalent-productivity (rho_eq)
machinery, the productivity-override parquet, and the production-model JSON.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from novomodelo_bridge.core.productivity import equivalent_productivity_from_coeffs
from novomodelo_bridge.decomp.converters.cadastro.effective import is_reservoir
from novomodelo_bridge.novomodelo import schemas as novomodelo_schemas

if TYPE_CHECKING:
    from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro
    from novomodelo_bridge.decomp.id_map import DecompIdMap


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
      :func:`convert_hydro_group_availability`'s hydraulic cap consume,
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


def _stored_energy_productivity_mw_per_m3s(
    effective: EffectiveCadastro, code: int, stage_index: int
) -> float:
    """The source model's own stored-energy ρ for plant *code* at one stage.

    A reservoir (:func:`is_reservoir`, ``"M"`` or ``"S"``) averages the head
    over the stage's ``[volume_minimo, volume_maximo]``; a run-of-river ``"D"``
    plant takes the point head at ``volume_referencia``. Every other input is
    the stage's effective value, as in :func:`_equivalent_productivity_mw_per_m3s`.
    """
    reference = (
        None
        if is_reservoir(effective, code)
        else effective.value(code, "volume_referencia", stage_index)
    )
    return _equivalent_productivity_mw_per_m3s(
        effective, code, stage_index, reference_volume_hm3=reference
    )


def convert_energy_productivity(
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
    initial_volumes: dict[int, float] | None = None,
) -> pa.Table:
    """Per-plant equivalent generation productivity, anchored at initial volume.

    ``ρ_eq = ρ_esp · h_net`` with the head taken at the plant's **initial
    reservoir volume** — the source model's FPHA fit anchor (see
    :func:`_equivalent_productivity_mw_per_m3s`). *initial_volumes*
    (``{code: hm³}``, from :func:`_operated_initial_volumes`) supplies that
    anchor; a plant absent from it — or the whole map being ``None`` — falls
    back to the full-range mean anchor (``reference_volume_hm3=None``), the
    pre-FPHA-fix behaviour.

    Every operated plant is listed, FPHA included, so :func:`convert_penalties`'
    system ρ_avg/ρ_max covers the whole fleet; the case parquet is
    :func:`convert_hydro_energy_productivity`. The per-stage head variation
    instead feeds :func:`convert_hydro_group_availability`'s own hydraulic
    ceiling.
    """
    vols = initial_volumes or {}
    hydro_ids = [id_map.hydro_id(code) for code in id_map.hydro_codes]
    values = [
        _equivalent_productivity_mw_per_m3s(
            effective, code, 0, reference_volume_hm3=vols.get(code)
        )
        for code in id_map.hydro_codes
    ]
    return pa.table(
        {
            "hydro_id": pa.array(hydro_ids, type=pa.int32()),
            "stage_id": pa.array([None] * len(hydro_ids), type=pa.int32()),
            "equivalent_productivity_mw_per_m3s": pa.array(values, type=pa.float64()),
        }
    )


def convert_hydro_energy_productivity(
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
    initial_volumes: dict[int, float] | None = None,
    fpha_codes: set[int] | None = None,
) -> pa.Table:
    """The ``hydro_energy_productivity.parquet`` table, one or more rows per plant.

    Each plant has a ``stage_id = NULL`` default row carrying:

    - ``equivalent_productivity_mw_per_m3s`` — the generation ρ_eq of
      :func:`convert_energy_productivity`; NULL for a plant in *fpha_codes*,
      which carries ``model: "fpha"`` and whose ρ_eq novomodelo derives from the
      geometry it fits the production function on.
    - ``integrated_equivalent_productivity_mw_per_m3s`` — the plant's own
      stored-energy ρ (:func:`_stored_energy_productivity_mw_per_m3s`), which
      novomodelo sums down the cascade for the integrated tags and the maximum
      stored energy that RHE reads. novomodelo's own integral over the VHA geometry
      ignores the per-stage ``AC`` overrides and integrates ``"D"`` plants, so
      the value is always supplied.

    When an ``AC`` override makes the stored-energy ρ vary by stage, the
    default row's integrated value is NULL and one row per stage carries it
    instead; novomodelo resolves each column per stage before the default.
    """
    vols = initial_volumes or {}
    fpha = fpha_codes or set()
    hydro_ids: list[int] = []
    stage_ids: list[int | None] = []
    rho_eq: list[float | None] = []
    integrated: list[float | None] = []
    for code in id_map.hydro_codes:
        hydro_id = id_map.hydro_id(code)
        series = [
            _stored_energy_productivity_mw_per_m3s(effective, code, stage)
            for stage in range(effective.n_stages)
        ]
        varies = len(set(series)) > 1
        hydro_ids.append(hydro_id)
        stage_ids.append(None)
        rho_eq.append(
            None
            if code in fpha
            else _equivalent_productivity_mw_per_m3s(
                effective, code, 0, reference_volume_hm3=vols.get(code)
            )
        )
        integrated.append(None if varies or not series else series[0])
        if varies:
            for stage, value in enumerate(series):
                hydro_ids.append(hydro_id)
                stage_ids.append(stage)
                rho_eq.append(None)
                integrated.append(value)

    nulls = [None] * len(hydro_ids)
    return pa.table(
        {
            "hydro_id": pa.array(hydro_ids, type=pa.int32()),
            "stage_id": pa.array(stage_ids, type=pa.int32()),
            "equivalent_productivity_mw_per_m3s": pa.array(rho_eq, type=pa.float64()),
            "reference_outflow_m3s": pa.array(nulls, type=pa.float64()),
            "specific_productivity_mw_per_m3s_per_m": pa.array(
                nulls, type=pa.float64()
            ),
            "integrated_equivalent_productivity_mw_per_m3s": pa.array(
                integrated, type=pa.float64()
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
    :func:`novomodelo_bridge.decomp.converters.fpha.fitting_window`) is emitted as ``model:
    "fpha"`` — novomodelo fits the production function from the plant geometry
    (``hydro_geometry.parquet``) + tailrace families (``tailrace_curves.parquet``)
    over the config's ``fitting_window`` — with its ``reference_volume`` (from
    *reference_volumes*, the initial-volume anchor) setting the FPHA reference /
    backwater level. Every other operated plant keeps ``constant_productivity``,
    its ρ_eq riding in ``hydro_energy_productivity.parquet``. *fpha_configs* and
    *reference_volumes* are pre-built by the pipeline so this module needs no
    import from :mod:`novomodelo_bridge.decomp.converters.fpha` (which imports it).
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
        "$schema": novomodelo_schemas.schema_url_for(
            "system/hydro_production_models.json"
        ),
        "production_models": models,
    }
