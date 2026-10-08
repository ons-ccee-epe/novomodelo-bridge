"""Minimum-outflow bounds for DECOMP-like decks (``RQ`` defaults).

Semantics pinned against the reference manual (§4.5.11):

- the default minimum defluence is zero;
- ``RQ`` supplies, per REE, per-stage percentages of the registry's
  historical minimum flow (``vazao_minima_historica`` — *not* the
  long-term mean), after ``AC VAZMIN`` overrides patch that registry field.
  One percentage per study stage (``vazao_1`` → stage 0, ``vazao_2`` →
  stage 1, …); the last declared percentage carries forward for any stage
  beyond the register's own columns (seasonal carry-forward, as
  ``convert_irrigation_withdrawal`` does for ``TI``);
- a ``UH``-declared minimum has priority over ``RQ`` and is fixed for all
  stages;
- a plant with an explicit defluence window in the flow-constraint family
  (``HQ``/``LQ``/``CQ`` on QDEF) **co-applies** that window with its
  ``RQ``/``UH`` default rather than either one replacing the other (user
  ruling, 2026-08-09): both constraints hold on the same ``(hydro, stage,
  block)`` cell, so this module still emits the ``RQ``/``UH`` contribution
  unconditionally, and the accumulator (``bounds_accumulator.resolve`` /
  :func:`~novomodelo_bridge.decomp.bounds_accumulator.intersect`) composes it
  with the RHQ ``QDEF``-derived ``outflow`` contribution
  (``single_term_bounds.single_term_bound_contributions``) via
  max-of-lowers/min-of-uppers — the tighter side of each source wins, never
  one source displacing the other. A prior version of this module instead
  *skipped* the ``RQ``/``UH`` contribution for any plant with a ``CQ``
  ``QDEF`` window — encoding the wrong "the window replaces the default"
  reading of the reference manual instead of the correct "both apply,
  tighter wins" one; that skip was correctly retired once the accumulator's
  ``intersect`` could express the co-apply composition directly instead of
  this module approximating it via a skip;
- the ``RQ``/``UH`` minimum-outflow default is released for a run-of-river
  plant from the cascade headwater down to — but excluding — the first
  reservoir, to avoid infeasibility on a stage/scenario with zero inflow.
  Such a plant
  (:func:`~novomodelo_bridge.decomp.converters.cadastro.unregulated_runofriver_codes`)
  contributes no ``outflow`` floor here (neither RQ nor UH). An explicit RHQ
  ``QDEF`` window is a user-declared constraint and is **not** released — it
  still lowers to its ``outflow`` bound in ``single_term_bounds``.

Both emitters here return :class:`~novomodelo_bridge.decomp.bounds_accumulator.
BoundContribution` lists — the accumulator, not this module, resolves
per-cell collisions and fans them into the novomodelo bound parquet rows.

The ``RQ``/``UH`` minimum-outflow floor is a stage-level value
(``block_id = None``) that holds across every block. ``block_id`` on the
``outflow`` axis is materialised by a per-patamar ``LU``/``LQ`` window
(``single_term_bounds``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa

from novomodelo_bridge.core.tolerances import floats_differ
from novomodelo_bridge.decomp.bounds_accumulator import BoundContribution
from novomodelo_bridge.decomp.converters.cadastro import (
    effective_storage_range,
    storage_envelope,
)

if TYPE_CHECKING:
    from novomodelo_bridge.decomp.case import DecompCase
    from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro
    from novomodelo_bridge.decomp.id_map import DecompIdMap


def convert_hydro_bounds(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    effective: EffectiveCadastro,
    unregulated_codes: set[int] | None = None,
) -> list[BoundContribution]:
    """Minimum-outflow contributions from the ``RQ``/``UH`` defaults.

    A ``UH``-declared plant contributes a constant stage-level
    (``block_id = None``) value every stage. An ``RQ``-derived plant
    contributes one stage-level (``block_id = None``) value per stage — the
    REE's per-stage percentage (``vazao_1`` → stage 0, …, last declared
    percentage carried forward past the register's columns) times the
    plant's effective ``vazao_minima_historica`` for that stage. A plant
    with an explicit ``QDEF`` flow window still contributes its RQ/UH value
    here; the accumulator co-applies it with that window's own contribution
    (``single_term_bounds``) via max-of-lowers/min-of-uppers rather than
    either one replacing the other. Any stage whose effective floor is
    non-positive (or ``NaN``) emits no contribution.

    *unregulated_codes* (see
    :func:`~novomodelo_bridge.decomp.converters.cadastro.unregulated_runofriver_codes`)
    is the set of run-of-river plant codes whose minimum-outflow floor is
    released: a plant in it contributes no ``outflow`` floor here at all, RQ
    or UH alike.
    """
    unregulated_codes = unregulated_codes or set()
    calendar = case.calendar
    dadger = case.dadger
    rq = dadger.rq(df=True)
    if rq is None or rq.empty:
        return []

    uh = dadger.uh(df=True)
    operated = uh[uh["volume_inicial"].notna()]
    ree_by_code: dict[int, int] = {}
    uh_declared: dict[int, float] = {}
    for _, row in operated.iterrows():
        code = int(row["codigo_usina"])
        ree_by_code[code] = int(row["codigo_ree"])
        declared = row.get("vazao_defluente_minima")
        if declared is not None and not pd.isna(declared):
            uh_declared[code] = float(declared)

    # ``pct_by_ree[ree]`` is the REE's per-stage percentage list of the
    # historical minimum flow (``vazao_1`` → stage 0, ``vazao_2`` → stage 1,
    # …).
    pct_by_ree: dict[int, list[float]] = {}
    for _, row in rq.iterrows():
        values = []
        k = 1
        while f"vazao_{k}" in rq.columns:
            value = row[f"vazao_{k}"]
            values.append(0.0 if pd.isna(value) else float(value))
            k += 1
        pct_by_ree[int(row["codigo_ree"])] = values

    contributions: list[BoundContribution] = []
    for code in id_map.hydro_codes:
        # A run-of-river plant above the first cascade reservoir has its
        # minimum-outflow restriction released — no RQ/UH floor.
        if code in unregulated_codes:
            continue
        # ``per_stage[stage.index]`` is the effective minimum-outflow floor
        # (m3/s) for this plant at that stage — a UH-declared constant, or
        # the RQ per-stage percentage times the effective historical minimum.
        if code in uh_declared:
            per_stage = [uh_declared[code]] * len(calendar)
            contributor = "UH"
        else:
            ree = ree_by_code.get(code)
            if ree is None or ree not in pct_by_ree or code not in effective.base.index:
                continue
            pct_by_stage = pct_by_ree[ree]
            per_stage = []
            for stage in calendar:
                # Carry the last declared percentage forward for any stage
                # past the register's own columns (a no-op when they align).
                pct = pct_by_stage[min(stage.index, len(pct_by_stage) - 1)]
                base = effective.value(code, "vazao_minima_historica", stage.index)
                per_stage.append(pct / 100.0 * base)
            contributor = "RQ"

        hydro_id = id_map.hydro_id(code)
        for stage in calendar:
            value = per_stage[stage.index]
            if pd.isna(value) or value <= 0.0:
                continue
            contributions.append(
                BoundContribution(
                    family="hydro",
                    entity_id=hydro_id,
                    stage_id=stage.index,
                    block_id=None,
                    axis="outflow",
                    lower=value,
                    upper=None,
                    contributor=contributor,
                )
            )

    return contributions


def convert_storage_bounds(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    effective: EffectiveCadastro,
) -> list[BoundContribution]:
    """Sparse per-stage storage contributions wherever a stage tightens the envelope.

    For each hydro *code*, the outer envelope is ``storage_envelope(effective,
    code)`` — the widest floor/ceiling the plant's per-stage volumes ever
    reach, and the default the entity ``reservoir`` block
    declares. A stage whose effective range (:func:`~novomodelo_bridge.decomp.
    cadastro.effective_storage_range`) differs from that envelope (past float
    noise) contributes a stage-level (``block_id = None``) override; a stage
    equal to the envelope contributes nothing and simply inherits it. A plant
    with no temporal ``VOLMIN``/``VOLMAX`` override never differs from its own
    envelope, so it contributes nothing at all — and neither does a
    run-of-river (``D``) plant, whose per-stage range is already the same
    single-point collapse as its envelope. Storage is a
    stage-level axis (``block_eligible=False``), so no ``block_id`` is ever
    emitted here.
    """
    calendar = case.calendar
    contributions: list[BoundContribution] = []
    for code in id_map.hydro_codes:
        env_min, env_max = storage_envelope(effective, code)
        hydro_id = id_map.hydro_id(code)
        for stage_index in range(len(calendar)):
            vmin, vmax = effective_storage_range(effective, code, stage_index)
            if not (floats_differ(vmin, env_min) or floats_differ(vmax, env_max)):
                continue
            contributions.append(
                BoundContribution(
                    family="hydro",
                    entity_id=hydro_id,
                    stage_id=stage_index,
                    block_id=None,
                    axis="storage",
                    lower=vmin,
                    upper=vmax,
                    contributor="storage-envelope",
                )
            )

    return contributions


def convert_volume_espera_bounds(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    effective: EffectiveCadastro,
) -> list[BoundContribution]:
    """Per-stage max-storage contributions from the ``VE`` (volume de espera).

    The ``VE`` register (manual §3.4.6.15) declares a flood-control storage
    ceiling for a hydro *with reservoir*, as a percentage of the plant's
    **useful** volume for each of the study's ``N`` stages (``volume_k`` →
    stage ``k − 1``). It is a **hard** maximum-storage limit — the register
    carries no penalty field — so the reservoir may not fill above it, which
    forces releases during the flood season (verified: DECOMP pins ITAPARICA
    at its 55.1 % VE ceiling every flood-season stage). novomodelo has no other
    input for it, so it is emitted here as a per-stage ``max_storage_hm3``
    upper bound in absolute hm³ (``env_min + VE% · (env_max − env_min)``, the
    same ``volume útil`` base the source reports storage against), one-sided
    (no lower — the floor stays the plant's own), and composed by the
    accumulator's min-of-uppers so it only ever tightens the declared
    reservoir ceiling.

    Sparse by construction: a stage whose VE ceiling is not strictly below the
    plant's storage envelope max (a ``VE = 100 %`` no-op, or a blank field)
    contributes nothing, and neither does a plant absent from the register or
    without a useful-volume reservoir.
    """
    calendar = case.calendar
    ve = case.dadger.ve(df=True)
    if ve is None or ve.empty:
        return []

    stage_columns = [
        column
        for _, column in sorted(
            (int(column.split("_")[1]), column)
            for column in ve.columns
            if column.startswith("volume_") and column.split("_")[1].isdigit()
        )
    ]
    operated = set(id_map.hydro_codes)
    n_stages = len(calendar)

    contributions: list[BoundContribution] = []
    for _, row in ve.iterrows():
        code = int(row["codigo_usina"])
        if code not in operated:
            continue
        env_min, env_max = storage_envelope(effective, code)
        useful = env_max - env_min
        if useful <= 0.0:
            continue
        hydro_id = id_map.hydro_id(code)
        for stage_index in range(min(n_stages, len(stage_columns))):
            pct = row[stage_columns[stage_index]]
            if pd.isna(pct):
                continue
            ceiling = env_min + float(pct) / 100.0 * useful
            # Only tightens: a VE ≥ the envelope max is a no-op, and the
            # bound must never *raise* the declared ceiling.
            if ceiling >= env_max or not floats_differ(ceiling, env_max):
                continue
            contributions.append(
                BoundContribution(
                    family="hydro",
                    entity_id=hydro_id,
                    stage_id=stage_index,
                    block_id=None,
                    axis="storage",
                    lower=None,
                    upper=ceiling,
                    contributor="VE",
                )
            )

    return contributions


#: novomodelo ``hydro_bounds`` column for a consumptive water withdrawal, in m³/s
#: (positive = water removed from the plant's balance). The DECOMP ``TI``
#: irrigation rate and the source model's ``dsvagua`` file both land here.
_WATER_WITHDRAWAL_SCHEMA = pa.schema(
    [
        pa.field("hydro_id", pa.int32()),
        pa.field("stage_id", pa.int32()),
        pa.field("water_withdrawal_m3s", pa.float64()),
    ]
)


def convert_irrigation_withdrawal(
    case: DecompCase,
    id_map: DecompIdMap,
) -> pa.Table | None:
    """Per-(hydro, stage) consumptive irrigation withdrawal from the ``TI`` register.

    The ``TI`` register (*taxas de irrigação por UHE*) declares the water a hydro
    loses to irrigation, one rate (m³/s) per study stage (``taxa_k`` → stage
    ``k − 1``). It is a **consumptive** withdrawal — the water leaves the river
    and is unavailable for generation downstream — so it maps 1:1 to novomodelo's
    ``hydro_bounds`` ``water_withdrawal_m3s`` column, the DECOMP counterpart of
    the source model's ``dsvagua`` water-withdrawal file
    (:func:`novomodelo_bridge.newave.converters.hydro.convert_water_withdrawal`). Omitting it
    leaves that flow in the balance, so novomodelo turbines it and over-generates.

    The ``TI`` rate is already a positive withdrawal, matching novomodelo's positive
    ``water_withdrawal_m3s`` convention (no sign flip — unlike the source model's
    negative-``valor`` ``dsvagua`` convention). A stage beyond the register's own
    ``taxa`` columns repeats the last declared rate (seasonal carry-forward,
    matching the post-study extension the load/inflow converters use); a
    zero-withdrawal ``(hydro, stage)`` contributes no row.

    Returns the table sorted by ``(hydro_id, stage_id)`` with schema
    ``(hydro_id: int32, stage_id: int32, water_withdrawal_m3s: float64)``, or
    ``None`` when the deck carries no ``TI`` register or no operated plant
    withdraws (so the pipeline leaves ``hydro_bounds`` unchanged).
    """
    calendar = case.calendar
    ti = case.dadger.ti(df=True)
    if ti is None or ti.empty:
        return None

    taxa_columns = [
        column
        for _, column in sorted(
            (int(column.split("_")[1]), column)
            for column in ti.columns
            if column.startswith("taxa_") and column.split("_")[1].isdigit()
        )
    ]
    if not taxa_columns:
        return None

    operated = set(id_map.hydro_codes)
    n_stages = len(calendar)

    rows_hydro: list[int] = []
    rows_stage: list[int] = []
    rows_value: list[float] = []
    for _, row in ti.iterrows():
        code = int(row["codigo_usina"])
        if code not in operated:
            continue
        hydro_id = id_map.hydro_id(code)
        for stage_index in range(n_stages):
            # Carry the last declared rate forward for any stage past the
            # register's own columns (a no-op when they already align).
            column = taxa_columns[min(stage_index, len(taxa_columns) - 1)]
            value = row[column]
            if pd.isna(value) or float(value) == 0.0:
                continue
            rows_hydro.append(hydro_id)
            rows_stage.append(stage_index)
            rows_value.append(float(value))

    if not rows_hydro:
        return None

    order = sorted(range(len(rows_hydro)), key=lambda i: (rows_hydro[i], rows_stage[i]))
    return pa.table(
        {
            "hydro_id": pa.array([rows_hydro[i] for i in order], type=pa.int32()),
            "stage_id": pa.array([rows_stage[i] for i in order], type=pa.int32()),
            "water_withdrawal_m3s": pa.array(
                [rows_value[i] for i in order], type=pa.float64()
            ),
        },
        schema=_WATER_WITHDRAWAL_SCHEMA,
    )
