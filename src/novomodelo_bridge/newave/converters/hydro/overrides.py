"""Source-file override readers for hydro entity conversion.

Reads MODIF.DAT (permanent + temporal overrides) and GHMIN.DAT. The
package's lowest layer: imports nothing from a sibling submodule.

MODIF.DAT volume records (VOLMAX, VOLMIN, VMAXT, VMINT) carry a unit column:
``h`` is hm³ and ``%`` is a percentage of the plant's useful volume. The
percentage always resolves against the ``hidr.dat`` registry volumes, never
against a volume another record already changed, so the result does not
depend on record order.
"""

from __future__ import annotations

import logging

import pandas as pd

from cobre_bridge.core.diagnostics import Diagnostic, DiagnosticTable, Severity, emit
from cobre_bridge.newave.case import NewaveCase
from cobre_bridge.newave.horizon import POST_STUDY_YEAR

_LOG = logging.getLogger(__name__)


# Temporal (dated) MODIF.DAT record types and the attribute holding each value.
_TEMPORAL_VALUE_ATTR = {
    "VAZMINT": "vazao",
    "VMAXT": "volume",
    "VMINT": "volume",
    "CFUGA": "nivel",
    "CMONT": "nivel",
    "TURBMINT": "turbinamento",
    "TURBMAXT": "turbinamento",
}
_TEMPORAL_OVERRIDE_TYPES = frozenset(_TEMPORAL_VALUE_ATTR)


def _raw_unit(rec: object) -> str:
    raw = getattr(rec, "unidade", None)
    return raw if isinstance(raw, str) else ""


def _volume_unit(rec: object) -> str | None:
    """``"h"`` (hm³) or ``"%"`` (percent of the useful volume) of a MODIF.DAT
    volume record; ``None`` when its ``unidade`` column is absent or unknown.
    The file quotes the unit (``'%'``), so quotes are stripped first."""
    unit = _raw_unit(rec).strip().strip("'\"").strip().lower()
    if unit in ("h", "hm3", "hm³"):
        return "h"
    if unit == "%":
        return "%"
    return None


def _polynomial_coefficients(raw: list[float | None]) -> list[float]:
    """The coefficients of a MODIF.DAT polynomial record, a0 first.

    A blank field is a zero coefficient, not a missing one: a low-order
    polynomial leaves its tail blank instead of spelling out ``0.``.
    """
    return [0.0 if pd.isna(value) else float(value) for value in raw]


def percent_of_useful_volume(percent: float, vol_min: float, vol_max: float) -> float:
    """hm³ for a MODIF.DAT volume given as a percentage of the useful volume."""
    return vol_min + (percent / 100.0) * (vol_max - vol_min)


def _emit_unknown_volume_units(
    code: str, scope: str, records: list[tuple[int, str, str]], assumed: str
) -> None:
    if not records:
        return
    emit(
        Diagnostic(
            code=code,
            severity=Severity.WARNING,
            category="Cadastro overrides",
            title=f"Volume override(s) with unknown unit ({len(records)})",
            summary=(
                f"MODIF.DAT {scope} volume records carry a unit column ('h' = hm³, "
                f"'%' = percent of the useful volume); {len(records)} record(s) "
                f"have none the converter recognises, so their values are taken "
                f"as {assumed}."
            ),
            table=DiagnosticTable(
                columns=["Code", "Type", "Unit"],
                rows=[[code_, type_name, unit] for code_, type_name, unit in records],
                justify=["right", "left", "left"],
            ),
            remediation="Check the unit column of the listed records in MODIF.DAT.",
        ),
        logger=_LOG,
    )


def _apply_permanent_overrides(
    cadastro: pd.DataFrame, case: NewaveCase
) -> pd.DataFrame:
    """Apply MODIF.DAT permanent overrides to the hidr.dat cadastro.

    Reads ``MODIF.DAT`` from *case* and applies its permanent (undated)
    override records to a *copy* of *cadastro*.  The original DataFrame is
    not mutated.

    Parameters
    ----------
    cadastro:
        The ``Hidr.cadastro`` DataFrame indexed by ``codigo_usina``.
    case:
        Parsed the source model case.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with permanent overrides applied.
    """
    modif = case.modif
    if modif is None:
        _LOG.debug("MODIF.DAT not found; skipping permanent overrides.")
        return cadastro

    result = cadastro.copy()

    # Ensure float dtype for columns that permanent overrides may assign floats
    # into.  Without this, pandas 2.x raises TypeError when the column was
    # inferred as int64 (e.g. vazao_minima_historica=[0, 0]).
    _float_override_cols = (
        "vazao_minima_historica",
        "volume_maximo",
        "volume_minimo",
    )
    for _col in _float_override_cols:
        if _col in result.columns and result[_col].dtype.kind == "i":
            result[_col] = result[_col].astype(float)

    usina_records = modif.usina()
    if not usina_records:
        return result

    # Loop-accumulate-then-emit-once: one record per skipped plant/record,
    # emitted after the loop (see the module's finalize_diagnostics de-dup).
    uncadastred: list[int] = []
    unsupported_perm: list[tuple[int, str]] = []
    unit_unknown: list[tuple[int, str, str]] = []

    for usina_rec in usina_records:
        code = int(usina_rec.codigo)
        if code not in result.index:
            uncadastred.append(code)
            continue

        for rec in modif.modificacoes_usina(code):
            type_name = type(rec).__name__

            # Skip temporal override types — handled separately.
            if type_name in _TEMPORAL_OVERRIDE_TYPES:
                continue

            if type_name == "VAZMIN":
                result.loc[code, "vazao_minima_historica"] = float(rec.vazao)

            elif type_name in ("VOLMAX", "VOLMIN"):
                unit = _volume_unit(rec)
                if unit is None:
                    unit_unknown.append((code, type_name, _raw_unit(rec)))
                    unit = "h"
                value = float(rec.volume)
                if unit == "%":
                    base = cadastro.loc[code]
                    value = percent_of_useful_volume(
                        value,
                        float(base["volume_minimo"]),
                        float(base["volume_maximo"]),
                    )
                column = "volume_maximo" if type_name == "VOLMAX" else "volume_minimo"
                result.loc[code, column] = value

            elif type_name == "NUMCNJ":
                result.loc[code, "numero_conjuntos_maquinas"] = int(rec.numero)

            elif type_name == "NUMMAQ":
                set_num = int(rec.conjunto)
                n_maq = int(rec.numero_maquinas)
                result.loc[code, f"maquinas_conjunto_{set_num}"] = n_maq

            elif type_name == "POTEFE":
                # ``potencia_nominal_conjunto_*`` is the conjunto's POTEF.
                set_num = int(rec.conjunto)
                result.loc[code, f"potencia_nominal_conjunto_{set_num}"] = float(
                    rec.potencia
                )

            elif type_name == "VOLCOTA":
                result.loc[code, [f"a{i}_volume_cota" for i in range(5)]] = (
                    _polynomial_coefficients(rec.polinomio_volume_cota)
                )

            elif type_name == "COTAREA":
                result.loc[code, [f"a{i}_cota_area" for i in range(5)]] = (
                    _polynomial_coefficients(rec.polinomio_cota_area)
                )

            elif type_name == "DefaultRegister":
                # inewave emits DefaultRegister for records it does not model.
                # These are benign for the conversion, so log at debug level
                # only — no user-facing warning.
                _LOG.debug(
                    "MODIF.DAT contains an unmodeled record (DefaultRegister)"
                    " for plant %d; skipping.",
                    code,
                )

            else:
                unsupported_perm.append((code, type_name))

    if uncadastred:
        emit(
            Diagnostic(
                code="modif-override-plant-uncadastred",
                severity=Severity.WARNING,
                category="Cadastro overrides",
                title=(f"MODIF.DAT references {len(uncadastred)} uncadastred plant(s)"),
                summary=(
                    f"MODIF.DAT references {len(uncadastred)} plant code(s) "
                    "not present in hidr.dat; skipping their overrides."
                ),
                table=DiagnosticTable(
                    columns=["Code"],
                    rows=[[code] for code in uncadastred],
                    justify=["right"],
                ),
            ),
            logger=_LOG,
        )
    if unsupported_perm:
        emit(
            Diagnostic(
                code="modif-permanent-override-unsupported",
                severity=Severity.WARNING,
                category="Cadastro overrides",
                title=(
                    f"Unsupported permanent override type(s) ({len(unsupported_perm)})"
                ),
                summary=(
                    f"MODIF.DAT contains {len(unsupported_perm)} unsupported "
                    "or unknown permanent override record(s); skipping."
                ),
                table=DiagnosticTable(
                    columns=["Code", "Type"],
                    rows=[[code, type_name] for code, type_name in unsupported_perm],
                    justify=["right", "left"],
                ),
            ),
            logger=_LOG,
        )
    _emit_unknown_volume_units(
        "modif-permanent-volume-unit-unknown", "permanent", unit_unknown, "hm³"
    )

    return result


def read_cadastro(case: NewaveCase) -> pd.DataFrame:
    """Read ``hidr.dat`` and apply permanent MODIF.DAT overrides.

    Parameters
    ----------
    case:
        Parsed the source model case.

    Returns
    -------
    pd.DataFrame
        The ``Hidr.cadastro`` DataFrame indexed by ``codigo_usina`` with all
        permanent MODIF.DAT overrides already applied.
    """
    cadastro = case.hidr.cadastro
    return _apply_permanent_overrides(cadastro, case)


def _extract_temporal_overrides(
    case: NewaveCase, confhd_codes: list[int]
) -> dict[int, list[dict]]:
    """Extract MODIF.DAT temporal overrides for plants in *confhd_codes*.

    Reads ``MODIF.DAT`` and returns a dict keyed by plant code.  Each value
    is a list of override dicts in file order::

        {"type": str, "month": int, "year": int | None,
         "period": "PRE" | "POS" | None, "value": float}

    A VAZMINT record may mark its year field ``PRE`` (pre-study period) or
    ``POS`` (post-study period); it then carries ``year=None`` and that
    ``period``. A VAZMINT record with no month, or with neither a year nor a
    marker, is skipped and reported.

    For CFUGA/CMONT the ``"value"`` field is the level in metres.  For
    TURBMINT/TURBMAXT it is the turbined flow in m³/s and for VAZMINT the flow
    in m³/s.  For VMAXT/VMINT it is the volume as stored in the record, with a
    ``"unit"`` key saying how to read it: ``"h"`` (hm³) or ``"%"`` (percent of
    the useful volume); a record with no recognisable unit is taken as ``"%"``
    and reported.

    Parameters
    ----------
    case:
        Parsed the source model case.
    confhd_codes:
        List of plant codes present in the study (from confhd.dat).  Records
        for plants not in this list are excluded.

    Returns
    -------
    dict[int, list[dict]]
        Temporal override records per plant code.  Empty dict if MODIF.DAT is
        absent.
    """
    modif = case.modif
    if modif is None:
        _LOG.debug("MODIF.DAT not found; no temporal overrides extracted.")
        return {}

    confhd_set = set(confhd_codes)
    result: dict[int, list[dict]] = {}

    usina_records = modif.usina()
    if not usina_records:
        return result

    # Loop-accumulate-then-emit-once (see _apply_permanent_overrides above).
    undated: list[tuple[int, str]] = []
    unit_unknown: list[tuple[int, str, str]] = []

    for usina_rec in usina_records:
        code = int(usina_rec.codigo)
        if code not in confhd_set:
            continue

        plant_overrides: list[dict] = []
        for rec in modif.modificacoes_usina(code):
            type_name = type(rec).__name__
            if type_name not in _TEMPORAL_OVERRIDE_TYPES:
                continue

            # VMAXT/VMINT carry a unit column ('h' hm³ or '%' of useful volume);
            # a record with no recognisable unit is taken as '%' and reported.
            unit: str | None = None
            if type_name in ("VMAXT", "VMINT"):
                unit = _volume_unit(rec)
                if unit is None:
                    unit_unknown.append((code, type_name, _raw_unit(rec)))
                    unit = "%"

            # Only VAZMINT admits the PRE/POS markers, so only it reads them.
            if type_name == "VAZMINT":
                period = rec.periodo
                start = rec.data_inicio
                if rec.mes is None or (period is None and start is None):
                    undated.append((code, type_name))
                    continue
                month = int(rec.mes)
                year = None if period is not None else int(start.year)
            else:
                period = None
                month = int(rec.data_inicio.month)
                year = int(rec.data_inicio.year)

            override: dict = {
                "type": type_name,
                "month": month,
                "year": year,
                "period": period,
                "value": float(getattr(rec, _TEMPORAL_VALUE_ATTR[type_name])),
            }
            if unit is not None:
                override["unit"] = unit
            plant_overrides.append(override)

        if plant_overrides:
            result[code] = plant_overrides

    if undated:
        emit(
            Diagnostic(
                code="modif-temporal-override-undated",
                severity=Severity.WARNING,
                category="Cadastro overrides",
                title=f"Dated override(s) without a date ({len(undated)})",
                summary=(
                    f"MODIF.DAT contains {len(undated)} dated override record(s) "
                    "with no month, or with neither a year nor a PRE/POS marker; "
                    "skipping."
                ),
                table=DiagnosticTable(
                    columns=["Code", "Type"],
                    rows=[[code, type_name] for code, type_name in undated],
                    justify=["right", "left"],
                ),
            ),
            logger=_LOG,
        )
    _emit_unknown_volume_units(
        "modif-temporal-volume-unit-unknown",
        "dated",
        unit_unknown,
        "a percentage of the useful volume",
    )

    return result


def _read_ghmin_per_stage(
    case: NewaveCase,
    start_year: int,
    start_month: int,
    study_months: int,
    total_stages: int,
) -> dict[int, dict[int, float]]:
    """Read GHMIN.DAT and expand into ``{plant_code: {stage_0based: min_gen_mw}}``.

    GHMIN values are time-varying minimum-generation requirements in MWmes that
    source-model enforces per plant per stage.  Each (plant, month, year) record sets
    the value from that stage forward until the next record overrides it (step
    function).  Records with ``year == 9999`` are post-study seasonal entries: each
    calendar month they appear for becomes the value used in every post-study stage with
    that calendar month, falling back to a seasonal repeat of the last study year for
    unspecified months.

    Only ``patamar == 0`` rows are used — they represent the all-blocks
    mean, which matches the per-stage granularity of
    ``hydro_bounds.parquet``.

    Returns an empty mapping when ``GHMIN.DAT`` is absent.

    Parameters
    ----------
    case:
        Parsed the source model case.
    start_year, start_month:
        Study start (Cobre stage 0 corresponds to this calendar month).
    study_months:
        Number of in-study stages.
    total_stages:
        Total number of stages (study + post-study).
    """
    ghmin = case.ghmin
    if ghmin is None:
        _LOG.debug("GHMIN.DAT not found; emitting no per-stage min_generation.")
        return {}

    df = ghmin.geracoes
    if df is None or df.empty:
        return {}

    patamar0 = df[df["patamar"] == 0]
    if patamar0.empty:
        return {}

    result: dict[int, dict[int, float]] = {}
    for code, group in patamar0.groupby("codigo_usina"):
        code_int = int(code)
        study_changes: list[tuple[int, float]] = []
        pos_by_month: dict[int, float] = {}
        for _, row in group.iterrows():
            dt = row["data"]
            yr = int(dt.year)
            mo = int(dt.month)
            value = float(row["geracao"])
            if yr == POST_STUDY_YEAR:
                pos_by_month[mo] = value
                continue
            sid = (yr - start_year) * 12 + (mo - start_month)
            if sid < 0:
                sid = 0
            study_changes.append((sid, value))
        study_changes.sort()

        per_stage: dict[int, float] = {}
        # Step function across the study period.
        if study_changes:
            first_stage = study_changes[0][0]
            cp_idx = 0
            current: float | None = None
            for stage_id in range(first_stage, study_months):
                while (
                    cp_idx < len(study_changes) and study_changes[cp_idx][0] <= stage_id
                ):
                    current = study_changes[cp_idx][1]
                    cp_idx += 1
                if current is not None:
                    per_stage[stage_id] = current

        # Seasonal pattern for post-study: prefer explicit POS entries
        # for each calendar month; fall back to the last study year's
        # value for that calendar month.
        if total_stages > study_months:
            last_year_seasonal: dict[int, float] = {}
            for stage_id in range(max(0, study_months - 12), study_months):
                if stage_id in per_stage:
                    cal = ((start_month - 1 + stage_id) % 12) + 1
                    last_year_seasonal[cal] = per_stage[stage_id]

            for stage_id in range(study_months, total_stages):
                cal = ((start_month - 1 + stage_id) % 12) + 1
                if cal in pos_by_month:
                    per_stage[stage_id] = pos_by_month[cal]
                elif cal in last_year_seasonal:
                    per_stage[stage_id] = last_year_seasonal[cal]

        if per_stage:
            result[code_int] = per_stage

    return result


def _per_stage_drop_overrides(
    drop_overrides: list[dict],
    case: NewaveCase,
    total_stages: int,
) -> list[tuple[float | None, float | None]]:
    """Per-stage effective ``(CFUGA, CMONT)`` from MODIF.DAT step-functions.

    Evolves the CFUGA / CMONT temporal overrides into a per-stage
    ``(canal_fuga, cmont)`` state: each event applies from its stage of effect
    forward; events before the horizon fold in at stage 0; and when
    ``sazonaliza_cfuga_cmont == 1`` the calendar month's latest-year value
    repeats after the last explicit event. Shared by both
    :func:`_per_stage_productivities` (point ρ, combined with VOLREF_SAZ) and
    :func:`_per_stage_equivalent_productivities` (PRODT, which ignores VOLREF_SAZ).
    """
    if not drop_overrides:
        return [(None, None)] * total_stages

    dger = case.dger
    start_year = int(dger.ano_inicio_estudo)
    start_month = int(dger.mes_inicio_estudo)
    seasonalize = int(getattr(dger, "sazonaliza_cfuga_cmont", 0) or 0) == 1

    events_by_stage: dict[int, list[tuple[float | None, float | None]]] = {}
    last_event_stage = -1
    for override in drop_overrides:
        stage_id = (override["year"] - start_year) * 12 + (
            override["month"] - start_month
        )
        last_event_stage = max(last_event_stage, stage_id)
        if override["type"] == "CFUGA":
            events_by_stage.setdefault(stage_id, []).append(
                (float(override["value"]), None)
            )
        else:  # CMONT
            events_by_stage.setdefault(stage_id, []).append(
                (None, float(override["value"]))
            )

    # sazonaliza_cfuga_cmont == 1: after the last explicit entry the source model
    # repeats the seasonal pattern from the latest year defining each calendar month.
    seasonal_cfuga: dict[int, float] = {}
    seasonal_cmont: dict[int, float] = {}
    if seasonalize:
        latest_cfuga: dict[int, int] = {}
        latest_cmont: dict[int, int] = {}
        for override in drop_overrides:
            year = int(override["year"])
            month = int(override["month"])
            value = float(override["value"])
            if override["type"] == "CFUGA":
                if month not in latest_cfuga or year > latest_cfuga[month]:
                    latest_cfuga[month] = year
                    seasonal_cfuga[month] = value
            elif month not in latest_cmont or year > latest_cmont[month]:
                latest_cmont[month] = year
                seasonal_cmont[month] = value

    drops: list[tuple[float | None, float | None]] = []
    active_cfuga: float | None = None
    active_cmont: float | None = None
    for stage_id in range(total_stages):
        if stage_id == 0:
            applicable_stages = sorted(s for s in events_by_stage if s <= 0)
        elif stage_id in events_by_stage:
            applicable_stages = [stage_id]
        else:
            applicable_stages = []
        for past_stage in applicable_stages:
            for cfuga_val, cmont_val in events_by_stage[past_stage]:
                if cfuga_val is not None:
                    active_cfuga = cfuga_val
                if cmont_val is not None:
                    active_cmont = cmont_val

        calendar_month = ((start_month - 1 + stage_id) % 12) + 1
        if seasonalize and stage_id > last_event_stage:
            if calendar_month in seasonal_cfuga:
                active_cfuga = seasonal_cfuga[calendar_month]
            if calendar_month in seasonal_cmont:
                active_cmont = seasonal_cmont[calendar_month]

        drops.append((active_cfuga, active_cmont))
    return drops
