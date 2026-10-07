"""Temporal data converter: maps the source model study horizon configuration to Cobre
JSON.

Converts ``dger.dat`` and ``patamar.dat`` into the ``stages.json`` and
``config.json`` formats expected by the Cobre solver.
"""

from __future__ import annotations

import calendar
import logging
import math
from datetime import date

from inewave.newave import Dger

from cobre_bridge.cobre import schemas as cobre_schemas
from cobre_bridge.core.season_calendar import block_names, monthly_season_definitions
from cobre_bridge.newave.case import NewaveCase
from cobre_bridge.newave.horizon import study_horizon
from cobre_bridge.newave.id_map import NewaveIdMap

logger = logging.getLogger(__name__)


def _month_start_date(year: int, month: int) -> date:
    """Return the first day of the given calendar month."""
    return date(year, month, 1)


def _month_end_date(year: int, month: int) -> date:
    """Return the first day of the *following* month (exclusive end date)."""
    if month == 12:
        return date(year + 1, 1, 1)
    return date(year, month + 1, 1)


def _month_hours(year: int, month: int) -> float:
    """Total number of hours in the given calendar month."""
    days_in_month = calendar.monthrange(year, month)[1]
    return float(days_in_month * 24)


def convert_stages(case: NewaveCase, id_map: NewaveIdMap) -> dict:  # noqa: ARG001
    """Convert the source model temporal configuration to a Cobre ``stages.json`` dict.

    Reads ``dger.dat`` and ``patamar.dat`` from *case* and produces a
    dict that conforms to ``stages.schema.json``.

    Parameters
    ----------
    case:
        Parsed the source model case.
    id_map:
        Entity ID map (not used for temporal conversion, accepted for API
        consistency with the other converters).

    Returns
    -------
    dict
        JSON-serializable dict conforming to ``stages.schema.json``.

    Raises
    ------
    ValueError
        If ``num_anos_estudo`` is 0 or None.
    """
    dger = case.dger
    patamar = case.patamar
    deterministic = _is_deterministic_mode(case, dger)

    num_anos = dger.num_anos_estudo
    if not num_anos:
        raise ValueError("NEWAVE case has zero study years")

    start_month: int = dger.mes_inicio_estudo
    start_year: int = dger.ano_inicio_estudo
    num_scenarios: int = dger.num_aberturas or 1
    taxa = dger.taxa_de_desconto or 0.0
    annual_discount_rate = float(taxa) / 100.0

    # ------------------------------------------------------------------
    # Read CVaR configuration from dger.dat and cvar.dat.
    # dger.cvar == 0 (or None): expectation
    # dger.cvar == 1: constant CVaR from cvar.dat::valores_constantes
    # dger.cvar == 2: temporal CVaR from cvar.dat::alfa_variavel / lambda_variavel
    # ------------------------------------------------------------------
    _raw_cvar = getattr(dger, "cvar", None)
    dger_cvar: int = int(_raw_cvar) if isinstance(_raw_cvar, int) else 0

    # Default: each stage uses "expectation"
    _cvar_by_stage: dict[int, dict] = {}  # stage_id -> {"alpha": ..., "lambda": ...}

    if dger_cvar in (1, 2) and case.cvar is not None:
        cvar_file = case.cvar
        const_values: list[float] = cvar_file.valores_constantes or [0.0, 0.0]
        const_alpha = const_values[0] / 100.0
        const_lambda = const_values[1] / 100.0

        if dger_cvar == 1:
            # Same constant CVaR for ALL stages — populate lazily in loop below.
            _cvar_constant: dict | None = {
                "cvar": {"alpha": const_alpha, "lambda": const_lambda}
            }
        else:
            # dger_cvar == 2: build per-stage override maps from DataFrames.
            _cvar_constant = None
            df_alpha = cvar_file.alfa_variavel
            df_lambda = cvar_file.lambda_variavel

            # Build {(year, month): alpha_value_fraction} for study rows only.
            alpha_override: dict[tuple[int, int], float] = {}
            if df_alpha is not None and not df_alpha.empty:
                for _, row in df_alpha.iterrows():
                    y = int(row["data"].year)
                    m = int(row["data"].month)
                    if y < 9000:  # skip post-study sentinel rows (year 9999)
                        val = float(row["valor"])
                        alpha_override[(y, m)] = (
                            val / 100.0 if val != 0.0 else const_alpha
                        )

            lambda_override: dict[tuple[int, int], float] = {}
            if df_lambda is not None and not df_lambda.empty:
                for _, row in df_lambda.iterrows():
                    y = int(row["data"].year)
                    m = int(row["data"].month)
                    if y < 9000:
                        val = float(row["valor"])
                        lambda_override[(y, m)] = (
                            val / 100.0 if val != 0.0 else const_lambda
                        )
    elif dger_cvar in (1, 2) and case.cvar is None:
        logger.warning(
            "dger.cvar=%d requires cvar.dat but file is missing; "
            "falling back to expectation for all stages.",
            dger_cvar,
        )
        dger_cvar = 0
        _cvar_constant = None
    else:
        _cvar_constant = None

    # Announce the chosen risk-measure mode so the conversion log makes
    # the cvar/expectation switch obvious without having to inspect
    # stages.json.  Deterministic mode always wins; otherwise dger.cvar
    # drives the selection (0 → expectation, 1 → constant CVaR, 2 →
    # per-stage CVaR).
    if deterministic:
        logger.info(
            "Risk measure: forcing 'expectation' on every stage "
            "(deterministic mode active — single inflow path, no CVaR tail)."
        )
    elif dger_cvar == 0:
        logger.info("Risk measure: 'expectation' on every stage (dger.cvar=0).")
    elif dger_cvar == 1 and _cvar_constant is not None:
        cvar_p = _cvar_constant["cvar"]
        logger.info(
            "Risk measure: constant CVaR(alpha=%.4f, lambda=%.4f) on every "
            "stage (dger.cvar=1, cvar.dat::valores_constantes).",
            cvar_p["alpha"],
            cvar_p["lambda"],
        )
    elif dger_cvar == 2:
        logger.info(
            "Risk measure: per-stage CVaR with constant fallback "
            "alpha=%.4f, lambda=%.4f (dger.cvar=2, "
            "cvar.dat::alfa_variavel + lambda_variavel).",
            const_alpha,
            const_lambda,
        )

    # Build block duration lookup: {(month, patamar) -> fraction}
    # Patamar.duracao_mensal_patamares columns: data (datetime), patamar (int),
    # valor (float, fraction of month).
    df_pat = patamar.duracao_mensal_patamares
    num_patamares: int = patamar.numero_patamares or 1

    # Index the patamar DataFrame by (year, calendar month, patamar index).
    # The ``data`` column is a datetime; we need year and calendar month.
    # Build a dict {(year, month_1_to_12, patamar_1_based) -> fraction}.
    pat_lookup: dict[tuple[int, int, int], float] = {}
    pat_last_year: int | None = None
    if df_pat is not None and not df_pat.empty:
        for _, row in df_pat.iterrows():
            cal_year = int(row["data"].year)
            cal_month = int(row["data"].month)
            pat_idx = int(row["patamar"])
            fraction = float(row["valor"])
            pat_lookup[(cal_year, cal_month, pat_idx)] = fraction
            if pat_last_year is None or cal_year > pat_last_year:
                pat_last_year = cal_year

    names = block_names(num_patamares)

    # ------------------------------------------------------------------
    # Build study stages (IDs 0 .. N-1).
    # ------------------------------------------------------------------
    stages: list[dict] = []
    transitions: list[dict] = []

    horizon = case.horizon
    # Study runs from mes_inicio to December of (ano_inicio + num_anos - 1).
    study_months = horizon.study_months
    # Post-study adds num_anos_pos full calendar years after that.
    pos_months = horizon.pos_months
    total_months = study_months + pos_months

    year = start_year
    month = start_month
    for stage_id in range(total_months):
        start_date = _month_start_date(year, month)
        end_date = _month_end_date(year, month)
        total_hours = _month_hours(year, month)

        blocks: list[dict] = []
        for pat_idx in range(1, num_patamares + 1):
            fraction = pat_lookup.get((year, month, pat_idx))
            if fraction is None and pat_last_year is not None:
                fraction = pat_lookup.get((pat_last_year, month, pat_idx))
            if fraction is None:
                fraction = 1.0 / num_patamares
                logger.warning(
                    "No patamar duration for year %d, calendar month %d, "
                    "patamar %d; using equal fraction %.4f",
                    year,
                    month,
                    pat_idx,
                    fraction,
                )
            block_hours = fraction * total_hours
            blocks.append(
                {
                    "id": pat_idx - 1,
                    "name": names[pat_idx - 1],
                    "hours": block_hours,
                }
            )

        # risk_measure per stage.  ``dger.cvar``: 0 → expectation; 1 → one constant
        # CVaR(alpha,lambda) on every stage (reuse ``_cvar_constant``); 2 → per-stage
        # alpha/lambda from cvar.dat, falling back to ``valores_constantes``.
        # Deterministic mode overrides all: a single inflow path has no tail for
        # CVaR to penalise, so expectation is correct.
        risk_measure: str | dict
        if deterministic or dger_cvar == 0:
            risk_measure = "expectation"
        elif dger_cvar == 1 and _cvar_constant is not None:
            risk_measure = _cvar_constant
        elif dger_cvar == 2:
            a = alpha_override.get((year, month), const_alpha)
            lbd = lambda_override.get((year, month), const_lambda)
            risk_measure = {"cvar": {"alpha": a, "lambda": lbd}}
        else:
            # Defensive: should only fire if dger.cvar ∈ {1, 2} but cvar.dat
            # was missing (the top-of-function block downgrades to
            # dger_cvar=0 and warns, so we shouldn't reach here in practice).
            risk_measure = "expectation"

        stage_entry: dict = {
            "id": stage_id,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "season_id": month - 1,  # 0-based: Jan=0, ..., Dec=11
            "blocks": blocks,
            "num_openings": num_scenarios,
            "risk_measure": risk_measure,
            "state_variables": {
                "storage": True,
                # Inflow lags add no information in deterministic mode — every
                # stage already sees a single fixed historical residual — so
                # drop them to keep the LP smaller.
                "inflow_lags": not deterministic,
            },
        }
        if deterministic:
            # The source model's deterministic mode: each stage samples its inflow
            # residual directly from the (single) historical scenario instead of drawing
            # from the synthetic PAR(p) distribution.
            stage_entry["sampling_method"] = "historical_residuals"
        stages.append(stage_entry)

        if stage_id < total_months - 1:
            transitions.append(
                {
                    "source_id": stage_id,
                    "target_id": stage_id + 1,
                    "probability": 1.0,
                }
            )

        month += 1
        if month > 12:
            month = 1
            year += 1

    # ------------------------------------------------------------------
    # Build pre-study stages (IDs -N .. -1), if requested.
    # ------------------------------------------------------------------
    pre_study_stages: list[dict] = []
    num_anos_pre = dger.num_anos_pre_estudo or 0
    if num_anos_pre > 0:
        num_pre_months = num_anos_pre * 12
        pre_year = start_year
        pre_month = start_month
        # Walk backwards num_pre_months months from the study start.
        pre_dates: list[tuple[int, int]] = []
        for _ in range(num_pre_months):
            pre_month -= 1
            if pre_month < 1:
                pre_month = 12
                pre_year -= 1
            pre_dates.append((pre_year, pre_month))

        # pre_dates is in reverse order; reverse it so IDs go -N..-1.
        pre_dates.reverse()

        for offset, (py, pm) in enumerate(pre_dates):
            pre_id = -(num_pre_months - offset)
            pre_study_stages.append(
                {
                    "id": pre_id,
                    "start_date": _month_start_date(py, pm).isoformat(),
                    "end_date": _month_end_date(py, pm).isoformat(),
                    "season_id": pm - 1,  # 0-based calendar month index
                }
            )

    policy_graph: dict = {
        "type": "finite_horizon",
        "annual_discount_rate": annual_discount_rate,
        "transitions": transitions,
    }

    season_definitions: dict = monthly_season_definitions()

    result: dict = {
        "$schema": cobre_schemas.schema_url_for("stages.json"),
        "season_definitions": season_definitions,
        "policy_graph": policy_graph,
        "stages": stages,
    }
    if pre_study_stages:
        result["pre_study_stages"] = pre_study_stages

    return result


def _is_deterministic_mode(
    case: NewaveCase,
    dger: Dger,
) -> bool:
    """True when the source model's hidden 'deterministic mode' is active.

    Triggered by the combination of:

    - ``dger.num_forwards == 1`` (one forward pass)
    - ``dger.num_aberturas == 1`` (one backward opening)
    - ``dger.tipo_simulacao_final == 2`` (historical final simulation)
    - ``shist.varredura == 0`` (explicit historical years)
    - ``shist.anos_inicio_simulacoes`` has exactly one entry

    When all five hold, the source model drops the stochastic machinery: training
    and simulation both replay the same single historical scenario and each stage
    samples residuals directly off the history instead of drawing from a synthetic
    distribution.  Mirror the same configuration here so the converted cobre case
    reproduces the source model's behavior.
    """
    if (dger.num_forwards or 0) != 1:
        return False
    if (dger.num_aberturas or 0) != 1:
        return False
    if dger.tipo_simulacao_final != 2:
        return False
    shist = case.shist
    if shist is None:
        return False
    if shist.varredura != 0:
        return False
    years = shist.anos_inicio_simulacoes
    return bool(years) and len(years) == 1


def _historical_years_from_shist(
    case: NewaveCase,
    dger: Dger,
) -> list[int] | dict[str, int]:
    """Build cobre's ``historical_years`` from ``shist.dat``.

    The source model's ``shist.dat`` controls which historical years drive the final
    simulation when ``dger.tipo_simulacao_final == 2``:

    - ``shist.varredura == 0`` — explicit list in ``anos_inicio_simulacoes``;
      emit as an array.
    - ``shist.varredura == 1`` — range starting at ``ano_inicio_varredura``;
      the end is the most recent historical year for which a scenario still
      has enough forward data to cover the full simulation horizon
      (``num_anos_estudo + num_anos_pos_estudo``).  Computed as
      ``ano_inicio_estudo − 1 − horizon_years + 1 = ano_inicio_estudo
      − horizon_years``: the last historical year is taken to be
      ``ano_inicio_estudo − 1`` and the scenario must fit inside it.

    When ``shist.dat`` is absent, fall back to the pre-existing default of
    ``{from: ano_inicial_historico + 1, to: ano_inicio_estudo - 1}`` so the
    output remains valid even for cases that ship without an Shist file.
    """
    ano_inicio: int = int(dger.ano_inicio_estudo or 2020)
    horizon = study_horizon(dger)
    num_anos = horizon.num_anos
    num_anos_pos = horizon.num_anos_pos

    shist = case.shist
    if shist is None:
        logger.debug("shist.dat not found; using legacy default historical range.")
        ano_ini_hist: int = int(dger.ano_inicial_historico or 1931)
        return {"from": ano_ini_hist + 1, "to": ano_inicio - 1}

    if shist.varredura == 0:
        years_raw = shist.anos_inicio_simulacoes
        if not years_raw:
            logger.warning(
                "shist.dat has varredura=0 but anos_inicio_simulacoes is empty;"
                " falling back to a single-year list at ano_inicio_estudo - 1."
            )
            return [ano_inicio - 1]
        return [int(y) for y in years_raw]

    # varredura == 1 → range.  The most recent valid start year is
    # ano_inicio_estudo - horizon_years so the scenario fits in history.
    horizon_years = num_anos + num_anos_pos
    range_from = int(shist.ano_inicio_varredura or (dger.ano_inicial_historico or 1931))
    range_to = ano_inicio - horizon_years
    if range_to < range_from:
        logger.warning(
            "shist.dat varredura range collapses: ano_inicio_varredura=%d but"
            " latest valid year=%d (horizon=%d years); clamping range_to=range_from.",
            range_from,
            range_to,
            horizon_years,
        )
        range_to = range_from
    return {"from": range_from, "to": range_to}


def _count_historical_years(
    historical_years: list[int] | dict[str, int],
) -> int:
    """Return how many distinct start-years are covered by a historical_years spec —
    equivalently the number of simulation scenarios the source model will run."""
    if isinstance(historical_years, list):
        return len(historical_years)
    # Range form: {"from": int, "to": int} — inclusive both ends.
    return max(0, int(historical_years["to"]) - int(historical_years["from"]) + 1)


def convert_config(case: NewaveCase) -> dict:
    """Convert the source model training parameters to a Cobre ``config.json`` dict.

    Reads ``dger.dat`` from *case* and produces a dict that conforms
    to ``config.schema.json``.

    Parameters
    ----------
    case:
        Parsed the source model case.

    Returns
    -------
    dict
        JSON-serializable dict conforming to ``config.schema.json``.
    """
    dger = case.dger

    forward_passes: int = dger.num_forwards or 1
    max_iterations: int = dger.num_max_iteracoes or 200
    num_series: int = dger.num_series_sinteticas or 200
    max_order: int = dger.ordem_maxima_parp or 6
    num_openings: int = dger.num_aberturas or 1

    # consideracao_media_anual_afluencias (dger.dat line 83):
    # 0 → classical PAR(p), maps to Cobre "pacf" 1, 2, 3 → PAR(p)-A variants; Cobre
    # implements the exact PDDE form (the source model option 3) under "pacf_annual".
    consideracao_anual: int | None = dger.consideracao_media_anual_afluencias
    if consideracao_anual is None:
        order_selection: str | None = None
    elif consideracao_anual == 0:
        order_selection = "pacf"
    else:
        if consideracao_anual not in (1, 2, 3):
            logger.warning(
                "Unexpected consideracao_media_anual_afluencias=%d; "
                "treating as PAR(p)-A (pacf_annual).",
                consideracao_anual,
            )
        elif consideracao_anual in (1, 2):
            logger.warning(
                "consideracao_media_anual_afluencias=%d (approximate PAR(p)-A) "
                "mapped to Cobre 'pacf_annual', which implements only the exact "
                "12-axis variant (NEWAVE option 3).",
                consideracao_anual,
            )
        order_selection = "pacf_annual"

    tipo_execucao: int = dger.tipo_execucao if dger.tipo_execucao is not None else 1
    tipo_simulacao_final: int = (
        dger.tipo_simulacao_final if dger.tipo_simulacao_final is not None else 1
    )
    considera_reamostragem: int = (
        dger.considera_reamostragem_cenarios
        if dger.considera_reamostragem_cenarios is not None
        else 0
    )

    # impressao_estados_geracao_cortes (dger.dat line 90): when 0, the source model
    # writes the visited cut-generation states (the per-stage cortese*.dat files).
    # Mirror that on the Cobre side via `exports.states` so the two models' visited
    # forward-pass trial points can be compared.  A None or non-zero value leaves the
    # Cobre default (states export off).
    export_states: bool = dger.impressao_estados_geracao_cortes == 0

    # tipo_execucao: 0 = simulation only, 1 = training (+ simulation).
    training_enabled: bool = tipo_execucao == 1

    # tipo_simulacao_final: 0 = disabled, 1 = out_of_sample, 2 = historical.
    # When tipo_execucao == 0 (simulation-only mode), simulation is always on.
    if tipo_execucao == 0:
        simulation_enabled = True
    else:
        simulation_enabled = tipo_simulacao_final != 0

    # -- Cut selection -- The source model's cut-selection knobs are independent for the
    # forward and backward passes (`selecao_de_cortes_forward` /
    # `selecao_de_cortes_backward`); cobre's training pipeline applies a single toggle
    # to both passes.  Mirror the union: cut selection is enabled if the source model
    # turned it on for at least one direction, and only disabled when both flags are 0.
    forward_sel: int = dger.selecao_de_cortes_forward or 0
    backward_sel: int = dger.selecao_de_cortes_backward or 0
    cut_selection_enabled: bool = (forward_sel == 1) or (backward_sel == 1)

    # cobre's `training.cut_selection` keeps two always-on knobs at the top level plus a
    # tagged `selection` object that names the method and carries only that method's
    # parameters; omitting `selection` disables row selection. We mirror The source
    # model's limited-memory Level-1 selection with `method = "lml1"`. The new lml1 is
    # value-based (`tie_tolerance`, default 1e-10), so the source model's old
    # memory-window knob has no equivalent and is intentionally dropped.
    cut_selection: dict = {"row_activity_tolerance": 1e-6}
    if cut_selection_enabled:
        cut_selection["selection"] = {
            "method": "lml1",
            "check_frequency": 1,
        }

    # -- Backward-pass scheduler --
    # Opt into ``by_node`` so each backward work unit is a (trial point, opening
    # block) pair rather than a whole trial point, with block size
    # ceil(num_openings / 2).  This coincides with cobre's own per-node default but
    # pins the value taken from the source deck rather than relying on it.
    parallelism: dict = {
        "backward_scheduler": {
            "method": "by_node",
            "block_size": math.ceil(num_openings / 2),
        }
    }

    # -- Training scenario source --
    training_section: dict = {
        "selection": {"method": "sampled", "forward_passes": forward_passes},
        "stopping_rules": [
            {"type": "iteration_limit", "limit": max_iterations},
        ],
        "cut_selection": cut_selection,
        "parallelism": parallelism,
    }
    if not training_enabled:
        training_section["enabled"] = False
    if training_enabled and considera_reamostragem == 1:
        training_section["scenario_source"] = {
            "seed": 42,
            "inflow": {"scheme": "out_of_sample"},
        }

    # Deterministic mode: training reuses the simulation's historical pool
    # (single year), so set the training scenario_source to mirror it.
    deterministic = _is_deterministic_mode(case, dger)
    if training_enabled and deterministic:
        training_section["scenario_source"] = {
            "seed": 42,
            "inflow": {"scheme": "historical"},
            "historical_years": _historical_years_from_shist(case, dger),
        }

    # -- Simulation scenario source --
    # Priority: tipo_simulacao_final=2 forces "historical"; otherwise
    # considera_reamostragem determines the scheme (1=out_of_sample, 0=in_sample).
    simulation_section: dict = {
        "enabled": simulation_enabled,
        "selection": {"method": "sampled", "num_scenarios": num_series},
    }
    if simulation_enabled:
        if tipo_simulacao_final == 2:
            inflow_scheme = "historical"
        elif considera_reamostragem == 1:
            inflow_scheme = "out_of_sample"
        else:
            inflow_scheme = "in_sample"
        sim_source: dict = {"seed": 42, "inflow": {"scheme": inflow_scheme}}
        if inflow_scheme == "historical":
            historical_years = _historical_years_from_shist(case, dger)
            sim_source["historical_years"] = historical_years
            # In historical mode each scenario is one (start-year, member) tuple; the
            # number of scenarios is fully determined by the size of the historical
            # pool.  Override num_series_sinteticas so the cobre case doesn't request
            # more scenarios than the source model generates.
            simulation_section["selection"]["num_scenarios"] = _count_historical_years(
                historical_years
            )
        simulation_section["scenario_source"] = sim_source

    # Deterministic-mode workaround: force max_order = 0 so the LP carries
    # no inflow-lag state.  Cobre's SDDP exhibits a negative-gap regression
    # when ``max_par_order > 0`` is combined with the sparse cut mask (see
    # the per-hydro lag exclusion in ``StageIndexer::set_nonzero_mask``);
    # disabling lags is the only safe knob from the bridge side until the
    # cobre-side fix lands.  Lag state has no informational value in a
    # deterministic case — every stage already sees a single fixed inflow
    # path — so this is a no-op for correctness on this run mode.
    if deterministic and max_order > 0:
        logger.info(
            "Deterministic mode: forcing estimation.max_order from %d to 0 "
            "to avoid cobre SDDP negative-gap regression triggered by the "
            "sparse cut mask when inflow-lag state is present.",
            max_order,
        )
        max_order = 0
        order_selection = "pacf"

    estimation: dict = {"max_order": max_order}
    if order_selection is not None:
        estimation["order_selection"] = order_selection

    config: dict = {
        "$schema": cobre_schemas.schema_url_for("config.json"),
        "estimation": estimation,
        "training": training_section,
        "modeling": {
            "inflow_non_negativity": {
                "method": "truncation_with_penalty",
            },
        },
        "exports": {
            "states": export_states,
            "stochastic": True,
        },
        "simulation": simulation_section,
    }

    return config
