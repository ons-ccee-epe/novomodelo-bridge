"""Builders used only by tests/decomp/."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date, timedelta

import pandas as pd

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
from novomodelo_bridge.decomp.temporal import OperativeStage


def flat_turbine_limits(
    id_map: DecompIdMap,
    n_stages: int,
    cap_m3s: float = 100.0,
    *,
    group_counts: Mapping[int, int] | None = None,
) -> TurbineLimits:
    """Every operated plant's turbined-flow cap at *cap_m3s* on every stage,
    and its FPHA fit range at its groups' summed caps.

    A plant has one unit group unless *group_counts* gives its count by code.
    """
    counts = group_counts or {}
    return TurbineLimits(
        {
            (code, group_id): (cap_m3s,) * n_stages
            for code in id_map.hydro_codes
            for group_id in range(counts.get(code, 1))
        },
        fit_maxima={
            code: (cap_m3s * counts.get(code, 1),) * n_stages
            for code in id_map.hydro_codes
        },
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


class StubRestricoes:
    """Return a preset DataFrame (or ``None``) for each ``Restricoes`` accessor."""

    #: The file's registers in order, as ``Restricoes.data`` holds them; a stub
    #: holds no line kept as plain text.
    data: tuple[object, ...] = ()

    def __init__(self, **frames: pd.DataFrame) -> None:
        self._frames = frames

    def expressao_eletrica(self, df: bool = True) -> pd.DataFrame | None:
        return self._frames.get("expressao_eletrica")

    def alias_eletrico_valor_periodo_patamar(
        self, df: bool = True
    ) -> pd.DataFrame | None:
        return self._frames.get("alias_eletrico_valor_periodo_patamar")

    def alias_eletrico(self, df: bool = True) -> pd.DataFrame | None:
        return self._frames.get("alias_eletrico")

    def restricao_eletrica_horizonte_periodo(
        self, df: bool = True
    ) -> pd.DataFrame | None:
        return self._frames.get("restricao_eletrica_horizonte_periodo")

    def restricao_eletrica_horizonte_data(self, df: bool = True) -> pd.DataFrame | None:
        return self._frames.get("restricao_eletrica_horizonte_data")

    def restricao_eletrica_limite_formula_periodo_patamar(
        self, df: bool = True
    ) -> pd.DataFrame | None:
        return self._frames.get("restricao_eletrica_limite_formula_periodo_patamar")

    def restricao_eletrica_limites_formula_data_patamar(
        self, df: bool = True
    ) -> pd.DataFrame | None:
        return self._frames.get("restricao_eletrica_limites_formula_data_patamar")

    def restricao_eletrica_inequacao_periodo_patamar(
        self, df: bool = True
    ) -> pd.DataFrame | None:
        return self._frames.get("restricao_eletrica_inequacao_periodo_patamar")

    def restricao_eletrica_habilita(self, df: bool = True) -> pd.DataFrame | None:
        return self._frames.get("restricao_eletrica_habilita")

    def restricao_eletrica_regra_ativacao(self, df: bool = True) -> pd.DataFrame | None:
        return self._frames.get("restricao_eletrica_regra_ativacao")

    def restricao_eletrica_tratamento_violacao(
        self, df: bool = True
    ) -> pd.DataFrame | None:
        return self._frames.get("restricao_eletrica_tratamento_violacao")

    def restricao_eletrica_tratamento_violacao_periodo(
        self, df: bool = True
    ) -> pd.DataFrame | None:
        return self._frames.get("restricao_eletrica_tratamento_violacao_periodo")

    def restricao_eletrica_formula(self, df: bool = True) -> pd.DataFrame | None:
        return self._frames.get("restricao_eletrica_formula")

    def restricao_eletrica_inequacao(self, df: bool = True) -> pd.DataFrame | None:
        return self._frames.get("restricao_eletrica_inequacao")

    def restricao_eletrica_formula_periodo_patamar(
        self, df: bool = True
    ) -> pd.DataFrame | None:
        return self._frames.get("restricao_eletrica_formula_periodo_patamar")

    def restricao_eletrica_formula_data_patamar(
        self, df: bool = True
    ) -> pd.DataFrame | None:
        return self._frames.get("restricao_eletrica_formula_data_patamar")

    def re(self, df: bool = True) -> pd.DataFrame | None:
        return self._frames.get("re")

    def re_horiz_per(self, df: bool = True) -> pd.DataFrame | None:
        return self._frames.get("re_horiz_per")

    def re_lim_form_per_pat(self, df: bool = True) -> pd.DataFrame | None:
        return self._frames.get("re_lim_form_per_pat")

    def re_horiz_data(self, df: bool = True) -> pd.DataFrame | None:
        return self._frames.get("re_horiz_data")

    def re_lim_form_data_pat(self, df: bool = True) -> pd.DataFrame | None:
        return self._frames.get("re_lim_form_data_pat")

    def alias_elet(self, df: bool = True) -> pd.DataFrame | None:
        return self._frames.get("alias_elet")

    def alias_elet_val_per_pat(self, df: bool = True) -> pd.DataFrame | None:
        return self._frames.get("alias_elet_val_per_pat")

    def re_regra_ativa(self, df: bool = True) -> pd.DataFrame | None:
        return self._frames.get("re_regra_ativa")

    def re_habilita(self, df: bool = True) -> pd.DataFrame | None:
        return self._frames.get("re_habilita")

    def re_trat_viol(self, df: bool = True) -> pd.DataFrame | None:
        return self._frames.get("re_trat_viol")

    def re_trat_viol_per(self, df: bool = True) -> pd.DataFrame | None:
        return self._frames.get("re_trat_viol_per")

    def re_per_pat(self, df: bool = True) -> pd.DataFrame | None:
        return self._frames.get("re_per_pat")

    def re_data_pat(self, df: bool = True) -> pd.DataFrame | None:
        return self._frames.get("re_data_pat")


def stage(index: int) -> OperativeStage:
    return OperativeStage(
        index=index,
        start_date=date(2026, 7, 4),
        end_date=date(2026, 7, 11),
        season_id=6,
        block_hours=(168.0,),
    )


def calendar(n_stages: int) -> list[OperativeStage]:
    return [stage(i) for i in range(n_stages)]


def weekly_calendar(n_stages: int) -> list[OperativeStage]:
    """Contiguous one-week stages from 2026-07-04, each with its own dates (the
    last ``end_date`` exclusive), unlike :func:`calendar`."""
    first = date(2026, 7, 4)
    return [
        OperativeStage(
            index=i,
            start_date=first + timedelta(weeks=i),
            end_date=first + timedelta(weeks=i + 1),
            season_id=6,
            block_hours=(168.0,),
        )
        for i in range(n_stages)
    ]


def expressao(*rows: tuple) -> pd.DataFrame:
    """EXPRESSAO-ELETRICA rows: (codigo_expressao, identificador_expressao, formula)."""
    return pd.DataFrame(
        rows, columns=["codigo_expressao", "identificador_expressao", "formula"]
    )


def alias(*rows: tuple) -> pd.DataFrame:
    """ALIAS-ELETRICO rows: (codigo_alias, identificador_alias)."""
    return pd.DataFrame(rows, columns=["codigo_alias", "identificador_alias"])


def alias_valor(*rows: tuple) -> pd.DataFrame:
    """ALIAS-ELETRICO-VALOR-PERIODO-PATAMAR rows: (codigo_alias, estagio_inicio,
    estagio_fim, patamar, valor)."""
    return pd.DataFrame(
        rows,
        columns=["codigo_alias", "estagio_inicio", "estagio_fim", "patamar", "valor"],
    )


def re_formula(*rows: tuple) -> pd.DataFrame:
    """RESTRICAO-ELETRICA-FORMULA rows: (codigo_restricao, formula)."""
    return pd.DataFrame(rows, columns=["codigo_restricao", "formula"])


def re_lim_formula(*rows: tuple) -> pd.DataFrame:
    """RESTRICAO-ELETRICA-LIMITES-FORMULA-PERIODO-PATAMAR rows: (codigo_restricao,
    estagio_inicio, estagio_fim, patamar, limite_inferior, limite_superior)."""
    return pd.DataFrame(
        rows,
        columns=[
            "codigo_restricao",
            "estagio_inicio",
            "estagio_fim",
            "patamar",
            "limite_inferior",
            "limite_superior",
        ],
    )


def re_formula_per_pat(*rows: tuple) -> pd.DataFrame:
    """RESTRICAO-ELETRICA-FORMULA-PERIODO-PATAMAR rows: (codigo_restricao,
    estagio_inicio, estagio_fim, patamar, formula)."""
    return pd.DataFrame(
        rows,
        columns=[
            "codigo_restricao",
            "estagio_inicio",
            "estagio_fim",
            "patamar",
            "formula",
        ],
    )


def re_inequacao(*rows: tuple) -> pd.DataFrame:
    """RESTRICAO-ELETRICA-INEQUACAO rows: (codigo_restricao, formula, operador,
    formula_limite)."""
    return pd.DataFrame(
        rows, columns=["codigo_restricao", "formula", "operador", "formula_limite"]
    )


def re_inequacao_per_pat(*rows: tuple) -> pd.DataFrame:
    """RESTRICAO-ELETRICA-INEQUACAO-PERIODO-PATAMAR rows: (codigo_restricao,
    estagio_inicio, estagio_fim, patamar, formula, operador, formula_limite)."""
    return pd.DataFrame(
        rows,
        columns=[
            "codigo_restricao",
            "estagio_inicio",
            "estagio_fim",
            "patamar",
            "formula",
            "operador",
            "formula_limite",
        ],
    )


def horizonte(*rows: tuple) -> pd.DataFrame:
    """RESTRICAO-ELETRICA-HORIZONTE-PERIODO rows: (codigo_restricao,
    estagio_inicio, estagio_fim)."""
    return pd.DataFrame(
        rows, columns=["codigo_restricao", "estagio_inicio", "estagio_fim"]
    )


def _read_as_datetimes(frame: pd.DataFrame) -> pd.DataFrame:
    """*frame* with ``data_inicio``/``data_fim`` as ``datetime64``, the dtype
    idecomp's date columns have (a blank or unparseable cell is ``NaT``)."""
    return frame.assign(
        data_inicio=pd.to_datetime(frame["data_inicio"]),
        data_fim=pd.to_datetime(frame["data_fim"]),
    )


def horizonte_data(*rows: tuple) -> pd.DataFrame:
    """RESTRICAO-ELETRICA-HORIZONTE-DATA rows: (codigo_restricao, data_inicio,
    data_fim), the dates as strings or datetimes."""
    return _read_as_datetimes(
        pd.DataFrame(rows, columns=["codigo_restricao", "data_inicio", "data_fim"])
    )


def re_lim_formula_data(*rows: tuple) -> pd.DataFrame:
    """RESTRICAO-ELETRICA-LIMITES-FORMULA-DATA-PATAMAR rows: (codigo_restricao,
    data_inicio, data_fim, patamar, limite_inferior, limite_superior), the dates
    as strings or datetimes."""
    return _read_as_datetimes(
        pd.DataFrame(
            rows,
            columns=[
                "codigo_restricao",
                "data_inicio",
                "data_fim",
                "patamar",
                "limite_inferior",
                "limite_superior",
            ],
        )
    )


def re_formula_data_pat(*rows: tuple) -> pd.DataFrame:
    """RESTRICAO-ELETRICA-FORMULA-DATA-PATAMAR rows: (codigo_restricao,
    data_inicio, data_fim, patamar, formula), the dates as strings or datetimes."""
    return _read_as_datetimes(
        pd.DataFrame(
            rows,
            columns=[
                "codigo_restricao",
                "data_inicio",
                "data_fim",
                "patamar",
                "formula",
            ],
        )
    )


def habilita(*rows: tuple) -> pd.DataFrame:
    """RESTRICAO-ELETRICA-HABILITA rows: (codigo_restricao, codigo_regra_ativacao)."""
    return pd.DataFrame(rows, columns=["codigo_restricao", "codigo_regra_ativacao"])


def regra(*rows: tuple) -> pd.DataFrame:
    """RESTRICAO-ELETRICA-REGRA-ATIVACAO rows: (codigo_regra_ativacao,
    regra_ativacao)."""
    return pd.DataFrame(rows, columns=["codigo_regra_ativacao", "regra_ativacao"])


def tratamento_violacao(*rows: tuple) -> pd.DataFrame:
    """RESTRICAO-ELETRICA-TRATAMENTO-VIOLACAO rows: (codigo_restricao,
    tipo_violacao, custo_violacao)."""
    return pd.DataFrame(
        rows, columns=["codigo_restricao", "tipo_violacao", "custo_violacao"]
    )


def tratamento_violacao_periodo(*rows: tuple) -> pd.DataFrame:
    """RESTRICAO-ELETRICA-TRATAMENTO-VIOLACAO-PERIODO rows: (codigo_restricao,
    estagio_inicio, estagio_fim, tipo_violacao, custo_violacao)."""
    return pd.DataFrame(
        rows,
        columns=[
            "codigo_restricao",
            "estagio_inicio",
            "estagio_fim",
            "tipo_violacao",
            "custo_violacao",
        ],
    )


SHORT_FORM_ACCESSORS = {
    "restricao_eletrica_formula": "re",
    "restricao_eletrica_horizonte_periodo": "re_horiz_per",
    "restricao_eletrica_limite_formula_periodo_patamar": "re_lim_form_per_pat",
    "restricao_eletrica_horizonte_data": "re_horiz_data",
    "restricao_eletrica_limites_formula_data_patamar": "re_lim_form_data_pat",
    "alias_eletrico": "alias_elet",
    "alias_eletrico_valor_periodo_patamar": "alias_elet_val_per_pat",
    "restricao_eletrica_regra_ativacao": "re_regra_ativa",
    "restricao_eletrica_habilita": "re_habilita",
    "restricao_eletrica_tratamento_violacao": "re_trat_viol",
    "restricao_eletrica_tratamento_violacao_periodo": "re_trat_viol_per",
    "restricao_eletrica_formula_periodo_patamar": "re_per_pat",
    "restricao_eletrica_formula_data_patamar": "re_data_pat",
}


def spelled(frames: Mapping[str, pd.DataFrame], *, short: bool) -> StubRestricoes:
    """A stub holding *frames* (keyed by long accessor) under the long accessor
    names, or under the short-form alias where the card has one."""
    return StubRestricoes(
        **{
            (SHORT_FORM_ACCESSORS.get(name, name) if short else name): frame
            for name, frame in frames.items()
        }
    )
