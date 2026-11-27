"""The LIBs-era electrical-constraint cards in ``convert_decomp_case`` -- the
generic-constraint files they are translated into and their census.
Discovering the file is covered with the rest of file discovery, in
``tests/decomp/test_files.py``.

Synthetic stub decks only, mirroring the ``test_pipeline`` module's own
``_run_cadastro_pipeline`` convention -- no ``import novomodelo`` at module scope.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.errors import FieldParseError, UnsupportedSourceError
from novomodelo_bridge.core.generic_constraint_builder import GENERIC_BOUNDS_SCHEMA
from novomodelo_bridge.core.libs_index import UNREAD_LINES_CODE
from novomodelo_bridge.decomp.bounds_accumulator import BoundContribution
from novomodelo_bridge.decomp.constraint_registers import (
    ConstraintCensus,
    ConstraintRecord,
    ConstraintTerm,
    StageBounds,
)
from novomodelo_bridge.decomp.converters.electrical_constraints import (
    GenericOutputs,
)
from novomodelo_bridge.decomp.converters.libs_electrical import (
    ElectricalAlias,
    ElectricalRestriction,
    LibsElectricalModel,
    ViolationRow,
)
from novomodelo_bridge.decomp.converters.network import _LINE_BOUNDS_SCHEMA
from novomodelo_bridge.decomp.converters.scalar_parameters import (
    build_decomp_scalar_parameters,
)
from novomodelo_bridge.decomp.converters.thermal import (
    _THERMAL_COST_SCHEMA,
    ThermalBounds,
)
from novomodelo_bridge.decomp.files import DecompFiles
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.pipeline import (
    ConversionReport,
    _electrical_census_diagnostic,
)
from novomodelo_bridge.decomp.temporal import build_operative_calendar
from novomodelo_bridge.novomodelo.schemas import SCHEMA_URLS
from tests.conftest import make_decomp_case
from tests.decomp.conftest import flat_turbine_limits

# ---------------------------------------------------------------------------
# Tier-1 pipeline wiring -- a fully synthetic mock deck, mirroring
# the ``test_pipeline`` module's ``_run_cadastro_pipeline`` convention.
# ---------------------------------------------------------------------------


def _plant_row(
    name: str, sub: int, jusante: int, vmin: float, vmax: float, posto: int
) -> dict:
    return {
        "nome_usina": name,
        "submercado": sub,
        "codigo_usina_jusante": jusante,
        "posto": posto,
        "desvio": 0,
        "volume_minimo": vmin,
        "volume_maximo": vmax,
        "tipo_regulacao": "M",
        "volume_referencia": vmin + 0.5 * (vmax - vmin),
        "numero_conjuntos_maquinas": 1,
        "maquinas_conjunto_1": 2,
        "vazao_nominal_conjunto_1": 100.0,
        "potencia_nominal_conjunto_1": 50.0,
        "teif": 0.0,
        "ip": 0.0,
        "a0_volume_cota": 100.0,
        "a1_volume_cota": 0.0,
        "a2_volume_cota": 0.0,
        "a3_volume_cota": 0.0,
        "a4_volume_cota": 0.0,
        "a0_cota_area": 1.0,
        "a1_cota_area": 0.0,
        "a2_cota_area": 0.0,
        "a3_cota_area": 0.0,
        "a4_cota_area": 0.0,
        "canal_fuga_medio": 20.0,
        "produtibilidade_especifica": 0.009,
        "tipo_perda": 0,
        "perdas": 0.0,
        "influencia_vertimento_canal_fuga": 1,
    }


_SPLIT_PLANT_CODE = 66
_ITAIPU_CODE = _SPLIT_PLANT_CODE


def _hidr_frame(second_plant_code: int = 2) -> pd.DataFrame:
    """Two plants, codes 1 and *second_plant_code*, both on bus (submercado) 1,
    no cascade; the split plant (code 66) has two machine sets."""
    second = _plant_row("PLANT_TWO", 1, 0, 10.0, 50.0, posto=12)
    if second_plant_code == _SPLIT_PLANT_CODE:
        second |= {
            "numero_conjuntos_maquinas": 2,
            "maquinas_conjunto_2": 2,
            "vazao_nominal_conjunto_2": 100.0,
            "potencia_nominal_conjunto_2": 50.0,
        }
    df = pd.DataFrame(
        {
            1: _plant_row("PLANT_ONE", 1, 0, 20.0, 100.0, posto=11),
            second_plant_code: second,
        }
    ).T
    df.index.name = "codigo_usina"
    return df


def _uh_frame(second_plant_code: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "codigo_usina": code,
                "volume_inicial": 50.0,
                "vazao_defluente_minima": None,
                "volume_morto_inicial": None,
                "codigo_ree": 1,
            }
            for code in (1, second_plant_code)
        ]
    )


def _id_map(second_plant_code: int = 2) -> DecompIdMap:
    return DecompIdMap(
        bus_codes=(1,), bus_names=("SE",), hydro_codes=(1, second_plant_code)
    )


def _split_plant_frequencies() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"codigo_usina": _SPLIT_PLANT_CODE, "frequencia": frequency}
            for frequency in (50.0, 60.0)
        ]
    )


_N_BLOCKS = 3


def _calendar():
    hours = [[15.0, 64.0, 89.0]] * 2 + [[63.0, 280.0, 401.0]]
    return build_operative_calendar(date(2026, 7, 18), hours)


class _MockDadger:
    """Covers only what the real (unmocked) consumers this fixture exercises
    need -- mirrors the ``test_pipeline`` module's own ``_CadastroDadger``."""

    class _Tx:
        taxa = 12.0

    class _Data:
        @staticmethod
        def of_type(_register: type) -> list:
            return []

    def __init__(
        self,
        second_plant_code: int = 2,
        *,
        pq: pd.DataFrame | None = None,
        ri: pd.DataFrame | None = None,
    ) -> None:
        self.tx = self._Tx()
        self._uh = _uh_frame(second_plant_code)
        self._pq = pq
        self._ri = ri
        self.data = self._Data()

    def uh(self, df: bool = False) -> pd.DataFrame:
        return self._uh

    def mp(self, df: bool = False) -> pd.DataFrame:
        return _split_plant_frequencies()

    def fd(self, df: bool = False) -> pd.DataFrame:
        return _split_plant_frequencies()

    def pq(self, df: bool = False) -> pd.DataFrame | None:
        return self._pq

    def ri(self, df: bool = False) -> pd.DataFrame | None:
        return self._ri

    def vi(self, df: bool = False) -> pd.DataFrame | None:
        return None

    def ve(self, df: bool = False) -> pd.DataFrame | None:
        return None

    def vl(self, df: bool = False) -> pd.DataFrame | None:
        return None

    def vu(self, df: bool = False) -> pd.DataFrame | None:
        return None

    def va(self, df: bool = False) -> pd.DataFrame | None:
        return None

    def ti(self, df: bool = False) -> pd.DataFrame | None:
        return None

    def ez(self, df: bool = False) -> pd.DataFrame | None:
        return None

    def fp(self, df: bool = False) -> pd.DataFrame | None:
        return None

    def ac(
        self,
        codigo_usina: int | None = None,
        modificacao: type | None = None,
        df: bool = False,
    ) -> pd.DataFrame | None:
        return None


#: A soft violation treatment with its own cost, so a restriction does not take
#: the default penalty the harness's deck (no deficit cost) cannot derive.
_SOFT_TREATMENT = (ViolationRow(None, None, "soft", 1000.0),)


def _cap_restriction(code: int, plant_code: int, cap: float) -> ElectricalRestriction:
    """A constant INEQUACAO restriction (``ger_usih(plant_code) <= cap``)."""
    return ElectricalRestriction(
        code=code,
        lhs=f"ger_usih({plant_code})",
        operator="<=",
        rhs=str(cap),
        is_formula=False,
        limits={},
        overrides=(),
        violations=_SOFT_TREATMENT,
    )


def _re_record(schedule: tuple[tuple[int, float], ...] = ()) -> ConstraintRecord:
    """A multi-term RE record: it never lowers to a bound, so it is emitted as
    a generic constraint. *schedule* (first coefficient 1.0) is the first
    term's, which a varying schedule makes a per-stage ``@`` parameter."""
    return ConstraintRecord(
        family="RE",
        constraint_id=901,
        stage_start=0,
        stage_end=2,
        terms=(
            ConstraintTerm(
                code=1, coefficient=1.0, variable="generation", schedule=schedule
            ),
            ConstraintTerm(code=2, coefficient=1.0, variable="generation"),
        ),
        bounds={
            stage: StageBounds(lower=(None,) * _N_BLOCKS, upper=(100.0,) * _N_BLOCKS)
            for stage in range(3)
        },
        per_block=True,
    )


def _run_libs_pipeline(
    tmp_path: Path,
    *,
    libs_restricao_eletrica: Path | None = None,
    libs_restricoes: object | None = None,
    libs_electrical_model: object | None = None,
    to_generic: tuple[ConstraintRecord, ...] = (),
    second_plant_code: int = 2,
    pq_frame: pd.DataFrame | None = None,
    ri_frame: pd.DataFrame | None = None,
    diagnostics_out: list[dx.Diagnostic] | None = None,
    report_out: list[ConversionReport] | None = None,
    mocks_out: dict[str, MagicMock] | None = None,
    dry_run: bool = False,
) -> Path:
    """Run ``convert_decomp_case`` against a fully synthetic mock deck,
    patching every converter not under test to a canned return
    value -- mirrors the ``test_pipeline`` module's
    ``_run_cadastro_pipeline`` (duplicated rather than imported, per this
    codebase's own convention of copying a fixture verbatim
    across test modules).

    *libs_restricao_eletrica*/*libs_electrical_model* control
    ``files.libs_restricao_eletrica`` and the (patched)
    ``read_libs_electrical`` return value; *libs_restricoes* is the parsed
    file, a reader with no lines by default; *second_plant_code* is the code of
    the deck's second plant (66 makes it the two-group split plant, and Itaipu,
    so an *ri_frame* is read); *pq_frame*/*ri_frame* are the ``PQ``/``RI``
    registers of the mock dadger; *mocks_out* receives each patched target's
    mock; *dry_run* is forwarded to ``convert_decomp_case``.
    """
    from novomodelo_bridge.decomp import pipeline as decomp_pipeline

    files = DecompFiles(
        revision="rv0",
        dadger=Path("unused/dadger.rv0"),
        vazoes=Path("unused/vazoes.rv0"),
        hidr=Path("unused/hidr.dat"),
        dadgnl=None,
        renovaveis=None,
        polinjus=None,
        libs_restricao_eletrica=libs_restricao_eletrica,
    )
    id_map = _id_map(second_plant_code)
    dadger = _MockDadger(second_plant_code, pq=pq_frame, ri=ri_frame)
    hidr = _hidr_frame(second_plant_code)
    calendar = _calendar()
    case = make_decomp_case(
        files,
        dadger=dadger,
        hidr=hidr,
        id_map=id_map,
        calendar=calendar,
        renovaveis=None,
        dadgnl=None,
        polinjus=None,
        libs_restricao_eletrica=(
            SimpleNamespace(data=[])
            if libs_restricoes is None
            else libs_restricoes
            if libs_restricao_eletrica is not None
            else None
        ),
    )

    productivity_table = pa.table(
        {"equivalent_productivity_mw_per_m3s": pa.array([0.5, 0.6], type=pa.float64())}
    )
    load_stats_table = pa.table(
        {
            "bus_id": pa.array([0, 0, 0], type=pa.int32()),
            "stage_id": pa.array([0, 1, 2], type=pa.int32()),
            "mean_mw": pa.array([10.0, 10.0, 10.0], type=pa.float64()),
        }
    )
    ncs_stats_table = pa.table(
        {
            "ncs_id": pa.array([0, 0, 0], type=pa.int32()),
            "stage_id": pa.array([0, 1, 2], type=pa.int32()),
            "mean": pa.array([5.0, 5.0, 5.0], type=pa.float64()),
        }
    )
    external_inflow_table = pa.table(
        {
            "hydro_id": pa.array([], type=pa.int32()),
            "stage_id": pa.array([], type=pa.int32()),
            "scenario_id": pa.array([], type=pa.int32()),
            "value_m3s": pa.array([], type=pa.float64()),
        }
    )
    baseline_hydro_bounds = [
        BoundContribution(
            family="hydro",
            entity_id=0,
            stage_id=0,
            block_id=None,
            axis="outflow",
            lower=5.0,
            upper=None,
            contributor="RQ",
        )
    ]

    patches: dict[str, object] = {
        "novomodelo_bridge.decomp.pipeline.DecompCase.from_directory": case,
        "novomodelo_bridge.decomp.pipeline.Vazoes.read": object(),
        "novomodelo_bridge.decomp.pipeline.scenarios_conv.terminal_fan_probabilities": [
            1.0
        ],
        "novomodelo_bridge.decomp.pipeline.config_conv.convert_config": {},
        "novomodelo_bridge.decomp.pipeline.network_conv.bus_deficit_costs": {},
        "novomodelo_bridge.decomp.pipeline"
        ".hydro_conv.convert_energy_productivity": productivity_table,
        "novomodelo_bridge.decomp.pipeline.network_conv.convert_buses": {"buses": []},
        "novomodelo_bridge.decomp.pipeline.network_conv.convert_lines": (
            {"lines": []},
            _LINE_BOUNDS_SCHEMA.empty_table(),
        ),
        "novomodelo_bridge.decomp.pipeline.network_conv.convert_pumping_stations": {
            "pumping_stations": []
        },
        "novomodelo_bridge.decomp.pipeline.thermal_conv.convert_thermals": {
            "thermals": []
        },
        "novomodelo_bridge.decomp.pipeline.thermal_conv.ct_submarkets": {},
        "novomodelo_bridge.decomp.pipeline.ncs_conv.convert_non_controllable_sources": {
            "non_controllable_sources": []
        },
        "novomodelo_bridge.decomp.pipeline"
        ".scenarios_conv.convert_external_inflows": external_inflow_table,
        "novomodelo_bridge.decomp.pipeline"
        ".scenarios_conv.convert_recent_observation_windows": [],
        "novomodelo_bridge.decomp.pipeline.load_conv.convert_load_stats": load_stats_table,
        "novomodelo_bridge.decomp.pipeline.load_conv.convert_load_factors": {},
        "novomodelo_bridge.decomp.pipeline.ncs_conv.convert_ncs_stats": ncs_stats_table,
        "novomodelo_bridge.decomp.pipeline.ncs_conv.convert_ncs_factors": {},
        "novomodelo_bridge.decomp.pipeline.thermal_conv.convert_thermal_bounds": (
            ThermalBounds(generation=[], cost=_THERMAL_COST_SCHEMA.empty_table())
        ),
        "novomodelo_bridge.decomp.pipeline"
        ".bounds_conv.convert_hydro_bounds": baseline_hydro_bounds,
        "novomodelo_bridge.decomp.pipeline.hydro_conv.convert_hydro_group_availability": {},
        "novomodelo_bridge.decomp.pipeline.scenarios_conv.expected_incremental_inflows": {},
        "novomodelo_bridge.decomp.pipeline.hydro_conv.tailrace_families": {},
        "novomodelo_bridge.decomp.pipeline.hydro_conv.stage_availability_factors": {},
        "novomodelo_bridge.decomp.pipeline.hydro_conv.compute_turbine_limits": (
            flat_turbine_limits(
                id_map,
                len(case.calendar),
                group_counts={_SPLIT_PLANT_CODE: 2},
            )
        ),
        "novomodelo_bridge.decomp.pipeline.contracts_conv.read_contracts": [],
        "novomodelo_bridge.decomp.pipeline.constraint_registers.read_constraints": (
            ConstraintCensus(by_family={}, to_bounds=(), to_generic=to_generic)
        ),
        "novomodelo_bridge.decomp.pipeline.network_conv.pumping_station_id_map": {},
        "novomodelo_bridge.decomp.pipeline"
        ".constraint_registers.detect_unconverted_rha": None,
        "novomodelo_bridge.decomp.pipeline.libs_electrical_conv.read_libs_electrical": (
            libs_electrical_model
        ),
    }
    with ExitStack() as stack:
        for target, value in patches.items():
            mock = stack.enter_context(patch(target, return_value=value))
            if mocks_out is not None:
                mocks_out[target] = mock
        dst = tmp_path / "case"
        report = decomp_pipeline.convert_decomp_case(
            Path("unused-src"), dst, dry_run=dry_run
        )
    if diagnostics_out is not None:
        diagnostics_out.extend(report.diagnostics)
    if report_out is not None:
        report_out.append(report)
    return dst


_GENERIC_CONSTRAINTS_PATH = "constraints/generic_constraints.json"
_GENERIC_BOUNDS_PATH = "constraints/generic_constraint_bounds.parquet"
#: The files the conversion no longer writes: the cards go to the generic ones.
_ELECTRICAL_PATHS = (
    "constraints/electrical_constraints.json",
    "constraints/electrical_constraint_bounds.parquet",
)
_LIBS_CSV = Path("unused/lib_restricao-eletrica-especial.csv")


def _electrical_codes(diagnostics: list[dx.Diagnostic]) -> set[str]:
    return {d.code for d in diagnostics if d.code.startswith("decomp-libs-electrical")}


class TestLibsElectricalPipelineWiring:
    def test_no_entry_is_a_no_op(self, tmp_path: Path) -> None:
        """No ``RESTRICAO-ELETRICA-ESPECIAL`` entry -- no electrical
        constraint emitted and no libs diagnostic produced."""
        diagnostics_out: list[dx.Diagnostic] = []
        dst = _run_libs_pipeline(
            tmp_path,
            libs_restricao_eletrica=None,
            libs_electrical_model=None,
            diagnostics_out=diagnostics_out,
        )

        assert not (dst / _GENERIC_CONSTRAINTS_PATH).exists()
        assert not (dst / _GENERIC_BOUNDS_PATH).exists()
        assert _electrical_codes(diagnostics_out) == set()

    def test_file_the_reader_finds_no_card_in_emits_no_electrical_diagnostic(
        self, tmp_path: Path
    ) -> None:
        """The entry is declared but ``read_libs_electrical`` returns ``None``:
        the lines the reader keeps as text are already reported by the
        unread-lines diagnostic, so nothing electrical is added."""
        diagnostics_out: list[dx.Diagnostic] = []
        dst = _run_libs_pipeline(
            tmp_path,
            libs_restricao_eletrica=_LIBS_CSV,
            libs_electrical_model=None,
            diagnostics_out=diagnostics_out,
        )

        assert _electrical_codes(diagnostics_out) == set()
        assert not (dst / _GENERIC_CONSTRAINTS_PATH).exists()

    def test_lines_the_reader_keeps_as_text_are_warned(self, tmp_path: Path) -> None:
        from idecomp.libs.restricoes import Restricoes

        libs_file = tmp_path / "lib_restricao-eletrica-especial.csv"
        libs_file.write_text(
            "& comentario\n\nrestricao-eletrica-formula; 1; ger_usih(1)\n",
            encoding="utf-8",
        )
        diagnostics_out: list[dx.Diagnostic] = []
        _run_libs_pipeline(
            tmp_path,
            libs_restricao_eletrica=libs_file,
            libs_restricoes=Restricoes.read(str(libs_file)),
            diagnostics_out=diagnostics_out,
        )

        [unread] = [d for d in diagnostics_out if d.code == UNREAD_LINES_CODE]
        assert unread.severity is dx.Severity.WARNING
        assert unread.table is not None
        assert unread.table.rows == [["restricao-eletrica-formula; 1; ger_usih(1)"]]


_GENERIC_PARAMETERS_PATH = "constraints/generic_parameters.json"
_CENSUS_CODE = "decomp-libs-electrical-translated"
_CAP_MW = 500.0
_ANDE_MW = (3505.0, 3332.0, 2817.5)
_PQ_MW = (10.0, 20.0, 30.0)


def _model(
    formula: str = "ger_usih(1) - carga_ande",
    *,
    violations: tuple[ViolationRow, ...] = _SOFT_TREATMENT,
) -> LibsElectricalModel:
    """One FORMULA restriction, 601: *formula* at most :data:`_CAP_MW` in
    every stage, for all blocks, under *violations*."""
    return LibsElectricalModel(
        expressions={},
        aliases={},
        restrictions={
            601: ElectricalRestriction(
                code=601,
                lhs=formula,
                operator=None,
                rhs=None,
                is_formula=True,
                limits={(stage, None): (None, _CAP_MW) for stage in range(3)},
                overrides=(),
                violations=violations,
            )
        },
        rules={},
    )


def _alias_model(name: str = "MM") -> LibsElectricalModel:
    """:func:`_model` of ``ger_usih(1) - <name>``, the alias *name* declared
    1.0 in every stage."""
    return replace(
        _model(f"ger_usih(1) - {name}"),
        aliases={
            name: ElectricalAlias(
                code=1, name=name, values={(stage, None): 1.0 for stage in range(3)}
            )
        },
    )


def _ri_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [(66, 1, 1, *_ANDE_MW)],
        columns=[
            "codigo_usina",
            "estagio",
            "codigo_submercado",
            *(f"carga_ande_{k}" for k in range(1, _N_BLOCKS + 1)),
        ],
    )


def _pq_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "codigo_submercado": 1,
                "estagio": 1,
                "nome": "SECO_PCHgd",
                **{f"geracao_{k}": mw for k, mw in enumerate(_PQ_MW, start=1)},
            }
        ]
    )


def _run_translated(
    tmp_path: Path,
    model: LibsElectricalModel | None = None,
    *,
    diagnostics_out: list[dx.Diagnostic] | None = None,
    report_out: list[ConversionReport] | None = None,
    dry_run: bool = False,
) -> Path:
    """:func:`_run_libs_pipeline` over *model* (:func:`_model` by default) on
    a deck that operates Itaipu (plant 66, hydro id 1) and declares the ANDE
    load."""
    return _run_libs_pipeline(
        tmp_path,
        libs_restricao_eletrica=_LIBS_CSV,
        libs_electrical_model=_model() if model is None else model,
        second_plant_code=_ITAIPU_CODE,
        ri_frame=_ri_frame(),
        diagnostics_out=diagnostics_out,
        report_out=report_out,
        dry_run=dry_run,
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def _constraint(dst: Path, constraint_id: int) -> dict:
    """The record of *constraint_id* in the case's generic constraints."""
    doc = _read_json(dst / _GENERIC_CONSTRAINTS_PATH)
    [record] = [c for c in doc["constraints"] if c["id"] == constraint_id]
    return record


def _bound_rows(dst: Path, constraint_id: int) -> list[dict]:
    """The rows of *constraint_id* in the case's generic constraint bounds."""
    bounds = pq.read_table(dst / _GENERIC_BOUNDS_PATH)
    return [row for row in bounds.to_pylist() if row["constraint_id"] == constraint_id]


def _block_values(mw: tuple[float, ...]) -> list[list[float]]:
    return [
        [stage, block, value] for stage in range(3) for block, value in enumerate(mw)
    ]


#: The ``generic_parameters.json`` entries of the deck's two hydros (ids 0 and 1).
_HYDRO_PARAMETERS = build_decomp_scalar_parameters([0, 1])["scalar_parameters"]


class TestElectricalConstraintsInGenericFiles:
    def test_cards_are_translated_into_the_generic_files(self, tmp_path: Path) -> None:
        dst = _run_translated(tmp_path)

        doc = _read_json(dst / _GENERIC_CONSTRAINTS_PATH)
        assert doc["$schema"] == SCHEMA_URLS[_GENERIC_CONSTRAINTS_PATH]
        assert _constraint(dst, 601) == {
            "id": 601,
            "name": "constraint_601",
            "expression": "hydro_generation(0) - @carga_ande",
            "slack": {"enabled": True, "penalty": 1000.0, "penalty_basis": "per_hour"},
        }
        assert pq.read_schema(dst / _GENERIC_BOUNDS_PATH).equals(GENERIC_BOUNDS_SCHEMA)
        assert _bound_rows(dst, 601) == [
            {
                "constraint_id": 601,
                "stage_id": stage,
                "block_id": None,
                "bound_lower": None,
                "bound_upper": _CAP_MW,
            }
            for stage in range(3)
        ]
        assert not any((dst / path).exists() for path in _ELECTRICAL_PATHS)
        parameters = _read_json(dst / _GENERIC_PARAMETERS_PATH)["scalar_parameters"]
        assert parameters == [
            *_HYDRO_PARAMETERS,
            {
                "id": len(_HYDRO_PARAMETERS),
                "name": "carga_ande",
                "kind": "per_stage_block",
                "block_values": _block_values(_ANDE_MW),
            },
        ]

    def test_electrical_parameters_follow_the_generic_constraint_parameters(
        self, tmp_path: Path
    ) -> None:
        dst = _run_libs_pipeline(
            tmp_path,
            libs_restricao_eletrica=_LIBS_CSV,
            libs_electrical_model=_alias_model(),
            to_generic=(_re_record(((0, 1.0), (1, 2.0))),),
        )

        parameters = _read_json(dst / _GENERIC_PARAMETERS_PATH)["scalar_parameters"]
        generic = build_decomp_scalar_parameters(
            [0, 1], per_stage={"coef_re901_t0": [1.0, 2.0, 2.0]}
        )["scalar_parameters"]
        assert parameters[:-1] == generic
        assert parameters[-1]["id"] == len(generic)
        assert parameters[-1]["name"] == "MM"

    def test_a_deck_name_a_generic_parameter_takes_is_renamed(
        self, tmp_path: Path
    ) -> None:
        dst = _run_libs_pipeline(
            tmp_path,
            libs_restricao_eletrica=_LIBS_CSV,
            libs_electrical_model=_alias_model("rho_eq_h0"),
        )

        assert _constraint(dst, 601)["expression"] == (
            "hydro_generation(0) - @rho_eq_h0_2"
        )
        parameters = _read_json(dst / _GENERIC_PARAMETERS_PATH)["scalar_parameters"]
        assert parameters[:-1] == _HYDRO_PARAMETERS
        assert parameters[-1]["name"] == "rho_eq_h0_2"

    def test_machine_sets_are_the_unit_groups_of_the_written_hydros(
        self, tmp_path: Path
    ) -> None:
        dst = _run_translated(tmp_path, _model("ger_conjh(66, 2)"))

        hydros = _read_json(dst / "system" / "hydros.json")["hydros"]
        [itaipu] = [hydro for hydro in hydros if hydro["id"] == 1]
        assert sorted(group["id"] for group in itaipu["unit_groups"]) == [0, 1]
        assert _constraint(dst, 601)["expression"] == (
            "hydro_generation(1, unit_group=1)"
        )

    def test_dry_run_reports_the_generic_files_without_writing_them(
        self, tmp_path: Path
    ) -> None:
        report_out: list[ConversionReport] = []
        diagnostics_out: list[dx.Diagnostic] = []
        dst = _run_translated(
            tmp_path,
            report_out=report_out,
            diagnostics_out=diagnostics_out,
            dry_run=True,
        )

        [report] = report_out
        would_write = {
            Path(written).relative_to(dst).as_posix()
            for written in report.would_write_paths
        }
        generic = {
            _GENERIC_CONSTRAINTS_PATH,
            _GENERIC_BOUNDS_PATH,
            _GENERIC_PARAMETERS_PATH,
        }
        assert generic <= would_write
        assert would_write.isdisjoint(_ELECTRICAL_PATHS)
        assert not any((dst / path).exists() for path in generic)
        assert _electrical_codes(diagnostics_out) == {_CENSUS_CODE}

    def test_no_electrical_cards_writes_no_file(self, tmp_path: Path) -> None:
        dst = _run_libs_pipeline(tmp_path)

        assert not any(
            (dst / path).exists()
            for path in (_GENERIC_CONSTRAINTS_PATH, _GENERIC_BOUNDS_PATH)
        )
        assert not any((dst / path).exists() for path in _ELECTRICAL_PATHS)

    def test_an_inequality_without_parameters_writes_activating_rows_only(
        self, tmp_path: Path
    ) -> None:
        model = LibsElectricalModel(
            expressions={},
            aliases={},
            restrictions={601: _cap_restriction(601, plant_code=1, cap=_CAP_MW)},
            rules={},
        )
        dst = _run_libs_pipeline(
            tmp_path, libs_restricao_eletrica=_LIBS_CSV, libs_electrical_model=model
        )

        assert _constraint(dst, 601)["expression"] == "hydro_generation(0) <= 500.0"
        assert _bound_rows(dst, 601) == [
            {
                "constraint_id": 601,
                "stage_id": stage,
                "block_id": None,
                "bound_lower": None,
                "bound_upper": None,
            }
            for stage in range(3)
        ]
        assert not (dst / _GENERIC_PARAMETERS_PATH).exists()

    def test_deck_constraints_follow_the_special_register_records(
        self, tmp_path: Path
    ) -> None:
        dst = _run_libs_pipeline(
            tmp_path,
            libs_restricao_eletrica=_LIBS_CSV,
            libs_electrical_model=_model("ger_usih(1)"),
            to_generic=(_re_record(),),
        )

        doc = _read_json(dst / _GENERIC_CONSTRAINTS_PATH)
        assert [c["name"] for c in doc["constraints"]] == ["RE_901", "constraint_601"]
        assert "expressions" not in doc
        bounds = pq.read_table(dst / _GENERIC_BOUNDS_PATH)
        [re_901] = [c["id"] for c in doc["constraints"] if c["name"] == "RE_901"]
        assert set(bounds.column("constraint_id").to_pylist()) == {re_901, 601}

    def test_the_census_counts_the_translated_records(self, tmp_path: Path) -> None:
        diagnostics_out: list[dx.Diagnostic] = []
        _run_translated(tmp_path, diagnostics_out=diagnostics_out)

        [census] = [d for d in diagnostics_out if d.code == _CENSUS_CODE]
        assert census.severity is dx.Severity.INFO
        assert census.category == "Special constraints"
        assert census.title == "LIBS electrical constraints translated (1)"
        assert census.table is not None
        assert census.table.columns == ["Record", "Count"]
        assert census.table.rows == [
            ["Constraints", 1],
            ["Expressions", 0],
            ["Parameters", 1],
            ["Bounds rows", 3],
            ["Constraints with a condition", 0],
        ]
        assert _electrical_codes(diagnostics_out) == {_CENSUS_CODE}

    def test_a_translation_error_stops_the_conversion(self, tmp_path: Path) -> None:
        with pytest.raises(
            UnsupportedSourceError,
            match=r"^lib_restricao-eletrica-especial\.csv: restriction 601: "
            ".*hydro code 99 is not in the converted case$",
        ):
            _run_translated(tmp_path, _model("ger_usih(99)"))

    def test_a_restriction_without_a_cost_needs_a_deficit_cost(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(
            FieldParseError,
            match=r"^lib_restricao-eletrica-especial\.csv: restriction 601: "
            "no violation cost",
        ):
            _run_translated(tmp_path, _model(violations=()))


_LOAD_STATS_TARGET = "novomodelo_bridge.decomp.pipeline.load_conv.convert_load_stats"


class TestDataTerms:
    def test_ande_and_pq_values_are_parameters_and_the_ande_stays_on_the_itaipu_bus(
        self, tmp_path: Path
    ) -> None:
        mocks: dict[str, MagicMock] = {}
        dst = _run_libs_pipeline(
            tmp_path,
            libs_restricao_eletrica=_LIBS_CSV,
            libs_electrical_model=_model(
                "ger_usih(1) - carga_ande - peq_SECO_PCHgd_SE"
            ),
            second_plant_code=_ITAIPU_CODE,
            pq_frame=_pq_frame(),
            ri_frame=_ri_frame(),
            mocks_out=mocks,
        )

        parameters = _read_json(dst / _GENERIC_PARAMETERS_PATH)["scalar_parameters"]
        assert [
            (p["name"], p["block_values"]) for p in parameters[len(_HYDRO_PARAMETERS) :]
        ] == [
            ("carga_ande", _block_values(_ANDE_MW)),
            ("peq_SECO_PCHgd_SE", _block_values(_PQ_MW)),
        ]
        extra_bus_loads = mocks[_LOAD_STATS_TARGET].call_args.kwargs["extra_bus_loads"]
        assert extra_bus_loads == {(0, stage): list(_ANDE_MW) for stage in range(3)}

    def test_a_deck_without_ri_and_pq_writes_only_the_declared_alias(
        self, tmp_path: Path
    ) -> None:
        mocks: dict[str, MagicMock] = {}
        dst = _run_libs_pipeline(
            tmp_path,
            libs_restricao_eletrica=_LIBS_CSV,
            libs_electrical_model=_alias_model(),
            second_plant_code=_ITAIPU_CODE,
            mocks_out=mocks,
        )

        parameters = _read_json(dst / _GENERIC_PARAMETERS_PATH)["scalar_parameters"]
        assert [p["name"] for p in parameters[len(_HYDRO_PARAMETERS) :]] == ["MM"]
        assert mocks[_LOAD_STATS_TARGET].call_args.kwargs["extra_bus_loads"] is None


class TestElectricalCensusDiagnostic:
    def test_counts_only_the_top_level_records_of_each_list(self) -> None:
        outputs = GenericOutputs(
            constraints=[
                {"id": 1, "active_when": "a", "slack": {"enabled": False}},
                {"id": 2, "slack": {"enabled": False}},
                {"id": 3, "active_when": "b", "slack": {"enabled": False}},
            ],
            expressions=[{"name": "A"}, {"name": "B"}],
            bounds=pa.Table.from_pylist(
                [
                    {"constraint_id": 1, "stage_id": stage, "block_id": None}
                    for stage in range(5)
                ],
                schema=GENERIC_BOUNDS_SCHEMA,
            ),
            parameters=[{"name": "C"}, {"name": "D"}, {"name": "E"}, {"name": "F"}],
        )

        diagnostic = _electrical_census_diagnostic(outputs)

        assert diagnostic.table is not None
        assert diagnostic.table.rows == [
            ["Constraints", 3],
            ["Expressions", 2],
            ["Parameters", 4],
            ["Bounds rows", 5],
            ["Constraints with a condition", 2],
        ]
        assert diagnostic.title.endswith("(3)")

    def test_summary_says_the_cards_were_translated_into_the_generic_files(
        self,
    ) -> None:
        diagnostic = _electrical_census_diagnostic(
            GenericOutputs(
                constraints=[],
                expressions=[],
                bounds=GENERIC_BOUNDS_SCHEMA.empty_table(),
                parameters=[],
            )
        )

        assert (
            "translated the LIBS electrical file to novomodelo's vocabulary and ids"
            in diagnostic.summary
        )
        for path in (
            _GENERIC_CONSTRAINTS_PATH,
            _GENERIC_BOUNDS_PATH,
            _GENERIC_PARAMETERS_PATH,
        ):
            assert path in diagnostic.summary
