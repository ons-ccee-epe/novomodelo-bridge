"""Tests for the topology/gauge ``AC`` override family.

Covers the plant-keyed, ``int``-valued mnemonics (``ACNUMJUS`` — the
water-routing downstream plant — ``ACNUMPOS`` — the inflow gauge —
``ACNPOSNW`` — the boundary-cut coupling gauge — and ``ACJUSENA`` — the
stored-energy downstream plant),
flowing through the shared
:func:`~novomodelo_bridge.decomp.converters.cadastro.overrides._read_keyed_overrides`
reader (with ``_plant_code_key``), and their effect on the re-sourced
cascade walk (``_downstream_operated``), the incremental-inflow cascade
(``_incremental_context``/``convert_external_inflows``), the entity
``downstream_id``, and the pipeline's ``cadastro-topology-relinked``
diagnostic. Tier-1 only: a synthetic ``_FakeDadger``/``_StubVazoes`` double
and a synthetic ``build_operative_calendar``, no real deck.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest
from idecomp.decomp.modelos.dadger import ACJUSENA, ACNPOSNW, ACNUMJUS, ACNUMPOS

from novomodelo_bridge.core.diagnostics import collect
from novomodelo_bridge.decomp.converters.cadastro import (
    EffectiveCadastro,
    build_effective_cadastro,
    topology_varies_diagnostic,
)
from novomodelo_bridge.decomp.converters.hydro import (
    convert_hydros,
    convert_initial_storage,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.pipeline import _topology_relink_diagnostic
from novomodelo_bridge.decomp.scenarios import (
    _incremental_context,
    convert_external_inflows,
)
from novomodelo_bridge.decomp.temporal import build_operative_calendar
from tests.conftest import make_decomp_case
from tests.decomp.conftest import flat_turbine_limits

if TYPE_CHECKING:
    from collections.abc import Mapping

    from novomodelo_bridge.core.diagnostics import Diagnostic
    from novomodelo_bridge.decomp.case import DecompCase
    from novomodelo_bridge.decomp.temporal import OperativeStage


class _FakeDadger:
    """Return a preset ``AC`` frame (or ``None``) dispatched by ``modificacao``
    class, plus a canned ``UH`` operated-plant frame.

    Mirrors ``test_cadastro_head.py``'s double of the same name —
    kept local rather than shared, per that file's own convention — extended
    with ``.uh(df=True)`` because the no-override case exercises
    ``convert_hydros`` directly.
    """

    def __init__(
        self,
        frames: dict[type, pd.DataFrame | None],
        uh: pd.DataFrame | None = None,
    ) -> None:
        self._frames = frames
        self._uh = uh

    def ac(
        self,
        codigo_usina: int | None = None,
        modificacao: type | None = None,
        df: bool = True,
    ) -> pd.DataFrame | None:
        if modificacao is None:
            return None
        return self._frames.get(modificacao)

    def uh(self, df: bool = True) -> pd.DataFrame | None:
        return self._uh


class _StubVazoes:
    """Trunk (2 rows) + terminal fan (1 scenario) for the 3-stage calendar
    below, keyed by ``posto`` station columns as strings.
    """

    def __init__(self, columns: Mapping[str, tuple[float, ...]]) -> None:
        self.previsoes = pd.DataFrame(
            [
                {"estagio": 1, **{k: v[0] for k, v in columns.items()}},
                {"estagio": 2, **{k: v[0] for k, v in columns.items()}},
            ]
        )
        self.cenarios_gerados = pd.DataFrame(
            [
                {
                    "estagio": 3,
                    "cenario": 1,
                    **{k: v[1] for k, v in columns.items()},
                }
            ]
        )


def _calendar() -> list[OperativeStage]:
    """Stages 0, 1 = July weekly stages; stage 2 = the August monthly stage —
    the same 3-stage pattern every sibling cadastro test file uses.
    """
    stage_block_hours = [[15.0, 64.0, 89.0]] * 2 + [[63.0, 280.0, 401.0]]
    return build_operative_calendar(date(2026, 7, 18), stage_block_hours)


def _plant_row(
    *,
    jusante: int,
    posto: int,
    name: str = "PLANT",
) -> dict:
    """One synthetic ``hidr`` row."""
    row: dict = {
        "nome_usina": name,
        "submercado": 1,
        "codigo_usina_jusante": jusante,
        "posto": posto,
        "desvio": 0,
        "volume_minimo": 10.0,
        "volume_maximo": 50.0,
        "numero_conjuntos_maquinas": 1,
        "maquinas_conjunto_1": 2,
        "vazao_nominal_conjunto_1": 50.0,
        "potencia_nominal_conjunto_1": 100.0,
        "teif": 0.0,
        "ip": 0.0,
        "canal_fuga_medio": 0.0,
        "produtibilidade_especifica": 0.0,
        "tipo_perda": 0,
        "perdas": 0.0,
    }
    for i, a in enumerate((100.0, 0.0, 0.0, 0.0, 0.0)):
        row[f"a{i}_volume_cota"] = a
    return row


def _hidr_frame(rows: dict[int, dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).T
    df.index.name = "codigo_usina"
    return df


def _uh_frame(codes: tuple[int, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "codigo_usina": code,
                "volume_inicial": 50.0,
                "vazao_defluente_minima": None,
                "volume_morto_inicial": None,
            }
            for code in codes
        ]
    )


# The shared A -> B -> C cascade (base topology) exercised across the topology tests:
# A=1 (posto 11), B=2 (posto 12), C=3 (posto 13); A's base downstream is B,
# B's base downstream is C, C is the sink.
def _abc_hidr() -> pd.DataFrame:
    return _hidr_frame(
        {
            1: _plant_row(jusante=2, posto=11, name="A"),
            2: _plant_row(jusante=3, posto=12, name="B"),
            3: _plant_row(jusante=0, posto=13, name="C"),
        }
    )


_ABC_ID_MAP = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1, 2, 3))


def _numjus_override(
    code: int, downstream: int, *, mes: str = "", semana: float | None = None
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "codigo_usina": code,
                "codigo_usina_jusante": downstream,
                "mes": mes,
                "semana": float("nan") if semana is None else semana,
                "ano": float("nan") if mes == "" else 2026.0,
            }
        ]
    )


# ---------------------------------------------------------------------------
# Ingestion + accessors (ACNUMJUS relinks A's downstream, B falls through).
# ---------------------------------------------------------------------------


def test_numjus_override_relinks_downstream() -> None:
    hidr = _abc_hidr()
    dadger = _FakeDadger({ACNUMJUS: _numjus_override(1, 3)})
    effective, report = build_effective_cadastro(dadger, hidr, _calendar())

    assert [effective.downstream_plant(1, s) for s in range(3)] == [3, 3, 3]
    base_b_downstream = int(hidr.loc[2, "codigo_usina_jusante"])
    assert all(effective.downstream_plant(2, s) == base_b_downstream for s in range(3))
    assert report.applied["codigo_usina_jusante"] == 1


def test_jusena_override_sets_the_energy_link_only() -> None:
    hidr = _abc_hidr()
    dadger = _FakeDadger({ACJUSENA: _numjus_override(1, 3)})
    effective, report = build_effective_cadastro(dadger, hidr, _calendar())

    assert [effective.energy_link(1, s) for s in range(3)] == [3, 3, 3]
    assert effective.energy_link(2, 0) is None
    assert effective.downstream_plant(1, 0) == 2
    assert topology_varies_diagnostic(effective) is None
    assert report.applied["jusante_energia"] == 1


def test_temporal_jusena_override_varies() -> None:
    hidr = _abc_hidr()
    dadger = _FakeDadger({ACJUSENA: _numjus_override(1, 3, mes="AGO", semana=1.0)})
    effective, _ = build_effective_cadastro(dadger, hidr, _calendar())

    assert [effective.energy_link(1, s) for s in range(3)] == [None, None, 3]


# ---------------------------------------------------------------------------
# The incremental inflow values are invariant to a NUMJUS relink.
# ---------------------------------------------------------------------------


def test_relink_does_not_shift_incremental_inflows() -> None:
    """DECOMP's inflow file is already incremental per gauge column, so an
    ``AC NUMJUS`` relink changes only the routing topology (a plant's
    ``downstream_id``), never the incremental inflow *values*,
    which are read straight from each plant's own gauge column.
    """
    hidr = _abc_hidr()
    dadger = _FakeDadger({ACNUMJUS: _numjus_override(1, 3)})
    effective, _ = build_effective_cadastro(dadger, hidr, _calendar())

    vazoes = _StubVazoes(
        {"11": (100.0, 110.0), "12": (200.0, 210.0), "13": (300.0, 310.0)}
    )
    case = make_decomp_case(Path("unused"), calendar=_calendar())
    table = convert_external_inflows(
        case, _ABC_ID_MAP, vazoes=vazoes, effective=effective
    )
    trunk = table.to_pandas()
    trunk = trunk[trunk["stage_id"] == 0].set_index("hydro_id")["value_m3s"]

    # Each plant reads its own gauge column directly, regardless of the relink.
    assert trunk.loc[0] == pytest.approx(100.0)  # A, gauge 11
    assert trunk.loc[1] == pytest.approx(200.0)  # B, gauge 12
    assert trunk.loc[2] == pytest.approx(300.0)  # C, gauge 13


# ---------------------------------------------------------------------------
# No topology/gauge override -> byte-identical to the base-topology
# result, for both the incremental table and every downstream_id.
# ---------------------------------------------------------------------------


def test_no_topology_override_is_identical() -> None:
    hidr = _hidr_frame(
        {
            1: _plant_row(jusante=2, posto=11, name="A"),
            2: _plant_row(jusante=0, posto=12, name="B"),
        }
    )
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1, 2))
    dadger = _FakeDadger({}, uh=_uh_frame((1, 2)))
    calendar = _calendar()
    effective, _ = build_effective_cadastro(dadger, hidr, calendar)
    base_effective = EffectiveCadastro(
        base=hidr, n_stages=len(calendar), stage_varying={}
    )

    vazoes = _StubVazoes({"11": (100.0, 110.0), "12": (200.0, 210.0)})
    case = make_decomp_case(Path("unused"), dadger=dadger, hidr=hidr, calendar=calendar)
    via_effective = convert_external_inflows(
        case, id_map, vazoes=vazoes, effective=effective
    )
    via_base = convert_external_inflows(
        case, id_map, vazoes=vazoes, effective=base_effective
    )
    assert via_effective.equals(via_base)

    limits = flat_turbine_limits(id_map, len(calendar))
    hydros_effective = convert_hydros(
        case, id_map, effective=effective, turbine_limits=limits
    )["hydros"]
    hydros_base = convert_hydros(
        case, id_map, effective=base_effective, turbine_limits=limits
    )["hydros"]
    assert [h["downstream_id"] for h in hydros_effective] == [
        h["downstream_id"] for h in hydros_base
    ]


# ---------------------------------------------------------------------------
# An ACNUMPOS override re-attributes a plant's inflow to another column.
# ---------------------------------------------------------------------------


def test_numpos_override_changes_station() -> None:
    hidr = _hidr_frame({1: _plant_row(jusante=0, posto=11, name="A")})
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))
    numpos = pd.DataFrame(
        [
            {
                "codigo_usina": 1,
                "codigo_posto": 99,
                "mes": "",
                "semana": float("nan"),
                "ano": float("nan"),
            }
        ]
    )
    dadger = _FakeDadger({ACNUMPOS: numpos})
    calendar = _calendar()
    effective, report = build_effective_cadastro(dadger, hidr, calendar)
    assert report.applied["posto"] == 1

    # Column "11" (the base gauge) carries a value that must NOT be read;
    # column "99" (the overridden gauge) carries the value that must be.
    vazoes = _StubVazoes({"11": (500.0, 500.0), "99": (77.0, 88.0)})
    case = make_decomp_case(Path("unused"), calendar=calendar)
    table = convert_external_inflows(
        case, id_map, vazoes=vazoes, effective=effective
    ).to_pandas()
    trunk = table[table["stage_id"] == 0]["value_m3s"].iloc[0]
    assert trunk == pytest.approx(77.0)


def test_nposnw_override_sets_the_coupling_gauge_only() -> None:
    """The last stage's ``ACNPOSNW`` is the coupling gauge; the operation
    gauge and a plant without the record keep the ``hidr.dat`` posto."""
    nposnw = pd.DataFrame(
        [
            {
                "codigo_usina": 1,
                "codigo_posto": 99,
                "mes": "AGO",
                "semana": 1.0,
                "ano": 2026.0,
            }
        ]
    )
    effective, report = build_effective_cadastro(
        _FakeDadger({ACNPOSNW: nposnw}), _abc_hidr(), _calendar()
    )

    assert report.applied["posto_acoplamento"] == 1
    assert effective.coupling_gauge(1) == 99
    assert effective.inflow_gauge(1, 2) == 11
    assert topology_varies_diagnostic(effective) is None
    assert effective.coupling_gauge(2) == 12


# ---------------------------------------------------------------------------
# The pipeline's relink diagnostic.
# ---------------------------------------------------------------------------


def test_relink_diagnostic_lists_operated_plants() -> None:
    hidr = _abc_hidr()
    dadger = _FakeDadger({ACNUMJUS: _numjus_override(1, 3)})
    effective, _ = build_effective_cadastro(dadger, hidr, _calendar())

    diagnostic = _topology_relink_diagnostic(effective, _ABC_ID_MAP)
    assert diagnostic is not None
    assert diagnostic.code == "cadastro-topology-relinked"
    assert diagnostic.table is not None
    rows = {(row[0], row[2]): (row[3], row[4]) for row in diagnostic.table.rows}
    assert rows[(1, "downstream plant")] == (2, 3)

    # The no-override deck must not produce a diagnostic.
    no_override_hidr = _hidr_frame(
        {
            1: _plant_row(jusante=2, posto=11, name="A"),
            2: _plant_row(jusante=0, posto=12, name="B"),
        }
    )
    no_override_id_map = DecompIdMap(
        bus_codes=(1,), bus_names=("SE",), hydro_codes=(1, 2)
    )
    no_override_effective, _ = build_effective_cadastro(
        _FakeDadger({}), no_override_hidr, _calendar()
    )
    assert (
        _topology_relink_diagnostic(no_override_effective, no_override_id_map) is None
    )


def test_relink_through_a_plant_left_out_is_itemized() -> None:
    """An ``AC NUMJUS`` on a plant outside the configuration changes the next
    operated plant below an operated one whose own link is unchanged."""
    hidr = _hidr_frame(
        {
            1: _plant_row(jusante=4, posto=11, name="A"),
            2: _plant_row(jusante=3, posto=12, name="B"),
            3: _plant_row(jusante=0, posto=13, name="C"),
            4: _plant_row(jusante=2, posto=14, name="D"),
        }
    )
    dadger = _FakeDadger({ACNUMJUS: _numjus_override(4, 3)})
    effective, _ = build_effective_cadastro(dadger, hidr, _calendar())

    diagnostic = _topology_relink_diagnostic(effective, _ABC_ID_MAP)

    assert diagnostic is not None
    assert diagnostic.table is not None
    assert diagnostic.table.rows == [[1, "A", "next operated plant", 2, 3]]
    assert diagnostic.title.endswith("(1)")


# ---------------------------------------------------------------------------
# A cascade register that changes between stages is reported; every walk uses
# the stage-0 effective value.
# ---------------------------------------------------------------------------


def test_temporal_topology_is_reported_and_uses_stage0() -> None:
    hidr = _abc_hidr()
    # Dated AGO (the 3-stage calendar's index-2 stage): stages 0/1 still read
    # the base downstream (B); stage 2 onward reads the override (C).
    dadger = _FakeDadger({ACNUMJUS: _numjus_override(1, 3, mes="AGO", semana=1.0)})
    effective, _ = build_effective_cadastro(dadger, hidr, _calendar())

    assert effective.downstream_plant(1, 0) == 2
    assert effective.downstream_plant(1, 1) == 2
    assert effective.downstream_plant(1, 2) == 3

    _, parents = _incremental_context(effective, _ABC_ID_MAP)
    diagnostic = topology_varies_diagnostic(effective)

    assert diagnostic is not None
    assert diagnostic.code == "decomp-ac-topology-varies"
    assert diagnostic.table is not None
    assert diagnostic.table.rows == [[1, "A", "AC NUMJUS", 2, "2"]]
    # The stage-0 (base) link is used for the whole horizon: A still feeds
    # B, not C, despite the override.
    assert parents[2] == [1]
    assert parents[3] == [2]


def test_temporal_gauge_is_reported_and_uses_stage0() -> None:
    """A temporal ``ACNUMPOS`` is reported the same way, and the stage-0
    gauge is used for the whole horizon.
    """
    hidr = _abc_hidr()
    numpos = pd.DataFrame(
        [
            {
                "codigo_usina": 1,
                "codigo_posto": 99,
                "mes": "AGO",
                "semana": 1.0,
                "ano": 2026.0,
            }
        ]
    )
    dadger = _FakeDadger({ACNUMPOS: numpos})
    effective, _ = build_effective_cadastro(dadger, hidr, _calendar())

    assert effective.inflow_gauge(1, 0) == 11
    assert effective.inflow_gauge(1, 2) == 99

    station_by_code, _ = _incremental_context(effective, _ABC_ID_MAP)
    diagnostic = topology_varies_diagnostic(effective)

    assert station_by_code[1] == "11"
    assert diagnostic is not None
    assert diagnostic.table is not None
    assert diagnostic.table.rows == [[1, "A", "AC NUMPOS", 11, "2"]]


def test_temporal_jusena_is_reported_with_no_first_stage_link() -> None:
    dadger = _FakeDadger({ACJUSENA: _numjus_override(1, 3, mes="AGO", semana=1.0)})
    effective, _ = build_effective_cadastro(dadger, _abc_hidr(), _calendar())

    diagnostic = topology_varies_diagnostic(effective)

    assert diagnostic is not None
    assert diagnostic.table is not None
    assert diagnostic.table.rows == [[1, "A", "AC JUSENA", "none", "2"]]


def test_a_register_constant_over_the_horizon_is_not_reported() -> None:
    dadger = _FakeDadger({ACNUMJUS: _numjus_override(1, 3)})
    effective, _ = build_effective_cadastro(dadger, _abc_hidr(), _calendar())

    assert topology_varies_diagnostic(effective) is None


# ---------------------------------------------------------------------------
# Findings convert_hydros and convert_initial_storage report once per kind.
# ---------------------------------------------------------------------------


def _two_plant_case(
    hidr: pd.DataFrame, uh: pd.DataFrame | None = None
) -> tuple[DecompCase, DecompIdMap, EffectiveCadastro]:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1, 2))
    dadger = _FakeDadger({}, uh=_uh_frame((1, 2)) if uh is None else uh)
    calendar = _calendar()
    effective, _ = build_effective_cadastro(dadger, hidr, calendar)
    case = make_decomp_case(Path("unused"), dadger=dadger, hidr=hidr, calendar=calendar)
    return case, id_map, effective


def _hydro_findings(hidr: pd.DataFrame) -> dict[str, Diagnostic]:
    case, id_map, effective = _two_plant_case(hidr)
    with collect() as found:
        convert_hydros(
            case,
            id_map,
            effective=effective,
            turbine_limits=flat_turbine_limits(id_map, len(_calendar())),
        )
    return {diagnostic.code: diagnostic for diagnostic in found}


def test_a_link_to_a_plant_missing_from_hidr_is_reported_once() -> None:
    hidr = _hidr_frame(
        {
            1: _plant_row(jusante=9, posto=11, name="A"),
            2: _plant_row(jusante=0, posto=12, name="B"),
        }
    )

    found = _hydro_findings(hidr)

    diagnostic = found["decomp-cascade-link-invalid"]
    assert diagnostic.table is not None
    assert diagnostic.table.rows == [[1, "A", 9]]


def test_a_plant_without_a_forebay_curve_is_reported_with_its_stages() -> None:
    flat = _plant_row(jusante=0, posto=12, name="B")
    for i in range(5):
        flat[f"a{i}_volume_cota"] = 0.0
    hidr = _hidr_frame({1: _plant_row(jusante=2, posto=11, name="A"), 2: flat})

    found = _hydro_findings(hidr)

    diagnostic = found["hydro-forebay-curve-missing"]
    assert diagnostic.table is not None
    assert diagnostic.table.rows == [[2, "B", "0-2"]]


def test_a_declared_initial_dead_volume_is_reported() -> None:
    uh = _uh_frame((1, 2))
    uh.loc[uh["codigo_usina"] == 2, "volume_morto_inicial"] = 4.5
    case, id_map, effective = _two_plant_case(_abc_hidr().loc[[1, 2]], uh)

    with collect() as found:
        convert_initial_storage(case, id_map, effective=effective)

    [diagnostic] = found
    assert diagnostic.code == "decomp-dead-volume-filling-unconverted"
    assert diagnostic.table is not None
    assert diagnostic.table.rows == [[2, "B", 4.5]]


def test_plants_on_the_artificial_gauge_are_reported() -> None:
    hidr = _hidr_frame(
        {
            1: _plant_row(jusante=3, posto=300, name="A"),
            2: _plant_row(jusante=3, posto=300, name="B"),
            3: _plant_row(jusante=0, posto=13, name="C"),
        }
    )
    effective, _ = build_effective_cadastro(_FakeDadger({}), hidr, _calendar())
    vazoes = _StubVazoes({"300": (0.0, 0.0), "13": (300.0, 310.0)})
    case = make_decomp_case(Path("unused"), calendar=_calendar())
    with collect() as diagnostics:
        convert_external_inflows(case, _ABC_ID_MAP, vazoes=vazoes, effective=effective)
    reported = [d for d in diagnostics if d.code == "artificial-gauge-plants"]
    assert len(reported) == 1
    assert reported[0].table is not None
    assert reported[0].table.rows == [[1, "A"], [2, "B"]]
