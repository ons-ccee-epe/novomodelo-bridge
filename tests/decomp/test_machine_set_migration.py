"""Tests for migrating capacity/availability consumers onto the
per-stage-effective machine-set view.

The entity/mirror-group capacity converter (``convert_hydros``) and
the B8 per-group per-stage availability overlay
(``convert_hydro_group_availability``) used to read the machine-set
overrides through a date-blind ``_read_ac_machine_overrides``/
``_AcOverrides`` pair. This migration re-sources both consumers onto
``EffectiveCadastro``'s per-stage-effective ``machine_conjunto_count``/
``machine_set`` accessors, so a plant whose machine set changes
mid-horizon gets the correct per-stage capacity — a fixture no real deck
(rv3) exercises, since every machine-set override there resolves to stage 0.
Tier-1: pure Python, no ``import novomodelo``, no ``example/`` read.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro, MachineSet
from novomodelo_bridge.decomp.converters.hydro import (
    convert_hydro_group_availability,
    convert_hydros,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.temporal import OperativeStage, build_operative_calendar
from tests.conftest import make_decomp_case

_HYDRO_CODE = 7
_ID_MAP = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(_HYDRO_CODE,))


def _calendar() -> list[OperativeStage]:
    """Three stages (two weekly + the aggregated final month), mirroring
    ``test_decomp_hydro_thermal.py``'s own fixture — so ``manutencao_k``/
    ``fator_k`` index unambiguously as 1, 2, 3."""
    hours = [[15.0, 64.0, 89.0]] * 2 + [[63.0, 280.0, 401.0]]
    return build_operative_calendar(date(2026, 7, 18), hours)


def _plant_row() -> dict:
    """One reservoir plant, one conjunto: 4 machines x 50 m3/s x 25 MW —
    base (un-overridden) rated capacity 200 m3/s / 100 MW. Flat cota
    (``a1..a4`` zero) and ``volume_minimo == volume_maximo`` keep
    ``_equivalent_productivity_mw_per_m3s`` a trivial constant (rho_eq =
    0.009 x (100 - 20) = 0.72 MW per m3/s) so the hydraulic ceiling
    (0.72 x 200 = 144 MW) never binds below the 100 MW installed power —
    the fixtures below isolate the machine-set/MP/FD behaviour from B8's
    hydraulic-cap behaviour, which is already covered elsewhere
    (``test_decomp_availability_rule.py``).

    ``queda_nominal_conjunto_1`` is pinned to 80.0 m — exactly the plant's
    own operating head (``h_op = rho_eq / rho_esp = 0.72 / 0.009 = 80``) —
    so the affinity ratio is a no-op (``(h_op/h_nom)**k == 1``)
    and this fixture stays a pure machine-set/MP-FD regression, isolated
    from head-affinity effects (covered elsewhere,
    ``test_decomp_head_aware_max_turbined.py``). The head-corrected
    ``max_turbined_m3s`` is not, however, isolated from the *power* cap
    (``Σ n·p_nom / rho_eq``): with this plant's numbers that cap (see each
    test's own comment) binds below the affinity-neutral rated flow.
    """
    return {
        "nome_usina": "SYN",
        "submercado": 1,
        "codigo_usina_jusante": 0,
        "volume_minimo": 10.0,
        "volume_maximo": 10.0,
        "numero_conjuntos_maquinas": 1,
        "maquinas_conjunto_1": 4,
        "vazao_nominal_conjunto_1": 50.0,
        "potencia_nominal_conjunto_1": 25.0,
        "queda_nominal_conjunto_1": 80.0,
        "teif": 0.0,
        "ip": 0.0,
        "a0_volume_cota": 100.0,
        "a1_volume_cota": 0.0,
        "a2_volume_cota": 0.0,
        "a3_volume_cota": 0.0,
        "a4_volume_cota": 0.0,
        "canal_fuga_medio": 20.0,
        "produtibilidade_especifica": 0.009,
        "tipo_perda": 0,
        "perdas": 0.0,
    }


def _hidr_frame() -> pd.DataFrame:
    df = pd.DataFrame({_HYDRO_CODE: _plant_row()}).T
    df.index.name = "codigo_usina"
    return df


def _uh_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "codigo_usina": _HYDRO_CODE,
                "volume_inicial": 100.0,
                "vazao_defluente_minima": None,
                "volume_morto_inicial": None,
            }
        ]
    )


class _StubDadger:
    """Never carries an ``AC`` machine-configuration override: both
    consumers under test source the machine set from ``EffectiveCadastro``
    exclusively now, never from the deck — the whole point of the
    migration. ``mp``/``fd`` default to ``None`` (no register at all, B8's
    own ``1.0``-factor default); pass a frame to exercise the MP/FD path.
    """

    def __init__(
        self,
        uh: pd.DataFrame | None = None,
        mp: pd.DataFrame | None = None,
        fd: pd.DataFrame | None = None,
    ) -> None:
        self._uh, self._mp, self._fd = uh, mp, fd

    def uh(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._uh

    def mp(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._mp

    def fd(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._fd

    def ac(  # noqa: ARG002
        self,
        codigo_usina: int | None = None,
        modificacao: type | None = None,
        df: bool = False,
    ) -> pd.DataFrame:
        # No ACALTEFE (or any other AC) row on this synthetic double — the
        # tracked-gap warning is covered separately in
        # test_head_aware_max_turbined.py.
        return pd.DataFrame()


def _no_override_effective(hidr: pd.DataFrame, n_stages: int) -> EffectiveCadastro:
    """Constant machine set: no override at all, falls through to the
    ``hidr`` base at every stage — the byte-identical regression fixture."""
    return EffectiveCadastro(base=hidr, n_stages=n_stages, stage_varying={})


def _dropping_machine_set_effective(
    hidr: pd.DataFrame, n_stages: int
) -> EffectiveCadastro:
    """Conjunto 1's machine count is 4 for stages 0-1 (matching the
    ``hidr`` base) and drops to 2 at the final stage — the synthetic
    mid-horizon ``NUMMAQ`` shrink no rv3 deck exercises (every machine-set
    override there resolves to stage 0, per the rv3 probe)."""
    machine_sets = tuple(
        MachineSet(4, 25.0, 50.0) if stage < n_stages - 1 else MachineSet(2, 25.0, 50.0)
        for stage in range(n_stages)
    )
    return EffectiveCadastro(
        base=hidr,
        n_stages=n_stages,
        stage_varying={},
        machine_sets={(_HYDRO_CODE, 1): machine_sets},
    )


def test_constant_machine_set_matches_base_rated_sum() -> None:
    """No override at all: the entity envelope and its mirror group's
    ``max_generation_mw`` equal the base ``hidr`` rated sum (4 x 25 = 100
    MW) — byte-identical to the date-blind value.
    ``max_turbined_m3s`` is head-corrected: the affinity ratio
    is a no-op here (``queda_nominal_conjunto_1`` pinned to ``h_op``), but
    the installed-power cap ``Σ n·p_nom / rho_eq`` (= 100 / 0.72) still
    binds below the rated 4 x 50 = 200 m3/s."""
    hidr = _hidr_frame()
    calendar = _calendar()
    effective = _no_override_effective(hidr, len(calendar))

    case = make_decomp_case(
        Path("unused"), dadger=_StubDadger(uh=_uh_frame()), hidr=hidr, calendar=calendar
    )
    doc = convert_hydros(case, _ID_MAP, effective=effective)
    gen = doc["hydros"][0]["generation"]
    assert gen["max_turbined_m3s"] == pytest.approx(100.0 / 0.72)
    assert gen["max_generation_mw"] == pytest.approx(100.0)

    group = doc["hydros"][0]["unit_groups"][0]
    assert group["max_turbined_m3s"] == pytest.approx(gen["max_turbined_m3s"])
    assert group["max_generation_mw"] == pytest.approx(gen["max_generation_mw"])


def test_mid_horizon_nummaq_drop_entity_declares_pre_drop_envelope() -> None:
    """A machine-count drop at the final stage does not lower the declared
    entity/mirror-group envelope: both stay at the pre-drop max-over-stages
    value (``max_generation_mw`` 100 MW rated; ``max_turbined_m3s`` the
    power-cap-bound 100 / 0.72 m3/s, same as the constant-machine-set case
    above), not the reduced stage's (50 MW / 50 x 0.72 m3/s)."""
    hidr = _hidr_frame()
    calendar = _calendar()
    effective = _dropping_machine_set_effective(hidr, len(calendar))

    case = make_decomp_case(
        Path("unused"), dadger=_StubDadger(uh=_uh_frame()), hidr=hidr, calendar=calendar
    )
    doc = convert_hydros(case, _ID_MAP, effective=effective)
    gen = doc["hydros"][0]["generation"]
    assert gen["max_turbined_m3s"] == pytest.approx(100.0 / 0.72)
    assert gen["max_generation_mw"] == pytest.approx(100.0)

    group = doc["hydros"][0]["unit_groups"][0]
    assert group["max_turbined_m3s"] == pytest.approx(gen["max_turbined_m3s"])
    assert group["max_generation_mw"] == pytest.approx(gen["max_generation_mw"])


def test_mid_horizon_nummaq_drop_overlay_only_at_reduced_stage() -> None:
    """The machine-count drop's own overlay row: sparse, lands only at the
    reduced final stage, carrying both the lowered ``max_generation_mw``
    (50 MW, rated) and the head-corrected ``max_turbined_m3s`` overlay
    (``Σ n·p_nom / rho_eq`` = 50 / 0.72, power-cap-bound — not the rated
    2 x 50 = 100 m3/s); the unchanged stages 0-1 get no row at all (MP=FD=1,
    hydraulic non-binding at every stage)."""
    hidr = _hidr_frame()
    calendar = _calendar()
    effective = _dropping_machine_set_effective(hidr, len(calendar))

    case = make_decomp_case(
        Path("unused"), dadger=_StubDadger(), hidr=hidr, calendar=calendar
    )
    values = convert_hydro_group_availability(case, _ID_MAP, effective=effective)
    hydro_id = _ID_MAP.hydro_id(_HYDRO_CODE)

    assert (hydro_id, 0, 0) not in values
    assert (hydro_id, 0, 1) not in values

    entry = values[(hydro_id, 0, 2)]
    assert entry.max_generation_mw == pytest.approx(50.0)
    assert entry.max_turbined_m3s == pytest.approx(50.0 / 0.72)


def test_constant_machine_set_availability_matches_mp_fd_product() -> None:
    """Regression guard: with a constant (un-overridden) machine set, the
    B8 overlay still reproduces the plain ``installed x MP x FD`` formula,
    capped by the (here, non-binding) hydraulic ceiling — unchanged from
    the date-blind behaviour, now sourced through ``effective``.
    A stage whose MP/FD factors do not move installed capacity below the
    envelope (stage 0) gets no row; the flow envelope is untouched by
    MP/FD (only by the machine set), so ``max_turbined_m3s`` stays absent
    on every row here."""
    hidr = _hidr_frame()
    calendar = _calendar()
    effective = _no_override_effective(hidr, len(calendar))
    mp = pd.DataFrame(
        [
            {
                "codigo_usina": _HYDRO_CODE,
                "manutencao_1": 1.0,
                "manutencao_2": 0.5,
                "manutencao_3": 1.0,
            }
        ]
    )
    fd = pd.DataFrame(
        [
            {
                "codigo_usina": _HYDRO_CODE,
                "fator_1": 1.0,
                "fator_2": 1.0,
                "fator_3": 0.8,
            }
        ]
    )

    case = make_decomp_case(
        Path("unused"), dadger=_StubDadger(mp=mp, fd=fd), hidr=hidr, calendar=calendar
    )
    values = convert_hydro_group_availability(case, _ID_MAP, effective=effective)
    hydro_id = _ID_MAP.hydro_id(_HYDRO_CODE)

    assert (hydro_id, 0, 0) not in values

    stage1 = values[(hydro_id, 0, 1)]
    assert stage1.max_generation_mw == pytest.approx(50.0)
    assert stage1.max_turbined_m3s is None

    stage2 = values[(hydro_id, 0, 2)]
    assert stage2.max_generation_mw == pytest.approx(80.0)
    assert stage2.max_turbined_m3s is None
