"""Tests for the DECOMP FPHA converter (``decomp/fpha.py``).

Tier-1: synthetic ``EffectiveCadastro`` / ``DecompIdMap`` / polinjus doubles,
no real deck. Covers the fitting window (a novomodelo-bridge parameter), eligibility,
turbine efficiency, the VHA geometry table, and the tailrace-curves wrapper.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.diagnostics import Severity
from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro, MachineSet
from novomodelo_bridge.decomp.converters.fpha import (
    FPHA_VOLUME_WINDOW_FRACTION,
    convert_hydro_geometry,
    convert_tailrace_curves,
    fitting_window,
    fpha_eligible_codes,
    is_fpha_eligible,
    turbine_efficiency,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap
from tests.conftest import make_decomp_case


def _plant_row(
    *,
    name: str = "PLANT",
    vmin: float = 100.0,
    vmax: float = 500.0,
    volume_cota: tuple[float, ...] = (200.0, 0.2, 0.0, 0.0, 0.0),
    cota_area: tuple[float, ...] = (1.0, 0.01, 0.0, 0.0, 0.0),
    rho_esp: float = 0.009,
    tipo_regulacao: str = "M",
    n_conjuntos: int = 1,
    maquinas: int = 2,
    potencia: float = 50.0,
    vazao: float = 100.0,
) -> dict:
    row: dict = {
        "nome_usina": name,
        "submercado": 1,
        "volume_minimo": vmin,
        "volume_maximo": vmax,
        "produtibilidade_especifica": rho_esp,
        "canal_fuga_medio": 20.0,
        "tipo_perda": 0,
        "perdas": 0.0,
        "tipo_regulacao": tipo_regulacao,
        "numero_conjuntos_maquinas": n_conjuntos,
    }
    for i, a in enumerate(volume_cota):
        row[f"a{i}_volume_cota"] = a
    for i, a in enumerate(cota_area):
        row[f"a{i}_cota_area"] = a
    # Machine set columns for up to 5 conjuntos
    for i in range(1, 6):
        if i <= n_conjuntos:
            row[f"maquinas_conjunto_{i}"] = maquinas
            row[f"potencia_nominal_conjunto_{i}"] = potencia
            row[f"vazao_nominal_conjunto_{i}"] = vazao
        else:
            row[f"maquinas_conjunto_{i}"] = 0
            row[f"potencia_nominal_conjunto_{i}"] = 0.0
            row[f"vazao_nominal_conjunto_{i}"] = 0.0
    return row


def _effective(rows: dict[int, dict]) -> EffectiveCadastro:
    hidr = pd.DataFrame(rows).T
    hidr.index.name = "codigo_usina"
    return EffectiveCadastro(base=hidr, n_stages=1, stage_varying={})


def _id_map(codes: tuple[int, ...]) -> DecompIdMap:
    return DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=codes)


# ---------------------------------------------------------------------------
# fitting_window
# ---------------------------------------------------------------------------


def test_fitting_window_bands_each_side_of_initial() -> None:
    # useful = 400; ±10% each side = ±40 around V_init=300 -> [260, 340].
    assert FPHA_VOLUME_WINDOW_FRACTION == 0.10
    assert fitting_window(300.0, 100.0, 500.0) == {
        "volume_min_hm3": 260.0,
        "volume_max_hm3": 340.0,
    }


def test_fitting_window_clamps_to_reservoir_bounds() -> None:
    # V_init near vmax: upper side clamps to vmax.
    w = fitting_window(480.0, 100.0, 500.0)
    assert w["volume_min_hm3"] == 440.0  # 480 - 40
    assert w["volume_max_hm3"] == 500.0  # clamped


def test_fitting_window_collapses_for_run_of_river() -> None:
    # Zero useful volume -> single-volume window.
    assert fitting_window(250.0, 250.0, 250.0) == {
        "volume_min_hm3": 250.0,
        "volume_max_hm3": 250.0,
    }


# ---------------------------------------------------------------------------
# eligibility
# ---------------------------------------------------------------------------


def test_is_fpha_eligible_true_for_valid_reservoir() -> None:
    eff = _effective({1: _plant_row()})
    assert is_fpha_eligible(eff, 1) is True


def test_is_fpha_eligible_false_for_degenerate_cota() -> None:
    eff = _effective({1: _plant_row(volume_cota=(0.0, 0.0, 0.0, 0.0, 0.0))})
    assert is_fpha_eligible(eff, 1) is False


def test_is_fpha_eligible_false_for_nonpositive_rho_esp() -> None:
    eff = _effective({1: _plant_row(rho_esp=0.0)})
    assert is_fpha_eligible(eff, 1) is False


def test_is_fpha_eligible_false_for_zero_generation_capacity() -> None:
    eff = _effective({1: _plant_row(potencia=0.0)})
    assert is_fpha_eligible(eff, 1) is False


def test_is_fpha_eligible_false_for_zero_turbined_capacity() -> None:
    eff = _effective({1: _plant_row(vazao=0.0)})
    assert is_fpha_eligible(eff, 1) is False


def _effective_with_machine_set(
    stages: tuple[MachineSet, ...],
) -> EffectiveCadastro:
    hidr = pd.DataFrame({1: _plant_row()}).T
    hidr.index.name = "codigo_usina"
    return EffectiveCadastro(
        base=hidr,
        n_stages=len(stages),
        stage_varying={},
        machine_sets={(1, 1): stages},
    )


def test_is_fpha_eligible_false_for_ac_potefe_zero() -> None:
    eff = _effective_with_machine_set((MachineSet(2, 0.0, 100.0),))
    assert is_fpha_eligible(eff, 1) is False


def test_is_fpha_eligible_false_for_ac_vazefe_zero() -> None:
    eff = _effective_with_machine_set((MachineSet(2, 50.0, 0.0),))
    assert is_fpha_eligible(eff, 1) is False


def test_is_fpha_eligible_true_when_capacity_positive_at_any_stage() -> None:
    # novomodelo fits against the max-over-stages envelope hydros.json declares.
    eff = _effective_with_machine_set(
        (MachineSet(0, 50.0, 100.0), MachineSet(2, 50.0, 100.0))
    )
    assert is_fpha_eligible(eff, 1) is True


def test_fpha_eligible_codes_reports_zero_capacity_plants() -> None:
    eff = _effective(
        {
            1: _plant_row(name="A"),
            2: _plant_row(name="B", maquinas=0),
            3: _plant_row(name="C", rho_esp=0.0, maquinas=0),
        }
    )
    with dx.collect() as collected:
        codes = fpha_eligible_codes(eff, _id_map((1, 2, 3)))

    assert codes == {1}
    [diagnostic] = [d for d in collected if d.code == "fpha-zero-capacity"]
    assert diagnostic.severity is Severity.INFO
    # Plant 3 already fails the curve inputs, so only plant 2 is reported.
    assert diagnostic.table is not None
    assert diagnostic.table.rows == [["B", 2, 0.0, 0.0]]


def test_fpha_eligible_codes_filters_operated() -> None:
    eff = _effective(
        {
            1: _plant_row(name="A"),
            2: _plant_row(name="B", rho_esp=0.0),  # ineligible
        }
    )
    assert fpha_eligible_codes(eff, _id_map((1, 2))) == {1}


# ---------------------------------------------------------------------------
# efficiency
# ---------------------------------------------------------------------------


def test_turbine_efficiency_is_rho_esp_over_k() -> None:
    eff = _effective({1: _plant_row(rho_esp=0.009)})
    # eta = 0.009 / 9.81e-3 ≈ 0.917
    assert abs(turbine_efficiency(eff, 1, "PLANT") - 0.009 / 9.81e-3) < 1e-9


# ---------------------------------------------------------------------------
# hydro geometry
# ---------------------------------------------------------------------------


def test_convert_hydro_geometry_samples_vha_curve() -> None:
    eff = _effective({1: _plant_row(vmin=100.0, vmax=500.0)})
    table = convert_hydro_geometry(eff, _id_map((1,)))
    df = table.to_pandas()
    assert set(df.columns) == {"hydro_id", "volume_hm3", "height_m", "area_km2"}
    assert len(df) == 100  # _GEOMETRY_N_POINTS
    assert df["volume_hm3"].min() == 100.0 and df["volume_hm3"].max() == 500.0
    # height = 200 + 0.2 * V (from volume_cota), monotone here
    assert abs(df.iloc[0]["height_m"] - (200.0 + 0.2 * 100.0)) < 1e-6
    assert (df["height_m"] >= 0).all() and (df["area_km2"] >= 0).all()


def test_convert_hydro_geometry_single_point_for_run_of_river() -> None:
    eff = _effective({1: _plant_row(vmin=250.0, vmax=250.0, tipo_regulacao="D")})
    df = convert_hydro_geometry(eff, _id_map((1,))).to_pandas()
    assert len(df) == 1
    assert df.iloc[0]["volume_hm3"] == 250.0


def test_convert_hydro_geometry_skips_degenerate_cota() -> None:
    eff = _effective({1: _plant_row(volume_cota=(0.0, 0.0, 0.0, 0.0, 0.0))})
    assert convert_hydro_geometry(eff, _id_map((1,))).num_rows == 0


# ---------------------------------------------------------------------------
# tailrace curves
# ---------------------------------------------------------------------------


class _FakePolinjus:
    def __init__(self, families: pd.DataFrame, segments: pd.DataFrame) -> None:
        self._families = families
        self._segments = segments

    def hidreletrica_curvajusante(self, df: bool = True) -> pd.DataFrame:  # noqa: ARG002
        return self._families

    def hidreletrica_curvajusante_polinomio_segmento(
        self,
        df: bool = True,  # noqa: ARG002
    ) -> pd.DataFrame:
        return self._segments


def test_convert_tailrace_curves_none_when_no_polinjus() -> None:
    case = make_decomp_case(Path("unused"), polinjus=None)
    assert convert_tailrace_curves(case, _id_map((1,))) is None


def test_convert_tailrace_curves_maps_families_to_hydro_ids() -> None:
    families = pd.DataFrame(
        {
            "codigo_usina": [1],
            "indice_familia": [1],
            "nivel_montante_referencia": [50.0],
        }
    )
    segments = pd.DataFrame(
        {
            "codigo_usina": [1],
            "indice_familia": [1],
            "indice_polinomio": [1],
            "limite_inferior_vazao_jusante": [0.0],
            "limite_superior_vazao_jusante": [500.0],
            "coeficiente_a0": [10.0],
            "coeficiente_a1": [0.01],
            "coeficiente_a2": [0.0],
            "coeficiente_a3": [0.0],
            "coeficiente_a4": [0.0],
        }
    )
    case = make_decomp_case(Path("unused"), polinjus=_FakePolinjus(families, segments))
    table = convert_tailrace_curves(case, _id_map((1,)))
    assert table is not None
    df = table.to_pandas()
    assert df.iloc[0]["hydro_id"] == 0  # code 1 -> dense id 0
    assert df.iloc[0]["outflow_max_m3s"] == 500.0
    assert df.iloc[0]["coefficient_0"] == 10.0
