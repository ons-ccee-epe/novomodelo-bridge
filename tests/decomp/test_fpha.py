"""Tests for the DECOMP FPHA converter (``decomp/fpha.py``).

Tier-1: synthetic ``EffectiveCadastro`` / ``DecompIdMap`` / polinjus doubles,
no real deck. Covers the volume window, the spillage fit range, eligibility,
turbine efficiency, the VHA geometry table, and the tailrace-curves wrapper.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.diagnostics import Severity
from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro, MachineSet
from novomodelo_bridge.decomp.converters.fpha import (
    VolumeWindowRule,
    convert_hydro_geometry,
    convert_tailrace_curves,
    fpha_config,
    fpha_eligible_codes,
    is_fpha_eligible,
    mlt_absent_diagnostic,
    spill_outside_tailrace,
    spillage_fit_max,
    turbine_efficiency,
    unconverted_fp_fields_diagnostic,
    volume_window_rule,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.temporal import OperativeStage
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
# volume window
# ---------------------------------------------------------------------------


def _staged_effective(
    rows: dict[int, dict], n_stages: int, stage_varying: dict | None = None
) -> EffectiveCadastro:
    hidr = pd.DataFrame(rows).T
    hidr.index.name = "codigo_usina"
    return EffectiveCadastro(
        base=hidr, n_stages=n_stages, stage_varying=stage_varying or {}
    )


def _fp(*records: tuple) -> pd.DataFrame:
    """``Dadger.fp(df=True)`` volume columns from ``(code, estagio, entry type,
    reduction, increase)`` tuples; ``None`` is a blank field."""
    return pd.DataFrame(
        [
            {
                "codigo_usina": code,
                "estagio": estagio,
                "tipo_entrada_janela_volume": entry,
                "limite_inferior_janela_volume": reduction,
                "limite_superior_janela_volume": increase,
            }
            for code, estagio, entry, reduction, increase in records
        ]
    )


def _windows(
    eff: EffectiveCadastro, rule: VolumeWindowRule, code: int = 1
) -> list[tuple[float, ...]]:
    return [
        tuple(rule.fitting_window(eff, code, t).values()) for t in range(eff.n_stages)
    ]


def test_fitting_window_defaults_to_the_weekly_and_monthly_bands() -> None:
    eff = _staged_effective({1: _plant_row(vmin=100.0, vmax=500.0)}, n_stages=2)
    rule = VolumeWindowRule.from_fp(None, {1: 25.0}, weekly_stages=1)
    assert _windows(eff, rule) == [(120.0, 280.0), (100.0, 400.0)]


def test_fitting_window_is_clamped_to_the_storage_range() -> None:
    eff = _staged_effective({1: _plant_row(vmin=100.0, vmax=500.0)}, n_stages=1)
    rule = VolumeWindowRule.from_fp(pd.DataFrame(), {1: 90.0}, weekly_stages=1)
    assert _windows(eff, rule) == [(380.0, 500.0)]


def test_fitting_window_takes_the_initial_percentage_of_each_stage_range() -> None:
    eff = _staged_effective(
        {1: _plant_row(vmin=100.0, vmax=500.0)},
        n_stages=2,
        stage_varying={(1, "volume_minimo"): (100.0, 300.0)},
    )
    rule = VolumeWindowRule.from_fp(None, {1: 50.0}, weekly_stages=2)
    assert _windows(eff, rule) == [(220.0, 380.0), (360.0, 440.0)]


def test_fitting_window_collapses_to_the_reference_volume_for_run_of_river() -> None:
    row = _plant_row(vmin=100.0, vmax=500.0, tipo_regulacao="D")
    row["volume_referencia"] = 250.0
    eff = _staged_effective({1: row}, n_stages=1)
    rule = VolumeWindowRule.from_fp(_fp((1, 1, 1, 50.0, 50.0)), {1: 80.0}, 1)
    assert _windows(eff, rule) == [(250.0, 250.0)]


def test_fp_record_applies_from_its_stage_on() -> None:
    eff = _staged_effective({1: _plant_row(vmin=100.0, vmax=500.0)}, n_stages=3)
    rule = VolumeWindowRule.from_fp(_fp((1, 2, 0, 5.0, 10.0)), {1: 50.0}, 3)
    assert _windows(eff, rule) == [(220.0, 380.0), (280.0, 340.0), (280.0, 340.0)]


def test_fp_deltas_resolve_field_by_field_plant_then_999_then_default() -> None:
    eff = _staged_effective(
        {1: _plant_row(vmin=100.0, vmax=500.0), 2: _plant_row(vmin=100.0, vmax=500.0)},
        n_stages=3,
    )
    fp = _fp(
        (999, 1, 0, 30.0, 10.0),
        (1, 1, 0, 5.0, None),
        (1, 3, 0, None, 15.0),
        (999, 2, 0, 40.0, 40.0),
    )
    rule = VolumeWindowRule.from_fp(fp, {1: 50.0, 2: 50.0}, weekly_stages=3)
    assert _windows(eff, rule, 1) == [(280.0, 340.0), (280.0, 460.0), (280.0, 360.0)]
    assert _windows(eff, rule, 2) == [(180.0, 340.0), (140.0, 460.0), (140.0, 460.0)]


def test_fp_entry_type_one_gives_the_deltas_in_hm3() -> None:
    eff = _staged_effective({1: _plant_row(vmin=100.0, vmax=500.0)}, n_stages=1)
    rule = VolumeWindowRule.from_fp(_fp((1, 1, 1, 50.0, 20.0)), {1: 50.0}, 1)
    assert _windows(eff, rule) == [(250.0, 320.0)]


def test_fp_blank_entry_type_reads_as_percent() -> None:
    eff = _staged_effective({1: _plant_row(vmin=100.0, vmax=500.0)}, n_stages=1)
    rule = VolumeWindowRule.from_fp(_fp((1, 1, None, 5.0, 5.0)), {1: 50.0}, 1)
    assert _windows(eff, rule) == [(280.0, 320.0)]


def test_fp_unknown_entry_type_is_refused() -> None:
    with pytest.raises(
        ValueError, match="plant 1 at stage 2: volume window entry type 2"
    ):
        VolumeWindowRule.from_fp(_fp((1, 2, 2, 5.0, 5.0)), {1: 50.0}, 1)


def test_fp_negative_delta_is_refused() -> None:
    with pytest.raises(ValueError, match="negative volume window delta"):
        VolumeWindowRule.from_fp(_fp((1, 1, 0, -5.0, 5.0)), {1: 50.0}, 1)


class _FpDadger:
    """Duck-types the two ``Dadger`` readers :func:`volume_window_rule` calls."""

    def __init__(self, fp: pd.DataFrame, initial_pct: dict[int, float]) -> None:
        self._fp = fp
        self._uh = pd.DataFrame(
            {
                "codigo_usina": list(initial_pct),
                "volume_inicial": list(initial_pct.values()),
            }
        )

    def fp(self, *, df: bool) -> pd.DataFrame:
        return self._fp

    def uh(self, *, df: bool) -> pd.DataFrame:
        return self._uh


def _calendar(*hours: float) -> list[OperativeStage]:
    start = date(2026, 3, 7)
    return [
        OperativeStage(
            index=i,
            start_date=start + timedelta(days=7 * i),
            end_date=start + timedelta(days=7 * (i + 1)),
            season_id=2,
            block_hours=(h,),
        )
        for i, h in enumerate(hours)
    ]


def test_volume_window_rule_reads_the_deck_fp_uh_and_weekly_stages() -> None:
    eff = _staged_effective({1: _plant_row(vmin=100.0, vmax=500.0)}, n_stages=3)
    dadger = _FpDadger(_fp((999, 3, 0, 10.0, None)), {1: 25.0})
    rule = volume_window_rule(dadger, _calendar(168.0, 168.0, 552.0))
    assert _windows(eff, rule) == [(120.0, 280.0), (120.0, 280.0), (160.0, 400.0)]


def _fit_fields(code: int, estagio: int, **overrides: float) -> dict:
    return {
        "codigo_usina": code,
        "estagio": estagio,
        "tipo_entrada_janela_turbinamento": 0,
        "limite_inferior_janela_turbinamento": 0.0,
        "limite_superior_janela_turbinamento": 100.0,
        "numero_pontos_turbinamento": 5,
        "numero_pontos_volume": 5,
    } | overrides


def test_unconverted_fp_fields_warn_only_when_off_the_default_fit() -> None:
    assert unconverted_fp_fields_diagnostic(None) is None
    assert unconverted_fp_fields_diagnostic(pd.DataFrame([_fit_fields(999, 1)])) is None
    fp = pd.DataFrame(
        [
            _fit_fields(999, 1, numero_pontos_volume=7),
            _fit_fields(6, 2, limite_superior_janela_turbinamento=80.0),
            _fit_fields(31, 1),
        ]
    )
    diagnostic = unconverted_fp_fields_diagnostic(fp)
    assert diagnostic is not None
    assert diagnostic.severity is Severity.WARNING
    assert "all plants from stage 1; plant 6 from stage 2." in diagnostic.summary
    assert "plant 31" not in diagnostic.summary


def test_unconverted_fp_fields_message_has_no_repo_internal_references() -> None:
    diagnostic = unconverted_fp_fields_diagnostic(
        pd.DataFrame([_fit_fields(6, 1, numero_pontos_turbinamento=9)])
    )
    assert diagnostic is not None
    message = f"{diagnostic.title} {diagnostic.summary} {diagnostic.remediation or ''}"
    for leak in ("docs/", "plans/", "~/git", ".py", "TRACKED", "_"):
        assert leak not in message


def test_hydro_geometry_spans_the_storage_envelope_of_every_stage() -> None:
    eff = _staged_effective(
        {1: _plant_row(vmin=100.0, vmax=500.0)},
        n_stages=3,
        stage_varying={(1, "volume_minimo"): (100.0, 100.0, 40.0)},
    )
    volumes = convert_hydro_geometry(eff, _id_map((1,))).column("volume_hm3")
    assert volumes[0].as_py() == 40.0 and volumes[-1].as_py() == 500.0


# ---------------------------------------------------------------------------
# spillage fit range
# ---------------------------------------------------------------------------


class _FakeMlt:
    """Duck-types ``idecomp.decomp.Mlt``: ``valores`` has a ``mes`` column and
    one column per posto, named by its number."""

    def __init__(self, means_by_posto: dict[int, list[float]]) -> None:
        self.valores = pd.DataFrame(
            {"mes": list(range(1, 13))}
            | {str(posto): means for posto, means in means_by_posto.items()}
        )


def _gauged_effective(
    posto: int, numpos: tuple[int, ...] | None = None
) -> EffectiveCadastro:
    row = _plant_row()
    row["posto"] = posto
    hidr = pd.DataFrame({1: row}).T
    hidr.index.name = "codigo_usina"
    return EffectiveCadastro(
        base=hidr,
        n_stages=2,
        stage_varying={},
        inflow_gauges={1: numpos} if numpos is not None else {},
    )


_MLT = _FakeMlt(
    {
        12: [100.0] * 11 + [700.0],
        40: [50.0] * 12,
        300: [0.0] * 12,
    }
)


def test_spillage_fit_max_is_twice_the_gauge_largest_monthly_mean() -> None:
    assert spillage_fit_max(_gauged_effective(12), _MLT, 1, 0) == 1400.0


def test_spillage_fit_max_follows_the_ac_numpos_gauge_of_the_stage() -> None:
    eff = _gauged_effective(12, numpos=(12, 40))
    assert spillage_fit_max(eff, _MLT, 1, 0) == 1400.0
    assert spillage_fit_max(eff, _MLT, 1, 1) == 100.0


def test_spillage_fit_max_none_on_the_artificial_gauge() -> None:
    assert (
        spillage_fit_max(_gauged_effective(12, numpos=(300, 300)), _MLT, 1, 0) is None
    )


def test_spillage_fit_max_none_without_mlt_or_for_an_unlisted_gauge() -> None:
    assert spillage_fit_max(_gauged_effective(12), None, 1, 0) is None
    assert spillage_fit_max(_gauged_effective(77), _MLT, 1, 0) is None


def test_fpha_config_carries_the_spillage_range_only_when_there_is_one() -> None:
    windows = VolumeWindowRule.from_fp(None, {1: 50.0}, weekly_stages=2)
    assert fpha_config(
        _gauged_effective(12), _MLT, 1, 0, windows, turbined_fit_max_m3s=250.0
    ) == {
        "source": "computed",
        "fitting_window": {"volume_min_hm3": 220.0, "volume_max_hm3": 380.0},
        "turbined_fit_max_m3s": 250.0,
        "spillage_fit_max_m3s": 1400.0,
    }
    assert "spillage_fit_max_m3s" not in fpha_config(
        _gauged_effective(12), None, 1, 0, windows, turbined_fit_max_m3s=250.0
    )


def test_mlt_absent_diagnostic_is_a_warning_naming_the_plant_count() -> None:
    diagnostic = mlt_absent_diagnostic(3)
    assert diagnostic.severity is Severity.WARNING
    assert "3 FPHA plant(s)" in diagnostic.summary
    assert "mlt.dat" in (diagnostic.remediation or "")


def test_mlt_absent_remediation_has_no_repo_internal_references() -> None:
    diagnostic = mlt_absent_diagnostic(3)
    message = f"{diagnostic.summary} {diagnostic.remediation}"
    for leak in ("docs/", "plans/", "~/git", ".py", "TRACKED", "_"):
        assert leak not in message


def _flagged_effective(
    flags: dict[int, int],
    stage_varying: dict[tuple[int, str], tuple[float, ...]] | None = None,
) -> EffectiveCadastro:
    hidr = pd.DataFrame(
        {
            code: {
                "nome_usina": f"PLANT {code} ",
                "influencia_vertimento_canal_fuga": flag,
            }
            for code, flag in flags.items()
        }
    ).T
    return EffectiveCadastro(base=hidr, n_stages=3, stage_varying=stage_varying or {})


_FPHA = {"model": "fpha"}
_OUTAGE = {"model": "constant_productivity"}


def test_spill_outside_tailrace_names_the_fpha_stages_of_each_flagged_plant() -> None:
    diagnostic = spill_outside_tailrace(
        _flagged_effective({1: 0, 2: 1}),
        {1: [_FPHA, _OUTAGE, _FPHA], 2: [_FPHA] * 3},
    )
    assert diagnostic is not None
    assert diagnostic.code == "fpha-spill-outside-tailrace"
    assert diagnostic.severity is Severity.INFO
    assert diagnostic.table is not None
    assert diagnostic.table.rows == [[1, "PLANT 1", "0, 2"]]


def test_spill_outside_tailrace_reads_the_flag_in_force_at_each_stage() -> None:
    effective = _flagged_effective(
        {1: 1}, {(1, "influencia_vertimento_canal_fuga"): (1.0, 0.0, 0.0)}
    )
    diagnostic = spill_outside_tailrace(effective, {1: [_FPHA] * 3})
    assert diagnostic is not None
    assert diagnostic.table is not None
    assert diagnostic.table.rows == [[1, "PLANT 1", "1-2"]]


def test_spill_outside_tailrace_none_when_every_spill_reaches_the_tailrace() -> None:
    assert spill_outside_tailrace(_flagged_effective({1: 1}), {1: [_FPHA] * 3}) is None


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
