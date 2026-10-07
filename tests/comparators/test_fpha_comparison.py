"""Tests for the production-function (FPHA) comparison data layer.

Covers the two source-model readers (``read_fpha_planes`` / ``read_fpha_grid``),
the Novomodelo reader (``read_novomodelo_fpha_planes``), the envelope evaluator
(``fpha.dense_grid``), and the comparison builder (``build_fpha_comparison``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from novomodelo_bridge.comparators.analyze import build_fpha_comparison
from novomodelo_bridge.comparators.fpha import dense_grid
from novomodelo_bridge.comparators.newave.alignment import HydroEntity
from novomodelo_bridge.comparators.newave.readers import (
    read_fpha_grid,
    read_fpha_planes,
)
from novomodelo_bridge.novomodelo.readers import read_novomodelo_fpha_planes

# --------------------------------------------------------------------------- #
# Readers — absence gate                                                       #
# --------------------------------------------------------------------------- #


def test_read_fpha_planes_returns_none_when_absent(tmp_path: Path) -> None:
    assert read_fpha_planes(tmp_path) is None


def test_read_fpha_grid_returns_none_when_absent(tmp_path: Path) -> None:
    assert read_fpha_grid(tmp_path) is None


def test_read_novomodelo_fpha_planes_returns_none_when_absent(tmp_path: Path) -> None:
    assert read_novomodelo_fpha_planes(tmp_path) is None


# --------------------------------------------------------------------------- #
# Readers — happy path (inewave patched, so no fixture file format needed)     #
# --------------------------------------------------------------------------- #


def _patch_inewave(
    monkeypatch: pytest.MonkeyPatch, attr: str, table: pd.DataFrame
) -> None:
    """Patch ``inewave.newave.<attr>`` so ``.read(path).tabela`` yields ``table``."""
    import inewave.newave

    class _Reader:
        tabela = table

    class _Fake:
        @staticmethod
        def read(_path: str) -> _Reader:
            return _Reader()

    monkeypatch.setattr(inewave.newave, attr, _Fake)


def test_read_fpha_planes_renames_and_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "fpha").mkdir()
    (tmp_path / "fpha" / "fpha_cortes.csv").write_text("placeholder", encoding="utf-8")
    table = pd.DataFrame(
        {
            "codigo_usina": [4, 4],
            "periodo": [9, 9],
            "indice_corte": [1, 2],
            "fator_correcao": [1.0, 0.99],
            "rhs_energia": [0.0, 2.5],
            "coeficiente_volume_util_MW_hm3": [0.0, 0.01],
            "coeficiente_vazao_turbinada_MW_m3s": [0.36, 0.34],
            "coeficiente_vazao_vertida_MW_m3s": [-0.003, -0.009],
            "coeficiente_vazao_lateral_MW_m3s": [0.0, 0.0],
        }
    )
    _patch_inewave(monkeypatch, "FphaCortes", table)

    out = read_fpha_planes(tmp_path)
    assert out is not None
    assert out.columns == [
        "newave_code",
        "periodo",
        "plane_id",
        "fator_correcao",
        "gamma_0",
        "gamma_v",
        "gamma_q",
        "gamma_s",
        "gamma_lat",
    ]
    assert out.schema["newave_code"] == pl.Int64
    assert out.schema["gamma_q"] == pl.Float64
    assert out["gamma_0"].to_list() == [0.0, 2.5]


def test_read_fpha_grid_renames_and_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "fpha").mkdir()
    (tmp_path / "fpha" / "fpha_eco.csv").write_text("placeholder", encoding="utf-8")
    table = pd.DataFrame(
        {
            "codigo_usina": [6],
            "periodo": [9],
            "nome_usina": ["FURNAS"],
            "tipo": [2],
            "conv": [1],
            "alfa": [1],
            "rems": [0],
            "numero_pontos_vazao_turbinada": [5],
            "vazao_turbinada_minima": [0.0],
            "vazao_turbinada_maxima": [1470.6],
            "numero_pontos_volume_armazenado": [5],
            "volume_armazenado_minimo": [5733.0],
            "volume_armazenado_maximo": [22950.0],
            "geracao_minima": [0.0],
            "geracao_maxima": [1216.0],
        }
    )
    _patch_inewave(monkeypatch, "FphaEco", table)

    out = read_fpha_grid(tmp_path)
    assert out is not None
    assert out["v_min_hm3"].to_list() == [5733.0]
    assert out["n_v"].to_list() == [5]
    assert out.schema["n_v"] == pl.Int64


# --------------------------------------------------------------------------- #
# Envelope evaluator                                                           #
# --------------------------------------------------------------------------- #


def test_evaluate_fpha_envelope_takes_min_over_planes() -> None:
    # Plane A: GH = q ; plane B: GH = 10 (constant). Envelope = min(q, 10).
    gamma_0 = np.array([0.0, 10.0])
    gamma_v = np.array([0.0, 0.0])
    gamma_q = np.array([1.0, 0.0])
    gamma_s = np.array([0.0, 0.0])
    mult = np.array([1.0, 1.0])
    q = np.array([5.0, 20.0])
    zeros = np.zeros_like(q)

    out = dense_grid(gamma_0, gamma_v, gamma_q, gamma_s, mult, zeros, q, zeros)
    assert out.tolist() == [5.0, 10.0]


def test_evaluate_fpha_envelope_applies_volume_offset() -> None:
    # GH = 0.1 * useful_volume, useful = v - offset. At v=110, offset=10 -> 10.0.
    out = dense_grid(
        np.array([0.0]),
        np.array([0.1]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([1.0]),
        np.array([110.0]),
        np.array([0.0]),
        np.array([0.0]),
        volume_offset=10.0,
    )
    assert out[0] == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# Comparison builder                                                           #
# --------------------------------------------------------------------------- #


def _nw_planes(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "newave_code": pl.Int64,
            "periodo": pl.Int64,
            "plane_id": pl.Int64,
            "fator_correcao": pl.Float64,
            "gamma_0": pl.Float64,
            "gamma_v": pl.Float64,
            "gamma_q": pl.Float64,
            "gamma_s": pl.Float64,
            "gamma_lat": pl.Float64,
        },
    )


def _nw_grid(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "newave_code": pl.Int64,
            "periodo": pl.Int64,
            "v_min_hm3": pl.Float64,
            "v_max_hm3": pl.Float64,
            "n_v": pl.Int64,
            "q_min_m3s": pl.Float64,
            "q_max_m3s": pl.Float64,
            "n_q": pl.Int64,
            "gh_min_mw": pl.Float64,
            "gh_max_mw": pl.Float64,
            "tipo": pl.Int64,
        },
    )


def _cb_planes(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "hydro_id": pl.Int32,
            "stage_id": pl.Int32,
            "plane_id": pl.Int32,
            "gamma_0": pl.Float64,
            "gamma_v": pl.Float64,
            "gamma_q": pl.Float64,
            "gamma_s": pl.Float64,
            "kappa": pl.Float64,
            "valid_v_min_hm3": pl.Float64,
            "valid_v_max_hm3": pl.Float64,
            "valid_q_max_m3s": pl.Float64,
        },
    )


_METRIC_COLUMNS = [
    "novomodelo_id",
    "plant_name",
    "stage",
    "n_planes_newave",
    "n_planes_novomodelo",
    "n_v",
    "nmae",
    "bias",
    "max_abs_dev",
    "gh_max_ratio",
]


def test_build_fpha_comparison_returns_empty_when_no_planes() -> None:
    metrics, surface, spill = build_fpha_comparison(None, None, None, [])
    assert metrics.is_empty()
    assert surface.is_empty()
    assert spill.is_empty()
    assert metrics.columns == _METRIC_COLUMNS
    assert surface.columns == [
        "novomodelo_id",
        "plant_name",
        "stage",
        "v_hm3",
        "q_m3s",
        "source",
        "gh_mw",
    ]
    assert spill.columns == [
        "novomodelo_id",
        "plant_name",
        "stage",
        "s_m3s",
        "source",
        "gh_mw",
    ]


def test_build_fpha_comparison_run_of_river_collapses_volume_axis() -> None:
    # Single-volume plant (n_v=1): identical planes GH = 0.5*Q on both sides.
    nw = _nw_planes(
        [
            {
                "newave_code": 10,
                "periodo": 9,
                "plane_id": 1,
                "fator_correcao": 1.0,
                "gamma_0": 0.0,
                "gamma_v": 0.0,
                "gamma_q": 0.5,
                "gamma_s": 0.0,
                "gamma_lat": 0.0,
            }
        ]
    )
    grid = _nw_grid(
        [
            {
                "newave_code": 10,
                "periodo": 9,
                "v_min_hm3": 100.0,
                "v_max_hm3": 100.0,
                "n_v": 1,
                "q_min_m3s": 0.0,
                "q_max_m3s": 100.0,
                "n_q": 5,
                "gh_min_mw": 0.0,
                "gh_max_mw": 50.0,
                "tipo": 2,
            }
        ]
    )
    cb = _cb_planes(
        [
            {
                "hydro_id": 0,
                "stage_id": 0,
                "plane_id": 0,
                "gamma_0": 0.0,
                "gamma_v": 0.0,
                "gamma_q": 0.5,
                "gamma_s": 0.0,
                "kappa": 1.0,
                "valid_v_min_hm3": None,
                "valid_v_max_hm3": None,
                "valid_q_max_m3s": None,
            }
        ]
    )
    hydros = [HydroEntity(newave_code=10, novomodelo_id=0, name="ROR")]

    metrics, surface, spill = build_fpha_comparison(nw, grid, cb, hydros)

    row = metrics.row(0, named=True)
    assert row["gh_max_ratio"] == pytest.approx(1.0)
    assert row["nmae"] == pytest.approx(0.0, abs=1e-9)
    assert row["n_v"] == 1
    # Volume axis collapses to a single point for run-of-river.
    assert surface.filter(pl.col("source") == "newave")["v_hm3"].n_unique() == 1
    assert sorted(surface["source"].unique().to_list()) == ["newave", "novomodelo"]
    assert not spill.is_empty()


def test_build_fpha_comparison_reconciles_useful_vs_absolute_volume() -> None:
    # Reservoir: source model GH = 0.1*(V-10) + 0.5*Q (useful volume);
    # the equivalent Novomodelo plane on absolute volume is -1.0 + 0.1*V + 0.5*Q.
    # If the offset handling is right, the two surfaces coincide exactly.
    nw = _nw_planes(
        [
            {
                "newave_code": 20,
                "periodo": 9,
                "plane_id": 1,
                "fator_correcao": 1.0,
                "gamma_0": 0.0,
                "gamma_v": 0.1,
                "gamma_q": 0.5,
                "gamma_s": 0.0,
                "gamma_lat": 0.0,
            }
        ]
    )
    grid = _nw_grid(
        [
            {
                "newave_code": 20,
                "periodo": 9,
                "v_min_hm3": 10.0,
                "v_max_hm3": 110.0,
                "n_v": 3,
                "q_min_m3s": 0.0,
                "q_max_m3s": 100.0,
                "n_q": 5,
                "gh_min_mw": 0.0,
                "gh_max_mw": 60.0,
                "tipo": 1,
            }
        ]
    )
    cb = _cb_planes(
        [
            {
                "hydro_id": 3,
                "stage_id": 0,
                "plane_id": 0,
                "gamma_0": -1.0,
                "gamma_v": 0.1,
                "gamma_q": 0.5,
                "gamma_s": 0.0,
                "kappa": 1.0,
                "valid_v_min_hm3": None,
                "valid_v_max_hm3": None,
                "valid_q_max_m3s": None,
            }
        ]
    )
    hydros = [HydroEntity(newave_code=20, novomodelo_id=3, name="RES")]

    metrics, surface, _ = build_fpha_comparison(nw, grid, cb, hydros)

    row = metrics.row(0, named=True)
    assert row["gh_max_ratio"] == pytest.approx(1.0)
    assert row["nmae"] == pytest.approx(0.0, abs=1e-9)
    assert row["bias"] == pytest.approx(0.0, abs=1e-9)
    # Sampled at the fitting grid: n_v=3 volumes x n_q=5 turbined points per source.
    assert surface.filter(pl.col("source") == "novomodelo").height == 15


def test_build_fpha_comparison_skips_plant_absent_on_one_side() -> None:
    nw = _nw_planes(
        [
            {
                "newave_code": 10,
                "periodo": 9,
                "plane_id": 1,
                "fator_correcao": 1.0,
                "gamma_0": 0.0,
                "gamma_v": 0.0,
                "gamma_q": 0.5,
                "gamma_s": 0.0,
                "gamma_lat": 0.0,
            }
        ]
    )
    grid = _nw_grid(
        [
            {
                "newave_code": 10,
                "periodo": 9,
                "v_min_hm3": 100.0,
                "v_max_hm3": 100.0,
                "n_v": 1,
                "q_min_m3s": 0.0,
                "q_max_m3s": 100.0,
                "n_q": 5,
                "gh_min_mw": 0.0,
                "gh_max_mw": 50.0,
                "tipo": 2,
            }
        ]
    )
    # Novomodelo has a different hydro (stage 0, id 7) — no overlap with code 10 -> id 0.
    cb = _cb_planes(
        [
            {
                "hydro_id": 7,
                "stage_id": 0,
                "plane_id": 0,
                "gamma_0": 0.0,
                "gamma_v": 0.0,
                "gamma_q": 0.5,
                "gamma_s": 0.0,
                "kappa": 1.0,
                "valid_v_min_hm3": None,
                "valid_v_max_hm3": None,
                "valid_q_max_m3s": None,
            }
        ]
    )
    hydros = [HydroEntity(newave_code=10, novomodelo_id=0, name="ROR")]

    metrics, surface, spill = build_fpha_comparison(nw, grid, cb, hydros)
    assert metrics.is_empty()
    assert surface.is_empty()
    assert spill.is_empty()


# --------------------------------------------------------------------------- #
# Charts                                                                       #
# --------------------------------------------------------------------------- #


def _surface_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "novomodelo_id": pl.Int64,
            "plant_name": pl.Utf8,
            "stage": pl.Int64,
            "v_hm3": pl.Float64,
            "q_m3s": pl.Float64,
            "source": pl.Utf8,
            "gh_mw": pl.Float64,
        },
    )


def _spill_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "novomodelo_id": pl.Int64,
            "plant_name": pl.Utf8,
            "stage": pl.Int64,
            "s_m3s": pl.Float64,
            "source": pl.Utf8,
            "gh_mw": pl.Float64,
        },
    )


def test_fpha_metrics_table_sorts_and_tints_worst() -> None:
    from novomodelo_bridge.comparators.charts import fpha_metrics_table

    metrics = pl.DataFrame(
        {
            "novomodelo_id": [0, 1],
            "plant_name": ["GOOD", "BADPLANT"],
            "stage": [0, 0],
            "n_planes_newave": [4, 11],
            "n_planes_novomodelo": [3, 6],
            "n_v": [1, 5],
            "nmae": [0.001, 0.12],
            "bias": [0.001, -0.10],
            "max_abs_dev": [0.5, 40.0],
            "gh_max_ratio": [1.0, 0.80],
        }
    )
    html = fpha_metrics_table(metrics)
    assert "fpha-metrics-table" in html
    assert "run-of-river" in html and "reservoir" in html
    # Sorted worst-NMAE first: BADPLANT (12%) before GOOD (0.1%).
    assert html.index("BADPLANT") < html.index("GOOD")
    # Worst NMAE > 5% is tinted.
    assert "#FEF3C7" in html


def test_fpha_metrics_table_empty_message() -> None:
    from novomodelo_bridge.comparators.charts import fpha_metrics_table

    out = fpha_metrics_table(pl.DataFrame())
    assert "No production-function (FPHA) data" in out


def test_fpha_detail_chart_reservoir_embeds_3d_surfaces() -> None:
    from novomodelo_bridge.comparators.charts import fpha_detail_chart

    # Reservoir: 2 volumes x 2 turbined points, both sources (v-major order).
    surface = _surface_frame(
        [
            {
                "novomodelo_id": 0,
                "plant_name": "RES",
                "stage": 0,
                "v_hm3": 10.0,
                "q_m3s": 0.0,
                "source": "newave",
                "gh_mw": 0.0,
            },
            {
                "novomodelo_id": 0,
                "plant_name": "RES",
                "stage": 0,
                "v_hm3": 10.0,
                "q_m3s": 5.0,
                "source": "newave",
                "gh_mw": 2.5,
            },
            {
                "novomodelo_id": 0,
                "plant_name": "RES",
                "stage": 0,
                "v_hm3": 20.0,
                "q_m3s": 0.0,
                "source": "newave",
                "gh_mw": 1.0,
            },
            {
                "novomodelo_id": 0,
                "plant_name": "RES",
                "stage": 0,
                "v_hm3": 20.0,
                "q_m3s": 5.0,
                "source": "newave",
                "gh_mw": 3.5,
            },
            {
                "novomodelo_id": 0,
                "plant_name": "RES",
                "stage": 0,
                "v_hm3": 10.0,
                "q_m3s": 0.0,
                "source": "novomodelo",
                "gh_mw": 0.0,
            },
            {
                "novomodelo_id": 0,
                "plant_name": "RES",
                "stage": 0,
                "v_hm3": 10.0,
                "q_m3s": 5.0,
                "source": "novomodelo",
                "gh_mw": 2.6,
            },
            {
                "novomodelo_id": 0,
                "plant_name": "RES",
                "stage": 0,
                "v_hm3": 20.0,
                "q_m3s": 0.0,
                "source": "novomodelo",
                "gh_mw": 1.1,
            },
            {
                "novomodelo_id": 0,
                "plant_name": "RES",
                "stage": 0,
                "v_hm3": 20.0,
                "q_m3s": 5.0,
                "source": "novomodelo",
                "gh_mw": 3.6,
            },
        ]
    )
    spill = _spill_frame(
        [
            {
                "novomodelo_id": 0,
                "plant_name": "RES",
                "stage": 0,
                "s_m3s": 0.0,
                "source": "newave",
                "gh_mw": 3.5,
            },
            {
                "novomodelo_id": 0,
                "plant_name": "RES",
                "stage": 0,
                "s_m3s": 10.0,
                "source": "newave",
                "gh_mw": 3.0,
            },
            {
                "novomodelo_id": 0,
                "plant_name": "RES",
                "stage": 0,
                "s_m3s": 0.0,
                "source": "novomodelo",
                "gh_mw": 3.6,
            },
            {
                "novomodelo_id": 0,
                "plant_name": "RES",
                "stage": 0,
                "s_m3s": 10.0,
                "source": "novomodelo",
                "gh_mw": 3.0,
            },
        ]
    )
    html = fpha_detail_chart(surface, spill)
    assert "fpha-plant" in html and "fpha-stage" in html
    assert "fpha-surf" in html
    assert "'surface'" in html  # 3D surface traces, not heatmaps
    assert "updatemenus" in html and "Difference" in html  # toggle buttons
    assert "fpha-spill" in html
    assert '<option value="0">RES</option>' in html
    # n_v=2 is embedded so the JS renders 3D surfaces (not the run-of-river line).
    assert '"n_v":2' in html.replace(" ", "")


def test_fpha_detail_chart_empty_message() -> None:
    from novomodelo_bridge.comparators.charts import fpha_detail_chart

    out = fpha_detail_chart(_surface_frame([]), _spill_frame([]))
    assert "No production-function (FPHA) data" in out
