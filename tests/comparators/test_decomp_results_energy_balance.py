"""Energy Balance tab tests for ``comparators.decomp.results``.

Second carve out of the legacy ``test_decomp_results_compare.py`` mega file:
the System tab's novomodelo bus percentile metadata and the Energy
Balance frames feeding ``build_decomp_dataset``. The remaining concern bands
(network, costs, performance, hydro/thermal detail, productivity, FPHA, REE,
evaporation, constraints, CLI) stay in the mega file pending their own carve.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from novomodelo_bridge.comparators.charts import _BALANCE_VARS
from novomodelo_bridge.comparators.decomp.results import (
    _energy_balance_frames,
    build_decomp_dataset,
)
from novomodelo_bridge.comparators.report_builder import build_comparison_report
from novomodelo_bridge.decomp.id_map import DecompIdMap
from tests.comparators.conftest import (
    _aligned_fixture,
    _balance_fixture,
    _patch_aligned_frames,
)


class TestSystemTabMetadata:
    """The System tab's novomodelo bus percentile band + the
    exclusion of the transhipment bus from ``results`` bus rows."""

    def _bus_percentiles(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "entity_id": [0],
                "stage_id": [0],
                "spot_price_p10": [40.0],
                "spot_price_p50": [44.0],
                "spot_price_p90": [48.0],
                "deficit_mw_p10": [0.0],
                "deficit_mw_p50": [0.0],
                "deficit_mw_p90": [0.0],
            }
        )

    def test_bus_percentiles_populate_metadata_when_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.novomodelo_readers."
            "read_novomodelo_bus_percentiles",
            lambda *_args, **_kwargs: self._bus_percentiles(),
        )

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        bus_pct = dataset.render.bus
        assert isinstance(bus_pct, pl.DataFrame)
        assert not bus_pct.is_empty()
        assert {
            "spot_price_p10",
            "spot_price_p50",
            "spot_price_p90",
            "deficit_mw_p10",
            "deficit_mw_p50",
            "deficit_mw_p90",
        }.issubset(set(bus_pct.columns))

    def test_bus_percentiles_stay_empty_when_novomodelo_output_lacks_them(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No percentile mock: ``tmp_path`` has no ``simulation/buses``
        partition, so ``read_novomodelo_bus_percentiles`` degrades to its own
        empty-frame default and the dataset must not fabricate a band."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        bus_pct = dataset.render.bus
        assert isinstance(bus_pct, pl.DataFrame)
        assert bus_pct.is_empty()

    def test_system_tab_renders_with_the_percentile_band(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.novomodelo_readers."
            "read_novomodelo_bus_percentiles",
            lambda *_args, **_kwargs: self._bus_percentiles(),
        )
        dataset = build_decomp_dataset(tmp_path, tmp_path)

        html = build_comparison_report(dataset)

        assert "Spot Price by Bus" in html
        assert "Deficit" in html
        assert "No spot_price data available." not in html
        assert "No deficit_mw data available." not in html
        assert "Plotly.newPlot" in html

    def test_system_tab_renders_without_a_band_when_percentiles_are_absent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No band, no error: the System tab still renders both sections
        from the ``results`` bus rows alone."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        html = build_comparison_report(dataset)

        assert "Spot Price by Bus" in html
        assert "Deficit" in html
        assert "No spot_price data available." not in html
        assert "No deficit_mw data available." not in html
        assert "Plotly.newPlot" in html


def _dec_oper_sist_frame() -> pl.DataFrame:
    """Two submarkets, one stage, two nodes (exercises scenario averaging),
    carrying the raw ``dec_oper_sist`` columns ``_energy_balance_frames``
    reads."""
    return pl.DataFrame(
        {
            "estagio": [1, 1, 1, 1],
            "no": [1, 2, 1, 2],
            "patamar": [None, None, None, None],
            "codigo_submercado": [1, 1, 2, 2],
            "demanda_MW": [1000.0, 1000.0, 500.0, 500.0],
            "geracao_hidroeletrica_MW": [600.0, 620.0, 300.0, 300.0],
            "geracao_termica_MW": [200.0, 200.0, 100.0, 100.0],
            "geracao_termica_antecipada_MW": [50.0, 50.0, 0.0, 0.0],
            "geracao_eolica_MW": [30.0, 30.0, 10.0, 10.0],
            "geracao_pequenas_usinas_MW": [20.0, 20.0, 5.0, 5.0],
            "deficit_MW": [0.0, 0.0, 0.0, 0.0],
            "ena_MWmes": [1200.0, 1200.0, 400.0, 400.0],
            "earm_final_MWmes": [5000.0, 5000.0, 2000.0, 2000.0],
        }
    )


class TestEnergyBalanceFrames:
    """``_energy_balance_frames`` -- the Energy Balance tab
    reference frames, built from ``dec_oper_sist``'s stage-aggregate rows."""

    def _bus_codes(self) -> dict[int, int]:
        return {1: 0, 2: 1}

    def _patch_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_sist",
            lambda *_args, **_kwargs: _dec_oper_sist_frame(),
        )

    def test_nw_market_carries_only_the_tokens_the_tab_consumes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch_source(monkeypatch)

        nw_market, _nw_net_load, _nw_sin = _energy_balance_frames(
            tmp_path, self._bus_codes()
        )

        tab_tokens = {nw_var for _, nw_var, _, _ in _BALANCE_VARS if nw_var}
        emitted = set(nw_market["variable"].unique().to_list())
        assert emitted, "fixture must exercise real GHTOT/GTERM/DEFT rows"
        assert emitted <= tab_tokens
        assert emitted == {"GHTOT", "GTERM", "DEFT"}
        assert "EXCESSO" not in emitted

    def test_ghtot_gterm_deft_values_use_the_mapped_novomodelo_bus_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch_source(monkeypatch)

        nw_market, _nw_net_load, _nw_sin = _energy_balance_frames(
            tmp_path, self._bus_codes()
        )

        by_bus = {
            (row["newave_code"], row["variable"]): row["value"]
            for row in nw_market.iter_rows(named=True)
        }
        # Submarket 1 -> novomodelo bus 0: hydro gen averaged over the two nodes
        # ((600+620)/2), GTERM = live (200) + anticipated (50).
        assert by_bus[(0, "GHTOT")] == pytest.approx(610.0)
        assert by_bus[(0, "GTERM")] == pytest.approx(250.0)
        assert by_bus[(0, "DEFT")] == pytest.approx(0.0)
        # Submarket 2 -> novomodelo bus 1.
        assert by_bus[(1, "GHTOT")] == pytest.approx(300.0)
        assert by_bus[(1, "GTERM")] == pytest.approx(100.0)
        assert by_bus[(1, "DEFT")] == pytest.approx(0.0)

    def test_stage_stays_one_based(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch_source(monkeypatch)

        nw_market, nw_net_load, nw_sin = _energy_balance_frames(
            tmp_path, self._bus_codes()
        )

        assert set(nw_market["stage"].unique().to_list()) == {1}
        assert set(nw_net_load["stage"].unique().to_list()) == {1}
        assert set(nw_sin["stage"].unique().to_list()) == {1}

    def test_net_load_subtracts_wind_and_small_plants_from_demand(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch_source(monkeypatch)

        _nw_market, nw_net_load, _nw_sin = _energy_balance_frames(
            tmp_path, self._bus_codes()
        )

        assert set(nw_net_load["variable"].unique().to_list()) == {"NET_LOAD"}
        by_bus = {
            row["newave_code"]: row["value"]
            for row in nw_net_load.iter_rows(named=True)
        }
        assert by_bus[0] == pytest.approx(1000.0 - 30.0 - 20.0)
        assert by_bus[1] == pytest.approx(500.0 - 10.0 - 5.0)

    def test_nw_sin_sums_earmf_and_ena_across_every_submarket(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch_source(monkeypatch)

        _nw_market, _nw_net_load, nw_sin = _energy_balance_frames(
            tmp_path, self._bus_codes()
        )

        by_var = {row["variable"]: row["value"] for row in nw_sin.iter_rows(named=True)}
        assert by_var["EARMF"] == pytest.approx(5000.0 + 2000.0)
        assert by_var["ENA"] == pytest.approx(1200.0 + 400.0)
        # The constant SIN placeholder, matching read_medias_sin's convention.
        assert set(nw_sin["newave_code"].unique().to_list()) == {0}

    def test_transhipment_bus_never_appears(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The converter-created transhipment bus has no ``codigo_submercado``
        of its own -- ``bus_codes`` (``{code: id_map.bus_id(code) for code in
        id_map.bus_codes}``) only ever holds Novomodelo ids in
        ``range(len(bus_codes))``, one short of ``transhipment_bus_id`` -- so
        it structurally cannot appear in ``newave_code``. Regression guard,
        mirroring ``TestBusSideExcludesTranshipment``."""
        id_map = DecompIdMap(bus_codes=(1, 2), bus_names=("SUDESTE", "SUL"))
        bus_codes = {code: id_map.bus_id(code) for code in id_map.bus_codes}
        self._patch_source(monkeypatch)

        nw_market, nw_net_load, _nw_sin = _energy_balance_frames(tmp_path, bus_codes)

        assert id_map.transhipment_bus_id not in nw_market["newave_code"].to_list()
        assert id_map.transhipment_bus_id not in nw_net_load["newave_code"].to_list()

    def test_empty_source_returns_empty_typed_frames(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.read_dec_oper_sist",
            lambda *_args, **_kwargs: pl.DataFrame(),
        )

        nw_market, nw_net_load, nw_sin = _energy_balance_frames(
            tmp_path, self._bus_codes()
        )

        for frame in (nw_market, nw_net_load, nw_sin):
            assert frame.is_empty()
            assert frame.columns == ["newave_code", "stage", "variable", "value"]


def _bus_aggregates_fixture() -> pl.DataFrame:
    """Per-bus Novomodelo percentile aggregates for all five ``_BALANCE_VARS``
    quantities -- the shape :func:`novomodelo_readers.read_novomodelo_bus_aggregates`
    returns."""
    return pl.DataFrame(
        {
            "bus_id": [0],
            "stage_id": [0],
            "hydro_gen_mw_p10": [580.0],
            "hydro_gen_mw_p50": [600.0],
            "hydro_gen_mw_p90": [620.0],
            "thermal_gen_mw_p10": [240.0],
            "thermal_gen_mw_p50": [250.0],
            "thermal_gen_mw_p90": [260.0],
            "net_load_mw_p10": [900.0],
            "net_load_mw_p50": [950.0],
            "net_load_mw_p90": [1000.0],
            "deficit_mw_p10": [0.0],
            "deficit_mw_p50": [0.0],
            "deficit_mw_p90": [0.0],
            "excess_mw_p10": [0.0],
            "excess_mw_p50": [0.0],
            "excess_mw_p90": [0.0],
        }
    )


def _novomodelo_hydro_means_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entity_id": [0],
            "stage_id": [0],
            "stored_energy_final_mwh": [5100000.0],
            "incremental_inflow_energy_mw": [1550.0],
        }
    )


class TestBuildDecompDatasetEnergyBalance:
    """Energy Balance tab metadata (demand / gen-by-source /
    EARM / ENA) filled by ``build_decomp_dataset``."""

    def _patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_aligned_frames(monkeypatch, _balance_fixture())
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.novomodelo_readers."
            "read_novomodelo_bus_aggregates",
            lambda *_args, **_kwargs: _bus_aggregates_fixture(),
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.novomodelo_readers."
            "read_novomodelo_bus_metadata",
            lambda *_args, **_kwargs: {0: {"name": "SE"}},
        )
        monkeypatch.setattr(
            "novomodelo_bridge.comparators.decomp.results.novomodelo_readers."
            "read_novomodelo_hydro_means",
            lambda *_args, **_kwargs: _novomodelo_hydro_means_fixture(),
        )

    def test_metadata_keys_are_present_and_typed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        for key in (
            "nw_market",
            "nw_net_load",
            "bus_aggregates",
            "novomodelo_hydro_means",
            "nw_sin",
        ):
            value = getattr(dataset.render, key)
            assert isinstance(value, pl.DataFrame)
            assert not value.is_empty()
        assert isinstance(dataset.render.novomodelo_bus_meta, dict)
        assert dataset.render.novomodelo_bus_meta
        # D-STAGE-OFFSET: fixed at 1 for DECOMP's 1-based estagio.
        assert dataset.render.nw_offset == 1

    def test_nw_market_tokens_are_all_consumed_by_the_tab(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        tab_tokens = {nw_var for _, nw_var, _, _ in _BALANCE_VARS if nw_var}
        emitted = set(dataset.render.nw_market["variable"].unique().to_list())
        assert emitted
        assert emitted <= tab_tokens

    def test_nw_sin_earm_ena_sums_match_the_fixture(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        nw_sin = dataset.render.nw_sin
        earmf = nw_sin.filter(pl.col("variable") == "EARMF")["value"].to_list()
        ena = nw_sin.filter(pl.col("variable") == "ENA")["value"].to_list()
        assert earmf == pytest.approx([7000.0])
        assert ena == pytest.approx([1600.0])

    def test_excess_panel_renders_novomodelo_only_with_no_fabricated_newave_row(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """DECOMP has no energy-excess quantity (see
        ``_energy_balance_frames``'s docstring): EXCESSO must never appear in
        ``nw_market`` (no dead row), while the tab's Excess panel still
        renders using Novomodelo data alone."""
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert "EXCESSO" not in dataset.render.nw_market["variable"].to_list()

        html = build_comparison_report(dataset)

        assert "Excess" in html

    def test_report_renders_energy_balance_tab_and_system_energy_section(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        html = build_comparison_report(dataset)

        assert "Hydro Generation" in html
        assert "Thermal Generation" in html
        assert "Net Load" in html
        assert "Deficit" in html
        assert "System Energy (EARM / ENA)" in html
        # The DECOMP overlay line on the SIN EARM/ENA charts.
        assert "NEWAVE SIN" in html
        assert "Plotly.newPlot" in html
