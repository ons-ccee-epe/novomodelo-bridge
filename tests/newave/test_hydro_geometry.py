"""Unit tests for the source model hydro geometry converter."""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.diagnostics import Severity
from novomodelo_bridge.newave.id_map import NewaveIdMap

# The title/summary/remediation/notes strings geometry.py emits reach a
# pip-installed user with no repo checkout — none may leak a repo-internal
# reference (mirrors test_constraints.py's own marker scan).
_REPO_INTERNAL_LEAKS = (
    "docs/",
    "plans/",
    "~/git",
    "feat/",
    "ticket-",
    "epic-",
    "src/",
    ".py",
)


def _assert_no_repo_internal_leaks(collected: list[dx.Diagnostic]) -> None:
    for diag in collected:
        strings = [diag.title, diag.summary, *diag.notes]
        if diag.remediation is not None:
            strings.append(diag.remediation)
        for s in strings:
            for leak in _REPO_INTERNAL_LEAKS:
                assert leak not in s, f"diagnostic {diag.code!r} leaks {leak!r}: {s!r}"


def _make_geometry_cadastro() -> pd.DataFrame:
    """Synthetic Hidr.cadastro for generate_hydro_geometry tests.

    Two plants using real inewave column names (a0_volume_cota, a0_cota_area):
    - Plant 1: reservoir plant with vol_min=100, vol_max=1000
      volume_cota: h(v) = 300 + 0.1*v  (a0=300, a1=0.1, rest zero)
      cota_area:   A(h) = 0.5*h         (a0=0, a1=0.5, rest zero)
    - Plant 2: run-of-river with vol_min == vol_max == 50
    """
    return pd.DataFrame(
        {
            "volume_minimo": [100.0, 50.0],
            "volume_maximo": [1000.0, 50.0],
            "a0_volume_cota": [300.0, 300.0],
            "a1_volume_cota": [0.1, 0.1],
            "a2_volume_cota": [0.0, 0.0],
            "a3_volume_cota": [0.0, 0.0],
            "a4_volume_cota": [0.0, 0.0],
            "a0_cota_area": [0.0, 0.0],
            "a1_cota_area": [0.5, 0.5],
            "a2_cota_area": [0.0, 0.0],
            "a3_cota_area": [0.0, 0.0],
            "a4_cota_area": [0.0, 0.0],
        },
        index=pd.Index([1, 2], name="codigo_usina"),
    )


class TestGenerateHydroGeometry:
    """Tests for hydro.generate_hydro_geometry."""

    def test_produces_100_rows_per_plant(self) -> None:
        """A reservoir plant yields exactly 100 rows in the output table."""
        from novomodelo_bridge.newave.converters.hydro import generate_hydro_geometry

        cadastro = _make_geometry_cadastro()
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1, 2], thermal_codes=[])
        table = generate_hydro_geometry(cadastro, id_map)

        # Plant 1 has vol range → 100 rows. Plant 2 is run-of-river → 1 row.
        assert isinstance(table, pa.Table)
        assert len(table) == 101

        novomodelo_id_1 = id_map.hydro_id(1)
        novomodelo_id_2 = id_map.hydro_id(2)
        ids = table.column("hydro_id").to_pylist()
        assert ids.count(novomodelo_id_1) == 100
        assert ids.count(novomodelo_id_2) == 1

    def test_run_of_river_emits_single_point(self) -> None:
        """Plant with vol_min == vol_max produces one geometry row."""
        from novomodelo_bridge.newave.converters.hydro import generate_hydro_geometry

        cadastro = _make_geometry_cadastro()
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[2], thermal_codes=[])
        table = generate_hydro_geometry(cadastro, id_map)

        assert len(table) == 1

    def test_correct_schema(self) -> None:
        """Output table has the required schema with correct column types."""
        from novomodelo_bridge.newave.converters.hydro import generate_hydro_geometry

        cadastro = _make_geometry_cadastro()
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1], thermal_codes=[])
        table = generate_hydro_geometry(cadastro, id_map)

        assert table.schema.field("hydro_id").type == pa.int32()
        assert table.schema.field("volume_hm3").type == pa.float64()
        assert table.schema.field("height_m").type == pa.float64()
        assert table.schema.field("area_km2").type == pa.float64()

    def test_correct_schema_roundtrip_parquet(self, tmp_path) -> None:
        """Schema is preserved when written and read back as Parquet."""
        import pyarrow.parquet as pq

        from novomodelo_bridge.newave.converters.hydro import generate_hydro_geometry

        cadastro = _make_geometry_cadastro()
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1], thermal_codes=[])
        table = generate_hydro_geometry(cadastro, id_map)

        out = tmp_path / "hydro_geometry.parquet"
        pq.write_table(table, out)
        reloaded = pq.read_table(out)

        assert reloaded.schema.field("hydro_id").type == pa.int32()
        assert reloaded.schema.field("volume_hm3").type == pa.float64()
        assert reloaded.schema.field("height_m").type == pa.float64()
        assert reloaded.schema.field("area_km2").type == pa.float64()
        assert len(reloaded) == 100

    def test_volumes_are_uniformly_spaced(self) -> None:
        """The 100 volume points are uniformly distributed on [vol_min, vol_max]."""
        import numpy as np

        from novomodelo_bridge.newave.converters.hydro import generate_hydro_geometry

        cadastro = _make_geometry_cadastro()
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1], thermal_codes=[])
        table = generate_hydro_geometry(cadastro, id_map)

        vols = table.column("volume_hm3").to_pylist()
        expected = np.linspace(100.0, 1000.0, 100).tolist()
        assert vols == pytest.approx(expected, rel=1e-9)

    def test_polynomial_evaluation_correctness(self) -> None:
        """Heights and areas match the expected polynomial values."""
        import numpy as np

        from novomodelo_bridge.newave.converters.hydro import generate_hydro_geometry

        cadastro = _make_geometry_cadastro()
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1], thermal_codes=[])
        table = generate_hydro_geometry(cadastro, id_map)

        vols = np.array(table.column("volume_hm3").to_pylist())
        heights = np.array(table.column("height_m").to_pylist())
        areas = np.array(table.column("area_km2").to_pylist())

        # h(v) = 300 + 0.1*v
        expected_heights = 300.0 + 0.1 * vols
        np.testing.assert_allclose(heights, expected_heights, rtol=1e-9)

        # A(h) = 0.5 * h
        expected_areas = 0.5 * expected_heights
        np.testing.assert_allclose(areas, expected_areas, rtol=1e-9)

    def test_skips_all_zero_volume_cota(self) -> None:
        """Plant with all-zero volume_cota coefficients is skipped (no rows emitted)."""
        from novomodelo_bridge.newave.converters.hydro import generate_hydro_geometry

        # Build a cadastro with all-zero volume_cota for plant 1.
        cadastro = _make_geometry_cadastro().copy()
        for i in range(5):
            cadastro.loc[1, f"a{i}_volume_cota"] = 0.0

        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1], thermal_codes=[])
        # Should not raise; plant is silently skipped after logging a warning.
        table = generate_hydro_geometry(cadastro, id_map)
        assert len(table) == 0

    def test_negative_values_clamped_to_zero(self) -> None:
        """Negative polynomial outputs are clamped to 0.0."""
        from novomodelo_bridge.newave.converters.hydro import generate_hydro_geometry

        # volume_cota: h(v) = -1000 + v  (negative at low volumes)
        # cota_area:   A(h) = -1000 + h  (negative at low heights)
        cadastro = _make_geometry_cadastro().copy()
        cadastro.loc[1, "a0_volume_cota"] = -1000.0
        cadastro.loc[1, "a1_volume_cota"] = 1.0
        cadastro.loc[1, "a2_volume_cota"] = 0.0
        cadastro.loc[1, "a3_volume_cota"] = 0.0
        cadastro.loc[1, "a4_volume_cota"] = 0.0
        cadastro.loc[1, "a0_cota_area"] = -1000.0
        cadastro.loc[1, "a1_cota_area"] = 1.0
        cadastro.loc[1, "a2_cota_area"] = 0.0
        cadastro.loc[1, "a3_cota_area"] = 0.0
        cadastro.loc[1, "a4_cota_area"] = 0.0

        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1], thermal_codes=[])
        table = generate_hydro_geometry(cadastro, id_map)

        heights = table.column("height_m").to_pylist()
        areas = table.column("area_km2").to_pylist()
        assert all(h >= 0.0 for h in heights), "Heights must be >= 0"
        assert all(a >= 0.0 for a in areas), "Areas must be >= 0"

    @pytest.mark.parametrize(
        ("tipo", "vmin", "vmax", "vref"),
        [
            ("D", 50.0, 50.0, 40.0),
            ("S", 100.0, 1000.0, 1200.0),
            ("D", 100.0, 1000.0, 500.0),
        ],
    )
    def test_daily_and_run_of_river_emit_one_point_at_reference_volume(
        self, tipo: str, vmin: float, vmax: float, vref: float
    ) -> None:
        """A 'D'/'S' plant's single row sits at ``volume_referencia``, the volume
        its reservoir is held at, even outside the hidr range."""
        from novomodelo_bridge.newave.converters.hydro import generate_hydro_geometry

        cadastro = _make_geometry_cadastro().copy()
        cadastro["tipo_regulacao"] = ["M", tipo]
        cadastro["volume_referencia"] = [550.0, vref]
        cadastro.loc[2, "volume_minimo"] = vmin
        cadastro.loc[2, "volume_maximo"] = vmax
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[2], thermal_codes=[])
        table = generate_hydro_geometry(cadastro, id_map)

        assert table.column("volume_hm3").to_pylist() == [vref]
        assert table.column("height_m").to_pylist() == pytest.approx(
            [300.0 + 0.1 * vref]
        )

    def test_monthly_reservoir_keeps_hidr_range(self) -> None:
        """An 'M' plant samples ``[volume_minimo, volume_maximo]`` whatever its
        ``volume_referencia``."""
        from novomodelo_bridge.newave.converters.hydro import generate_hydro_geometry

        cadastro = _make_geometry_cadastro().copy()
        cadastro["tipo_regulacao"] = ["M", "D"]
        cadastro["volume_referencia"] = [1500.0, 50.0]
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1], thermal_codes=[])
        vols = (
            generate_hydro_geometry(cadastro, id_map).column("volume_hm3").to_pylist()
        )

        assert len(vols) == 100
        assert (vols[0], vols[-1]) == pytest.approx((100.0, 1000.0))


class TestGeometryCoversReferenceVolumes:
    """The geometry reaches every volume novomodelo reads a downstream level at."""

    def test_reference_volume_above_vmax_extends_geometry(self) -> None:
        from novomodelo_bridge.newave.converters.hydro import generate_hydro_geometry

        cadastro = _make_geometry_cadastro()
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1], thermal_codes=[])
        with dx.collect() as collected:
            table = generate_hydro_geometry(
                cadastro, id_map, {1: [300.0] * 11 + [1200.0]}
            )

        volumes = table.column("volume_hm3").to_pylist()
        heights = table.column("height_m").to_pylist()
        assert volumes[0] == pytest.approx(100.0)
        assert volumes[-1] == pytest.approx(1200.0)
        assert heights[-1] == pytest.approx(300.0 + 0.1 * 1200.0)
        [diagnostic] = [d for d in collected if d.code == "hydro-geometry-extended"]
        assert diagnostic.severity is Severity.INFO
        assert diagnostic.table is not None
        assert [row[1:] for row in diagnostic.table.rows] == [
            [1, 100.0, 1000.0, 100.0, 1200.0]
        ]
        _assert_no_repo_internal_leaks(collected)

    def test_reference_volume_below_vmin_extends_geometry(self) -> None:
        from novomodelo_bridge.newave.converters.hydro import generate_hydro_geometry

        cadastro = _make_geometry_cadastro()
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1], thermal_codes=[])
        with dx.collect() as collected:
            table = generate_hydro_geometry(cadastro, id_map, {1: [60.0, 300.0]})

        volumes = table.column("volume_hm3").to_pylist()
        assert volumes[0] == pytest.approx(60.0)
        assert volumes[-1] == pytest.approx(1000.0)
        [diagnostic] = [d for d in collected if d.code == "hydro-geometry-extended"]
        assert diagnostic.table is not None
        assert [row[1:] for row in diagnostic.table.rows] == [
            [1, 100.0, 1000.0, 60.0, 1000.0]
        ]

    def test_reference_volumes_within_range_keep_geometry(self) -> None:
        from novomodelo_bridge.newave.converters.hydro import generate_hydro_geometry

        cadastro = _make_geometry_cadastro()
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1], thermal_codes=[])
        with dx.collect() as collected:
            table = generate_hydro_geometry(cadastro, id_map, {1: [300.0] * 12})

        assert max(table.column("volume_hm3").to_pylist()) == pytest.approx(1000.0)
        assert [d for d in collected if d.code == "hydro-geometry-extended"] == []

    def test_single_volume_plant_ignores_reference_volumes(self) -> None:
        from novomodelo_bridge.newave.converters.hydro import generate_hydro_geometry

        cadastro = _make_geometry_cadastro()
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[2], thermal_codes=[])
        table = generate_hydro_geometry(cadastro, id_map, {2: [80.0] * 12})

        assert table.column("volume_hm3").to_pylist() == [50.0]


class TestOperatingStorageRange:
    """The one storage range the reservoir bounds, geometry and FPHA window read."""

    @pytest.mark.parametrize(
        ("tipo", "vref", "expected"),
        [
            ("M", 500.0, (100.0, 1000.0)),
            ("D", 500.0, (500.0, 500.0)),
            ("S", 1200.0, (1200.0, 1200.0)),
            ("D", float("nan"), (100.0, 1000.0)),
            (" ", 500.0, (100.0, 1000.0)),
        ],
    )
    def test_collapses_daily_and_run_of_river_to_reference_volume(
        self, tipo: str, vref: float, expected: tuple[float, float]
    ) -> None:
        from novomodelo_bridge.newave.converters.hydro.geometry import (
            operating_storage_range,
        )

        hreg = pd.Series(
            {
                "tipo_regulacao": tipo,
                "volume_minimo": 100.0,
                "volume_maximo": 1000.0,
                "volume_referencia": vref,
            }
        )
        assert operating_storage_range(hreg) == expected


class TestGenerateHydroGeometryDiagnostics:
    """Structured-diagnostic coverage for the loop-accumulated geometry skips."""

    def test_skips_emit_one_table_distinguishing_both_causes(self) -> None:
        """Both skip causes fold into one ``hydro-geometry-skipped`` table."""
        from novomodelo_bridge.newave.converters.hydro import generate_hydro_geometry

        cadastro = _make_geometry_cadastro().copy()
        for i in range(5):
            cadastro.loc[1, f"a{i}_volume_cota"] = 0.0
        cadastro.loc[1, "nome_usina"] = "PLANT ONE"

        # Code 3 is referenced by id_map but absent from cadastro entirely.
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1, 3], thermal_codes=[])

        with dx.collect() as collected:
            table = generate_hydro_geometry(cadastro, id_map)

        assert len(table) == 0
        assert len(collected) == 1
        diag = collected[0]
        assert diag.code == "hydro-geometry-skipped"
        assert diag.severity is Severity.WARNING
        assert diag.category == "Hydro geometry"
        assert diag.table is not None
        assert diag.table.columns == ["Plant", "Code", "Reason"]
        assert len(diag.table.rows) == 2

        by_code = {row[1]: row for row in diag.table.rows}
        assert by_code[1][0] == "PLANT ONE"
        assert "polynomial" in str(by_code[1][2])
        assert by_code[3][0] == "?"
        assert "cadastro" in str(by_code[3][2])
        assert by_code[1][2] != by_code[3][2]

        _assert_no_repo_internal_leaks(collected)

    def test_no_skips_emits_no_diagnostic(self) -> None:
        from novomodelo_bridge.newave.converters.hydro import generate_hydro_geometry

        cadastro = _make_geometry_cadastro()
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1, 2], thermal_codes=[])

        with dx.collect() as collected:
            generate_hydro_geometry(cadastro, id_map)

        assert collected == []

    def test_skips_carry_no_legacy_warning_under_collect(self) -> None:
        from novomodelo_bridge.newave.converters.hydro import generate_hydro_geometry

        cadastro = _make_geometry_cadastro().copy()
        for i in range(5):
            cadastro.loc[1, f"a{i}_volume_cota"] = 0.0
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1], thermal_codes=[])

        with dx.collect() as collected:
            generate_hydro_geometry(cadastro, id_map)

        assert not any(d.code == "legacy-warning" for d in collected)

    def test_no_sink_fallback_logs_one_warning(self, caplog) -> None:
        """With no active collect() sink, emit() degrades to a single logging
        record — the pre-migration caplog contract keeps working."""
        import logging

        from novomodelo_bridge.newave.converters.hydro import generate_hydro_geometry

        cadastro = _make_geometry_cadastro().copy()
        for i in range(5):
            cadastro.loc[1, f"a{i}_volume_cota"] = 0.0
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1], thermal_codes=[])

        with caplog.at_level(logging.WARNING):
            generate_hydro_geometry(cadastro, id_map)

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
