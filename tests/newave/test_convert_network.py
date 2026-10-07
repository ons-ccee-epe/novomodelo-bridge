"""Unit tests for the source model network converter (buses, lines, penalties)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pytest

from novomodelo_bridge.newave.case import NewaveCase
from novomodelo_bridge.newave.id_map import NewaveIdMap
from tests.conftest import make_case
from tests.newave.conftest import _make_sistema_mock

# ---------------------------------------------------------------------------
# Bus conversion
# ---------------------------------------------------------------------------


class TestConvertBuses:
    def _make_id_map(self) -> NewaveIdMap:
        # Subsystems: 1, 2, 99 (fictitious)
        return NewaveIdMap(
            subsystem_ids=[1, 2, 99],
            hydro_codes=[],
            thermal_codes=[],
        )

    def test_returns_buses_key(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.network import convert_buses

        case = make_case(tmp_path, sistema=_make_sistema_mock())
        result = convert_buses(case, self._make_id_map())
        assert "buses" in result

    def test_bus_count_includes_fictitious(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.network import convert_buses

        case = make_case(tmp_path, sistema=_make_sistema_mock())
        result = convert_buses(case, self._make_id_map())
        # 3 subsystems total: 1, 2, 99.
        assert len(result["buses"]) == 3

    def test_bus_ids_are_zero_based_sorted(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.network import convert_buses

        case = make_case(tmp_path, sistema=_make_sistema_mock())
        result = convert_buses(case, self._make_id_map())
        ids = [b["id"] for b in result["buses"]]
        assert ids == sorted(ids)
        assert ids[0] == 0

    def test_bus_has_deficit_segments(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.network import convert_buses

        case = make_case(tmp_path, sistema=_make_sistema_mock())
        result = convert_buses(case, self._make_id_map())
        for b in result["buses"]:
            assert "deficit_segments" in b
            assert isinstance(b["deficit_segments"], list)
            assert len(b["deficit_segments"]) == 2  # 2 patamares

    def test_last_deficit_segment_depth_is_null(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.network import convert_buses

        case = make_case(tmp_path, sistema=_make_sistema_mock())
        result = convert_buses(case, self._make_id_map())
        for b in result["buses"]:
            last_seg = b["deficit_segments"][-1]
            assert last_seg["depth_mw"] is None


# ---------------------------------------------------------------------------
# Line conversion
# ---------------------------------------------------------------------------


class TestConvertLines:
    def _make_id_map(self) -> NewaveIdMap:
        return NewaveIdMap(
            subsystem_ids=[1, 2, 99],
            hydro_codes=[],
            thermal_codes=[],
        )

    def _make_case(self, tmp_path):
        dger = MagicMock()
        dger.mes_inicio_estudo = 1
        dger.ano_inicio_estudo = 2023
        return make_case(tmp_path, sistema=_make_sistema_mock(), dger=dger)

    def test_returns_lines_key(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.network import convert_lines

        result = convert_lines(self._make_case(tmp_path), self._make_id_map())
        assert "lines" in result

    def test_line_count_three_pairs(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.network import convert_lines

        result = convert_lines(self._make_case(tmp_path), self._make_id_map())
        assert len(result["lines"]) == 3

    def test_line_capacity_structure(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.network import convert_lines

        result = convert_lines(self._make_case(tmp_path), self._make_id_map())
        for line in result["lines"]:
            assert "capacity" in line
            assert "direct_mw" in line["capacity"]
            assert "reverse_mw" in line["capacity"]
            assert "source_bus_id" in line
            assert "target_bus_id" in line

    def test_line_ids_sequential(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.network import convert_lines

        result = convert_lines(self._make_case(tmp_path), self._make_id_map())
        ids = [ln["id"] for ln in result["lines"]]
        assert ids == list(range(len(ids)))

    def test_first_month_used_for_capacity(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.network import convert_lines

        result = convert_lines(self._make_case(tmp_path), self._make_id_map())
        line_12 = next(
            ln
            for ln in result["lines"]
            if ln["source_bus_id"] == 0 and ln["target_bus_id"] == 1
        )
        assert line_12["capacity"]["direct_mw"] == pytest.approx(3000.0)
        assert line_12["capacity"]["reverse_mw"] == pytest.approx(2500.0)

    def test_fictitious_lines_get_half_exchange_cost(self, tmp_path) -> None:
        from novomodelo_bridge.core.penalties import PINT
        from novomodelo_bridge.newave.converters.network import (
            _PINT_FICTITIOUS_DISCOUNT,
            convert_lines,
        )

        id_map = NewaveIdMap(
            subsystem_ids=[1, 2, 99],
            hydro_codes=[],
            thermal_codes=[],
        )
        result = convert_lines(self._make_case(tmp_path), id_map)

        fict_bus_id = id_map.bus_id(99)
        expected = PINT * _PINT_FICTITIOUS_DISCOUNT
        for ln in result["lines"]:
            touches_fict = (
                ln["source_bus_id"] == fict_bus_id or ln["target_bus_id"] == fict_bus_id
            )
            if touches_fict:
                assert ln["exchange_cost"] == pytest.approx(expected)
            else:
                assert "exchange_cost" not in ln


# ---------------------------------------------------------------------------
# Line bounds conversion (folds patamar.dat exchange factors into per-block
# rows — novomodelo decision 10)
# ---------------------------------------------------------------------------


def _make_line_bounds_patamar_df() -> pd.DataFrame:
    """Per-block exchange factors for line (1, 2) and line (2, 99).

    Line (1, 2) [line_id=0]: block 0 has a non-uniform direct factor (0.9),
    block 1 is uniform (1.0/1.0) and so should emit no row.
    Line (2, 99) [line_id=2]: both blocks differ (direct and reverse).
    Line (1, 99) [line_id=1] gets no rows at all: every block defaults to
    factor 1.0 (uniform), so it must emit zero block rows.
    """
    import datetime

    d = datetime.datetime(2023, 1, 1)
    rows = [
        # Line (1, 2): block 0 (patamar=1) direct factor 0.9; block 1 uniform.
        {
            "submercado_de": 1,
            "submercado_para": 2,
            "patamar": 1,
            "data": d,
            "valor": 0.9,
        },
        {
            "submercado_de": 1,
            "submercado_para": 2,
            "patamar": 2,
            "data": d,
            "valor": 1.0,
        },
        # Line (2, 99): block 0 differs in both directions.
        {
            "submercado_de": 2,
            "submercado_para": 99,
            "patamar": 1,
            "data": d,
            "valor": 0.8,
        },
        {
            "submercado_de": 99,
            "submercado_para": 2,
            "patamar": 1,
            "data": d,
            "valor": 0.95,
        },
        # Line (2, 99): block 1 differs in the direct direction only.
        {
            "submercado_de": 2,
            "submercado_para": 99,
            "patamar": 2,
            "data": d,
            "valor": 1.1,
        },
    ]
    return pd.DataFrame(rows)


class TestConvertLineBounds:
    def _make_id_map(self) -> NewaveIdMap:
        return NewaveIdMap(
            subsystem_ids=[1, 2, 99],
            hydro_codes=[],
            thermal_codes=[],
        )

    def _make_case(self, tmp_path, patamar_df: pd.DataFrame):
        dger = MagicMock()
        dger.mes_inicio_estudo = 1
        dger.ano_inicio_estudo = 2023
        patamar = MagicMock()
        patamar.intercambio_patamares = patamar_df
        return make_case(
            tmp_path, sistema=_make_sistema_mock(), dger=dger, patamar=patamar
        )

    def test_block_id_column_is_nullable_int32(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.network import convert_line_bounds

        case = self._make_case(tmp_path, _make_line_bounds_patamar_df())
        table = convert_line_bounds(case, self._make_id_map())
        field = table.schema.field("block_id")
        assert field.type == pa.int32()
        assert field.nullable

    def test_stage_level_base_rows_carry_none_block_id(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.network import convert_line_bounds

        case = self._make_case(tmp_path, _make_line_bounds_patamar_df())
        table = convert_line_bounds(case, self._make_id_map())
        df = table.to_pandas()

        base_rows = df[df["block_id"].isna()]
        # One base row per (line, stage): 3 lines x 24 stages (12 study +
        # 12 post-study, per the dger fixture below).
        assert len(base_rows) == 3 * 24

    def test_differing_block_factor_folds_into_absolute_mw(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.network import convert_line_bounds

        case = self._make_case(tmp_path, _make_line_bounds_patamar_df())
        table = convert_line_bounds(case, self._make_id_map())
        df = table.to_pandas()

        # Line 0 = (1, 2): base direct=3000.0, reverse=2500.0 at stage 0.
        row = df[(df["line_id"] == 0) & (df["stage_id"] == 0) & (df["block_id"] == 0)]
        assert len(row) == 1
        assert row.iloc[0]["direct_mw"] == pytest.approx(3000.0 * 0.9, rel=1e-9)
        assert row.iloc[0]["reverse_mw"] == pytest.approx(2500.0 * 1.0, rel=1e-9)

        # Line 2 = (2, 99): base direct=1500.0, reverse=1200.0 at stage 0.
        row_b0 = df[
            (df["line_id"] == 2) & (df["stage_id"] == 0) & (df["block_id"] == 0)
        ]
        assert row_b0.iloc[0]["direct_mw"] == pytest.approx(1500.0 * 0.8, rel=1e-9)
        assert row_b0.iloc[0]["reverse_mw"] == pytest.approx(1200.0 * 0.95, rel=1e-9)

        row_b1 = df[
            (df["line_id"] == 2) & (df["stage_id"] == 0) & (df["block_id"] == 1)
        ]
        assert row_b1.iloc[0]["direct_mw"] == pytest.approx(1500.0 * 1.1, rel=1e-9)
        assert row_b1.iloc[0]["reverse_mw"] == pytest.approx(1200.0 * 1.0, rel=1e-9)

    def test_uniform_block_factor_emits_no_block_row(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.network import convert_line_bounds

        case = self._make_case(tmp_path, _make_line_bounds_patamar_df())
        table = convert_line_bounds(case, self._make_id_map())
        df = table.to_pandas()

        # Line 0's block 1 (patamar=2, factor 1.0/1.0) is uniform with the
        # base, so it must not appear.
        rows = df[(df["line_id"] == 0) & (df["stage_id"] == 0) & (df["block_id"] == 1)]
        assert rows.empty

        # Line 1 = (1, 99) never appears in the patamar fixture: every block
        # defaults to factor 1.0, so it must get zero block rows entirely.
        rows = df[(df["line_id"] == 1) & df["block_id"].notna()]
        assert rows.empty

    def test_block_row_count_matches_differing_combinations(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.network import convert_line_bounds

        case = self._make_case(tmp_path, _make_line_bounds_patamar_df())
        table = convert_line_bounds(case, self._make_id_map())
        df = table.to_pandas()

        block_rows = df[df["block_id"].notna()]
        # line 0 block 0 + line 2 blocks 0 and 1 = 3 differing combinations,
        # each recurring twice: at stage 0 (January 2023, the fixture's dated
        # record) and at stage 12 (January 2024). Unlike the base direct/
        # reverse bounds — which freeze at the last study stage's value for
        # the whole post-study tail — the factor lookup seasonally repeats by
        # calendar month post-study (ported unchanged from the deleted
        # ``convert_exchange_factors``), so January's factor recurs every
        # January thereafter. 3 combinations x 2 recurrences = 6.
        assert len(block_rows) == 6

    def test_block_id_within_declared_block_count(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.network import convert_line_bounds

        case = self._make_case(tmp_path, _make_line_bounds_patamar_df())
        table = convert_line_bounds(case, self._make_id_map())
        df = table.to_pandas()

        block_rows = df[df["block_id"].notna()]
        # The fixture declares 2 blocks (patamar 1 and 2).
        assert (block_rows["block_id"] >= 0).all()
        assert (block_rows["block_id"] < 2).all()

    def test_no_patamar_data_emits_only_base_rows(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.network import convert_line_bounds

        case = self._make_case(tmp_path, pd.DataFrame())
        table = convert_line_bounds(case, self._make_id_map())
        df = table.to_pandas()

        assert df["block_id"].isna().all()
        assert len(df) == 3 * 24


# ---------------------------------------------------------------------------
# line_bounds zero-capability.
#
# The synthetic shape tests above (``TestConvertLineBounds``) pin the folding
# mechanics on a small hand-built fixture. This class pins one more claim: a
# synthetic zero block factor -- unrepresentable in the deleted
# strictly-positive-factor encoding -- converts to an ordinary
# ``direct_mw == 0.0`` bound without raising.
# ---------------------------------------------------------------------------


def _newave_raw_factor_tables(
    factors_df: pd.DataFrame,
) -> tuple[
    dict[tuple[int, int, int, int, int], float],
    dict[tuple[int, int, int, int, int], float],
    dict[tuple[int, int, int, int], float],
    dict[tuple[int, int, int, int], float],
]:
    """Independent per-block direct/reverse factor tables read straight off
    ``patamar.dat`` -- built fresh here, never touching
    ``convert_line_bounds``'s own ``direct_factor_map``/``last_direct_factor``
    state.

    Returns ``(raw_direct, raw_reverse, latest_direct, latest_reverse)``:
    the first two keyed ``(src, tgt, year, month, block_id)`` for an exact
    dated hit, the last two keyed ``(src, tgt, month, block_id)`` holding the
    most recent year on file for that calendar month (the seasonal-repeat
    fallback the module docstring documents for post-study stages).
    """
    raw_direct: dict[tuple[int, int, int, int, int], float] = {}
    raw_reverse: dict[tuple[int, int, int, int, int], float] = {}
    for _, row in factors_df.iterrows():
        de, para = int(row["submercado_de"]), int(row["submercado_para"])
        block_id = int(row["patamar"]) - 1
        year, month = int(row["data"].year), int(row["data"].month)
        val = float(row["valor"])
        src, tgt = (de, para) if de < para else (para, de)
        key = (src, tgt, year, month, block_id)
        if de < para:
            raw_direct[key] = val
        else:
            raw_reverse[key] = val

    def _latest_by_month(
        raw: dict[tuple[int, int, int, int, int], float],
    ) -> dict[tuple[int, int, int, int], float]:
        latest: dict[tuple[int, int, int, int], tuple[int, float]] = {}
        for (src, tgt, year, month, block_id), val in raw.items():
            k = (src, tgt, month, block_id)
            cur = latest.get(k)
            if cur is None or year > cur[0]:
                latest[k] = (year, val)
        return {k: v for k, (_, v) in latest.items()}

    return (
        raw_direct,
        raw_reverse,
        _latest_by_month(raw_direct),
        _latest_by_month(raw_reverse),
    )


def _newave_expected_factor(
    raw: dict[tuple[int, int, int, int, int], float],
    latest_by_month: dict[tuple[int, int, int, int], float],
    src: int,
    tgt: int,
    year: int,
    month: int,
    block_id: int,
    *,
    is_post_study: bool,
) -> float:
    """The documented factor-resolution rule (``convert_line_bounds``'s own
    docstring): post-study stages seasonally repeat the latest recorded
    value for the same calendar month; study-period stages take an exact
    ``(year, month)`` hit, else fall back to the same seasonal repeat, else
    the neutral factor ``1.0`` (uniform, no override).
    """
    if is_post_study:
        return latest_by_month.get((src, tgt, month, block_id), 1.0)
    exact = raw.get((src, tgt, year, month, block_id))
    if exact is not None:
        return exact
    return latest_by_month.get((src, tgt, month, block_id), 1.0)


class TestConvertLineBoundsZeroCapability:
    """novomodelo decision 10 makes ``direct_mw = 0.0`` an ordinary bound. No
    current deck ever records a zero exchange factor (measured: 1152
    factors across the converted example cases, zero zeros, min 0.5582), so
    only a synthetic fixture can pin the new capability."""

    def _make_id_map(self) -> NewaveIdMap:
        return NewaveIdMap(subsystem_ids=[1, 2, 99], hydro_codes=[], thermal_codes=[])

    def _make_case(self, tmp_path: Path, patamar_df: pd.DataFrame) -> NewaveCase:
        dger = MagicMock()
        dger.mes_inicio_estudo = 1
        dger.ano_inicio_estudo = 2023
        patamar = MagicMock()
        patamar.intercambio_patamares = patamar_df
        return make_case(
            tmp_path, sistema=_make_sistema_mock(), dger=dger, patamar=patamar
        )

    def test_zero_block_factor_converts_without_raising(self, tmp_path: Path) -> None:
        import datetime

        from novomodelo_bridge.newave.converters.network import convert_line_bounds

        d = datetime.datetime(2023, 1, 1)
        patamar_df = pd.DataFrame(
            [
                # Line (1, 2) [line_id=0]: block 0's direct factor is
                # exactly zero -- a line fully out in that block, the shape
                # the deleted strictly-positive factor encoding could never
                # represent. Block 1 is non-zero and distinct from 1.0, to
                # prove it is genuinely unaffected rather than coincidentally
                # matching the base.
                {
                    "submercado_de": 1,
                    "submercado_para": 2,
                    "patamar": 1,
                    "data": d,
                    "valor": 0.0,
                },
                {
                    "submercado_de": 1,
                    "submercado_para": 2,
                    "patamar": 2,
                    "data": d,
                    "valor": 0.75,
                },
            ]
        )
        case = self._make_case(tmp_path, patamar_df)

        table = convert_line_bounds(case, self._make_id_map())  # must not raise
        df = table.to_pandas()

        # Base for line (1, 2) at stage 0: direct=3000.0, reverse=2500.0
        # (see `_make_intercambio_df`).
        zero_row = df[
            (df["line_id"] == 0) & (df["stage_id"] == 0) & (df["block_id"] == 0)
        ]
        assert len(zero_row) == 1
        assert zero_row.iloc[0]["direct_mw"] == 0.0
        # The reverse direction has no patamar row for this block => factor
        # 1.0 => untouched by the zero direct factor.
        assert zero_row.iloc[0]["reverse_mw"] == pytest.approx(2500.0)

        other_block = df[
            (df["line_id"] == 0) & (df["stage_id"] == 0) & (df["block_id"] == 1)
        ]
        assert len(other_block) == 1
        assert other_block.iloc[0]["direct_mw"] == pytest.approx(
            3000.0 * 0.75, rel=1e-9
        )


# ---------------------------------------------------------------------------
# Penalties conversion
# ---------------------------------------------------------------------------


class TestConvertPenalties:
    def test_returns_required_keys(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.network import convert_penalties

        case = make_case(tmp_path, sistema=_make_sistema_mock())
        result = convert_penalties(
            case,
            {
                "hydros": [
                    {
                        "generation": {"productivity_mw_per_m3s": 1.0},
                        "reservoir": {
                            "max_storage_hm3": 1000.0,
                            "min_storage_hm3": 100.0,
                        },
                    }
                ]
            },
        )
        for key in ("bus", "hydro", "line", "non_controllable_source"):
            assert key in result

    def test_bus_deficit_uses_first_subsystem_first_tier(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.network import convert_penalties

        case = make_case(tmp_path, sistema=_make_sistema_mock())
        result = convert_penalties(
            case,
            {
                "hydros": [
                    {
                        "generation": {"productivity_mw_per_m3s": 1.0},
                        "reservoir": {
                            "max_storage_hm3": 1000.0,
                            "min_storage_hm3": 100.0,
                        },
                    }
                ]
            },
        )
        # First subsystem=1, patamar=1: custo = 500.0*1 = 500.0
        seg = result["bus"]["deficit_segments"][0]
        assert seg["cost"] == pytest.approx(500.0)

    def test_hydro_has_all_penalty_fields(self, tmp_path) -> None:
        from novomodelo_bridge.newave.converters.network import convert_penalties

        case = make_case(tmp_path, sistema=_make_sistema_mock())
        result = convert_penalties(
            case,
            {
                "hydros": [
                    {
                        "generation": {"productivity_mw_per_m3s": 1.0},
                        "reservoir": {
                            "max_storage_hm3": 1000.0,
                            "min_storage_hm3": 100.0,
                        },
                    }
                ]
            },
        )
        required = {
            "spillage_cost",
            "turbined_cost",
            "diversion_cost",
            "storage_violation_below_cost",
            "filling_target_violation_cost",
            "turbined_violation_below_cost",
            "outflow_violation_below_cost",
            "outflow_violation_above_cost",
            "generation_violation_below_cost",
            "evaporation_violation_cost",
            "water_withdrawal_violation_cost",
            "inflow_nonnegativity_cost",
        }
        assert required == set(result["hydro"].keys())


class TestHydroPenaltyCosts:
    """The pure ρ-scaling helper shared by the base and per-stage paths."""

    def test_flow_penalties_scale_linearly_with_rho_avg(self) -> None:
        from novomodelo_bridge.core.penalties import _PEVERT, hydro_penalty_costs

        single = hydro_penalty_costs(
            rho_avg=1.0, rho_max_acum=2.0, penalid_costs={}, max_deficit_cost=100.0
        )
        double = hydro_penalty_costs(
            rho_avg=2.0, rho_max_acum=2.0, penalid_costs={}, max_deficit_cost=100.0
        )
        # spillage_cost = _PEVERT * rho_avg → doubles with rho_avg.
        assert single["spillage_cost"] == pytest.approx(_PEVERT * 1.0)
        assert double["spillage_cost"] == pytest.approx(2.0 * single["spillage_cost"])
        # water_withdrawal uses rho_max_acum (held fixed) → unchanged.
        assert double["water_withdrawal_violation_cost"] == pytest.approx(
            single["water_withdrawal_violation_cost"]
        )

    def test_water_withdrawal_scales_with_rho_max_acum(self) -> None:
        from novomodelo_bridge.core.penalties import hydro_penalty_costs

        low = hydro_penalty_costs(
            rho_avg=1.0, rho_max_acum=1.0, penalid_costs={}, max_deficit_cost=100.0
        )
        high = hydro_penalty_costs(
            rho_avg=1.0, rho_max_acum=3.0, penalid_costs={}, max_deficit_cost=100.0
        )
        assert high["water_withdrawal_violation_cost"] == pytest.approx(
            3.0 * low["water_withdrawal_violation_cost"]
        )
        # spillage (rho_avg only) is unaffected by rho_max_acum.
        assert high["spillage_cost"] == pytest.approx(low["spillage_cost"])

    def test_storage_floor_derived_from_deficit_at_evaporation_tier(self) -> None:
        from novomodelo_bridge.core.penalties import (
            _STORAGE_VIOLATION_DEFICIT_MULT,
            hydro_penalty_costs,
        )
        from novomodelo_bridge.core.units import HM3_TO_MWH_PER_RHO

        max_deficit_cost, rho_max_acum = 100.0, 2.0
        costs = hydro_penalty_costs(
            rho_avg=1.0,
            rho_max_acum=rho_max_acum,
            penalid_costs={},
            max_deficit_cost=max_deficit_cost,
        )
        # storage == 10 × max_deficit × rho_max_acum × 277.78 (energy-equivalent).
        expected = (
            _STORAGE_VIOLATION_DEFICIT_MULT
            * max_deficit_cost
            * rho_max_acum
            * HM3_TO_MWH_PER_RHO
        )
        assert costs["storage_violation_below_cost"] == pytest.approx(expected)
        # ... which is exactly the evaporation energy-equivalent.
        assert costs["storage_violation_below_cost"] == pytest.approx(
            costs["evaporation_violation_cost"] * HM3_TO_MWH_PER_RHO
        )

    def test_filling_target_derived_a_little_below_deficit(self) -> None:
        from novomodelo_bridge.core.penalties import (
            _FILLING_TARGET_DEFICIT_FRACTION,
            hydro_penalty_costs,
        )
        from novomodelo_bridge.core.units import HM3_TO_MWH_PER_RHO

        max_deficit_cost, rho_max_acum = 100.0, 2.0
        costs = hydro_penalty_costs(
            rho_avg=1.0,
            rho_max_acum=rho_max_acum,
            penalid_costs={},
            max_deficit_cost=max_deficit_cost,
        )
        expected = (
            _FILLING_TARGET_DEFICIT_FRACTION
            * max_deficit_cost
            * rho_max_acum
            * HM3_TO_MWH_PER_RHO
        )
        assert costs["filling_target_violation_cost"] == pytest.approx(expected)

    def test_storage_floor_is_the_largest_hydro_penalty(self) -> None:
        from novomodelo_bridge.core.penalties import hydro_penalty_costs
        from novomodelo_bridge.core.units import HM3_TO_MWH_PER_RHO

        costs = hydro_penalty_costs(
            rho_avg=1.0, rho_max_acum=2.0, penalid_costs={}, max_deficit_cost=100.0
        )
        storage = costs["storage_violation_below_cost"]
        # The storage floor is the strictest deterrent: it dominates every other
        # hydro penalty, including the evaporation energy-equivalent it is derived
        # from and the (cheaper) filling target.
        others = [v for k, v in costs.items() if k != "storage_violation_below_cost"]
        assert storage == pytest.approx(max(costs.values()))
        assert all(storage > v for v in others)
        assert storage > costs["evaporation_violation_cost"]
        assert storage > costs["filling_target_violation_cost"]
        # Sanity: the ordering above is the energy-equiv blow-up (×277.78), not a
        # coincidence of the chosen ρ.
        assert storage == pytest.approx(
            costs["evaporation_violation_cost"] * HM3_TO_MWH_PER_RHO
        )

    def test_storage_penalid_volmin_uses_rho_max_acum(self) -> None:
        from novomodelo_bridge.core.penalties import hydro_penalty_costs
        from novomodelo_bridge.core.units import HM3_TO_MWH_PER_RHO

        volmin_rate = 5.0
        penalid = {"VOLMIN": volmin_rate}
        # rho_avg deliberately != rho_max_acum so the two are distinguishable.
        costs = hydro_penalty_costs(
            rho_avg=1.0,
            rho_max_acum=2.0,
            penalid_costs=penalid,
            max_deficit_cost=100.0,
        )
        assert costs["storage_violation_below_cost"] == pytest.approx(
            volmin_rate * 2.0 * HM3_TO_MWH_PER_RHO
        )
        # Changing rho_avg alone must NOT move the storage floor (rho_max_acum-only).
        moved_rho_avg = hydro_penalty_costs(
            rho_avg=9.0,
            rho_max_acum=2.0,
            penalid_costs=penalid,
            max_deficit_cost=100.0,
        )
        assert moved_rho_avg["storage_violation_below_cost"] == pytest.approx(
            costs["storage_violation_below_cost"]
        )
        # Doubling rho_max_acum doubles the storage floor.
        doubled = hydro_penalty_costs(
            rho_avg=1.0,
            rho_max_acum=4.0,
            penalid_costs=penalid,
            max_deficit_cost=100.0,
        )
        assert doubled["storage_violation_below_cost"] == pytest.approx(
            2.0 * costs["storage_violation_below_cost"]
        )


class TestConvertHydroPenaltyOverrides:
    """Per-stage, SIN-uniform hydro penalty override parquet."""

    @patch(
        "novomodelo_bridge.newave.converters.network._read_penalid_costs",
        return_value={},
    )
    def test_sin_uniform_sparse_per_stage(self, _mock_penalid, tmp_path) -> None:
        from novomodelo_bridge.core.penalties import _PEVERT, hydro_penalty_costs
        from novomodelo_bridge.newave.converters.network import (
            convert_hydro_penalty_overrides,
        )

        case = make_case(tmp_path, sistema=_make_sistema_mock())
        # Build the base via the same helper the override diffs against, with
        # the mocked max_deficit_cost (max custo = 500*2 = 1000). Stage 1 then
        # uses exactly the base (ρ_avg=0.6, ρ_max_acum=2.0) → no override.
        base = hydro_penalty_costs(
            rho_avg=0.6, rho_max_acum=2.0, penalid_costs={}, max_deficit_cost=1000.0
        )
        table = convert_hydro_penalty_overrides(
            case,
            hydro_ids=[0, 1],
            base_hydro_penalties=base,
            per_stage_rho_avg=[0.5, 0.6, 0.55],
            per_stage_rho_max_acum=[2.0, 2.0, 2.0],
        )
        assert table is not None
        df = table.to_pandas()

        # Required key columns + only ρ-scaled columns that differ are present.
        assert {"hydro_id", "stage_id"}.issubset(df.columns)
        assert "generation_violation_below_cost" not in df.columns
        assert "filling_target_violation_cost" not in df.columns

        # Stage 1 matches the base exactly → no rows emitted for it (sparse).
        assert sorted(df["stage_id"].unique().tolist()) == [0, 2]

        # SIN-uniform: both hydros share one value per stage.
        s0 = df[df["stage_id"] == 0]
        assert s0["hydro_id"].tolist() == [0, 1]
        assert s0["spillage_cost"].nunique() == 1
        assert s0["spillage_cost"].iloc[0] == pytest.approx(_PEVERT * 0.5)

        # Output obeys the (hydro_id, stage_id) ordering contract.
        ordered = df.sort_values(["hydro_id", "stage_id"]).reset_index(drop=True)
        assert df.reset_index(drop=True).equals(ordered)

    @patch(
        "novomodelo_bridge.newave.converters.network._read_penalid_costs",
        return_value={},
    )
    def test_returns_none_when_no_stage_differs(self, _mock_penalid, tmp_path) -> None:
        from novomodelo_bridge.core.penalties import hydro_penalty_costs
        from novomodelo_bridge.newave.converters.network import (
            convert_hydro_penalty_overrides,
        )

        case = make_case(tmp_path, sistema=_make_sistema_mock())
        # max_deficit_cost from the mocked deficit df (max custo = 500*2 = 1000).
        base = hydro_penalty_costs(
            rho_avg=0.6, rho_max_acum=2.0, penalid_costs={}, max_deficit_cost=1000.0
        )
        # Every stage uses exactly the base ρ → fully sparse → None.
        table = convert_hydro_penalty_overrides(
            case,
            hydro_ids=[0, 1],
            base_hydro_penalties=base,
            per_stage_rho_avg=[0.6, 0.6],
            per_stage_rho_max_acum=[2.0, 2.0],
        )
        assert table is None
