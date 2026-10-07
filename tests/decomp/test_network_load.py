"""Tests for the DECOMP bus and load converters."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from novomodelo_bridge.decomp.converters.network import convert_buses, convert_lines
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.load import convert_load_factors, convert_load_stats
from novomodelo_bridge.decomp.temporal import OperativeStage, build_operative_calendar
from tests.conftest import make_decomp_case

_ID_MAP = DecompIdMap(
    bus_codes=(1, 2, 3, 4, 11),
    bus_names=("SE", "S", "NE", "N", "FC"),
)


def _calendar_rv3():
    hours = [[15.0, 64.0, 89.0]] * 2 + [[63.0, 280.0, 401.0]]
    return build_operative_calendar(date(2026, 7, 18), hours)


def _single_stage_calendar(start_date: date) -> list[OperativeStage]:
    """A minimal one-stage calendar carrying only *start_date* — for tests
    that assert on ``operational_start_date`` without needing a real
    DECOMP-shaped (weekly + aggregated-month) calendar."""
    return [
        OperativeStage(
            index=0,
            start_date=start_date,
            end_date=start_date,
            season_id=start_date.month - 1,
            block_hours=(1.0,),
        )
    ]


class _StubDadger:
    """Minimal Dadger stand-in carrying sb/cd/dp/ia DataFrames."""

    def __init__(
        self,
        sb: pd.DataFrame | None = None,
        cd: pd.DataFrame | None = None,
        dp: pd.DataFrame | None = None,
        ia: pd.DataFrame | None = None,
    ) -> None:
        self._sb, self._cd, self._dp, self._ia = sb, cd, dp, ia

    def sb(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._sb

    def cd(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._cd

    def dp(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._dp

    def ia(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._ia


def _cd_frame(
    cost: float = 7810.62,
    limit: float = 100.0,
    codes: tuple[int, ...] = (1, 2, 3, 4),
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "codigo_curva": 1,
                "codigo_submercado": code,
                "estagio": 1,
                "nome_curva": "1PDEF",
                "custo_1": cost,
                "custo_2": cost,
                "custo_3": cost,
                "limite_superior_1": limit,
                "limite_superior_2": limit,
                "limite_superior_3": limit,
            }
            for code in codes
        ]
    )


def _dp_frame(calendar_stages: int = 3) -> pd.DataFrame:
    rows = []
    weekly = {"duracao_1": 15.0, "duracao_2": 64.0, "duracao_3": 89.0}
    monthly = {"duracao_1": 63.0, "duracao_2": 280.0, "duracao_3": 401.0}
    for estagio in range(1, calendar_stages + 1):
        durations = monthly if estagio == calendar_stages else weekly
        for sbm, base in ((1, 40000.0), (2, 15000.0), (11, None)):
            rows.append(
                {
                    "codigo_submercado": sbm,
                    "estagio": estagio,
                    "numero_patamares": 3,
                    "carga_1": None if base is None else base * 1.2,
                    "carga_2": None if base is None else base,
                    "carga_3": None if base is None else base * 0.8,
                    **durations,
                }
            )
    return pd.DataFrame(rows)


class TestDecompIdMap:
    def test_deterministic_ids_and_transhipment(self) -> None:
        assert _ID_MAP.bus_id(1) == 0
        assert _ID_MAP.bus_id(11) == 4
        assert _ID_MAP.transhipment_bus_id == 5
        assert _ID_MAP.n_buses == 6
        assert _ID_MAP.bus_id_by_name("SE") == 0
        assert _ID_MAP.bus_id_by_name("IV") == 5
        assert _ID_MAP.bus_name(5) == "IV"

    def test_unknown_code_raises(self) -> None:
        with pytest.raises(KeyError, match="99"):
            _ID_MAP.bus_id(99)

    def test_reserved_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="reserved"):
            DecompIdMap(bus_codes=(1, 2), bus_names=("SE", "IV"))


class TestConvertBuses:
    def test_buses_with_and_without_deficit(self) -> None:
        dadger = _StubDadger(cd=_cd_frame())
        case = make_decomp_case(
            Path("unused"),
            dadger=dadger,
            calendar=_single_stage_calendar(date(2024, 8, 31)),
        )
        doc = convert_buses(case, _ID_MAP)
        buses = doc["buses"]
        assert [b["id"] for b in buses] == [0, 1, 2, 3, 4, 5]
        assert buses[0]["deficit_segments"] == [{"depth_mw": None, "cost": 7810.62}]
        assert "deficit_segments" not in buses[4]  # FC
        assert buses[5]["name"] == "IV"
        assert "deficit_segments" not in buses[5]
        assert buses[0]["operational_start_date"] == "2024-08-31"

    def test_rejects_non_full_depth(self) -> None:
        dadger = _StubDadger(cd=_cd_frame(limit=95.0))
        case = make_decomp_case(
            Path("unused"),
            dadger=dadger,
            calendar=_single_stage_calendar(date(2024, 8, 31)),
        )
        with pytest.raises(ValueError, match="depth"):
            convert_buses(case, _ID_MAP)

    def test_rejects_block_varying_cost(self) -> None:
        cd = _cd_frame()
        cd.loc[0, "custo_2"] = 9000.0
        case = make_decomp_case(
            Path("unused"),
            dadger=_StubDadger(cd=cd),
            calendar=_single_stage_calendar(date(2024, 8, 31)),
        )
        with pytest.raises(ValueError, match="uniform"):
            convert_buses(case, _ID_MAP)

    def test_rejects_multi_segment_curves(self) -> None:
        cd = pd.concat([_cd_frame(), _cd_frame(cost=9000.0)], ignore_index=True)
        cd.loc[cd.index[-4:], "codigo_curva"] = 2
        case = make_decomp_case(
            Path("unused"),
            dadger=_StubDadger(cd=cd),
            calendar=_single_stage_calendar(date(2024, 8, 31)),
        )
        with pytest.raises(ValueError, match="multi-segment"):
            convert_buses(case, _ID_MAP)


class TestConvertLoad:
    def test_stats_cover_every_bus_and_stage(self) -> None:
        case = make_decomp_case(
            Path("unused"), dadger=_StubDadger(dp=_dp_frame()), calendar=_calendar_rv3()
        )
        table = convert_load_stats(case, _ID_MAP)
        df = table.to_pandas()
        assert len(df) == 6 * 3
        assert set(df["std_mw"]) == {0.0}
        # SE stage 0: (48000·15 + 40000·64 + 32000·89)/168
        se = df[(df["bus_id"] == 0) & (df["stage_id"] == 0)]["mean_mw"].iloc[0]
        assert se == pytest.approx((48000 * 15 + 40000 * 64 + 32000 * 89) / 168)
        # FC (NaN cargas) and IV (absent from DP) carry zero load.
        assert set(df[df["bus_id"].isin([4, 5])]["mean_mw"]) == {0.0}

    def test_factors_invariant_and_zero_mean_omission(self) -> None:
        calendar = _calendar_rv3()
        case = make_decomp_case(
            Path("unused"), dadger=_StubDadger(dp=_dp_frame()), calendar=calendar
        )
        doc = convert_load_factors(case, _ID_MAP)
        entries = doc["load_factors"]
        assert {e["bus_id"] for e in entries} == {0, 1}
        for entry in entries:
            stage = calendar[entry["stage_id"]]
            weighted = sum(
                bf["factor"] * stage.block_hours[bf["block_id"]]
                for bf in entry["block_factors"]
            )
            assert weighted == pytest.approx(stage.total_hours, rel=1e-12)

    def test_rejects_block_count_mismatch(self) -> None:
        dp = _dp_frame()
        dp.loc[0, "numero_patamares"] = 2
        case = make_decomp_case(
            Path("unused"), dadger=_StubDadger(dp=dp), calendar=_calendar_rv3()
        )
        with pytest.raises(ValueError, match="blocks"):
            convert_load_stats(case, _ID_MAP)

    def test_rejects_stage_outside_calendar(self) -> None:
        dp = _dp_frame(calendar_stages=4)
        case = make_decomp_case(
            Path("unused"), dadger=_StubDadger(dp=dp), calendar=_calendar_rv3()
        )
        with pytest.raises(ValueError, match="outside the calendar"):
            convert_load_stats(case, _ID_MAP)

    def test_extra_bus_loads_add_to_existing_dp_row(self) -> None:
        """``extra_bus_loads`` on a bus ``DP`` already declares (Itaipu's SE
        bus, for ``carga_ande``) sums element-wise onto the ``DP`` row
        rather than replacing it -- a regression to a plain ``dict.update``
        merge would silently erase SE's declared demand."""
        calendar = _calendar_rv3()
        case = make_decomp_case(
            Path("unused"), dadger=_StubDadger(dp=_dp_frame()), calendar=calendar
        )
        extra = {0: [500.0, 400.0, 300.0], 1: [10.0, 10.0, 10.0], 2: [0.0, 0.0, 0.0]}
        extra_bus_loads = {(0, stage): values for stage, values in extra.items()}

        dp_only = convert_load_stats(case, _ID_MAP).to_pandas()
        combined = convert_load_stats(
            case, _ID_MAP, extra_bus_loads=extra_bus_loads
        ).to_pandas()

        dp_se = dp_only[dp_only["bus_id"] == 0].set_index("stage_id")["mean_mw"]
        combined_se = combined[combined["bus_id"] == 0].set_index("stage_id")["mean_mw"]
        for stage in calendar:
            expected_extra = (
                sum(
                    v * h
                    for v, h in zip(extra[stage.index], stage.block_hours, strict=True)
                )
                / stage.total_hours
            )
            assert combined_se.loc[stage.index] == pytest.approx(
                dp_se.loc[stage.index] + expected_extra
            )

        # Every other bus's row is untouched by the SE-keyed addition.
        pd.testing.assert_frame_equal(
            combined[combined["bus_id"] != 0].reset_index(drop=True),
            dp_only[dp_only["bus_id"] != 0].reset_index(drop=True),
        )

    def test_extra_bus_loads_insert_for_bus_absent_from_dp(self) -> None:
        """A bus ``DP`` never declares (the ``IV`` transhipment bus) has no
        row to collide with, so ``extra_bus_loads`` is simply inserted --
        the backward-compatible shape for a deck with no colliding demand."""
        calendar = _calendar_rv3()
        case = make_decomp_case(
            Path("unused"), dadger=_StubDadger(dp=_dp_frame()), calendar=calendar
        )
        iv_bus = _ID_MAP.transhipment_bus_id
        extra = {0: [50.0, 40.0, 30.0], 1: [50.0, 40.0, 30.0], 2: [50.0, 40.0, 30.0]}
        extra_bus_loads = {(iv_bus, stage): values for stage, values in extra.items()}

        combined = convert_load_stats(
            case, _ID_MAP, extra_bus_loads=extra_bus_loads
        ).to_pandas()
        iv_rows = combined[combined["bus_id"] == iv_bus]
        assert (iv_rows["mean_mw"] > 0).all()

    def test_extra_bus_loads_reject_block_count_mismatch(self) -> None:
        """A colliding key whose ``extra_bus_loads`` block count disagrees
        with ``DP``'s own for that (bus, stage) fails loud instead of
        silently zipping a truncated/padded sum."""
        calendar = _calendar_rv3()
        case = make_decomp_case(
            Path("unused"), dadger=_StubDadger(dp=_dp_frame()), calendar=calendar
        )
        extra_bus_loads = {(0, 0): [500.0, 400.0]}  # 2 values vs DP's 3

        with pytest.raises(ValueError, match="block"):
            convert_load_stats(case, _ID_MAP, extra_bus_loads=extra_bus_loads)


def _ia_zero_block_frame() -> pd.DataFrame:
    """One SE-IV exchange row whose block-0 ``de_para`` limit sits at
    exactly zero and whose other blocks are non-zero and distinct from the
    base -- the shape the deleted multiplicative-factor encoding could
    never represent (factors are strictly positive)."""
    return pd.DataFrame(
        [
            {
                "estagio": 1,
                "nome_submercado_de": "SE",
                "nome_submercado_para": "IV",
                "limite_de_para_1": 0.0,
                "limite_de_para_2": 5000.0,
                "limite_de_para_3": 6000.0,
                "limite_para_de_1": 9000.0,
                "limite_para_de_2": 9000.0,
                "limite_para_de_3": 9000.0,
            }
        ]
    )


class TestConvertLinesZeroCapability:
    """novomodelo decision 10 makes ``direct_mw = 0.0`` an ordinary bound; the
    ``raise`` this replaced only ever fired on a zero-limit round-trip that
    no longer exists (see the capability-gain comment at ``convert_lines``,
    ``decomp/network.py``). No current deck exercises a zero IA limit
    (measured: zero zeros in both production decks' ``IA`` records), so
    this synthetic fixture pins the new capability."""

    def test_zero_block_limit_converts_without_raising(self) -> None:
        calendar = _calendar_rv3()
        case = make_decomp_case(
            Path("unused"),
            dadger=_StubDadger(ia=_ia_zero_block_frame()),
            calendar=calendar,
        )
        lines_doc, bounds = convert_lines(case, _ID_MAP)  # must not raise
        line_id = lines_doc["lines"][0]["id"]
        df = bounds.to_pandas()

        zero_row = df[
            (df["line_id"] == line_id) & (df["stage_id"] == 0) & (df["block_id"] == 0)
        ]
        assert len(zero_row) == 1
        assert zero_row.iloc[0]["direct_mw"] == 0.0
        assert zero_row.iloc[0]["reverse_mw"] == 9000.0

        # Block 1 is non-zero and distinct from the base (6000.0), proving
        # it is genuinely unaffected rather than coincidentally matching it.
        other_block = df[
            (df["line_id"] == line_id) & (df["stage_id"] == 0) & (df["block_id"] == 1)
        ]
        assert len(other_block) == 1
        assert other_block.iloc[0]["direct_mw"] == 5000.0
        assert other_block.iloc[0]["reverse_mw"] == 9000.0
