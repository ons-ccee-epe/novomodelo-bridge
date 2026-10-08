"""Tests for the DECOMP ``IV`` transshipment node wiring.

Covers the converter-created SE<->IV line, which carries Itaipu's 60 Hz
group's output (``network.append_iv_se_line`` /
``network._itaipu_60hz_capacity_mw``), and Itaipu's own SE bus's
``carga_ande`` addition (the ``extra_bus_loads`` parameter on
``load.convert_load_stats`` / ``convert_load_factors``, netted ADDITIVELY
onto SE's real ``DP`` demand -- never replacing it).

Stub-deck, tier-1 tests only: every fixture below is hand-built, mirroring
the exact column shape idecomp's ``df=True`` accessors expand (mirroring
the ``_StubDadger``/``_ri`` fixtures in
``test_libs_electrical.py``). These exercise the functions
``pipeline.py`` orchestrates with exactly the arguments it computes for
each of Itaipu's three deck shapes (with ``RI``, without ``RI``, no Itaipu
at all); a full ``convert_decomp_case`` run is out of scope here.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.converters.libs_electrical import read_carga_ande
from novomodelo_bridge.decomp.converters.network import (
    _LINE_BOUNDS_SCHEMA,
    _UNBOUNDED_LINE_CAPACITY_MW,
    _itaipu_60hz_capacity_mw,
    append_iv_se_line,
    convert_lines,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.load import convert_load_factors, convert_load_stats
from novomodelo_bridge.decomp.temporal import OperativeStage, build_operative_calendar
from tests.conftest import make_decomp_case

_ITAIPU_CODE = 66

_ID_MAP_ITAIPU = DecompIdMap(
    bus_codes=(1, 2), bus_names=("SE", "S"), hydro_codes=(_ITAIPU_CODE,)
)
_ID_MAP_NO_ITAIPU = DecompIdMap(bus_codes=(1, 2), bus_names=("SE", "S"))


def _calendar() -> list[OperativeStage]:
    hours = [[15.0, 64.0, 89.0]] * 2 + [[63.0, 280.0, 401.0]]
    return build_operative_calendar(date(2026, 7, 18), hours)


def _case(
    dadger: object = None, calendar: list[OperativeStage] | None = None
) -> DecompCase:
    """A ``DecompCase`` pre-filled with *dadger* (unused by
    ``append_iv_se_line``, which never reads it) and *calendar* (defaults to
    ``_calendar()``, matching the raw ``calendar``/``start`` every test here
    used to pass directly)."""
    return make_decomp_case(
        Path("unused"), dadger=dadger, calendar=calendar or _calendar()
    )


class _StubDadger:
    """Minimal Dadger stand-in carrying ``ia``/``ri``/``dp`` DataFrames --
    the only accessors the functions under test read."""

    def __init__(
        self,
        ia: pd.DataFrame | None = None,
        ri: pd.DataFrame | None = None,
        dp: pd.DataFrame | None = None,
    ) -> None:
        self._ia, self._ri, self._dp = ia, ri, dp

    def ia(self, df: bool = True) -> pd.DataFrame | None:  # noqa: ARG002
        return self._ia

    def ri(self, df: bool = True) -> pd.DataFrame | None:  # noqa: ARG002
        return self._ri

    def dp(self, df: bool = True) -> pd.DataFrame | None:  # noqa: ARG002
        return self._dp


def _ia_frame() -> pd.DataFrame:
    """One SE->S exchange pair, uniform across every block (no per-block
    override rows), declared at estágio 1 and forward-filled."""
    return pd.DataFrame(
        [
            {
                "estagio": 1,
                "nome_submercado_de": "SE",
                "nome_submercado_para": "S",
                "limite_de_para_1": 500.0,
                "limite_de_para_2": 500.0,
                "limite_de_para_3": 500.0,
                "limite_para_de_1": 400.0,
                "limite_para_de_2": 400.0,
                "limite_para_de_3": 400.0,
            }
        ]
    )


def _ri_frame() -> pd.DataFrame:
    """Three ``RI`` rows (one per stage) carrying both
    ``geracao_maxima_60_hz`` and ``carga_ande`` per-patamar values -- the
    two column families ``_itaipu_60hz_capacity_mw``/``read_carga_ande``
    each read independently. The largest declared ``geracao_maxima_60_hz``
    is 3700.0 (stage 3, patamar 1)."""
    per_stage = [
        (3500.0, 3600.0, 3550.0, 1200.0, 1100.0, 1150.0),
        (3400.0, 3300.0, 3450.0, 1300.0, 1250.0, 1280.0),
        (3700.0, 3550.0, 3600.0, 1150.0, 1400.0, 1300.0),
    ]
    rows = [
        {
            "codigo_usina": _ITAIPU_CODE,
            "estagio": estagio,
            "codigo_submercado": 1,
            "geracao_maxima_60_hz_1": g1,
            "geracao_maxima_60_hz_2": g2,
            "geracao_maxima_60_hz_3": g3,
            "carga_ande_1": c1,
            "carga_ande_2": c2,
            "carga_ande_3": c3,
        }
        for estagio, (g1, g2, g3, c1, c2, c3) in enumerate(per_stage, start=1)
    ]
    return pd.DataFrame(rows)


def _dp_frame() -> pd.DataFrame:
    """``DP`` loads for SE (1) and S (2) across all 3 stages, 3 patamares
    each -- SE's own row is what the ``carga_ande`` addition must land on
    top of, never replace (the ``IV`` bus, absent here, carries no load of
    its own)."""
    rows = []
    for estagio in (1, 2, 3):
        rows.append(
            {
                "codigo_submercado": 1,
                "estagio": estagio,
                "numero_patamares": 3,
                "carga_1": 1000.0,
                "carga_2": 900.0,
                "carga_3": 950.0,
            }
        )
        rows.append(
            {
                "codigo_submercado": 2,
                "estagio": estagio,
                "numero_patamares": 3,
                "carga_1": 400.0,
                "carga_2": 380.0,
                "carga_3": 390.0,
            }
        )
    return pd.DataFrame(rows)


class TestAppendIvSeLine:
    """The append helper's own shape contract, independent of Itaipu
    detection (its caller's job) -- mirrors ``convert_lines``'s ``lines.json``
    entry and stage-level ``line_bounds`` base-row shape."""

    def test_appends_after_existing_lines_without_mutating_them(self) -> None:
        calendar = _calendar()
        dadger = _StubDadger(ia=_ia_frame())
        case = _case(dadger, calendar)
        lines_doc, line_bounds = convert_lines(case, _ID_MAP_ITAIPU)
        assert len(lines_doc["lines"]) == 1
        original_line = dict(lines_doc["lines"][0])
        original_row_count = line_bounds.num_rows

        extended_doc, extended_bounds = append_iv_se_line(
            case,
            lines_doc=lines_doc,
            line_bounds=line_bounds,
            source_bus_id=_ID_MAP_ITAIPU.transhipment_bus_id,
            target_bus_id=_ID_MAP_ITAIPU.bus_id(1),
            capacity_mw=3700.0,
        )

        # The original doc/list is never mutated in place -- the existing
        # IA line keeps its id (build_fi_line_map stays stable).
        assert lines_doc["lines"] == [original_line]
        assert [line["id"] for line in extended_doc["lines"]] == [0, 1]
        assert extended_doc["lines"][0] == original_line

        new_line = extended_doc["lines"][1]
        assert new_line["name"] == "IV-SE"
        assert new_line["source_bus_id"] == _ID_MAP_ITAIPU.transhipment_bus_id
        assert new_line["target_bus_id"] == _ID_MAP_ITAIPU.bus_id(1)
        assert new_line["capacity"] == {"direct_mw": 3700.0, "reverse_mw": 3700.0}

        assert extended_bounds.num_rows == original_row_count + len(calendar)
        df = extended_bounds.to_pandas()
        new_rows = df[df["line_id"] == 1]
        assert len(new_rows) == len(calendar)
        assert set(new_rows["stage_id"]) == {s.index for s in calendar}
        assert new_rows["block_id"].isna().all()
        assert (new_rows["direct_mw"] == 3700.0).all()
        assert (new_rows["reverse_mw"] == 3700.0).all()

    def test_next_free_id_with_no_existing_lines(self) -> None:
        calendar = _calendar()
        empty_doc = {"$schema": "irrelevant", "lines": []}

        extended_doc, extended_bounds = append_iv_se_line(
            _case(calendar=calendar),
            lines_doc=empty_doc,
            line_bounds=_LINE_BOUNDS_SCHEMA.empty_table(),
            source_bus_id=2,
            target_bus_id=0,
            capacity_mw=99999.0,
        )

        assert extended_doc["lines"][0]["id"] == 0
        assert extended_bounds.num_rows == len(calendar)

    def test_dedup_skip_when_pair_already_wired(self) -> None:
        """The deck's own ``IA`` register already connects the pair
        (either orientation) -- the call is a silent no-op, returning the
        docs unchanged with no duplicate line and no diagnostic (a line here
        is expected, so the reuse needs no announcement)."""
        calendar = _calendar()
        start = calendar[0].start_date
        existing_line = {
            "id": 0,
            "name": "SE-IV",
            "operational_start_date": start.isoformat(),
            "source_bus_id": 0,
            "target_bus_id": 5,
            "capacity": {"direct_mw": 4000.0, "reverse_mw": 3500.0},
        }
        lines_doc = {"$schema": "irrelevant", "lines": [existing_line]}
        line_bounds = _LINE_BOUNDS_SCHEMA.empty_table()

        with dx.collect() as collected:
            result_doc, result_bounds = append_iv_se_line(
                _case(calendar=calendar),
                lines_doc=lines_doc,
                line_bounds=line_bounds,
                # Reversed orientation vs. the existing line -- the deck's
                # IA register may declare either direction.
                source_bus_id=5,
                target_bus_id=0,
                capacity_mw=99999.0,
            )

        assert result_doc is lines_doc
        assert result_bounds is line_bounds
        assert len(result_doc["lines"]) == 1
        assert collected == []

    def test_islanded_synthesize_emits_no_info(self) -> None:
        """No existing line between the pair -- the line is
        synthesized as before (prior behavior) and no diagnostic is
        emitted."""
        calendar = _calendar()
        lines_doc = {"$schema": "irrelevant", "lines": []}
        line_bounds = _LINE_BOUNDS_SCHEMA.empty_table()

        with dx.collect() as collected:
            extended_doc, extended_bounds = append_iv_se_line(
                _case(calendar=calendar),
                lines_doc=lines_doc,
                line_bounds=line_bounds,
                source_bus_id=5,
                target_bus_id=0,
                capacity_mw=99999.0,
            )

        assert len(extended_doc["lines"]) == 1
        assert extended_doc["lines"][0]["id"] == 0
        assert extended_doc["lines"][0]["name"] == "IV-SE"
        assert extended_bounds.num_rows == len(calendar)
        assert collected == []


class TestItaipu60HzCapacity:
    def test_max_over_every_stage_and_patamar(self) -> None:
        dadger = _StubDadger(ri=_ri_frame())
        assert _itaipu_60hz_capacity_mw(dadger) == 3700.0

    def test_unbounded_sentinel_when_no_ri_register(self) -> None:
        dadger = _StubDadger(ri=None)
        assert _itaipu_60hz_capacity_mw(dadger) == _UNBOUNDED_LINE_CAPACITY_MW


class TestItaipuWithRi:
    """AC 1+2: Itaipu (66) operated with an ``RI`` register -- the IV-SE
    line (sized off the 60 Hz group's own generation ceiling) plus Itaipu's
    own SE bus's ``carga_ande`` addition, both computed exactly the way
    ``pipeline.py`` assembles them."""

    def _build(self) -> tuple[list[OperativeStage], date, _StubDadger]:
        calendar = _calendar()
        start = calendar[0].start_date
        dadger = _StubDadger(ia=_ia_frame(), ri=_ri_frame(), dp=_dp_frame())
        return calendar, start, dadger

    def test_line_append_and_base_bound_row(self) -> None:
        calendar, _start, dadger = self._build()
        case = _case(dadger, calendar)
        lines_doc, line_bounds = convert_lines(case, _ID_MAP_ITAIPU)
        capacity = _itaipu_60hz_capacity_mw(dadger)
        assert capacity == 3700.0

        lines_doc, line_bounds = append_iv_se_line(
            case,
            lines_doc=lines_doc,
            line_bounds=line_bounds,
            source_bus_id=_ID_MAP_ITAIPU.transhipment_bus_id,
            target_bus_id=_ID_MAP_ITAIPU.bus_id(1),
            capacity_mw=capacity,
        )

        iv_se = next(line for line in lines_doc["lines"] if line["name"] == "IV-SE")
        assert iv_se["id"] == 1  # next free id after the one IA line
        assert iv_se["source_bus_id"] == _ID_MAP_ITAIPU.transhipment_bus_id
        assert iv_se["target_bus_id"] == _ID_MAP_ITAIPU.bus_id(1)

        df = line_bounds.to_pandas()
        base_rows = df[(df["line_id"] == 1) & df["block_id"].isna()]
        assert len(base_rows) == len(calendar)
        assert (base_rows["direct_mw"] == 3700.0).all()
        assert (base_rows["reverse_mw"] == 3700.0).all()

    def test_ande_nets_additively_onto_se_bus(self) -> None:
        """The corrected model: ``carga_ande`` keys Itaipu's own SE bus and
        must ADD to SE's real ``DP`` demand, never replace it -- a
        regression to a plain ``dict.update``/replace merge would silently
        erase SE's declared load, so both the preserved DP component and the
        added ANDE component are asserted here."""
        calendar, _start, dadger = self._build()
        carga_ande = read_carga_ande(dadger, calendar)
        assert carga_ande  # RI present -> non-empty
        se_bus = _ID_MAP_ITAIPU.bus_id(1)
        extra_bus_loads = {
            (se_bus, stage): values for stage, values in carga_ande.items()
        }

        case = _case(dadger, calendar)
        dp_only = convert_load_stats(case, _ID_MAP_ITAIPU).to_pandas()
        with_ande = convert_load_stats(
            case, _ID_MAP_ITAIPU, extra_bus_loads=extra_bus_loads
        ).to_pandas()

        dp_se = dp_only[dp_only["bus_id"] == se_bus].set_index("stage_id")["mean_mw"]
        combined_se = with_ande[with_ande["bus_id"] == se_bus].set_index("stage_id")[
            "mean_mw"
        ]
        assert (dp_se > 0).all()  # SE's real DP demand is non-trivial
        assert (combined_se > dp_se).all()  # ANDE strictly adds on top of it
        for stage in calendar:
            expected_ande = (
                sum(
                    v * h
                    for v, h in zip(
                        carga_ande[stage.index], stage.block_hours, strict=True
                    )
                )
                / stage.total_hours
            )
            assert combined_se.loc[stage.index] == pytest.approx(
                dp_se.loc[stage.index] + expected_ande
            )

        # The IV bus itself carries no load under the corrected model.
        iv_bus = _ID_MAP_ITAIPU.transhipment_bus_id
        assert set(with_ande[with_ande["bus_id"] == iv_bus]["mean_mw"]) == {0.0}

    def test_se_factors_entry_satisfies_hours_invariant(self) -> None:
        calendar, _start, dadger = self._build()
        carga_ande = read_carga_ande(dadger, calendar)
        se_bus = _ID_MAP_ITAIPU.bus_id(1)
        extra_bus_loads = {
            (se_bus, stage): values for stage, values in carga_ande.items()
        }

        doc = convert_load_factors(
            _case(dadger, calendar), _ID_MAP_ITAIPU, extra_bus_loads=extra_bus_loads
        )
        se_entries = [e for e in doc["load_factors"] if e["bus_id"] == se_bus]
        assert len(se_entries) == len(calendar)
        for entry in se_entries:
            stage = calendar[entry["stage_id"]]
            weighted = sum(
                bf["factor"] * stage.block_hours[bf["block_id"]]
                for bf in entry["block_factors"]
            )
            assert weighted == pytest.approx(stage.total_hours, rel=1e-12)

        iv_bus = _ID_MAP_ITAIPU.transhipment_bus_id
        assert all(e["bus_id"] != iv_bus for e in doc["load_factors"])


class TestItaipuNoRi:
    """AC 3: Itaipu operated but no ``RI`` register -- the line is still
    present (sized to the unbounded sentinel), but SE carries only its own
    ``DP`` demand (``read_carga_ande`` returns ``{}``, so ``pipeline.py``
    builds no ``extra_bus_loads``)."""

    def _build(self) -> tuple[list[OperativeStage], date, _StubDadger]:
        calendar = _calendar()
        start = calendar[0].start_date
        dadger = _StubDadger(ia=_ia_frame(), ri=None, dp=_dp_frame())
        return calendar, start, dadger

    def test_line_present_with_unbounded_capacity(self) -> None:
        calendar, _start, dadger = self._build()
        case = _case(dadger, calendar)
        lines_doc, line_bounds = convert_lines(case, _ID_MAP_ITAIPU)
        capacity = _itaipu_60hz_capacity_mw(dadger)
        assert capacity == _UNBOUNDED_LINE_CAPACITY_MW

        lines_doc, _line_bounds = append_iv_se_line(
            case,
            lines_doc=lines_doc,
            line_bounds=line_bounds,
            source_bus_id=_ID_MAP_ITAIPU.transhipment_bus_id,
            target_bus_id=_ID_MAP_ITAIPU.bus_id(1),
            capacity_mw=capacity,
        )
        iv_se = next(line for line in lines_doc["lines"] if line["name"] == "IV-SE")
        assert iv_se["capacity"] == {
            "direct_mw": _UNBOUNDED_LINE_CAPACITY_MW,
            "reverse_mw": _UNBOUNDED_LINE_CAPACITY_MW,
        }

    def test_se_load_stays_at_dp_only(self) -> None:
        calendar, _start, dadger = self._build()
        carga_ande = read_carga_ande(dadger, calendar)
        assert carga_ande == {}  # no RI -> pipeline.py builds no extra_bus_loads

        case = _case(dadger, calendar)
        dp_only = convert_load_stats(case, _ID_MAP_ITAIPU).to_pandas()
        with_none = convert_load_stats(
            case, _ID_MAP_ITAIPU, extra_bus_loads=None
        ).to_pandas()
        se_bus = _ID_MAP_ITAIPU.bus_id(1)
        pd.testing.assert_frame_equal(
            dp_only[dp_only["bus_id"] == se_bus].reset_index(drop=True),
            with_none[with_none["bus_id"] == se_bus].reset_index(drop=True),
        )

        iv_bus = _ID_MAP_ITAIPU.transhipment_bus_id
        assert set(with_none[with_none["bus_id"] == iv_bus]["mean_mw"]) == {0.0}

        doc = convert_load_factors(case, _ID_MAP_ITAIPU, extra_bus_loads=None)
        assert all(e["bus_id"] != iv_bus for e in doc["load_factors"])


class TestNoItaipu:
    """AC 4: a deck that never operates Itaipu (66) -- no IV-SE line, and
    the ``IV`` bus load stays zero, exactly like today."""

    def test_itaipu_operated_gate_is_false(self) -> None:
        assert _ITAIPU_CODE not in _ID_MAP_NO_ITAIPU.hydro_codes

    def test_no_iv_se_line(self) -> None:
        calendar = _calendar()
        dadger = _StubDadger(ia=_ia_frame())
        lines_doc, _line_bounds = convert_lines(
            _case(dadger, calendar), _ID_MAP_NO_ITAIPU
        )
        assert len(lines_doc["lines"]) == 1
        assert all(line["name"] != "IV-SE" for line in lines_doc["lines"])

    def test_iv_load_stays_zero(self) -> None:
        calendar = _calendar()
        dadger = _StubDadger(dp=_dp_frame())
        stats = convert_load_stats(
            _case(dadger, calendar), _ID_MAP_NO_ITAIPU, extra_bus_loads=None
        ).to_pandas()
        iv_bus = _ID_MAP_NO_ITAIPU.transhipment_bus_id
        assert set(stats[stats["bus_id"] == iv_bus]["mean_mw"]) == {0.0}

        doc = convert_load_factors(
            _case(dadger, calendar), _ID_MAP_NO_ITAIPU, extra_bus_loads=None
        )
        assert all(e["bus_id"] != iv_bus for e in doc["load_factors"])
