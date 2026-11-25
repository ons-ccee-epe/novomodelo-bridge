"""Tests for the tailrace lateral flows of a DECOMP deck: the resolver over its VL,
VU and VA records, and their emission in the converted hydros.

The decomp_mini deck pins the converted shape of a real deck; synthetic
register doubles pin each resolution rule.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from novomodelo_bridge.core.diagnostics import Diagnostic, Severity, collect
from novomodelo_bridge.core.errors import FieldParseError
from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.converters.cadastro import (
    EffectiveCadastro,
    build_effective_cadastro,
)
from novomodelo_bridge.decomp.converters.hydro import convert_hydros
from novomodelo_bridge.decomp.converters.tailrace_flow import (
    ResolvedTailraceFlow,
    TailraceFlowEntry,
    TailraceFlowResolution,
    TailraceRegisters,
    convert_tailrace_flow,
    piecewise_tailrace_codes,
    resolve_tailrace_flow,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.pipeline import convert_decomp_case
from tests.conftest import _FakeDadger, make_decomp_case
from tests.decomp.conftest import flat_turbine_limits

_LOGGER = "novomodelo_bridge.decomp.converters.tailrace_flow"
_VERTJU = "influencia_vertimento_canal_fuga"
_COEFFICIENTS = [f"coeficiente_a{i}" for i in range(5)]


def _effective(
    plants: Sequence[int],
    *,
    gauges: Mapping[int, int] | None = None,
    no_spill: Collection[int] = (),
    inflow_gauges: Mapping[int, tuple[int, ...]] | None = None,
    stage_varying: Mapping[tuple[int, str], tuple[float, ...]] | None = None,
) -> EffectiveCadastro:
    posto = {code: 100 + code for code in plants} | dict(gauges or {})
    base = pd.DataFrame(
        {
            "nome_usina": [f"Plant{code}  " for code in plants],
            "posto": [posto[code] for code in plants],
            _VERTJU: [0 if code in no_spill else 1 for code in plants],
        },
        index=pd.Index(plants, name="codigo_usina"),
    )
    return EffectiveCadastro(
        base=base,
        n_stages=2,
        stage_varying=dict(stage_varying or {}),
        inflow_gauges=dict(inflow_gauges or {}),
    )


def _vl(*rows: tuple[int, float] | tuple[int, float, Sequence[float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "codigo_usina_influenciada": row[0],
                "fator_impacto_defluencia": row[1],
                **dict(
                    zip(
                        _COEFFICIENTS,
                        row[2] if len(row) == 3 else [None] * 5,
                        strict=True,
                    )
                ),
            }
            for row in rows
        ]
    )


def _vu(*rows: tuple[int, int, float]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "codigo_usina_influenciada",
            "codigo_usina_influenciadora",
            "fator_impacto_defluencia",
        ],
    )


def _va(*rows: tuple[int, int, float]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "codigo_usina_influenciada",
            "codigo_posto_influenciador",
            "fator_impacto_incremental",
        ],
    )


def _resolve(
    effective: EffectiveCadastro,
    *,
    vl: pd.DataFrame | None = None,
    vu: pd.DataFrame | None = None,
    va: pd.DataFrame | None = None,
    case: Collection[int] | None = None,
    piecewise: Collection[int] = (),
) -> TailraceFlowResolution:
    frames = {
        name: frame
        for name, frame in (("vl", vl), ("vu", vu), ("va", va))
        if frame is not None
    }
    return resolve_tailrace_flow(
        cast("TailraceRegisters", _FakeDadger(**frames)),
        effective,
        effective.base.index.tolist() if case is None else case,
        piecewise,
    )


def _rows(resolution: TailraceFlowResolution, code: str) -> list[list[object]]:
    (diagnostic,) = [d for d in resolution.diagnostics if d.code == code]
    assert diagnostic.table is not None
    return [list(row) for row in diagnostic.table.rows]


def _codes(resolution: TailraceFlowResolution) -> list[str]:
    return [d.code for d in resolution.diagnostics]


@pytest.fixture(scope="module")
def decomp_mini_effective(decomp_mini_case: DecompCase) -> EffectiveCadastro:
    effective, _ = build_effective_cadastro(
        decomp_mini_case.dadger, decomp_mini_case.hidr, decomp_mini_case.calendar
    )
    return effective


@pytest.fixture(scope="module")
def decomp_mini_resolution(
    decomp_mini_case: DecompCase, decomp_mini_effective: EffectiveCadastro
) -> TailraceFlowResolution:
    return resolve_tailrace_flow(
        decomp_mini_case.dadger,
        decomp_mini_effective,
        decomp_mini_case.id_map.operated_hydro_codes,
        set(),
    )


class TestDecompMini:
    def test_decomp_mini_declarations(
        self, decomp_mini_resolution: TailraceFlowResolution
    ) -> None:
        assert decomp_mini_resolution.declarations == {
            147: ResolvedTailraceFlow(1.0, ((146, 1.0),), (), None),
            146: ResolvedTailraceFlow(1.0, ((147, 1.0),), (), None),
            288: ResolvedTailraceFlow(1.0, ((314, 1.0),), ((314, 0.07),), None),
            66: ResolvedTailraceFlow(1.03, ((83, 1.17),), (), None),
        }

    def test_decomp_mini_has_no_findings(
        self, decomp_mini_resolution: TailraceFlowResolution
    ) -> None:
        assert decomp_mini_resolution.diagnostics == ()

    def test_decomp_mini_lists_every_record_in_deck_order(
        self, decomp_mini_resolution: TailraceFlowResolution
    ) -> None:
        assert decomp_mini_resolution.records == (
            ("VL", 147),
            ("VL", 146),
            ("VL", 288),
            ("VL", 66),
            ("VU", 147),
            ("VU", 146),
            ("VU", 288),
            ("VU", 66),
            ("VA", 288),
        )


class TestAbsentRegisters:
    def test_a_deck_without_the_registers_resolves_to_nothing(self) -> None:
        resolution = _resolve(_effective([1, 2]))
        assert resolution == TailraceFlowResolution({}, (), ())


class TestInfluencedPlantNotInCase:
    def test_records_of_a_plant_outside_the_case_are_dropped_silently(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        effective = _effective([1, 2, 9])
        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            resolution = _resolve(
                effective,
                vl=_vl((9, 1.2), (1, 1.1)),
                vu=_vu((9, 2, 0.5)),
                va=_va((9, 102, 0.1)),
                case=[1, 2],
            )
        assert list(resolution.declarations) == [1]
        assert resolution.diagnostics == ()
        assert ("VU", 9) in resolution.records
        assert {r.levelno for r in caplog.records if r.name == _LOGGER} == {
            logging.DEBUG
        }


class TestSpillOutsideTailrace:
    def test_every_record_of_the_plant_is_skipped(self) -> None:
        resolution = _resolve(
            _effective([1, 2, 3], no_spill={2}),
            vl=_vl((1, 1.1), (2, 1.2)),
            vu=_vu((2, 3, 0.5)),
            va=_va((2, 103, 0.1)),
        )
        assert list(resolution.declarations) == [1]
        assert _codes(resolution) == ["tailrace-spill-outside-tailrace-skipped"]
        assert _rows(resolution, "tailrace-spill-outside-tailrace-skipped") == [
            [2, "Plant2", "VL, VU, VA"]
        ]

    def test_a_plant_listed_once_names_the_registers_it_lost(self) -> None:
        resolution = _resolve(
            _effective([1, 2, 3], no_spill={2}),
            vu=_vu((2, 3, 0.5), (2, 1, 0.4)),
            vl=_vl((2, 1.2)),
        )
        assert resolution.declarations == {}
        assert _rows(resolution, "tailrace-spill-outside-tailrace-skipped") == [
            [2, "Plant2", "VL, VU"]
        ]

    def test_a_plant_that_spills_outside_the_tailrace_at_a_later_stage_is_skipped(
        self,
    ) -> None:
        resolution = _resolve(
            _effective([1, 2], stage_varying={(2, _VERTJU): (1.0, 0.0)}),
            vl=_vl((1, 1.0), (2, 1.0)),
        )
        assert list(resolution.declarations) == [1]
        assert _codes(resolution) == ["tailrace-spill-outside-tailrace-skipped"]


class TestDuplicateVl:
    def test_the_first_vl_record_wins(self) -> None:
        resolution = _resolve(
            _effective([1]),
            vl=_vl((1, 1.1, [0, 0, 0, 0, 0]), (1, 1.5, [1, 2, 3, 4, 5]), (1, 1.7)),
        )
        assert resolution.declarations == {1: ResolvedTailraceFlow(1.1, (), (), None)}
        assert _codes(resolution) == ["tailrace-duplicate-vl-dropped"]
        assert _rows(resolution, "tailrace-duplicate-vl-dropped") == [[1, "Plant1", 2]]


class TestGaugeMapping:
    def test_a_gauge_maps_to_the_plant_that_has_it(self) -> None:
        resolution = _resolve(
            _effective([1, 2], gauges={2: 500}), va=_va((1, 500, 0.1))
        )
        assert resolution.declarations == {
            1: ResolvedTailraceFlow(1.0, (), ((2, 0.1),), None)
        }
        assert resolution.diagnostics == ()

    def test_the_mapping_reads_the_gauge_in_force_at_the_first_stage(self) -> None:
        resolution = _resolve(
            _effective([1, 2], inflow_gauges={2: (500, 600)}), va=_va((1, 500, 0.1))
        )
        assert resolution.declarations[1].incremental_inflow_influences == ((2, 0.1),)

    def test_a_gauge_no_plant_has_drops_the_record(self) -> None:
        resolution = _resolve(
            _effective([1, 2]), vl=_vl((2, 1.0)), va=_va((1, 999, 0.1))
        )
        assert list(resolution.declarations) == [2]
        assert _codes(resolution) == ["tailrace-gauge-not-in-case"]
        assert _rows(resolution, "tailrace-gauge-not-in-case") == [[1, "Plant1", 999]]

    def test_a_gauge_with_several_plants_drops_the_record(self) -> None:
        resolution = _resolve(
            _effective([1, 2, 3], gauges={2: 500, 3: 500}),
            vl=_vl((1, 1.0)),
            va=_va((1, 500, 0.1)),
        )
        assert resolution.declarations[1].incremental_inflow_influences == ()
        assert _codes(resolution) == ["tailrace-gauge-ambiguous"]
        assert _rows(resolution, "tailrace-gauge-ambiguous") == [
            [1, "Plant1", 500, "2, 3"]
        ]

    def test_a_gauge_that_varies_is_kept_at_the_first_stage_and_reported(self) -> None:
        resolution = _resolve(
            _effective([1, 2], inflow_gauges={2: (500, 600)}),
            va=_va((1, 500, 0.1)),
        )
        assert resolution.declarations[1].incremental_inflow_influences == ((2, 0.1),)
        assert _codes(resolution) == ["tailrace-gauge-varies"]
        assert _rows(resolution, "tailrace-gauge-varies") == [[2, "Plant2", 500]]

    def test_a_constant_gauge_override_is_not_reported_as_varying(self) -> None:
        resolution = _resolve(
            _effective([1, 2], inflow_gauges={2: (500, 500)}),
            va=_va((1, 500, 0.1)),
        )
        assert resolution.diagnostics == ()


class TestVuInfluencerNotInCase:
    def test_the_record_is_dropped_without_a_diagnostic(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            resolution = _resolve(
                _effective([1, 2, 9]),
                vl=_vl((1, 1.1)),
                vu=_vu((1, 9, 0.5), (1, 2, 0.25)),
                case=[1, 2],
            )
        assert resolution.declarations == {
            1: ResolvedTailraceFlow(1.1, ((2, 0.25),), (), None)
        }
        assert resolution.diagnostics == ()
        assert {r.levelno for r in caplog.records if r.name == _LOGGER} == {
            logging.DEBUG
        }

    def test_a_plant_with_no_other_record_is_not_declared(self) -> None:
        resolution = _resolve(_effective([1, 2, 9]), vu=_vu((1, 9, 0.5)), case=[1, 2])
        assert resolution.declarations == {}


class TestSelfVu:
    def test_the_factor_is_added_to_the_vl_factor(self) -> None:
        resolution = _resolve(
            _effective([1, 2]), vl=_vl((1, 1.03)), vu=_vu((1, 1, 0.5), (1, 2, 0.25))
        )
        assert resolution.declarations == {
            1: ResolvedTailraceFlow(1.03 + 0.5, ((2, 0.25),), (), None)
        }
        assert _codes(resolution) == ["tailrace-self-outflow-folded"]
        assert _rows(resolution, "tailrace-self-outflow-folded") == [[1, "Plant1", 0.5]]
        assert resolution.diagnostics[0].severity is Severity.INFO

    def test_the_factor_is_added_to_one_without_a_vl_record(self) -> None:
        resolution = _resolve(_effective([1]), vu=_vu((1, 1, 0.5)))
        assert resolution.declarations == {
            1: ResolvedTailraceFlow(1.0 + 0.5, (), (), None)
        }

    def test_repeated_self_records_fold_into_one_total(self) -> None:
        resolution = _resolve(_effective([1]), vu=_vu((1, 1, 0.25), (1, 1, 0.5)))
        assert resolution.declarations[1].own_outflow_factor == 1.75
        assert _rows(resolution, "tailrace-self-outflow-folded") == [
            [1, "Plant1", 0.75]
        ]
        assert _codes(resolution) == ["tailrace-self-outflow-folded"]


class TestDuplicateInfluence:
    def test_a_repeated_vu_influencer_is_summed(self) -> None:
        resolution = _resolve(_effective([1, 2]), vu=_vu((1, 2, 0.25), (1, 2, 0.5)))
        assert resolution.declarations[1].outflow_influences == ((2, 0.75),)
        assert _codes(resolution) == ["tailrace-duplicate-influence-summed"]
        assert _rows(resolution, "tailrace-duplicate-influence-summed") == [
            [1, "Plant1", "VU", 2, 0.75]
        ]
        assert resolution.diagnostics[0].severity is Severity.INFO

    def test_repeated_va_records_on_one_gauge_are_summed_after_mapping(self) -> None:
        resolution = _resolve(
            _effective([1, 2], gauges={2: 500}),
            va=_va((1, 500, 0.25), (1, 500, 0.5)),
        )
        assert resolution.declarations[1].incremental_inflow_influences == ((2, 0.75),)
        assert _rows(resolution, "tailrace-duplicate-influence-summed") == [
            [1, "Plant1", "VA", 2, 0.75]
        ]

    def test_the_same_influencer_in_a_vu_and_a_va_is_not_a_duplicate(self) -> None:
        resolution = _resolve(
            _effective([1, 2], gauges={2: 500}),
            vu=_vu((1, 2, 0.25)),
            va=_va((1, 500, 0.5)),
        )
        assert resolution.declarations == {
            1: ResolvedTailraceFlow(1.0, ((2, 0.25),), ((2, 0.5),), None)
        }
        assert resolution.diagnostics == ()


class TestSelfVa:
    def test_a_gauge_mapping_to_the_plant_itself_stays_a_self_entry(self) -> None:
        resolution = _resolve(_effective([1, 2]), va=_va((1, 101, 0.07)))
        assert resolution.declarations == {
            1: ResolvedTailraceFlow(1.0, (), ((1, 0.07),), None)
        }
        assert resolution.diagnostics == ()

    def test_self_and_repeated_records_on_the_gauge_sum_into_one_self_entry(
        self,
    ) -> None:
        resolution = _resolve(_effective([1, 2]), va=_va((1, 101, 0.25), (1, 101, 0.5)))
        assert resolution.declarations[1].incremental_inflow_influences == ((1, 0.75),)
        assert _codes(resolution) == ["tailrace-duplicate-influence-summed"]

    def test_a_gauge_number_equal_to_the_plant_code_is_not_a_self_entry(self) -> None:
        resolution = _resolve(
            _effective([10, 20], gauges={10: 30, 20: 10}), va=_va((10, 10, 0.07))
        )
        assert resolution.declarations[10].incremental_inflow_influences == (
            (20, 0.07),
        )


class TestVlPolynomial:
    _POLYNOMIAL = [1.0, 0.5, 0.0, 0.0, 0.25]

    def test_a_polynomial_is_kept_for_a_plant_without_a_piecewise_curve(self) -> None:
        resolution = _resolve(
            _effective([1, 2]), vl=_vl((1, 1.0, self._POLYNOMIAL)), piecewise={2}
        )
        assert resolution.declarations[1].polynomial == (1.0, 0.5, 0.0, 0.0, 0.25)
        assert resolution.diagnostics == ()

    def test_a_polynomial_is_dropped_for_a_plant_with_a_piecewise_curve(self) -> None:
        resolution = _resolve(
            _effective([1, 2]), vl=_vl((1, 1.0, self._POLYNOMIAL)), piecewise={1}
        )
        assert resolution.declarations[1].polynomial is None
        assert _codes(resolution) == ["tailrace-vl-polynomial-shadowed"]
        assert _rows(resolution, "tailrace-vl-polynomial-shadowed") == [[1, "Plant1"]]
        assert resolution.diagnostics[0].severity is Severity.WARNING

    def test_all_zero_coefficients_are_no_polynomial(self) -> None:
        resolution = _resolve(
            _effective([1]), vl=_vl((1, 1.0, [0.0] * 5)), piecewise={1}
        )
        assert resolution.declarations[1].polynomial is None
        assert resolution.diagnostics == ()


class TestFactorValidation:
    @pytest.mark.parametrize("factor", [-0.1, float("nan"), float("inf")])
    @pytest.mark.parametrize("register", ["VL", "VU", "VA"])
    def test_an_unusable_factor_raises_naming_register_and_plant(
        self, register: str, factor: float
    ) -> None:
        frames = {
            "VL": {"vl": _vl((1, factor))},
            "VU": {"vu": _vu((1, 2, factor))},
            "VA": {"va": _va((1, 102, factor))},
        }[register]
        with pytest.raises(FieldParseError, match=rf"{register} record of plant 1"):
            _resolve(_effective([1, 2]), **frames)

    def test_a_zero_factor_is_kept(self) -> None:
        resolution = _resolve(_effective([1, 2]), vu=_vu((1, 2, 0.0)))
        assert resolution.declarations[1].outflow_influences == ((2, 0.0),)


class TestPurity:
    def test_the_resolver_emits_nothing_and_leaves_the_frames_alone(self) -> None:
        vl, vu, va = _vl((1, 1.1), (1, 1.2)), _vu((1, 1, 0.5)), _va((1, 999, 0.1))
        before = [frame.copy() for frame in (vl, vu, va)]
        with collect() as sink:
            resolution = _resolve(_effective([1]), vl=vl, vu=vu, va=va)
        assert sink == []
        assert resolution.diagnostics
        for frame, original in zip((vl, vu, va), before, strict=True):
            pd.testing.assert_frame_equal(frame, original)


def _mixed_deck() -> tuple[EffectiveCadastro, dict[str, pd.DataFrame]]:
    effective = _effective([1, 2, 3, 4], gauges={3: 500, 4: 600})
    return effective, {
        "vl": _vl((1, 1.03), (2, 1.0, [1.0, 2.0, 3.0, 4.0, 5.0]), (3, 1.2)),
        "vu": _vu(
            (1, 2, 0.1),
            (1, 2, 0.2),
            (1, 2, 0.3),
            (1, 1, 0.1),
            (1, 1, 0.7),
            (1, 3, 0.4),
            (2, 4, 0.6),
            (3, 1, 0.9),
            (4, 4, 0.35),
            (4, 4, 0.1),
            (4, 4, 0.2),
        ),
        "va": _va(
            (1, 500, 0.1),
            (1, 500, 0.2),
            (1, 500, 0.3),
            (1, 101, 0.07),
            (2, 600, 0.5),
            (3, 999, 0.5),
        ),
    }


class TestDeterminism:
    @pytest.mark.parametrize("seed", [1, 2, 3, 4])
    def test_the_result_is_identical_under_any_permutation_of_the_rows(
        self, seed: int
    ) -> None:
        effective, frames = _mixed_deck()
        reference = _resolve(effective, **frames)
        shuffled = {
            name: frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            for name, frame in frames.items()
        }
        permuted = _resolve(effective, **shuffled)
        assert permuted.declarations == reference.declarations
        assert permuted.diagnostics == reference.diagnostics

    def test_influence_lists_hold_no_duplicate_code_and_no_self_outflow(self) -> None:
        effective, frames = _mixed_deck()
        resolution = _resolve(effective, **frames)
        for plant, declaration in resolution.declarations.items():
            outflow = [code for code, _ in declaration.outflow_influences]
            incremental = [
                code for code, _ in declaration.incremental_inflow_influences
            ]
            assert outflow == sorted(set(outflow))
            assert incremental == sorted(set(incremental))
            assert plant not in outflow
        assert resolution.declarations[1].incremental_inflow_influences == (
            (1, 0.07),
            (3, pytest.approx(0.6)),
        )

    def test_diagnostics_are_sorted_by_code_and_rows_by_plant(self) -> None:
        resolution = _resolve(
            _effective([1, 2, 3]),
            vl=_vl((3, 1.0), (3, 1.0), (1, 1.0), (1, 1.0), (2, 1.0), (2, 1.0)),
            vu=_vu((3, 3, 0.5), (1, 1, 0.5)),
        )
        assert _codes(resolution) == sorted(_codes(resolution))
        assert [
            row[0] for row in _rows(resolution, "tailrace-duplicate-vl-dropped")
        ] == [
            1,
            2,
            3,
        ]
        assert [
            row[0] for row in _rows(resolution, "tailrace-self-outflow-folded")
        ] == [
            1,
            3,
        ]


class TestTailraceRaise:
    def test_the_resolution_raises_the_tailrace_like_the_raw_records(self) -> None:
        effective, frames = _mixed_deck()
        resolution = _resolve(effective, **frames)
        outflow = {1: 40.0, 2: 55.0, 3: 31.0, 4: 22.0}
        spill = {1: 5.0, 2: 6.0, 3: 7.0, 4: 8.0}
        incremental = {1: 3.0, 2: 4.0, 3: 5.0, 4: 6.0}
        total = {code: outflow[code] + spill[code] for code in outflow}
        plant_of_gauge = {
            effective.inflow_gauge(code, 0): code for code in effective.base.index
        }

        def raw(plant: int) -> float:
            vl = frames["vl"]
            own = vl.loc[
                vl["codigo_usina_influenciada"] == plant, "fator_impacto_defluencia"
            ]
            vu = frames["vu"]
            va = frames["va"]
            return (
                (float(own.iloc[0]) if len(own) else 1.0) * total[plant]
                + sum(
                    float(f) * total[int(j)]
                    for p, j, f in vu.itertuples(index=False)
                    if p == plant
                )
                + sum(
                    float(f) * incremental[plant_of_gauge[int(g)]]
                    for p, g, f in va.itertuples(index=False)
                    if p == plant and int(g) in plant_of_gauge
                )
            )

        for plant, declaration in resolution.declarations.items():
            resolved = (
                declaration.own_outflow_factor * total[plant]
                + sum(f * total[j] for j, f in declaration.outflow_influences)
                + sum(
                    f * incremental[k]
                    for k, f in declaration.incremental_inflow_influences
                )
            )
            assert resolved == pytest.approx(raw(plant), rel=1e-12)


class TestDiagnosticContract:
    def test_every_code_carries_its_severity(self) -> None:
        effective = _effective(
            [1, 2, 3, 4, 5, 6],
            gauges={3: 500, 4: 500, 5: 600},
            no_spill={2},
            inflow_gauges={5: (600, 700)},
        )
        resolution = _resolve(
            effective,
            vl=_vl((1, 1.0, [1.0, 0, 0, 0, 0]), (1, 1.1), (2, 1.0)),
            vu=_vu((1, 1, 0.5), (1, 6, 0.25), (1, 6, 0.25)),
            va=_va((1, 999, 0.1), (1, 500, 0.1), (1, 600, 0.1)),
            piecewise={1},
        )
        assert {d.code: d.severity for d in resolution.diagnostics} == {
            "tailrace-spill-outside-tailrace-skipped": Severity.WARNING,
            "tailrace-duplicate-vl-dropped": Severity.WARNING,
            "tailrace-gauge-not-in-case": Severity.WARNING,
            "tailrace-gauge-ambiguous": Severity.WARNING,
            "tailrace-gauge-varies": Severity.WARNING,
            "tailrace-self-outflow-folded": Severity.INFO,
            "tailrace-duplicate-influence-summed": Severity.INFO,
            "tailrace-vl-polynomial-shadowed": Severity.WARNING,
        }
        assert len(resolution.diagnostics) == 8


def _id_map(*codes: int, dormant: tuple[int, ...] = ()) -> DecompIdMap:
    return DecompIdMap(
        bus_codes=(1,),
        bus_names=("SE",),
        hydro_codes=codes,
        dormant_hydro_codes=dormant,
    )


def _declaration(
    own: float = 1.0,
    outflow: tuple[tuple[int, float], ...] = (),
    incremental: tuple[tuple[int, float], ...] = (),
    polynomial: tuple[float, ...] | None = None,
) -> ResolvedTailraceFlow:
    return ResolvedTailraceFlow(own, outflow, incremental, polynomial)


def _convert(
    declarations: Mapping[int, ResolvedTailraceFlow],
    *,
    stage: Collection[int],
    eligible: Collection[int] | None = None,
    records: Sequence[tuple[str, int]] = (),
) -> tuple[dict[int, TailraceFlowEntry], list[Diagnostic]]:
    codes = sorted(declarations)
    resolution = TailraceFlowResolution(declarations, tuple(records), ())
    with collect() as found:
        entries = convert_tailrace_flow(
            resolution,
            stage,
            _id_map(*codes),
            fpha_codes=stage if eligible is None else eligible,
            effective=_effective(codes),
        )
    return entries, found


class TestConvertTailraceFlow:
    def test_a_constant_productivity_plant_gets_no_entry_and_is_reported(
        self,
    ) -> None:
        entries, found = _convert(
            {
                1: _declaration(outflow=((2, 1.0),)),
                2: _declaration(outflow=((1, 1.0),)),
                3: _declaration(1.05),
            },
            stage={1, 2},
            records=[("VL", 3), ("VU", 3), ("VU", 1), ("VU", 2), ("VL", 1)],
        )
        assert set(entries) == {1, 2}
        (diagnostic,) = found
        assert diagnostic.code == "tailrace-constant-productivity-omitted"
        assert diagnostic.severity is Severity.WARNING
        assert diagnostic.table is not None
        assert diagnostic.table.columns == ["Code", "Name", "Records"]
        assert [list(row) for row in diagnostic.table.rows] == [[3, "Plant3", "VL, VU"]]

    def test_every_omitted_plant_is_one_row_in_code_order(self) -> None:
        _, found = _convert(
            {5: _declaration(1.1), 2: _declaration(polynomial=(1.0, 0, 0, 0, 0))},
            stage=(),
            records=[("VA", 5), ("VL", 2)],
        )
        (diagnostic,) = found
        assert diagnostic.table is not None
        assert [list(row) for row in diagnostic.table.rows] == [
            [2, "Plant2", "VL"],
            [5, "Plant5", "VA"],
        ]

    def test_an_eligible_plant_in_outage_at_every_stage_is_dropped_silently(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG, logger=_LOGGER)
        entries, found = _convert(
            {1: _declaration(1.05), 2: _declaration(1.05)},
            stage={1},
            eligible={1, 2},
        )
        assert set(entries) == {1}
        assert found == []
        assert [r.levelno for r in caplog.records] == [logging.DEBUG]
        assert "plant 2" in caplog.records[0].getMessage()

    def test_a_default_declaration_on_a_non_fpha_plant_reports_nothing(self) -> None:
        entries, found = _convert({1: _declaration()}, stage=())
        assert entries == {}
        assert found == []

    def test_a_default_declaration_on_an_fpha_plant_carries_nothing(self) -> None:
        entries, _ = _convert({1: _declaration()}, stage={1})
        assert entries == {1: TailraceFlowEntry(None, None)}

    def test_lists_are_sorted_by_hydro_id_and_a_self_entry_stays_incremental(
        self,
    ) -> None:
        entries, _ = _convert(
            {
                10: _declaration(),
                20: _declaration(
                    1.0,
                    outflow=((30, 0.5), (10, 0.25)),
                    incremental=((30, 0.1), (20, 0.3), (10, 0.2)),
                ),
                30: _declaration(),
            },
            stage={10, 20, 30},
        )
        assert entries[20].tailrace_flow == {
            "own_outflow_factor": 1.0,
            "outflow_influences": [
                {"hydro_id": 0, "factor": 0.25},
                {"hydro_id": 2, "factor": 0.5},
            ],
            "incremental_inflow_influences": [
                {"hydro_id": 0, "factor": 0.2},
                {"hydro_id": 1, "factor": 0.3},
                {"hydro_id": 2, "factor": 0.1},
            ],
        }

    def test_a_non_default_own_factor_alone_is_written_with_empty_lists(
        self,
    ) -> None:
        entries, _ = _convert({1: _declaration(1.03)}, stage={1})
        assert entries[1].tailrace_flow == {
            "own_outflow_factor": 1.03,
            "outflow_influences": [],
            "incremental_inflow_influences": [],
        }

    def test_the_polynomial_alone_leaves_the_declaration_default(self) -> None:
        polynomial = (3.0, 0.5, 0.0, 0.0, 0.0)
        entries, _ = _convert({1: _declaration(polynomial=polynomial)}, stage={1})
        assert entries == {1: TailraceFlowEntry(None, polynomial)}

    def test_the_findings_of_the_resolution_are_emitted_once(self) -> None:
        effective = _effective([1, 2])
        resolution = _resolve(
            effective, vl=_vl((1, 1.0), (1, 1.1)), vu=_vu((1, 1, 0.5))
        )
        assert _codes(resolution) == [
            "tailrace-duplicate-vl-dropped",
            "tailrace-self-outflow-folded",
        ]
        with collect() as found:
            convert_tailrace_flow(
                resolution,
                {1},
                _id_map(1, 2),
                fpha_codes={1},
                effective=effective,
            )
        assert [d.code for d in found] == _codes(resolution)


class _StubPolinjus:
    def __init__(self, families: pd.DataFrame) -> None:
        self._families = families

    def hidreletrica_curvajusante(self, *, df: bool) -> pd.DataFrame:
        return self._families


def _case_with_polinjus(polinjus: object | None) -> DecompCase:
    return make_decomp_case(Path("unused"), polinjus=polinjus)


class TestPiecewiseTailraceCodes:
    def test_a_deck_without_polinjus_has_no_piecewise_plant(self) -> None:
        case = _case_with_polinjus(None)
        assert piecewise_tailrace_codes(case, _id_map(1, 2)) == frozenset()

    def test_a_polinjus_without_families_has_no_piecewise_plant(self) -> None:
        case = _case_with_polinjus(_StubPolinjus(pd.DataFrame()))
        assert piecewise_tailrace_codes(case, _id_map(1, 2)) == frozenset()

    def test_only_operated_plants_with_a_family_count(self) -> None:
        families = pd.DataFrame({"codigo_usina": [1, 1, 3, 4, 9]})
        case = _case_with_polinjus(_StubPolinjus(families))
        id_map = _id_map(1, 2, 3, 4, dormant=(4,))
        assert piecewise_tailrace_codes(case, id_map) == {1, 3}

    def test_decomp_mini_plants_with_declarations_have_a_family(
        self, decomp_mini_case: DecompCase
    ) -> None:
        codes = piecewise_tailrace_codes(decomp_mini_case, decomp_mini_case.id_map)
        assert {147, 146, 288, 66} <= codes


_FLOW = {
    "own_outflow_factor": 1.1,
    "outflow_influences": [],
    "incremental_inflow_influences": [],
}


def _hydros_by_id(
    case: DecompCase,
    effective: EffectiveCadastro,
    **flows: Any,
) -> dict[int, dict]:
    doc = convert_hydros(
        case,
        case.id_map,
        effective=effective,
        turbine_limits=flat_turbine_limits(
            case.id_map, len(case.calendar), group_counts={66: 2}
        ),
        fpha_codes={147, 146},
        **flows,
    )
    return {hydro["id"]: hydro for hydro in doc["hydros"]}


class TestHydroEntry:
    def test_a_vl_polynomial_replaces_the_constant_tailrace(
        self, decomp_mini_case: DecompCase, decomp_mini_effective: EffectiveCadastro
    ) -> None:
        case, effective = decomp_mini_case, decomp_mini_effective
        polynomial = (7.5, 0.25, 0.0, 0.0, 0.0)
        base = _hydros_by_id(case, effective)
        stamped = _hydros_by_id(
            case,
            effective,
            tailrace_flows={
                147: TailraceFlowEntry(_FLOW, polynomial),
                146: TailraceFlowEntry(None, None),
            },
        )
        fontes_ab = case.id_map.hydro_id(147)
        assert base[fontes_ab]["tailrace"]["coefficients"] != list(polynomial)
        assert stamped[fontes_ab]["tailrace"] == {
            "type": "polynomial",
            "coefficients": list(polynomial),
        }
        assert stamped[fontes_ab]["tailrace_flow"] == _FLOW
        assert {i: h for i, h in stamped.items() if i != fontes_ab} == {
            i: h for i, h in base.items() if i != fontes_ab
        }

    def test_an_entry_of_a_plant_that_is_not_fpha_is_ignored(
        self, decomp_mini_case: DecompCase, decomp_mini_effective: EffectiveCadastro
    ) -> None:
        flows = {288: TailraceFlowEntry(_FLOW, (1.0, 0.0, 0.0, 0.0, 0.0))}
        assert _hydros_by_id(
            decomp_mini_case, decomp_mini_effective, tailrace_flows=flows
        ) == _hydros_by_id(decomp_mini_case, decomp_mini_effective)

    def test_no_flows_leaves_the_hydros_as_without_the_keyword(
        self, decomp_mini_case: DecompCase, decomp_mini_effective: EffectiveCadastro
    ) -> None:
        assert _hydros_by_id(
            decomp_mini_case, decomp_mini_effective, tailrace_flows=None
        ) == _hydros_by_id(decomp_mini_case, decomp_mini_effective)


class TestDecompMiniEmission:
    def test_hydros_carry_the_declarations_of_the_deck(
        self, decomp_mini_deck: Path, decomp_mini_case: DecompCase, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        report = convert_decomp_case(decomp_mini_deck, out, force=True)
        hydros = json.loads((out / "system" / "hydros.json").read_text("utf-8"))[
            "hydros"
        ]
        id_map = decomp_mini_case.id_map

        def influences(*pairs: tuple[int, float]) -> list[dict]:
            return [
                {"hydro_id": id_map.hydro_id(code), "factor": factor}
                for code, factor in pairs
            ]

        assert {
            hydro["name"]: hydro["tailrace_flow"]
            for hydro in hydros
            if "tailrace_flow" in hydro
        } == {
            "FONTES AB": {
                "own_outflow_factor": 1.0,
                "outflow_influences": influences((146, 1.0)),
                "incremental_inflow_influences": [],
            },
            "FONTES C": {
                "own_outflow_factor": 1.0,
                "outflow_influences": influences((147, 1.0)),
                "incremental_inflow_influences": [],
            },
            "BELO MONTE": {
                "own_outflow_factor": 1.0,
                "outflow_influences": influences((314, 1.0)),
                "incremental_inflow_influences": influences((314, 0.07)),
            },
            "ITAIPU": {
                "own_outflow_factor": 1.03,
                "outflow_influences": influences((83, 1.17)),
                "incremental_inflow_influences": [],
            },
        }
        assert not [d for d in report.diagnostics if d.code.startswith("tailrace-")]
