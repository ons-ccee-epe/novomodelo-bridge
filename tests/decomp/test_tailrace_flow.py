"""Tests for the tailrace lateral-flow resolver over a DECOMP deck's VL, VU and VA records.

The decomp_mini deck pins the converted shape of a real deck; synthetic
register doubles pin each resolution rule.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping, Sequence

import pandas as pd
import pytest

from novomodelo_bridge.core.diagnostics import Severity, collect
from novomodelo_bridge.core.errors import FieldParseError
from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.converters.cadastro import (
    EffectiveCadastro,
    build_effective_cadastro,
)
from novomodelo_bridge.decomp.converters.tailrace_flow import (
    ResolvedTailraceFlow,
    TailraceFlowResolution,
    resolve_tailrace_flow,
)
from tests.conftest import _FakeDadger

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
        _FakeDadger(**frames),
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
def decomp_mini_resolution(decomp_mini_case: DecompCase) -> TailraceFlowResolution:
    effective, _ = build_effective_cadastro(
        decomp_mini_case.dadger, decomp_mini_case.hidr, decomp_mini_case.calendar
    )
    return resolve_tailrace_flow(
        decomp_mini_case.dadger,
        effective,
        decomp_mini_case.id_map.hydro_codes,
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
