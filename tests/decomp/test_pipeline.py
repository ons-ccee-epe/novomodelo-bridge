"""Tests for the DECOMP scenario emitters, pipeline, and CLI command."""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from contextlib import ExitStack
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from novomodelo_bridge.cli import app
from novomodelo_bridge.core.diagnostics import Diagnostic
from novomodelo_bridge.decomp.bounds_accumulator import BoundContribution
from novomodelo_bridge.decomp.constraint_registers import (
    ConstraintCensus,
    ConstraintRecord,
    ConstraintTerm,
    HeMeta,
    StageBounds,
)
from novomodelo_bridge.decomp.converters.anticipated import GnlEmission
from novomodelo_bridge.decomp.converters.cadastro import (
    DiversionChannel,
    EffectiveCadastro,
)
from novomodelo_bridge.decomp.converters.network import _LINE_BOUNDS_SCHEMA
from novomodelo_bridge.decomp.converters.single_term_bounds import HydroCapacities
from novomodelo_bridge.decomp.converters.thermal import (
    _THERMAL_COST_SCHEMA,
    ThermalBounds,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.pipeline import (
    ConversionReport,
    _base_diversion_channels,
    _diversion_channels,
)
from novomodelo_bridge.decomp.scenarios import (
    convert_external_inflows,
    convert_inflow_stats_identity,
    convert_scenario_probabilities,
    deterministic_external_scenarios,
    terminal_fan_probabilities,
)
from novomodelo_bridge.decomp.temporal import build_operative_calendar
from novomodelo_bridge.novomodelo import schemas as novomodelo_schemas
from tests.conftest import make_decomp_case

_ID_MAP = DecompIdMap(
    bus_codes=(1, 2),
    bus_names=("SE", "S"),
    hydro_codes=(1, 2),
)


def _calendar():
    hours = [[15.0, 64.0, 89.0]] * 2 + [[63.0, 280.0, 401.0]]
    return build_operative_calendar(date(2026, 7, 18), hours)


def _scenario_case():
    """A ``DecompCase`` carrying only ``calendar`` — the sole cached slot
    ``convert_external_inflows``/``convert_inflow_stats_identity`` read."""
    return make_decomp_case(Path("unused"), calendar=_calendar())


def _hidr_frame() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            1: {"posto": 11, "codigo_usina_jusante": 2},
            2: {"posto": 12, "codigo_usina_jusante": 0},
        }
    ).T
    df.index.name = "codigo_usina"
    return df


def _effective_no_override(hidr: pd.DataFrame) -> EffectiveCadastro:
    """``convert_external_inflows`` now takes the effective
    cadastro, not the bare ``hidr`` frame; every call site in this module
    that carries no ``AC NUMJUS``/``NUMPOS`` override wraps *hidr* through
    this — the resulting downstream/gauge reads fall through to *hidr*
    unchanged (see the ``test_cadastro_topology`` module for the
    override-bearing cases).
    """
    return EffectiveCadastro(base=hidr, n_stages=len(_calendar()), stage_varying={})


class _StubVazoes:
    def __init__(self) -> None:
        self.previsoes = pd.DataFrame(
            [
                {"estagio": 1, "11": 100.0, "12": 150.0},
                {"estagio": 2, "11": 110.0, "12": 160.0},
            ]
        )
        self.cenarios_gerados = pd.DataFrame(
            [
                {"estagio": 3, "cenario": 1, "11": 90.0, "12": 140.0},
                {"estagio": 3, "cenario": 2, "11": 120.0, "12": 180.0},
            ]
        )
        self.probabilidades = pd.DataFrame(
            [
                {"estagio": 1, "no": 1, "cenario": 1, "probabilidade": 1.0},
                {"estagio": 2, "no": 2, "cenario": 1, "probabilidade": 1.0},
                {"estagio": 3, "no": 3, "cenario": 1, "probabilidade": 0.6},
                {"estagio": 3, "no": 4, "cenario": 2, "probabilidade": 0.4},
            ]
        )


class TestScenarioEmitters:
    def test_external_inflows_are_incremental(self) -> None:
        hidr = _hidr_frame()
        table = convert_external_inflows(
            _scenario_case(),
            _ID_MAP,
            vazoes=_StubVazoes(),
            effective=_effective_no_override(hidr),
        ).to_pandas()
        # 4 tree nodes × 2 hydros.
        assert len(table) == 8
        trunk = table[(table["stage_id"] == 0) & (table["hydro_id"] == 1)]
        # DECOMP's inflow file is already incremental per gauge column, so
        # plant 2's value passes through directly (no upstream subtraction).
        assert trunk["value_m3s"].iloc[0] == pytest.approx(150.0)
        fan = table[(table["stage_id"] == 2) & (table["scenario_id"] == 1)]
        assert fan[fan["hydro_id"] == 0]["value_m3s"].iloc[0] == pytest.approx(120.0)
        assert fan[fan["hydro_id"] == 1]["value_m3s"].iloc[0] == pytest.approx(180.0)

    def test_identity_stats(self) -> None:
        stats = convert_inflow_stats_identity(_scenario_case(), _ID_MAP).to_pandas()
        assert len(stats) == 2 * 3
        assert set(stats["mean_m3s"]) == {0.0}
        assert set(stats["std_m3s"]) == {1.0}

    def test_deterministic_external_scenarios_replicate_across_columns(self) -> None:
        stats = pa.table(
            {
                "ncs_id": pa.array([0, 0, 1, 1], type=pa.int32()),
                "stage_id": pa.array([0, 1, 0, 1], type=pa.int32()),
                "mean": pa.array([0.5, 0.6, 0.7, 0.8], type=pa.float64()),
            }
        )
        # Trunk stage 0 -> 1 column; terminal stage 1 -> a 3-member fan.
        out = deterministic_external_scenarios(
            stats,
            entity_column="ncs_id",
            value_in="mean",
            value_out="value",
            scenario_counts=[1, 3],
        ).to_pandas()
        # stage 0: 2 entities × 1 col; stage 1: 2 entities × 3 cols.
        assert len(out) == 2 + 6
        fan = out[(out["stage_id"] == 1) & (out["ncs_id"] == 0)].sort_values(
            "scenario_id"
        )
        assert list(fan["scenario_id"]) == [0, 1, 2]
        assert list(fan["value"]) == [0.6, 0.6, 0.6]

    def test_probabilities_shape_and_sums(self) -> None:
        table = convert_scenario_probabilities(_StubVazoes(), _calendar()).to_pandas()
        sums = table.groupby("stage_id")["probability"].sum()
        assert sums.tolist() == pytest.approx([1.0, 1.0, 1.0])
        assert table["scenario_id"].min() == 0

    def test_bad_probability_sum_raises(self) -> None:
        vazoes = _StubVazoes()
        vazoes.probabilidades.loc[3, "probabilidade"] = 0.3
        with pytest.raises(ValueError, match="sum"):
            convert_scenario_probabilities(vazoes, _calendar())

    def test_terminal_fan_probabilities_ordered(self) -> None:
        # Terminal (1-based stage 3) fan: cenario 1 -> 0.6, cenario 2 -> 0.4,
        # returned ordered by 0-based scenario id.
        probs = terminal_fan_probabilities(_StubVazoes(), _calendar())
        assert probs == pytest.approx([0.6, 0.4])

    def test_terminal_fan_probabilities_bad_sum_raises(self) -> None:
        vazoes = _StubVazoes()
        vazoes.probabilidades.loc[3, "probabilidade"] = 0.3  # terminal sums 0.9
        with pytest.raises(ValueError, match="sum"):
            terminal_fan_probabilities(vazoes, _calendar())

    def test_pre_terminal_fan_raises(self) -> None:
        vazoes = _StubVazoes()
        vazoes.cenarios_gerados.loc[0, "estagio"] = 2
        with pytest.raises(ValueError, match="node-graph"):
            convert_external_inflows(
                _scenario_case(),
                _ID_MAP,
                vazoes=vazoes,
                effective=_effective_no_override(_hidr_frame()),
            )


class TestPipeline:
    def test_missing_deck_raises(self, tmp_path: Path) -> None:
        from novomodelo_bridge.decomp.pipeline import convert_decomp_case

        with pytest.raises(FileNotFoundError, match="caso.dat"):
            convert_decomp_case(tmp_path, tmp_path / "out")

    def test_discover_decomp_files_no_caso_raises_source_file_error(
        self, tmp_path: Path
    ) -> None:
        from novomodelo_bridge.core.errors import SourceFileError
        from novomodelo_bridge.decomp.files import discover_decomp_files

        with pytest.raises(SourceFileError) as excinfo:
            discover_decomp_files(tmp_path)

        exc = excinfo.value
        assert isinstance(exc, FileNotFoundError)
        assert exc.path == str(tmp_path)
        assert exc.field == "caso.dat"
        assert "caso.dat" in str(exc)

    def test_discover_decomp_files_no_dadger_raises_source_file_error(
        self, tmp_path: Path
    ) -> None:
        from novomodelo_bridge.core.errors import SourceFileError
        from novomodelo_bridge.decomp.files import discover_decomp_files

        (tmp_path / "caso.dat").write_text("rv0\n", encoding="latin-1")

        with pytest.raises(SourceFileError) as excinfo:
            discover_decomp_files(tmp_path)

        exc = excinfo.value
        assert isinstance(exc, FileNotFoundError)
        assert exc.path == str(tmp_path)
        assert exc.field == "dadger"
        assert "no dadger* file found" in str(exc)


class TestPhaseLabels:
    """``DECOMP_CONVERSION_PHASE_LABELS`` + the
    ``on_phase`` progress callback, mirroring the source model's own
    ``CONVERSION_PHASE_LABELS``/``on_phase`` wiring."""

    def test_decomp_conversion_phase_labels_is_the_expected_tuple(self) -> None:
        from novomodelo_bridge.decomp.pipeline import DECOMP_CONVERSION_PHASE_LABELS

        assert DECOMP_CONVERSION_PHASE_LABELS == (
            "Discovering deck",
            "Converting entities",
            "Converting scenarios",
            "Resolving bounds",
            "Converting constraints",
            "Writing outputs",
        )


class TestDecompCaseArtifacts:
    """The bundle ``_discover``/``_convert_core_entities``/
    ``_convert_scenarios`` thread across the discovery/entity/scenario phases,
    and the still-inline bounds/constraints/write phases read back off — must
    carry every field those five functions (plus the FCF importer,
    which reads ``config``/``initial_conditions``) depend on."""

    def test_field_contract(self) -> None:
        from novomodelo_bridge.decomp.pipeline import DecompCaseArtifacts

        field_names = {f.name for f in dataclasses.fields(DecompCaseArtifacts)}
        assert {
            "case",
            "id_map",
            "dadger",
            "calendar",
            "vazoes",
            "effective",
            "itaipu_operated",
            "fan_probabilities",
            "tx",
            "config",
            "stages_dict",
            "hydros_dict",
            "thermals_dict",
            "buses_doc",
            "lines_doc",
            "line_bounds",
            "initial_conditions",
            "deficit_cost",
            "has_travel_time",
            "fpha_codes",
        } <= field_names

        # _discover's own fields are required (no default); every later-phase
        # field defaults to None until _convert_core_entities/_convert_scenarios
        # populate it.
        artifacts = DecompCaseArtifacts(
            case=object(),
            id_map=_ID_MAP,
            dadger=object(),
            calendar=[],
            vazoes=object(),
            effective=object(),
            itaipu_operated=False,
            fan_probabilities=[1.0],
            tx=0.1,
        )
        assert artifacts.config is None
        assert artifacts.initial_conditions is None
        assert artifacts.stages_dict is None
        assert artifacts.hydros_dict is None
        assert artifacts.thermals_dict is None
        assert artifacts.lines_doc is None
        assert artifacts.line_bounds is None

        artifacts.config = {"a": 1}
        artifacts.initial_conditions = {"storage": []}
        artifacts.stages_dict = {"stages": []}
        artifacts.hydros_dict = {"hydros": []}
        artifacts.thermals_dict = {"thermals": []}
        artifacts.lines_doc = {"lines": []}
        artifacts.line_bounds = pa.table({"line_id": pa.array([], type=pa.int32())})
        assert isinstance(artifacts.config, dict)
        assert isinstance(artifacts.initial_conditions, dict)
        assert isinstance(artifacts.stages_dict, dict)
        assert isinstance(artifacts.hydros_dict, dict)
        assert isinstance(artifacts.thermals_dict, dict)
        assert isinstance(artifacts.id_map, DecompIdMap)
        assert isinstance(artifacts.lines_doc, dict)
        assert isinstance(artifacts.line_bounds, pa.Table)


class TestEmissionCheckWiring:
    """The post-emission self-checks run inside
    ``convert_decomp_case``, before the constraint writes."""

    def test_run_and_gate_raises_on_duplicate_bound_row(self) -> None:
        """tier-1 (no ``example/`` deck): drives the same
        ``emission_checks.run_and_gate`` call ``_convert_decomp_case_impl``
        now makes, over a synthetic in-memory bounds table
        carrying one duplicate ``(hydro_id, stage_id, block_id, column)`` row
        (novomodelo rule 36, ``check_bound_row_uniqueness``) — the gate must raise
        ``EmissionCheckError``, and that exception must still satisfy
        ``isinstance(exc, ValueError)`` for any existing
        ``pytest.raises(ValueError)`` call site."""
        from novomodelo_bridge.core import emission_checks

        hydro_bounds = pa.table(
            {
                "hydro_id": pa.array([0, 0], type=pa.int32()),
                "stage_id": pa.array([1, 1], type=pa.int32()),
                "block_id": pa.array([None, None], type=pa.int32()),
                "min_outflow_m3s": pa.array([5.0, 5.0], type=pa.float64()),
            }
        )
        bound_families = [
            emission_checks.BoundFamily("Hydro", "hydro_id", hydro_bounds)
        ]

        def _run_emission_checks() -> None:
            emission_checks.check_bound_row_uniqueness(bound_families)

        with pytest.raises(emission_checks.EmissionCheckError) as excinfo:
            emission_checks.run_and_gate(_run_emission_checks)
        assert isinstance(excinfo.value, ValueError)

    def test_decomp_shaped_violation_flips_convert_status_through_the_same_function(
        self,
    ) -> None:
        """A violation built from DECOMP-shaped artifacts still flips the
        verdict via ``cli.app._convert_status`` — the single function both
        pipelines' convert verdicts key off, not a bare inspection of
        the diagnostic."""
        from novomodelo_bridge.cli.verdict import _convert_status
        from novomodelo_bridge.core import diagnostics as dx
        from novomodelo_bridge.core.emission_checks import check_hydro_bounds_no_raising

        hydros = {
            "hydros": [
                {
                    "id": 0,
                    "generation": {
                        "max_turbined_m3s": 100.0,
                        "max_generation_mw": 50.0,
                    },
                    "unit_groups": [
                        {"max_turbined_m3s": 100.0, "max_generation_mw": 50.0}
                    ],
                }
            ]
        }
        hydro_bounds = pa.table(
            {
                "hydro_id": pa.array([0], type=pa.int32()),
                "stage_id": pa.array([1], type=pa.int32()),
                "max_turbined_m3s": pa.array([999.0], type=pa.float64()),
            }
        )

        with dx.collect() as collected:
            check_hydro_bounds_no_raising(hydros, hydro_bounds)

        assert _convert_status(collected, success="ok") == "error"
        assert _convert_status([], success="ok") == "ok"


class TestBoundAccumulatorWiring:
    """The E2 accumulator is the *single* merge point
    for every per-entity bound, and the new pumping family is wired
    end-to-end."""

    def test_pipeline_source_has_no_concat_tables_for_entity_bounds(self) -> None:
        """The naive ``pa.concat_tables`` the old hydro-bounds combine used
        is gone — every ENTITY bound table pipeline.py writes comes from
        ``bounds_accumulator.resolve`` + ``build_bound_tables``.
        The generic emitters' own ``pa.concat_tables(generic_bound_tables)`` call is a
        different, sanctioned merge (folding the RE/RHQ/RHV/RHE
        generic-constraint bound tables into one, mirroring the source
        model's own generic-constraints pipeline) — not a regression of the
        entity-bounds combine this test guards against."""
        import inspect

        from novomodelo_bridge.decomp import pipeline

        source = inspect.getsource(pipeline)
        assert source.count("concat_tables") == 1
        assert "pa.concat_tables(generic_bound_tables)" in source

    def test_no_pumping_records_writes_no_pumping_bounds_file(
        self, tmp_path: Path
    ) -> None:
        """A deck with no QBOM records writes no ``pumping_bounds.parquet``
        at all (the negative half) — the fully-synthetic mock deck
        (``_run_cadastro_pipeline``) carries none."""
        dst = _run_cadastro_pipeline(tmp_path, ac_volmax_frame=None)
        assert not (dst / "constraints" / "pumping_bounds.parquet").exists()


class TestCli:
    def test_convert_decomp_invokes_pipeline(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with patch(
            "novomodelo_bridge.decomp.pipeline.convert_decomp_case"
        ) as mock_convert:
            result = runner.invoke(
                app,
                # --no-fcf: the empty tmp_path is not a discoverable deck, and
                # this test only asserts the pipeline is invoked (convert is
                # mocked), so skip the default boundary-FCF discovery/import.
                [
                    "convert",
                    "decomp",
                    str(tmp_path),
                    str(tmp_path / "out"),
                    "--force",
                    "--no-fcf",
                ],
            )
        assert result.exit_code == 0
        mock_convert.assert_called_once()
        assert mock_convert.call_args.kwargs["force"] is True

    def test_convert_decomp_failure_exits_one(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with patch(
            "novomodelo_bridge.decomp.pipeline.convert_decomp_case",
            side_effect=FileNotFoundError("caso.dat not found"),
        ):
            result = runner.invoke(
                app, ["convert", "decomp", str(tmp_path), str(tmp_path / "out")]
            )
        assert result.exit_code == 1

    def test_convert_decomp_emission_check_error_exits_one(
        self, tmp_path: Path
    ) -> None:
        """The ``ValueError`` ``convert_decomp_case`` raises on an
        ERROR-severity post-emission self-check finding (Finding 1) is caught
        by the same except tuple as a missing/existing-destination failure —
        no dedicated CLI branch is needed."""
        runner = CliRunner()
        with patch(
            "novomodelo_bridge.decomp.pipeline.convert_decomp_case",
            side_effect=ValueError(
                "DECOMP conversion failed 1 post-emission self-check error(s)"
            ),
        ):
            result = runner.invoke(
                app, ["convert", "decomp", str(tmp_path), str(tmp_path / "out")]
            )
        assert result.exit_code == 1


# A fully synthetic Tier-1 harness for ``convert_decomp_case`` — no real
# deck under ``example/``. Every converter the pipeline calls other than the
# five wired here (``build_effective_cadastro``, ``convert_initial_storage``,
# ``convert_hydros``, ``convert_hydro_bounds``, ``convert_storage_bounds``)
# is patched to a canned return value, so the test exercises only the
# orchestration: threading the effective cadastro through, combining
# ``hydro_bounds`` with the storage-bounds overlay, and the resolution-report
# summary diagnostic.


def _cadastro_plant_row(
    name: str, sub: int, jusante: int, vmin: float, vmax: float, posto: int = 0
) -> dict:
    return {
        "nome_usina": name,
        "submercado": sub,
        "codigo_usina_jusante": jusante,
        "posto": posto,
        "desvio": 0,
        "volume_minimo": vmin,
        "volume_maximo": vmax,
        # emit_rhe_generics computes a per-stage integrated ρ_acum
        # for every operated plant (not only ones an HE record references), so
        # this synthetic fixture needs a regulation type ("M" -- a reservoir --
        # never branches through the run-of-river collapse the rest of this
        # fixture is already validated against) and a reference volume, even
        # though no HE record here ever reads either.
        "tipo_regulacao": "M",
        "volume_referencia": vmin + 0.5 * (vmax - vmin),
        "numero_conjuntos_maquinas": 1,
        "maquinas_conjunto_1": 2,
        "vazao_nominal_conjunto_1": 100.0,
        "potencia_nominal_conjunto_1": 50.0,
        "teif": 0.0,
        "ip": 0.0,
        "a0_volume_cota": 100.0,
        "a1_volume_cota": 0.0,
        "a2_volume_cota": 0.0,
        "a3_volume_cota": 0.0,
        "a4_volume_cota": 0.0,
        # cota→area coefficients: this reservoir is FPHA-eligible (non-degenerate
        # cota + rho_esp > 0), so convert_hydro_geometry reads these.
        "a0_cota_area": 1.0,
        "a1_cota_area": 0.0,
        "a2_cota_area": 0.0,
        "a3_cota_area": 0.0,
        "a4_cota_area": 0.0,
        "canal_fuga_medio": 20.0,
        "produtibilidade_especifica": 0.009,
        "tipo_perda": 0,
        "perdas": 0.0,
    }


def _cadastro_hidr_frame() -> pd.DataFrame:
    """Two plants, codes 1 and 2, both on bus (submercado) 1, no cascade."""
    df = pd.DataFrame(
        {
            1: _cadastro_plant_row("PLANT_ONE", 1, 0, 20.0, 100.0, posto=11),
            2: _cadastro_plant_row("PLANT_TWO", 1, 0, 10.0, 50.0, posto=12),
        }
    ).T
    df.index.name = "codigo_usina"
    return df


def _cadastro_uh_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "codigo_usina": 1,
                "volume_inicial": 50.0,
                "vazao_defluente_minima": None,
                "volume_morto_inicial": None,
                # convert_decomp_case builds hydro_to_ree off this
                "codigo_ree": 1,
            },
            {
                "codigo_usina": 2,
                "volume_inicial": 50.0,
                "vazao_defluente_minima": None,
                "volume_morto_inicial": None,
                "codigo_ree": 1,
            },
        ]
    )


_CADASTRO_ID_MAP = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1, 2))


class _CadastroDadger:
    """Mock ``Dadger`` covering only what the real consumers below need:
    ``.tx.taxa`` (the discount rate), ``.uh(df=True)`` (the operated
    registrations both ``convert_hydros``/``convert_initial_storage`` read),
    and ``.ac(...)`` — returning *ac_volmax_frame* for the ``ACVOLMAX``
    subtype (the real idecomp representation: string month abbreviation,
    float ``semana``/``ano``) and ``None`` (no override) for every other
    ``AC`` subtype, including ``ACVOLMIN`` and the four machine-configuration
    overrides ``convert_hydros`` also probes.
    """

    class _Tx:
        taxa = 12.0

    class _Data:
        """Register API `resolve_cvar` probes: no ``AR`` register (this mock
        deck is risk-neutral), so CVaR resolution yields expectation."""

        @staticmethod
        def of_type(_register: type) -> list[object]:
            return []

    def __init__(self, ac_volmax_frame: pd.DataFrame | None) -> None:
        self.tx = self._Tx()
        self.data = self._Data()
        self._uh = _cadastro_uh_frame()
        self._ac_volmax_frame = ac_volmax_frame

    def uh(self, df: bool = False) -> pd.DataFrame:  # noqa: ARG002
        return self._uh

    def vi(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return None

    def ve(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return None

    def ti(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return None

    def ac(
        self,
        codigo_usina: int | None = None,  # noqa: ARG002
        modificacao: type | None = None,
        df: bool = False,  # noqa: ARG002
    ) -> pd.DataFrame | None:
        from idecomp.decomp.modelos.dadger import ACVOLMAX

        if modificacao is ACVOLMAX:
            return self._ac_volmax_frame
        return None


def _run_cadastro_pipeline(
    tmp_path: Path,
    ac_volmax_frame: pd.DataFrame | None,
    *,
    to_generic: tuple[ConstraintRecord, ...] = (),
    unreadable_electrical: tuple[object, ...] = (),
    libs_electrical: object | None = None,
    diagnostics_out: list[Diagnostic] | None = None,
    dry_run: bool = False,
    report_out: list[ConversionReport] | None = None,
    gnl_emission: GnlEmission | None = None,
    convert_gnl_mock_out: list[MagicMock] | None = None,
) -> Path:
    """Run ``convert_decomp_case`` against the fully synthetic mock deck
    above, patching every converter not under test to a canned
    return value. Returns the case directory.

    *to_generic* feeds ``constraint_registers.read_constraints``'s
    ``ConstraintCensus.to_generic`` — every RE/RHQ/RHV/RHE record the
    generic-constraint emitters consume, none of which the mock deck's
    ``Dadger`` can produce for real (it exposes no
    ``RE``/``HQ``/``HV``/``HE``/``CM`` accessors).
    *unreadable_electrical*/*libs_electrical* feed the E1
    detection helpers the same way — both otherwise patched to their
    empty/absent default so this shared fixture keeps regressing the
    bound-combine logic it was built for, undisturbed.

    *diagnostics_out*: ``convert_decomp_case`` now owns its own
    top-level ``dx.collect()``, so a caller-side ``with dx.collect():``
    wrapped around this helper would be shadowed and see nothing. A caller
    that needs the run's diagnostics passes a list here; it is extended in
    place with ``report.diagnostics`` after the (patched) conversion returns.

    *dry_run* threads straight through to ``convert_decomp_case``.
    *report_out*, mirroring *diagnostics_out*'s out-param shape, lets a caller
    inspect the full returned ``ConversionReport`` (e.g. ``would_write_paths``)
    without changing this helper's ``Path``-only return type.

    *gnl_emission* drives the pipeline's GNL wiring
    block (``pipeline.py``'s ``files.dadgnl is not None`` branch), which every
    other caller skips by leaving *gnl_emission* at its ``None`` default. When
    supplied: ``files.dadgnl`` points at a placeholder path so the branch is
    entered; ``Dadger.Dadgnl.read``/``anticipated_conv.read_gnl_model`` are
    patched to a sentinel non-``None`` model (their own decode logic is out of
    scope — the ``test_anticipated`` module owns it);
    ``anticipated_conv.convert_gnl`` is patched to return *gnl_emission*
    verbatim; and ``thermal_conv.convert_thermals`` is swapped from the empty
    default to a single CT thermal (id ``0``) so ``first_thermal_id`` (``max(id)
    + 1``) resolves to ``1`` instead of raising on an empty sequence.
    *convert_gnl_mock_out*, mirroring *diagnostics_out*'s out-param shape, lets
    a caller recover the ``convert_gnl`` mock (and thus its call args) after
    the patched run — populated only alongside *gnl_emission*.
    """
    from novomodelo_bridge.decomp.files import DecompFiles
    from novomodelo_bridge.decomp.pipeline import convert_decomp_case

    files = DecompFiles(
        revision="rv0",
        dadger=Path("unused/dadger.rv0"),
        vazoes=Path("unused/vazoes.rv0"),
        hidr=Path("unused/hidr.dat"),
        dadgnl=None,
        renovaveis=None,
        polinjus=None,
    )
    dadger = _CadastroDadger(ac_volmax_frame)
    hidr = _cadastro_hidr_frame()
    calendar = _calendar()
    case = make_decomp_case(
        files,
        dadger=dadger,
        hidr=hidr,
        id_map=_CADASTRO_ID_MAP,
        calendar=calendar,
        renovaveis=None,
        # A non-None sentinel enters convert_decomp_case's GNL wiring branch
        # (gated on ``case.dadgnl is not None``); the model's own decode is
        # patched below via ``anticipated_conv.read_gnl_model``, so the
        # sentinel's identity never matters.
        dadgnl=object() if gnl_emission is not None else None,
        polinjus=None,
        libs_restricao_eletrica=None,
    )

    productivity_table = pa.table(
        {"equivalent_productivity_mw_per_m3s": pa.array([0.5, 0.6], type=pa.float64())}
    )
    load_stats_table = pa.table(
        {
            "bus_id": pa.array([0, 0, 0], type=pa.int32()),
            "stage_id": pa.array([0, 1, 2], type=pa.int32()),
            "mean_mw": pa.array([10.0, 10.0, 10.0], type=pa.float64()),
        }
    )
    ncs_stats_table = pa.table(
        {
            "ncs_id": pa.array([0, 0, 0], type=pa.int32()),
            "stage_id": pa.array([0, 1, 2], type=pa.int32()),
            "mean": pa.array([5.0, 5.0, 5.0], type=pa.float64()),
        }
    )
    external_inflow_table = pa.table(
        {
            "hydro_id": pa.array([], type=pa.int32()),
            "stage_id": pa.array([], type=pa.int32()),
            "scenario_id": pa.array([], type=pa.int32()),
            "value_m3s": pa.array([], type=pa.float64()),
        }
    )
    # The pre-override baseline hydro_bounds: one RQ/UH-derived min-outflow
    # contribution — the combine logic (one resolve() +
    # build_bound_tables() pass over this plus the real convert_storage_bounds
    # output) is what is under test, not convert_hydro_bounds' own RQ/UH logic
    # (out of scope).
    baseline_hydro_bounds = [
        BoundContribution(
            family="hydro",
            entity_id=0,
            stage_id=0,
            block_id=None,
            axis="outflow",
            lower=5.0,
            upper=None,
            contributor="RQ",
        )
    ]

    patches: dict[str, object] = {
        "novomodelo_bridge.decomp.pipeline.DecompCase.from_directory": case,
        "novomodelo_bridge.decomp.pipeline.Vazoes.read": object(),
        "novomodelo_bridge.decomp.pipeline.scenarios_conv.terminal_fan_probabilities": [
            1.0
        ],
        "novomodelo_bridge.decomp.pipeline.config_conv.convert_config": {},
        "novomodelo_bridge.decomp.pipeline.network_conv._bus_deficit_costs": {},
        "novomodelo_bridge.decomp.pipeline"
        ".hydro_conv.convert_energy_productivity": productivity_table,
        "novomodelo_bridge.decomp.pipeline.network_conv.convert_buses": {"buses": []},
        "novomodelo_bridge.decomp.pipeline.network_conv.convert_lines": (
            {"lines": []},
            _LINE_BOUNDS_SCHEMA.empty_table(),
        ),
        "novomodelo_bridge.decomp.pipeline.network_conv.convert_pumping_stations": {
            "pumping_stations": []
        },
        "novomodelo_bridge.decomp.pipeline.thermal_conv.convert_thermals": {
            "thermals": []
        },
        "novomodelo_bridge.decomp.pipeline.ncs_conv.convert_non_controllable_sources": {
            "non_controllable_sources": []
        },
        "novomodelo_bridge.decomp.pipeline"
        ".scenarios_conv.convert_external_inflows": external_inflow_table,
        "novomodelo_bridge.decomp.pipeline"
        ".scenarios_conv.convert_recent_observation_windows": [],
        "novomodelo_bridge.decomp.pipeline.load_conv.convert_load_stats": load_stats_table,
        "novomodelo_bridge.decomp.pipeline.load_conv.convert_load_factors": {},
        "novomodelo_bridge.decomp.pipeline.ncs_conv.convert_ncs_stats": ncs_stats_table,
        "novomodelo_bridge.decomp.pipeline.ncs_conv.convert_ncs_factors": {},
        "novomodelo_bridge.decomp.pipeline.thermal_conv.convert_thermal_bounds": (
            ThermalBounds(generation=[], cost=_THERMAL_COST_SCHEMA.empty_table())
        ),
        "novomodelo_bridge.decomp.pipeline"
        ".bounds_conv.convert_hydro_bounds": baseline_hydro_bounds,
        "novomodelo_bridge.decomp.pipeline.hydro_conv.convert_hydro_group_availability": {},
        "novomodelo_bridge.decomp.pipeline.contracts_conv.read_contracts": [],
        # The mock deck (_CadastroDadger) exposes no
        # RE/HQ/HV/UE accessors, so the special-constraint census and the
        # pumping id map must be patched too — the bound-combine
        # logic is under test, not the special-constraint reader or the
        # pumping id map (both out of scope, exercised elsewhere).
        "novomodelo_bridge.decomp.pipeline.constraint_registers.read_constraints": (
            ConstraintCensus(by_family={}, to_bounds=(), to_generic=to_generic)
        ),
        "novomodelo_bridge.decomp.pipeline.network_conv.pumping_station_id_map": {},
        # The mock deck exposes no real files (DecompCase.from_directory
        # is patched wholesale above, so no real file I/O happens anywhere in this
        # fixture) — the E1 detection helpers read the raw deck files directly, so
        # they must be patched here too, the same way the special-constraint
        # reader above is.
        "novomodelo_bridge.decomp.pipeline"
        ".constraint_registers.detect_unreadable_electrical": list(
            unreadable_electrical
        ),
        "novomodelo_bridge.decomp.pipeline.constraint_registers.detect_libs_electrical": (
            libs_electrical
        ),
    }
    if gnl_emission is not None:
        # Route the GNL wiring block through
        # its own patches rather than the empty/absent default above —
        # convert_gnl's own decode/placement logic stays out of scope
        # (the ``test_anticipated`` module owns it), only the routing of its
        # *return value* into the written case files is under test here.
        # ``case.dadgnl`` is already the non-None sentinel set above, so only
        # its downstream decode (read_gnl_model) needs patching here.
        patches["novomodelo_bridge.decomp.pipeline.anticipated_conv.read_gnl_model"] = (
            object()
        )
        patches["novomodelo_bridge.decomp.pipeline.anticipated_conv.convert_gnl"] = (
            gnl_emission
        )
        patches["novomodelo_bridge.decomp.pipeline.thermal_conv.convert_thermals"] = {
            "thermals": [{"id": 0}]
        }
    with ExitStack() as stack:
        entered: dict[str, MagicMock] = {}
        for target, value in patches.items():
            entered[target] = stack.enter_context(patch(target, return_value=value))
        if convert_gnl_mock_out is not None:
            convert_gnl_mock_out.append(
                entered[
                    "novomodelo_bridge.decomp.pipeline.anticipated_conv.convert_gnl"
                ]
            )
        dst = tmp_path / "case"
        report = convert_decomp_case(Path("unused-src"), dst, dry_run=dry_run)
    if diagnostics_out is not None:
        diagnostics_out.extend(report.diagnostics)
    if report_out is not None:
        report_out.append(report)
    return dst


# Two hand-built ``GnlEmission`` fixtures driving
# ``TestGnlWiring`` below via ``_run_cadastro_pipeline``'s ``gnl_emission``
# param — no real ``dadgnl`` deck, no ``convert_gnl`` execution (it is
# mocked). "Populated" carries a non-empty post-study ``thermal_bounds``
# carrier; "empty" mirrors a GS-calendar-only plant (e.g. PSERGIPE I) whose
# post-study stages are all class-4 já-comandada, so none get a
# ``thermal_bounds`` row.
_POPULATED_GNL_EMISSION = GnlEmission(
    thermals=[
        {"id": 1, "name": "GNL A", "anticipated_config": {"lead_time_hours": 168.0}},
        {"id": 2, "name": "GNL B", "anticipated_config": {"lead_time_hours": 336.0}},
    ],
    past_anticipated_commitments=[
        {"thermal_id": 1, "stage_id": 0, "mw": 50.0},
        {"thermal_id": 2, "stage_id": 0, "mw": 30.0},
    ],
    post_study_stages={
        "stages": [{"id": 12, "start_date": "2027-07-06"}],
        "thermal_bounds": [{"thermal_id": 1, "stage_id": 12, "max_mw": 50.0}],
    },
)

_EMPTY_THERMAL_BOUNDS_GNL_EMISSION = GnlEmission(
    thermals=[
        {"id": 1, "name": "GNL A", "anticipated_config": {"lead_time_hours": 168.0}},
    ],
    past_anticipated_commitments=[{"thermal_id": 1, "stage_id": 0, "mw": 20.0}],
    post_study_stages={
        "stages": [{"id": 12, "start_date": "2027-07-06"}],
        "thermal_bounds": [],
    },
)


class TestGnlWiring:
    """Regression-guard the pre-existing GNL wiring
    block. ``convert_gnl``
    synthesises a GS-driven post-study calendar and free (not just pinned)
    forward deliveries, but the pipeline call site already carried the
    unchanged ``GnlEmission`` shape to disk — no tier-1 test exercised it,
    since the mock deck's ``DecompFiles`` always carried ``dadgnl=None``.
    These tests drive the block via ``_run_cadastro_pipeline``'s
    ``gnl_emission`` param without a real ``dadgnl`` deck or a real
    ``convert_gnl`` call (mocked)."""

    def test_convert_gnl_called_with_first_thermal_id_bus_id_of_and_calendar(
        self, tmp_path: Path
    ) -> None:
        """``first_thermal_id`` resolves to 1 past the single mocked CT
        thermal (id 0); ``bus_id_of`` is the run's own ``id_map.bus_id``;
        ``calendar`` is the operative calendar ``stages.json`` is written from."""
        convert_gnl_mock_out: list[MagicMock] = []
        dst = _run_cadastro_pipeline(
            tmp_path,
            ac_volmax_frame=None,
            gnl_emission=_POPULATED_GNL_EMISSION,
            convert_gnl_mock_out=convert_gnl_mock_out,
        )

        [convert_gnl_mock] = convert_gnl_mock_out
        convert_gnl_mock.assert_called_once()
        call_kwargs = convert_gnl_mock.call_args.kwargs
        assert call_kwargs["first_thermal_id"] == 1
        assert call_kwargs["bus_id_of"] == _CADASTRO_ID_MAP.bus_id
        written_stages = json.loads((dst / "stages.json").read_text())["stages"]
        assert [
            (s.start_date.isoformat(), s.end_date.isoformat())
            for s in call_kwargs["calendar"]
        ] == [(s["start_date"], s["end_date"]) for s in written_stages]

    def test_populated_emission_routes_thermals_and_past_boundary(
        self, tmp_path: Path
    ) -> None:
        """Every created GNL thermal id lands in ``thermals.json``, sorted
        ascending; the past boundary lands in ``initial_conditions.json``
        verbatim, with no ``future_anticipated_deliveries`` key (retired);
        the post-study ``thermal_bounds`` carrier is still written and
        non-empty."""
        dst = _run_cadastro_pipeline(
            tmp_path, ac_volmax_frame=None, gnl_emission=_POPULATED_GNL_EMISSION
        )

        thermal_ids = [
            t["id"]
            for t in json.loads((dst / "system" / "thermals.json").read_text())[
                "thermals"
            ]
        ]
        assert thermal_ids == sorted(thermal_ids)
        assert {1, 2}.issubset(set(thermal_ids))

        initial_conditions = json.loads((dst / "initial_conditions.json").read_text())
        assert (
            initial_conditions["past_anticipated_commitments"]
            == _POPULATED_GNL_EMISSION.past_anticipated_commitments
        )
        assert "future_anticipated_deliveries" not in initial_conditions

        post_study = json.loads((dst / "post_study_stages.json").read_text())
        assert post_study["thermal_bounds"]

    def test_empty_thermal_bounds_still_writes_post_study_stages(
        self, tmp_path: Path
    ) -> None:
        """A GS-calendar-only plant whose post-study stages are all class-4
        (empty ``post_study_stages["thermal_bounds"]``) still gets
        ``post_study_stages.json`` (non-``None`` calendar);
        ``initial_conditions.json`` carries no
        ``future_anticipated_deliveries`` key (retired)."""
        dst = _run_cadastro_pipeline(
            tmp_path,
            ac_volmax_frame=None,
            gnl_emission=_EMPTY_THERMAL_BOUNDS_GNL_EMISSION,
        )

        post_study = json.loads((dst / "post_study_stages.json").read_text())
        assert post_study["thermal_bounds"] == []
        expected_post_study = _EMPTY_THERMAL_BOUNDS_GNL_EMISSION.post_study_stages
        assert expected_post_study is not None
        assert post_study["stages"] == expected_post_study["stages"]

        initial_conditions = json.loads((dst / "initial_conditions.json").read_text())
        assert "future_anticipated_deliveries" not in initial_conditions

    def test_summary_log_names_only_thermal_count(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The summary log names only the emitted GNL thermal count -- no
        "future anticipated deliver(y/ies)" clause (the retired free lane)
        and no "post-horizon deliver(y/ies)" wording (an older phrasing)."""
        with caplog.at_level(logging.INFO, logger="novomodelo_bridge.decomp.pipeline"):
            _run_cadastro_pipeline(
                tmp_path, ac_volmax_frame=None, gnl_emission=_POPULATED_GNL_EMISSION
            )

        thermal_matches = [
            r
            for r in caplog.records
            if "emitted 2 GNL anticipated thermal(s) from dadgnl" in r.message
        ]
        assert len(thermal_matches) == 1
        assert not any(
            "future anticipated deliver" in r.message for r in caplog.records
        )
        assert not any("post-horizon deliver" in r.message for r in caplog.records)


class TestCadastroPipelineWiring:
    """``build_effective_cadastro`` threads into
    ``convert_decomp_case`` and the storage-bounds overlay folds into the
    same ``hydro_bounds.parquet`` the RQ/UH minimum-outflow rows populate."""

    def test_no_override_regresses_hydro_bounds_and_entity_output(
        self, tmp_path: Path
    ) -> None:
        """No ``AC`` volume record: the combined table equals the pre-override
        baseline (no storage rows), and ``initial_conditions.json`` /
        ``system/hydros.json`` reflect the base registry values unchanged."""
        dst = _run_cadastro_pipeline(tmp_path, ac_volmax_frame=None)

        table = pq.read_table(dst / "constraints" / "hydro_bounds.parquet")
        assert table.num_rows == 1
        assert table["min_outflow_m3s"].to_pylist() == [5.0]
        assert table["min_storage_hm3"].to_pylist() == [None]
        assert table["max_storage_hm3"].to_pylist() == [None]

        hydros = json.loads((dst / "system" / "hydros.json").read_text())["hydros"]
        reservoirs = {h["id"]: h["reservoir"] for h in hydros}
        assert reservoirs[0] == {"min_storage_hm3": 20.0, "max_storage_hm3": 100.0}
        assert reservoirs[1] == {"min_storage_hm3": 10.0, "max_storage_hm3": 50.0}

        storage = json.loads((dst / "initial_conditions.json").read_text())["storage"]
        values = {row["hydro_id"]: row["value_hm3"] for row in storage}
        # 50% of the base [20, 100] / [10, 50] ranges.
        assert values[0] == pytest.approx(60.0)
        assert values[1] == pytest.approx(30.0)

    def test_initial_conditions_json_stamps_schema_as_first_key(
        self, tmp_path: Path
    ) -> None:
        """``$schema`` is now stamped on DECOMP ``initial_conditions.json`` too
        (a previously deferred twin-track asymmetry), first key to match the
        source model's own key order, pinned to the registry."""
        dst = _run_cadastro_pipeline(tmp_path, ac_volmax_frame=None)

        doc = json.loads((dst / "initial_conditions.json").read_text())
        assert next(iter(doc)) == "$schema"
        assert doc["$schema"] == novomodelo_schemas.schema_url_for(
            "initial_conditions.json"
        )

    def test_temporal_override_adds_storage_rows_and_raises_the_entity_envelope(
        self, tmp_path: Path
    ) -> None:
        """An ``ACVOLMAX`` row raising plant 1's ``volume_maximo`` to 250.0
        from the final stage forward widens its entity envelope to 250.0
        (``system/hydros.json``) and adds override rows to the combined
        ``hydro_bounds`` table. Per Rule A (``storage_envelope``):
        the raised (final) stage itself now *equals* the widened envelope and
        needs no override row; the earlier stages, which still sit at the
        narrower pre-raise ceiling, are what differ from it and get the
        override rows — non-null ``max_storage_hm3``, null ``min_outflow_m3s``.
        """
        ac_volmax_frame = pd.DataFrame(
            [
                {
                    "codigo_usina": 1,
                    "volume": 250.0,
                    "mes": "AGO",
                    "semana": 0.0,
                    "ano": 2026.0,
                }
            ]
        )
        dst = _run_cadastro_pipeline(tmp_path, ac_volmax_frame=ac_volmax_frame)

        table = pq.read_table(dst / "constraints" / "hydro_bounds.parquet")
        storage_rows = table.to_pylist()
        storage_only = [
            row
            for row in storage_rows
            if row["max_storage_hm3"] is not None and row["min_outflow_m3s"] is None
        ]
        assert len(storage_only) >= 1
        assert all(row["max_storage_hm3"] == 100.0 for row in storage_only)
        # The pre-override baseline row (min_outflow, no storage columns)
        # is still present, untouched by the combine.
        outflow_only = [row for row in storage_rows if row["min_outflow_m3s"] == 5.0]
        assert len(outflow_only) == 1

        hydros = json.loads((dst / "system" / "hydros.json").read_text())["hydros"]
        reservoirs = {h["id"]: h["reservoir"] for h in hydros}
        assert reservoirs[0]["max_storage_hm3"] == pytest.approx(250.0)

    def test_temporal_override_emits_exactly_one_summary_diagnostic(
        self, tmp_path: Path
    ) -> None:
        """The resolution-report summary is a single INFO diagnostic naming
        ``volume_maximo`` among the applied overrides.

        ``convert_decomp_case`` now owns its own top-level
        ``dx.collect()``, so this reads the diagnostics via
        ``_run_cadastro_pipeline``'s ``diagnostics_out`` rather than an outer
        ``dx.collect()`` (which would be shadowed and see nothing)."""
        from novomodelo_bridge.core import diagnostics as dx

        ac_volmax_frame = pd.DataFrame(
            [
                {
                    "codigo_usina": 1,
                    "volume": 250.0,
                    "mes": "AGO",
                    "semana": 0.0,
                    "ano": 2026.0,
                }
            ]
        )
        collected: list[dx.Diagnostic] = []
        _run_cadastro_pipeline(
            tmp_path, ac_volmax_frame=ac_volmax_frame, diagnostics_out=collected
        )

        cadastro_diagnostics = [
            d for d in collected if d.code == "cadastro-overrides-applied"
        ]
        assert len(cadastro_diagnostics) == 1
        assert cadastro_diagnostics[0].severity is dx.Severity.INFO
        assert "volume_maximo" in cadastro_diagnostics[0].summary


# Synthetic RE/RHQ/RHV/RHE ``to_generic`` records for
# ``TestGenericConstraintWiring`` below. Every record uses plant codes 1/2 —
# the shared ``_cadastro_hidr_frame``/``_cadastro_uh_frame`` fixture's two
# plants, both members of REE 1 (``_cadastro_uh_frame``'s ``codigo_ree``) —
# and is deliberately multi-term (RE/HQ/HV) or the whole-REE energy sum (HE),
# so none of the four ever lowers to an entity bound (``lowers_to_bound``);
# every one lands in ``census.to_generic``, the generic-constraint territory.
_N_BLOCKS = 3  # matches _calendar()'s 3-block stages


def _synthetic_re_record() -> ConstraintRecord:
    """Two-hydro-generation RE record, upper-bounded only."""
    return ConstraintRecord(
        family="RE",
        constraint_id=901,
        stage_start=0,
        stage_end=2,
        terms=(
            ConstraintTerm(code=1, coefficient=1.0, variable="generation"),
            ConstraintTerm(code=2, coefficient=1.0, variable="generation"),
        ),
        bounds={
            stage: StageBounds(lower=(None,) * _N_BLOCKS, upper=(100.0,) * _N_BLOCKS)
            for stage in range(3)
        },
        per_block=True,
    )


def _synthetic_rhq_record() -> ConstraintRecord:
    """Two-term HQ flow-mix record, bounded on both sides."""
    return ConstraintRecord(
        family="HQ",
        constraint_id=902,
        stage_start=0,
        stage_end=2,
        terms=(
            ConstraintTerm(code=1, coefficient=1.0, variable="QDEF"),
            ConstraintTerm(code=2, coefficient=-1.0, variable="QDEF"),
        ),
        bounds={
            stage: StageBounds(lower=(-50.0,) * _N_BLOCKS, upper=(50.0,) * _N_BLOCKS)
            for stage in range(3)
        },
        per_block=True,
    )


def _synthetic_rhv_record() -> ConstraintRecord:
    """Two-``VARM`` (multi-plant stored-volume) HV record, lower-bounded only."""
    return ConstraintRecord(
        family="HV",
        constraint_id=903,
        stage_start=0,
        stage_end=2,
        terms=(
            ConstraintTerm(code=1, coefficient=1.0, variable="VARM"),
            ConstraintTerm(code=2, coefficient=1.0, variable="VARM"),
        ),
        bounds={stage: StageBounds(lower=(5.0,), upper=(None,)) for stage in range(3)},
        per_block=False,
    )


#: The RHE record's percentage limit — deliberately a *percentage*
#: (``tipo_limite=2``) rather than an absolute MWmes limit, so its RHS
#: genuinely depends on ρ_acum (an absolute limit would not exercise the
#: "LHS sigil == the RHS ρ_acum it drove" wiring at all).
_RHE_LIMITE_PCT = 40.0


def _synthetic_rhe_record() -> ConstraintRecord:
    """One-REE HE record, a percentage-of-EARM limit pulling in both plants
    of REE 1 — both monthly-regulating (``tipo_regulacao="M"``) reservoirs
    per :func:`_cadastro_plant_row`, so both are participating reservoirs."""
    return ConstraintRecord(
        family="HE",
        constraint_id=904,
        stage_start=0,
        stage_end=2,
        terms=(ConstraintTerm(code=1, coefficient=1.0, variable="energy"),),
        bounds={
            stage: StageBounds(lower=(_RHE_LIMITE_PCT,), upper=(None,))
            for stage in range(3)
        },
        per_block=False,
        tipo_limite=2,
        he_meta=HeMeta(valor_penalidade=500.0),
    )


def _all_synthetic_generics() -> tuple[ConstraintRecord, ...]:
    """RE, then the combined RHQ/RHV family, then RHE — the E7 emission
    order, in one census, for the cross-emitter id-allocation tests."""
    return (
        _synthetic_re_record(),
        _synthetic_rhq_record(),
        _synthetic_rhv_record(),
        _synthetic_rhe_record(),
    )


class TestGenericConstraintWiring:
    """The E4/E5 generic-constraint emitters wired
    into ``convert_decomp_case`` over one shared 0-based id allocator.
    Tier-1 synthetic only — the ``_run_cadastro_pipeline`` mock deck,
    extended with a ``to_generic`` census; no real deck, no ``import novomodelo``.
    """

    def test_ids_are_dense_and_unique_across_the_emitters(self, tmp_path: Path) -> None:
        """RE (one upper-only id) -> the combined RHQ/RHV emitter (one
        genuinely two-sided HQ id under novomodelo's F3 interval model + one
        lower-only HV id) -> RHE (one id) share a single running allocator,
        so the 4 emitted ids form a gap-free ``range(4)`` with no
        collision, regardless of which emitter produced which id."""
        dst = _run_cadastro_pipeline(
            tmp_path, ac_volmax_frame=None, to_generic=_all_synthetic_generics()
        )
        doc = json.loads((dst / "constraints" / "generic_constraints.json").read_text())
        ids = [c["id"] for c in doc["constraints"]]
        assert len(ids) == 4
        assert sorted(ids) == list(range(len(ids)))
        assert len(ids) == len(set(ids))
        # E7's fixed order: RE, then HQ/HV (combined), then RHE. The
        # genuinely two-sided HQ record is one F3 constraint, not two.
        by_family = [c["name"].split("_")[0] for c in doc["constraints"]]
        assert by_family == ["RE", "HQ", "HV", "RHE"]

    def test_writes_both_generic_files_when_generics_present(
        self, tmp_path: Path
    ) -> None:
        """Positive half: a deck with >= 1 surviving generic writes
        both ``generic_constraints.json`` and
        ``generic_constraint_bounds.parquet``."""
        dst = _run_cadastro_pipeline(
            tmp_path, ac_volmax_frame=None, to_generic=_all_synthetic_generics()
        )
        assert (dst / "constraints" / "generic_constraints.json").is_file()
        assert (dst / "constraints" / "generic_constraint_bounds.parquet").is_file()

    def test_writes_neither_generic_file_when_none_survive(
        self, tmp_path: Path
    ) -> None:
        """Negative half: the baseline mock deck (no ``to_generic``
        records at all) writes neither file."""
        dst = _run_cadastro_pipeline(tmp_path, ac_volmax_frame=None)
        assert not (dst / "constraints" / "generic_constraints.json").exists()
        assert not (dst / "constraints" / "generic_constraint_bounds.parquet").exists()

    def test_rhe_rho_acum_sigil_matches_the_rhs_it_drove(self, tmp_path: Path) -> None:
        """The ``@rho_acum_h{id}`` LHS sigil ``generic_parameters.json``
        declares for a referenced hydro resolves to the SAME per-stage value
        ``emit_rhe_generics`` used to compute that same constraint's own RHS
        (a percentage-of-EARM limit, so the RHS genuinely depends on it) —
        proving the pipeline wires the scalar-parameters half and the
        RHE emitter half together, not two independently
        computed values that merely happen to look alike."""
        dst = _run_cadastro_pipeline(
            tmp_path, ac_volmax_frame=None, to_generic=(_synthetic_rhe_record(),)
        )

        doc = json.loads((dst / "constraints" / "generic_constraints.json").read_text())
        [rhe] = doc["constraints"]
        assert rhe["name"].startswith("RHE_")
        referenced_hids = sorted(
            int(hid) for hid in re.findall(r"rho_acum_h(\d+)", rhe["expression"])
        )
        assert referenced_hids == [0, 1]  # both fixture plants (REE 1)

        params = json.loads(
            (dst / "constraints" / "generic_parameters.json").read_text()
        )["scalar_parameters"]
        per_stage_by_hid = {
            hid: dict(
                next(
                    p
                    for p in params
                    if p["name"] == f"rho_acum_h{hid}" and p["kind"] == "per_stage"
                )["values"]
            )
            for hid in referenced_hids
        }

        bounds = pq.read_table(
            dst / "constraints" / "generic_constraint_bounds.parquet"
        ).to_pylist()
        [bound_row] = [
            row
            for row in bounds
            if row["constraint_id"] == rhe["id"] and row["stage_id"] == 0
        ]

        # volume_maximo per fixture plant (codes 1, 2 -> hids 0, 1) — no
        # ACVOLMAX override active in this test (ac_volmax_frame=None), so
        # the effective value equals the base _cadastro_plant_row vmax.
        volume_maximo_by_hid = {0: 100.0, 1: 50.0}
        expected_rhs = (_RHE_LIMITE_PCT / 100.0) * sum(
            per_stage_by_hid[hid][0] * volume_maximo_by_hid[hid]
            for hid in referenced_hids
        )
        assert bound_row["bound_lower"] == pytest.approx(expected_rhs)
        assert bound_row["bound_upper"] is None

    def test_detection_diagnostics_flow_through_the_dx_sink(
        self, tmp_path: Path
    ) -> None:
        """The E1 FE/RHA/LIBs-electrical detection diagnostics are
        captured on the returned ``ConversionReport``, not only logged.

        ``convert_decomp_case`` now owns its own top-level
        ``dx.collect()``, so this reads the diagnostics via
        ``_run_cadastro_pipeline``'s ``diagnostics_out`` rather than an outer
        ``dx.collect()`` (which would be shadowed and see nothing)."""
        from novomodelo_bridge.core import diagnostics as dx

        fe_diagnostic = dx.Diagnostic(
            code="decomp-fe-participation-unreadable",
            severity=dx.Severity.WARNING,
            category="Special constraints",
            title="FE electrical participation present but unreadable",
            summary="synthetic FE finding",
        )
        rha_diagnostic = dx.Diagnostic(
            code="decomp-rha-family-unconverted",
            severity=dx.Severity.WARNING,
            category="Special constraints",
            title="RHA family not converted",
            summary="synthetic RHA finding",
        )
        libs_diagnostic = dx.Diagnostic(
            code="decomp-libs-electrical-present",
            severity=dx.Severity.WARNING,
            category="Special constraints",
            title="LIBs electrical constraints present",
            summary="synthetic LIBs finding",
        )

        collected: list[dx.Diagnostic] = []
        _run_cadastro_pipeline(
            tmp_path,
            ac_volmax_frame=None,
            unreadable_electrical=(fe_diagnostic, rha_diagnostic),
            libs_electrical=libs_diagnostic,
            diagnostics_out=collected,
        )

        codes = {d.code for d in collected}
        assert {
            "decomp-fe-participation-unreadable",
            "decomp-rha-family-unconverted",
            "decomp-libs-electrical-present",
        } <= codes

    def test_generic_bounds_are_zstd_and_envelope_matches_the_shared_writer(
        self, tmp_path: Path
    ) -> None:
        """``generic_constraint_bounds.parquet`` compresses with zstd
        (novomodelo C3: snappy unsupported), and ``generic_constraints.json``'s
        envelope key + ``$schema`` match the registry entry the source
        model's own generic-constraints writer (``converters/constraints.py``)
        also reads, confirmed by loading both."""
        dst = _run_cadastro_pipeline(
            tmp_path, ac_volmax_frame=None, to_generic=_all_synthetic_generics()
        )

        bounds_path = dst / "constraints" / "generic_constraint_bounds.parquet"
        metadata = pq.ParquetFile(bounds_path).metadata
        for row_group_index in range(metadata.num_row_groups):
            row_group = metadata.row_group(row_group_index)
            for column_index in range(row_group.num_columns):
                assert row_group.column(column_index).compression == "ZSTD"

        doc = json.loads((dst / "constraints" / "generic_constraints.json").read_text())
        assert doc["$schema"] == novomodelo_schemas.schema_url_for(
            "constraints/generic_constraints.json"
        )
        assert set(doc) == {"$schema", "constraints"}


class TestDryRun:
    """``dry_run`` threads through both DECOMP write
    seams (the in-impl ``_write_json``/``_write_parquet`` closures and the
    ``write_scalar_parameters`` seam) and the partial-write cleanup arm."""

    def test_dry_run_writes_nothing_and_records_would_write_paths(
        self, tmp_path: Path
    ) -> None:
        report_out: list[ConversionReport] = []
        dst = _run_cadastro_pipeline(
            tmp_path, ac_volmax_frame=None, dry_run=True, report_out=report_out
        )

        assert not dst.exists()
        [report] = report_out
        assert str(dst / "config.json") in report.would_write_paths
        assert str(dst / "stages.json") in report.would_write_paths

    def test_dry_run_defers_generic_parameters_but_records_its_path(
        self, tmp_path: Path
    ) -> None:
        """A surviving RHE record drives
        ``write_scalar_parameters`` -- under ``dry_run=True`` it must defer
        the write while still recording the path, proving the second write
        seam (outside the ``_write_json``/``_write_parquet`` closures) is
        dry-run-aware too."""
        report_out: list[ConversionReport] = []
        dst = _run_cadastro_pipeline(
            tmp_path,
            ac_volmax_frame=None,
            to_generic=(_synthetic_rhe_record(),),
            dry_run=True,
            report_out=report_out,
        )

        assert not (dst / "constraints" / "generic_parameters.json").exists()
        [report] = report_out
        assert (
            str(dst / "constraints" / "generic_parameters.json")
            in report.would_write_paths
        )

    def test_real_run_failure_clears_partial_writes(self, tmp_path: Path) -> None:
        """A mid-conversion failure on a real run removes the known outputs
        already written before the raise, so a plain (no ``--force``) retry
        is not refused as "destination not empty"."""
        from novomodelo_bridge.decomp import pipeline as decomp_pipeline

        dst = tmp_path / "case"

        def _fake_impl(*args: object, **kwargs: object) -> ConversionReport:
            dst.mkdir(parents=True, exist_ok=True)
            (dst / "config.json").write_text("{}", encoding="utf-8")
            raise ValueError("boom")

        with (
            patch.object(
                decomp_pipeline, "_convert_decomp_case_impl", side_effect=_fake_impl
            ),
            pytest.raises(ValueError, match="boom"),
        ):
            decomp_pipeline.convert_decomp_case(tmp_path / "src", dst)

        assert not (dst / "config.json").exists()

    def test_dry_run_failure_preserves_pre_existing_dst_contents(
        self, tmp_path: Path
    ) -> None:
        """A dry-run failure must never clear ``dst``: it wrote nothing, and
        ``dst`` may be a pre-existing populated directory the user never
        asked to clear. ``config.json`` is one of ``clear_dst_contents``'s
        own removal-list names, so its survival proves cleanup was skipped
        entirely, not merely that this particular name was spared.

        ``force=True`` here only clears the unrelated non-empty-``dst``
        refusal guard (now checked up front by ``convert_decomp_case``
        itself, ahead of the mocked impl) so this test can reach and isolate
        the failure-clearing behavior under ``dry_run`` that it actually
        targets; see ``test_real_run_against_populated_dst_refuses_without_clearing``
        for the refusal-guard behavior on its own."""
        from novomodelo_bridge.decomp import pipeline as decomp_pipeline

        dst = tmp_path / "case"
        dst.mkdir()
        existing = dst / "config.json"
        existing.write_text("keep me", encoding="utf-8")

        def _fake_impl(*args: object, **kwargs: object) -> ConversionReport:
            raise ValueError("boom")

        with (
            patch.object(
                decomp_pipeline, "_convert_decomp_case_impl", side_effect=_fake_impl
            ),
            pytest.raises(ValueError, match="boom"),
        ):
            decomp_pipeline.convert_decomp_case(
                tmp_path / "src", dst, dry_run=True, force=True
            )

        assert existing.read_text(encoding="utf-8") == "keep me"

    def test_force_preclears_stale_decomp_artifacts(self, tmp_path: Path) -> None:
        """A successful ``--force`` re-run over a populated ``dst`` pre-clears
        the previous case's full artifact set first, so a conditional
        artifact the new run does not reproduce (``post_study_stages.json``,
        ``boundary/``) cannot survive on top of the fresh case."""
        from novomodelo_bridge.decomp import pipeline as decomp_pipeline

        dst = tmp_path / "case"
        dst.mkdir()
        post_study = dst / "post_study_stages.json"
        post_study.write_text("{}", encoding="utf-8")
        boundary = dst / "boundary"
        boundary.mkdir()
        (boundary / "metadata.json").write_text("{}", encoding="utf-8")

        def _fake_impl(*args: object, **kwargs: object) -> ConversionReport:
            return ConversionReport()

        with patch.object(
            decomp_pipeline, "_convert_decomp_case_impl", side_effect=_fake_impl
        ):
            decomp_pipeline.convert_decomp_case(tmp_path / "src", dst, force=True)

        assert not post_study.exists()
        assert not boundary.exists()

    def test_force_dry_run_preserves_stale_decomp_artifacts(
        self, tmp_path: Path
    ) -> None:
        """``--force --dry-run`` over the same populated ``dst`` must not
        pre-clear: a dry run never mutates ``dst``."""
        from novomodelo_bridge.decomp import pipeline as decomp_pipeline

        dst = tmp_path / "case"
        dst.mkdir()
        post_study = dst / "post_study_stages.json"
        post_study.write_text("{}", encoding="utf-8")
        boundary = dst / "boundary"
        boundary.mkdir()
        (boundary / "metadata.json").write_text("{}", encoding="utf-8")

        def _fake_impl(*args: object, **kwargs: object) -> ConversionReport:
            return ConversionReport()

        with patch.object(
            decomp_pipeline, "_convert_decomp_case_impl", side_effect=_fake_impl
        ):
            decomp_pipeline.convert_decomp_case(
                tmp_path / "src", dst, force=True, dry_run=True
            )

        assert post_study.exists()
        assert boundary.exists()

    def test_real_run_against_populated_dst_refuses_without_clearing(
        self, tmp_path: Path
    ) -> None:
        """Regression for the data-loss bug: the non-empty/``force``
        refusal must fire before the clearing ``try``/``except`` so a plain
        (no ``--force``) run against a pre-existing, populated ``dst`` raises
        cleanly and never deletes the user's existing case. Exercises the
        real guard (no mocking of ``_convert_decomp_case_impl``), since a
        guard living inside the mocked-out implementation would not be
        caught by a test that replaces it with a fake."""
        from novomodelo_bridge.decomp.pipeline import convert_decomp_case

        dst = tmp_path / "case"
        dst.mkdir()
        existing = dst / "config.json"
        existing.write_bytes(b"keep me")

        with pytest.raises(FileExistsError, match="force"):
            convert_decomp_case(tmp_path / "src", dst)

        assert existing.read_bytes() == b"keep me"

    def test_clear_dst_contents_removes_decomp_only_artifacts(
        self, tmp_path: Path
    ) -> None:
        """The DECOMP set clears the shared artifacts plus its own root-level
        ``post_study_stages.json`` and ``boundary/`` tree."""
        from novomodelo_bridge.core.conversion import clear_dst_contents
        from novomodelo_bridge.decomp.pipeline import DECOMP_CLEARED_ARTIFACTS

        dst = tmp_path / "dst"
        dst.mkdir()
        post_study = dst / "post_study_stages.json"
        post_study.write_text("{}", encoding="utf-8")
        boundary = dst / "boundary"
        boundary.mkdir()
        (boundary / "metadata.json").write_text("{}", encoding="utf-8")
        config = dst / "config.json"
        config.write_text("{}", encoding="utf-8")
        notes = dst / "notes.txt"
        notes.write_text("keep me", encoding="utf-8")

        clear_dst_contents(dst, DECOMP_CLEARED_ARTIFACTS)

        assert not post_study.exists()
        assert not boundary.exists()
        assert not config.exists()
        assert notes.exists()

    def test_newave_cleared_set_leaves_decomp_only_artifacts(
        self, tmp_path: Path
    ) -> None:
        """Regression: the NEWAVE set does not name
        DECOMP-only artifacts, so they survive a NEWAVE-set clear."""
        from novomodelo_bridge.core.conversion import clear_dst_contents
        from novomodelo_bridge.newave.pipeline import NEWAVE_CLEARED_ARTIFACTS

        dst = tmp_path / "dst"
        dst.mkdir()
        post_study = dst / "post_study_stages.json"
        post_study.write_text("{}", encoding="utf-8")
        boundary = dst / "boundary"
        boundary.mkdir()
        (boundary / "metadata.json").write_text("{}", encoding="utf-8")
        config = dst / "config.json"
        config.write_text("{}", encoding="utf-8")

        clear_dst_contents(dst, NEWAVE_CLEARED_ARTIFACTS)

        assert post_study.exists()
        assert boundary.exists()
        assert not config.exists()


_READ_TRAVEL_TIMES = (
    "novomodelo_bridge.decomp.pipeline.travel_time_conv.read_travel_times"
)


class TestDeferralWarning:
    """The flat ``deferred at this milestone`` warning now names ONLY water
    travel time, and only when the deck actually carries a ``VI`` register.
    Boundary FCF is imported by default, reservoir evaporation is converted
    (novomodelo >= 0.14's C11 fix), and windowed inflow inputs do not apply to the
    external explicit tree the DECOMP path emits — so none of those are
    deferred, and a deck with no travel time emits no deferral warning at all.
    """

    def test_no_warning_when_nothing_is_deferred(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The mock deck carries no VI travel-time register, so nothing is
        deferred and no ``deferred at this milestone`` warning is emitted."""
        with caplog.at_level(
            logging.WARNING, logger="novomodelo_bridge.decomp.pipeline"
        ):
            _run_cadastro_pipeline(tmp_path, ac_volmax_frame=None)

        assert not [
            r for r in caplog.records if "deferred at this milestone" in r.message
        ]

    def test_names_only_water_travel_time_when_present(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """With a VI register present, the warning fires naming water travel
        time alone — never boundary FCF, windowed inflow inputs, GNL
        anticipation, or reservoir evaporation (all emitted, not deferred)."""
        with (
            caplog.at_level(
                logging.WARNING, logger="novomodelo_bridge.decomp.pipeline"
            ),
            patch(_READ_TRAVEL_TIMES, return_value={1: 24.0}),
        ):
            _run_cadastro_pipeline(tmp_path, ac_volmax_frame=None)

        deferral = next(
            r.message
            for r in caplog.records
            if "deferred at this milestone" in r.message
        )
        assert "water travel time (VI present)" in deferral
        for absent in ("boundary FCF", "windowed inflow", "GNL", "evaporation"):
            assert absent not in deferral

    def test_fe_rha_libs_are_reported_through_the_dx_sink_not_the_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A deck with FE/RHA/LIBs surfaces reports them exactly once, through
        the structured ``Diagnostic``s -- even when the deferral
        warning fires (VI present here), it names only water travel time,
        never these special-constraint surfaces (the warning text and the
        ``dx`` sink are disjoint on these items).

        ``convert_decomp_case`` owns its own top-level
        ``dx.collect()``, so the diagnostics half reads
        ``_run_cadastro_pipeline``'s ``diagnostics_out`` rather than an outer
        ``dx.collect()`` (which would be shadowed and see nothing)."""
        from novomodelo_bridge.core import diagnostics as dx

        fe_diagnostic = dx.Diagnostic(
            code="decomp-fe-participation-unreadable",
            severity=dx.Severity.WARNING,
            category="Special constraints",
            title="FE electrical participation present but unreadable",
            summary="synthetic FE finding",
        )
        rha_diagnostic = dx.Diagnostic(
            code="decomp-rha-family-unconverted",
            severity=dx.Severity.WARNING,
            category="Special constraints",
            title="RHA family not converted",
            summary="synthetic RHA finding",
        )
        libs_diagnostic = dx.Diagnostic(
            code="decomp-libs-electrical-present",
            severity=dx.Severity.WARNING,
            category="Special constraints",
            title="LIBs electrical constraints present",
            summary="synthetic LIBs finding",
        )

        collected: list[dx.Diagnostic] = []
        with (
            caplog.at_level(
                logging.WARNING, logger="novomodelo_bridge.decomp.pipeline"
            ),
            patch(_READ_TRAVEL_TIMES, return_value={1: 24.0}),
        ):
            _run_cadastro_pipeline(
                tmp_path,
                ac_volmax_frame=None,
                unreadable_electrical=(fe_diagnostic, rha_diagnostic),
                libs_electrical=libs_diagnostic,
                diagnostics_out=collected,
            )

        codes = {d.code for d in collected}
        assert {
            "decomp-fe-participation-unreadable",
            "decomp-rha-family-unconverted",
            "decomp-libs-electrical-present",
        } <= codes

        deferral = next(
            r.message
            for r in caplog.records
            if "deferred at this milestone" in r.message
        )
        for needle in ("FE electrical participation", "RHA family", "LIBs electrical"):
            assert needle not in deferral


class TestDiversionChannels:
    """``_diversion_channels`` couples a positive QDES diversion floor to the
    source-model diversion channel novomodelo requires for it (BILLINGS/PIMENTAL in
    the real decks)."""

    @staticmethod
    def _id_map() -> DecompIdMap:
        # hydro_id(code) == hydro_codes.index(code): 118 -> 0, 119 -> 1.
        return DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(118, 119))

    @staticmethod
    def _effective(diversions: dict, n_stages: int = 2) -> EffectiveCadastro:
        return EffectiveCadastro(
            base=pd.DataFrame(),
            n_stages=n_stages,
            stage_varying={},
            diversions=diversions,
        )

    @staticmethod
    def _bounds(hydro_id: int, min_diversion: float | None) -> pa.Table:
        return pa.table(
            {
                "hydro_id": pa.array([hydro_id], type=pa.int32()),
                "min_diversion_m3s": pa.array([min_diversion], type=pa.float64()),
            }
        )

    def test_floored_hydro_with_channel_gets_the_channel(self) -> None:
        channel = DiversionChannel(downstream=119, limit=100.0)
        channels, unresolved = _diversion_channels(
            self._bounds(0, 6.0),
            self._id_map(),
            self._effective({118: (channel, channel)}),
        )
        assert channels == {0: {"downstream_id": 1, "max_flow_m3s": 100.0}}
        assert unresolved == []

    def test_floored_hydro_without_channel_is_unresolved(self) -> None:
        channels, unresolved = _diversion_channels(
            self._bounds(0, 6.0), self._id_map(), self._effective({})
        )
        assert channels == {}
        assert unresolved == [0]

    def test_channel_with_no_limit_is_unresolved(self) -> None:
        channel = DiversionChannel(downstream=119, limit=None)
        channels, unresolved = _diversion_channels(
            self._bounds(0, 6.0),
            self._id_map(),
            self._effective({118: (channel, channel)}),
        )
        assert channels == {}
        assert unresolved == [0]

    def test_channel_to_unmapped_downstream_is_unresolved(self) -> None:
        channel = DiversionChannel(downstream=999, limit=100.0)
        channels, unresolved = _diversion_channels(
            self._bounds(0, 6.0),
            self._id_map(),
            self._effective({118: (channel, channel)}),
        )
        assert channels == {}
        assert unresolved == [0]

    def test_zero_floor_gets_no_channel(self) -> None:
        channel = DiversionChannel(downstream=119, limit=100.0)
        channels, unresolved = _diversion_channels(
            self._bounds(0, 0.0),
            self._id_map(),
            self._effective({118: (channel, channel)}),
        )
        assert channels == {}
        assert unresolved == []

    def test_absent_diversion_column_returns_empty(self) -> None:
        table = pa.table({"hydro_id": pa.array([0], type=pa.int32())})
        channels, unresolved = _diversion_channels(
            table, self._id_map(), self._effective({})
        )
        assert channels == {}
        assert unresolved == []


class TestBaseDiversionChannels:
    """``_base_diversion_channels`` models the BASE ``desvio`` diversions that
    carry no QDES flow bound (e.g. MOXOTO -> P.AFONSO 4): without them novomodelo pins
    the diversion column to ``[0, 0]`` and the downstream plant is stranded."""

    @staticmethod
    def _id_map() -> DecompIdMap:
        # 173 (MOXOTO source) -> 0, 175 (P.AFONSO 4 receiver) -> 1.
        return DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(173, 175))

    @staticmethod
    def _effective(diversions: dict, n_stages: int = 2) -> EffectiveCadastro:
        return EffectiveCadastro(
            base=pd.DataFrame(),
            n_stages=n_stages,
            stage_varying={},
            diversions=diversions,
        )

    @staticmethod
    def _caps() -> dict[int, HydroCapacities]:
        return {
            0: HydroCapacities(max_generation_mw=400.0, max_turbined_m3s=2120.5),
            1: HydroCapacities(max_generation_mw=2462.4, max_turbined_m3s=2406.8),
        }

    def test_base_channel_capped_at_receiver_turbine(self) -> None:
        channel = DiversionChannel(downstream=175, limit=None)
        channels, contribs = _base_diversion_channels(
            self._effective({173: (channel, channel)}),
            self._id_map(),
            self._caps(),
            already_bounded_ids=set(),
            stage_ids=[0, 1],
        )
        assert channels == {0: {"downstream_id": 1, "max_flow_m3s": 2406.8}}
        # one diversion contribution per stage, opened [0, receiver-turbine]
        assert [
            (c.entity_id, c.stage_id, c.axis, c.lower, c.upper) for c in contribs
        ] == [
            (0, 0, "diversion", 0.0, 2406.8),
            (0, 1, "diversion", 0.0, 2406.8),
        ]

    def test_ac_limit_channel_uses_the_explicit_limit(self) -> None:
        channel = DiversionChannel(downstream=175, limit=500.0)
        channels, contribs = _base_diversion_channels(
            self._effective({173: (channel, channel)}),
            self._id_map(),
            self._caps(),
            already_bounded_ids=set(),
            stage_ids=[0, 1],
        )
        assert channels == {0: {"downstream_id": 1, "max_flow_m3s": 500.0}}
        assert all(c.upper == 500.0 for c in contribs)

    def test_already_bounded_source_is_skipped(self) -> None:
        channel = DiversionChannel(downstream=175, limit=None)
        channels, contribs = _base_diversion_channels(
            self._effective({173: (channel, channel)}),
            self._id_map(),
            self._caps(),
            already_bounded_ids={0},
            stage_ids=[0, 1],
        )
        assert channels == {}
        assert contribs == []

    def test_unmapped_downstream_is_skipped(self) -> None:
        channel = DiversionChannel(downstream=999, limit=None)
        channels, contribs = _base_diversion_channels(
            self._effective({173: (channel, channel)}),
            self._id_map(),
            self._caps(),
            already_bounded_ids=set(),
            stage_ids=[0, 1],
        )
        assert channels == {}
        assert contribs == []

    def test_no_diversion_returns_empty(self) -> None:
        channels, contribs = _base_diversion_channels(
            self._effective({}),
            self._id_map(),
            self._caps(),
            already_bounded_ids=set(),
            stage_ids=[0, 1],
        )
        assert channels == {}
        assert contribs == []


def _fc_line(tipo: str, caminho: str) -> str:
    """One fixed-width ``FC`` register line: identifier at columns 0:4,
    ``tipo`` mnemonic at 4:10, ``caminho`` from column 14 -- matching
    ``idecomp.decomp.modelos.dadger.FC``'s own layout (confirmed against
    ``example/decomp-mar-26-rv2/dadger.rv2``'s real ``FC  NEWV21
    cortesh.dat`` line)."""
    return f"FC  {tipo:<6}    {caminho}"


class TestDiscoverDecompFilesBoundaryFcf:
    """``discover_decomp_files`` resolves the deck's optional
    boundary-FCF cut files (``cortesh``/``cortes``), gated on their presence
    -- the discovery prerequisite for the boundary-FCF importer, which
    ``convert decomp`` now runs by default (``--no-fcf`` opts out). Mirrors
    ``TestDiscoverDecompFilesLibsElectrical``'s synthetic-deck-dir fixture
    pattern (in the ``test_libs_electrical_pipeline`` module)."""

    @staticmethod
    def _minimal_deck(deck_dir: Path, *, dadger_text: str = "") -> None:
        (deck_dir / "caso.dat").write_text("rv0", encoding="latin-1")
        (deck_dir / "rv0").write_text(
            "dadger.rv0\nvazoes.rv0\nhidr.dat\n", encoding="latin-1"
        )
        (deck_dir / "dadger.rv0").write_text(dadger_text, encoding="latin-1")
        (deck_dir / "vazoes.rv0").write_text("", encoding="latin-1")
        (deck_dir / "hidr.dat").write_text("", encoding="latin-1")

    def test_cortesh_and_cortes_resolved_when_present(self, tmp_path: Path) -> None:
        from novomodelo_bridge.decomp.files import discover_decomp_files

        self._minimal_deck(tmp_path)
        (tmp_path / "cortesh.dat").write_text("", encoding="latin-1")
        (tmp_path / "cortes-004.dat").write_text("", encoding="latin-1")

        files = discover_decomp_files(tmp_path)

        assert files.cortesh == tmp_path / "cortesh.dat"
        assert files.cortes == tmp_path / "cortes-004.dat"

    def test_cortesh_and_cortes_none_when_absent(self, tmp_path: Path) -> None:
        from novomodelo_bridge.decomp.files import discover_decomp_files

        self._minimal_deck(tmp_path)

        files = discover_decomp_files(tmp_path)

        assert files.cortesh is None
        assert files.cortes is None

    def test_cortes_prefers_single_stage_export_over_consolidated_archive(
        self, tmp_path: Path
    ) -> None:
        from novomodelo_bridge.decomp.files import discover_decomp_files

        self._minimal_deck(tmp_path)
        (tmp_path / "cortesh.dat").write_text("", encoding="latin-1")
        (tmp_path / "cortes.dat").write_text("", encoding="latin-1")
        (tmp_path / "cortes-004.dat").write_text("", encoding="latin-1")

        files = discover_decomp_files(tmp_path)

        assert files.cortes == tmp_path / "cortes-004.dat"

    def test_cortesh_resolved_via_fc_record_outside_deck_dir(
        self, tmp_path: Path
    ) -> None:
        """The ``FC`` record's own ``caminho`` may be a relative path
        pointing outside the deck directory (e.g. a shared upstream run
        directory); the glob idiom alone could never find it there."""
        from novomodelo_bridge.decomp.files import discover_decomp_files

        deck_dir = tmp_path / "deck"
        deck_dir.mkdir()
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        target = shared_dir / "cortesh.dat"
        target.write_text("", encoding="latin-1")

        self._minimal_deck(
            deck_dir,
            dadger_text=_fc_line("NEWV21", "../shared/cortesh.dat") + "\n",
        )

        files = discover_decomp_files(deck_dir)

        assert files.cortesh == target

    def test_fc_record_naming_missing_file_falls_back_to_glob(
        self, tmp_path: Path
    ) -> None:
        """A malformed/stale ``FC`` record (naming a file that does not
        exist) must never raise -- discovery falls through to the deck-local
        glob idiom instead."""
        from novomodelo_bridge.decomp.files import discover_decomp_files

        self._minimal_deck(
            tmp_path,
            dadger_text=_fc_line("NEWV21", "does-not-exist.dat") + "\n",
        )
        target = tmp_path / "cortesh.dat"
        target.write_text("", encoding="latin-1")

        files = discover_decomp_files(tmp_path)

        assert files.cortesh == target

    def test_decomp_files_still_constructs_without_the_new_fields(self) -> None:
        """Every pre-existing ``DecompFiles(...)`` call site keeps
        constructing unchanged -- both new fields default to ``None``."""
        from novomodelo_bridge.decomp.files import DecompFiles

        files = DecompFiles(
            revision="rv0",
            dadger=Path("dadger.rv0"),
            vazoes=Path("vazoes.rv0"),
            hidr=Path("hidr.dat"),
            dadgnl=None,
            renovaveis=None,
            polinjus=None,
        )

        assert files.cortesh is None
        assert files.cortes is None
