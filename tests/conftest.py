"""Shared test helpers for building the source model case inputs without real file I/O.

``make_nw_files`` constructs a :class:`~novomodelo_bridge.newave.files.NewaveFiles`
with paths under a tmp dir (no filesystem access). ``make_case`` wraps it in a
:class:`~novomodelo_bridge.newave.case.NewaveCase` and pre-fills the requested cached reader
slots, so a converter under test reads the supplied mock objects instead of
parsing files. ``make_decomp_files``/``make_decomp_case`` are the DECOMP-track
twins, wrapping a :class:`~novomodelo_bridge.decomp.case.DecompCase`.
``hydro_with_group`` builds a 0.13-shaped ``hydros.json`` hydro
dict (no top-level ``bus_id``, one mirror ``unit_groups`` entry) for tests that
hand-build a hydro fixture rather than calling a converter. ``_FakeDadger``
is a decomp-side test double returning preset DataFrames for register
accessors. These builders are referenced from two or more of
``tests/{cli,novomodelo,comparators,dashboard,decomp,newave}/``; a builder used by
only one of those directories lives in that directory's own ``conftest.py``
instead (e.g. the hydro/thermal/network entity-conversion fixtures in
``tests/newave/conftest.py``). Import them with ``from tests.conftest import
make_case``.
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from novomodelo_bridge.comparators.dataset import (
    SUMMARY_SCHEMA,
    TIDY_SCHEMA,
    ComparisonDataset,
)
from novomodelo_bridge.core.hydro_units import build_mirror_unit_group
from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.files import DecompFiles
from novomodelo_bridge.newave.case import NewaveCase
from novomodelo_bridge.newave.files import NewaveFiles

# Shared skip marker for tier-2 tests that need `novomodelo-python` (`import
# novomodelo`). It is a core dev dependency (installed via `.[dev]`), so it is
# present on every CI job; this ``skipif`` guards only a deliberately
# novomodelo-free environment (e.g. a partial dev checkout). Use ``find_spec``
# (import-free) so this module stays importable there too. ``condition`` is
# passed by keyword (rather than positionally) so ``.kwargs["condition"]`` is
# introspectable.
# Import via ``from tests.conftest import requires_novomodelo_python``.
requires_novomodelo_python = pytest.mark.skipif(
    condition=importlib.util.find_spec("novomodelo") is None,
    reason=(
        "requires novomodelo-python, a core dev dependency; this environment is "
        "missing it (a deliberately novomodelo-free checkout)"
    ),
)


def has_writer_binding() -> bool:
    """Whether an installed novomodelo wheel exposes ``write_policy_checkpoint``.

    Checked via ``importlib.util.find_spec`` before any import — the same
    import-free-first convention as ``requires_novomodelo_python`` above — so
    calling this (or importing this module) never requires novomodelo to be
    installed. A novomodelo wheel older than the one that added the writer
    binding is import-able (``find_spec`` succeeds) but lacks the attribute;
    ``requires_writer_binding`` below turns that case into a clean SKIP
    instead of a runtime ``AttributeError``.
    """
    if importlib.util.find_spec("novomodelo") is None:
        return False
    import novomodelo

    return hasattr(novomodelo, "write_policy_checkpoint")


# Additional skip marker for tier-2 tests whose path calls an in-wheel
# binding newer than the mere import-ability ``requires_novomodelo_python``
# checks. Stack this alongside ``requires_novomodelo_python`` on any test that
# calls ``novomodelo.write_policy_checkpoint`` directly (or transitively via a
# helper that does), so an old-but-importable novomodelo wheel skips cleanly
# rather than failing at runtime with ``AttributeError``.
# Import via ``from tests.conftest import requires_writer_binding``.
requires_writer_binding = pytest.mark.skipif(
    condition=not has_writer_binding(),
    reason=(
        "requires a novomodelo-python wheel exposing the write_policy_checkpoint "
        "writer binding"
    ),
)


@pytest.fixture
def decomp_mini_deck() -> Path:
    """Path to the committed DECOMP mini-deck under ``tests/decks/decomp_mini/``.

    A tier-2 end-to-end test converts this deck with the real
    ``convert_decomp_case`` and validates the result with the real
    ``novomodelo.io.validate`` — see ``tests/decomp/test_convert_decomp_e2e.py``.
    """
    return Path(__file__).parent / "decks" / "decomp_mini"


@pytest.fixture
def newave_mini_deck() -> Path:
    """Path to the committed NEWAVE mini-deck under ``tests/decks/newave_mini/``.

    A tier-2 end-to-end test converts this deck with the real
    ``convert_newave_case`` and validates the result with the real
    ``novomodelo.io.validate`` — see ``tests/newave/test_convert_newave_e2e.py``.
    """
    return Path(__file__).parent / "decks" / "newave_mini"


@pytest.fixture
def dumb_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force a plain, fixed-width terminal for Typer ``--help`` content tests.

    Typer's Rich help (``rich_markup_mode="rich"``) renders its options table
    from the detected terminal. On CI, GitHub Actions sets
    ``GITHUB_ACTIONS``/``FORCE_COLOR``, pushing Rich into "force terminal" mode;
    inside Typer's ``CliRunner`` (whose captured stream is not a real tty) that
    path renders an *empty* options box, so option-presence assertions pass
    locally but fail on CI. ``TERM=dumb`` makes Rich treat the captured output as
    a plain, fixed-width stream and emit the help content deterministically.

    Only the ``--help`` content tests opt in (via this fixture) so the fixed
    width does not perturb width-sensitive rendering in the rest of the suite.
    The real CLI, writing to a normal stdout, is unaffected.
    """
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.setenv("COLUMNS", "80")


@pytest.fixture(autouse=True)
def _restore_novomodelo_bridge_logger() -> Iterator[None]:
    """Snapshot/restore the ``novomodelo_bridge`` logger around every test.

    ``cli._configure_logging`` flips ``propagate``/level for a run, and the CLI's
    Typer ``CliRunner`` tests invoke the app directly (bypassing ``main``'s restore),
    so without this fixture that state would leak into a later ``caplog``-by-logger
    test in the same interpreter and silently swallow its records.
    """
    pkg = logging.getLogger("novomodelo_bridge")
    prior_propagate = pkg.propagate
    prior_level = pkg.level
    try:
        yield
    finally:
        pkg.propagate = prior_propagate
        pkg.setLevel(prior_level)


_REQUIRED = (
    "dger",
    "confhd",
    "conft",
    "sistema",
    "clast",
    "term",
    "ree",
    "patamar",
    "hidr",
    "vazoes",
)
_OPTIONAL = (
    "modif",
    "ghmin",
    "penalid",
    "vazpast",
    "dsvagua",
    "curva",
    "expt",
    "exph",
    "manutt",
    "c_adic",
    "cvar",
    "agrint",
    "re_dat",
    "volref_saz",
    "shist",
    "adterm",
    "polinjus",
    "tratamento_fpha",
)


def make_nw_files(tmp_path: Path, **overrides: Path | None) -> NewaveFiles:
    """Build a ``NewaveFiles`` pointing into *tmp_path*; no I/O.

    Required files default to ``tmp_path / "<name>.dat"`` and optional files to
    ``None``. Pass any attribute as a keyword to override (e.g.
    ``make_nw_files(tmp_path, cvar=tmp_path / "cvar.dat")``).
    """
    fields: dict[str, Path | None] = {"directory": tmp_path}
    for name in _REQUIRED:
        fields[name] = tmp_path / f"{name}.dat"
    for name in _OPTIONAL:
        fields[name] = None
    fields.update(overrides)
    return NewaveFiles(**fields)  # type: ignore[arg-type]


def make_case(files_or_tmp: NewaveFiles | Path, **parsed: Any) -> NewaveCase:
    """Build a ``NewaveCase`` with the given parsed readers pre-cached.

    *files_or_tmp* is either a ``NewaveFiles`` or a tmp dir (then
    ``make_nw_files`` builds one). Each ``parsed`` keyword sets the matching
    cached-property slot directly, so accessing e.g. ``case.dger`` returns the
    supplied object without parsing. Pass ``cvar=None`` to force an optional
    reader to the absent state.
    """
    files = (
        files_or_tmp
        if isinstance(files_or_tmp, NewaveFiles)
        else make_nw_files(files_or_tmp)
    )
    case = NewaveCase(files=files)
    for name, value in parsed.items():
        case.__dict__[name] = value
    # ``case.active_hydros``/``case.id_map`` now consult the hidr cadastro to
    # decide which orphaned FICT reservoirs to keep. Tests that don't exercise
    # that path get a default empty-cadastro hidr so those accessors don't try
    # to read a real binary hidr.dat (an empty cadastro keeps the historical
    # "exclude all FICT" behavior). Tests needing real geometry pass ``hidr=``.
    if "hidr" not in parsed:
        default_hidr = MagicMock()
        default_hidr.cadastro = pd.DataFrame()
        case.__dict__["hidr"] = default_hidr
    return case


_DECOMP_REQUIRED = ("dadger", "vazoes", "hidr")
_DECOMP_OPTIONAL = (
    "dadgnl",
    "renovaveis",
    "polinjus",
    "libs_restricao_eletrica",
    "cortesh",
    "cortes",
)


def make_decomp_files(tmp_path: Path, **overrides: Any) -> DecompFiles:
    """Build a ``DecompFiles`` pointing into *tmp_path*; no I/O.

    Required paths default to ``tmp_path / "<name>"``, optional paths to
    ``None``, and ``revision`` to ``"rv0"``. Pass any attribute as a keyword
    to override (e.g. ``make_decomp_files(tmp_path, dadgnl=tmp_path /
    "dadgnl.rv0")``).
    """
    fields: dict[str, Any] = {"revision": "rv0"}
    for name in _DECOMP_REQUIRED:
        fields[name] = tmp_path / name
    for name in _DECOMP_OPTIONAL:
        fields[name] = None
    fields.update(overrides)
    return DecompFiles(**fields)


def make_decomp_case(files_or_tmp: DecompFiles | Path, **parsed: Any) -> DecompCase:
    """Build a ``DecompCase`` with the given parsed readers pre-cached.

    *files_or_tmp* is either a ``DecompFiles`` or a tmp dir (then
    ``make_decomp_files`` builds one). Each ``parsed`` keyword sets the
    matching cached-property slot directly, so accessing e.g. ``case.dadger``
    returns the supplied object without parsing.
    """
    files = (
        files_or_tmp
        if isinstance(files_or_tmp, DecompFiles)
        else make_decomp_files(files_or_tmp)
    )
    case = DecompCase(files=files)
    for name, value in parsed.items():
        case.__dict__[name] = value
    return case


def hydro_with_group(
    hydro_id: int,
    bus_id: int,
    *,
    name: str | None = None,
    min_generation_mw: float = 0.0,
    max_generation_mw: float = 50.0,
    min_turbined_m3s: float = 0.0,
    max_turbined_m3s: float = 100.0,
    **extra: Any,
) -> dict[str, Any]:
    """Build a 0.13-shaped ``hydros.json`` hydro dict for tests.

    Matches the shape every converter now emits (novomodelo decisions 13/14 →
    §7.6, §7.8, via
    :func:`novomodelo_bridge.core.hydro_units.build_mirror_unit_group`): no
    top-level ``bus_id`` and a single seven-field mirror ``unit_groups``
    entry whose four bounds equal this dict's own ``generation`` envelope
    (novomodelo rule 41's mirror invariant).

    Use this wherever a test needs "some hydro on bus N" rather than a
    real converter output — it replaces a hand-rolled seven-field group
    literal duplicated across test modules.

    Parameters
    ----------
    hydro_id:
        The Novomodelo 0-based hydro id.
    bus_id:
        The plant's bus id (relocated into ``unit_groups[0].bus_id``; no
        longer emitted at the top level).
    name:
        The plant name. Defaults to ``f"HYDRO_{hydro_id}"``.
    min_generation_mw, max_generation_mw, min_turbined_m3s, max_turbined_m3s:
        The plant's generation envelope. Pass these where a test asserts
        specific bounds, so the mirror group tracks them.
    **extra:
        Any additional top-level keys a test needs (``reservoir``,
        ``outflow``, ``downstream_id``, etc.), merged in verbatim.
    """
    plant_name = name if name is not None else f"HYDRO_{hydro_id}"
    hydro: dict[str, Any] = {
        "id": hydro_id,
        "name": plant_name,
        "generation": {
            "min_generation_mw": min_generation_mw,
            "max_generation_mw": max_generation_mw,
            "min_turbined_m3s": min_turbined_m3s,
            "max_turbined_m3s": max_turbined_m3s,
        },
        "unit_groups": [
            build_mirror_unit_group(
                name=plant_name,
                bus_id=bus_id,
                min_generation_mw=min_generation_mw,
                max_generation_mw=max_generation_mw,
                min_turbined_m3s=min_turbined_m3s,
                max_turbined_m3s=max_turbined_m3s,
            )
        ],
    }
    hydro.update(extra)
    return hydro


class _FakeDadger:
    """Return a preset DataFrame (or ``None``) for each register accessor.

    Shared across the decomp constraint-register and results-compare test
    modules; import with ``from tests.conftest import _FakeDadger``.
    """

    def __init__(self, **frames: pd.DataFrame) -> None:
        self._frames = frames

    def __getattr__(self, name: str):  # noqa: ANN204 - test double
        if name.startswith("_"):
            raise AttributeError(name)

        def accessor(df: bool = True) -> pd.DataFrame | None:
            return self._frames.get(name)

        return accessor


def _fake_dataset(*, all_within_tol: bool = False) -> ComparisonDataset:
    """A canonical two-stage, two-variable dataset for the ``compare decomp``
    CLI tests. ``generation_mw`` always matches; ``turbined_m3s`` diverges
    unless *all_within_tol* asks for a fully-passing dataset instead."""
    tidy = pl.DataFrame(
        {
            "entity_type": ["hydro", "hydro", "hydro", "hydro"],
            "entity_id": [0, 0, 0, 0],
            "entity_name": ["A", "A", "A", "A"],
            "bus": [-1, -1, -1, -1],
            "stage": [0, 0, 1, 1],
            "block": [-1, -1, -1, -1],
            "variable": [
                "generation_mw",
                "generation_mw",
                "turbined_m3s",
                "turbined_m3s",
            ],
            "source": ["newave", "novomodelo", "newave", "novomodelo"],
            "value": [100.0, 100.0, 100.0, 100.0 if all_within_tol else 90.0],
        },
        schema=TIDY_SCHEMA,
    )
    turbined_within_tol_rate = 1.0 if all_within_tol else 0.0
    turbined_smape = 0.0 if all_within_tol else 0.12
    summary = pl.DataFrame(
        {
            "variable": ["generation_mw", "turbined_m3s"],
            "count": [1, 1],
            "mean_abs_diff": [0.0, 0.0 if all_within_tol else 10.0],
            "max_abs_diff": [0.0, 0.0 if all_within_tol else 10.0],
            "mean_smape": [0.0, turbined_smape],
            "max_smape": [0.0, turbined_smape],
            "within_tol_rate": [1.0, turbined_within_tol_rate],
            "correlation": [1.0, 0.9],
        },
        schema=SUMMARY_SCHEMA,
    )
    return ComparisonDataset(
        tidy=tidy,
        summary=summary,
        metadata={"unmapped": {"hydro": [], "thermal": [86, 224], "bus": []}},
    )


def _empty_fake_dataset() -> ComparisonDataset:
    return ComparisonDataset(
        tidy=pl.DataFrame(schema=TIDY_SCHEMA),
        summary=pl.DataFrame(schema=SUMMARY_SCHEMA),
        metadata={"unmapped": {"hydro": [], "thermal": [], "bus": []}},
    )


# ---------------------------------------------------------------------------
# Canned pipeline-mock payloads shared by the novomodelo schema-registry and
# newave pipeline tests (via ``_all_converter_patches``).
# ---------------------------------------------------------------------------

# Fake data for converter functions.
_FAKE_LOAD_FACTORS: dict = {"load_factors": []}
_FAKE_LINE_BOUNDS_TABLE = pa.table(
    {
        "line_id": pa.array([], type=pa.int32()),
        "stage_id": pa.array([], type=pa.int32()),
        "direct_mw": pa.array([], type=pa.float64()),
        "reverse_mw": pa.array([], type=pa.float64()),
        "block_id": pa.array([], type=pa.int32()),
    }
)
_FAKE_NCS: dict = {"non_controllable_sources": []}
_FAKE_NCS_FACTORS: dict = {"non_controllable_factors": []}
_FAKE_NCS_BOUNDS_TABLE = pa.table(
    {
        "ncs_id": pa.array([], type=pa.int32()),
        "stage_id": pa.array([], type=pa.int32()),
        "available_generation_mw": pa.array([], type=pa.float64()),
    }
)

_ARQUIVOS_DAT_CONTENT = """\
DADOS GERAIS                : dger.dat
DADOS DOS SUBSISTEMAS       : sistema.dat
CONFIGURACAO HIDRAULICA     : confhd.dat
ALTERACAO DADOS USINAS HIDRO: modif.dat
CONFIGURACAO TERMICA        : conft.dat
DADOS DAS USINAS TERMICAS   : term.dat
DADOS DAS CLASSES TERMICAS  : clast.dat
DADOS DE EXPANSAO HIDRAULICA: exph.dat
ARQUIVO DE EXPANSAO TERMICA : expt.dat
ARQUIVO DE PATAMARES MERCADO: patamar.dat
ARQUIVO DE CORTES DE BENDERS: cortes.dat
ARQUIVO DE CABECALHO CORTES : cortesh.dat
RELATORIO DE CONVERGENCIA   : pmo.dat
RELATORIO DE E. SINTETICAS  : parp.dat
RELATORIO DETALHADO FORWARD : forward.dat
ARQUIVO DE CABECALHO FORWARD: forwarh.dat
ARQUIVO DE S.HISTORICAS S.F.: shist.dat
ARQUIVO DE MANUT.PROG. UTE'S: manutt.dat
ARQUIVO P/DESPACHO HIDROTERM: newdesp.dat
ARQUIVO C/TEND. HIDROLOGICA : vazpast.dat
ARQUIVO C/DADOS DE ITAIPU   : itaipu.dat
ARQUIVO C/DEMAND S. BIDDING : bid.dat
ARQUIVO C/CARGAS ADICIONAIS : c_adic.dat
ARQUIVO C/FATORES DE PERDAS : loss.dat
ARQUIVO C/PATAMARES GTMIN   : gtminpat.dat
ARQUIVO ENSO 1              : elnino.dat
ARQUIVO ENSO 2              : ensoaux.dat
ARQUIVO DSVAGUA             : dsvagua.dat
ARQUIVO P/PENALID. POR DESV.: penalid.dat
ARQUIVO C.GUIA / PENAL.VMINT: curva.dat
ARQUIVO AGRUPAMENTO LIVRE   : agrint.dat
ARQUIVO DESP. ANTEC. GNL    : adterm.dat
ARQUIVO GER. HIDR. MIN      : ghmin.dat
ARQUIVO AVERSAO RISCO - SAR : sar.dat
ARQUIVO AVERSAO RISCO - CVAR: cvar.dat
DADOS DOS RESER.EQ.ENERGIA  : ree.dat
ARQUIVO RESTRICOES ELETRICAS: re.dat
ARQUIVO DE TECNOLOGIAS      : tecno.dat
DADOS DE ABERTURAS          : abertura.dat
ARQUIVO DE EMISSOES GEE     : gee.dat
ARQUIVO DE RESTRICAO DE GAS : clasgas.dat
ARQUIVO DE DADOS SIM. FINAL : simfinal.dat
ARQ. DE CORTES POS ESTUDO   : cortes-pos.dat
ARQ. DE CABECALHO CORTES POS: cortesh-pos.dat
ARQ. C/ VOLUME REF. SAZONAL : volref_saz.dat
"""

_REQUIRED_STUB_FILES = [
    "dger.dat",
    "confhd.dat",
    "conft.dat",
    "sistema.dat",
    "clast.dat",
    "term.dat",
    "ree.dat",
    "patamar.dat",
    "hidr.dat",
    "vazoes.dat",
]


def _make_fake_newave_dir(tmp_path: Path) -> Path:
    """Create a directory with caso.dat, arquivos.dat, and all required stub files."""
    newave_dir = tmp_path / "newave_case"
    newave_dir.mkdir()
    (newave_dir / "caso.dat").write_text("arquivos.dat\n")
    (newave_dir / "arquivos.dat").write_text(_ARQUIVOS_DAT_CONTENT)
    for filename in _REQUIRED_STUB_FILES:
        (newave_dir / filename).write_text("stub")
    return newave_dir


# Canned return values used across pipeline tests.
_FAKE_HYDROS = {"$schema": "http://example", "hydros": [{"id": 0}, {"id": 1}]}
_FAKE_THERMALS = {"$schema": "http://example", "thermals": [{"id": 0}]}
_FAKE_BUSES = {"$schema": "http://example", "buses": [{"id": 0}, {"id": 1}, {"id": 2}]}
_FAKE_LINES = {"$schema": "http://example", "lines": [{"id": 0}]}
_FAKE_PENALTIES = {"bus": {}, "hydro": {}, "line": {}, "non_controllable_source": {}}
_FAKE_STAGES = {
    "$schema": "http://example",
    "policy_graph": {"type": "finite_horizon"},
    "stages": [{"id": i} for i in range(12)],
}
_FAKE_CONFIG = {
    "$schema": "http://example",
    "training": {
        "selection": {"method": "sampled", "forward_passes": 5},
        "stopping_rules": [],
    },
    "simulation": {
        "enabled": True,
        "selection": {"method": "sampled", "num_scenarios": 200},
    },
}
_FAKE_IC = {"$schema": "http://example", "storage": [], "filling_storage": []}
_FAKE_INFLOW_TABLE = pa.table(
    {
        "hydro_id": pa.array([0], type=pa.int32()),
        "stage_id": pa.array([0], type=pa.int32()),
        "mean_m3s": pa.array([100.0], type=pa.float64()),
        "std_m3s": pa.array([10.0], type=pa.float64()),
    }
)
_FAKE_LOAD_TABLE = pa.table(
    {
        "bus_id": pa.array([0], type=pa.int32()),
        "stage_id": pa.array([0], type=pa.int32()),
        "mean_mw": pa.array([500.0], type=pa.float64()),
        "std_mw": pa.array([0.0], type=pa.float64()),
    }
)
_FAKE_HYDRO_ENERGY_PRODUCTIVITY_TABLE = pa.table(
    {
        "hydro_id": pa.array([0, 1], type=pa.int32()),
        "stage_id": pa.array([None, None], type=pa.int32()),
        "equivalent_productivity_mw_per_m3s": pa.array([0.5, 0.6], type=pa.float64()),
        "reference_outflow_m3s": pa.array([None, None], type=pa.float64()),
        "specific_productivity_mw_per_m3s_per_m": pa.array(
            [None, None], type=pa.float64()
        ),
    }
)


def _all_converter_patches(fake_id_map: MagicMock) -> list:  # type: ignore[type-arg]
    """Return patch context managers for all converter functions.

    The parsed case is mocked at ``NewaveCase.from_directory``; its ``id_map``
    is the supplied ``fake_id_map`` (the pipeline now reads ``case.id_map``).
    """
    fake_case = MagicMock()
    fake_case.id_map = fake_id_map
    return [
        patch(
            "novomodelo_bridge.newave.pipeline.NewaveCase.from_directory",
            return_value=fake_case,
        ),
        patch(
            "novomodelo_bridge.newave.pipeline.hydro_conv.convert_hydros",
            return_value=_FAKE_HYDROS,
        ),
        patch(
            "novomodelo_bridge.newave.pipeline.thermal_conv.convert_thermals",
            return_value=_FAKE_THERMALS,
        ),
        patch(
            "novomodelo_bridge.newave.pipeline.network_conv.convert_buses",
            return_value=_FAKE_BUSES,
        ),
        patch(
            "novomodelo_bridge.newave.pipeline.network_conv.convert_lines",
            return_value=_FAKE_LINES,
        ),
        patch(
            "novomodelo_bridge.newave.pipeline.network_conv.convert_penalties",
            return_value=_FAKE_PENALTIES,
        ),
        patch(
            "novomodelo_bridge.newave.pipeline.temporal_conv.convert_stages",
            return_value=_FAKE_STAGES,
        ),
        patch(
            "novomodelo_bridge.newave.pipeline.temporal_conv.convert_config",
            return_value=_FAKE_CONFIG,
        ),
        patch(
            "novomodelo_bridge.newave.pipeline.ic_conv.convert_initial_conditions",
            return_value=_FAKE_IC,
        ),
        patch(
            "novomodelo_bridge.newave.pipeline.stochastic_conv.convert_inflow_stats",
            return_value=_FAKE_INFLOW_TABLE,
        ),
        patch(
            "novomodelo_bridge.newave.pipeline.stochastic_conv.convert_load_stats",
            return_value=_FAKE_LOAD_TABLE,
        ),
        patch(
            "novomodelo_bridge.newave.pipeline.inflow_windows.convert_recent_observation_windows",
            return_value=[],
        ),
        patch(
            "novomodelo_bridge.newave.pipeline.inflow_windows.convert_inflow_history_windows",
            return_value=_FAKE_INFLOW_TABLE,
        ),
        patch(
            "novomodelo_bridge.newave.pipeline.hydro_conv.read_cadastro",
            return_value=MagicMock(),
        ),
        patch(
            "novomodelo_bridge.newave.pipeline.hydro_conv.generate_hydro_geometry",
            return_value=_FAKE_INFLOW_TABLE,  # reuse any small pa.Table
        ),
        patch(
            "novomodelo_bridge.newave.pipeline.constraints_conv.convert_vminop_constraints",
            return_value=None,
        ),
        patch(
            "novomodelo_bridge.newave.pipeline.constraints_conv.convert_electric_constraints",
            return_value=None,
        ),
        patch(
            "novomodelo_bridge.newave.pipeline.constraints_conv.convert_agrint_constraints",
            return_value=None,
        ),
        patch(
            "novomodelo_bridge.newave.pipeline.stochastic_conv.convert_load_factors",
            return_value=_FAKE_LOAD_FACTORS,
        ),
        patch(
            "novomodelo_bridge.newave.pipeline.network_conv.convert_line_bounds",
            return_value=_FAKE_LINE_BOUNDS_TABLE,
        ),
        patch(
            "novomodelo_bridge.newave.pipeline.network_conv.convert_non_controllable_sources",
            return_value=_FAKE_NCS,
        ),
        patch(
            "novomodelo_bridge.newave.pipeline.network_conv.convert_ncs_factors",
            return_value=_FAKE_NCS_FACTORS,
        ),
        patch(
            "novomodelo_bridge.newave.pipeline.network_conv.convert_ncs_stats",
            return_value=_FAKE_NCS_BOUNDS_TABLE,
        ),
        patch(
            "novomodelo_bridge.newave.pipeline.hydro_conv.convert_production_models",
            return_value={"production_models": []},
        ),
        patch(
            "novomodelo_bridge.newave.pipeline.hydro_conv.compute_base_productivities",
            return_value={},
        ),
        patch(
            "novomodelo_bridge.newave.pipeline.hydro_conv.convert_hydro_energy_productivity",
            return_value=_FAKE_HYDRO_ENERGY_PRODUCTIVITY_TABLE,
        ),
        patch(
            "novomodelo_bridge.newave.pipeline.thermal_conv.convert_thermal_bounds",
            return_value=None,
        ),
        patch(
            "novomodelo_bridge.newave.pipeline.hydro_conv.convert_storage_bounds",
            return_value=None,
        ),
    ]
