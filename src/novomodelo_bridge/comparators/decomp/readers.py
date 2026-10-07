"""DECOMP output readers for results comparison.

Reads the ``dec_oper_*.csv`` operation tables and the convergence report
from a DECOMP case directory via ``idecomp``, returning Polars
frames with the source's native column names (1-based ``estagio``,
node/scenario indices as written). Alignment onto Cobre entity ids and
stage indices happens in the comparison layer, not here.

Reader-failure contract: an absent OPTIONAL input yields a typed-empty
frame plus a WARNING log; a present-but-unreadable file, or an absent
REQUIRED input, raises a typed error (``CobreReadError``, ``ValueError``,
or ``FileNotFoundError``) — never a silent empty, since an empty frame
from real-but-broken data fabricates a false zero-vs-zero match
(``.claude/rules/comments.md`` §4; reads route through this layer per
``.claude/rules/bridge.md`` §5). This module and ``cobre_readers`` raise on
present-but-corrupt data; ``newave_readers`` is the genuine outlier — it
best-effort degrades present-but-broken optional inputs to empty/None plus a
WARNING (see its module docstring). DECOMP's operation tables are required,
the source model's and Cobre's optional partitions are not. Every ``dec_oper_*``/
``relato``-backed table here is a REQUIRED input: an absent file raises
``FileNotFoundError`` and a present file that parses empty raises
``ValueError`` — silent-empty DataFrames are idecomp's characteristic
failure mode on unexpected syntax. ``read_relato2_costs`` is the one
deliberate exception, an OPTIONAL per-realization report that degrades
to an empty frame when absent.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Protocol

import pandas as pd
import polars as pl
from idecomp.decomp import (
    DecDesvFpha,
    DecEstatFpha,
    Decomptim,
    DecOperEvap,
    DecOperGnl,
    DecOperInterc,
    DecOperRee,
    DecOperRheSoft,
    DecOperSist,
    DecOperUsih,
    DecOperUsit,
    EcoFpha,
    Relato,
)

from cobre_bridge.core.paths import find_case_insensitive

_LOG = logging.getLogger(__name__)


class _TableFile(Protocol):
    """The idecomp reader surface the operation readers rely on."""

    tabela: pd.DataFrame | None


class _TableReader(Protocol):
    @staticmethod
    def read(path: str) -> _TableFile: ...


def _resolve_result_file(case_dir: Path, filename: str) -> Path | None:
    """Resolve *filename* case-insensitively in *case_dir* (the deck root).

    Every DECOMP result file is read from the case root; a missing
    *case_dir*, or *filename* absent from it, is a silent miss, not an
    error -- callers translate a `None` return into `FileNotFoundError`
    themselves.
    """
    return find_case_insensitive(case_dir, filename)


def _finalize_table(path: Path, table: pd.DataFrame | None) -> pl.DataFrame:
    """Reject an empty/None parse (see the module docstring), debug-log the
    row count, and convert to Polars."""
    if table is None or table.empty:
        raise ValueError(
            f"{path} parsed empty; the run's outputs look incomplete or the "
            "file syntax is unsupported"
        )
    _LOG.debug("Read %s: %d rows", path.name, len(table))
    return pl.from_pandas(table)


def _read_dec_oper(
    case_dir: Path,
    filename: str,
    reader_cls: _TableReader,
) -> pl.DataFrame:
    """Read one ``dec_oper_*`` table, rejecting missing or empty parses."""
    path = _resolve_result_file(case_dir, filename)
    if path is None:
        raise FileNotFoundError(f"{filename} not found in {case_dir}")
    return _finalize_table(path, reader_cls.read(str(path)).tabela)


def read_dec_oper_sist(case_dir: Path) -> pl.DataFrame:
    """Per-(stage, node, scenario, block, submarket) system operation.

    Includes demand, generation by family (hydro/thermal/anticipated/wind/
    small plants), exchanges, the Itaipu 50/60 Hz split, deficit, stored
    energy, and the marginal cost (``cmo``).
    """
    return _read_dec_oper(case_dir, "dec_oper_sist.csv", DecOperSist)


def read_dec_oper_usih(case_dir: Path) -> pl.DataFrame:
    """Per-hydro operation: storage, flows (natural/incremental/turbined/
    spilled/withdrawn/evaporated), generation, and available capacity
    (``potencia_disponivel_MW`` — the availability-rule oracle)."""
    return _read_dec_oper(case_dir, "dec_oper_usih.csv", DecOperUsih)


def read_dec_oper_usit(case_dir: Path) -> pl.DataFrame:
    """Per-thermal operation: generation with its effective min/max bounds
    and incremental cost."""
    return _read_dec_oper(case_dir, "dec_oper_usit.csv", DecOperUsit)


def read_dec_oper_interc(case_dir: Path) -> pl.DataFrame:
    """Per-exchange operation: origin/destination flows, losses, and the
    effective capacity."""
    return _read_dec_oper(case_dir, "dec_oper_interc.csv", DecOperInterc)


def read_dec_oper_rhesoft(case_dir: Path) -> pl.DataFrame:
    """Per-RHE-soft-constraint operation: achieved stored energy against its
    limit, and the violation the soft treatment absorbed.

    Native columns: ``estagio``, ``no``, ``cenario``, ``codigo_restricao``,
    ``limite_MW``, ``valor_MW``, ``violacao_absoluta_MW``,
    ``violacao_percentual``. ``codigo_restricao`` is the same ``HE``/``CM``
    register id (``constraint_registers.ConstraintRecord.constraint_id``)
    the conversion-time RHE emitter (``decomp.converters.constraints.
    emit_rhe_generics``) names its cobre constraint after (``"RHE_<id>"``).

    This is the RHE (soft minimum-stored-energy) constraints'
    own achieved LHS, straight from the source model -- the Constraints tab's
    DECOMP-side LHS derivation (`results._rhe_lhs_lookup`) reads
    ``valor_MW``/``violacao_absoluta_MW`` from here rather than re-deriving
    the register's ρ_acum-weighted cascade sum a second time."""
    return _read_dec_oper(case_dir, "dec_oper_rhesoft.csv", DecOperRheSoft)


def read_dec_oper_gnl(case_dir: Path) -> pl.DataFrame:
    """Per-anticipated-thermal (GNL) operation: dispatch bounds, incremental
    cost, and the fuel cost (``custo_geracao``, native k$). Resolved via the
    same root-only lookup shared with every other ``dec_oper_*`` table."""
    return _read_dec_oper(case_dir, "dec_oper_gnl.csv", DecOperGnl)


def read_dec_oper_evap(case_dir: Path) -> pl.DataFrame:
    """Per-hydro/stage/node/scenario reservoir evaporation.

    Verified (idecomp 1.14.2, ``DecOperEvap.tabela``) native columns:
    ``estagio``, ``no``, ``cenario``, ``codigo_usina``, ``nome_usina``,
    ``codigo_submercado``, ``codigo_ree``, ``volume_util_inicial_hm3``,
    ``volume_util_inicial_percentual``, ``volume_util_final_hm3``,
    ``volume_util_final_percentual``, ``evaporacao_modelo_hm3`` (the fitted
    monthly-coefficient estimate), ``evaporacao_calculada_hm3`` (the volume
    the run's own water balance actually applied over the stage -- hm³, a
    volume, not a flow), ``desvio_absoluto_hm3``, ``desvio_percentual``.
    Unlike `read_dec_oper_usih`/`read_dec_oper_ree`, this table carries no
    ``patamar`` column -- it is already one row per (stage, node, scenario,
    plant), with no sub-stage block breakdown to fold. The
    source for the evaporation comparison (`results.
    _evaporation_result_comparisons`), which reconciles
    ``evaporacao_calculada_hm3`` (hm³) against Cobre's ``evaporation_m3s``
    (m³/s) via the stage's own hours."""
    return _read_dec_oper(case_dir, "dec_oper_evap.csv", DecOperEvap)


def read_dec_oper_ree(case_dir: Path) -> pl.DataFrame:
    """Per-REE (reservoir-equivalent-energy) operation: natural inflow energy
    (``ena_MWmes``) and stored energy (``earm_inicial``/``earm_final``, both
    absolute ``_MWmes`` and ``_percentual``, plus ``earm_maximo_MWmes``), one
    row per (stage, node, scenario, REE). The DECOMP-side source
    for the REE energy rollup -- Cobre has no REE entity, so its counterpart
    is a membership-weighted sum of plant output (see
    `results._ree_result_comparisons`)."""
    return _read_dec_oper(case_dir, "dec_oper_ree.csv", DecOperRee)


def _resolve_revisioned_file(case_dir: Path, stem: str) -> Path | None:
    """Locate a revision-suffixed result file (``<stem>.rvN``) in *case_dir*
    (the deck root).

    Shared by every reader whose filename carries the deck's own revision
    number as a suffix instead of a fixed extension -- the general report
    (``relato.rvN``) and the FPHA deviation/grid-echo files
    (``dec_estatfpha.rvN``, ``dec_desvfpha.rvN``, ``eco_fpha.rvN``), unlike
    the fixed-name ``dec_oper_*.csv``/``decomp.tim`` tables
    `_resolve_result_file` resolves.
    """
    pattern = re.compile(rf"^{re.escape(stem)}\.rv\d+$", re.IGNORECASE)
    try:
        for entry in sorted(case_dir.iterdir()):
            if entry.is_file() and pattern.match(entry.name):
                return entry
    except OSError:
        pass
    return None


def _resolve_relato(case_dir: Path) -> Path | None:
    """Locate the revision-suffixed general report (``relato.rvN``) in
    *case_dir* (the deck root)."""
    return _resolve_revisioned_file(case_dir, "relato")


def _resolve_relato2(case_dir: Path) -> Path | None:
    """Locate the revision-suffixed second report (``relato2.rvN``) in
    *case_dir* (the deck root).

    ``relato2`` holds the per-realization operation report for the
    scenario-fan stage(s) the aggregate ``relato`` omits; it is optional -- a
    deck without a fan stage has none. The ``^relato\\.rv\\d+$`` pattern
    `_resolve_relato` uses does not match ``relato2.rvN`` (the char after
    ``relato`` is a digit, not ``.``), so the two never collide.
    """
    return _resolve_revisioned_file(case_dir, "relato2")


def _read_relato_table(case_dir: Path, attr: str) -> pl.DataFrame:
    """Read one named pandas table off the general report (``relato.rvN``).

    Resolves the relato file via `_resolve_relato` (root-only), reads it
    via ``Relato.read``, and pulls the table at ``getattr(relato, attr)``.
    Shared by every ``relato``-backed reader (convergence, energy balance,
    and later costs and membership).
    """
    path = _resolve_relato(case_dir)
    if path is None:
        raise FileNotFoundError(f"no relato.rvN found in {case_dir}")
    table = getattr(Relato.read(str(path)), attr)
    if table is None or table.empty:
        raise ValueError(f"{path} has no {attr} table")
    return pl.from_pandas(table)


def read_relato_convergence(case_dir: Path) -> pl.DataFrame:
    """Read the convergence table (``iteracao``, ``zinf``, ``zsup``,
    ``gap_percentual``, …) from the general report."""
    return _read_relato_table(case_dir, "convergencia")


def read_decomp_tim(case_dir: Path) -> pl.DataFrame:
    """Read the wall-clock timing table (``Etapa``, ``Tempo``) from
    ``decomp.tim``.

    ``Etapa`` names the phase (e.g. ``Leitura de Dados``, ``Convergencia``,
    ``Impressao``, ``Tempo Total``); columns are returned unrenamed, with no
    phase-name mapping — that belongs to the caller.
    """
    path = _resolve_result_file(case_dir, "decomp.tim")
    if path is None:
        raise FileNotFoundError(f"decomp.tim not found in {case_dir}")
    return _finalize_table(path, Decomptim.read(str(path)).tempos_etapas)


def read_relato_balance(case_dir: Path) -> pl.DataFrame:
    """Read the per-submarket energy balance table (demand, generation by
    source, purchase/sale, ENA, and EARM in/out) from the general report."""
    return _read_relato_table(case_dir, "balanco_energetico")


def read_relato_costs(case_dir: Path) -> pl.DataFrame:
    """Read the per-(stage, scenario) operating cost table (present/future
    cost, thermal generation, deviation/spillage/turbining penalties, and
    the per-submarket marginal cost) from the general report.

    Costs are **native k$**, unconverted -- converted once in the analysis
    layer, see `results.reconcile_kdollars_to_reais`.
    """
    return _read_relato_table(case_dir, "relatorio_operacao_custos")


def read_relato2_costs(case_dir: Path) -> pl.DataFrame:
    """Read the fan-stage per-(stage, scenario) operating cost table from the
    second report (``relato2.rvN``).

    ``relato2`` carries the detailed per-realization operation report for the
    scenario-fan (e.g. monthly) stage(s) that ``relato``'s aggregate
    ``relatorio_operacao_custos`` omits -- the *same* cost columns
    (``custo_presente``/``custo_futuro``/``geracao_termica``/penalties) with
    one row per (``estagio``, ``cenario``) and the real tree ``probabilidade``
    per row. Parsed with idecomp's ``Relato`` reader, which reads ``relato2``
    with the same block format as ``relato``.

    **Optional** -- returns an empty frame (not an error) when ``relato2`` is
    absent, so a caller can fall back to ``relato`` alone: a deck with no
    scenario fan has no second report. This is the one deliberately
    silent-empty reader (a missing optional file is not a degraded input),
    unlike the mandatory ``relato``/``dec_oper_*`` readers.

    Costs are **native k$**, unconverted -- converted once in the analysis
    layer, see `results.reconcile_kdollars_to_reais`.
    """
    path = _resolve_relato2(case_dir)
    if path is None:
        return pl.DataFrame()
    table = Relato.read(str(path)).relatorio_operacao_custos
    if table is None or table.empty:
        return pl.DataFrame()
    return pl.from_pandas(table)


def read_relato_expected_cost(case_dir: Path) -> pl.DataFrame:
    """Read the per-parcela expected operating cost table (one ``estagio_N``
    column per stage) from the general report.

    Costs are **native k$**, unconverted -- converted once in the analysis
    layer, see `results.reconcile_kdollars_to_reais`.
    """
    return _read_relato_table(case_dir, "custo_operacao_valor_esperado")


def read_relato_membership(case_dir: Path) -> pl.DataFrame:
    """Read the hydro-plant -> REE -> submarket membership table
    (``codigo_usina``, ``nome_usina``, ``codigo_ree``, ``nome_ree``,
    ``codigo_submercado``, ``nome_submercado``, ``nome_submercado_newave``)
    from the general report.

    The sole source that attributes a hydro plant to its REE --
    neither `DecompIdMap` nor any ``dec_oper_*`` table carries that
    membership, so `results._ree_result_comparisons` rolls Cobre's
    per-plant energy up to the REE level through this table instead.
    """
    return _read_relato_table(case_dir, "uhes_rees_submercados")


# --- FPHA (fitted production function) readers ---
#
# Three files, all resolved via `_resolve_revisioned_file` (root-only,
# `<stem>.rvN`): `dec_desvfpha` (per-hydro/stage/scenario/block deviation
# between the "true" nonlinear generation and the piecewise-linear fit the
# LP actually consumed), `eco_fpha` (per-hydro/stage fitting-grid bounds and
# node counts), and `dec_estatfpha` (a single deck-wide deviation summary,
# no per-hydro/stage key). None of the three carries the fitted PLANE
# coefficients themselves (those live in a fourth, undeclared file,
# `avl_cortesfpha.rvN` / idecomp's `AvlCortesFpha`) -- see
# `results._fpha_metrics`'s docstring for how it works
# around that gap.


def _read_revisioned_table(
    case_dir: Path,
    stem: str,
    reader_cls: _TableReader,
) -> pl.DataFrame:
    """Read one revision-suffixed (``<stem>.rvN``) ``.tabela``-shaped table,
    rejecting missing or empty parses.

    Mirrors :func:`_read_dec_oper`'s shape exactly, but resolves via
    `_resolve_revisioned_file` instead of an exact filename -- shared by the
    FPHA readers whose filenames carry the deck's own revision suffix the
    way ``dec_oper_*.csv`` does not.
    """
    path = _resolve_revisioned_file(case_dir, stem)
    if path is None:
        raise FileNotFoundError(f"{stem}.rvN not found in {case_dir}")
    return _finalize_table(path, reader_cls.read(str(path)).tabela)


def read_dec_desvfpha(case_dir: Path) -> pl.DataFrame:
    """Read the per-hydro FPHA deviation table (``dec_desvfpha.rvN``).

    One row per (hydro, stage, node, block): the realized ``(volume_total_hm3,
    vazao_turbinada_m3s, vazao_vertida_m3s)`` operating point, the "true"
    nonlinear generation (``geracao_hidraulica_fph``), the piecewise-linear
    fit the LP actually consumed at that point (``geracao_hidraulica_fpha``),
    and their signed deviation (``desvio_absoluto_MW``/``desvio_percentual``
    -- "absoluto" means "in MW", not ``abs()``-ed; ``fpha - fph``).
    """
    return _read_revisioned_table(case_dir, "dec_desvfpha", DecDesvFpha)


def read_eco_fpha(case_dir: Path) -> pl.DataFrame:
    """Read the per-hydro/stage FPHA fitting-grid echo (``eco_fpha.rvN``).

    One row per (hydro, stage): the volume/turbined-flow fitting-grid bounds
    and node counts (``numero_pontos_volume_armazenado``/
    ``numero_pontos_vazao_turbinada``, ``volume_armazenado_minimo/maximo``,
    ``vazao_turbinada_minima/maxima``) and the plant's generation bounds --
    the grid the source model's own fit was built on, not the fitted
    coefficients themselves.
    """
    return _read_revisioned_table(case_dir, "eco_fpha", EcoFpha)


def read_dec_estatfpha(case_dir: Path) -> pl.DataFrame:
    """Read the deck-wide FPHA deviation summary (``dec_estatfpha.rvN``).

    A ``variavel``/``valor`` key-value table (e.g. mean deviation in MW, %
    of total generation, split reservoir vs. run-of-river) -- unlike every
    other reader in this module, it carries no per-hydro or per-stage key,
    so it cannot join into a per-(hydro, stage) frame; it is surfaced as
    deck-wide context only (see `results._log_decomp_fpha_deck_summary`).
    """
    path = _resolve_revisioned_file(case_dir, "dec_estatfpha")
    if path is None:
        raise FileNotFoundError(f"dec_estatfpha.rvN not found in {case_dir}")
    return _finalize_table(path, DecEstatFpha.read(str(path)).estatisticas_desvios)
