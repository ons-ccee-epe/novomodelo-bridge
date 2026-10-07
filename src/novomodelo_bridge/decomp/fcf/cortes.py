"""Boundary cut model and ``CORTESH``/``cortes`` readers for the source model.

The source model stores its boundary FCF as a pair of files: a header
(``cortesh.dat``) describing the cut state space — plant index order,
submercados, patamar count, GNL lag depth, and the individualized-vs-
aggregated flag — and a record file (``cortes.dat`` for the consolidated
multi-stage archive, ``cortes-<estagio>.dat`` for a single-stage partition
export; the canonical name is set by the deck's ``FC`` record) carrying the
cut coefficients. ``inewave.newave.Cortesh`` parses the header into
:class:`CortesHeader`, the manifest both :func:`read_cortes` and the mapper
key off. :func:`read_cortes` reads the boundary-stage records into
:class:`StageCutRecord`/:class:`BoundaryCuts` via ``inewave`` 1.15.0's
``Cortes.from_cortesh``, which returns a ``.cortes`` frame of named columns
(``rhs``, ``pi_gnl_sbm{s}_pat{p}_lag{l}``, ``pi_varm_uhe{code}``,
``pi_qafl_uhe{code}_lag{l}``, ``pi_mx_sar_uhe{code}``, plus the per-cut
provenance columns ``indice_corte``, ``iteracao_construcao``,
``indice_forward``, ``iteracao_desativacao``) — a faithful transcription of
the named layout, not raw byte arithmetic.

``Cortesh.STORAGE`` is binary and its ``read`` classmethod accepts a file
path directly (confirmed against both a non-GNL deck,
``example/newave_rodada/cortesh.dat``, and a GNL deck,
``example/decomp-set-24-rv0/cortesh.dat``) — the same ``Cortesh.read(str(path))``
idiom :func:`cobre_bridge.decomp.converters.hydro.entity.read_hidr` uses
for ``Hidr.read``, not a decoded-text buffer.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from inewave.newave import Cortes, Cortesh

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd  # type: ignore[import-untyped]  # pandas-stubs not installed

_NONZERO_TOLERANCE = 1e-9


@dataclass(frozen=True)
class CortesHeader:
    """The ``cortesh.dat`` manifest: cut state-space shape and chain heads.

    ``plant_codes`` is in ``indice_usina`` slot order (the state-slot
    order the cut coefficients are laid out in) — **not** sorted by
    ``codigo_usina``, which would silently permute every coefficient.
    """

    plant_codes: tuple[int, ...]
    submercado_codes: tuple[int, ...]
    n_patamares: int
    lag_maximo_gnl: int
    n_plants: int
    individualized: bool
    record_size: int
    last_cut_record_by_stage: tuple[int, ...]


@dataclass(frozen=True)
class StageCutRecord:
    """One boundary cut's provenance and coefficients, in header slot order.

    Populated by the record reader; defined here so the
    header reader and its consumers share one type. ``cut_id``,
    ``iteration``, ``forward_pass_index``, and ``is_active`` are the
    per-cut provenance columns the checkpoint writer
    needs — carried verbatim from the ``from_cortesh`` frame's
    ``indice_corte``, ``iteracao_construcao``, ``indice_forward``, and
    ``iteracao_desativacao == 0`` respectively. The reader carries every
    record regardless of ``is_active``; deactivated-cut filtering is the
    writer's concern, not the reader's.
    """

    cut_id: int
    iteration: int
    forward_pass_index: int
    is_active: bool
    rhs: float
    pi_varm: tuple[float, ...]
    pi_qafl: tuple[tuple[float, ...], ...]
    pi_gnl: tuple[float, ...]


@dataclass(frozen=True)
class BoundaryCuts:
    """The boundary-stage cut family: header plus its cut records.

    Assembled by the record reader; defined here so the
    header reader and its consumers share one type.
    """

    header: CortesHeader
    boundary_stage: int
    records: tuple[StageCutRecord, ...]


@dataclass(frozen=True)
class CutFamilySummary:
    """Which cut coefficient families are live in a :class:`BoundaryCuts`.

    A header triage over the active cuts: which plants carry a nonzero
    storage or inflow-lag coefficient, which GNL slots are live, and the
    RHS coefficient scale. Plain data — no logging, no
    :class:`~cobre_bridge.core.diagnostics.Diagnostic` — so it stays reusable
    both as a self-check and as diagnostic input.
    """

    n_active_cuts: int
    storage_nonzero_plants: int
    lag_nonzero_by_depth: tuple[int, ...]
    gnl_nonzero_slots: int
    rhs_min: float
    rhs_max: float


def summarize_cut_families(cuts: BoundaryCuts) -> CutFamilySummary:
    """Triage which cut coefficient families are nonzero in ``cuts``.

    A single pass over ``cuts.records``: OR-accumulates, per plant, whether
    ``pi_varm`` is ever nonzero across the active cuts; per (plant, lag
    depth 1..12), whether ``pi_qafl`` is ever nonzero at that depth; and per
    GNL slot, whether ``pi_gnl`` is ever nonzero — each a plant-level (or
    slot-level) OR across records, never a per-cut sum. ``rhs_min``/
    ``rhs_max`` track the RHS coefficient scale across the active cuts.

    Raises ``ValueError`` if ``cuts.records`` is empty — a boundary with no
    active cuts is a read error, not an empty-but-valid state.
    """
    if not cuts.records:
        raise ValueError(
            f"BoundaryCuts for stage {cuts.boundary_stage} has no active cuts"
        )

    n_plants = cuts.header.n_plants
    n_gnl_slots = len(cuts.records[0].pi_gnl)

    storage_nonzero = [False] * n_plants
    lag_nonzero_by_plant = [[False] * 12 for _ in range(n_plants)]
    gnl_nonzero = [False] * n_gnl_slots
    rhs_min = float("inf")
    rhs_max = float("-inf")

    for record in cuts.records:
        rhs_min = min(rhs_min, record.rhs)
        rhs_max = max(rhs_max, record.rhs)
        for i, value in enumerate(record.pi_varm):
            if abs(value) > _NONZERO_TOLERANCE:
                storage_nonzero[i] = True
        for i, lags in enumerate(record.pi_qafl):
            plant_lags = lag_nonzero_by_plant[i]
            for lag_index, value in enumerate(lags):
                if abs(value) > _NONZERO_TOLERANCE:
                    plant_lags[lag_index] = True
        for i, value in enumerate(record.pi_gnl):
            if abs(value) > _NONZERO_TOLERANCE:
                gnl_nonzero[i] = True

    lag_nonzero_by_depth = tuple(
        sum(1 for plant_lags in lag_nonzero_by_plant if plant_lags[lag_index])
        for lag_index in range(12)
    )

    return CutFamilySummary(
        n_active_cuts=len(cuts.records),
        storage_nonzero_plants=sum(storage_nonzero),
        lag_nonzero_by_depth=lag_nonzero_by_depth,
        gnl_nonzero_slots=sum(gnl_nonzero),
        rhs_min=rhs_min,
        rhs_max=rhs_max,
    )


def required_inflow_lag_depth(summary: CutFamilySummary) -> int:
    """The deepest inflow-lag depth any boundary cut references, ``1``-based.

    The boundary cuts price inflow-lag state only out to the deepest lag with a
    nonzero ``pi_qafl`` coefficient (``summary.lag_nonzero_by_depth[d-1] > 0``
    for calendar-month depth ``d`` in ``1..12``). That depth is exactly the
    number of ``HydroInflowLag`` slots cobre must reserve so the terminal
    boundary cut can price its conditioning history — the bridge-side equivalent
    of cobre's ``boundary_cut_lag_depth`` (which reads the same quantity off the
    written manifest at load).

    Declared to ``cobre.write_policy_checkpoint`` so the writer reserves that many
    canonical ``HydroInflowLag`` slots in the checkpoint: a DECOMP case has no
    PAR(p) model for cobre to infer the depth from, so the depth must be carried
    in the boundary checkpoint itself (cobre's programmatic inflow-lag-depth
    bootstrap via ``write_policy_checkpoint(inflow_lag_depth=N)``).
    It is never written to ``config.json`` (cobre 0.14 retired that user field).

    Returns ``0`` when no cut carries a nonzero lag coefficient — the boundary
    prices no inflow-lag state, so no slots need reserving.
    """
    return max(
        (
            depth
            for depth, count in enumerate(summary.lag_nonzero_by_depth, start=1)
            if count > 0
        ),
        default=0,
    )


def read_cortesh(path: Path) -> CortesHeader:
    """Read ``cortesh.dat`` into a :class:`CortesHeader`.

    Resolves the deck's own individualized boundary stage
    (``estagio_individualizado_inicial``) to drive :func:`_build_header`'s
    per-stage gate for this standalone read. A production deck for the
    source model is routinely **hybrid** (``tipo_agregacao_caso == 0``):
    aggregated stages up to some point, individualized (plant-space) stages
    after it — this reader accepts both that shape and a fully-
    individualized deck (``tipo_agregacao_caso == 1``), and excludes
    fictitious (accounting-only) plants from the returned
    ``n_plants``/``plant_codes``.

    Raises ``ValueError`` if the deck's own ``estagio_individualizado_inicial``
    stage is not itself within the declared individualized band (not
    expected from a valid deck; defensive), or if the retained
    non-fictitious ``indice_usina`` values are not unique and within
    ``1..numero_maximo_uhes``.
    """
    cortesh = Cortesh.read(str(path))
    return _build_header(
        cortesh, boundary_stage=cortesh.estagio_individualizado_inicial
    )


def _build_header(cortesh: Cortesh, *, boundary_stage: int) -> CortesHeader:
    """Build a :class:`CortesHeader` from an already-loaded ``Cortesh``.

    Gates on whether ``boundary_stage`` itself is individualized rather than
    on the deck's overall aggregation mode: accepts both a fully-
    individualized deck (``tipo_agregacao_caso == 1``) and a hybrid deck
    (``tipo_agregacao_caso == 0``) whose ``boundary_stage`` falls inside the
    declared individualized band (``estagio_individualizado_inicial``..
    ``estagio_individualizado_final``) — a production deck for the source
    model routinely runs aggregated stages up to some point and
    individualized (plant-space, real boundary-cut coefficients) stages
    after it.

    Fictitious (accounting-only) plants in ``dados_uhes`` carry no cut
    columns in the record file, so they are excluded from
    ``n_plants``/``plant_codes``; the non-fictitious subset legitimately
    has holes in the deck-wide ``1..numero_maximo_uhes`` ``indice_usina``
    range (contiguity holds only over *all* plants, not the non-fictitious
    subset).

    Shared by :func:`read_cortesh` (which loads the file itself and
    resolves the deck's own individualized boundary stage) and
    :func:`read_cortes` (which passes its resolved boundary stage — trailer-
    derived or caller-supplied — and also resolves the consolidated
    archive's global chain-head index).
    """
    band_start = cortesh.estagio_individualizado_inicial
    band_end = cortesh.estagio_individualizado_final
    if not (band_start <= boundary_stage <= band_end):
        raise ValueError(
            f"boundary_stage {boundary_stage} is outside the individualized "
            f"band [{band_start}, {band_end}]; the importer only reads "
            "plant-space cuts at an individualized boundary stage"
        )

    n_max_plants = cortesh.numero_maximo_uhes
    uhes = cortesh.dados_uhes[cortesh.dados_uhes["ficticia"] == 0].sort_values(
        "indice_usina"
    )
    indices = [int(value) for value in uhes["indice_usina"]]
    n_plants = len(indices)
    if len(set(indices)) != n_plants or any(
        not (1 <= index <= n_max_plants) for index in indices
    ):
        raise ValueError(
            "non-fictitious indice_usina values are not unique and within "
            f"1..{n_max_plants}: {indices}"
        )
    plant_codes = tuple(int(code) for code in uhes["codigo_usina"])

    submercado_codes = tuple(
        int(code) for code in cortesh.dados_submercados["codigo_submercado"]
    )
    last_cut_record_by_stage = tuple(
        int(value)
        for value in cortesh.ultimo_registro_cortes_estagio["indice_ultimo_corte"]
    )

    return CortesHeader(
        plant_codes=plant_codes,
        submercado_codes=submercado_codes,
        n_patamares=cortesh.numero_patamares,
        lag_maximo_gnl=cortesh.lag_maximo_gnl,
        n_plants=n_plants,
        individualized=True,
        record_size=cortesh.tamanho_registro_individualizado,
        last_cut_record_by_stage=last_cut_record_by_stage,
    )


def _read_trailer(
    cortes_path: Path, record_size: int
) -> tuple[int, int, int, int] | None:
    """Recover the single-stage partition export's trailer, if present.

    ``inewave`` 1.15.0's ``Cortes.from_cortesh``/``SecaoDadosCortes`` has no
    accessor for it — it silently discards the sentinel record (the record
    whose ``rhs == 0.0``) when building ``.cortes``. This reads the file's
    last physical ``record_size`` bytes directly and returns the leading
    ``int32[4]`` — ``(study_start_month, study_start_year, cut_stage_month,
    cut_stage_year)`` — when that record is the sentinel (confirmed
    ``(9, 2024, 10, 2024)`` on the GNL deck's ``cortes-010.dat``). Returns
    ``None`` for the consolidated archive, whose last physical record is a
    real cut (nonzero ``rhs``) and carries no trailer.
    """
    file_size = cortes_path.stat().st_size
    if file_size == 0 or file_size % record_size != 0:
        raise ValueError(
            f"{cortes_path} size {file_size} is not a positive multiple of "
            f"record_size {record_size}"
        )
    with cortes_path.open("rb") as handle:
        handle.seek(file_size - record_size)
        leading_ints = struct.unpack("<4i", handle.read(16))
        rhs = struct.unpack("<d", handle.read(8))[0]
    if rhs != 0.0:
        return None
    return leading_ints


def _resolve_global_chain_head(cortesh: Cortesh, boundary_stage: int) -> int:
    """Resolve the consolidated archive's chain-head index for ``boundary_stage``.

    Looks up the ``(tipo_estagio == "estudo", estagio == boundary_stage)``
    row of ``cortesh.ultimo_registro_cortes_estagio``.
    """
    urce = cortesh.ultimo_registro_cortes_estagio
    matches = urce[
        (urce["tipo_estagio"] == "estudo") & (urce["estagio"] == boundary_stage)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one 'estudo' row for estagio={boundary_stage} in "
            f"cortesh.ultimo_registro_cortes_estagio, found {len(matches)}"
        )
    return int(matches["indice_ultimo_corte"].iloc[0])


def _gnl_columns(cortesh: Cortesh, header: CortesHeader) -> list[str]:
    """Named GNL columns in ``[sbm-major x pat x lag]`` order.

    Built from ``cortesh.numero_submercados`` (the *real*, non-fictitious
    submercado count ``from_cortesh`` itself uses to lay out the GNL block
    positionally 1..S) — **not** ``len(header.submercado_codes)``, which
    includes a fictitious no-market submercado (``NOFICT1``) that carries no
    GNL columns at all. Empty when ``header.lag_maximo_gnl == 0``.
    """
    return [
        f"pi_gnl_sbm{sbm}_pat{pat}_lag{lag}"
        for sbm in range(1, cortesh.numero_submercados + 1)
        for pat in range(1, header.n_patamares + 1)
        for lag in range(1, header.lag_maximo_gnl + 1)
    ]


def _validate_sar_zero(
    df: pd.DataFrame, sar_cols: list[str], plant_codes: tuple[int, ...]
) -> None:
    """Assert every ``pi_mx_sar_uhe*`` value is ~0; raise naming the offender.

    The importer never maps the risk-aversion-surface (SAR) tail; a nonzero
    value means the layout assumption behind the column grouping is
    violated and must surface loudly, not be silently dropped.
    """
    if not sar_cols:
        return
    sar_values = df[sar_cols].to_numpy(dtype=float)
    bad_rows, bad_cols = np.nonzero(np.abs(sar_values) >= _NONZERO_TOLERANCE)
    if bad_rows.size == 0:
        return
    plant_code = plant_codes[int(bad_cols[0])]
    value = float(sar_values[bad_rows[0], bad_cols[0]])
    raise ValueError(
        f"nonzero SAR coefficient pi_mx_sar_uhe{plant_code} = {value!r}; "
        "the importer never maps SAR"
    )


def _build_records(
    df: pd.DataFrame, header: CortesHeader, cortesh: Cortesh
) -> tuple[StageCutRecord, ...]:
    """Build :class:`StageCutRecord` tuples from ``.cortes``'s named columns.

    Coefficients are sliced by explicit column name (never by position) so
    ``pi_varm[i]``/``pi_qafl[i]`` always correspond to ``header.plant_codes[i]``,
    regardless of the frame's own column order.
    """
    varm_cols = [f"pi_varm_uhe{code}" for code in header.plant_codes]
    qafl_cols = [
        f"pi_qafl_uhe{code}_lag{lag}"
        for code in header.plant_codes
        for lag in range(1, 13)
    ]
    sar_cols = [f"pi_mx_sar_uhe{code}" for code in header.plant_codes]
    gnl_cols = _gnl_columns(cortesh, header)

    if len(varm_cols) != header.n_plants:
        raise ValueError(
            f"pi_varm width {len(varm_cols)} != n_plants {header.n_plants}"
        )
    expected_gnl_width = (
        cortesh.numero_submercados * header.n_patamares * header.lag_maximo_gnl
    )
    if len(gnl_cols) != expected_gnl_width:
        raise ValueError(
            f"GNL column count {len(gnl_cols)} does not match expected width "
            f"{expected_gnl_width} (numero_submercados={cortesh.numero_submercados}, "
            f"n_patamares={header.n_patamares}, lag_maximo_gnl={header.lag_maximo_gnl})"
        )

    _validate_sar_zero(df, sar_cols, header.plant_codes)

    cut_id_values = df["indice_corte"].to_numpy(dtype=int).tolist()
    iteration_values = df["iteracao_construcao"].to_numpy(dtype=int).tolist()
    forward_pass_values = df["indice_forward"].to_numpy(dtype=int).tolist()
    deactivation_values = df["iteracao_desativacao"].to_numpy(dtype=int).tolist()
    rhs_values = df["rhs"].to_numpy(dtype=float).tolist()
    varm_values = df[varm_cols].to_numpy(dtype=float).tolist()
    qafl_values = (
        df[qafl_cols]
        .to_numpy(dtype=float)
        .reshape(len(df), header.n_plants, 12)
        .tolist()
    )
    gnl_values = df[gnl_cols].to_numpy(dtype=float).tolist() if gnl_cols else None

    return tuple(
        StageCutRecord(
            cut_id=cut_id_values[i],
            iteration=iteration_values[i],
            forward_pass_index=forward_pass_values[i],
            is_active=deactivation_values[i] == 0,
            rhs=rhs_values[i],
            pi_varm=tuple(varm_values[i]),
            pi_qafl=tuple(tuple(lags) for lags in qafl_values[i]),
            pi_gnl=tuple(gnl_values[i]) if gnl_values is not None else (),
        )
        for i in range(len(df))
    )


def read_cortes(
    cortes_path: Path,
    cortesh: Cortesh,
    *,
    boundary_stage: int | None = None,
) -> BoundaryCuts:
    """Read the boundary-stage cut records from ``cortes_path``.

    Takes the raw ``Cortesh`` (not :class:`CortesHeader`) so this can both
    build its own header (:func:`_build_header`, the same logic
    :func:`read_cortesh` uses) and resolve the consolidated archive's
    global chain-head index from ``cortesh.ultimo_registro_cortes_estagio``.
    The header is built from the *resolved* boundary stage below (trailer-
    derived or caller-supplied), which also drives :func:`_build_header`'s
    individualized-band gate and its fictitious-plant exclusion.

    Detects the file shape automatically:

    - **Single-stage partition export** (e.g. ``cortes-010.dat``): carries a
      sentinel/trailer record. Read via ``por_estagio=True``; when
      ``boundary_stage`` is ``None`` it is derived, calendar-anchored, from
      the trailer's ``(cut_stage_month, cut_stage_year)`` as
      ``(cut_stage_year - cortesh.ano_inicio_estudo) * 12 + cut_stage_month``.
    - **Consolidated archive** (``cortes.dat``): no trailer.
      ``boundary_stage`` is *required* — the caller supplies the DECOMP
      coupling stage — and its global chain-head index is resolved from
      ``cortesh.ultimo_registro_cortes_estagio`` before reading with
      ``por_estagio=False``.

    Raises ``ValueError`` if the resolved boundary stage falls outside the
    deck's individualized band, if a ``pi_mx_sar_uhe*`` coefficient is
    nonzero, if the file size is not a multiple of the header's record
    size, or (for the consolidated archive) if ``boundary_stage`` is
    missing or has no unique ``estudo`` row in
    ``cortesh.ultimo_registro_cortes_estagio``.
    """
    record_size = cortesh.tamanho_registro_individualizado
    trailer = _read_trailer(cortes_path, record_size)

    resolved_boundary_stage: int
    if trailer is not None:
        _study_month, _study_year, cut_stage_month, cut_stage_year = trailer
        if boundary_stage is None:
            resolved_boundary_stage = (
                cut_stage_year - cortesh.ano_inicio_estudo
            ) * 12 + cut_stage_month
        else:
            resolved_boundary_stage = boundary_stage
        header = _build_header(cortesh, boundary_stage=resolved_boundary_stage)
        cortes = Cortes.from_cortesh(str(cortes_path), cortesh, por_estagio=True)
    else:
        if boundary_stage is None:
            raise ValueError(
                "boundary_stage is required for the consolidated archive "
                f"{cortes_path} (no file-derivable trailer); supply the "
                "DECOMP coupling stage"
            )
        resolved_boundary_stage = boundary_stage
        header = _build_header(cortesh, boundary_stage=resolved_boundary_stage)
        indice_ultimo_corte = _resolve_global_chain_head(cortesh, boundary_stage)
        cortes = Cortes.from_cortesh(
            str(cortes_path),
            cortesh,
            indice_ultimo_corte=indice_ultimo_corte,
            por_estagio=False,
        )

    cortes_df = cortes.cortes
    if cortes_df is None:
        raise ValueError(f"{cortes_path} produced no cut coefficient section")

    records = _build_records(cortes_df, header, cortesh)
    return BoundaryCuts(
        header=header,
        boundary_stage=resolved_boundary_stage,
        records=records,
    )
