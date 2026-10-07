"""Read the special-constraint register families into joined, classified records.

The source model expresses special constraints as four register *triples* — a
declaration register (which stages the constraint spans), a per-stage limit
register, and a coefficient/participation register naming the entities:

    family  decl  limit        coeff     variable(s)                    limit axis
    ----    ----  -----        -----     -----------                    ----------
    RE      RE    LU (1..5)    FU/FT/FI  hydro/thermal gen, interchange per-block
    HQ      HQ    LQ (1..5)    CQ        flow, per-term CQ.tipo         per-block
    HV      HV    LV           CV        hydro storage (hm3)            per-stage
    HE      HE    (inline)     CM        energy over a REE cascade      per-stage

RE spreads its participation over up to three coefficient registers — ``FU``
(hydro generation), ``FT`` (thermal generation), ``FI`` (interchange between two
submarkets) — any combination of which may be present; a fourth, ``FE``, is not
readable via idecomp. Participation (FU/FT/FI, CQ, CV) is declared at a base
stage and inherited forward across the constraint's
``[estagio_inicial, estagio_final]`` range; the limit registers (LU/LQ/LV) are
likewise sparse per stage and inherited forward — the same sparse-stage
convention the ``CT`` reader uses.

Each *term* carries its own variable: an ``HQ`` constraint may mix flow variables
(e.g. ``QDEF`` on one plant and ``QDES`` on another), so the variable is a
per-term property, not a per-constraint one. A constraint therefore lowers to an
entity **bound** only when it is a *single term*, its coefficient is ``±1``, and
its variable has a cobre bounds axis — this covers hydro generation (``FU``),
thermal generation (``FT``), pumping flow (``QBOM``), and the ``HQ`` flow
family (``QDEF``/``QTUR``/``QDES``/``QVER``); every other shape (multiple
terms, a non-unit coefficient, or the ``HE`` energy sum) becomes a **generic
constraint**.

This module only reads, joins, and classifies. The lowering itself lives in the
bounds emitters and the generic-constraints emitter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from cobre_bridge.core.diagnostics import Diagnostic, Severity

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from idecomp.decomp import Dadger

_LOG = logging.getLogger(__name__)

#: Shared diagnostic category for every finding this module emits.
_CATEGORY = "Special constraints"

#: The most block slots the source model's per-block limit rows carry
#: (``limite_{inferior,superior}_1`` … ``_5``).
_MAX_BLOCK_SLOTS = 5

#: Variables that have a cobre entity-bounds axis — a single-term constraint on
#: one of these lowers to a plant bound (`constraints/bounds.rs`). This is a
#: family-agnostic *membership* set: the only consumer is `lowers_to_bound`'s
#: `variable in _BOUNDS_AXIS` test, so the values are never read (do not
#: convert this to a family-keyed structure — nothing consumes it).
_BOUNDS_AXIS: dict[str, str] = {
    "generation": "generation",  # RE FU (hydro)   -> min/max_generation_mw
    "thermal_generation": "generation",  # RE FT (thermal) -> min/max_generation_mw
    "QDEF": "outflow",  # HQ defluencia   -> min/max_outflow_m3s
    "QTUR": "turbined",  # HQ turbinado    -> min/max_turbined_m3s
    "QDES": "diversion",  # HQ desvio       -> min/max_diversion_m3s
    "QVER": "spillage",  # HQ vertimento   -> min/max_spillage_m3s
    "QBOM": "flow",  # HQ pumped flow  -> pumping_bounds min/max_m3s
    "VARM": "storage",  # HV volume       -> min/max_storage_hm3
}


@dataclass(frozen=True)
class ConstraintTerm:
    """One ``(entity, variable)`` participation in a constraint.

    ``code`` is the plant code (``codigo_usina``) for RE/HQ/HV or the REE code
    (``codigo_ree``) for HE. ``variable`` is what the term bounds — ``generation``
    (RE hydro, ``FU``), ``thermal_generation`` (RE thermal, ``FT``),
    ``interchange`` (RE submarket pair, ``FI``), the ``CQ.tipo`` flow
    (``QDEF``/``QTUR``/``QDES``/``QBOM``) for HQ, ``VARM`` for HV, or ``energy``
    for HE. ``frequency`` is the 50/60 Hz tag carried only by RE ``FU`` terms
    (``None`` elsewhere); a plant split across both frequencies contributes two
    distinct terms. ``submarket`` is the ``FT`` thermal's submarket code
    (``codigo_submercado``). ``submarket_de``/``submarket_para`` are the ``FI``
    interchange pair's submarket mnemonics — an interchange term has no plant
    code, so ``code`` carries the documented sentinel ``0`` for it; resolving
    the pair to a cobre ``line_id`` is a later stage's job, not this reader's.
    """

    code: int
    coefficient: float
    variable: str
    frequency: float | None = None
    submarket: int | None = None
    submarket_de: str | None = None
    submarket_para: str | None = None


@dataclass(frozen=True)
class StageBounds:
    """Per-slot lower/upper limits at one stage.

    One slot per block for the per-block families (RE/HQ, up to
    :data:`_MAX_BLOCK_SLOTS`), a single slot for the stage-level families
    (HV/HE). ``None`` marks an absent bound (a one-sided limit).
    """

    lower: tuple[float | None, ...]
    upper: tuple[float | None, ...]


@dataclass(frozen=True)
class HeMeta:
    """The ``HE`` columns beyond the per-stage limit itself.

    ``forma_calculo_produtibilidades``/``tipo_valores_produtibilidades``/
    ``arquivo_produtibilidades`` describe how the REE cascade's productivity
    feeds the energy sum; ``valor_penalidade``/``tipo_penalidade`` are the
    constraint's soft-penalty. This reader only captures them raw — E5
    interprets them (percentage-vs-absolute RHS, energy factor, penalty).
    """

    forma_calculo_produtibilidades: int | None = None
    tipo_valores_produtibilidades: int | None = None
    arquivo_produtibilidades: str | None = None
    valor_penalidade: float | None = None
    tipo_penalidade: int | None = None


@dataclass(frozen=True)
class ConstraintRecord:
    """One special constraint, joined across its register triple and made dense
    over its stage range."""

    family: str  # "RE" | "HQ" | "HV" | "HE"
    constraint_id: int
    stage_start: int  # 0-based, inclusive
    stage_end: int  # 0-based, inclusive
    terms: tuple[ConstraintTerm, ...]
    bounds: Mapping[int, StageBounds]  # 0-based stage index -> bounds, dense
    per_block: bool
    #: HE unit flag on ``limite``: 1 = absolute MWmes, 2 = % of EarmMax. Never a
    #: side selector — ``limite`` is always the lower side regardless of it.
    tipo_limite: int | None = None
    he_meta: HeMeta | None = None  # HE productivity/penalty columns

    @property
    def n_entities(self) -> int:
        """Distinct participating entities (plants, or REEs for HE).

        ⚠️ Counts over the bare ``code`` only, so it is **unreliable for RE**: a
        hydro ``FU`` and a thermal ``FT`` term share a 1-based ``codigo_usina``
        namespace (both could be code 5 → counted as one), and every interchange
        ``FI`` term carries the ``code = 0`` sentinel (two distinct pairs →
        counted as one). Lowering never uses this — it keys on ``is_single_term``
        — and any later consumer that needs a true RE entity count must
        discriminate on ``(variable, code, submarket_de, submarket_para)``.
        """
        return len({t.code for t in self.terms})

    @property
    def is_single_entity(self) -> bool:
        """True when exactly one entity participates (may still be multi-term).

        Inherits the :meth:`n_entities` RE caveat — do not use it to gate RE
        lowering; use ``is_single_term``.
        """
        return self.n_entities == 1

    @property
    def is_single_term(self) -> bool:
        """True when the constraint has exactly one ``(entity, variable)`` term."""
        return len(self.terms) == 1


def lowers_to_bound(record: ConstraintRecord) -> bool:
    """A constraint lowers to an entity bound iff it is a single ``(entity,
    variable)`` term, its coefficient is ``±1``, and the variable has a cobre
    bounds axis; otherwise it is a generic constraint.

    A non-unit coefficient (e.g. ``0.5·QDEF``) is not a face-value bound — it
    must reach the generic emitter, which honours the coefficient. The ``HE``
    energy sum has no bound column, so it always stays generic. Pumping
    (``QBOM``, M2), diversion (``QDES``, M3), and spillage (``QVER``, M5)
    each have one and lower to their respective entity axis as a single unit
    term.
    """
    return (
        record.is_single_term
        and abs(record.terms[0].coefficient) == 1.0
        and record.terms[0].variable in _BOUNDS_AXIS
    )


@dataclass(frozen=True)
class _FamilyConfig:
    """Static description of a declaration/limit/coefficient register triple."""

    family: str
    decl_attr: str  # "re" / "hq" / "hv"
    limit_attr: str  # "lu" / "lq" / "lv"
    coeff_attr: str  # "fu" / "cq" / "cv"
    per_block: bool  # True for RE/HQ, False for HV
    #: Fixed per-term variable (RE = "generation"), or ``None`` to read it from
    #: the coefficient register's ``tipo`` column (HQ/HV).
    fixed_variable: str | None
    has_frequency: bool  # RE carries a 50/60 Hz tag on FU
    #: Override for reading participation terms, given ``dadger`` and returning
    #: ``{constraint_id: [terms]}``. ``None`` uses the generic single-register
    #: path (:func:`_read_terms` against ``coeff_attr``); RE sets this to
    #: :func:`_read_re_terms` to merge its three coefficient registers.
    term_reader: Callable[[Dadger], dict[int, list[ConstraintTerm]]] | None = None


def _df(dadger: Dadger, attr: str) -> pd.DataFrame | None:
    """Read one register as a DataFrame, treating an absent register as empty."""
    frame = getattr(dadger, attr)(df=True)
    if frame is None or getattr(frame, "empty", True):
        return None
    return frame


def _opt_float(value: object) -> float | None:
    """A cell as ``float``, or ``None`` when blank (NaN)."""
    return None if pd.isna(value) else float(value)  # type: ignore[arg-type]


def _opt_int(value: object) -> int | None:
    """A cell as ``int``, or ``None`` when blank (NaN)."""
    return None if pd.isna(value) else int(value)  # type: ignore[call-overload]


def _opt_str(value: object) -> str | None:
    """A cell as ``str``, or ``None`` when blank (NaN)."""
    return None if pd.isna(value) else str(value)


def _read_stage_bounds(row: pd.Series, per_block: bool) -> StageBounds:
    """One limit row -> its per-slot lower/upper bounds.

    Per-block rows expose ``limite_{inferior,superior}_1`` … ``_N``; the
    stage-level rows expose bare ``limite_inferior`` / ``limite_superior``.
    """
    if not per_block:
        return StageBounds(
            lower=(_opt_float(row["limite_inferior"]),),
            upper=(_opt_float(row["limite_superior"]),),
        )
    lower: list[float | None] = []
    upper: list[float | None] = []
    for slot in range(1, _MAX_BLOCK_SLOTS + 1):
        lo_col, up_col = f"limite_inferior_{slot}", f"limite_superior_{slot}"
        if lo_col not in row.index:
            break
        lower.append(_opt_float(row[lo_col]))
        upper.append(_opt_float(row[up_col]))
    return StageBounds(lower=tuple(lower), upper=tuple(upper))


def _forward_fill_bounds(
    declared: Mapping[int, StageBounds],
    stage_start: int,
    stage_end: int,
) -> dict[int, StageBounds]:
    """Densify sparse per-stage limits over ``[stage_start, stage_end]``.

    A stage inherits the most recently declared limit at or before it; stages
    before the first declaration inherit that first declaration (mirroring the
    ``CT`` sparse-stage rule). Returns ``{}`` when nothing is declared.
    """
    if not declared:
        return {}
    first = declared[min(declared)]
    dense: dict[int, StageBounds] = {}
    current = first
    for stage in range(stage_start, stage_end + 1):
        if stage in declared:
            current = declared[stage]
        dense[stage] = current
    return dense


def _read_terms(
    coeff: pd.DataFrame,
    config: _FamilyConfig,
) -> dict[int, list[ConstraintTerm]]:
    """Group a coefficient register into ``{constraint_id: [terms]}``.

    Each ``(entity, variable, frequency)`` identity contributes one term; a
    repeated identity (declared at several stages) collapses to its first
    coefficient — the source coefficients are pure ``±1``/``1`` signs, constant
    across stages.
    """
    terms: dict[int, list[ConstraintTerm]] = {}
    seen: dict[int, set[tuple[int, str, float | None]]] = {}
    for _, row in coeff.iterrows():
        cid = int(row["codigo_restricao"])
        code = int(row["codigo_usina"])
        variable = (
            config.fixed_variable
            if config.fixed_variable is not None
            else str(row["tipo"]).strip()
        )
        frequency = (
            _opt_float(row["frequencia"])
            if config.has_frequency and "frequencia" in row.index
            else None
        )
        identity = (code, variable, frequency)
        if identity in seen.setdefault(cid, set()):
            continue
        seen[cid].add(identity)
        terms.setdefault(cid, []).append(
            ConstraintTerm(
                code=code,
                coefficient=float(row["coeficiente"]),
                variable=variable,
                frequency=frequency,
            )
        )
    return terms


def _read_re_terms(dadger: Dadger) -> dict[int, list[ConstraintTerm]]:
    """Group RE's three participation registers into ``{constraint_id: [terms]}``.

    Unlike HQ/HV, an RE's participation is not confined to one coefficient
    register: it may carry ``FU`` (hydro generation), ``FT`` (thermal
    generation), ``FI`` (submarket interchange), or any combination of the
    three, so all three are read and merged by ``codigo_restricao``. A
    missing/empty register contributes no terms; an RE named in at least one of
    the three has terms. Each ``(entity, variable, frequency, submarket,
    submarket_de, submarket_para)`` identity contributes one term — the same
    dedup-by-identity rule :func:`_read_terms` applies to a single register.
    """
    terms: dict[int, list[ConstraintTerm]] = {}
    seen: dict[
        int, set[tuple[int, str, float | None, int | None, str | None, str | None]]
    ] = {}

    def _add(cid: int, term: ConstraintTerm) -> None:
        identity = (
            term.code,
            term.variable,
            term.frequency,
            term.submarket,
            term.submarket_de,
            term.submarket_para,
        )
        if identity in seen.setdefault(cid, set()):
            return
        seen[cid].add(identity)
        terms.setdefault(cid, []).append(term)

    fu = _df(dadger, "fu")
    if fu is not None:
        for _, row in fu.iterrows():
            frequency = (
                _opt_float(row["frequencia"]) if "frequencia" in row.index else None
            )
            _add(
                int(row["codigo_restricao"]),
                ConstraintTerm(
                    code=int(row["codigo_usina"]),
                    coefficient=float(row["coeficiente"]),
                    variable="generation",
                    frequency=frequency,
                ),
            )

    ft = _df(dadger, "ft")
    if ft is not None:
        for _, row in ft.iterrows():
            _add(
                int(row["codigo_restricao"]),
                ConstraintTerm(
                    code=int(row["codigo_usina"]),
                    coefficient=float(row["coeficiente"]),
                    variable="thermal_generation",
                    submarket=int(row["codigo_submercado"]),
                ),
            )

    fi = _df(dadger, "fi")
    if fi is not None:
        for _, row in fi.iterrows():
            _add(
                int(row["codigo_restricao"]),
                ConstraintTerm(
                    code=0,  # interchange has no plant code — documented sentinel
                    coefficient=float(row["coeficiente"]),
                    variable="interchange",
                    submarket_de=str(row["codigo_submercado_de"]),
                    submarket_para=str(row["codigo_submercado_para"]),
                ),
            )

    return terms


_TRIPLE_FAMILIES: tuple[_FamilyConfig, ...] = (
    _FamilyConfig(
        "RE", "re", "lu", "fu", True, "generation", True, term_reader=_read_re_terms
    ),
    _FamilyConfig("HQ", "hq", "lq", "cq", True, None, False),
    _FamilyConfig("HV", "hv", "lv", "cv", False, None, False),
)


def _read_triple_family(
    dadger: Dadger,
    config: _FamilyConfig,
) -> list[ConstraintRecord]:
    """Join one RE/HQ/HV register triple into classified constraint records."""
    decl = _df(dadger, config.decl_attr)
    if decl is None:
        return []
    if config.term_reader is not None:
        terms_by_cid = config.term_reader(dadger)
    else:
        coeff = _df(dadger, config.coeff_attr)
        terms_by_cid = _read_terms(coeff, config) if coeff is not None else {}
    if not terms_by_cid:
        return []
    limit = _df(dadger, config.limit_attr)

    limits_by_cid: dict[int, dict[int, StageBounds]] = {}
    if limit is not None:
        for _, row in limit.iterrows():
            cid = int(row["codigo_restricao"])
            stage0 = int(row["estagio"]) - 1
            limits_by_cid.setdefault(cid, {})[stage0] = _read_stage_bounds(
                row, config.per_block
            )

    records: list[ConstraintRecord] = []
    skipped_no_terms = 0
    for _, row in decl.iterrows():
        cid = int(row["codigo_restricao"])
        terms = terms_by_cid.get(cid)
        if not terms:
            skipped_no_terms += 1
            continue
        stage_start = int(row["estagio_inicial"]) - 1
        stage_end = int(row["estagio_final"]) - 1
        bounds = _forward_fill_bounds(
            limits_by_cid.get(cid, {}), stage_start, stage_end
        )
        records.append(
            ConstraintRecord(
                family=config.family,
                constraint_id=cid,
                stage_start=stage_start,
                stage_end=stage_end,
                terms=tuple(terms),
                bounds=bounds,
                per_block=config.per_block,
            )
        )
    if skipped_no_terms:
        _LOG.info(
            "%s: %d declared constraint(s) carry no participation rows and were "
            "skipped (inactive / defined elsewhere)",
            config.family,
            skipped_no_terms,
        )
    return records


def _read_energy_constraints(dadger: Dadger) -> list[ConstraintRecord]:
    """Join the ``HE``/``CM`` energy family (VminOP-style, over a REE cascade).

    ``HE`` rows carry the per-stage ``limite`` inline — **always** a lower limit
    on the REE's stored energy, regardless of ``tipo_limite`` (a unit flag, not a
    side selector) — plus the five productivity/penalty columns captured onto
    :class:`HeMeta`; ``CM`` carries the per-REE productivity coefficients. Every
    HE is a generic constraint — the ``energy`` variable expands to
    ``Σ ρ_acum · storage`` over the REE's plants downstream (E5, RHE energy).
    """
    he = _df(dadger, "he")
    cm = _df(dadger, "cm")
    if he is None or cm is None:
        return []

    terms_by_cid: dict[int, list[ConstraintTerm]] = {}
    for _, row in cm.iterrows():
        cid = int(row["codigo_restricao"])
        terms_by_cid.setdefault(cid, []).append(
            ConstraintTerm(
                code=int(row["codigo_ree"]),
                coefficient=float(row["coeficiente"]),
                variable="energy",
            )
        )

    per_stage: dict[int, dict[int, StageBounds]] = {}
    meta: dict[int, tuple[int | None, HeMeta]] = {}
    for _, row in he.iterrows():
        cid = int(row["codigo_restricao"])
        stage0 = int(row["estagio"]) - 1
        per_stage.setdefault(cid, {})[stage0] = StageBounds(
            lower=(_opt_float(row["limite"]),), upper=(None,)
        )
        meta[cid] = (
            _opt_int(row["tipo_limite"]),
            HeMeta(
                forma_calculo_produtibilidades=_opt_int(
                    row.get("forma_calculo_produtibilidades")
                ),
                tipo_valores_produtibilidades=_opt_int(
                    row.get("tipo_valores_produtibilidades")
                ),
                arquivo_produtibilidades=_opt_str(row.get("arquivo_produtibilidades")),
                valor_penalidade=_opt_float(row.get("valor_penalidade")),
                tipo_penalidade=_opt_int(row.get("tipo_penalidade")),
            ),
        )

    records: list[ConstraintRecord] = []
    skipped_no_limits = 0  # CM productivity terms whose cid has no HE limit rows
    for cid, terms in terms_by_cid.items():
        stages = per_stage.get(cid)
        if not stages:
            skipped_no_limits += 1
            continue
        stage_start, stage_end = min(stages), max(stages)
        bounds = _forward_fill_bounds(stages, stage_start, stage_end)
        tipo_limite, he_meta = meta[cid]
        records.append(
            ConstraintRecord(
                family="HE",
                constraint_id=cid,
                stage_start=stage_start,
                stage_end=stage_end,
                terms=tuple(terms),
                bounds=bounds,
                per_block=False,
                tipo_limite=tipo_limite,
                he_meta=he_meta,
            )
        )
    # HE limit rows whose cid has no CM productivity term never become a record;
    # surface both halves of the join so a dropped energy constraint is loud, not
    # silent (mirroring ``_read_triple_family``'s ``skipped_no_terms``).
    skipped_no_terms = sum(1 for cid in per_stage if cid not in terms_by_cid)
    if skipped_no_limits or skipped_no_terms:
        _LOG.info(
            "HE: %d constraint(s) with a productivity (CM) term but no limit (HE) "
            "rows, and %d with limit rows but no CM productivity term, were skipped "
            "(incomplete join / defined elsewhere)",
            skipped_no_limits,
            skipped_no_terms,
        )
    return records


@dataclass(frozen=True)
class ConstraintCensus:
    """The four families read and split by their lowering target."""

    #: Every record, keyed by family ("RE"/"HQ"/"HV"/"HE").
    by_family: Mapping[str, Sequence[ConstraintRecord]]
    #: Single-term records on a bounded variable — lower to entity bounds.
    to_bounds: Sequence[ConstraintRecord] = field(default_factory=tuple)
    #: Everything else — lower to generic constraints.
    to_generic: Sequence[ConstraintRecord] = field(default_factory=tuple)


def read_constraints(dadger: Dadger) -> ConstraintCensus:
    """Read all four special-constraint families and split them by lowering target.

    Returns a :class:`ConstraintCensus`: every record grouped by family, plus the
    two lowering buckets — ``to_bounds`` (single-term on a bounded variable) and
    ``to_generic`` (multi-term, unbounded-variable, or ``HE``).
    """
    by_family: dict[str, list[ConstraintRecord]] = {}
    for config in _TRIPLE_FAMILIES:
        by_family[config.family] = _read_triple_family(dadger, config)
    by_family["HE"] = _read_energy_constraints(dadger)

    to_bounds: list[ConstraintRecord] = []
    to_generic: list[ConstraintRecord] = []
    for records in by_family.values():
        for record in records:
            (to_bounds if lowers_to_bound(record) else to_generic).append(record)

    _LOG.info(
        "special constraints read: %s; %d -> bounds, %d -> generic",
        {fam: len(recs) for fam, recs in by_family.items()},
        len(to_bounds),
        len(to_generic),
    )
    return ConstraintCensus(
        by_family={fam: tuple(recs) for fam, recs in by_family.items()},
        to_bounds=tuple(to_bounds),
        to_generic=tuple(to_generic),
    )


#: The ``FE`` register (electrical participation) has no idecomp accessor —
#: presence-only detection, never parsed into a term or a constraint id.
_MNEMONIC_FE = "FE"

#: The whole ``RHA`` family (``HA``/``LA``/``CA``) has no idecomp accessor.
_RHA_MNEMONICS = frozenset({"HA", "LA", "CA"})

#: The ``indices.csv`` entry a LIBs-era deck uses to point at its richer
#: electrical-constraint file (``lib_restricao-eletrica-especial.csv``), E9
#: (blocked).
_LIBS_ELECTRICAL_TOKEN = "RESTRICAO-ELETRICA-ESPECIAL"


def detect_unreadable_electrical(dadger_path: Path) -> list[Diagnostic]:
    """Detect ``FE``/``RHA`` register lines the reader cannot ingest and warn
    loudly instead of silently under-reading them.

    idecomp 1.13.0 exposes no accessor for ``FE`` electrical-participation
    terms or for the whole ``RHA`` family (``HA``/``LA``/``CA``). An ``FE``
    term on a constraint, if left unaccounted for, makes that constraint's
    electrical model *wrong*, not merely incomplete — so a deck carrying one
    must be flagged rather than silently treated as fully read. This scans the
    deck's raw lines for the **leading** register mnemonic only (a mnemonic
    appearing mid-line, inside another field, never counts) and returns at
    most two ``WARNING`` diagnostics: one for ``FE`` presence, one for ``RHA``
    presence. It does not parse a constraint id out of either family and does
    not drop any ``RE`` record — the per-id ``FE`` skip this would require is
    deferred (E8).

    Parameters
    ----------
    dadger_path:
        Path to the deck's ``dadger`` register file.

    Returns
    -------
    Zero, one, or two diagnostics; empty when the deck carries neither
    ``FE`` nor ``RHA`` lines.
    """
    has_fe = False
    has_rha = False
    for line in dadger_path.read_text(encoding="latin-1").splitlines():
        tokens = line.strip().split(maxsplit=1)
        if not tokens:
            continue
        mnemonic = tokens[0]
        if mnemonic == _MNEMONIC_FE:
            has_fe = True
        elif mnemonic in _RHA_MNEMONICS:
            has_rha = True

    diagnostics: list[Diagnostic] = []
    if has_fe:
        diagnostics.append(
            Diagnostic(
                code="decomp-fe-participation-unreadable",
                severity=Severity.WARNING,
                category=_CATEGORY,
                title="FE electrical participation present but unreadable",
                summary=(
                    "The deck declares FE electrical-participation terms, "
                    "which have no reader accessor; the electrical "
                    "constraints (RE) on this deck may be under-read, so the "
                    "electrical model is not authoritative for this deck."
                ),
                remediation=(
                    "FE term reading and the per-constraint skip it requires "
                    "are deferred; no action needed to convert."
                ),
            )
        )
    if has_rha:
        diagnostics.append(
            Diagnostic(
                code="decomp-rha-family-unconverted",
                severity=Severity.WARNING,
                category=_CATEGORY,
                title="RHA family (HA/LA/CA) not converted",
                summary=(
                    "The deck declares HA/LA/CA (RHA family) register lines; "
                    "this conversion does not read or convert them."
                ),
                remediation=(
                    "RHA conversion is deferred; no action needed to convert."
                ),
            )
        )
    return diagnostics


def detect_libs_electrical(deck_dir: Path) -> Diagnostic | None:
    """Warn when a LIBs-era deck moves its electrical constraints out of
    ``dadger``.

    A deck using the LIBs electrical-constraint family declares it in
    ``indices.csv`` under a ``RESTRICAO-ELETRICA-ESPECIAL`` entry; that
    richer format is unconverted only for its short-form ``RE``/``RE-*`` and
    date-indexed ``-HORIZONTE-DATA``/``-FORMULA-DATA-PATAMAR`` variants (the
    period-keyed long-form cards ARE converted — see
    :mod:`cobre_bridge.decomp.converters.libs_electrical`). This reads
    ``indices.csv`` (when present) and reports the entry's presence; it does
    not parse or convert the LIBs file itself, and does not know whether the
    long-form subset converted — the caller (:func:`~cobre_bridge.decomp.
    pipeline.convert_decomp_case`) is the one that reads
    :func:`~cobre_bridge.decomp.converters.libs_electrical.read_libs_electrical` and
    decides whether this diagnostic's return value is still worth emitting:
    it suppresses it once the long-form subset converts (the census INFO
    diagnostic, ``decomp-libs-electrical-converted``, is authoritative
    then), and still emits it for a short-form-only entry. This function
    itself is unchanged either way — always returning the same diagnostic
    for the same *deck_dir*.

    Parameters
    ----------
    deck_dir:
        The deck's source directory (holding ``indices.csv`` alongside
        ``caso.dat``).

    Returns
    -------
    A ``WARNING`` diagnostic when ``indices.csv`` lists a
    ``RESTRICAO-ELETRICA-ESPECIAL`` entry (case-insensitive), else ``None`` —
    including when ``indices.csv`` itself is absent.
    """
    indices_path = deck_dir / "indices.csv"
    if not indices_path.is_file():
        return None
    text = indices_path.read_text(encoding="latin-1")
    if not any(_LIBS_ELECTRICAL_TOKEN in line.upper() for line in text.splitlines()):
        return None
    return Diagnostic(
        code="decomp-libs-electrical-present",
        severity=Severity.WARNING,
        category=_CATEGORY,
        title="LIBs electrical constraints present",
        summary=(
            "The deck's indices.csv lists a RESTRICAO-ELETRICA-ESPECIAL "
            "entry (the LIBs-era electrical-constraint family); this richer "
            "format is not converted, so the electrical model on this deck "
            "is not authoritative."
        ),
        remediation=(
            "LIBs electrical-constraint conversion is deferred; no action "
            "needed to convert."
        ),
    )


def resolve_libs_electrical_path(deck_dir: Path) -> Path | None:
    """Resolve the deck's LIBs electrical-constraint file path.

    Reads *deck_dir*'s ``indices.csv`` for the ``RESTRICAO-ELETRICA-ESPECIAL``
    entry's own ``Caminho`` (the third ``;``-delimited field), matched by its
    first field **exactly** — never the ``-ATIVACAO``/``-VIOLACAO`` sibling
    entries :func:`detect_libs_electrical`'s substring scan also matches —
    and resolved relative to *deck_dir*. Falls back to a
    ``lib_restricao-eletrica-especial*.csv`` glob in *deck_dir* when
    ``indices.csv`` is absent, carries no such entry, or the entry's own path
    does not resolve to an existing file — mirroring
    :func:`~cobre_bridge.decomp.files.discover_decomp_files`'s own
    ``find`` fallback idiom for the deck's other optional files.

    Parameters
    ----------
    deck_dir:
        The deck's source directory (holding ``indices.csv`` alongside
        ``caso.dat``).

    Returns
    -------
    The resolved file path, or ``None`` when neither resolves.
    """
    indices_path = deck_dir / "indices.csv"
    if indices_path.is_file():
        for line in indices_path.read_text(encoding="latin-1").splitlines():
            fields = line.split(";")
            if len(fields) < 3 or fields[0].strip().upper() != _LIBS_ELECTRICAL_TOKEN:
                continue
            filename = fields[2].strip()
            if not filename:
                continue
            candidate = deck_dir / filename
            if candidate.is_file():
                return candidate
    matches = sorted(deck_dir.glob("lib_restricao-eletrica-especial*.csv"))
    return matches[0] if matches else None
