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
readable via idecomp. Each participation row (FU/FT/FI, CQ, CV) declares a
term's coefficient from its stage onward, across the constraint's
``[estagio_inicial, estagio_final]`` range, so a coefficient may change from
stage to stage; a term first declared after the constraint's first stage is
absent before it. The limit registers (LU/LQ/LV) are likewise sparse per stage
and inherited forward — the same sparse-stage convention the ``CT`` reader
uses.

Each *term* carries its own variable: an ``HQ`` constraint may mix flow variables
(e.g. ``QDEF`` on one plant and ``QDES`` on another), so the variable is a
per-term property, not a per-constraint one. A constraint therefore lowers to an
entity **bound** only when it is a *single term*, its coefficient is ``±1`` at
every stage, and its variable has a novomodelo bounds axis — this covers hydro
generation (``FU``), thermal generation (``FT``), pumping flow (``QBOM``), and
the ``HQ`` flow family (``QDEF``/``QTUR``/``QDES``/``QVER``); every other shape
(multiple terms, a non-unit coefficient, or the ``HE`` energy sum) becomes a
**generic constraint**. So does a single-term ``HQ`` whose limits leave no
common value with another's on the same plant variable: a plant bound cannot
hold both, and each restriction keeps its own slack.

This module only reads, joins, and classifies. The lowering itself lives in the
bounds emitters and the generic-constraints emitter.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import TYPE_CHECKING

import pandas as pd

from novomodelo_bridge.core.diagnostics import Diagnostic, DiagnosticTable, Severity
from novomodelo_bridge.core.errors import FieldParseError
from novomodelo_bridge.core.generic_constraint_builder import is_bounded
from novomodelo_bridge.core.tolerances import relative_tolerance

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

#: Variables that have a novomodelo entity-bounds axis — a single-term constraint on
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
    the pair to a novomodelo ``line_id`` is a later stage's job, not this reader's.

    ``schedule`` is empty when ``coefficient`` holds at every stage of the
    constraint. Otherwise it lists ascending ``(stage_index, coefficient)``
    pairs (0-based), each holding until the next; the term is absent
    (coefficient ``0``) before the first pair, and ``coefficient`` is the
    first pair's value. Read a stage's coefficient through
    :meth:`coefficient_at`.
    """

    code: int
    coefficient: float
    variable: str
    frequency: float | None = None
    submarket: int | None = None
    submarket_de: str | None = None
    submarket_para: str | None = None
    schedule: tuple[tuple[int, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.schedule:
            return
        stages = [stage for stage, _ in self.schedule]
        if stages != sorted(set(stages)):
            raise ValueError(f"schedule stages must be strictly ascending: {stages}")
        if self.coefficient != self.schedule[0][1]:
            raise ValueError(
                f"coefficient {self.coefficient} differs from the first "
                f"scheduled coefficient {self.schedule[0][1]}"
            )

    def coefficient_at(self, stage: int) -> float:
        """The coefficient at 0-based *stage*."""
        if not self.schedule:
            return self.coefficient
        value = 0.0
        for start, coefficient in self.schedule:
            if start > stage:
                break
            value = coefficient
        return value


#: ``{constraint_id: [(term, {stage_index: coefficient})]}`` — every distinct
#: participation of a constraint with the coefficient each of its rows declares.
_DeclaredTerms = dict[int, list[tuple[ConstraintTerm, dict[int, float]]]]


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
    constraint's soft-penalty. This reader only captures them raw; the RHE
    emitter interprets them (percentage-vs-absolute RHS, energy factor,
    penalty).
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
    variable)`` term, its coefficient is ``±1`` at every stage, and the
    variable has a novomodelo bounds axis; otherwise it is a generic constraint.

    A non-unit coefficient (e.g. ``0.5·QDEF``) is not a face-value bound — it
    must reach the generic emitter, which honours the coefficient. The ``HE``
    energy sum has no bound column, so it always stays generic. Pumping
    (``QBOM``), diversion (``QDES``), and spillage (``QVER``) each have one
    and lower to their respective entity axis as a single unit term.
    """
    if not record.is_single_term:
        return False
    term = record.terms[0]
    return term.variable in _BOUNDS_AXIS and all(
        abs(term.coefficient_at(stage)) == 1.0
        for stage in range(record.stage_start, record.stage_end + 1)
    )


def sided_bounds(
    lower: float | None, upper: float | None, coefficient: float
) -> tuple[float | None, float | None]:
    """The coefficient-sign -> side map for a single term ``c*x in [lower, upper]``.

    ``coefficient > 0`` (``+1``) leaves the entity bound as-is:
    ``x in [lower, upper]``. ``coefficient < 0`` (``-1``) flips and swaps the
    sides: ``-x in [lower, upper] => x in [-upper, -lower]``. Only ``+-1``
    ever reaches here (``lowers_to_bound`` gates ``abs(coefficient) == 1`` at
    every stage), so the two branches are exhaustive.

    The ``+-1e21`` unbounded sentinel is not re-derived here — a bound is
    passed through unchanged (negating it keeps its magnitude at the
    sentinel) and its consumers drop it through ``is_bounded``. Only an
    actual ``None`` is treated as "no bound on that side".
    """
    if coefficient > 0:
        return lower, upper
    return (None if upper is None else -upper, None if lower is None else -lower)


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
    #: its declared terms. ``None`` uses the generic single-register
    #: path (:func:`_read_terms` against ``coeff_attr``); RE sets this to
    #: :func:`_read_re_terms` to merge its three coefficient registers.
    term_reader: Callable[[Dadger], _DeclaredTerms] | None = None


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


class _TermCollector:
    """Group participation rows into one term per ``(constraint, identity)``.

    The identity is every term field except the coefficient, so a term
    declared at several stages is one term with several declarations. A
    repeated row at the same stage keeps its first coefficient.
    """

    def __init__(self) -> None:
        self._terms: dict[
            int, dict[ConstraintTerm, tuple[ConstraintTerm, dict[int, float]]]
        ] = {}

    def add(self, cid: int, stage: int, term: ConstraintTerm) -> None:
        identity = replace(term, coefficient=0.0)
        _, declared = self._terms.setdefault(cid, {}).setdefault(identity, (term, {}))
        declared.setdefault(stage, term.coefficient)

    def result(self) -> _DeclaredTerms:
        return {cid: list(terms.values()) for cid, terms in self._terms.items()}


def _row_stage(row: pd.Series) -> int:
    """A coefficient row's 0-based stage (the register's ``estagio`` is 1-based)."""
    return int(row["estagio"]) - 1


def _read_terms(coeff: pd.DataFrame, config: _FamilyConfig) -> _DeclaredTerms:
    """Group a coefficient register into its declared terms per constraint."""
    collector = _TermCollector()
    for _, row in coeff.iterrows():
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
        collector.add(
            int(row["codigo_restricao"]),
            _row_stage(row),
            ConstraintTerm(
                code=int(row["codigo_usina"]),
                coefficient=float(row["coeficiente"]),
                variable=variable,
                frequency=frequency,
            ),
        )
    return collector.result()


def _read_re_terms(dadger: Dadger) -> _DeclaredTerms:
    """Group RE's three participation registers into its declared terms.

    Unlike HQ/HV, an RE's participation is not confined to one coefficient
    register: it may carry ``FU`` (hydro generation), ``FT`` (thermal
    generation), ``FI`` (submarket interchange), or any combination of the
    three, so all three are read and merged by ``codigo_restricao``. A
    missing/empty register contributes no terms.
    """
    collector = _TermCollector()

    fu = _df(dadger, "fu")
    if fu is not None:
        for _, row in fu.iterrows():
            frequency = (
                _opt_float(row["frequencia"]) if "frequencia" in row.index else None
            )
            collector.add(
                int(row["codigo_restricao"]),
                _row_stage(row),
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
            collector.add(
                int(row["codigo_restricao"]),
                _row_stage(row),
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
            collector.add(
                int(row["codigo_restricao"]),
                _row_stage(row),
                ConstraintTerm(
                    code=0,  # interchange has no plant code — documented sentinel
                    coefficient=float(row["coeficiente"]),
                    variable="interchange",
                    submarket_de=str(row["codigo_submercado_de"]),
                    submarket_para=str(row["codigo_submercado_para"]),
                ),
            )

    return collector.result()


def _scheduled_term(
    term: ConstraintTerm,
    declared: Mapping[int, float],
    stage_start: int,
    stage_end: int,
) -> ConstraintTerm | None:
    """*term* with its coefficient schedule over ``[stage_start, stage_end]``.

    A declaration holds until the next one. A term first declared after
    ``stage_start`` is absent before that stage, as the source model reads
    it; a declaration before ``stage_start`` holds from ``stage_start``.
    Returns ``None`` for a term declared only after ``stage_end``.
    """
    schedule: list[tuple[int, float]] = []
    for declared_stage in sorted(declared):
        stage = max(declared_stage, stage_start)
        if stage > stage_end:
            break
        coefficient = declared[declared_stage]
        if schedule and schedule[-1][0] == stage:
            schedule[-1] = (stage, coefficient)
        elif not schedule or schedule[-1][1] != coefficient:
            schedule.append((stage, coefficient))
    if not schedule:
        return None
    first_stage, first_coefficient = schedule[0]
    if len(schedule) == 1 and first_stage == stage_start:
        return replace(term, coefficient=first_coefficient, schedule=())
    return replace(term, coefficient=first_coefficient, schedule=tuple(schedule))


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
    """Join one RE/HQ/HV register triple into classified constraint records.

    Raises
    ------
    FieldParseError
        A constraint has no term at its first stage, a deck the source model
        rejects.
    """
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
        stage_start = int(row["estagio_inicial"]) - 1
        stage_end = int(row["estagio_final"]) - 1
        terms = [
            scheduled
            for term, declared in terms_by_cid.get(cid, [])
            if (scheduled := _scheduled_term(term, declared, stage_start, stage_end))
            is not None
        ]
        if not terms:
            skipped_no_terms += 1
            continue
        if all(term.schedule and term.schedule[0][0] > stage_start for term in terms):
            raise FieldParseError(
                f"{config.family} constraint {cid} has no coefficient at its "
                f"first stage ({stage_start + 1}); the source model rejects "
                "such a deck.",
                field=f"{config.family} constraint {cid}",
            )
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
    ``Σ ρ_acum · storage`` over the REE's plants downstream (RHE energy).
    ``CM`` has no stage column, so every HE term has a constant coefficient.
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


#: Families whose single-term records stay generic when their limits conflict.
#: The source model prices every RHQ limit's violation per unit at one slack
#: penalty and violates the cheaper side, whatever the restrictions' order or
#: numbers; a plant bound cannot hold two disjoint ranges.
_SOFT_CONFLICT_FAMILIES = frozenset({"HQ"})


@dataclass(frozen=True)
class BoundConflict:
    """Single-term records on one plant variable that no value meets at once.

    ``stage_index`` (0-based) and ``block`` (0-based) locate the first slot
    where ``floor``, the highest lower limit among ``constraint_ids``, exceeds
    ``ceiling``, the lowest upper limit, both on the variable itself.
    """

    family: str
    code: int
    variable: str
    constraint_ids: tuple[int, ...]
    stage_index: int
    block: int
    floor: float
    ceiling: float


@dataclass(frozen=True)
class ConstraintCensus:
    """The four families read and split by their lowering target."""

    #: Every record, keyed by family ("RE"/"HQ"/"HV"/"HE").
    by_family: Mapping[str, Sequence[ConstraintRecord]]
    #: Single-term records on a bounded variable — lower to entity bounds.
    to_bounds: Sequence[ConstraintRecord] = field(default_factory=tuple)
    #: Everything else — lower to generic constraints.
    to_generic: Sequence[ConstraintRecord] = field(default_factory=tuple)
    #: Single-term records kept in ``to_generic`` because their limits conflict.
    bound_conflicts: Sequence[BoundConflict] = field(default_factory=tuple)


_Slot = tuple[int, int]


def _entity_slots(
    record: ConstraintRecord,
) -> dict[_Slot, tuple[float | None, float | None]]:
    """``{(stage_index, block): (lower, upper)}`` of a single-term record on its
    variable, unbounded sides as ``None``."""
    term = record.terms[0]
    slots: dict[_Slot, tuple[float | None, float | None]] = {}
    for stage_index, stage_bounds in record.bounds.items():
        coefficient = term.coefficient_at(stage_index)
        for block, (lower, upper) in enumerate(
            zip(stage_bounds.lower, stage_bounds.upper, strict=True)
        ):
            lower, upper = sided_bounds(lower, upper, coefficient)
            lower = lower if is_bounded(lower) else None
            upper = upper if is_bounded(upper) else None
            if lower is not None or upper is not None:
                slots[(stage_index, block)] = (lower, upper)
    return slots


def _floor_above(floor: float | None, ceiling: float | None) -> bool:
    return (
        floor is not None
        and ceiling is not None
        and floor - ceiling > relative_tolerance(ceiling)
    )


def _group_conflict(
    family: str, group: Sequence[ConstraintRecord]
) -> BoundConflict | None:
    """The conflict among *group* (one plant variable), or ``None``.

    Only pairs of distinct records count: a record whose own floor exceeds its
    own ceiling stays a bound and fails at the bound intersection.
    """
    slots = {record.constraint_id: _entity_slots(record) for record in group}
    involved: set[int] = set()
    first: _Slot | None = None
    for a, b in combinations(sorted(slots), 2):
        for slot in slots[a].keys() & slots[b].keys():
            (lower_a, upper_a), (lower_b, upper_b) = slots[a][slot], slots[b][slot]
            if _floor_above(lower_a, upper_b) or _floor_above(lower_b, upper_a):
                involved.update((a, b))
                first = slot if first is None else min(first, slot)
    if first is None:
        return None
    lowers = [slots[cid][first][0] for cid in involved if first in slots[cid]]
    uppers = [slots[cid][first][1] for cid in involved if first in slots[cid]]
    term = group[0].terms[0]
    return BoundConflict(
        family=family,
        code=term.code,
        variable=term.variable,
        constraint_ids=tuple(sorted(involved)),
        stage_index=first[0],
        block=first[1],
        floor=max(value for value in lowers if value is not None),
        ceiling=min(value for value in uppers if value is not None),
    )


def _split_bound_conflicts(
    to_bounds: Sequence[ConstraintRecord],
) -> tuple[list[ConstraintRecord], list[ConstraintRecord], list[BoundConflict]]:
    """``(kept, rerouted, conflicts)``: the records of a conflicting pair move
    from the bounds to the generic constraints, each with its own slack."""
    groups: dict[tuple[str, int, str], list[ConstraintRecord]] = defaultdict(list)
    for record in to_bounds:
        if record.family in _SOFT_CONFLICT_FAMILIES:
            term = record.terms[0]
            groups[(record.family, term.code, term.variable)].append(record)
    conflicts: list[BoundConflict] = []
    for (family, _, _), group in sorted(groups.items()):
        if len(group) > 1:
            conflict = _group_conflict(family, group)
            if conflict is not None:
                conflicts.append(conflict)
    moved = {
        (conflict.family, cid)
        for conflict in conflicts
        for cid in conflict.constraint_ids
    }
    kept = [r for r in to_bounds if (r.family, r.constraint_id) not in moved]
    rerouted = [r for r in to_bounds if (r.family, r.constraint_id) in moved]
    return kept, rerouted, conflicts


def bound_conflicts_diagnostic(
    conflicts: Sequence[BoundConflict], plant_names: Mapping[int, str]
) -> Diagnostic:
    """WARNING listing the conflicting restrictions kept as soft constraints."""
    rows = [
        [
            conflict.code,
            plant_names.get(conflict.code, ""),
            conflict.variable,
            ", ".join(f"{conflict.family} {cid}" for cid in conflict.constraint_ids),
            conflict.stage_index + 1,
            conflict.block + 1,
            conflict.floor,
            conflict.ceiling,
        ]
        for conflict in conflicts
    ]
    return Diagnostic(
        code="decomp-conflicting-flow-restrictions",
        severity=Severity.WARNING,
        category=_CATEGORY,
        title=f"Conflicting flow restrictions kept soft ({len(rows)})",
        summary=(
            f"{len(rows)} plant flow(s) carry single-plant HQ restrictions that "
            "no value meets at once. Each listed restriction is converted as a "
            "generic constraint with its own slack at the common penalty "
            "instead of a plant bound, so novomodelo violates whichever is cheaper, "
            "as the source model does."
        ),
        table=DiagnosticTable(
            columns=[
                "Code",
                "Name",
                "Variable",
                "Restrictions",
                "First stage",
                "Block",
                "Floor (m3/s)",
                "Ceiling (m3/s)",
            ],
            rows=rows,
        ),
        remediation=(
            "Check the listed restrictions in the deck: one of them is violated "
            "in every scenario, and the source model reports which one in its "
            "infeasibility report (inviab)."
        ),
    )


def read_constraints(dadger: Dadger) -> ConstraintCensus:
    """Read all four special-constraint families and split them by lowering target.

    Returns a :class:`ConstraintCensus`: every record grouped by family, plus the
    two lowering buckets — ``to_bounds`` (single-term on a bounded variable) and
    ``to_generic`` (multi-term, unbounded-variable, ``HE``, or a single-term RHQ
    whose limits conflict with another's, listed in ``bound_conflicts``).
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
    to_bounds, rerouted, conflicts = _split_bound_conflicts(to_bounds)
    to_generic.extend(rerouted)

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
        bound_conflicts=tuple(conflicts),
    )


#: The ``FE`` register (electrical participation) has no idecomp accessor —
#: presence-only detection, never parsed into a term or a constraint id.
_MNEMONIC_FE = "FE"

#: The whole ``RHA`` family (``HA``/``LA``/``CA``) has no idecomp accessor.
_RHA_MNEMONICS = frozenset({"HA", "LA", "CA"})


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
    not implemented.

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


def detect_libs_electrical(libs_file: Path | None) -> Diagnostic | None:
    """Warn when a LIBs-era deck moves its electrical constraints out of
    ``dadger``.

    *libs_file* is the deck's ``RESTRICAO-ELETRICA-ESPECIAL`` file
    (``DecompFiles.libs_restricao_eletrica``). Only that format's long-form
    period-keyed cards are converted (:mod:`novomodelo_bridge.decomp.converters.
    libs_electrical`), so the caller suppresses this warning once that subset
    converts and the ``decomp-libs-electrical-converted`` census is
    authoritative.

    Returns:
        A ``WARNING`` when the deck declares the file, else ``None``.
    """
    if libs_file is None:
        return None
    return Diagnostic(
        code="decomp-libs-electrical-present",
        severity=Severity.WARNING,
        category=_CATEGORY,
        title="LIBs electrical constraints present",
        summary=(
            "The deck's LIBS index lists a RESTRICAO-ELETRICA-ESPECIAL "
            "entry (the LIBs-era electrical-constraint family); this richer "
            "format is not converted, so the electrical model on this deck "
            "is not authoritative."
        ),
        remediation=(
            "LIBs electrical-constraint conversion is deferred; no action "
            "needed to convert."
        ),
    )
