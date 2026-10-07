"""Tier-1 tests for the special-constraint register readers (M2.1 T1).

Synthetic-fixture only — a fake ``Dadger`` returns pandas DataFrames shaped like
``idecomp``'s ``df=True`` accessors, so these run in CI with no real deck.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from novomodelo_bridge.core.diagnostics import Severity
from novomodelo_bridge.decomp.constraint_registers import (
    ConstraintRecord,
    ConstraintTerm,
    detect_libs_electrical,
    detect_unreadable_electrical,
    lowers_to_bound,
    read_constraints,
)
from novomodelo_bridge.decomp.converters.cadastro import EffectiveCadastro
from novomodelo_bridge.decomp.converters.constraints import (
    emit_re_generics,
    emit_rhq_rhv_generics,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.temporal import OperativeStage
from tests.conftest import _FakeDadger, make_decomp_case


def _stage(index: int, n_blocks: int) -> OperativeStage:
    return OperativeStage(
        index=index,
        start_date=date(2026, 7, 4),
        end_date=date(2026, 7, 11),
        season_id=6,
        block_hours=tuple(168.0 / n_blocks for _ in range(n_blocks)),
    )


def _decl(*rows: tuple[int, int, int]) -> pd.DataFrame:
    """RE/HQ/HV declaration rows: (constraint_id, estagio_inicial, estagio_final)."""
    return pd.DataFrame(
        rows, columns=["codigo_restricao", "estagio_inicial", "estagio_final"]
    )


def _coeff(*rows: tuple, tipo: bool = False, freq: bool = False) -> pd.DataFrame:
    """FU/CQ/CV rows: (constraint_id, codigo_usina, coeficiente, estagio[, tipo|freq])."""
    cols = ["codigo_restricao", "codigo_usina", "coeficiente", "estagio"]
    if tipo:
        cols.append("tipo")
    if freq:
        cols.append("frequencia")
    return pd.DataFrame(rows, columns=cols)


def _ft(*rows: tuple) -> pd.DataFrame:
    """FT rows: (constraint_id, codigo_submercado, codigo_usina, coeficiente, estagio)."""
    return pd.DataFrame(
        rows,
        columns=[
            "codigo_restricao",
            "codigo_submercado",
            "codigo_usina",
            "coeficiente",
            "estagio",
        ],
    )


def _fi(*rows: tuple) -> pd.DataFrame:
    """FI rows: (cid, codigo_submercado_de, codigo_submercado_para, coeficiente, estagio)."""
    return pd.DataFrame(
        rows,
        columns=[
            "codigo_restricao",
            "codigo_submercado_de",
            "codigo_submercado_para",
            "coeficiente",
            "estagio",
        ],
    )


def _lu(*rows: tuple) -> pd.DataFrame:
    """Per-block limit rows: (cid, estagio, li1, ls1, li2, ls2, li3, ls3)."""
    return pd.DataFrame(
        rows,
        columns=[
            "codigo_restricao",
            "estagio",
            "limite_inferior_1",
            "limite_superior_1",
            "limite_inferior_2",
            "limite_superior_2",
            "limite_inferior_3",
            "limite_superior_3",
        ],
    )


def _lv(*rows: tuple) -> pd.DataFrame:
    """Stage-level limit rows: (cid, estagio, limite_inferior, limite_superior)."""
    return pd.DataFrame(
        rows,
        columns=["codigo_restricao", "estagio", "limite_inferior", "limite_superior"],
    )


def _he(*rows: tuple) -> pd.DataFrame:
    """HE rows, all 9 idecomp columns: (cid, estagio, limite, tipo_limite,
    forma_calculo_produtibilidades, tipo_valores_produtibilidades,
    arquivo_produtibilidades, valor_penalidade, tipo_penalidade)."""
    return pd.DataFrame(
        rows,
        columns=[
            "codigo_restricao",
            "estagio",
            "limite",
            "tipo_limite",
            "forma_calculo_produtibilidades",
            "tipo_valores_produtibilidades",
            "arquivo_produtibilidades",
            "valor_penalidade",
            "tipo_penalidade",
        ],
    )


def _cm(*rows: tuple) -> pd.DataFrame:
    """CM rows: (constraint_id, codigo_ree, coeficiente)."""
    return pd.DataFrame(rows, columns=["codigo_restricao", "codigo_ree", "coeficiente"])


def _only(census_records: object, cid: int) -> ConstraintRecord:
    records = [r for r in census_records if r.constraint_id == cid]  # type: ignore[attr-defined]
    assert len(records) == 1, f"expected exactly one record for {cid}"
    return records[0]


def _term(coefficient: float, variable: str = "QDEF") -> ConstraintTerm:
    """A single flow term on one plant, for classifier-only tests."""
    return ConstraintTerm(code=30, coefficient=coefficient, variable=variable)


def _single_term_record(coefficient: float, variable: str = "QDEF") -> ConstraintRecord:
    """A minimal single-term HQ record built directly, no ``Dadger`` fixture —
    only ``lowers_to_bound``'s classification is under test."""
    return ConstraintRecord(
        family="HQ",
        constraint_id=1,
        stage_start=0,
        stage_end=0,
        terms=(_term(coefficient, variable),),
        bounds={},
        per_block=True,
    )


def test_re_single_term_lowers_to_generation_bound() -> None:
    dadger = _FakeDadger(
        re=_decl((10, 1, 2)),
        fu=_coeff((10, 21, 1.0, 1, float("nan")), freq=True),
        lu=_lu((10, 1, 50.0, 212.0, 50.0, 212.0, 50.0, 212.0)),
    )
    census = read_constraints(dadger)
    rec = _only(census.by_family["RE"], 10)
    assert rec.is_single_term
    assert rec.terms[0].variable == "generation"
    assert lowers_to_bound(rec)
    assert rec in census.to_bounds


def test_re_multi_term_lowers_to_generic() -> None:
    dadger = _FakeDadger(
        re=_decl((401, 1, 3)),
        fu=_coeff(
            (401, 285, 1.0, 1, float("nan")),
            (401, 287, 1.0, 1, float("nan")),
            freq=True,
        ),
        lu=_lu((401, 1, 0.0, 500.0, 0.0, 500.0, 0.0, 500.0)),
    )
    census = read_constraints(dadger)
    rec = _only(census.by_family["RE"], 401)
    assert not rec.is_single_term
    assert not lowers_to_bound(rec)
    assert rec in census.to_generic


def test_re_thermal_only_read() -> None:
    """FT-only participation (no FU) still produces a term, not skipped_no_terms."""
    dadger = _FakeDadger(
        re=_decl((10, 1, 2)),
        ft=_ft((10, 1, 5, 1.0, 1)),
    )
    census = read_constraints(dadger)
    rec = _only(census.by_family["RE"], 10)
    assert len(rec.terms) == 1
    assert rec.terms[0].variable == "thermal_generation"
    assert rec.terms[0].submarket == 1


def test_re_thermal_only_single_term_lowers_to_generation_bound() -> None:
    """A single-term ``FT`` RE (no ``FU``/``FI``) lowers to a thermal
    ``generation`` bound: the record moves into ``to_bounds`` and out of
    ``to_generic`` (M1)."""
    dadger = _FakeDadger(
        re=_decl((10, 1, 2)),
        ft=_ft((10, 1, 5, 1.0, 1)),
    )
    census = read_constraints(dadger)
    rec = _only(census.by_family["RE"], 10)
    assert rec.terms[0].variable == "thermal_generation"
    assert lowers_to_bound(rec)
    assert rec in census.to_bounds
    assert rec not in census.to_generic


def test_re_thermal_only_no_double_emission() -> None:
    """The same FT-only census never reaches ``emit_re_generics``: once the
    reader moves it to ``to_bounds``, the generic emitter's ``to_generic``
    filter no longer sees it, so it emits nothing (no special-casing needed
    in ``constraints.py`` — the split moving out of ``to_generic`` is the
    whole mechanism)."""
    dadger = _FakeDadger(
        re=_decl((10, 1, 2)),
        ft=_ft((10, 1, 5, 1.0, 1)),
    )
    census = read_constraints(dadger)
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), thermal_codes=(5,))
    calendar = [_stage(0, 1), _stage(1, 1)]

    case = make_decomp_case(Path("unused"), calendar=calendar)
    result = emit_re_generics(case, id_map, census=census, line_map={}, big_m=0.0)

    assert result is None


def test_re_interchange_only_is_generic() -> None:
    """FI-only participation (no FU) reads an interchange term and stays generic."""
    dadger = _FakeDadger(
        re=_decl((11, 1, 1)),
        fi=_fi((11, "SE", "S", 1.0, 1)),
    )
    census = read_constraints(dadger)
    rec = _only(census.by_family["RE"], 11)
    assert len(rec.terms) == 1
    term = rec.terms[0]
    assert term.variable == "interchange"
    assert term.submarket_de == "SE"
    assert term.submarket_para == "S"
    assert rec in census.to_generic


def test_re_all_three_registers() -> None:
    """FU + FT + FI on the same RE merge into three distinct terms."""
    dadger = _FakeDadger(
        re=_decl((12, 1, 1)),
        fu=_coeff((12, 21, 1.0, 1, float("nan")), freq=True),
        ft=_ft((12, 1, 5, 1.0, 1)),
        fi=_fi((12, "SE", "S", 1.0, 1)),
    )
    census = read_constraints(dadger)
    rec = _only(census.by_family["RE"], 12)
    assert len(rec.terms) == 3
    assert {t.variable for t in rec.terms} == {
        "generation",
        "thermal_generation",
        "interchange",
    }


def test_hq_qdef_single_lowers_to_outflow_bound() -> None:
    dadger = _FakeDadger(
        hq=_decl((5, 1, 1)),
        cq=_coeff((5, 30, 1.0, 1, "QDEF"), tipo=True),
        lq=_lu((5, 1, 10.0, 100.0, 10.0, 100.0, 10.0, 100.0)),
    )
    census = read_constraints(dadger)
    rec = _only(census.by_family["HQ"], 5)
    assert rec.terms[0].variable == "QDEF"
    assert lowers_to_bound(rec)


def test_hq_qbom_single_lowers_to_pumping_bound() -> None:
    """A single-term ``QBOM`` RHQ lowers to a pumping ``flow`` bound: the
    record moves into ``to_bounds`` and out of ``to_generic`` (M2)."""
    dadger = _FakeDadger(
        hq=_decl((5, 1, 1)),
        cq=_coeff((5, 30, 1.0, 1, "QBOM"), tipo=True),
        lq=_lu((5, 1, 5.0, 80.0, 5.0, 80.0, 5.0, 80.0)),
    )
    census = read_constraints(dadger)
    rec = _only(census.by_family["HQ"], 5)
    assert rec.terms[0].variable == "QBOM"
    assert lowers_to_bound(rec) is True
    assert rec in census.to_bounds
    assert rec not in census.to_generic


def test_hq_qbom_no_double_emission() -> None:
    """The same QBOM-only census never reaches ``emit_rhq_rhv_generics``:
    once the reader moves it to ``to_bounds``, the generic emitter's
    ``to_generic`` filter (``family in ("HQ", "HV")``) no longer sees it, so
    it returns ``None`` (no special-casing needed in ``constraints.py`` —
    the split moving out of ``to_generic`` is the whole mechanism)."""
    dadger = _FakeDadger(
        hq=_decl((5, 1, 1)),
        cq=_coeff((5, 30, 1.0, 1, "QBOM"), tipo=True),
        lq=_lu((5, 1, 5.0, 80.0, 5.0, 80.0, 5.0, 80.0)),
    )
    census = read_constraints(dadger)
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",))
    df = pd.DataFrame({5: {}}).T
    df.index.name = "codigo_usina"
    effective = EffectiveCadastro(base=df, n_stages=1, stage_varying={})
    calendar = [_stage(0, 1)]

    case = make_decomp_case(Path("unused"), calendar=calendar)
    result = emit_rhq_rhv_generics(
        case,
        id_map,
        census=census,
        pumping_station_ids={},
        effective=effective,
        big_m=0.0,
    )

    assert result is None


def test_hq_qdes_single_lowers_to_diversion_bound() -> None:
    """A single-term ``QDES`` RHQ lowers to a two-sided hydro ``diversion``
    bound (M3), now that novomodelo's
    generic-constraint-authoring landed ``min_diversion_m3s``."""
    dadger = _FakeDadger(
        hq=_decl((6, 1, 1)),
        cq=_coeff((6, 31, 1.0, 1, "QDES"), tipo=True),
        lq=_lu((6, 1, 0.0, 50.0, 0.0, 50.0, 0.0, 50.0)),
    )
    census = read_constraints(dadger)
    rec = _only(census.by_family["HQ"], 6)
    assert rec.is_single_term
    assert rec.terms[0].variable == "QDES"
    assert lowers_to_bound(rec)
    assert rec in census.to_bounds
    assert rec not in census.to_generic


def test_hq_qver_single_lowers_to_spillage_bound() -> None:
    """A single-term ``QVER`` RHQ lowers to a two-sided hydro ``spillage``
    bound (M5), now that novomodelo's
    generic-constraint-authoring landed ``min/max_spillage_m3s``."""
    dadger = _FakeDadger(
        hq=_decl((9, 1, 1)),
        cq=_coeff((9, 31, 1.0, 1, "QVER"), tipo=True),
        lq=_lu((9, 1, 0.0, 50.0, 0.0, 50.0, 0.0, 50.0)),
    )
    census = read_constraints(dadger)
    rec = _only(census.by_family["HQ"], 9)
    assert rec.is_single_term
    assert rec.terms[0].variable == "QVER"
    assert lowers_to_bound(rec)
    assert rec in census.to_bounds
    assert rec not in census.to_generic


def test_hq_mixed_flow_variables_one_plant_is_multi_term_generic() -> None:
    """QDEF + QDES on the same plant is two terms -> a generic constraint."""
    dadger = _FakeDadger(
        hq=_decl((164, 1, 1)),
        cq=_coeff(
            (164, 182, 1.0, 1, "QDEF"),
            (164, 182, 1.0, 1, "QDES"),
            tipo=True,
        ),
        lq=_lu((164, 1, 0.0, 900.0, 0.0, 900.0, 0.0, 900.0)),
    )
    census = read_constraints(dadger)
    rec = _only(census.by_family["HQ"], 164)
    assert rec.is_single_entity  # one plant …
    assert not rec.is_single_term  # … but two variables
    assert not lowers_to_bound(rec)


def test_non_unit_single_term_is_generic() -> None:
    """A single QDEF term with a non-unit coefficient (``0.5·QDEF ≤ U`` means
    ``QDEF ≤ 2U``) must not lower to a face-value bound — it routes to the
    generic emitter, which honours the coefficient."""
    record = _single_term_record(0.5)
    assert not lowers_to_bound(record)


def test_unit_negative_single_term_lowers_to_bound() -> None:
    """``|coef| == 1`` still lowers to a bound when the sign is negative."""
    record = _single_term_record(-1.0)
    assert lowers_to_bound(record)


def test_non_unit_single_term_qdes_is_generic() -> None:
    """A non-unit ``QDES`` coefficient stays generic too — the coefficient
    check applies uniformly across the whole ``_BOUNDS_AXIS`` membership,
    diversion included."""
    record = _single_term_record(0.5, variable="QDES")
    assert not lowers_to_bound(record)


def test_non_unit_single_term_qver_is_generic() -> None:
    """Same as above for ``QVER`` — the coefficient check is variable-agnostic."""
    record = _single_term_record(0.5, variable="QVER")
    assert not lowers_to_bound(record)


def test_hv_varm_single_lowers_to_storage_bound() -> None:
    dadger = _FakeDadger(
        hv=_decl((7, 1, 2)),
        cv=_coeff((7, 40, 1.0, 1, "VARM"), tipo=True),
        lv=_lv((7, 1, 100.0, 900.0)),
    )
    census = read_constraints(dadger)
    rec = _only(census.by_family["HV"], 7)
    assert not rec.per_block
    assert rec.terms[0].variable == "VARM"
    assert lowers_to_bound(rec)
    # Stage-level bounds carry a single slot.
    assert rec.bounds[0].lower == (100.0,)
    assert rec.bounds[0].upper == (900.0,)


def test_he_percentage_is_lower_bound() -> None:
    """``tipo_limite = 2`` (percentage of the REE's max stored energy) is a lower
    limit — the unit flag never selects which side the limit lands on."""
    dadger = _FakeDadger(
        he=_he((7, 1, 20.0, 2, 0, 1, "PRODRHE.DAT", 3370.0, 1)),
        cm=_cm((7, 3, 1.0)),
    )
    census = read_constraints(dadger)
    rec = _only(census.by_family["HE"], 7)
    assert rec.bounds[0].lower == (20.0,)
    assert rec.bounds[0].upper == (None,)


def test_he_absolute_is_lower_bound() -> None:
    """``tipo_limite = 1`` (absolute MWmes) lands on the same lower side as
    ``tipo_limite = 2`` — same side, different unit."""
    dadger = _FakeDadger(
        he=_he((7, 1, 500.0, 1, 0, 1, "PRODRHE.DAT", 3370.0, 1)),
        cm=_cm((7, 3, 1.0)),
    )
    census = read_constraints(dadger)
    rec = _only(census.by_family["HE"], 7)
    assert rec.bounds[0].lower == (500.0,)
    assert rec.bounds[0].upper == (None,)


def test_he_metadata_captured() -> None:
    dadger = _FakeDadger(
        he=_he((7, 1, 20.0, 2, 0, 1, "PRODRHE.DAT", 1234.5, 1)),
        cm=_cm((7, 3, 1.0)),
    )
    census = read_constraints(dadger)
    rec = _only(census.by_family["HE"], 7)
    assert rec.he_meta is not None
    assert rec.he_meta.forma_calculo_produtibilidades == 0
    assert rec.he_meta.arquivo_produtibilidades == "PRODRHE.DAT"
    assert rec.he_meta.valor_penalidade == 1234.5
    assert rec.he_meta.tipo_penalidade == 1


def test_he_is_generic() -> None:
    dadger = _FakeDadger(
        he=_he((7, 1, 20.0, 2, 0, 1, "PRODRHE.DAT", 3370.0, 1)),
        cm=_cm((7, 3, 1.0)),
    )
    census = read_constraints(dadger)
    rec = _only(census.by_family["HE"], 7)
    assert rec.family == "HE"
    assert rec.per_block is False
    assert rec.terms[0].variable == "energy"
    assert rec.is_single_term  # one REE …
    assert not lowers_to_bound(rec)  # … but energy has no bounds axis
    assert rec in census.to_generic


def test_sparse_limits_forward_fill_over_stage_range() -> None:
    """A limit declared only at the first stage densifies over the whole range."""
    dadger = _FakeDadger(
        re=_decl((10, 1, 3)),
        fu=_coeff((10, 21, 1.0, 1, float("nan")), freq=True),
        lu=_lu((10, 1, 5.0, 9.0, 5.0, 9.0, 5.0, 9.0)),  # declared at stage 1 only
    )
    census = read_constraints(dadger)
    rec = _only(census.by_family["RE"], 10)
    assert sorted(rec.bounds) == [0, 1, 2]  # dense over the 0-based range
    assert all(rec.bounds[s].lower[0] == 5.0 for s in (0, 1, 2))


def test_declared_constraint_without_participation_is_skipped() -> None:
    dadger = _FakeDadger(
        re=_decl((10, 1, 1), (11, 1, 1)),  # 11 has no FU rows
        fu=_coeff((10, 21, 1.0, 1, float("nan")), freq=True),
        lu=_lu((10, 1, 5.0, 9.0, 5.0, 9.0, 5.0, 9.0)),
    )
    census = read_constraints(dadger)
    assert {r.constraint_id for r in census.by_family["RE"]} == {10}


def test_per_block_bounds_read_all_declared_slots() -> None:
    dadger = _FakeDadger(
        hq=_decl((5, 1, 1)),
        cq=_coeff((5, 30, 1.0, 1, "QTUR"), tipo=True),
        lq=_lu((5, 1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)),
    )
    census = read_constraints(dadger)
    rec = _only(census.by_family["HQ"], 5)
    assert rec.per_block
    assert rec.bounds[0].lower == (1.0, 3.0, 5.0)
    assert rec.bounds[0].upper == (2.0, 4.0, 6.0)


def test_frequency_split_plant_contributes_two_terms() -> None:
    """A plant declared at 50 and 60 Hz is two terms -> generic, not one bound."""
    dadger = _FakeDadger(
        re=_decl((20, 1, 1)),
        fu=_coeff(
            (20, 66, 1.0, 1, 50.0),
            (20, 66, 1.0, 1, 60.0),
            freq=True,
        ),
        lu=_lu((20, 1, 0.0, 14000.0, 0.0, 14000.0, 0.0, 14000.0)),
    )
    census = read_constraints(dadger)
    rec = _only(census.by_family["RE"], 20)
    assert rec.is_single_entity  # one plant code …
    assert not rec.is_single_term  # … split across two frequencies
    assert {t.frequency for t in rec.terms} == {50.0, 60.0}
    assert not lowers_to_bound(rec)


def test_fe_line_warns_under_read(tmp_path: Path) -> None:
    """An ``FE`` participation line warns the electrical model may be
    under-read, and names no specific constraint id."""
    dadger_path = tmp_path / "dadger.rv0"
    dadger_path.write_text(
        "RE    10    1    2\nFE    10   21  1.0\n", encoding="latin-1"
    )

    diagnostics = detect_unreadable_electrical(dadger_path)

    assert len(diagnostics) == 1
    diag = diagnostics[0]
    assert diag.severity is Severity.WARNING
    assert "FE" in diag.summary
    assert "under-read" in diag.summary
    assert "10" not in diag.summary  # no specific constraint id named


def test_rha_lines_warn(tmp_path: Path) -> None:
    """``HA``/``LA``/``CA`` lines warn the RHA family is unconverted, and a
    mnemonic buried mid-line (not the leading token) never triggers it."""
    dadger_path = tmp_path / "dadger.rv0"
    dadger_path.write_text(
        "HA    1    2\n"
        "LA    1    2\n"
        "CA    1    2\n"
        "RE  10   1   2   XCA field\n",  # "XCA" leading token must not match
        encoding="latin-1",
    )

    diagnostics = detect_unreadable_electrical(dadger_path)

    assert len(diagnostics) == 1
    diag = diagnostics[0]
    assert diag.severity is Severity.WARNING
    assert "RHA" in diag.summary or "HA/LA/CA" in diag.summary
    assert "not" in diag.summary


def test_clean_deck_no_findings(tmp_path: Path) -> None:
    """No ``FE``/``HA``/``LA``/``CA`` leading tokens yields an empty list; a
    ``CA``-like substring inside another field must not trigger a false
    positive."""
    dadger_path = tmp_path / "dadger.rv0"
    dadger_path.write_text(
        "RE    10    1    2\n"
        "FU    10   21  1.0\n"
        "some field carrying a CA substring mid-line\n",
        encoding="latin-1",
    )

    assert detect_unreadable_electrical(dadger_path) == []


def test_libs_indices_warns(tmp_path: Path) -> None:
    """``indices.csv`` listing ``RESTRICAO-ELETRICA-ESPECIAL`` warns; a deck
    without that row, and a deck with no ``indices.csv`` at all, return
    ``None``."""
    libs_dir = tmp_path / "with_libs"
    libs_dir.mkdir()
    (libs_dir / "indices.csv").write_text(
        "RESTRICAO-ELETRICA-ESPECIAL;Descricao;lib_restricao-eletrica-especial.csv\n",
        encoding="latin-1",
    )
    diagnostic = detect_libs_electrical(libs_dir)
    assert diagnostic is not None
    assert diagnostic.severity is Severity.WARNING

    clean_dir = tmp_path / "without_libs"
    clean_dir.mkdir()
    (clean_dir / "indices.csv").write_text(
        "HIDRELETRICA-CADASTRO-RESERVATORIO-CURVAJUSANTE;Descricao;polinjus.csv\n",
        encoding="latin-1",
    )
    assert detect_libs_electrical(clean_dir) is None

    missing_dir = tmp_path / "missing_indices"
    missing_dir.mkdir()
    assert detect_libs_electrical(missing_dir) is None
