"""Tests for the canonical active-hydro filter helpers.

A plant is fictitious **structurally**: zero productivity sharing an inflow
gauge (``posto``) with a generating plant. No name-prefix involved.
"""

from __future__ import annotations

import pandas as pd

from novomodelo_bridge.newave.plants import (
    active_hydro_codes,
    active_hydros,
    existing_hydros,
    fictitious_codes,
    filling_hydro_codes,
)


def _confhd(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _cadastro(rho: dict[int, float]) -> pd.DataFrame:
    """Hidr-like cadastro indexed by code: {code: produtibilidade_especifica}."""
    return pd.DataFrame(
        [{"codigo_usina": c, "produtibilidade_especifica": r} for c, r in rho.items()]
    ).set_index("codigo_usina")


# PLANT A (generates) shares posto 10 with FICT TWIN (ρ=0) -> twin is fictitious.
# ORPHAN (ρ=0) sits alone on posto 57 -> a real regulation reservoir, kept.
# FUTURE is NE (not existing).
_CASE = _confhd(
    [
        {
            "codigo_usina": 10,
            "nome_usina": "PLANT A",
            "posto": 10,
            "usina_existente": "EX",
        },
        {
            "codigo_usina": 3,
            "nome_usina": "FICT TWIN",
            "posto": 10,
            "usina_existente": "EX",
        },
        {
            "codigo_usina": 7,
            "nome_usina": "PLANT B",
            "posto": 7,
            "usina_existente": "EX",
        },
        {
            "codigo_usina": 99,
            "nome_usina": "FUTURE",
            "posto": 99,
            "usina_existente": "NE",
        },
        {
            "codigo_usina": 42,
            "nome_usina": "ORPHAN",
            "posto": 57,
            "usina_existente": "EX",
        },
    ]
)
_CAD = _cadastro({10: 0.01, 3: 0.0, 7: 0.01, 42: 0.0})


def test_existing_hydros_keeps_all_ex_including_fictitious() -> None:
    existing = existing_hydros(_CASE)
    assert sorted(existing["codigo_usina"]) == [3, 7, 10, 42]  # NE (99) dropped


def test_fictitious_codes_flags_zero_rho_sharing_a_generating_posto() -> None:
    # Code 3 (ρ=0) shares posto 10 with PLANT A (ρ>0) -> fictitious.
    # Code 42 (ρ=0) is alone on posto 57 -> NOT fictitious (real reservoir).
    assert fictitious_codes(_CASE, _CAD) == {3}


def test_active_hydros_drops_fictitious_keeps_orphan_rho_zero() -> None:
    # FICT TWIN (3) dropped; ORPHAN (42, ρ=0 but no generating posto-mate) kept.
    assert active_hydro_codes(_CASE, _CAD) == [10, 7, 42]
    assert list(active_hydros(_CASE, _CAD)["codigo_usina"]) == [10, 7, 42]


def test_active_hydros_preserves_declaration_order() -> None:
    # 10 is declared before 7; the id-map relies on this ordering.
    assert active_hydro_codes(_CASE, _CAD)[:2] == [10, 7]


def test_full_case_twin_is_fictitious() -> None:
    # MAUA (generates) + FICT.MAUA (ρ=0) on the same posto -> FICT.MAUA filtered.
    confhd = _confhd(
        [
            {
                "codigo_usina": 57,
                "nome_usina": "MAUA",
                "posto": 57,
                "usina_existente": "EX",
            },
            {
                "codigo_usina": 319,
                "nome_usina": "FICT.MAUA",
                "posto": 57,
                "usina_existente": "EX",
            },
        ]
    )
    cad = _cadastro({57: 0.009, 319: 0.0})
    assert fictitious_codes(confhd, cad) == {319}
    assert active_hydro_codes(confhd, cad) == [57]


def test_reduced_case_orphaned_twin_is_kept() -> None:
    # MAUA removed: FICT.MAUA (ρ=0) is now alone on posto 57 -> NOT fictitious,
    # kept in the LP as a routing/regulation reservoir.
    confhd = _confhd(
        [
            {
                "codigo_usina": 61,
                "nome_usina": "CAPIVARA",
                "posto": 61,
                "usina_existente": "EX",
            },
            {
                "codigo_usina": 319,
                "nome_usina": "FICT.MAUA",
                "posto": 57,
                "usina_existente": "EX",
            },
        ]
    )
    cad = _cadastro({61: 0.009, 319: 0.0})
    assert fictitious_codes(confhd, cad) == set()
    assert active_hydro_codes(confhd, cad) == [61, 319]


def test_two_zero_rho_on_one_posto_without_generator_are_both_real() -> None:
    # No generating plant on the shared posto -> neither is fictitious.
    confhd = _confhd(
        [
            {"codigo_usina": 1, "nome_usina": "A", "posto": 5, "usina_existente": "EX"},
            {"codigo_usina": 2, "nome_usina": "B", "posto": 5, "usina_existente": "EX"},
        ]
    )
    cad = _cadastro({1: 0.0, 2: 0.0})
    assert fictitious_codes(confhd, cad) == set()


def test_fictitious_codes_empty_without_data() -> None:
    # No cadastro (empty) or no posto column -> cannot classify -> empty set.
    assert fictitious_codes(_CASE, pd.DataFrame()) == set()
    no_posto = _confhd(
        [{"codigo_usina": 1, "nome_usina": "A", "usina_existente": "EX"}]
    )
    assert fictitious_codes(no_posto, _cadastro({1: 0.0})) == set()


def test_empty_confhd_yields_empty_active_set() -> None:
    empty = _confhd(
        [{"codigo_usina": 1, "nome_usina": "GONE", "posto": 1, "usina_existente": "NE"}]
    )
    assert active_hydro_codes(empty, _cadastro({1: 0.01})) == []


# --- Dead-volume filling admission ------------------------------


def _exph(rows: list[dict]) -> pd.DataFrame:
    """Build an exph-like expansoes DataFrame with datetime fill-start column."""
    df = pd.DataFrame(rows)
    df["data_inicio_enchimento"] = pd.to_datetime(df["data_inicio_enchimento"])
    return df


def test_filling_hydro_codes_excludes_nc() -> None:
    # An NC plant carrying a filling row is NOT admitted; only NE-with-filling is.
    confhd_df = _confhd(
        [
            {"codigo_usina": 1, "nome_usina": "EX1", "usina_existente": "EX"},
            {"codigo_usina": 2, "nome_usina": "NE_FILL", "usina_existente": "NE"},
            {"codigo_usina": 3, "nome_usina": "NC_FILL", "usina_existente": "NC"},
            {"codigo_usina": 4, "nome_usina": "NE_BARE", "usina_existente": "NE"},
        ]
    )
    exph_df = _exph(
        [
            {"codigo_usina": 2, "data_inicio_enchimento": "2024-10-01"},
            {"codigo_usina": 3, "data_inicio_enchimento": "2024-10-01"},
            {"codigo_usina": 4, "data_inicio_enchimento": None},
        ]
    )
    # 3 is NC (excluded), 4 is NE but bare (no filling row) -> only {2}.
    assert filling_hydro_codes(confhd_df, exph_df) == {2}


def test_filling_hydro_codes_none_exph() -> None:
    confhd_df = _confhd(
        [{"codigo_usina": 2, "nome_usina": "NE", "usina_existente": "NE"}]
    )
    assert filling_hydro_codes(confhd_df, None) == set()
    assert filling_hydro_codes(confhd_df, pd.DataFrame()) == set()


def test_active_hydros_includes_filling_in_declaration_order() -> None:
    # NE-with-filling (309-like code 5) sits at its confhd declaration position,
    # not appended at the end.
    confhd_df = _confhd(
        [
            {
                "codigo_usina": 10,
                "nome_usina": "A",
                "posto": 10,
                "usina_existente": "EX",
            },
            {
                "codigo_usina": 5,
                "nome_usina": "FILL",
                "posto": 5,
                "usina_existente": "NE",
            },
            {"codigo_usina": 7, "nome_usina": "B", "posto": 7, "usina_existente": "EX"},
        ]
    )
    cad = _cadastro({10: 0.01, 7: 0.01})
    exph_df = _exph([{"codigo_usina": 5, "data_inicio_enchimento": "2024-10-01"}])
    codes = active_hydro_codes(confhd_df, cad, exph_df)
    # 5 is admitted and appears between 10 and 7 (declaration order), not last.
    assert codes == [10, 5, 7]
    assert list(active_hydros(confhd_df, cad, exph_df)["codigo_usina"]) == [10, 5, 7]


def test_active_hydros_unchanged_when_exph_none() -> None:
    # With exph_df=None the NE plant is dropped -> identical to the in-service set.
    confhd_df = _confhd(
        [
            {
                "codigo_usina": 10,
                "nome_usina": "A",
                "posto": 10,
                "usina_existente": "EX",
            },
            {
                "codigo_usina": 5,
                "nome_usina": "FILL",
                "posto": 5,
                "usina_existente": "NE",
            },
            {"codigo_usina": 7, "nome_usina": "B", "posto": 7, "usina_existente": "EX"},
        ]
    )
    cad = _cadastro({10: 0.01, 7: 0.01})
    assert active_hydro_codes(confhd_df, cad) == [10, 7]
    assert active_hydro_codes(confhd_df, cad, None) == [10, 7]
    assert list(active_hydros(confhd_df, cad)["codigo_usina"]) == [10, 7]


# --- In-service admission: EE ("existente em expansão") ---------


_EE_CASE = _confhd(
    [
        {"codigo_usina": 10, "nome_usina": "A", "posto": 10, "usina_existente": "EX"},
        {
            "codigo_usina": 5,
            "nome_usina": "EXPANDING",
            "posto": 5,
            "usina_existente": "EE",
        },
        {"codigo_usina": 7, "nome_usina": "B", "posto": 7, "usina_existente": "EX"},
        {"codigo_usina": 9, "nome_usina": "GONE", "posto": 9, "usina_existente": "NC"},
    ]
)
_EE_CAD = _cadastro({10: 0.01, 5: 0.01, 7: 0.01})


def test_existing_hydros_includes_ee() -> None:
    assert sorted(existing_hydros(_EE_CASE)["codigo_usina"]) == [5, 7, 10]


def test_existing_hydros_tolerates_padded_status() -> None:
    padded = _confhd(
        [{"codigo_usina": 1, "nome_usina": "A", "posto": 1, "usina_existente": " EE "}]
    )
    assert list(existing_hydros(padded)["codigo_usina"]) == [1]


def test_active_hydros_admits_ee_at_declaration_position_without_exph() -> None:
    # EE admission reads confhd alone, so a path-only caller (the comparators'
    # ``build_id_map``, which passes exph=None) enumerates the pipeline's set.
    assert active_hydro_codes(_EE_CASE, _EE_CAD) == [10, 5, 7]
    assert list(active_hydros(_EE_CASE, _EE_CAD, None)["codigo_usina"]) == [10, 5, 7]
