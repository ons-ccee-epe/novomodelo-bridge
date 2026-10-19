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


def _structural(confhd_df: pd.DataFrame, cadastro: pd.DataFrame) -> frozenset[int]:
    """The structural fictitious set, with no pumping station in the case."""
    return fictitious_codes(confhd_df, cadastro, pumping_related_codes=frozenset())


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
    assert _structural(_CASE, _CAD) == {3}


def test_active_hydros_drops_fictitious_keeps_orphan_rho_zero() -> None:
    # FICT TWIN (3) dropped; ORPHAN (42, ρ=0 but no generating posto-mate) kept.
    assert active_hydro_codes(_CASE, fictitious=_structural(_CASE, _CAD)) == [10, 7, 42]
    assert list(
        active_hydros(_CASE, fictitious=_structural(_CASE, _CAD))["codigo_usina"]
    ) == [10, 7, 42]


def test_active_hydros_preserves_declaration_order() -> None:
    # 10 is declared before 7; the id-map relies on this ordering.
    assert active_hydro_codes(_CASE, fictitious=_structural(_CASE, _CAD))[:2] == [10, 7]


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
    assert _structural(confhd, cad) == {319}
    assert active_hydro_codes(confhd, fictitious=_structural(confhd, cad)) == [57]


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
    assert _structural(confhd, cad) == set()
    assert active_hydro_codes(confhd, fictitious=_structural(confhd, cad)) == [61, 319]


def test_two_zero_rho_on_one_posto_without_generator_are_both_real() -> None:
    # No generating plant on the shared posto -> neither is fictitious.
    confhd = _confhd(
        [
            {"codigo_usina": 1, "nome_usina": "A", "posto": 5, "usina_existente": "EX"},
            {"codigo_usina": 2, "nome_usina": "B", "posto": 5, "usina_existente": "EX"},
        ]
    )
    cad = _cadastro({1: 0.0, 2: 0.0})
    assert _structural(confhd, cad) == set()


def test_fictitious_codes_empty_without_data() -> None:
    # No cadastro (empty) or no posto column -> cannot classify -> empty set.
    assert _structural(_CASE, pd.DataFrame()) == set()
    no_posto = _confhd(
        [{"codigo_usina": 1, "nome_usina": "A", "usina_existente": "EX"}]
    )
    assert _structural(no_posto, _cadastro({1: 0.0})) == set()


def test_empty_confhd_yields_empty_active_set() -> None:
    empty = _confhd(
        [{"codigo_usina": 1, "nome_usina": "GONE", "posto": 1, "usina_existente": "NE"}]
    )
    assert (
        active_hydro_codes(empty, fictitious=_structural(empty, _cadastro({1: 0.01})))
        == []
    )


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
    codes = active_hydro_codes(
        confhd_df, fictitious=_structural(confhd_df, cad), exph_df=exph_df
    )
    # 5 is admitted and appears between 10 and 7 (declaration order), not last.
    assert codes == [10, 5, 7]
    assert list(
        active_hydros(
            confhd_df, fictitious=_structural(confhd_df, cad), exph_df=exph_df
        )["codigo_usina"]
    ) == [10, 5, 7]


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
    assert active_hydro_codes(confhd_df, fictitious=_structural(confhd_df, cad)) == [
        10,
        7,
    ]
    assert active_hydro_codes(
        confhd_df, fictitious=_structural(confhd_df, cad), exph_df=None
    ) == [10, 7]
    assert list(
        active_hydros(confhd_df, fictitious=_structural(confhd_df, cad))["codigo_usina"]
    ) == [10, 7]


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
    # EE admission reads confhd alone, so it does not depend on exph.
    assert active_hydro_codes(_EE_CASE, fictitious=_structural(_EE_CASE, _EE_CAD)) == [
        10,
        5,
        7,
    ]
    assert list(
        active_hydros(
            _EE_CASE, fictitious=_structural(_EE_CASE, _EE_CAD), exph_df=None
        )["codigo_usina"]
    ) == [10, 5, 7]


# ---------------------------------------------------------------------------
# Pumping-related codes exemption
# ---------------------------------------------------------------------------


def test_pumping_related_codes_excluded_from_fictitious() -> None:
    """Pumping-related codes are excluded from fictitious classification."""
    # Same setup as full-case twin test: 319 would be fictitious
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

    # Without pumping exclusion: 319 is fictitious
    fict_without = _structural(confhd, cad)
    assert 319 in fict_without

    # With pumping exclusion: 319 is preserved
    fict_with = fictitious_codes(confhd, cad, pumping_related_codes=frozenset({319}))
    assert 319 not in fict_with


def test_pumping_exemption_propagates_to_active_hydros() -> None:
    """active_hydros respects pumping exemption via fictitious parameter."""
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

    # Without exemption: 319 is dropped
    codes_without = active_hydro_codes(confhd, fictitious=_structural(confhd, cad))
    assert 319 not in codes_without

    # With exemption via fictitious parameter: 319 is kept
    fict = fictitious_codes(confhd, cad, pumping_related_codes=frozenset({319}))
    codes_with = active_hydro_codes(confhd, fictitious=fict)
    assert 319 in codes_with


def test_fictitious_codes_returns_frozenset() -> None:
    """fictitious_codes returns frozenset for immutability."""
    confhd = _confhd(
        [
            {
                "codigo_usina": 57,
                "nome_usina": "MAUA",
                "posto": 57,
                "usina_existente": "EX",
            },
        ]
    )
    cad = _cadastro({57: 0.009})

    result = _structural(confhd, cad)
    assert isinstance(result, frozenset)


# ---------------------------------------------------------------------------
# Regression test: VIGARIO / cascade cycle scenario (PR #33)
# ---------------------------------------------------------------------------


def test_vigario_scenario_pumping_exemption_prevents_cascade_cycle() -> None:
    """Regression test for the VIGARIO/cycle 131-133-182 scenario.

    In the real deck:
    - NILO PECANHA (131) and P. PASSOS (133) share posto 300 with VIGARIO (182)
    - P. PASSOS (133) is terminal (downstream = 0)
    - VIGARIO (182) has ρ = 0 (zero productivity)
    - VIGARIO is a pumping destination

    Without the pumping exemption:
    1. VIGARIO (182) is classified as fictitious (ρ=0, shares posto with real plant)
    2. P. PASSOS (133) gets VIGARIO as its "twin" (rule 3 of resolve_cascade)
    3. VIGARIO's downstream is NILO PECANHA (131)
    4. But 131 → 133 in the real cascade
    5. This creates a cycle: 133 → 182 → 131 → 133

    With the pumping exemption, VIGARIO stays an LP plant and no cycle forms.
    """
    # Minimal confhd reproducing the scenario:
    # - 131: NILO PECANHA, posto 300, downstream = 133
    # - 133: P. PASSOS, posto 300, downstream = 0 (terminal)
    # - 182: VIGARIO, posto 300, downstream = 131
    confhd = _confhd(
        [
            {
                "codigo_usina": 131,
                "nome_usina": "NILO PECANHA",
                "posto": 300,
                "usina_existente": "EX",
                "codigo_usina_jusante": 133,
            },
            {
                "codigo_usina": 133,
                "nome_usina": "P. PASSOS",
                "posto": 300,
                "usina_existente": "EX",
                "codigo_usina_jusante": 0,
            },
            {
                "codigo_usina": 182,
                "nome_usina": "VIGARIO",
                "posto": 300,
                "usina_existente": "EX",
                "codigo_usina_jusante": 131,
            },
        ]
    )

    # 182 has zero productivity (would be fictitious without exemption)
    cad = _cadastro({131: 0.01, 133: 0.01, 182: 0.0})

    # Without pumping exemption: 182 is fictitious
    fict_without = _structural(confhd, cad)
    assert 182 in fict_without, "182 should be fictitious without pumping exemption"

    # With pumping exemption: 182 is NOT fictitious
    pumping_related = frozenset({182})  # VIGARIO is a pumping destination
    fict_with = fictitious_codes(confhd, cad, pumping_related_codes=pumping_related)
    assert 182 not in fict_with, "182 should NOT be fictitious with pumping exemption"

    # All three plants should appear in active hydros
    codes = active_hydro_codes(confhd, fictitious=fict_with)
    assert 131 in codes
    assert 133 in codes
    assert 182 in codes

    # Verify cascade resolution doesn't form a cycle
    from novomodelo_bridge.newave.converters.fict_cascade import resolve_cascade

    resolutions = resolve_cascade(confhd, cad, fictitious=fict_with)

    # With the exemption applied:
    # - 131 (NILO PECANHA) → 133 (P. PASSOS) - direct cascade
    # - 133 (P. PASSOS) → terminal (no twin needed, 182 is not fictitious)
    # - 182 (VIGARIO) → 131 (NILO PECANHA) - direct cascade
    assert resolutions[131].downstream_code == 133
    assert resolutions[133].downstream_code is None  # terminal
    assert resolutions[182].downstream_code == 131

    # No cycle - this would have raised BridgeError in _cascade_sum if there was one
    from novomodelo_bridge.newave.converters.constraints import _cascade_sum

    # Build a downstream map from resolutions
    downstream_map = {code: r.downstream_code for code, r in resolutions.items()}
    own_prod = {131: 0.01, 133: 0.01, 182: 0.0}

    # This should complete without raising BridgeError
    acc = _cascade_sum(downstream_map, own_prod)
    assert 131 in acc
    assert 133 in acc
    assert 182 in acc
