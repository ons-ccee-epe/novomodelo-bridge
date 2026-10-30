"""Tests for dessem/exporters/fcf/content/audit_binary_mirror.py."""

from novomodelo_bridge.dessem.exporters.fcf.content.audit_binary_mirror import (
    cortdeco_columns,
)


def test_cortdeco_columns_fixed_leading_columns() -> None:
    assert cortdeco_columns((7,), (3, 4), 2, 2)[:5] == (
        "record_index",
        "next_index",
        "is_extra_record",
        "rhs",
        "pi_varm_uhe7",
    )


def test_cortdeco_columns_gnl_labels_follow_layout_order() -> None:
    assert cortdeco_columns((7,), (3, 4), 2, 2)[5:] == (
        "pi_gnl_sbm3_pat1_lag1",
        "pi_gnl_sbm3_pat2_lag1",
        "pi_gnl_sbm3_pat1_lag2",
        "pi_gnl_sbm3_pat2_lag2",
        "pi_gnl_sbm4_pat1_lag1",
        "pi_gnl_sbm4_pat2_lag1",
        "pi_gnl_sbm4_pat1_lag2",
        "pi_gnl_sbm4_pat2_lag2",
    )


def test_cortdeco_columns_repeated_submarket_code_keeps_duplicate_labels() -> None:
    columns = cortdeco_columns((7,), (3, 3), 1, 1)
    assert columns.count("pi_gnl_sbm3_pat1_lag1") == 2


def test_cortdeco_columns_no_submarket_codes_has_no_gnl_labels() -> None:
    columns = cortdeco_columns((7,), (), 3, 3)
    assert not [name for name in columns if name.startswith("pi_gnl")]
