"""CLI-level coverage for the converged compare ``--json`` status vocabulary.

Both ``compare newave`` and ``compare decomp`` derive their ``--json``
``status`` from the shared ``compare_status(dataset)`` helper, including the
empty-dataset ``"no-comparable-rows"`` value.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import polars as pl
import pytest
from typer.testing import CliRunner

from novomodelo_bridge.cli import app
from novomodelo_bridge.comparators.analyze import build_results_dataset
from novomodelo_bridge.comparators.dataset import (
    SUMMARY_SCHEMA,
    TIDY_SCHEMA,
    ComparisonDataset,
)
from novomodelo_bridge.comparators.model import PercentileData, ResultComparison


def _newave_dataset(*, all_within_tol: bool) -> ComparisonDataset:
    results = [
        ResultComparison(
            entity_type="hydro",
            entity_name="ITAIPU",
            newave_code=10,
            novomodelo_id=0,
            stage=0,
            variable="generation_mw",
            newave_value=100.0,
            novomodelo_value=100.0 if all_within_tol else 110.0,
            abs_diff=0.0 if all_within_tol else 10.0,
            rel_diff=0.0 if all_within_tol else 0.1,
        ),
    ]
    return build_results_dataset(results, PercentileData(), 1e-2)


def _empty_newave_dataset() -> ComparisonDataset:
    return build_results_dataset([], PercentileData(), 1e-2)


def _decomp_dataset(*, all_within_tol: bool) -> ComparisonDataset:
    tidy = pl.DataFrame(
        {
            "entity_type": ["hydro", "hydro"],
            "entity_id": [0, 0],
            "entity_name": ["A", "A"],
            "bus": [-1, -1],
            "stage": [0, 0],
            "block": [-1, -1],
            "variable": ["generation_mw", "generation_mw"],
            "source": ["newave", "novomodelo"],
            "value": [100.0, 100.0 if all_within_tol else 110.0],
        },
        schema=TIDY_SCHEMA,
    )
    summary = pl.DataFrame(
        {
            "variable": ["generation_mw"],
            "count": [1],
            "mean_abs_diff": [0.0 if all_within_tol else 10.0],
            "max_abs_diff": [0.0 if all_within_tol else 10.0],
            "mean_smape": [0.0 if all_within_tol else 0.1],
            "max_smape": [0.0 if all_within_tol else 0.1],
            "within_tol_rate": [1.0 if all_within_tol else 0.0],
            "correlation": [1.0],
        },
        schema=SUMMARY_SCHEMA,
    )
    return ComparisonDataset(
        tidy=tidy,
        summary=summary,
        metadata={"unmapped": {"hydro": [], "thermal": [], "bus": []}},
    )


def _empty_decomp_dataset() -> ComparisonDataset:
    return ComparisonDataset(
        tidy=pl.DataFrame(schema=TIDY_SCHEMA),
        summary=pl.DataFrame(schema=SUMMARY_SCHEMA),
        metadata={"unmapped": {"hydro": [], "thermal": [], "bus": []}},
    )


def _patch_newave_context(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.conftest import make_nw_files

    # ``.files`` must be a real ``NewaveFiles`` dataclass (not a further
    # MagicMock attribute) — ``hash_input_files`` reflects over it via
    # ``dataclasses.fields``, which raises on a non-dataclass. The paths
    # need not exist: a missing file degrades to a ``None`` hash/size.
    monkeypatch.setattr(
        "novomodelo_bridge.newave.case.NewaveCase.from_directory",
        classmethod(
            lambda cls, _dir: MagicMock(
                id_map=MagicMock(), files=make_nw_files(Path("nw"))
            )
        ),
    )
    monkeypatch.setattr(
        "novomodelo_bridge.comparators.newave.alignment.build_entity_alignment",
        lambda *a, **k: MagicMock(),
    )
    monkeypatch.setattr(
        "novomodelo_bridge.novomodelo.readers.read_novomodelo_lines", lambda _dir: []
    )


def _invoke_newave(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dataset: ComparisonDataset
) -> Any:
    _patch_newave_context(monkeypatch)
    monkeypatch.setattr(
        "novomodelo_bridge.comparators.newave.results.compare_results",
        lambda **_kwargs: dataset,
    )
    novomodelo_dir = tmp_path / "novomodelo"
    novomodelo_dir.mkdir()
    return CliRunner().invoke(
        app, ["compare", "newave", str(tmp_path / "nw"), str(novomodelo_dir), "--json"]
    )


def _invoke_decomp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dataset: ComparisonDataset
) -> Any:
    from tests.conftest import make_decomp_case

    # ``DecompCase.from_directory`` is re-invoked (for manifest hashing) after
    # ``build_decomp_dataset`` is mocked away; give it a real ``DecompFiles``
    # dataclass instead of trying to discover a deck under the fake ``tmp_path``.
    monkeypatch.setattr(
        "novomodelo_bridge.decomp.case.DecompCase.from_directory",
        classmethod(lambda cls, _dir: make_decomp_case(Path("decomp"))),
    )
    monkeypatch.setattr(
        "novomodelo_bridge.comparators.decomp.results.build_decomp_dataset",
        lambda *_args, **_kwargs: dataset,
    )
    return CliRunner().invoke(
        app, ["compare", "decomp", str(tmp_path), str(tmp_path), "--json"]
    )


class TestCompareNewaveStatusConvergence:
    """``compare newave --json`` now uses ``compare_status``'s vocabulary,
    including ``"no-comparable-rows"`` on an empty dataset."""

    def test_empty_dataset_reports_no_comparable_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = _invoke_newave(tmp_path, monkeypatch, _empty_newave_dataset())

        assert result.exit_code == 0
        assert json.loads(result.stdout)["status"] == "no-comparable-rows"

    def test_all_within_tol_reports_ok(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = _invoke_newave(
            tmp_path, monkeypatch, _newave_dataset(all_within_tol=True)
        )

        assert result.exit_code == 0
        assert json.loads(result.stdout)["status"] == "ok"

    def test_out_of_tolerance_reports_mismatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = _invoke_newave(
            tmp_path, monkeypatch, _newave_dataset(all_within_tol=False)
        )

        assert result.exit_code == 0
        assert json.loads(result.stdout)["status"] == "mismatch"


class TestCompareDecompStatusConvergence:
    """``compare decomp --json`` status is unchanged by the convergence, but
    now shares its implementation with ``compare newave``."""

    def test_empty_dataset_reports_no_comparable_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = _invoke_decomp(tmp_path, monkeypatch, _empty_decomp_dataset())

        assert result.exit_code == 0
        assert json.loads(result.stdout)["status"] == "no-comparable-rows"

    def test_all_within_tol_reports_ok(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = _invoke_decomp(
            tmp_path, monkeypatch, _decomp_dataset(all_within_tol=True)
        )

        assert result.exit_code == 0
        assert json.loads(result.stdout)["status"] == "ok"

    def test_out_of_tolerance_reports_mismatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = _invoke_decomp(
            tmp_path, monkeypatch, _decomp_dataset(all_within_tol=False)
        )

        assert result.exit_code == 0
        assert json.loads(result.stdout)["status"] == "mismatch"
