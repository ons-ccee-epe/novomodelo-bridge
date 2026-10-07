"""Unit tests for the ``compare decomp`` artifact export.

``compare decomp`` writes its
machine-readable artifacts through the same shared
:func:`novomodelo_bridge.comparators.export.write_artifacts` entry point as
``compare newave`` — there is no DECOMP-specific writer any more. This module
exercises that shared path against a DECOMP-shaped
:class:`~novomodelo_bridge.comparators.dataset.ComparisonDataset` (built the way
``build_decomp_dataset`` builds one), focusing on the one thing worth
DECOMP-specific coverage for: that the ``unmapped`` entity provenance
(``dataset.metadata["unmapped"]``) survives into the written artifact set.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from novomodelo_bridge.comparators.dataset import (
    SUMMARY_SCHEMA,
    TIDY_SCHEMA,
    ComparisonDataset,
)
from novomodelo_bridge.comparators.export import write_artifacts


def _make_decomp_dataset() -> ComparisonDataset:
    """A small validated dataset shaped like ``build_decomp_dataset``'s output.

    One hydro-generation row and one thermal-generation row, plus a
    DECOMP-flavoured ``unmapped`` metadata entry (entities present in the
    deck's own outputs that the id map could not resolve into Novomodelo ids).
    """
    tidy = pl.DataFrame(
        {
            "entity_type": ["hydro", "thermal"],
            "entity_id": [0, 0],
            "entity_name": ["A", "T"],
            "bus": [-1, -1],
            "stage": [0, 0],
            "block": [-1, -1],
            "variable": ["generation_mw", "generation_mw"],
            "source": ["newave", "novomodelo"],
            "value": [100.0, 90.0],
        },
        schema=TIDY_SCHEMA,
    )
    summary = pl.DataFrame(
        {
            "variable": ["generation_mw"],
            "count": [1],
            "mean_abs_diff": [10.0],
            "max_abs_diff": [10.0],
            "mean_smape": [0.105],
            "max_smape": [0.105],
            "within_tol_rate": [0.0],
            "correlation": [None],
        },
        schema=SUMMARY_SCHEMA,
    )
    metadata: dict[str, object] = {
        "unmapped": {"hydro": [], "thermal": [86, 224], "bus": []},
    }
    dataset = ComparisonDataset(tidy=tidy, summary=summary, metadata=metadata)
    dataset.validate()
    return dataset


def test_write_artifacts_emits_expected_files(tmp_path: Path) -> None:
    dataset = _make_decomp_dataset()
    decomp_dir = tmp_path / "decomp"
    novomodelo_output_dir = tmp_path / "output"
    out = tmp_path / "artifacts"

    write_artifacts(
        dataset,
        command="compare decomp",
        source_dir=decomp_dir,
        novomodelo_output_dir=novomodelo_output_dir,
        tolerance=1e-2,
        out_dir=out,
        formats=["parquet", "json", "csv"],
    )

    assert (out / "comparison.parquet").exists()
    assert (out / "summary.parquet").exists()
    assert (out / "metadata.json").exists()
    assert (out / "summary.json").exists()
    assert (out / "top_divergences.json").exists()
    assert (out / "comparison.csv").exists()
    assert (out / "summary.csv").exists()
    assert (out / "comparison.json").exists()


def test_unmapped_provenance_survives_in_metadata_json(tmp_path: Path) -> None:
    """The DECOMP-specific concern: ``dataset.metadata["unmapped"]`` (the
    per-level unresolved entity codes) is not DECOMP-shaped export state — it
    rides the dataset's own generic ``metadata.json`` serialization, exactly
    like every other metadata key."""
    dataset = _make_decomp_dataset()
    out = tmp_path / "artifacts"

    write_artifacts(
        dataset,
        command="compare decomp",
        source_dir=tmp_path / "decomp",
        novomodelo_output_dir=tmp_path / "output",
        tolerance=1e-2,
        out_dir=out,
        formats=["parquet"],
    )

    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["unmapped"] == {"hydro": [], "thermal": [86, 224], "bus": []}


def test_manifest_records_command_tolerance_and_artifacts(tmp_path: Path) -> None:
    dataset = _make_decomp_dataset()
    out = tmp_path / "artifacts"

    manifest = write_artifacts(
        dataset,
        command="compare decomp",
        source_dir=tmp_path / "decomp",
        novomodelo_output_dir=tmp_path / "output",
        tolerance=1e-2,
        out_dir=out,
        formats=["parquet", "json", "csv"],
    )

    payload = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert payload["command"] == "compare decomp"
    assert payload["tolerance"] == 1e-2
    assert payload["artifacts"] == sorted(payload["artifacts"])
    assert "comparison.json" in payload["artifacts"]
    assert manifest.command == "compare decomp"


def test_manifest_records_source_dir_input_files_and_diagnostics(
    tmp_path: Path,
) -> None:
    """The mislabel fix: ``comparison.json`` records the decomp deck
    under ``source_dir`` (never ``newave_dir``), plus the hashed input files and
    diagnostics threaded through from the caller."""
    dataset = _make_decomp_dataset()
    decomp_dir = tmp_path / "decomp"
    out = tmp_path / "artifacts"
    input_files = [
        {"field": "dadger", "path": "dadger.rv0", "sha256": "cafe", "size_bytes": 8}
    ]
    diagnostics = [{"code": "y", "severity": "info", "category": "Test"}]

    write_artifacts(
        dataset,
        command="compare decomp",
        source_dir=decomp_dir,
        novomodelo_output_dir=tmp_path / "output",
        tolerance=1e-2,
        out_dir=out,
        formats=["json"],
        input_files=input_files,
        diagnostics=diagnostics,
    )

    payload = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert payload["source_dir"] == str(decomp_dir)
    assert "newave_dir" not in payload
    assert payload["input_files"] == input_files
    assert payload["diagnostics"] == diagnostics


def test_unknown_format_raises_valueerror(tmp_path: Path) -> None:
    dataset = _make_decomp_dataset()

    with pytest.raises(ValueError, match="xml"):
        write_artifacts(
            dataset,
            command="compare decomp",
            source_dir=tmp_path / "decomp",
            novomodelo_output_dir=tmp_path / "output",
            tolerance=1e-2,
            out_dir=tmp_path / "artifacts",
            formats=["xml"],
        )


def test_empty_dataset_still_writes_typed_artifacts(tmp_path: Path) -> None:
    empty_dataset = ComparisonDataset(
        tidy=pl.DataFrame(schema=TIDY_SCHEMA),
        summary=pl.DataFrame(schema=SUMMARY_SCHEMA),
        metadata={"unmapped": {"hydro": [], "thermal": [], "bus": []}},
    )
    out = tmp_path / "artifacts"

    manifest = write_artifacts(
        empty_dataset,
        command="compare decomp",
        source_dir=tmp_path / "decomp",
        novomodelo_output_dir=tmp_path / "output",
        tolerance=1e-2,
        out_dir=out,
        formats=["parquet", "json", "csv"],
    )

    assert (out / "comparison.parquet").exists()
    assert (out / "summary.parquet").exists()
    assert (out / "comparison.json").exists()
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["unmapped"] == {"hydro": [], "thermal": [], "bus": []}
    assert manifest.artifacts == sorted(manifest.artifacts)
