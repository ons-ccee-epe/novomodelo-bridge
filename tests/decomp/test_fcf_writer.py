"""Tests for the boundary-checkpoint writer (``fcf/writer.py``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from novomodelo_bridge.decomp.fcf.mapper import MappingResult
from novomodelo_bridge.decomp.fcf.writer import (
    build_metadata,
    build_stage_cuts_payload,
    write_boundary_checkpoint,
)
from tests._fcf_fixtures import make_manifest, make_mapped_cut, make_slot
from tests.conftest import requires_novomodelo_python


def test_build_stage_cuts_payload_shape() -> None:
    manifest = make_manifest([make_slot(0, i, 0) for i in range(4)])
    active_cut = make_mapped_cut(
        coefficients=(1.0, 2.0, 3.0, 4.0),
        cut_id=101,
        iteration=5,
        forward_pass_index=2,
        is_active=True,
    )
    inactive_cut = make_mapped_cut(
        coefficients=(5.0, 6.0, 7.0, 8.0),
        cut_id=202,
        iteration=6,
        forward_pass_index=3,
        is_active=False,
    )
    mapping = MappingResult(cuts=(active_cut, inactive_cut), dropped=())

    payload = build_stage_cuts_payload(
        mapping,
        manifest,
        stage_id=10,
        cost_scale_factor=1.0,
        node_id=0,
        graph_stage_id=4,
        priced_state_date=manifest.priced_state_date,
    )

    assert payload["stage_id"] == 10
    assert payload["state_dimension"] == 4
    assert payload["entity_manifest"] == list(manifest.entity_manifest)
    assert payload["cost_scale_factor"] == 1.0
    assert payload["node_id"] == 0
    assert payload["graph_stage_id"] == 4
    assert payload["priced_state_date"] == manifest.priced_state_date
    assert len(payload["cuts"]) == 2
    for cut in payload["cuts"]:
        assert len(cut["coefficients"]) == 4

    first, second = payload["cuts"]
    assert first["cut_id"] == active_cut.cut_id
    assert first["iteration"] == active_cut.iteration
    assert first["forward_pass_index"] == active_cut.forward_pass_index
    assert first["is_active"] is True
    assert second["cut_id"] == inactive_cut.cut_id
    assert second["iteration"] == inactive_cut.iteration
    assert second["forward_pass_index"] == inactive_cut.forward_pass_index
    assert second["is_active"] is False

    assert payload["active_cut_indices"] == [0]
    assert payload["populated_count"] == 2


def test_build_metadata_refuses_none_cost_scale_factor() -> None:
    with pytest.raises(RuntimeError, match="cost_scale_factor"):
        build_metadata(
            num_stages=1,
            cost_scale_factor=None,
            completed_iterations=1,
            final_lower_bound=0.0,
            max_iterations=1,
            forward_passes=1,
            warm_start_cuts=0,
            rng_seed=0,
            created_at="2026-08-03T00:00:00Z",
        )


@requires_novomodelo_python
def test_write_boundary_checkpoint_creates_files(tmp_path: Path) -> None:
    import novomodelo

    manifest = make_manifest([make_slot(0, i, 0) for i in range(2)])
    mapping = MappingResult(
        cuts=(make_mapped_cut(coefficients=(1.0, 2.0), cut_id=101, is_active=True),),
        dropped=(),
    )
    payload = build_stage_cuts_payload(
        mapping,
        manifest,
        stage_id=10,
        cost_scale_factor=1.0,
        node_id=0,
        graph_stage_id=10,
        priced_state_date=manifest.priced_state_date,
    )
    metadata = build_metadata(
        num_stages=1,
        cost_scale_factor=1.0,
        completed_iterations=1,
        final_lower_bound=0.0,
        max_iterations=1,
        forward_passes=1,
        warm_start_cuts=0,
        rng_seed=0,
        created_at="2026-08-03T00:00:00Z",
    )

    boundary_dir = tmp_path / "boundary"
    write_boundary_checkpoint(boundary_dir, payload, metadata)

    assert (boundary_dir / "manifest.bin").exists()
    # novomodelo 0.14 keys the cut file by pool id (stage_id 10 -> "010.bin"),
    # replacing the old positional "stage_NNN.bin".
    assert (boundary_dir / "cuts" / "010.bin").exists()
    assert (boundary_dir / "basis").is_dir()

    reloaded = novomodelo.results.load_policy(
        boundary_dir.parent, policy_subdir=boundary_dir.name
    )
    # 0.14 nests the algorithm provenance (incl. cost_scale_factor) under a
    # "producer" block; state_dimension is no longer a metadata field.
    assert "state_dimension" not in reloaded["metadata"]
    assert reloaded["metadata"]["producer"]["cost_scale_factor"] is not None
    # Only the software and version that wrote a checkpoint load it, so the
    # boundary must carry the installed novomodelo-python's own identity.
    assert reloaded["metadata"]["software"] == "novomodelo"
    assert reloaded["metadata"]["software_version"] == novomodelo.__version__
    reloaded_stage = reloaded["stage_cuts"][0]
    assert reloaded_stage["state_dimension"] == 2
    assert len(reloaded_stage["cuts"]) == 1


def test_build_payload_rejects_wrong_coefficient_length() -> None:
    manifest = make_manifest([make_slot(0, i, 0) for i in range(4)])
    mapping = MappingResult(
        cuts=(make_mapped_cut(coefficients=(1.0, 2.0, 3.0)),),  # length 3 != 4
        dropped=(),
    )

    with pytest.raises(ValueError, match="state_dimension"):
        build_stage_cuts_payload(
            mapping,
            manifest,
            stage_id=10,
            cost_scale_factor=1.0,
            node_id=0,
            graph_stage_id=10,
            priced_state_date=manifest.priced_state_date,
        )


def test_build_stage_cuts_payload_refuses_none_cost_scale_factor() -> None:
    manifest = make_manifest([make_slot(0, 0, 0)])
    mapping = MappingResult(cuts=(make_mapped_cut(coefficients=(0.5,)),), dropped=())

    with pytest.raises(RuntimeError, match="cost_scale_factor"):
        build_stage_cuts_payload(
            mapping,
            manifest,
            stage_id=10,
            cost_scale_factor=None,
            node_id=0,
            graph_stage_id=10,
            priced_state_date=manifest.priced_state_date,
        )


def test_build_stage_cuts_payload_refuses_shared_pool_sentinel_node_id() -> None:
    manifest = make_manifest([make_slot(0, 0, 0)])
    mapping = MappingResult(cuts=(make_mapped_cut(coefficients=(0.5,)),), dropped=())

    with pytest.raises(ValueError, match="shared-pool sentinel"):
        build_stage_cuts_payload(
            mapping,
            manifest,
            stage_id=10,
            cost_scale_factor=1.0,
            node_id=-1,
            graph_stage_id=10,
            priced_state_date=manifest.priced_state_date,
        )


def test_build_stage_cuts_payload_carries_keyed_inflow_lag_coefficients() -> None:
    manifest = make_manifest([make_slot(0, 0, 0)])  # one storage slot
    cut = make_mapped_cut(
        coefficients=(0.5,),
        inflow_lag_coefficients={0: (2.0, 3.0, 0.0)},
    )
    payload = build_stage_cuts_payload(
        MappingResult(cuts=(cut,), dropped=()),
        manifest,
        stage_id=10,
        cost_scale_factor=1.0,
        node_id=0,
        graph_stage_id=10,
        priced_state_date=manifest.priced_state_date,
    )
    # The keyed lag coefficients ride alongside the storage-aligned vector, as
    # lists novomodelo's write_policy_checkpoint consumes to place its reserved slots.
    assert payload["cuts"][0]["inflow_lag_coefficients"] == {0: [2.0, 3.0, 0.0]}


def test_build_stage_cuts_payload_empty_lag_coefficients_by_default() -> None:
    manifest = make_manifest([make_slot(0, 0, 0)])
    cut = make_mapped_cut(coefficients=(0.5,))
    payload = build_stage_cuts_payload(
        MappingResult(cuts=(cut,), dropped=()),
        manifest,
        stage_id=10,
        cost_scale_factor=1.0,
        node_id=0,
        graph_stage_id=10,
        priced_state_date=manifest.priced_state_date,
    )
    assert payload["cuts"][0]["inflow_lag_coefficients"] == {}
