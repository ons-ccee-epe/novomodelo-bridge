"""Unit tests for ``novomodelo_bridge.decomp.converters.scalar_parameters``.

Tier-1: pure Python, no ``example/`` deck, no ``import novomodelo``. Exercises the
thin delegation to ``novomodelo.scalar_parameters.build_scalar_parameters``
and the ``constraints/generic_parameters.json`` write round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path

from novomodelo_bridge.decomp.converters.scalar_parameters import (
    build_decomp_scalar_parameters,
    rho_acum_scaled_name,
    write_scalar_parameters,
)
from novomodelo_bridge.novomodelo.case_writer import CaseWriter
from novomodelo_bridge.novomodelo.scalar_parameters import (
    build_scalar_parameters,
)


def test_build_decomp_scalar_parameters_matches_converter() -> None:
    decomp_result = build_decomp_scalar_parameters([0, 1])
    converter_result = build_scalar_parameters([0, 1])

    assert decomp_result == converter_result
    assert len(decomp_result["scalar_parameters"]) == 10
    assert all(e["kind"] == "computed" for e in decomp_result["scalar_parameters"])


def test_write_scalar_parameters_round_trip(tmp_path: Path) -> None:
    params = build_decomp_scalar_parameters([0])
    writer = CaseWriter(tmp_path)

    write_scalar_parameters(writer, params)

    written = tmp_path / "constraints" / "generic_parameters.json"
    assert written.exists()
    assert json.loads(written.read_text()) == params


def test_rho_acum_scaled_name_reexported() -> None:
    assert rho_acum_scaled_name(3) == "rho_acum_scaled_h3"


def test_build_decomp_scalar_parameters_empty_ids_empty_list(
    tmp_path: Path,
) -> None:
    params = build_decomp_scalar_parameters([])
    writer = CaseWriter(tmp_path)

    assert params["scalar_parameters"] == []

    write_scalar_parameters(writer, params)

    written = tmp_path / "constraints" / "generic_parameters.json"
    assert written.exists()
    assert json.loads(written.read_text()) == params
