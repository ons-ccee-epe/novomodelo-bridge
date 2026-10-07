"""Registry-integrity guard for `cobre_schemas.SCHEMA_URLS`/`schema_url_for`.

Every per-track schema-URL constant and inline `$schema` literal has been
repointed at this registry, so there is no live constant left to import here
— the tests below pin the registry's own shape plus a sample of real emit
sites (not a duplicated literal) to prove the emitted value still traces back
to `SCHEMA_URLS`.
"""

from __future__ import annotations

import contextlib
import datetime
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pytest

from cobre_bridge.cobre import schemas as cobre_schemas
from cobre_bridge.newave.id_map import NewaveIdMap
from tests.conftest import _all_converter_patches, _make_fake_newave_dir, make_case

_CANONICAL_PREFIX = (
    "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/schemas/"
)


def test_schema_urls_has_sixteen_entries() -> None:
    assert len(cobre_schemas.SCHEMA_URLS) == 16


def test_schema_url_for_hit() -> None:
    assert cobre_schemas.schema_url_for("system/buses.json") == (
        f"{_CANONICAL_PREFIX}buses.schema.json"
    )


def test_schema_url_for_unregistered_raises_key_error() -> None:
    with pytest.raises(KeyError):
        cobre_schemas.schema_url_for("does/not/exist.json")


def test_all_urls_have_canonical_shape() -> None:
    for relpath, url in cobre_schemas.SCHEMA_URLS.items():
        assert url.startswith(_CANONICAL_PREFIX), relpath
        assert url.endswith(".schema.json"), relpath


# ---------------------------------------------------------------------------
# NEWAVE emit-output pins: the registry is proven the single source by
# calling the real emit site, not by re-declaring the literal it should
# return.
# ---------------------------------------------------------------------------


def _dger_mock() -> MagicMock:
    """A ``dger.dat`` mock sufficient for ``convert_stages``/``convert_config``.

    Deterministic-mode detection short-circuits on ``num_forwards != 1``
    before touching ``case.shist``/``case.cvar``, so neither reader needs a
    fixture here.
    """
    dger = MagicMock()
    dger.mes_inicio_estudo = 1
    dger.ano_inicio_estudo = 2020
    dger.num_anos_estudo = 1
    dger.num_anos_pre_estudo = 0
    dger.num_anos_pos_estudo = 0
    dger.num_forwards = 20
    dger.num_max_iteracoes = 200
    dger.num_series_sinteticas = 500
    dger.taxa_de_desconto = 12.0
    dger.num_aberturas = 10
    dger.cvar = 0
    dger.tipo_execucao = 1
    dger.tipo_simulacao_final = 1
    dger.considera_reamostragem_cenarios = 0
    dger.ano_inicial_historico = 1931
    dger.consideracao_media_anual_afluencias = None
    dger.selecao_de_cortes_forward = 1
    dger.selecao_de_cortes_backward = 1
    dger.ordem_maxima_parp = 6
    dger.impressao_estados_geracao_cortes = None
    return dger


def _patamar_mock() -> MagicMock:
    """A single-block ``patamar.dat`` mock sufficient for ``convert_stages``."""
    rows = [
        {"data": datetime.datetime(2020, month, 1), "patamar": 1, "valor": 1.0}
        for month in range(1, 13)
    ]
    patamar = MagicMock()
    patamar.duracao_mensal_patamares = pd.DataFrame(rows)
    patamar.numero_patamares = 1
    return patamar


def test_stages_json_emit_matches_registry(tmp_path) -> None:
    from cobre_bridge.newave.converters.temporal import convert_stages

    case = make_case(tmp_path, dger=_dger_mock(), patamar=_patamar_mock())
    id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[], thermal_codes=[])

    result = convert_stages(case, id_map)

    assert result["$schema"] == cobre_schemas.SCHEMA_URLS["stages.json"]


def test_config_json_emit_matches_registry(tmp_path) -> None:
    from cobre_bridge.newave.converters.temporal import convert_config

    case = make_case(tmp_path, dger=_dger_mock())

    result = convert_config(case)

    assert result["$schema"] == cobre_schemas.SCHEMA_URLS["config.json"]


def test_generic_constraints_json_emit_matches_registry(tmp_path) -> None:
    """``constraints/generic_constraints.json`` is stamped by ``pipeline.py``'s
    own merge site (not a single converter), so this exercises the real
    pipeline run with every converter mocked except ``convert_vminop_constraints``
    — the minimum needed to reach the merge and get a non-empty file."""
    from cobre_bridge.core.generic_constraint_builder import GENERIC_BOUNDS_SCHEMA
    from cobre_bridge.newave.converters.constraints import VminopResult
    from cobre_bridge.newave.pipeline import convert_newave_case

    src = _make_fake_newave_dir(tmp_path)
    dst = tmp_path / "cobre_case"

    fake_bounds = pa.table(
        {
            "constraint_id": pa.array([], type=pa.int32()),
            "stage_id": pa.array([], type=pa.int32()),
            "block_id": pa.array([], type=pa.int32()),
            "bound_lower": pa.array([], type=pa.float64()),
            "bound_upper": pa.array([], type=pa.float64()),
        },
        schema=GENERIC_BOUNDS_SCHEMA,
    )
    fake_vminop = VminopResult(
        constraints_dict={"$schema": "http://example", "constraints": [{"id": 0}]},
        bounds=fake_bounds,
        referenced_hydro_ids=[],
        rho_acum_overrides={},
    )

    fake_id_map = MagicMock()
    with contextlib.ExitStack() as stack:
        for p in _all_converter_patches(fake_id_map):
            stack.enter_context(p)
        # Override the vminop patch (entered last -> exits first) so the
        # pipeline's merge actually runs and writes the file.
        stack.enter_context(
            patch(
                "cobre_bridge.newave.pipeline.constraints_conv.convert_vminop_constraints",
                return_value=fake_vminop,
            )
        )
        convert_newave_case(src, dst)

    doc = json.loads((dst / "constraints" / "generic_constraints.json").read_text())
    assert (
        doc["$schema"]
        == cobre_schemas.SCHEMA_URLS["constraints/generic_constraints.json"]
    )
