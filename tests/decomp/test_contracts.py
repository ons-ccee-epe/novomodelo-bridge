"""Tests for the DECOMP energy-contract reader and emitter (``CI``/``CE``)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import jsonschema
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.decomp.case import DecompCase
from novomodelo_bridge.decomp.converters.contracts import (
    _CONTRACT_BOUNDS_SCHEMA,
    Contract,
    ContractStage,
    _signed_price,
    convert_contract_bounds,
    convert_energy_contracts,
    read_contracts,
    warn_nonnull_loss_factor,
)
from novomodelo_bridge.decomp.id_map import DecompIdMap
from novomodelo_bridge.decomp.temporal import OperativeStage, build_operative_calendar
from novomodelo_bridge.novomodelo import schemas as novomodelo_schemas
from tests.conftest import make_decomp_case

# Vendored from the novomodelo repository (``schemas/`` and the
# ``d41-energy-contracts`` deterministic example) at the paired novomodelo release;
# refresh both when MIN_NOVOMODELO_VERSION moves.
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_NOVOMODELO_SCHEMA = _FIXTURES / "novomodelo_schemas" / "energy_contracts.schema.json"
_D41_DIR = _FIXTURES / "novomodelo_d41"
_D41_REQUIRED_CONTRACT_KEYS = frozenset(
    {
        "id",
        "name",
        "operational_start_date",
        "bus_id",
        "type",
        "price_per_mwh",
        "limits",
    }
)


class _StubDadger:
    """Minimal Dadger stand-in carrying ci/ce DataFrames."""

    def __init__(
        self,
        ci: pd.DataFrame | None = None,
        ce: pd.DataFrame | None = None,
    ) -> None:
        self._ci, self._ce = ci, ce

    def ci(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._ci

    def ce(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._ce


def _calendar() -> list[OperativeStage]:
    hours = [[15.0, 64.0, 89.0]] * 2 + [[63.0, 280.0, 401.0]]
    return build_operative_calendar(date(2026, 7, 18), hours)


def _uniform_calendar() -> list[OperativeStage]:
    """A calendar whose per-stage blocks all carry equal hours."""
    hours = [[56.0, 56.0, 56.0]] * 2 + [[248.0, 248.0, 248.0]]
    return build_operative_calendar(date(2026, 7, 18), hours)


def _bus_id_map() -> DecompIdMap:
    return DecompIdMap(bus_codes=(1, 2), bus_names=("SE", "S"))


def _case(calendar: list[OperativeStage]) -> DecompCase:
    return make_decomp_case(Path("unused"), calendar=calendar)


# ``decomp/pipeline.py`` no longer exposes module-level
# ``_write_json``/``_write_parquet`` helpers (they became dry-run-aware
# closures local to ``_convert_decomp_case_impl``), so
# ``test_synthetic_contracts_validate_in_a_real_case`` below carries its own
# minimal writers to overwrite the two files after a real conversion. The
# case's ``system/``/``constraints/`` directories already exist post-
# conversion, so neither writer needs to create them.
def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def _write_parquet(path: Path, table: pa.Table) -> None:
    pq.write_table(table, path, compression="zstd")


def _contract_row(
    numero: int,
    estagio: int,
    *,
    nome: str = "CONTRATO",
    bus_code: int = 1,
    min_mw: tuple[float, float, float] = (0.0, 0.0, 0.0),
    max_mw: tuple[float, float, float] = (0.0, 0.0, 0.0),
    custo: tuple[float, float, float] = (0.0, 0.0, 0.0),
    fator_perdas: float | None = None,
) -> dict:
    row: dict = {
        "numero_contrato": numero,
        "codigo_submercado": bus_code,
        "nome_contrato": nome,
        "estagio": estagio,
        "fator_perdas": fator_perdas,
    }
    for k in range(3):
        row[f"limite_inferior_{k + 1}"] = min_mw[k]
        row[f"limite_superior_{k + 1}"] = max_mw[k]
        row[f"custo_{k + 1}"] = custo[k]
    return row


def _contracts_stub() -> _StubDadger:
    """Shared integrated fixture: CI 1 (stages 1+3) + CI 9 placeholder + CE 1.

    ``CI`` numero 1 is declared at stages 1 and 3 only, so stage 2 is
    forward-filled per D1; stage 1's limits are block-uniform and stage 3's
    ``limite_superior`` is non-uniform, to exercise the per-block override
    rows. ``CE`` numero 1 carries a non-zero ``custo`` (D3 sign) and
    deliberately reuses the same ``numero_contrato`` as the ``CI`` row, to
    genuinely exercise D5's ``(kind, numero)`` id assignment. ``CI`` numero 9
    is a blank-name, all-zero-bound placeholder (D6).
    """
    ci = pd.DataFrame(
        [
            _contract_row(
                1,
                1,
                nome="IMPORTACAO",
                bus_code=1,
                min_mw=(1.0, 1.0, 1.0),
                max_mw=(50.0, 50.0, 50.0),
                custo=(200.0, 200.0, 200.0),
            ),
            _contract_row(
                1,
                3,
                nome="IMPORTACAO",
                bus_code=1,
                min_mw=(2.0, 2.0, 2.0),
                max_mw=(40.0, 50.0, 60.0),
                custo=(210.0, 210.0, 210.0),
            ),
            _contract_row(
                9,
                1,
                nome="",
                bus_code=1,
                min_mw=(0.0, 0.0, 0.0),
                max_mw=(0.0, 0.0, 0.0),
                custo=(0.0, 0.0, 0.0),
            ),
        ]
    )
    ce = pd.DataFrame(
        [
            _contract_row(
                1,
                1,
                nome="EXPORTACAO",
                bus_code=2,
                min_mw=(0.0, 0.0, 0.0),
                max_mw=(30.0, 30.0, 30.0),
                custo=(150.0, 150.0, 150.0),
            )
        ]
    )
    return _StubDadger(ci=ci, ce=ce)


def test_read_contracts_forward_fills_and_assigns_dense_ids() -> None:
    ci = pd.DataFrame(
        [
            _contract_row(
                1,
                1,
                nome="IMPORTACAO",
                min_mw=(1.0, 1.0, 1.0),
                max_mw=(10.0, 10.0, 10.0),
                custo=(5.0, 5.0, 5.0),
            ),
            _contract_row(
                1,
                3,
                nome="IMPORTACAO",
                min_mw=(2.0, 2.0, 2.0),
                max_mw=(20.0, 20.0, 20.0),
                custo=(6.0, 6.0, 6.0),
            ),
        ]
    )
    ce = pd.DataFrame(
        [
            _contract_row(
                1,
                1,
                nome="EXPORTACAO",
                min_mw=(0.5, 0.5, 0.5),
                max_mw=(8.0, 8.0, 8.0),
                custo=(4.0, 4.0, 4.0),
            )
        ]
    )
    dadger = _StubDadger(ci=ci, ce=ce)

    contracts = read_contracts(dadger, _calendar())

    assert len(contracts) == 2
    imported = next(c for c in contracts if c.kind == "import")
    exported = next(c for c in contracts if c.kind == "export")
    assert imported.id == 0
    assert exported.id == 1
    assert imported.numero == 1
    assert exported.numero == 1

    # Stage-2 (index 1) is absent in the declared data; it is forward-filled
    # from stage 1 (index 0). Stage 3 (index 2) carries its own declared
    # values, distinct from stage 1's.
    assert imported.stages[1] == imported.stages[0]
    assert imported.stages[2] != imported.stages[0]
    assert imported.stages[2].max_mw == [20.0, 20.0, 20.0]
    assert imported.stages[2].min_mw == [2.0, 2.0, 2.0]
    assert imported.stages[2].custo == [6.0, 6.0, 6.0]


def test_read_contracts_skips_blank_all_zero_placeholder() -> None:
    ci = pd.DataFrame(
        [
            _contract_row(
                1,
                1,
                nome="",
                min_mw=(0.0, 0.0, 0.0),
                max_mw=(0.0, 0.0, 0.0),
            )
        ]
    )
    dadger = _StubDadger(ci=ci, ce=None)

    assert read_contracts(dadger, _calendar()) == []


def test_read_contracts_synthesizes_name_when_blank_but_nonzero() -> None:
    ci = pd.DataFrame(
        [
            _contract_row(
                2,
                1,
                nome="",
                min_mw=(0.0, 0.0, 0.0),
                max_mw=(5.0, 0.0, 0.0),
            )
        ]
    )
    dadger = _StubDadger(ci=ci, ce=None)

    contracts = read_contracts(dadger, _calendar())

    assert len(contracts) == 1
    assert contracts[0].name == "CI 2"


def test_read_contracts_missing_stage_one_raises() -> None:
    ci = pd.DataFrame(
        [
            _contract_row(
                5,
                2,
                nome="SOMENTE ESTAGIO 2",
                max_mw=(10.0, 10.0, 10.0),
            )
        ]
    )
    dadger = _StubDadger(ci=ci, ce=None)

    with pytest.raises(ValueError, match=r"CI 5"):
        read_contracts(dadger, _calendar())


def test_signed_price_negates_export_only() -> None:
    assert _signed_price(100.0, "import") == 100.0
    assert _signed_price(100.0, "export") == -100.0


def test_convert_energy_contracts_import_base_shape() -> None:
    calendar = _uniform_calendar()
    id_map = _bus_id_map()
    contract = Contract(
        id=0,
        kind="import",
        numero=1,
        name="IMPORTACAO",
        bus_code=1,
        stages=[
            ContractStage(
                min_mw=[0.0, 0.0, 0.0],
                max_mw=[50.0, 50.0, 50.0],
                custo=[200.0, 200.0, 200.0],
                loss_factor=None,
            )
        ],
    )

    case = _case(calendar)
    result = convert_energy_contracts(case, id_map, contracts=[contract])

    assert result["contracts"][0] == {
        "id": 0,
        "name": "IMPORTACAO",
        "operational_start_date": calendar[0].start_date.isoformat(),
        "bus_id": 0,
        "type": "import",
        "price_per_mwh": 200.0,
        "limits": {"min_mw": 0.0, "max_mw": 50.0},
    }
    assert "entry_stage_id" not in result["contracts"][0]
    assert "exit_stage_id" not in result["contracts"][0]


def test_convert_energy_contracts_export_price_is_negative() -> None:
    calendar = _uniform_calendar()
    id_map = _bus_id_map()
    contract = Contract(
        id=1,
        kind="export",
        numero=1,
        name="EXPORTACAO",
        bus_code=1,
        stages=[
            ContractStage(
                min_mw=[0.0, 0.0, 0.0],
                max_mw=[30.0, 30.0, 30.0],
                custo=[150.0, 150.0, 150.0],
                loss_factor=None,
            )
        ],
    )

    case = _case(calendar)
    result = convert_energy_contracts(case, id_map, contracts=[contract])

    assert result["contracts"][0]["price_per_mwh"] == -150.0


def test_convert_energy_contracts_validates_against_schema() -> None:
    calendar = _uniform_calendar()
    id_map = _bus_id_map()
    contracts = [
        Contract(
            id=0,
            kind="import",
            numero=1,
            name="IMPORTACAO",
            bus_code=1,
            stages=[
                ContractStage(
                    min_mw=[0.0, 0.0, 0.0],
                    max_mw=[50.0, 50.0, 50.0],
                    custo=[200.0, 200.0, 200.0],
                    loss_factor=None,
                )
            ],
        ),
        Contract(
            id=1,
            kind="export",
            numero=1,
            name="EXPORTACAO",
            bus_code=2,
            stages=[
                ContractStage(
                    min_mw=[0.0, 0.0, 0.0],
                    max_mw=[30.0, 30.0, 30.0],
                    custo=[150.0, 150.0, 150.0],
                    loss_factor=None,
                )
            ],
        ),
    ]

    case = _case(calendar)
    result = convert_energy_contracts(case, id_map, contracts=contracts)

    schema = json.loads(_NOVOMODELO_SCHEMA.read_text(encoding="utf-8"))
    jsonschema.validate(result, schema)


def test_convert_energy_contracts_empty_is_total() -> None:
    calendar = _uniform_calendar()
    id_map = _bus_id_map()
    case = _case(calendar)

    result = convert_energy_contracts(case, id_map, contracts=[])

    assert result == {
        "$schema": novomodelo_schemas.schema_url_for("system/energy_contracts.json"),
        "contracts": [],
    }


def test_contract_bounds_uniform_emits_base_rows_only() -> None:
    calendar = _uniform_calendar()
    stage = ContractStage(
        min_mw=[0.0, 0.0, 0.0],
        max_mw=[50.0, 50.0, 50.0],
        custo=[200.0, 200.0, 200.0],
        loss_factor=None,
    )
    contract = Contract(
        id=0,
        kind="import",
        numero=1,
        name="IMPORTACAO",
        bus_code=1,
        stages=[stage, stage, stage],
    )

    table = convert_contract_bounds(_case(calendar), contracts=[contract])

    assert table.num_rows == 3
    assert table.column("block_id").null_count == 3
    for row in table.to_pylist():
        assert row["min_mw"] == 0.0
        assert row["max_mw"] == 50.0
        assert row["price_per_mwh"] == 200.0


def test_contract_bounds_nonuniform_max_emits_per_block_rows() -> None:
    calendar = _uniform_calendar()
    nonuniform_stage = ContractStage(
        min_mw=[0.0, 0.0, 0.0],
        max_mw=[40.0, 50.0, 60.0],
        custo=[200.0, 200.0, 200.0],
        loss_factor=None,
    )
    uniform_stage = ContractStage(
        min_mw=[0.0, 0.0, 0.0],
        max_mw=[50.0, 50.0, 50.0],
        custo=[200.0, 200.0, 200.0],
        loss_factor=None,
    )
    contract = Contract(
        id=0,
        kind="import",
        numero=1,
        name="IMPORTACAO",
        bus_code=1,
        stages=[nonuniform_stage, uniform_stage, uniform_stage],
    )

    table = convert_contract_bounds(_case(calendar), contracts=[contract])
    stage0 = [row for row in table.to_pylist() if row["stage_id"] == 0]

    assert len(stage0) == 4
    base = next(row for row in stage0 if row["block_id"] is None)
    assert base["max_mw"] == 50.0  # hours-weighted, uniform-hours calendar
    overrides = sorted(
        (row for row in stage0 if row["block_id"] is not None),
        key=lambda r: r["block_id"],
    )
    assert [o["block_id"] for o in overrides] == [0, 1, 2]
    assert [o["max_mw"] for o in overrides] == [40.0, 50.0, 60.0]


def test_contract_bounds_export_price_negated_per_block() -> None:
    calendar = _uniform_calendar()
    nonuniform_stage = ContractStage(
        min_mw=[0.0, 0.0, 0.0],
        max_mw=[30.0, 30.0, 30.0],
        custo=[100.0, 200.0, 300.0],
        loss_factor=None,
    )
    uniform_stage = ContractStage(
        min_mw=[0.0, 0.0, 0.0],
        max_mw=[30.0, 30.0, 30.0],
        custo=[150.0, 150.0, 150.0],
        loss_factor=None,
    )
    contract = Contract(
        id=1,
        kind="export",
        numero=1,
        name="EXPORTACAO",
        bus_code=1,
        stages=[nonuniform_stage, uniform_stage, uniform_stage],
    )

    table = convert_contract_bounds(_case(calendar), contracts=[contract])
    overrides = sorted(
        (
            row
            for row in table.to_pylist()
            if row["stage_id"] == 0 and row["block_id"] is not None
        ),
        key=lambda r: r["block_id"],
    )

    assert [o["price_per_mwh"] for o in overrides] == [-100.0, -200.0, -300.0]


def test_contract_bounds_schema_roundtrips_through_parquet(tmp_path: Path) -> None:
    calendar = _uniform_calendar()
    stage = ContractStage(
        min_mw=[0.0, 0.0, 0.0],
        max_mw=[50.0, 50.0, 50.0],
        custo=[200.0, 200.0, 200.0],
        loss_factor=None,
    )
    contract = Contract(
        id=0,
        kind="import",
        numero=1,
        name="IMPORTACAO",
        bus_code=1,
        stages=[stage, stage, stage],
    )
    table = convert_contract_bounds(_case(calendar), contracts=[contract])
    out = tmp_path / "contract_bounds.parquet"

    pq.write_table(table, out)
    read_back = pq.read_table(out)

    assert read_back.schema.names == [
        "contract_id",
        "stage_id",
        "min_mw",
        "max_mw",
        "price_per_mwh",
        "block_id",
    ]
    assert [str(field_type) for field_type in read_back.schema.types] == [
        "int32",
        "int32",
        "double",
        "double",
        "double",
        "int32",
    ]


def test_contract_bounds_empty_is_total() -> None:
    calendar = _uniform_calendar()

    table = convert_contract_bounds(_case(calendar), contracts=[])

    assert table.num_rows == 0
    assert table.schema == _CONTRACT_BOUNDS_SCHEMA


def test_loss_factor_all_null_emits_nothing() -> None:
    contract = Contract(
        id=0,
        kind="import",
        numero=1,
        name="X",
        bus_code=1,
        stages=[
            ContractStage(
                min_mw=[0.0, 0.0, 0.0],
                max_mw=[10.0, 10.0, 10.0],
                custo=[5.0, 5.0, 5.0],
                loss_factor=None,
            ),
            ContractStage(
                min_mw=[0.0, 0.0, 0.0],
                max_mw=[10.0, 10.0, 10.0],
                custo=[5.0, 5.0, 5.0],
                loss_factor=None,
            ),
        ],
    )

    with dx.collect() as collected:
        warn_nonnull_loss_factor([contract])

    assert collected == []


def test_loss_factor_nonnull_emits_one_warning() -> None:
    contract = Contract(
        id=0,
        kind="import",
        numero=1,
        name="X",
        bus_code=1,
        stages=[
            ContractStage(
                min_mw=[0.0, 0.0, 0.0],
                max_mw=[10.0, 10.0, 10.0],
                custo=[5.0, 5.0, 5.0],
                loss_factor=3.5,
            )
        ],
    )

    with dx.collect() as collected:
        warn_nonnull_loss_factor([contract])

    assert len(collected) == 1
    diagnostic = collected[0]
    assert diagnostic.severity is dx.Severity.WARNING
    assert diagnostic.code == "contract-loss-factor-unmapped"
    assert diagnostic.table is not None
    assert any(row[0] == 0 and row[1] == "X" for row in diagnostic.table.rows)


def test_loss_factor_zero_is_treated_as_null() -> None:
    contract = Contract(
        id=0,
        kind="import",
        numero=1,
        name="X",
        bus_code=1,
        stages=[
            ContractStage(
                min_mw=[0.0, 0.0, 0.0],
                max_mw=[10.0, 10.0, 10.0],
                custo=[5.0, 5.0, 5.0],
                loss_factor=0.0,
            )
        ],
    )

    with dx.collect() as collected:
        warn_nonnull_loss_factor([contract])

    assert collected == []


def test_loss_factor_is_not_folded_into_numbers() -> None:
    calendar = _uniform_calendar()
    id_map = _bus_id_map()

    def _contract(loss_factor: float | None) -> Contract:
        return Contract(
            id=0,
            kind="import",
            numero=1,
            name="IMPORTACAO",
            bus_code=1,
            stages=[
                ContractStage(
                    min_mw=[0.0, 0.0, 0.0],
                    max_mw=[50.0, 50.0, 50.0],
                    custo=[200.0, 200.0, 200.0],
                    loss_factor=loss_factor,
                )
            ],
        )

    case = _case(calendar)
    with_loss_factor = convert_energy_contracts(
        case, id_map, contracts=[_contract(5.0)]
    )
    without_loss_factor = convert_energy_contracts(
        case, id_map, contracts=[_contract(None)]
    )

    assert with_loss_factor == without_loss_factor


# --- Integrated golden-fixture suite -----------------------------------
#
# The narrow tests above exercise read_contracts/convert_energy_contracts/
# convert_contract_bounds one at a time. These tests drive the same
# _contracts_stub() fixture through the full reader -> both-emitters flow
# in one shot, over a single 3-stage calendar with known block hours.


def test_integrated_ids_and_placeholder_and_window() -> None:
    calendar = _calendar()
    id_map = _bus_id_map()
    dadger = _contracts_stub()

    contracts = read_contracts(dadger, calendar)
    result = convert_energy_contracts(_case(calendar), id_map, contracts=contracts)

    assert len(result["contracts"]) == 2
    by_id = {entry["id"]: entry for entry in result["contracts"]}
    assert by_id[0]["type"] == "import"
    assert by_id[1]["type"] == "export"

    # D6: the blank-name, all-zero CI numero 9 placeholder never becomes a
    # contract, so it cannot surface in the json output either.
    assert not any(c.numero == 9 for c in contracts)

    # D1: every contract is emitted always-active.
    for entry in result["contracts"]:
        assert "entry_stage_id" not in entry
        assert "exit_stage_id" not in entry


def test_integrated_export_sign_and_bus_mapping() -> None:
    calendar = _calendar()
    id_map = _bus_id_map()
    dadger = _contracts_stub()

    contracts = read_contracts(dadger, calendar)
    result = convert_energy_contracts(_case(calendar), id_map, contracts=contracts)

    export_contract = next(c for c in contracts if c.kind == "export")
    export_entry = next(e for e in result["contracts"] if e["type"] == "export")

    assert export_entry["price_per_mwh"] < 0
    assert export_entry["bus_id"] == id_map.bus_id(export_contract.bus_code)


def test_integrated_carry_forward_and_per_block_sparsity() -> None:
    calendar = _calendar()
    dadger = _contracts_stub()

    contracts = read_contracts(dadger, calendar)
    table = convert_contract_bounds(_case(calendar), contracts=contracts)

    import_contract = next(c for c in contracts if c.kind == "import")
    import_rows = [
        row for row in table.to_pylist() if row["contract_id"] == import_contract.id
    ]
    base_by_stage = {
        row["stage_id"]: row for row in import_rows if row["block_id"] is None
    }

    # D1 carry-forward: the forward-filled stage-2 (stage_id==1) base row is
    # identical to its declared stage-1 (stage_id==0) base row.
    assert base_by_stage[0]["min_mw"] == base_by_stage[1]["min_mw"]
    assert base_by_stage[0]["max_mw"] == base_by_stage[1]["max_mw"]
    assert base_by_stage[0]["price_per_mwh"] == base_by_stage[1]["price_per_mwh"]

    # Uniform stages (1 and 2, i.e. stage_id 0 and 1) emit base-only rows.
    overrides_stage0 = [
        row
        for row in import_rows
        if row["stage_id"] == 0 and row["block_id"] is not None
    ]
    overrides_stage1 = [
        row
        for row in import_rows
        if row["stage_id"] == 1 and row["block_id"] is not None
    ]
    assert overrides_stage0 == []
    assert overrides_stage1 == []

    # D4: the non-uniform declared stage 3 (stage_id==2) emits per-block
    # override rows for every block.
    overrides_stage2 = sorted(
        (
            row
            for row in import_rows
            if row["stage_id"] == 2 and row["block_id"] is not None
        ),
        key=lambda row: row["block_id"],
    )
    assert [o["block_id"] for o in overrides_stage2] == [0, 1, 2]


def test_integrated_json_schema_and_parquet_roundtrip(tmp_path: Path) -> None:
    calendar = _calendar()
    id_map = _bus_id_map()
    dadger = _contracts_stub()

    contracts = read_contracts(dadger, calendar)
    case = _case(calendar)
    result = convert_energy_contracts(case, id_map, contracts=contracts)
    table = convert_contract_bounds(case, contracts=contracts)

    schema = json.loads(_NOVOMODELO_SCHEMA.read_text(encoding="utf-8"))
    jsonschema.validate(result, schema)

    out = tmp_path / "contract_bounds.parquet"
    pq.write_table(table, out)
    read_back = pq.read_table(out)

    assert read_back.schema.names == [
        "contract_id",
        "stage_id",
        "min_mw",
        "max_mw",
        "price_per_mwh",
        "block_id",
    ]
    assert [str(field_type) for field_type in read_back.schema.types] == [
        "int32",
        "int32",
        "double",
        "double",
        "double",
        "int32",
    ]


# --- End-to-end d41 round-trip (vendored fixture) --------------------------


def test_d41_json_shape_roundtrip() -> None:
    """Every d41 ``RawContract`` required key must also be present in the
    bridge's emitted contracts for an equivalent import+export pair.

    Value identity is intentionally not asserted: d41's export contract
    carries ``entry_stage_id``/``exit_stage_id`` (a commissioning window),
    while the bridge always emits D1's always-active window (neither key at
    all) — both are legal shapes under the same required-key set.
    """
    d41_doc = json.loads(
        (_D41_DIR / "system" / "energy_contracts.json").read_text(encoding="utf-8")
    )
    assert d41_doc["contracts"], "the d41 fixture must carry at least one contract"
    for entry in d41_doc["contracts"]:
        assert _D41_REQUIRED_CONTRACT_KEYS.issubset(entry.keys())

    calendar = _uniform_calendar()
    id_map = _bus_id_map()
    contracts = [
        Contract(
            id=0,
            kind="import",
            numero=1,
            name="Import equivalent",
            bus_code=1,
            stages=[
                ContractStage(
                    min_mw=[0.0, 0.0, 0.0],
                    max_mw=[50.0, 50.0, 50.0],
                    custo=[200.0, 200.0, 200.0],
                    loss_factor=None,
                )
            ],
        ),
        Contract(
            id=1,
            kind="export",
            numero=2,
            name="Export equivalent",
            bus_code=1,
            stages=[
                ContractStage(
                    min_mw=[0.0, 0.0, 0.0],
                    max_mw=[30.0, 30.0, 30.0],
                    custo=[150.0, 150.0, 150.0],
                    loss_factor=None,
                )
            ],
        ),
    ]

    result = convert_energy_contracts(_case(calendar), id_map, contracts=contracts)

    assert len(result["contracts"]) == 2
    for entry in result["contracts"]:
        assert _D41_REQUIRED_CONTRACT_KEYS.issubset(entry.keys())


def test_d41_parquet_schema_superset() -> None:
    """The bridge's ``contract_bounds`` schema is a superset of d41's
    hand-built 5-column shape, with identical arrow types on the shared
    columns.

    d41's parquet has no ``block_id`` column; the bridge's always does (a
    legal superset per decision 6), so value identity is not asserted here —
    only column presence and shared-column types.
    """
    d41_table = pq.read_table(_D41_DIR / "constraints" / "contract_bounds.parquet")

    calendar = _uniform_calendar()
    stage = ContractStage(
        min_mw=[0.0, 0.0, 0.0],
        max_mw=[50.0, 50.0, 50.0],
        custo=[200.0, 200.0, 200.0],
        loss_factor=None,
    )
    contract = Contract(
        id=0,
        kind="import",
        numero=1,
        name="IMPORTACAO",
        bus_code=1,
        stages=[stage, stage, stage],
    )
    bridge_table = convert_contract_bounds(_case(calendar), contracts=[contract])

    d41_names = set(d41_table.schema.names)
    bridge_names = set(bridge_table.schema.names)
    assert d41_names.issubset(bridge_names), (
        f"bridge columns {bridge_names} must be a superset of d41's {d41_names}"
    )
    for name in ["contract_id", "stage_id", "min_mw", "max_mw", "price_per_mwh"]:
        assert bridge_table.schema.field(name).type == d41_table.schema.field(name).type
