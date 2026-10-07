"""Energy-contract reading for the source model (``CI``/``CE`` registers).

The source model declares external energy contracts in two field-identical
register families: ``CI`` (import — energy flows into the system) and
``CE`` (export — energy flows out). This module turns both into a
structured, deterministically-id'd :class:`Contract` model — the input
every downstream emitter consumes. ``custo`` stays raw and unsigned here;
the export sign convention and any ``fator_perdas`` diagnostics are later
concerns, not this reader's.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa

from cobre_bridge.cobre import schemas as cobre_schemas
from cobre_bridge.core import diagnostics as dx
from cobre_bridge.decomp.converters.thermal import _hours_weighted

if TYPE_CHECKING:
    from collections.abc import Sequence

    from idecomp.decomp import Dadger

    from cobre_bridge.decomp.case import DecompCase
    from cobre_bridge.decomp.id_map import DecompIdMap
    from cobre_bridge.decomp.temporal import OperativeStage

_LOG = logging.getLogger(__name__)

_LOWER_LIMIT_COLUMN = re.compile(r"^limite_inferior_\d+$")
_UPPER_LIMIT_COLUMN = re.compile(r"^limite_superior_\d+$")


@dataclass(frozen=True)
class ContractStage:
    """One stage's per-block bounds, cost, and loss factor for a contract."""

    min_mw: list[float]
    max_mw: list[float]
    custo: list[float]
    loss_factor: float | None


@dataclass(frozen=True)
class Contract:
    """A deterministically-id'd energy contract (``CI``/``CE``)."""

    id: int
    kind: str
    numero: int
    name: str
    bus_code: int
    stages: list[ContractStage]


def _register_label(kind: str) -> str:
    return "CI" if kind == "import" else "CE"


def _is_placeholder_row(row: pd.Series) -> bool:
    """D6: a blank-name row whose limits are all zero/NaN is a placeholder.

    Some decks carry a leftover ``CI``/``CE`` row with no real contract
    behind it — a blank name and every bound at zero. A blank-name row
    with a genuinely non-zero bound is still a real contract; only its
    name gets synthesized.
    """
    if str(row["nome_contrato"]).strip():
        return False
    limit_columns = [
        column
        for column in row.index
        if _LOWER_LIMIT_COLUMN.match(column) or _UPPER_LIMIT_COLUMN.match(column)
    ]
    return all(
        pd.isna(row[column]) or float(row[column]) == 0.0 for column in limit_columns
    )


def read_contracts(
    dadger: Dadger,
    calendar: Sequence[OperativeStage],
) -> list[Contract]:
    """Read ``CI``/``CE`` into a deterministic, densely-id'd contract list.

    Declared per-stage block values are forward-filled over the calendar
    (stage 1 mandatory); a blank-name, all-zero-bound row is a placeholder
    and is skipped entirely. Contracts get dense ids by sorting on
    ``(kind, numero)`` — imports before exports — since the ``CI``/``CE``
    ``numero`` spaces collide and are never a key on their own.
    """
    declared: dict[tuple[str, int], dict] = {}
    for kind, frame in (
        ("import", dadger.ci(df=True)),
        ("export", dadger.ce(df=True)),
    ):
        if frame is None or frame.empty:
            continue
        register = _register_label(kind)
        for _, row in frame.iterrows():
            if _is_placeholder_row(row):
                continue

            numero = int(row["numero_contrato"])
            declared_stage = int(row["estagio"])
            stage_index = declared_stage - 1
            if not 0 <= stage_index < len(calendar):
                raise ValueError(
                    f"{register} {numero}: stage {declared_stage} outside "
                    f"the calendar (1..{len(calendar)})"
                )
            n_blocks = len(calendar[stage_index].block_hours)

            def _blocks(
                prefix: str, row: pd.Series = row, n: int = n_blocks
            ) -> list[float]:
                return [
                    0.0
                    if pd.isna(row[f"{prefix}_{k}"])
                    else float(row[f"{prefix}_{k}"])
                    for k in range(1, n + 1)
                ]

            name = str(row["nome_contrato"]).strip()
            entry = declared.setdefault(
                (kind, numero),
                {
                    "register": register,
                    "name": name,
                    "bus_code": int(row["codigo_submercado"]),
                    "declared": {},
                },
            )
            fator_perdas = row["fator_perdas"]
            entry["declared"][stage_index] = ContractStage(
                min_mw=_blocks("limite_inferior"),
                max_mw=_blocks("limite_superior"),
                custo=_blocks("custo"),
                loss_factor=None if pd.isna(fator_perdas) else float(fator_perdas),
            )

    provisional: list[dict] = []
    for (kind, numero), entry in declared.items():
        register = entry["register"]
        per_stage: dict[int, ContractStage] = entry["declared"]
        if 0 not in per_stage:
            raise ValueError(
                f"{register} {numero} does not declare stage 1; "
                "sparse-stage inheritance has no base"
            )
        dense: list[ContractStage] = []
        for stage in calendar:
            dense.append(
                per_stage[stage.index] if stage.index in per_stage else dense[-1]
            )
        name = entry["name"] or f"{register} {numero}"
        provisional.append(
            {
                "kind": kind,
                "numero": numero,
                "name": name,
                "bus_code": entry["bus_code"],
                "stages": dense,
            }
        )

    provisional.sort(key=lambda c: (0 if c["kind"] == "import" else 1, c["numero"]))
    return [
        Contract(
            id=position,
            kind=c["kind"],
            numero=c["numero"],
            name=c["name"],
            bus_code=c["bus_code"],
            stages=c["stages"],
        )
        for position, c in enumerate(provisional)
    ]


def warn_nonnull_loss_factor(contracts: Sequence[Contract]) -> None:
    """D2: WARN (never fold, never reject) on a non-null, non-zero ``fator_perdas``.

    cobre's ``energy_contracts`` entity has no loss-factor field, so the
    source model's per-stage ``fator_perdas`` is intentionally excluded from
    every emitted number (``price_per_mwh``, ``limits``). A deck that
    declares one anyway is not an error — it is a WARNING, and the WARNING
    is the record; nothing is folded and the deck is never rejected. A
    declared ``0.0`` is treated the same as ``None`` (the expected case) and
    does not trigger this.
    """
    offenders: list[tuple[int, str, float]] = []
    for contract in contracts:
        non_null_non_zero = [
            stage.loss_factor
            for stage in contract.stages
            if stage.loss_factor is not None and stage.loss_factor != 0.0
        ]
        if non_null_non_zero:
            offenders.append((contract.id, contract.name, max(non_null_non_zero)))

    if not offenders:
        return

    dx.emit(
        dx.Diagnostic(
            code="contract-loss-factor-unmapped",
            severity=dx.Severity.WARNING,
            category="Energy contracts",
            title=(
                f"Non-null fator_perdas on {len(offenders)} energy "
                "contract(s) is ignored"
            ),
            summary=(
                "The source model's fator_perdas (loss factor) has no cobre "
                "energy_contracts target; it is ignored — not folded into "
                "price_per_mwh or limits, and does not reject the deck"
            ),
            table=dx.DiagnosticTable(
                columns=["Id", "Name", "fator_perdas %"],
                rows=[
                    [contract_id, name, round(value, 3)]
                    for contract_id, name, value in offenders
                ],
                justify=["right", "left", "right"],
            ),
        ),
        logger=_LOG,
    )


def _signed_price(custo: float, kind: str) -> float:
    """D3: export revenue is negative; import cost stays positive."""
    return -custo if kind == "export" else custo


def convert_energy_contracts(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    contracts: Sequence[Contract],
) -> dict:
    """Build ``energy_contracts.json`` from ``CI``/``CE`` (stage-1 base values).

    D1: every contract is emitted always-active — ``entry_stage_id`` and
    ``exit_stage_id`` are omitted, matching the schema's null default and the
    d41 import-contract shape. Total over ``contracts``: an empty list yields
    an empty ``"contracts"`` array without raising.
    """
    op_date = case.start_date.isoformat()
    first = case.calendar[0]

    out: list[dict] = []
    for c in contracts:
        base = c.stages[0]
        out.append(
            {
                "id": c.id,
                "name": c.name,
                "operational_start_date": op_date,
                "bus_id": id_map.bus_id(c.bus_code),
                "type": c.kind,
                "price_per_mwh": _signed_price(
                    _hours_weighted(base.custo, first), c.kind
                ),
                "limits": {
                    "min_mw": _hours_weighted(base.min_mw, first),
                    "max_mw": _hours_weighted(base.max_mw, first),
                },
            }
        )
    return {
        "$schema": cobre_schemas.schema_url_for("system/energy_contracts.json"),
        "contracts": out,
    }


_CONTRACT_BOUNDS_SCHEMA = pa.schema(
    [
        pa.field("contract_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("min_mw", pa.float64(), nullable=True),
        pa.field("max_mw", pa.float64(), nullable=True),
        pa.field("price_per_mwh", pa.float64(), nullable=True),
        pa.field("block_id", pa.int32(), nullable=True),
    ]
)


def convert_contract_bounds(
    case: DecompCase,
    *,
    contracts: Sequence[Contract],
) -> pa.Table:
    """Contract bounds: a stage-level base row plus sparse per-block overrides.

    Every ``(contract, stage)`` gets a base row (``block_id = None``) carrying
    the hours-weighted ``min_mw``/``max_mw`` and the D3-signed hours-weighted
    ``price_per_mwh``. Unlike ``convert_thermal_bounds`` — whose
    ``cost_per_mwh`` stays stage-level only (cobre rule 37) — all three value
    columns here are block-eligible (decision 6: contract price legitimately
    varies per block), so the uniformity test spans ``min_mw``, ``max_mw``,
    *and* the signed per-block price. Where the stage is not uniform across
    all three, one override row per block is emitted carrying that block's
    exact ``min_mw``/``max_mw`` and signed ``price_per_mwh``. This mirrors
    ``convert_thermal_bounds``' and ``convert_lines``' sparse
    base-plus-override convention exactly.
    """
    calendar = case.calendar
    contract_ids: list[int] = []
    stage_ids: list[int] = []
    mins: list[float] = []
    maxs: list[float] = []
    prices: list[float] = []
    block_ids: list[int | None] = []

    for c in contracts:
        for stage in calendar:
            sv = c.stages[stage.index]
            priced = [_signed_price(x, c.kind) for x in sv.custo]

            contract_ids.append(c.id)
            stage_ids.append(stage.index)
            mins.append(_hours_weighted(sv.min_mw, stage))
            maxs.append(_hours_weighted(sv.max_mw, stage))
            prices.append(_signed_price(_hours_weighted(sv.custo, stage), c.kind))
            block_ids.append(None)

            uniform = (
                all(m == sv.min_mw[0] for m in sv.min_mw)
                and all(m == sv.max_mw[0] for m in sv.max_mw)
                and all(p == priced[0] for p in priced)
            )
            if not uniform:
                for b in range(len(sv.max_mw)):
                    contract_ids.append(c.id)
                    stage_ids.append(stage.index)
                    mins.append(sv.min_mw[b])
                    maxs.append(sv.max_mw[b])
                    prices.append(priced[b])
                    block_ids.append(b)

    return pa.table(
        {
            "contract_id": pa.array(contract_ids, type=pa.int32()),
            "stage_id": pa.array(stage_ids, type=pa.int32()),
            "min_mw": pa.array(mins, type=pa.float64()),
            "max_mw": pa.array(maxs, type=pa.float64()),
            "price_per_mwh": pa.array(prices, type=pa.float64()),
            "block_id": pa.array(block_ids, type=pa.int32()),
        },
        schema=_CONTRACT_BOUNDS_SCHEMA,
    )
