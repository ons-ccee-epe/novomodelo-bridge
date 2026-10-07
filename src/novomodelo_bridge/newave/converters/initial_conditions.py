"""Initial conditions converter: maps the source model initial storage to Novomodelo JSON."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from novomodelo_bridge.newave.case import NewaveCase
from novomodelo_bridge.newave.converters.anticipated import read_anticipated_dispatch
from novomodelo_bridge.newave.converters.hydro import read_cadastro
from novomodelo_bridge.newave.converters.thermal import thermal_generation_bounds
from novomodelo_bridge.newave.id_map import NewaveIdMap
from novomodelo_bridge.newave.plants import filling_hydro_codes
from novomodelo_bridge.novomodelo import schemas as novomodelo_schemas

_LOG = logging.getLogger(__name__)


def _filling_row(exph_df: pd.DataFrame, code: int) -> pd.Series:
    """Return the first ``exph`` schedule row for *code* (the filling-start row).

    Mirrors the selector ``convert_hydros`` uses (``hydro.py`` lines 1036-1039):
    the filling schedule lives on the first ``exph`` row per plant, identified by
    a non-null ``data_inicio_enchimento`` (unit rows carry ``NaT`` there). Callers
    must only invoke this for a ``code`` already admitted by
    :func:`novomodelo_bridge.newave.plants.filling_hydro_codes`, which guarantees the
    predicate selects at least one row (so ``.iloc[0]`` never raises).
    """
    return exph_df.loc[
        (exph_df["codigo_usina"] == code) & exph_df["data_inicio_enchimento"].notna()
    ].iloc[0]


def _delivery_window(start_year: int, start_month: int, k: int) -> tuple[str, str]:
    """ISO ``(start_date, end_date)`` for delivery stage *k* (0-based).

    Delivery stage ``k`` is the monthly study stage ``k`` months after the
    study start; the window is ``[first-of-month, first-of-next-month)`` — the
    exclusive end novomodelo's windowed-record validator expects.
    """
    total = (start_month - 1) + k
    year = start_year + total // 12
    month = total % 12 + 1
    start = date(year, month, 1)
    end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return start.isoformat(), end.isoformat()


def convert_initial_conditions(case: NewaveCase, id_map: NewaveIdMap) -> dict:
    """Convert the source model initial reservoir storage to a Novomodelo initial_conditions
    dict.

    Reads ``hidr.dat`` and ``confhd.dat`` from *case*.  Initial
    storage is derived from ``Confhd.usinas.volume_inicial_percentual``
    (a percentage of ``volume_maximo`` from Hidr).

    Values outside ``[0, 100]`` are clamped with a warning.

    Parameters
    ----------
    case:
        Parsed the source model case.
    id_map:
        Pre-built ID mapping for hydro IDs.

    Raises
    ------
    ValueError
        If a hydro in ``confhd.dat`` references a code absent in
        ``hidr.dat``.
    """
    # Apply permanent MODIF.DAT overrides (notably VOLMIN) so the initial-storage
    # percentage is taken of the *operational* useful volume — the same min the bounds
    # converter uses.  The source model applies ``volume_inicial_percentual`` to the
    # VOLMIN-adjusted useful range, not the raw hidr.dat one (verified against pmo.dat
    # "VOLUME ARMAZENADO INICIAL": I. SOLTEIRA V.INIC 3940.9 hm³ @ 71.70% is 71.70% of
    # (vmax − VOLMIN), not (vmax − raw_min)).
    cadastro = read_cadastro(case)

    existing = case.active_hydros

    # ``NE`` plants carrying an exph dead-volume filling row are seeded into the
    # separate ``filling_storage`` list, never ``storage`` — novomodelo's IC reader
    # rejects a hydro that appears in both arrays.  Computed once here from the same
    # admission predicate ``case.active_hydros`` uses; ``set()`` when there is no
    # filling plant, so the in-loop guard below never fires.
    exph_df = case.exph.expansoes if case.exph is not None else None
    filling = filling_hydro_codes(case.confhd.usinas, exph_df)

    storage: list[dict] = []
    filling_storage: list[dict] = []
    for _, row in existing.iterrows():
        newave_code = int(row["codigo_usina"])
        name = str(row["nome_usina"]).strip()

        if newave_code not in cadastro.index:
            raise ValueError(
                f"Hydro plant '{name}' (code {newave_code}) from confhd.dat"
                f" not found in hidr.dat"
            )

        hreg = cadastro.loc[newave_code]
        vol_min = float(hreg["volume_minimo"])
        vol_max = float(hreg["volume_maximo"])

        # ``NE``-with-filling plant: seed ``filling_storage`` (never ``storage``)
        # with the already-impounded fraction of the dead volume at filling start.
        # ``volume_morto`` (a percentage) lives on the plant's exph schedule row;
        # the seed is that fraction of the hidr ``volume_minimo`` floor.  Routed
        # here *before* the 'S'/'D' anchored branch so a run-of-river filling plant
        # (JURUENA) does not also land in ``storage``.  ``exph_df`` is non-None
        # because ``newave_code in filling`` implies the admission predicate had
        # an exph row to admit it.
        if newave_code in filling:
            assert exph_df is not None  # filling membership implies exph present
            morto = float(_filling_row(exph_df, newave_code)["volume_morto"])
            if morto < 0.0 or morto > 100.0:
                _LOG.warning(
                    "volume_morto for plant '%s' (code %d) is %.2f"
                    " — clamping to [0, 100]",
                    name,
                    newave_code,
                    morto,
                )
                morto = max(0.0, min(100.0, morto))
            filling_storage.append(
                {
                    "hydro_id": id_map.hydro_id(newave_code),
                    # novomodelo-io's ``RawHydroStorage`` keys both ``storage`` and
                    # ``filling_storage`` on ``value_hm3``; this must match the
                    # ``storage`` entries' key below or novomodelo fails to deserialize.
                    "value_hm3": (morto / 100.0) * vol_min,
                }
            )
            continue

        # Fio-d'água plants don't accumulate water across stages, so the
        # bounds converter collapses their storage to a single point in
        # ``hydro.py``.  Anchor the initial storage to that same point so the
        # initial condition stays inside the (collapsed) [min, max] range.
        # Both 'D' (daily) and 'S' (run-of-river) freeze at ``volume_referencia``
        # — the same anchor the reservoir-range collapse uses in ``hydro.py``.
        # When ``volume_referencia`` is absent/NaN the range is not collapsed
        # there, so we skip the anchor here too and fall through to the normal
        # ``volume_inicial_percentual`` seeding below.
        tipo_reg = str(hreg.get("tipo_regulacao", "")).strip()
        vol_ref_raw = hreg.get("volume_referencia")
        anchored: float | None = None
        if (
            tipo_reg in ("D", "S")
            and vol_ref_raw is not None
            and not pd.isna(vol_ref_raw)
        ):
            anchored = float(vol_ref_raw)
        if anchored is not None:
            storage.append(
                {
                    "hydro_id": id_map.hydro_id(newave_code),
                    "value_hm3": anchored,
                }
            )
            continue

        pct = float(row["volume_inicial_percentual"])
        if pct < 0.0 or pct > 100.0:
            _LOG.warning(
                "volume_inicial_percentual for plant '%s' (code %d) is %.2f"
                " — clamping to [0, 100]",
                name,
                newave_code,
                pct,
            )
            pct = max(0.0, min(100.0, pct))

        # Percentage is of useful volume (max - min), not of max.
        value_hm3 = (pct / 100.0) * (vol_max - vol_min) + vol_min

        storage.append(
            {
                "hydro_id": id_map.hydro_id(newave_code),
                "value_hm3": value_hm3,
            }
        )

    storage.sort(key=lambda s: s["hydro_id"])
    filling_storage.sort(key=lambda s: s["hydro_id"])

    # ── Past anticipated thermal commitments (from adterm.dat) ──────────
    # Empty for non-GNL cases (despacho_antecipado_gnl=0 in dger.dat). Each entry maps
    # a thermal's source-model code to its novomodelo thermal_id.
    #
    # Novomodelo honours non-zero pre-horizon seeds: the always-active anticipated "fishing"
    # equality pins generation to the committed MW at each delivery stage. The committed
    # value must lie within the plant's static generation bounds ``[min_mw, max_mw]``
    # (``thermals.json`` / ``novomodelo-io`` semantic validator); an out-of-range seed makes
    # that stage's fishing equality infeasible, so we clamp into range and warn rather
    # than emit a case novomodelo would reject.
    anticipated = read_anticipated_dispatch(case)
    gen_bounds = thermal_generation_bounds(case) if anticipated else {}
    if anticipated:
        start_month = int(case.dger.mes_inicio_estudo)
        start_year = int(case.dger.ano_inicio_estudo)
    past_anticipated_commitments: list[dict] = []
    for newave_code, dispatch in anticipated.items():
        try:
            thermal_id = id_map.thermal_id(newave_code)
        except KeyError:
            _LOG.warning(
                "adterm.dat references thermal code=%d that is absent from "
                "the novomodelo id map; skipping its anticipated commitment.",
                newave_code,
            )
            continue
        lo, hi = gen_bounds.get(newave_code, (float("-inf"), float("inf")))
        seeded = [min(max(v, lo), hi) for v in dispatch.values_mw]
        if any(abs(s - v) > 1e-9 for s, v in zip(seeded, dispatch.values_mw)):
            _LOG.warning(
                "adterm.dat thermal code=%d committed dispatch [%s] MW falls "
                "outside the plant's generation bounds [%.4f, %.4f]; clamping to "
                "[%s] so the plant's committed (anticipated) dispatch stays "
                "within its generation bounds.",
                newave_code,
                ", ".join(f"{v:.4f}" for v in dispatch.values_mw),
                lo,
                hi,
                ", ".join(f"{s:.4f}" for s in seeded),
            )
        # novomodelo 0.14 takes past_anticipated_commitments as windowed records
        # {thermal_id, start_date, end_date, value_mw}, one per leading delivery
        # stage, tiling the plant's lead horizon exactly (strict full coverage,
        # zero-MW stages written explicitly — a missing lead stage is rejected).
        for k, value_mw in enumerate(seeded):
            start_date, end_date = _delivery_window(start_year, start_month, k)
            past_anticipated_commitments.append(
                {
                    "thermal_id": thermal_id,
                    "start_date": start_date,
                    "end_date": end_date,
                    "value_mw": value_mw,
                }
            )
    past_anticipated_commitments.sort(key=lambda c: (c["thermal_id"], c["start_date"]))

    result: dict = {
        "$schema": novomodelo_schemas.schema_url_for("initial_conditions.json"),
        "storage": storage,
        "filling_storage": filling_storage,
    }
    if past_anticipated_commitments:
        result["past_anticipated_commitments"] = past_anticipated_commitments
    return result
