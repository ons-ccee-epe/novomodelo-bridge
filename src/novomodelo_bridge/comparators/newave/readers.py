"""The source model output file readers for results comparison.

Reads MEDIAS CSV files (hydro, thermal, system) directly from the case
directory and pmo.dat convergence/productivity data.

MEDIAS files are parsed with Polars, not inewave's MEDIAS readers: those drop
the rows whose entity code is 0 or less and the last column of a line that
does not end in a comma. pmo.dat is read via ``inewave.newave.Pmo``.

Reader-failure contract: every file this module reads (MEDIAS CSVs,
pmo.dat, sistema.dat, newave.tim, the FPHA reports, the nwlistop
tables) is an OPTIONAL
input, and this module contains no ``raise`` statement — an absent
input AND a present-but-unparseable input both degrade to a typed-empty
frame (or ``None``, for the pmo/FPHA readers) plus a ``_LOG.warning``.
This is a genuine behaviour difference from ``novomodelo_readers``/
``decomp_readers``, which raise (``NovomodeloReadError``/``ValueError``/
``FileNotFoundError``) on a present-but-corrupt file or an absent
required one: an empty frame from real-but-broken data can fabricate a
false zero-vs-zero match (``.claude/rules/comments.md`` §4), but every
caller of this module already treats an empty/None result as "nothing
to compare here" rather than "no divergence found", so the softer
contract is safe for this module's inputs specifically.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

from novomodelo_bridge.core.pandas_utils import as_float, as_timestamp, is_na
from novomodelo_bridge.core.paths import find_case_insensitive
from novomodelo_bridge.newave.files import NewaveFiles

if TYPE_CHECKING:
    from inewave.nwlistop import ArquivoComValores

_LOG = logging.getLogger(__name__)


def _find_pmo(newave_dir: Path) -> Path | None:
    """Locate ``pmo.dat`` directly in the source model case directory."""
    return find_case_insensitive(newave_dir, "pmo.dat")


# -------------------------------------------------------------------
# MEDIAS CSV readers
# -------------------------------------------------------------------

# MEDIAS CSV structure (semicolon-separated):
#   SÉRIE/CENÁRIO ; USINA ; PATAMAR ; EST001 ; EST002 ; ...
# We filter for PATAMAR=0 (full-stage average) and pivot from wide
# to long format: (newave_code, stage, value).


def _read_medias_csv(
    case_dir: Path,
    filename: str,
    variable_filter: str | None = None,
) -> pl.DataFrame:
    """Read a MEDIAS CSV and return long-format DataFrame.

    Returns DataFrame with columns:
    - ``newave_code`` (Int64): entity code (usina or submercado)
    - ``stage`` (Int64): 1-indexed stage number
    - ``variable`` (Utf8): variable name from CSV (if multi-variable)
    - ``value`` (Float64): stage mean value

    Returns an empty DataFrame with the correct schema if the file
    is not found.
    """
    empty = pl.DataFrame(
        schema={
            "newave_code": pl.Int64,
            "stage": pl.Int64,
            "variable": pl.Utf8,
            "value": pl.Float64,
        }
    )

    path = find_case_insensitive(case_dir, filename)
    if path is None:
        _LOG.warning("%s not found in %s", filename, case_dir)
        return empty

    # Try comma first (the source model v29+ format), fall back to semicolon.
    for sep in (",", ";"):
        try:
            df = pl.read_csv(
                path,
                separator=sep,
                truncate_ragged_lines=True,
                infer_schema_length=100,
            )
            if len(df.columns) > 2:
                break
        except Exception:
            continue
    else:
        _LOG.warning("Failed to parse %s", path)
        return empty

    df = df.rename({c: c.strip() for c in df.columns})

    # Identify entity column, variable column, patamar column, and stage
    # value columns.
    #
    # The source model MEDIAS files come in two layouts:
    #   v29+:  USIH_ext, VAR, 3, 4, 5, ...   (comma-sep, no PATAMAR/SERIE)
    #   older: SERIE; USINA; PATAMAR; EST001; EST002; ...  (semicolon-sep)
    entity_col: str | None = None
    patamar_col: str | None = None
    variable_col: str | None = None
    stage_cols: list[str] = []

    for col in df.columns:
        upper = col.upper()
        # v29+ entity columns: USIH_ext, USIT_ext, SBM_ext, etc.
        if upper.endswith("_EXT") or upper in (
            "USINA",
            "CODIGO_USINA",
            "SUBMERCADO",
            "CODIGO_SUBMERCADO",
        ):
            entity_col = col
        elif upper in ("PATAMAR", "PAT"):
            patamar_col = col
        elif upper in ("SERIE", "CENARIO", "SÉRIE/CENÁRIO", "SERIE/CENARIO"):
            pass  # skip scenario column
        elif upper in ("VAR", "VARIAVEL"):
            variable_col = col
        elif upper.startswith("EST") or upper.startswith("ESTAGIO"):
            stage_cols.append(col)

    if entity_col is None:
        # Fallback: first column is usually the entity code.
        entity_col = df.columns[0]

    # If no EST* columns found, look for bare-integer column names (v29+).
    if not stage_cols:
        exclude = {entity_col, patamar_col, variable_col}
        for col in df.columns:
            if col in exclude:
                continue
            # Bare integer column names like "3", "4", ..., "60".
            stripped = col.strip()
            if stripped.isdigit():
                stage_cols.append(col)
                continue
            # Also accept numeric-typed columns as fallback.
            if df[col].dtype in (pl.Float64, pl.Int64, pl.Float32):
                stage_cols.append(col)

    if not stage_cols:
        _LOG.warning("No stage columns found in %s", filename)
        return empty

    # Filter patamar=0 (stage-level average) — only present in older format.
    if patamar_col is not None:
        df = df.filter(pl.col(patamar_col).cast(pl.Int64, strict=False) == 0)

    # Filter to mean scenario — only present in older format.
    serie_col: str | None = None
    for col in df.columns:
        upper = col.upper()
        if upper in ("SERIE", "CENARIO", "SÉRIE/CENÁRIO", "SERIE/CENARIO"):
            serie_col = col
            break

    if serie_col is not None:
        df = df.filter(pl.col(serie_col).cast(pl.Int64, strict=False) == 1)

    if variable_filter is not None and variable_col is not None:
        df = df.filter(
            pl.col(variable_col).str.strip_chars().str.to_uppercase() == variable_filter
        )

    id_cols = [entity_col]
    if variable_col is not None:
        id_cols.append(variable_col)

    try:
        long_df = df.unpivot(
            on=stage_cols,
            index=id_cols,
            variable_name="stage_col",
            value_name="value",
        )
    except Exception:
        _LOG.warning("Failed to unpivot %s", filename)
        return empty

    # Extract stage number from column name.
    # For bare-integer names ("3", "4") this extracts the number directly.
    # For EST001-style names it strips the prefix.
    long_df = long_df.with_columns(
        pl.col("stage_col")
        .str.replace_all(r"[^0-9]", "")
        .cast(pl.Int64, strict=False)
        .alias("stage")
    )

    # Build output.  Strip whitespace before casting — MEDIAS values
    # are whitespace-padded (e.g. "   1", "   525.55").
    result = long_df.select(
        pl.col(entity_col)
        .str.strip_chars()
        .cast(pl.Int64, strict=False)
        .alias("newave_code"),
        pl.col("stage"),
        (
            pl.col(variable_col).str.strip_chars().alias("variable")
            if variable_col is not None
            else pl.lit("value").alias("variable")
        ),
        pl.col("value").str.strip_chars().cast(pl.Float64, strict=False),
    ).drop_nulls(subset=["newave_code", "stage"])

    return result


def read_medias_hydro(case_dir: Path) -> pl.DataFrame:
    """Read MEDIAS-USIH.CSV and return hydro results.

    Returns DataFrame with columns: ``newave_code``, ``stage``,
    ``variable``, ``value``.  Variables include VARMUH, GHIDUH,
    QTURUH, QVERTUH, QAFLUH, etc.

    Returns empty DataFrame if file not found.
    """
    return _read_medias_csv(case_dir, "MEDIAS-USIH.CSV")


def read_medias_thermal(case_dir: Path) -> pl.DataFrame:
    """Read MEDIAS-USIT.CSV and return thermal results.

    Returns DataFrame with columns: ``newave_code``, ``stage``,
    ``variable``, ``value``.

    Returns empty DataFrame if file not found.
    """
    return _read_medias_csv(case_dir, "MEDIAS-USIT.CSV")


def read_medias_system(case_dir: Path) -> pl.DataFrame:
    """Read MEDIAS-MERC.CSV and return system/market results.

    Returns DataFrame with columns: ``newave_code`` (submercado code),
    ``stage``, ``variable``, ``value``.

    Returns empty DataFrame if file not found.
    """
    return _read_medias_csv(case_dir, "MEDIAS-MERC.CSV")


# -------------------------------------------------------------------
# PMO readers
# -------------------------------------------------------------------


# The pmo.dat convergence table runs from its header to the next page header.
# Only its rows are read: a row of any other table with seven numbers (an
# electric-constraint year row, say) parses as an iteration too.
_PMO_CONV_HEADER = re.compile(
    r"^\s*ITER\s+LIM\.INF\.\s+ZINF\s+LIM\.SUP\.\s+ZSUP\s+DZINF\s+ZSUP ITER\.",
    re.MULTILINE,
)
_PMO_PAGE_HEADER = re.compile(r"^\s*CEPEL\s", re.MULTILINE)

# Match one data row of the pmo.dat convergence table:
#   iter   LIM.INF   ZINF   LIM.SUP   ZSUP   DZINF   ZSUP ITER   [tempo]
# ZSUP is the lowest ZSUP ITER so far, never the iteration's own forward cost:
# set against Novomodelo's per-iteration upper bound it hides every rise. DZINF is
# "-" on the first iteration and numeric afterwards; the trailing wall-clock
# field is optional. The deficit-risk year rows interleaved with the iterations
# are indented far past the iteration column, which keeps them out.
_PMO_CONV_ROW = re.compile(
    r"^ {0,12}(\d+)"  # iteration -> group 1
    r"\s+(?:\d+\.\d+)"  # LIM.INF (not captured)
    r"\s+(\d+\.\d+)"  # ZINF -> group 2
    r"\s+(?:\d+\.\d+)"  # LIM.SUP (not captured)
    r"\s+(\d+\.\d+)"  # ZSUP -> group 3
    r"\s+(?:-|\d+\.\d+)"  # DZINF
    r"\s+(\d+\.\d+)"  # ZSUP ITER -> group 4
    r"(?:\s+\S+)?\s*$",  # optional tempo column
    re.MULTILINE,
)

_PMO_CONVERGENCE_SCHEMA: dict[str, type[pl.DataType]] = {
    "iteration": pl.Int64,
    "lower_bound": pl.Float64,
    "upper_bound_mean": pl.Float64,
    "upper_bound_best": pl.Float64,
}


def _parse_pmo_convergence_text(
    text: str,
) -> dict[int, tuple[float, float, float]]:
    """Extract the iteration table from raw pmo.dat text.

    Returns ``{iteration: (zinf, zsup_iter, zsup)}`` keyed on iteration number.
    The source model writes two lines per iteration, three with several forward
    series (a 99% interval line between them); only the last carries DZINF and
    ZSUP ITER, which the regex requires, and it holds the values the source
    model's own ``CONVERG.TMP`` records.
    """
    header = _PMO_CONV_HEADER.search(text)
    if header is None:
        return {}
    page = _PMO_PAGE_HEADER.search(text, header.end())
    section = text[header.end() : len(text) if page is None else page.start()]
    out: dict[int, tuple[float, float, float]] = {}
    for match in _PMO_CONV_ROW.finditer(section):
        out[int(match.group(1))] = (
            float(match.group(2)),
            float(match.group(4)),
            float(match.group(3)),
        )
    return out


def read_pmo_convergence(newave_dir: Path) -> pl.DataFrame:
    """Read pmo.dat convergence table, in R$.

    Returns ``iteration`` (Int64) and the Float64 bounds ``lower_bound``
    (``ZINF``), ``upper_bound_mean`` (``ZSUP ITER``, the iteration's forward
    cost, Novomodelo's ``upper_bound_mean``) and ``upper_bound_best`` (``ZSUP``, the
    lowest ``ZSUP ITER`` so far).

    The table is parsed with a regex, not ``Pmo.convergencia``: a run with
    several forward series prints three rows per iteration, which inewave reads
    by fixed columns that the unaligned high-precision numbers overrun.

    Returns empty DataFrame if pmo.dat is not found or the iteration
    table cannot be located.
    """
    empty = pl.DataFrame(schema=_PMO_CONVERGENCE_SCHEMA)

    pmo_path = _find_pmo(newave_dir)
    if pmo_path is None:
        _LOG.warning("pmo.dat not found in %s", newave_dir)
        return empty

    try:
        text = pmo_path.read_text(encoding="latin-1")
    except OSError:
        _LOG.exception("Failed to read pmo.dat at %s", pmo_path)
        return empty

    rows = _parse_pmo_convergence_text(text)
    if not rows:
        _LOG.warning(
            "No convergence-table rows matched in %s; "
            "expected the NEWAVE 'ITER LIM.INF ZINF LIM.SUP ZSUP DZINF "
            "ZSUP ITER' table",
            pmo_path,
        )
        return empty

    iters = sorted(rows)
    # pmo.dat prints the bounds in 10^6 R$; Novomodelo's are R$.
    return pl.DataFrame(
        {
            "iteration": iters,
            "lower_bound": [rows[i][0] * 1e6 for i in iters],
            "upper_bound_mean": [rows[i][1] * 1e6 for i in iters],
            "upper_bound_best": [rows[i][2] * 1e6 for i in iters],
        },
        schema=_PMO_CONVERGENCE_SCHEMA,
    )


# pmo.dat ``produtibilidades_equivalentes`` source columns → output names. These are the
# per-plant productivities the source model's FPHA produces at different heads (altura
# min/65/max), the equivalent productivity between min/max volume, and the accumulated
# productivity used for EARM. Many run-of-river plants leave the altura_* / acumulada
# columns NaN — preserved as null so the per-comparison drop-NaN logic in the report
# keeps them out of each scatter without crashing.
_PMO_PROD_COLUMNS: dict[str, str] = {
    "produtibilidade_altura_minima": "altura_min",
    "produtibilidade_altura_65": "altura_65",
    "produtibilidade_altura_maxima": "altura_max",
    "produtibilidade_equivalente_volmin_volmax": "equivalent",
    "produtibilidade_acumulada_calculo_earm": "accumulated_earm",
}


def read_pmo_productivity_detail(newave_dir: Path) -> pl.DataFrame:
    """Read the per-plant productivity breakdown from pmo.dat.

    Pulls ``pmo.produtibilidades_equivalentes`` (filtered to the first configuration,
    ``configuracao == min``) and surfaces the head-dependent productivities the source
    model's FPHA computes.

    Returns DataFrame with columns: ``plant_name`` (Utf8), ``altura_min``,
    ``altura_65``, ``altura_max``, ``equivalent``, ``accumulated_earm``
    (all Float64; NaN preserved as null so run-of-river plants with no
    head-dependence don't crash per-column comparisons).

    Returns an empty DataFrame with the correct schema if pmo.dat is not
    found or the productivity table is unavailable.
    """
    empty = pl.DataFrame(
        schema={
            "plant_name": pl.Utf8,
            "altura_min": pl.Float64,
            "altura_65": pl.Float64,
            "altura_max": pl.Float64,
            "equivalent": pl.Float64,
            "accumulated_earm": pl.Float64,
        }
    )

    pmo_path = _find_pmo(newave_dir)
    if pmo_path is None:
        _LOG.warning("pmo.dat not found in %s", newave_dir)
        return empty

    try:
        from inewave.newave import Pmo

        pmo = Pmo.read(str(pmo_path))
        prod_df = pmo.produtibilidades_equivalentes
    except Exception:
        _LOG.warning("Failed to read productivities from pmo.dat")
        return empty

    if prod_df is None or prod_df.empty:
        return empty
    if "nome_usina" not in prod_df.columns:
        _LOG.warning(
            "pmo.dat productivities lack nome_usina column: %s",
            list(prod_df.columns),
        )
        return empty

    df = prod_df
    if "configuracao" in df.columns:
        df = df[df["configuracao"] == df["configuracao"].min()]

    source_cols = [c for c in _PMO_PROD_COLUMNS if c in df.columns]
    result = pl.from_pandas(df[["nome_usina", *source_cols]])
    result = result.rename(
        {"nome_usina": "plant_name", **{c: _PMO_PROD_COLUMNS[c] for c in source_cols}}
    )
    casts = [pl.col("plant_name").str.strip_chars()]
    casts += [
        pl.col(_PMO_PROD_COLUMNS[c]).cast(pl.Float64, strict=False) for c in source_cols
    ]
    result = result.with_columns(casts).drop_nulls(subset=["plant_name"])

    # Backfill any column the source frame lacked so the schema is stable.
    for out_name in _PMO_PROD_COLUMNS.values():
        if out_name not in result.columns:
            result = result.with_columns(pl.lit(None).cast(pl.Float64).alias(out_name))

    return result.select(
        "plant_name",
        "altura_min",
        "altura_65",
        "altura_max",
        "equivalent",
        "accumulated_earm",
    )


_INT_FILENAME_RE = re.compile(r"^int(\d{3})(\d{3})\.out$", re.IGNORECASE)


def read_nwlistop_intercambio(case_dir: Path) -> pl.DataFrame:
    """Read NWLISTOP ``intXXXYYY.out`` files into long-format flow data.

    NWLISTOP emits one file per directional submercado pair containing
    block-decomposed flow (``patamar``) plus an aggregated ``TOTAL``
    row whose value is the block-hours-weighted month average — the
    natural counterpart to the per-stage interchange we want to
    compare against Novomodelo's ``net_flow_mw``.

    Stage numbering follows the MEDIAS convention: 1 = first calendar
    month of the file (typically January of the source model's first study year), 9 =
    September year 1, 21 = September year 2, etc. This matches MEDIAS column headers
    (which start at the first study month, e.g. 9 for a September-start study) so the
    existing ``_nw_stage_offset`` logic translates both consistently into Novomodelo's
    0-based ``stage_id``.

    Returns columns ``from_submarket_code`` (Int64),
    ``to_submarket_code`` (Int64), ``from_name`` (Utf8),
    ``to_name`` (Utf8), ``stage`` (Int64), ``variable`` (Utf8, always
    "INTERC"), ``value`` (Float64).
    Returns empty DataFrame when no int*.out files are present.
    """
    empty = pl.DataFrame(
        schema={
            "from_submarket_code": pl.Int64,
            "to_submarket_code": pl.Int64,
            "from_name": pl.Utf8,
            "to_name": pl.Utf8,
            "stage": pl.Int64,
            "variable": pl.Utf8,
            "value": pl.Float64,
        }
    )
    if not case_dir.is_dir():
        return empty

    matches: list[tuple[int, int, Path]] = []
    for path in case_dir.iterdir():
        m = _INT_FILENAME_RE.match(path.name)
        if m is None:
            continue
        matches.append((int(m.group(1)), int(m.group(2)), path))
    if not matches:
        return empty

    from inewave.nwlistop import Intercambio  # local import — heavy module

    rows: list[dict] = []
    for from_code, to_code, path in matches:
        try:
            obj = Intercambio.read(str(path))
        except Exception:
            _LOG.warning("Failed to parse %s", path)
            continue
        df = obj.valores
        if df is None or df.empty:
            continue
        total = df[df["patamar"] == "TOTAL"].copy()
        if total.empty:
            continue
        total["data"] = pd.to_datetime(total["data"])
        min_year = int(total["data"].dt.year.min())
        total["stage"] = (total["data"].dt.year - min_year) * 12 + total[
            "data"
        ].dt.month
        # Series cover disjoint years in NWLISTOP output; mean is a no-op when
        # only one series carries a given stage, and remains correct otherwise.
        agg = total.groupby("stage")["valor"].mean().reset_index()

        from_name = str(obj.submercado_de or "").strip()
        to_name = str(obj.submercado_para or "").strip()
        for _, r in agg.iterrows():
            rows.append(
                {
                    "from_submarket_code": from_code,
                    "to_submarket_code": to_code,
                    "from_name": from_name,
                    "to_name": to_name,
                    "stage": int(r["stage"]),
                    "variable": "INTERC",
                    "value": float(r["valor"]),
                }
            )
    if not rows:
        return empty
    return pl.from_dicts(rows).cast(
        {
            "from_submarket_code": pl.Int64,
            "to_submarket_code": pl.Int64,
            "stage": pl.Int64,
            "value": pl.Float64,
        }
    )


def read_medias_market(case_dir: Path) -> pl.DataFrame:
    """Read MEDIAS-MERC.CSV with all variables in long format.

    Returns DataFrame with columns: ``newave_code``, ``stage``,
    ``variable``, ``value``.  Unlike ``read_medias_system`` which only
    returns CMO/DEFT, this returns all market variables for energy
    balance analysis.
    """
    return _read_medias_csv(case_dir, "MEDIAS-MERC.CSV")


def read_medias_sin(case_dir: Path) -> pl.DataFrame:
    """Read MEDIAS-SIN.CSV (system interconnected aggregate).

    Returns DataFrame with columns ``newave_code`` (always 0 for SIN),
    ``stage``, ``variable``, ``value``.  Useful variables include
    ``EARMF`` (stored energy final, MWmes), ``ENA`` (natural energy
    inflow, MWmes), ``EARMFP`` (percentage of max storage).
    """
    return _read_medias_csv(case_dir, "MEDIAS-SIN.CSV")


def read_medias_ree(case_dir: Path) -> pl.DataFrame:
    """Read MEDIAS-REE.CSV (per equivalent-energy-reservoir aggregate).

    Returns DataFrame with columns ``newave_code`` (the REE code), ``stage`` (1-based
    MEDIAS stage), ``variable``, ``value``.  The key variable for VminOP validation is
    ``EARMF`` — the REE's final stored energy in MWmes, measured *relative to the
    minimum operative volume* (i.e. useful energy).  This is the physical quantity the
    VminOP generic constraints bound, so it is the natural the source-model-side LHS for
    those constraints.
    """
    return _read_medias_csv(case_dir, "MEDIAS-REE.CSV")


_MERCL_FILE_RE = re.compile(r"mercl(\d+)\.out$", re.IGNORECASE)


def _read_mercl(path: Path) -> dict[tuple[int, int], float]:
    """``{(year, month): net load in MWmes}`` of one nwlistop ``mercl<NNN>.out``;
    calendar months outside the horizon are written ``0.``. Empty when the file
    does not parse."""
    from inewave.nwlistop import Mercl

    try:
        table = Mercl.read(str(path)).valores
    except Exception:
        _LOG.warning("Failed to parse %s", path, exc_info=True)
        return {}
    if table is None:
        return {}
    months: dict[tuple[int, int], float] = {}
    for stamp, value in zip(table["data"], table["valor"], strict=True):
        if not is_na(value):
            day = as_timestamp(stamp)
            months[(day.year, day.month)] = as_float(value)
    return months


def read_newave_net_load_nwlistop(newave_dir: Path) -> pl.DataFrame:
    """Read net load (MERCADO LIQUIDO) from per-submarket ``mercl*.out`` files.

    nwlistop writes one ``mercl<NNN>.out`` per submarket (``mercl001`` =
    submarket 1, ...) plus ``merclsin.out`` for the SIN total.  Unlike the
    ``sistema.dat`` reconstruction in :func:`read_newave_net_load`, these cover
    the **full horizon** (study + post-study), so the energy-balance charts and
    the comparison data do not drop after the study period.

    Returns the same schema as :func:`read_newave_net_load`, or an empty frame
    when no per-submarket ``mercl`` files are present.
    """
    empty = pl.DataFrame(
        schema={
            "newave_code": pl.Int64,
            "stage": pl.Int64,
            "variable": pl.Utf8,
            "value": pl.Float64,
        }
    )
    if not newave_dir.is_dir():
        return empty
    # Per-submarket files only; merclsin.out (the SIN total) has no digits and
    # is excluded by the regex.
    files = sorted(p for p in newave_dir.iterdir() if _MERCL_FILE_RE.match(p.name))
    if not files:
        return empty

    parsed: dict[int, dict[tuple[int, int], float]] = {}
    for p in files:
        m = _MERCL_FILE_RE.match(p.name)
        if m is None:
            continue
        code = int(m.group(1))  # mercl001 -> submarket 1 (sequential codes)
        parsed[code] = _read_mercl(p)
    parsed = {code: months for code, months in parsed.items() if months}
    if not parsed:
        return empty

    # MEDIAS stage numbering: stage = (year - base_year) * 12 + month, so the
    # study's first calendar month (e.g. September -> 9) keeps its MEDIAS index.
    base_year = min(year for months in parsed.values() for (year, _m) in months)
    rows: list[tuple[int, int, str, float]] = []
    for code, months in parsed.items():
        for (year, month), value in months.items():
            # Drop the pre-study calendar months the source model writes as 0.
            if value == 0.0:
                continue
            stage = (year - base_year) * 12 + month
            rows.append((code, stage, "NET_LOAD", value))
    if not rows:
        return empty
    return pl.DataFrame(
        rows,
        schema=["newave_code", "stage", "variable", "value"],
        orient="row",
    ).cast({"newave_code": pl.Int64, "stage": pl.Int64, "value": pl.Float64})


def read_newave_net_load(files: NewaveFiles) -> pl.DataFrame:
    """Read deterministic net load from the case's subsystem and additional-load
    files (``sistema`` and ``c_adic`` as ``arquivos`` names them).

    Computes ``net_load = mercado_energia + c_adic - sum(geracao_usinas_nao_simuladas)``
    per submarket and date.  The C_ADIC contribution (must-take energy from
    Itaipu, ANDE, MMGD, etc.) is added so the newave load is comparable to
    novomodelo's simulation output, which already includes C_ADIC.

    Returns a DataFrame compatible with the ``nw_market`` schema used in
    the energy balance charts:

    - ``newave_code`` (Int64): The source model submarket code
    - ``stage`` (Int64): stage number (aligned with MEDIAS column naming)
    - ``variable`` (Utf8): always ``"NET_LOAD"``
    - ``value`` (Float64): net load in MW·med

    Prefers the nwlistop ``mercl*.out`` files (full horizon) when present,
    falling back to this ``sistema.dat`` reconstruction (study period only,
    since post-study dates are written under a filtered-out sentinel year).
    """
    nwlistop = read_newave_net_load_nwlistop(files.directory)
    if not nwlistop.is_empty():
        return nwlistop

    empty = pl.DataFrame(
        schema={
            "newave_code": pl.Int64,
            "stage": pl.Int64,
            "variable": pl.Utf8,
            "value": pl.Float64,
        }
    )

    sistema_path = files.sistema
    if not sistema_path.is_file():
        _LOG.warning("%s not found in %s", sistema_path.name, files.directory)
        return empty

    try:
        from inewave.newave import Sistema

        sistema = Sistema.read(str(sistema_path))
        load_df = sistema.mercado_energia
        ncs_df = sistema.geracao_usinas_nao_simuladas
    except Exception:
        _LOG.warning("Failed to read %s for net load", sistema_path.name)
        return empty

    if load_df is None or load_df.empty:
        return empty

    # Filter to valid study dates (drop NaN load values and sentinel years).
    load_df = load_df.dropna(subset=["valor"])
    load_df = load_df[load_df["data"].dt.year < 9000]

    # Add C_ADIC must-take energy (Itaipu, ANDE, MMGD, etc.) to load.
    cadical_path = files.c_adic
    if cadical_path is not None:
        try:
            from novomodelo_bridge.newave.converters.stochastic import parse_cadical

            cadical = parse_cadical(cadical_path)
            load_df = load_df.copy()
            load_df["valor"] = load_df.apply(
                lambda row: (
                    row["valor"]
                    + cadical.get(
                        (
                            int(row["codigo_submercado"]),
                            int(row["data"].year),
                            int(row["data"].month),
                        ),
                        0.0,
                    )
                ),
                axis=1,
            )
        except Exception:
            _LOG.warning(
                "Failed to parse %s for net load adjustment", cadical_path.name
            )

    # NCS: sum across all source types per (submarket, date).
    ncs_total = None
    if ncs_df is not None and not ncs_df.empty:
        ncs_df = ncs_df.dropna(subset=["valor"])
        ncs_df = ncs_df[ncs_df["data"].dt.year < 9000]
        ncs_total = (
            ncs_df.groupby(["codigo_submercado", "data"])["valor"]
            .sum()
            .reset_index()
            .rename(columns={"valor": "ncs"})
        )

    # Compute net load = load - NCS.
    if ncs_total is not None:
        merged = load_df.merge(
            ncs_total,
            on=["codigo_submercado", "data"],
            how="left",
        )
        merged["ncs"] = merged["ncs"].fillna(0.0)
        merged["net_load"] = merged["valor"] - merged["ncs"]
    else:
        merged = load_df.copy()
        merged["net_load"] = merged["valor"]

    # Assign stage numbers: sequential from the first study month's number
    # (matching MEDIAS-MERC column naming convention).
    merged = merged.sort_values(["codigo_submercado", "data"])
    first_month = int(merged["data"].min().month)
    stage_map = {
        date: first_month + i for i, date in enumerate(sorted(merged["data"].unique()))
    }
    merged["stage"] = merged["data"].map(stage_map)

    result = pl.from_pandas(merged[["codigo_submercado", "stage", "net_load"]]).rename(
        {"codigo_submercado": "newave_code", "net_load": "value"}
    )
    result = result.with_columns(pl.lit("NET_LOAD").alias("variable"))
    result = result.cast(
        {"newave_code": pl.Int64, "stage": pl.Int64, "value": pl.Float64}
    )

    return result


def read_pmo_cost_breakdown(newave_dir: Path) -> dict[str, float]:
    """Read cost breakdown from pmo.dat ``custo_operacao_series_simuladas``.

    Returns ``{category: expected_value_in_R$}`` where values are
    converted from 10^6 R$ to R$.  Zero-cost categories are excluded.
    """
    pmo_path = _find_pmo(newave_dir)
    if pmo_path is None:
        return {}

    try:
        from inewave.newave import Pmo

        pmo = Pmo.read(str(pmo_path))
        df = pmo.custo_operacao_series_simuladas
    except Exception:
        _LOG.warning("Failed to read cost breakdown from pmo.dat")
        return {}

    if df is None or df.empty:
        return {}

    result: dict[str, float] = {}
    for _, row in df.iterrows():
        name = str(row["parcela"]).strip()
        value = float(row["valor_esperado"]) * 1e6  # Convert 10^6 R$ to R$
        if abs(value) > 0.01:
            result[name] = value

    return result


_TIM_ITERATION_SCHEMA: dict[str, type[pl.DataType]] = {
    "iteration": pl.Int64,
    "backward_seconds": pl.Float64,
    "forward_seconds": pl.Float64,
    "total_seconds": pl.Float64,
}


def read_newave_tim_iterations(newave_dir: Path) -> pl.DataFrame:
    """Read per-iteration timings from ``newave.tim``.

    Returns DataFrame with columns ``iteration`` (Int64),
    ``backward_seconds``, ``forward_seconds``, ``total_seconds``
    (Float64). Empty when the file is missing or unparseable.

    Note: The source model's first iteration row carries clock-initialisation
    garbage on the Backward / Forward fields (sometimes hundreds of
    hours). Values are returned as-is so the caller can decide how to
    surface them (e.g. annotate the chart but keep the data).
    """
    empty = pl.DataFrame(schema=_TIM_ITERATION_SCHEMA)
    tim_path = find_case_insensitive(newave_dir, "newave.tim")
    if tim_path is None:
        _LOG.warning("newave.tim not found in %s", newave_dir)
        return empty
    try:
        from inewave.newave import Newavetim

        table = Newavetim.read(str(tim_path)).tempos_iteracoes
    except Exception:
        _LOG.warning("Failed to parse newave.tim at %s", tim_path, exc_info=True)
        return empty
    if table is None or table.empty:
        return empty
    return pl.DataFrame(
        {
            "iteration": table["iteracao"].to_list(),
            "backward_seconds": table["backward"].dt.total_seconds().to_list(),
            "forward_seconds": table["forward"].dt.total_seconds().to_list(),
            "total_seconds": table["total"].dt.total_seconds().to_list(),
        },
        schema=_TIM_ITERATION_SCHEMA,
    ).sort("iteration")


def read_newave_tim_stages(newave_dir: Path) -> dict[str, float]:
    """Read stage timings from ``newave.tim`` via ``inewave.Newavetim``.

    Returns ``{etapa_name: seconds}`` keyed on the Portuguese stage labels the source
    model emits (e.g. ``"Tempo Total"``, ``"Calculo da Politica"``, ``"Simulacao
    Final"``). Empty dict on parse failure.
    """
    tim_path = find_case_insensitive(newave_dir, "newave.tim")
    if tim_path is None:
        return {}
    try:
        from inewave.newave import Newavetim

        tim = Newavetim.read(str(tim_path))
    except Exception:
        _LOG.warning("Failed to parse newave.tim via inewave at %s", tim_path)
        return {}
    df = tim.tempos_etapas
    if df is None or df.empty:
        return {}
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        name = str(row["etapa"]).strip()
        tdelta = row["tempo"]
        out[name] = float(tdelta.total_seconds())
    return out


def _find_fpha_report(newave_dir: Path, filename: str) -> Path | None:
    """Locate a production-function report under ``<newave_dir>/fpha`` (or root).

    The source model writes the production-function (FPHA) reports into an
    ``fpha`` subdirectory; some cases keep them at the case root. Search both,
    ignoring case, and return the first match or ``None`` when absent.
    """
    for candidate_dir in (newave_dir / "fpha", newave_dir):
        if candidate_dir.is_dir():
            hit = find_case_insensitive(candidate_dir, filename)
            if hit is not None:
                return hit
    return None


#: Rename map for :func:`read_fpha_planes`: the ``inewave`` source column ->
#: the descriptive coefficient name. The source model bounds generation by
#: ``GH <= fator_correcao * (gamma_0 + gamma_v * useful_volume + gamma_q *
#: turbined + gamma_s * spilled + gamma_lat * lateral)``, one plane per row.
_FPHA_PLANE_COLUMNS: dict[str, str] = {
    "codigo_usina": "newave_code",
    "periodo": "periodo",
    "indice_corte": "plane_id",
    "fator_correcao": "fator_correcao",
    "rhs_energia": "gamma_0",
    "coeficiente_volume_util_MW_hm3": "gamma_v",
    "coeficiente_vazao_turbinada_MW_m3s": "gamma_q",
    "coeficiente_vazao_vertida_MW_m3s": "gamma_s",
    "coeficiente_vazao_lateral_MW_m3s": "gamma_lat",
}

_FPHA_PLANE_SCHEMA: dict[str, type[pl.DataType]] = {
    "newave_code": pl.Int64,
    "periodo": pl.Int64,
    "plane_id": pl.Int64,
    "fator_correcao": pl.Float64,
    "gamma_0": pl.Float64,
    "gamma_v": pl.Float64,
    "gamma_q": pl.Float64,
    "gamma_s": pl.Float64,
    "gamma_lat": pl.Float64,
}


def read_fpha_planes(newave_dir: Path) -> pl.DataFrame | None:
    """Read the source model's fitted production hyperplanes (``fpha_cortes``).

    Each plant/period carries a set of hyperplanes whose lower envelope (the
    minimum over planes) is the production surface the operating model consumes.
    This reader surfaces those coefficients for the production-model comparison.
    The ``gamma_v`` coefficient multiplies *useful* volume (storage above the
    plant minimum); ``fator_correcao`` scales the whole plane.

    Parameters
    ----------
    newave_dir:
        The source model case directory (the report lives in an ``fpha``
        subdirectory).

    Returns
    -------
    polars.DataFrame | None
        One row per (plant, period, plane) conforming to
        :data:`_FPHA_PLANE_SCHEMA`, or ``None`` when the case ships no
        ``fpha_cortes`` report (or it is empty / malformed).
    """
    path = _find_fpha_report(newave_dir, "fpha_cortes.csv")
    if path is None:
        return None
    from inewave.newave import FphaCortes

    table = FphaCortes.read(str(path)).tabela
    if table is None or table.empty:
        return None
    missing = set(_FPHA_PLANE_COLUMNS) - set(table.columns)
    if missing:
        _LOG.warning("fpha_cortes is missing columns %s; skipping", sorted(missing))
        return None
    frame = pl.from_pandas(table[list(_FPHA_PLANE_COLUMNS)]).rename(_FPHA_PLANE_COLUMNS)
    return frame.cast(_FPHA_PLANE_SCHEMA)


#: Rename map for :func:`read_fpha_grid`: the ``inewave`` source column -> the
#: descriptive fitting-grid name. The grid defines, per plant/period, the
#: volume/turbined domain (and point counts) over which the planes were fitted.
_FPHA_GRID_COLUMNS: dict[str, str] = {
    "codigo_usina": "newave_code",
    "periodo": "periodo",
    "volume_armazenado_minimo": "v_min_hm3",
    "volume_armazenado_maximo": "v_max_hm3",
    "numero_pontos_volume_armazenado": "n_v",
    "vazao_turbinada_minima": "q_min_m3s",
    "vazao_turbinada_maxima": "q_max_m3s",
    "numero_pontos_vazao_turbinada": "n_q",
    "geracao_minima": "gh_min_mw",
    "geracao_maxima": "gh_max_mw",
    "tipo": "tipo",
}

_FPHA_GRID_SCHEMA: dict[str, type[pl.DataType]] = {
    "newave_code": pl.Int64,
    "periodo": pl.Int64,
    "v_min_hm3": pl.Float64,
    "v_max_hm3": pl.Float64,
    "n_v": pl.Int64,
    "q_min_m3s": pl.Float64,
    "q_max_m3s": pl.Float64,
    "n_q": pl.Int64,
    "gh_min_mw": pl.Float64,
    "gh_max_mw": pl.Float64,
    "tipo": pl.Int64,
}


def read_fpha_grid(newave_dir: Path) -> pl.DataFrame | None:
    """Read the source model's production-function fitting grid (``fpha_eco``).

    Supplies, per (plant, period), the volume/turbined domain the planes were
    fitted over plus the discretization point counts. The volume minimum doubles
    as the useful-volume reference for :func:`read_fpha_planes` (its ``gamma_v``
    multiplies storage above that minimum).

    Parameters
    ----------
    newave_dir:
        The source model case directory.

    Returns
    -------
    polars.DataFrame | None
        One row per (plant, period) conforming to :data:`_FPHA_GRID_SCHEMA`, or
        ``None`` when the case ships no ``fpha_eco`` report (or it is empty /
        malformed).
    """
    path = _find_fpha_report(newave_dir, "fpha_eco.csv")
    if path is None:
        return None
    from inewave.newave import FphaEco

    table = FphaEco.read(str(path)).tabela
    if table is None or table.empty:
        return None
    missing = set(_FPHA_GRID_COLUMNS) - set(table.columns)
    if missing:
        _LOG.warning("fpha_eco is missing columns %s; skipping", sorted(missing))
        return None
    frame = pl.from_pandas(table[list(_FPHA_GRID_COLUMNS)]).rename(_FPHA_GRID_COLUMNS)
    return frame.cast(_FPHA_GRID_SCHEMA)


_NWLISTOP_PLANT_RE = re.compile(r"^ghiduh(\d{3})\.out$", re.IGNORECASE)

#: The per-block nwlistop plant tables :func:`read_nwlistop_hydro_blocks`
#: reads: file prefix -> output column.
_NWLISTOP_BLOCK_TABLES: dict[str, str] = {
    "ghiduh": "generation_mwmes",
    "qturuh": "turbined_m3s",
    "qvertuh": "spillage_m3s",
    "gh_fphexat": "exact_generation_mwmes",
}

_NWLISTOP_KEYS = ["stage", "scenario", "block"]


type _NwlistopReader = Callable[[str], ArquivoComValores]


def _nwlistop_readers() -> dict[str, _NwlistopReader]:
    from inewave.nwlistop import Cmarg, GhFphexat, Ghiduh, Qturuh, Qvertuh, Varmuh

    return {
        "ghiduh": Ghiduh.read,
        "qturuh": Qturuh.read,
        "qvertuh": Qvertuh.read,
        "gh_fphexat": GhFphexat.read,
        "varmuh": Varmuh.read,
        "cmarg": Cmarg.read,
    }


def _read_nwlistop_table(
    path: Path, read: _NwlistopReader, start_year: int, start_month: int
) -> pl.DataFrame | None:
    """One nwlistop table keyed by study stage, series and block.

    ``stage`` counts months from the study start, the earlier months dropped.
    ``scenario`` is the series' rank within its calendar year, from 0: a
    historical final simulation labels each year's series by its inflow year.
    ``block`` is ``patamar − 1`` in a per-block table, its ``TOTAL`` rows
    dropped. ``None`` when the file does not parse.
    """
    try:
        table = read(str(path)).valores
    except Exception:
        _LOG.warning("Failed to parse %s", path, exc_info=True)
        return None
    if table is None or table.empty:
        return None
    frame = pl.from_pandas(table)
    keys = ["stage", "scenario"]
    if "patamar" in frame.columns:
        frame = frame.filter(
            pl.col("patamar").cast(pl.Utf8).str.contains(r"^\d+$")
        ).with_columns((pl.col("patamar").cast(pl.Int64) - 1).alias("block"))
        keys.append("block")
    return (
        frame.with_columns(
            (
                (pl.col("data").dt.year().cast(pl.Int64) - start_year) * 12
                + pl.col("data").dt.month().cast(pl.Int64)
                - start_month
            ).alias("stage"),
            (pl.col("serie").rank("dense").over(pl.col("data").dt.year()) - 1)
            .cast(pl.Int64)
            .alias("scenario"),
        )
        .filter(pl.col("stage") >= 0)
        .select(*keys, pl.col("valor").cast(pl.Float64))
    )


def read_nwlistop_hydro_blocks(
    newave_dir: Path, start_year: int, start_month: int
) -> pl.DataFrame:
    """Read each plant's per-block operation from the nwlistop plant tables.

    For each plant ``NNN`` its ``nwlistop.dat`` plant list names, nwlistop
    writes ``ghiduh<NNN>.out`` (generation, MWmes), ``qturuh<NNN>.out`` and
    ``qvertuh<NNN>.out`` (turbined and spilled flow, m³/s) and, in an FPHA
    run, ``gh_fphexat<NNN>.out`` (the exact production function at the
    operating point, MWmes), per block and series. A block's MWmes is its
    mean MW times its share of the month.

    Returns columns ``newave_code``, ``stage``, ``scenario``, ``block`` (keyed
    as :func:`_read_nwlistop_table`), ``generation_mwmes``, ``turbined_m3s``,
    ``spillage_m3s`` and ``exact_generation_mwmes`` (null without the
    ``gh_fphexat`` table). Empty when no plant has a ``ghiduh`` table.
    """
    schema: dict[str, type[pl.DataType]] = {
        "newave_code": pl.Int64,
        "stage": pl.Int64,
        "scenario": pl.Int64,
        "block": pl.Int64,
        **dict.fromkeys(_NWLISTOP_BLOCK_TABLES.values(), pl.Float64),
    }
    if not newave_dir.is_dir():
        return pl.DataFrame(schema=schema)
    codes = sorted(
        int(m.group(1))
        for p in newave_dir.iterdir()
        if (m := _NWLISTOP_PLANT_RE.match(p.name)) is not None
    )
    readers = _nwlistop_readers()
    plants: list[pl.DataFrame] = []
    for code in codes:
        plant: pl.DataFrame | None = None
        for prefix, column in _NWLISTOP_BLOCK_TABLES.items():
            path = find_case_insensitive(newave_dir, f"{prefix}{code:03d}.out")
            table = (
                None
                if path is None
                else _read_nwlistop_table(
                    path, readers[prefix], start_year, start_month
                )
            )
            if table is None:
                continue
            table = table.rename({"valor": column})
            plant = (
                table
                if plant is None
                else plant.join(table, on=_NWLISTOP_KEYS, how="full", coalesce=True)
            )
        if plant is not None:
            plants.append(plant.with_columns(pl.lit(code).alias("newave_code")))
    if not plants:
        return pl.DataFrame(schema=schema)
    frame = pl.concat(plants, how="diagonal_relaxed")
    return (
        frame.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(column)
            for column in _NWLISTOP_BLOCK_TABLES.values()
            if column not in frame.columns
        )
        .select(list(schema))
        .cast(pl.Schema(schema))
        .sort("newave_code", *_NWLISTOP_KEYS)
    )


def read_nwlistop_hydro_storage(
    newave_dir: Path, start_year: int, start_month: int
) -> pl.DataFrame:
    """Read each reservoir's end-of-stage useful storage from ``varmuh<NNN>.out``.

    nwlistop writes the table for each listed plant that has a reservoir, per
    stage and series, in hm³ above the plant's minimum storage.

    Returns columns ``newave_code``, ``stage``, ``scenario`` (keyed as
    :func:`_read_nwlistop_table`) and ``useful_storage_final_hm3``; empty
    without the tables.
    """
    schema: dict[str, type[pl.DataType]] = {
        "newave_code": pl.Int64,
        "stage": pl.Int64,
        "scenario": pl.Int64,
        "useful_storage_final_hm3": pl.Float64,
    }
    return _read_numbered_tables(
        newave_dir,
        "varmuh",
        "newave_code",
        "useful_storage_final_hm3",
        schema,
        start_year,
        start_month,
    )


def read_nwlistop_block_cmo(
    newave_dir: Path, start_year: int, start_month: int
) -> pl.DataFrame:
    """Read each submarket's per-block marginal cost from ``cmarg<NNN>.out``.

    nwlistop writes the table for each submarket ``NNN``, per block and
    series, in $/MWh with two decimals.

    Returns columns ``submarket_code``, ``stage``, ``scenario``, ``block``
    (keyed as :func:`_read_nwlistop_table`) and ``cmo``; empty without the
    tables.
    """
    schema: dict[str, type[pl.DataType]] = {
        "submarket_code": pl.Int64,
        "stage": pl.Int64,
        "scenario": pl.Int64,
        "block": pl.Int64,
        "cmo": pl.Float64,
    }
    return _read_numbered_tables(
        newave_dir, "cmarg", "submarket_code", "cmo", schema, start_year, start_month
    )


def _read_numbered_tables(
    newave_dir: Path,
    prefix: str,
    code_column: str,
    value_column: str,
    schema: dict[str, type[pl.DataType]],
    start_year: int,
    start_month: int,
) -> pl.DataFrame:
    """Every ``<prefix><NNN>.out`` table, ``NNN`` in *code_column*."""
    if not newave_dir.is_dir():
        return pl.DataFrame(schema=schema)
    pattern = re.compile(rf"^{prefix}(\d{{3}})\.out$", re.IGNORECASE)
    read = _nwlistop_readers()[prefix]
    tables: list[pl.DataFrame] = []
    for path in sorted(newave_dir.iterdir()):
        m = pattern.match(path.name)
        if m is None:
            continue
        table = _read_nwlistop_table(path, read, start_year, start_month)
        if table is not None:
            tables.append(
                table.rename({"valor": value_column}).with_columns(
                    pl.lit(int(m.group(1))).alias(code_column)
                )
            )
    if not tables:
        return pl.DataFrame(schema=schema)
    return (
        pl.concat(tables)
        .select(list(schema))
        .cast(pl.Schema(schema))
        .sort(code_column, *(k for k in _NWLISTOP_KEYS if k in schema))
    )
