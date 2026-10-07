"""Canonical tidy/long comparison data model for the ANALYZE layer.

This module defines the single, serializable source of truth shared across the
console, HTML report, and export paths: :class:`ComparisonDataset`. It holds a
tidy/long value frame (one observation per row), a per-variable summary frame,
a typed :class:`RenderInputs` side-table for the non-tidy artifacts the HTML
report reads, and a ``metadata`` dict for provenance not read by the report.

This is a leaf module: it must not import from any other comparator (notably
``results.py``) so that callers can wire it in without creating import cycles.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, cast

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

    from cobre_bridge.comparators.model import ResultComparison


class SchemaError(ValueError):
    """Raised when a :class:`ComparisonDataset` violates its column contract.

    Subclasses :class:`ValueError` so callers can catch it as a value problem.
    The message always names the offending column(s) or source value(s).
    """


#: Allowed values for the tidy ``source`` column.
VALID_SOURCES: frozenset[str] = frozenset({"newave", "cobre", "p10", "p50", "p90"})

#: Ordered schema of the canonical tidy/long value frame. Every consumer
#: relies on this exact column contract.
TIDY_SCHEMA: dict[str, type[pl.DataType]] = {
    "entity_type": pl.Utf8,
    "entity_id": pl.Int64,
    "entity_name": pl.Utf8,
    "bus": pl.Int64,
    "stage": pl.Int64,
    "block": pl.Int64,
    "variable": pl.Utf8,
    "source": pl.Utf8,
    "value": pl.Float64,
}

#: Wrapper key marking a metadata value that was a polars DataFrame. A
#: wrapped entry has the shape ``{_FRAME_SENTINEL: "polars", "records": [...],
#: "columns": {name: dtype_name}}``. The ``columns`` map preserves the schema
#: so empty frames keep their columns/dtypes on round-trip.
_FRAME_SENTINEL: str = "__frame__"

#: File names written/read by :meth:`ComparisonDataset.to_dir` /
#: :meth:`ComparisonDataset.from_dir`, in their canonical order.
_TIDY_FILE: str = "comparison.parquet"
_SUMMARY_FILE: str = "summary.parquet"
_METADATA_FILE: str = "metadata.json"

#: Reserved top-level key under which :meth:`ComparisonDataset.to_dir` nests
#: the serialized :class:`RenderInputs` payload inside ``metadata.json`` —
#: kept out of the provenance ``metadata`` dict's own namespace so a render
#: field name can never collide with a provenance key.
_RENDER_KEY: str = "__render__"

#: ``RenderInputs`` fields whose value is a ``dict`` keyed by a Cobre/source
#: entity id (``int``), not ``str``. JSON object keys are always strings, so
#: -- unlike a generic ``metadata`` dict, whose non-string-key coercion is a
#: one-way, accepted trade-off -- these three are looked up by ``int`` id
#: downstream (``report_builder``/``analyze._bus_name_lookups``); a string
#: key there is a silent lookup miss, not a type-checker nuisance, so
#: :func:`_render_from_json` coerces them back to ``int`` explicitly.
_INT_KEYED_RENDER_FIELDS: frozenset[str] = frozenset(
    {"nw_bus_names", "cobre_bus_meta", "cobre_hydro_meta"}
)

#: Ordered schema of the per-variable summary frame.
SUMMARY_SCHEMA: dict[str, type[pl.DataType]] = {
    "variable": pl.Utf8,
    "count": pl.Int64,
    "mean_abs_diff": pl.Float64,
    "max_abs_diff": pl.Float64,
    "mean_smape": pl.Float64,
    "max_smape": pl.Float64,
    "within_tol_rate": pl.Float64,
    "correlation": pl.Float64,
}


@dataclass
class RenderInputs:
    """Typed render-input side-table for the HTML report builder.

    Replaces the untyped ``metadata`` grab-bag: every field is one of the
    non-tidy artifacts ``report_builder.build_comparison_report`` reads —
    percentile-band frames, cost/name dicts, and the raw ``results`` list.
    Each field's empty default reproduces the legacy ``_meta_*`` accessors'
    empty-on-miss behaviour, so an absent input renders the same "No data"
    fallback as before, never a ``KeyError``/``AttributeError``.

    ``results`` is typed under ``TYPE_CHECKING`` so this module stays a leaf:
    it never imports ``results.py`` at runtime (see the module docstring).
    """

    bus: pl.DataFrame = field(default_factory=pl.DataFrame)
    bus_aggregates: pl.DataFrame = field(default_factory=pl.DataFrame)
    cobre_bus_meta: dict[int, dict] = field(default_factory=dict)
    cobre_convergence: pl.DataFrame = field(default_factory=pl.DataFrame)
    cobre_costs: dict[str, float] = field(default_factory=dict)
    cobre_hydro_means: pl.DataFrame = field(default_factory=pl.DataFrame)
    cobre_hydro_meta: dict[int, dict] = field(default_factory=dict)
    cobre_hydro_per_stage_bounds: pl.DataFrame = field(default_factory=pl.DataFrame)
    cobre_iteration_timing: pl.DataFrame = field(default_factory=pl.DataFrame)
    cobre_stage_costs: pl.DataFrame = field(default_factory=pl.DataFrame)
    cobre_training_seconds: float = 0.0
    fpha_metrics: pl.DataFrame = field(default_factory=pl.DataFrame)
    fpha_spill: pl.DataFrame = field(default_factory=pl.DataFrame)
    fpha_surface: pl.DataFrame = field(default_factory=pl.DataFrame)
    gc_bounds: pl.DataFrame = field(default_factory=pl.DataFrame)
    gc_constraints: list[dict] = field(default_factory=list)
    gc_lhs_cobre: pl.DataFrame = field(default_factory=pl.DataFrame)
    gc_lhs_newave: pl.DataFrame = field(default_factory=pl.DataFrame)
    hydro: pl.DataFrame = field(default_factory=pl.DataFrame)
    line: pl.DataFrame = field(default_factory=pl.DataFrame)
    line_bounds: pl.DataFrame = field(default_factory=pl.DataFrame)
    line_meta: list[dict] = field(default_factory=list)
    nw_bus_names: dict[int, str] = field(default_factory=dict)
    nw_convergence: pl.DataFrame = field(default_factory=pl.DataFrame)
    nw_costs: dict[str, float] = field(default_factory=dict)
    nw_hydro_slacks: pl.DataFrame = field(default_factory=pl.DataFrame)
    nw_market: pl.DataFrame = field(default_factory=pl.DataFrame)
    nw_max_stage: int | None = None
    nw_net_load: pl.DataFrame = field(default_factory=pl.DataFrame)
    nw_offset: int = 0
    nw_sin: pl.DataFrame = field(default_factory=pl.DataFrame)
    nw_tim_iterations: pl.DataFrame = field(default_factory=pl.DataFrame)
    nw_tim_stages: dict[str, float] = field(default_factory=dict)
    productivity_detail: pl.DataFrame = field(default_factory=pl.DataFrame)
    productivity_per_stage: pl.DataFrame = field(default_factory=pl.DataFrame)
    results: list[ResultComparison] = field(default_factory=list)
    thermal: pl.DataFrame = field(default_factory=pl.DataFrame)


@dataclass
class ComparisonDataset:
    """Central tidy/long comparison dataset for the ANALYZE layer.

    Attributes:
        tidy: The long value frame conforming to :data:`TIDY_SCHEMA`. One row per
            (entity, stage, block, variable, source) observation.
        summary: Per-variable summary statistics conforming to
            :data:`SUMMARY_SCHEMA`.
        metadata: Side-table holding provenance carried verbatim, not read by
            the HTML report (``top_divergences``, ``footer_counts``,
            ``nw_hydro_names``, and the DECOMP-track ``unmapped``).
        render: The typed non-tidy inputs the HTML report builder reads (see
            :class:`RenderInputs`).
    """

    tidy: pl.DataFrame
    summary: pl.DataFrame
    metadata: dict[str, object] = field(default_factory=dict)
    render: RenderInputs = field(default_factory=RenderInputs)

    @classmethod
    def empty(cls) -> ComparisonDataset:
        """Return a valid empty dataset matching the schemas.

        Returns:
            A dataset whose ``tidy`` and ``summary`` frames are empty but typed
            per :data:`TIDY_SCHEMA` and :data:`SUMMARY_SCHEMA`, with an empty
            ``metadata`` dict and default (empty) ``render``.
        """
        return cls(
            tidy=pl.DataFrame(schema=TIDY_SCHEMA),
            summary=pl.DataFrame(schema=SUMMARY_SCHEMA),
            metadata={},
            render=RenderInputs(),
        )

    def validate(self) -> None:
        """Validate the dataset against its column contract.

        Raises:
            SchemaError: If the ``tidy`` frame columns do not equal the ordered
                :data:`TIDY_SCHEMA` keys, if any ``tidy`` or ``summary`` column
                has a dtype differing from its schema, if any ``source`` value
                (when the frame is non-empty) is outside :data:`VALID_SOURCES`,
                or if the ``summary`` frame columns do not equal the ordered
                :data:`SUMMARY_SCHEMA` keys.
        """
        expected_tidy = list(TIDY_SCHEMA)
        actual_tidy = list(self.tidy.columns)
        if actual_tidy != expected_tidy:
            msg = f"tidy frame columns {actual_tidy} != expected {expected_tidy}"
            raise SchemaError(msg)
        _check_dtypes(self.tidy, TIDY_SCHEMA, "tidy")

        if self.tidy.height > 0:
            seen = set(self.tidy["source"].unique().to_list())
            invalid = seen - VALID_SOURCES
            if invalid:
                msg = (
                    f"tidy frame has invalid source values {sorted(invalid)}; "
                    f"allowed sources are {sorted(VALID_SOURCES)}"
                )
                raise SchemaError(msg)

        expected_summary = list(SUMMARY_SCHEMA)
        actual_summary = list(self.summary.columns)
        if actual_summary != expected_summary:
            msg = (
                f"summary frame columns {actual_summary} != expected {expected_summary}"
            )
            raise SchemaError(msg)
        _check_dtypes(self.summary, SUMMARY_SCHEMA, "summary")

    def to_dir(self, out_dir: Path) -> list[Path]:
        """Serialize this dataset to ``out_dir`` as Parquet + JSON.

        Validates first (fail fast), creates ``out_dir`` if needed, then writes
        the tidy frame to ``comparison.parquet``, the summary frame to
        ``summary.parquet``, and the JSON-native view of :attr:`metadata` plus
        :attr:`render` (nested under the reserved :data:`_RENDER_KEY`) to
        ``metadata.json`` — the on-disk shape stays three files; ``render``
        rides inside the same ``metadata.json`` rather than a fourth file, so
        the round trip reproduces the report byte-identically.

        Metadata serialization (see :func:`_metadata_to_json`): JSON-native
        values pass through; ``pl.DataFrame`` values are wrapped as
        ``{"__frame__": "polars", "records": [...]}``; any other value raises
        :class:`TypeError` naming its key. ``render``'s
        frame fields serialize the same way; its ``results`` field serializes
        as a list of :func:`dataclasses.asdict` records (see
        :func:`_render_to_json`).

        Note:
            JSON object keys are always strings, so a metadata dict with
            non-string keys (e.g. ``{0: "ITAIPU"}``) round-trips with string
            keys (``{"0": "ITAIPU"}``). This coercion is intentional and not
            reversed on load.

        Args:
            out_dir: Destination directory; created with ``parents=True``.

        Returns:
            The three written paths in the order
            ``[comparison.parquet, summary.parquet, metadata.json]``.

        Raises:
            SchemaError: If :meth:`validate` fails.
            TypeError: If a metadata or render value is neither JSON-native
                nor a frame.
        """
        self.validate()
        out_dir.mkdir(parents=True, exist_ok=True)

        tidy_path = out_dir / _TIDY_FILE
        summary_path = out_dir / _SUMMARY_FILE
        metadata_path = out_dir / _METADATA_FILE

        self.tidy.write_parquet(tidy_path)
        self.summary.write_parquet(summary_path)
        metadata_view = _metadata_to_json(self.metadata)
        metadata_view[_RENDER_KEY] = _render_to_json(self.render)
        metadata_path.write_text(json.dumps(metadata_view, indent=2), encoding="utf-8")

        return [tidy_path, summary_path, metadata_path]

    @classmethod
    def from_dir(cls, in_dir: Path) -> ComparisonDataset:
        """Reconstruct a :class:`ComparisonDataset` written by :meth:`to_dir`.

        Reads ``comparison.parquet``, ``summary.parquet``, and ``metadata.json``
        from ``in_dir``, reconstructs frame-wrapped metadata/render values back
        to their original frame type, validates the result, and returns it.

        Note:
            JSON object keys are strings, so metadata dicts with originally
            non-string keys come back with string keys (see :meth:`to_dir`).

        Args:
            in_dir: Directory containing the three serialized files.

        Returns:
            The validated reconstructed dataset, its ``render`` rebuilt from
            the ``metadata.json``'s nested :data:`_RENDER_KEY` payload.

        Raises:
            FileNotFoundError: If any of the three files is absent; the message
                names the first missing path.
            SchemaError: If the reconstructed dataset fails :meth:`validate`.
            TypeError: If a wrapped frame entry declares an unknown frame type,
                or the render payload is malformed.
        """
        tidy_path = in_dir / _TIDY_FILE
        summary_path = in_dir / _SUMMARY_FILE
        metadata_path = in_dir / _METADATA_FILE

        for path in (tidy_path, summary_path, metadata_path):
            if not path.exists():
                msg = f"missing dataset file: {path}"
                raise FileNotFoundError(msg)

        tidy = pl.read_parquet(tidy_path)
        summary = pl.read_parquet(summary_path)
        metadata_view = json.loads(metadata_path.read_text(encoding="utf-8"))
        render_view = metadata_view.pop(_RENDER_KEY, {})
        if not isinstance(render_view, dict):
            type_name = type(render_view).__name__
            msg = f"metadata key {_RENDER_KEY!r} must be a dict, got {type_name}"
            raise TypeError(msg)
        render = _render_from_json(render_view)
        metadata = _metadata_from_json(metadata_view)

        dataset = cls(tidy=tidy, summary=summary, metadata=metadata, render=render)
        dataset.validate()
        return dataset


def _metadata_to_json(meta: dict[str, object]) -> dict[str, object]:
    """Build the JSON-serializable view of a metadata dict.

    JSON-native values (``str, int, float, bool, None`` and ``list``/``dict``
    thereof) are passed through verbatim. ``pl.DataFrame`` values are wrapped
    as ``{_FRAME_SENTINEL: "polars", "records": [...], "columns": {name:
    dtype_name}}`` using the frame's native record export. The ``columns`` map
    carries the schema so that an *empty* frame (whose ``records`` list is
    ``[]``) keeps its column names and dtypes on round-trip instead of
    collapsing to a zero-column frame.

    Args:
        meta: The metadata side-table.

    Returns:
        A dict whose values are all JSON-serializable.

    Raises:
        TypeError: If a value is neither JSON-native nor a polars frame; the
            message names the offending key.
    """
    view: dict[str, object] = {}
    for key, value in meta.items():
        if isinstance(value, pl.DataFrame):
            columns = {name: str(dtype) for name, dtype in value.schema.items()}
            view[key] = {
                _FRAME_SENTINEL: "polars",
                "records": value.to_dicts(),
                "columns": columns,
            }
        elif _is_json_native(value):
            view[key] = value
        else:
            type_name = type(value).__name__
            msg = f"metadata key {key!r} has non-serializable value of type {type_name}"
            raise TypeError(msg)
    return view


def _metadata_from_json(view: dict[str, object]) -> dict[str, object]:
    """Reconstruct a metadata dict from its JSON view (inverse of to-json).

    Entries shaped like ``{_FRAME_SENTINEL: "polars", "records": [...],
    "columns": {name: dtype_name}}`` are rebuilt into a polars frame; all
    other entries pass through unchanged. When ``records`` is empty and a
    ``columns`` map is present, the frame is reconstructed *with* an explicit
    schema so that an empty frame keeps its column names and dtypes (see
    :func:`_metadata_to_json`).

    Args:
        view: The JSON-deserialized metadata view.

    Returns:
        The reconstructed metadata side-table.

    Raises:
        TypeError: If a wrapped entry is missing its ``records`` key, or declares
            an unknown frame type; the message names the offending key.
    """
    meta: dict[str, object] = {}
    for key, value in view.items():
        if isinstance(value, dict) and _FRAME_SENTINEL in value:
            kind = value[_FRAME_SENTINEL]
            if kind != "polars":
                msg = f"metadata key {key!r} has unknown frame type {kind!r}"
                raise TypeError(msg)
            if "records" not in value:
                msg = f"metadata key {key!r} frame wrapper is missing 'records' key"
                raise TypeError(msg)
            records = value["records"]
            if not isinstance(records, list):
                type_name = type(records).__name__
                msg = (
                    f"metadata key {key!r} frame 'records' must be a list, "
                    f"got {type_name}"
                )
                raise TypeError(msg)
            columns = value.get("columns")
            meta[key] = _build_polars_frame(records, columns)
        else:
            meta[key] = value
    return meta


def _render_to_json(render: RenderInputs) -> dict[str, object]:
    """Build the JSON-serializable view of a :class:`RenderInputs`.

    Reuses :func:`_metadata_to_json`'s frame-wrapping for every ``pl``
    frame field. The one field that is neither JSON-native nor a frame
    (``results``, a ``list[ResultComparison]``) is flattened via
    :func:`dataclasses.asdict` first — ``ResultComparison`` is a flat frozen
    dataclass of JSON-native scalars, so no bespoke object serializer is
    needed.

    Args:
        render: The render-input side-table to serialize.

    Returns:
        A dict whose values are all JSON-serializable, one entry per
        :class:`RenderInputs` field.
    """
    raw: dict[str, object] = {}
    for render_field in fields(RenderInputs):
        value = getattr(render, render_field.name)
        if render_field.name == "results":
            raw[render_field.name] = [asdict(r) for r in value]
        else:
            raw[render_field.name] = value
    return _metadata_to_json(raw)


def _render_from_json(view: dict[str, object]) -> RenderInputs:
    """Reconstruct a :class:`RenderInputs` from its JSON view.

    Inverse of :func:`_render_to_json`: reuses :func:`_metadata_from_json` to
    rebuild every frame field, restores ``int`` keys on the three
    :data:`_INT_KEYED_RENDER_FIELDS` dicts (JSON stringified them), then
    reconstructs ``results`` as ``[ResultComparison(**record) for record in
    ...]``. The import is local to keep this module a leaf (see the module
    docstring).

    Args:
        view: The JSON-deserialized render view.

    Returns:
        The reconstructed :class:`RenderInputs`.

    Raises:
        TypeError: If ``results`` (or one of its records) is malformed, or a
            wrapped frame entry declares an unknown frame type; the message
            names the offending key.
    """
    from cobre_bridge.comparators.model import ResultComparison

    raw = _metadata_from_json(view)

    for key in _INT_KEYED_RENDER_FIELDS:
        value = raw.get(key)
        if isinstance(value, dict):
            str_keyed = cast("dict[str, object]", value)
            raw[key] = {int(k): v for k, v in str_keyed.items()}

    results_raw = raw.get("results", [])
    if not isinstance(results_raw, list):
        type_name = type(results_raw).__name__
        msg = f"render key 'results' must be a list, got {type_name}"
        raise TypeError(msg)

    results: list[ResultComparison] = []
    for record in results_raw:
        if not isinstance(record, dict):
            type_name = type(record).__name__
            msg = f"render key 'results' record must be a dict, got {type_name}"
            raise TypeError(msg)
        try:
            # record is dict[str, object]; ty can't narrow each value to its
            # field's tighter type without a per-field cast.
            results.append(
                ResultComparison(**cast("dict[str, object]", record))  # ty: ignore[invalid-argument-type]
            )
        except TypeError as exc:
            msg = f"render key 'results' has a malformed record: {exc}"
            raise TypeError(msg) from exc
    raw["results"] = results

    # raw is dict[str, object]; ty can't narrow each value to its field's
    # tighter type without a per-field cast.
    return RenderInputs(**raw)  # ty: ignore[invalid-argument-type]


def _build_polars_frame(records: list[object], columns: object) -> pl.DataFrame:
    """Rebuild a polars frame from records, preserving schema when empty.

    Args:
        records: The ``records`` payload (a list of row dicts).
        columns: The ``columns`` map (column-name -> polars dtype-name), or
            ``None`` for legacy wrappers without a schema.

    Returns:
        The reconstructed polars frame. When ``records`` is empty and
        ``columns`` is a mapping, an explicitly-schemed empty frame is built so
        column names and dtypes survive; otherwise the records drive inference.
    """
    if not records and isinstance(columns, dict):
        schema = {name: getattr(pl, dtype_name) for name, dtype_name in columns.items()}
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(records)


def _is_json_native(value: object) -> bool:
    """Return whether ``value`` is composed only of JSON-native types.

    JSON-native means ``None``/``bool``/``int``/``float``/``str`` scalars, or a
    ``list`` of such, or a ``dict`` whose values are such (recursively). Dict
    keys are not constrained here because JSON coerces them to strings on dump.

    Args:
        value: The value to inspect.

    Returns:
        ``True`` if ``json.dumps`` would accept ``value`` without a custom
        encoder, ``False`` otherwise.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, list):
        return all(_is_json_native(item) for item in value)
    if isinstance(value, dict):
        return all(_is_json_native(item) for item in value.values())
    return False


def _check_dtypes(
    frame: pl.DataFrame,
    schema: dict[str, type[pl.DataType]],
    label: str,
) -> None:
    """Verify each column dtype of ``frame`` matches its schema entry.

    The schema maps column name to a polars dtype *class* (e.g. ``pl.Int64``);
    the class is instantiated (``pl.Int64()``) before comparison because the
    frame's runtime schema holds dtype *instances*. Assumes column names/order
    have already been validated against ``schema``.

    Args:
        frame: The frame whose dtypes to check.
        schema: The column-name -> dtype-class contract.
        label: A human label for the frame (``"tidy"`` / ``"summary"``) used in
            the error message.

    Raises:
        SchemaError: If any column's dtype differs from its expected dtype; the
            message names the column, its actual dtype, and the expected dtype.
    """
    actual_schema = frame.schema
    for name, dtype_cls in schema.items():
        expected = dtype_cls()
        actual = actual_schema[name]
        if actual != expected:
            msg = (
                f"{label} frame column {name!r} has dtype {actual} "
                f"!= expected {expected}"
            )
            raise SchemaError(msg)
