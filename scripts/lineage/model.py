"""Load and validate ``docs/lineage/<track>.toml``.

The TOML is the single authored source for a track's data map: one ``[[output]]``
per file the converter writes, one ``[[output.field]]`` per leaf field or column,
and ``[[source]]`` entries for deck files and records the bridge does not
convert. Everything a page shows is derived from it; ``validate`` is what keeps
it honest against the files dataclasses, the idecomp register set, and itself.
"""

from __future__ import annotations

import tomllib
import types
from dataclasses import dataclass, field
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Any, get_type_hints

REPO_ROOT = Path(__file__).resolve().parents[2]
LINEAGE_DIR = REPO_ROOT / "docs" / "lineage"
TRACKS = ("newave", "decomp")

OUTPUT_KINDS = ("json", "parquet", "checkpoint")
FIELD_STATUSES = ("converted", "derived", "constant", "null")
SOURCE_STATUSES = ("deferred", "unread", "index")
REGISTER_FILES = ("dadger", "dadgnl")

# Case-relative output directories in page order; "" is the case root.
DIRECTORY_ORDER = ("", "system", "scenarios", "constraints", "boundary")


class LineageError(Exception):
    """A structural problem in a lineage TOML (wrong keys, wrong types)."""


@dataclass(frozen=True)
class SourceFile:
    """One deck file the registry knows: its display name and whether the
    files dataclass tolerates its absence."""

    key: str
    label: str
    optional: bool
    arquivos_attr: str | None = None


_NEWAVE_LABELS: dict[str, tuple[str, str | None]] = {
    "dger": ("dger.dat", "dger"),
    "confhd": ("confhd.dat", "confhd"),
    "conft": ("conft.dat", "conft"),
    "sistema": ("sistema.dat", "sistema"),
    "clast": ("clast.dat", "clast"),
    "term": ("term.dat", "term"),
    "ree": ("ree.dat", "ree"),
    "patamar": ("patamar.dat", "patamar"),
    "hidr": ("hidr.dat", None),
    "vazoes": ("vazoes.dat", None),
    "modif": ("modif.dat", "modif"),
    "ghmin": ("ghmin.dat", "ghmin"),
    "penalid": ("penalid.dat", "penalid"),
    "vazpast": ("vazpast.dat", "vazpast"),
    "dsvagua": ("dsvagua.dat", "dsvagua"),
    "curva": ("curva.dat", "curva"),
    "expt": ("expt.dat", "expt"),
    "exph": ("exph.dat", "exph"),
    "manutt": ("manutt.dat", "manutt"),
    "c_adic": ("c_adic.dat", "c_adic"),
    "cvar": ("cvar.dat", "cvar"),
    "agrint": ("agrint.dat", "agrint"),
    "re_dat": ("re.dat", "re"),
    "volref_saz": ("volref_saz.dat", "volume_referencia_sazonal"),
    "shist": ("shist.dat", "shist"),
    "adterm": ("adterm.dat", "adterm"),
    "polinjus": ("polinjus.csv", None),
    "tratamento_fpha": ("tratamento-fpha.csv", None),
}
# Deck files the converter reads without a ``NewaveFiles`` field: the index
# files and the LIBs electric-constraint file found by directory scan.
_NEWAVE_EXTRAS: dict[str, tuple[str, str | None]] = {
    "caso": ("caso.dat", None),
    "arquivos": ("arquivos.dat", "arquivos"),
    "restricao_eletrica": ("restricao-eletrica.csv", None),
    "indices": ("indices.csv", None),
}
_DECOMP_LABELS: dict[str, str] = {
    "dadger": "dadger.rvN",
    "vazoes": "vazoes.rvN",
    "hidr": "hidr.dat",
    "dadgnl": "dadgnl.rvN",
    "renovaveis": "renovaveis*",
    "polinjus": "polinjus.csv",
    "libs_restricao_eletrica": "lib_restricao-eletrica-especial*.csv",
    "cortesh": "cortesh.dat",
    "cortes": "cortes*.dat",
}
# Deck files the converter reads without a ``DecompFiles`` field: the index
# files and ``mlt.dat``, which the boundary import locates itself.
_DECOMP_EXTRAS: dict[str, str] = {
    "caso": "caso.dat",
    "indices": "indices.csv",
    "mlt": "mlt.dat",
}


def _is_optional(hint: Any) -> bool:
    return isinstance(hint, types.UnionType) and type(None) in hint.__args__


def registry(track: str) -> dict[str, SourceFile]:
    """The deck files a track can cite, keyed by files-dataclass field name.

    Built from the live ``NewaveFiles`` / ``DecompFiles`` type hints so the
    optional flag cannot drift; the display names are the one hand-kept table,
    and ``label_table_matches_files_dataclass`` reports any key mismatch.
    """
    if track == "newave":
        from novomodelo_bridge.newave.files import NewaveFiles

        hints = get_type_hints(NewaveFiles)
        out = {
            key: SourceFile(key, label, _is_optional(hints.get(key)), attr)
            for key, (label, attr) in _NEWAVE_LABELS.items()
        }
        out.update(
            {
                key: SourceFile(key, label, False, attr)
                for key, (label, attr) in _NEWAVE_EXTRAS.items()
            }
        )
        return out
    if track == "decomp":
        from novomodelo_bridge.decomp.files import DecompFiles

        hints = get_type_hints(DecompFiles)
        out = {
            key: SourceFile(key, label, _is_optional(hints.get(key)))
            for key, label in _DECOMP_LABELS.items()
        }
        out.update(
            {
                key: SourceFile(key, label, False)
                for key, label in _DECOMP_EXTRAS.items()
            }
        )
        return out
    raise ValueError(f"unknown track {track!r}")


def label_table_matches_files_dataclass(track: str) -> list[str]:
    """Keys the label table and the files dataclass disagree on."""
    if track == "newave":
        from novomodelo_bridge.newave.files import NewaveFiles

        expected = {f.name for f in dc_fields(NewaveFiles)} - {"directory"}
        known = set(_NEWAVE_LABELS)
    else:
        from novomodelo_bridge.decomp.files import DecompFiles

        expected = {f.name for f in dc_fields(DecompFiles)} - {"revision"}
        known = set(_DECOMP_LABELS)
    return sorted(f"unlabelled files field: {k}" for k in expected - known) + sorted(
        f"label without files field: {k}" for k in known - expected
    )


# Registers the bridge detects by a text scan of ``dadger.rvN`` (to report
# them as not converted) and for which idecomp has no model class.
_EXTRA_REGISTERS: dict[str, tuple[str, ...]] = {"dadger": ("CA", "FE", "HA", "LA")}


def registers() -> dict[str, tuple[str, ...]]:
    """Upper-case register mnemonics per register file, from idecomp's models.

    Derived from the register classes of ``idecomp.decomp.modelos.dadger`` and
    ``...dadgnl`` rather than from the accessor methods on ``Dadger``: a few
    registers (``AR``, ``EZ``) have a class but no accessor, and the ``AC``
    sub-mnemonics are one class each (``ACVOLMAX``), all folded to ``AC``.
    """
    import inspect

    from cfinterface.components.register import Register
    from idecomp.decomp.modelos import dadger, dadgnl

    def mnemonics(module: object) -> tuple[str, ...]:
        names = {
            name[:2]
            for name, cls in inspect.getmembers(module, inspect.isclass)
            if issubclass(cls, Register)
            and cls is not Register
            and cls.__module__ == module.__name__
            and name[:2].isupper()
        }
        return tuple(sorted(names))

    return {
        "dadger": tuple(sorted({*mnemonics(dadger), *_EXTRA_REGISTERS["dadger"]})),
        "dadgnl": mnemonics(dadgnl),
    }


def newave_arquivos_attrs() -> tuple[str, ...]:
    """Every input-file slot NEWAVE's ``arquivos.dat`` index can name."""
    from inewave.newave import Arquivos

    return tuple(n for n, v in vars(Arquivos).items() if isinstance(v, property))


@dataclass(frozen=True)
class SourceRef:
    file: str
    register: str | None = None
    item: str | None = None

    @property
    def token(self) -> tuple[str, str | None]:
        return (self.file, self.register)


@dataclass(frozen=True)
class Field:
    path: str
    from_: tuple[SourceRef, ...]
    how: str
    status: str
    when: str


@dataclass(frozen=True)
class Output:
    path: str
    kind: str
    when: str
    module: str
    reads: tuple[SourceRef, ...]
    summary: str
    fields: tuple[Field, ...]

    @property
    def directory(self) -> str:
        return self.path.rsplit("/", 1)[0] if "/" in self.path else ""

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1] or self.path

    @property
    def conditional(self) -> bool:
        return bool(self.when)


@dataclass(frozen=True)
class SourceItem:
    register: str | None
    item: str
    status: str
    note: str


@dataclass(frozen=True)
class Source:
    file: str
    label: str
    status: str
    note: str
    covers: tuple[str, ...]
    items: tuple[SourceItem, ...]


@dataclass(frozen=True)
class Lineage:
    track: str
    outputs: tuple[Output, ...]
    sources: tuple[Source, ...]

    def output(self, path: str) -> Output:
        for out in self.outputs:
            if out.path == path:
                return out
        raise KeyError(path)


@dataclass
class _Reader:
    """Strict table reader: unknown keys and wrong types are errors, not silence."""

    where: str
    table: dict[str, Any]
    allowed: frozenset[str]
    problems: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for key in self.table:
            if key not in self.allowed:
                raise LineageError(f"{self.where}: unknown key {key!r}")

    def str_(self, key: str, *, required: bool = False) -> str:
        value = self.table.get(key, "")
        if required and not value:
            raise LineageError(f"{self.where}: missing {key!r}")
        if not isinstance(value, str):
            raise LineageError(f"{self.where}: {key!r} must be a string")
        return value.strip()

    def list_(self, key: str) -> list[Any]:
        value = self.table.get(key, [])
        if not isinstance(value, list):
            raise LineageError(f"{self.where}: {key!r} must be a list")
        return value


def _ref(where: str, raw: Any, *, allow_item: bool) -> SourceRef:
    if not isinstance(raw, dict):
        raise LineageError(f"{where}: source references must be inline tables")
    reader = _Reader(where, raw, frozenset({"file", "register", "item"}))
    file = reader.str_("file", required=True)
    register = reader.str_("register") or None
    item = reader.str_("item") or None
    if item is not None and not allow_item:
        raise LineageError(f"{where}: `reads` entries name files/registers, not items")
    return SourceRef(file, register, item)


def load(track: str, path: Path | None = None) -> Lineage:
    """Parse ``docs/lineage/<track>.toml`` (or *path*) into a :class:`Lineage`."""
    path = path or LINEAGE_DIR / f"{track}.toml"
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    top = _Reader(str(path), data, frozenset({"track", "output", "source"}))
    if top.str_("track", required=True) != track:
        raise LineageError(f"{path}: `track` must be {track!r}")

    outputs: list[Output] = []
    for i, raw in enumerate(top.list_("output")):
        where = f"output[{i}]"
        r = _Reader(
            where,
            raw,
            frozenset({"path", "kind", "when", "module", "reads", "summary", "field"}),
        )
        out_path = r.str_("path", required=True)
        where = f"output {out_path}"
        fields_: list[Field] = []
        for j, fraw in enumerate(r.list_("field")):
            fwhere = f"{where} field[{j}]"
            fr = _Reader(
                fwhere, fraw, frozenset({"path", "from", "how", "status", "when"})
            )
            fields_.append(
                Field(
                    path=fr.str_("path", required=True),
                    from_=tuple(
                        _ref(fwhere, x, allow_item=True) for x in fr.list_("from")
                    ),
                    how=fr.str_("how"),
                    status=fr.str_("status") or "converted",
                    when=fr.str_("when"),
                )
            )
        outputs.append(
            Output(
                path=out_path,
                kind=r.str_("kind", required=True),
                when=r.str_("when"),
                module=r.str_("module", required=True),
                reads=tuple(_ref(where, x, allow_item=False) for x in r.list_("reads")),
                summary=r.str_("summary"),
                fields=tuple(fields_),
            )
        )

    sources: list[Source] = []
    for i, raw in enumerate(top.list_("source")):
        where = f"source[{i}]"
        r = _Reader(
            where, raw, frozenset({"file", "label", "status", "note", "covers", "item"})
        )
        file = r.str_("file", required=True)
        where = f"source {file}"
        items: list[SourceItem] = []
        for j, iraw in enumerate(r.list_("item")):
            iwhere = f"{where} item[{j}]"
            ir = _Reader(
                iwhere, iraw, frozenset({"register", "item", "status", "note"})
            )
            items.append(
                SourceItem(
                    register=ir.str_("register") or None,
                    item=ir.str_("item"),
                    status=ir.str_("status", required=True),
                    note=ir.str_("note"),
                )
            )
        sources.append(
            Source(
                file=file,
                label=r.str_("label"),
                status=r.str_("status"),
                note=r.str_("note"),
                covers=tuple(str(c) for c in r.list_("covers")),
                items=tuple(items),
            )
        )
    return Lineage(track=track, outputs=tuple(outputs), sources=tuple(sources))


def validate(lineage: Lineage) -> list[str]:
    """Semantic problems in *lineage*; empty means the TOML is consistent.

    Checks, in order: output shape and uniqueness; every cited file and register
    exists; every field's sources are declared in its output's ``reads``;
    statuses agree with the presence of sources; the source inventory does not
    contradict the field entries; and every deck file the files dataclass
    resolves (plus every idecomp register, and every ``arquivos.dat`` slot on
    the NEWAVE side) is accounted for as read, deferred, or unread.
    """
    track = lineage.track
    reg = registry(track)
    regs = registers() if track == "decomp" else {}
    problems: list[str] = []

    def check_ref(where: str, ref: SourceRef) -> None:
        if ref.file not in reg:
            problems.append(f"{where}: unknown file key {ref.file!r}")
            return
        if track == "decomp" and ref.file in REGISTER_FILES:
            if ref.register is None:
                problems.append(f"{where}: {ref.file} references need a `register`")
            elif ref.register not in regs[ref.file]:
                problems.append(f"{where}: {ref.file} has no register {ref.register!r}")
        elif ref.register is not None:
            problems.append(f"{where}: `register` is only for dadger/dadgnl")

    seen_paths: set[str] = set()
    read_tokens: set[tuple[str, str | None]] = set()
    used_items: set[tuple[str, str | None, str]] = set()
    for out in lineage.outputs:
        where = f"output {out.path}"
        if out.path in seen_paths:
            problems.append(f"{where}: duplicate output path")
        seen_paths.add(out.path)
        if out.kind not in OUTPUT_KINDS:
            problems.append(f"{where}: kind must be one of {OUTPUT_KINDS}")
        expected_suffix = {"json": ".json", "parquet": ".parquet", "checkpoint": "/"}
        if out.kind in expected_suffix and not out.path.endswith(
            expected_suffix[out.kind]
        ):
            problems.append(f"{where}: path does not match kind {out.kind!r}")
        if out.directory not in DIRECTORY_ORDER:
            problems.append(f"{where}: directory {out.directory!r} is not a case dir")
        if not (REPO_ROOT / out.module).is_file():
            problems.append(f"{where}: module {out.module!r} does not exist")
        if not out.reads:
            problems.append(f"{where}: `reads` is empty")
        declared: set[tuple[str, str | None]] = set()
        for ref in out.reads:
            check_ref(f"{where} reads", ref)
            if ref.token in declared:
                problems.append(f"{where}: duplicate read {ref.token}")
            declared.add(ref.token)
        read_tokens |= declared
        if out.kind == "checkpoint" and out.fields:
            problems.append(f"{where}: checkpoint outputs carry no fields")
        if out.kind != "checkpoint" and not out.fields:
            problems.append(f"{where}: no fields documented")
        seen_fields: set[str] = set()
        for f in out.fields:
            fwhere = f"{where} field {f.path}"
            if f.path in seen_fields:
                problems.append(f"{fwhere}: duplicate field path")
            seen_fields.add(f.path)
            if f.status not in FIELD_STATUSES:
                problems.append(f"{fwhere}: status must be one of {FIELD_STATUSES}")
            if f.status in ("constant", "null") and f.from_:
                problems.append(f"{fwhere}: status {f.status!r} takes no `from`")
            if f.status == "converted" and not f.from_:
                problems.append(f"{fwhere}: a converted field needs a `from`")
            if not f.how and f.status != "converted":
                problems.append(f"{fwhere}: status {f.status!r} needs a `how`")
            if "REVISAR" in f.how:
                problems.append(f"{fwhere}: unresolved REVISAR note")
            for ref in f.from_:
                check_ref(fwhere, ref)
                if ref.token not in declared and (ref.file, None) not in declared:
                    problems.append(
                        f"{fwhere}: source {ref.token} is not in the output's `reads`"
                    )
                if ref.item:
                    used_items.add((ref.file, ref.register, ref.item))
        if "REVISAR" in out.summary or "REVISAR" in out.when:
            problems.append(f"{where}: unresolved REVISAR note")

    source_keys: set[str] = set()
    covered: set[str] = set()
    for src in lineage.sources:
        where = f"source {src.file}"
        if src.file in source_keys:
            problems.append(f"{where}: duplicate source")
        source_keys.add(src.file)
        registered = src.file in reg
        if not registered and not src.label:
            problems.append(f"{where}: unregistered files need a `label`")
        if not registered and src.status != "unread":
            problems.append(f"{where}: an unregistered file can only be `unread`")
        if src.status and src.status not in SOURCE_STATUSES:
            problems.append(f"{where}: status must be one of {SOURCE_STATUSES}")
        if src.status in ("unread", "deferred") and src.file in {
            t[0] for t in read_tokens
        }:
            problems.append(
                f"{where}: status {src.status!r} but an output reads this file"
            )
        if registered and not src.status and not src.items and not src.note:
            problems.append(f"{where}: entry adds nothing (no status, items or note)")
        covered |= set(src.covers)
        if "REVISAR" in src.note:
            problems.append(f"{where}: unresolved REVISAR note")
        for it in src.items:
            iwhere = f"{where} item {it.item or it.register}"
            if not it.item and not it.register:
                problems.append(f"{iwhere}: needs an `item` (or a `register`)")
            if it.status not in SOURCE_STATUSES:
                problems.append(f"{iwhere}: status must be one of {SOURCE_STATUSES}")
            if track == "decomp" and src.file in REGISTER_FILES:
                if it.register is None:
                    problems.append(f"{iwhere}: {src.file} items need a `register`")
                elif it.register not in regs[src.file]:
                    problems.append(f"{iwhere}: no register {it.register!r}")
            elif it.register is not None:
                problems.append(f"{iwhere}: `register` is only for dadger/dadgnl")
            if (src.file, it.register, it.item) in used_items:
                problems.append(f"{iwhere}: listed as unconverted but a field uses it")
            whole_register = it.register is not None and not it.item
            if (
                whole_register
                and it.status == "unread"
                and (src.file, it.register) in read_tokens
            ):
                problems.append(
                    f"{iwhere}: register {it.register} is read by an output"
                )
            if not it.note:
                problems.append(f"{iwhere}: needs a `note`")
            if "REVISAR" in it.note:
                problems.append(f"{iwhere}: unresolved REVISAR note")

    # Every registered deck file is accounted for: read by an output, or
    # listed with a status.
    read_files = {t[0] for t in read_tokens}
    for key in reg:
        if key not in read_files and key not in source_keys:
            problems.append(
                f"deck file {key!r} ({reg[key].label}) is neither read nor "
                "listed under [[source]]"
            )
    if track == "decomp":
        register_items = {
            (src.file, it.register)
            for src in lineage.sources
            for it in src.items
            if it.register is not None
        }
        for file, mnemonics in regs.items():
            for mnemonic in mnemonics:
                if (file, mnemonic) not in read_tokens and (
                    file,
                    mnemonic,
                ) not in register_items:
                    problems.append(
                        f"{file} register {mnemonic} is neither read nor listed "
                        "under [[source.item]]"
                    )
    if track == "newave":
        attr_to_key = {
            sf.arquivos_attr: key for key, sf in reg.items() if sf.arquivos_attr
        }
        for attr in newave_arquivos_attrs():
            key = attr_to_key.get(attr)
            accounted = (
                (key is not None and (key in read_files or key in source_keys))
                or attr in source_keys
                or attr in covered
            )
            if not accounted:
                problems.append(
                    f"arquivos.dat slot {attr!r} is neither read nor listed under "
                    "[[source]] (directly or via `covers`)"
                )
    return problems


def field_tokens(out: Output) -> set[tuple[str, str | None]]:
    return {ref.token for f in out.fields for ref in f.from_}


def used_items_by_token(
    lineage: Lineage,
) -> dict[tuple[str, str | None], list[tuple[str, Output, Field]]]:
    """``{(file, register): [(item, output, field), ...]}`` from the field entries."""
    index: dict[tuple[str, str | None], list[tuple[str, Output, Field]]] = {}
    for out in lineage.outputs:
        for f in out.fields:
            for ref in f.from_:
                index.setdefault(ref.token, []).append((ref.item or "", out, f))
    return index
