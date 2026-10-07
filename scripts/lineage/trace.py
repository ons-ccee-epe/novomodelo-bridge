"""Derive file-level lineage from the code: output path -> deck sources read.

Walks a track's ``pipeline.py`` for ``writer.write_json`` / ``write_parquet``
sites, follows each written value back through the assignments that produced
it to the converter calls, and collects the ``case.<file>`` attributes (and,
on the DECOMP track, the ``dadger.<xx>()`` / ``dadgnl.<xx>()`` register calls)
those converters reach transitively within the track package.

The trace under-approximates on purpose. A value that reaches a write site
through a tuple unpack or a phase dataclass is left untraced rather than
attributed the reads of every sibling it was produced with, because the gate
built on this (``documented ⊇ traced``) must never fail on a read the output
does not have. Untraced outputs are reported as absent, not empty.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from .model import REPO_ROOT, registers

SRC = REPO_ROOT / "src" / "novomodelo_bridge"
Token = tuple[str, str | None]

# ``NewaveCase`` / ``DecompCase`` properties that are not deck files themselves
# but derive from them; the tracer expands each to the files it reads.
_DERIVED: dict[str, dict[str, tuple[Token, ...]]] = {
    "newave": {
        "active_hydros": (("confhd", None), ("hidr", None), ("exph", None)),
        "active_hydro_codes": (("confhd", None), ("hidr", None), ("exph", None)),
        "plants": (("confhd", None), ("hidr", None), ("exph", None)),
        "hydro_registry": (("hidr", None),),
        "id_map": (
            ("confhd", None),
            ("hidr", None),
            ("exph", None),
            ("conft", None),
            ("sistema", None),
        ),
        "horizon": (("dger", None),),
        "fpha_enabled": (("dger", None),),
        "switches": (("dger", None),),
    },
    "decomp": {
        "id_map": (("dadger", "SB"), ("dadger", "CT"), ("dadger", "UH")),
        "calendar": (("dadger", "DP"),),
        "start_date": (("dadger", "DP"),),
    },
}
# ``case.files.<attr>`` reads that are not files-dataclass fields.
_FILES_ATTR: dict[str, dict[str, Token]] = {
    "newave": {"directory": ("restricao_eletrica", None)},
    "decomp": {},
}
_RECEIVERS = {"dadger": "dadger", "dg": "dadger", "dadgnl": "dadgnl", "gnl": "dadgnl"}


@dataclass
class _Package:
    track: str
    modules: dict[str, ast.Module]
    imports: dict[str, dict[str, str]] = field(default_factory=dict)
    functions: dict[str, dict[str, ast.FunctionDef]] = field(default_factory=dict)
    cache: dict[tuple[str, str], set[Token]] = field(default_factory=dict)
    registers: dict[str, set[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, mod in self.modules.items():
            self.imports[name] = _imports(mod)
            self.functions[name] = {
                n.name: n for n in mod.body if isinstance(n, ast.FunctionDef)
            }
        if self.track == "decomp":
            self.registers = {k: set(v) for k, v in registers().items()}

    def resolve(self, module: str, func: ast.AST) -> tuple[str, str] | None:
        """``(module, function)`` for a call target inside the track package."""
        if isinstance(func, ast.Name):
            if func.id in self.functions[module]:
                return (module, func.id)
            target = self.imports[module].get(func.id)
            if target and "." in target:
                mod, fn = target.rsplit(".", 1)
                if fn in self.functions.get(mod, {}):
                    return (mod, fn)
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            target = self.imports[module].get(func.value.id)
            if target in self.modules and func.attr in self.functions[target]:
                return (target, func.attr)
        return None

    def reads(self, module: str, fn: str, stack: frozenset = frozenset()) -> set[Token]:
        key = (module, fn)
        if key in self.cache:
            return self.cache[key]
        if key in stack:
            return set()
        stack = stack | {key}
        out: set[Token] = set()
        derived = _DERIVED[self.track]
        files_attr = _FILES_ATTR[self.track]
        for node in ast.walk(self.functions[module][fn]):
            if isinstance(node, ast.Attribute):
                base = node.value
                if isinstance(base, ast.Name) and base.id == "case":
                    if node.attr == "files":
                        continue
                    if node.attr in derived:
                        out.update(derived[node.attr])
                    else:
                        out.add((node.attr, None))
                elif (
                    isinstance(base, ast.Attribute)
                    and isinstance(base.value, ast.Name)
                    and base.value.id == "case"
                    and base.attr == "files"
                ):
                    out.add(files_attr.get(node.attr, (node.attr, None)))
            if isinstance(node, ast.Attribute) and self.registers:
                recv = node.value
                recv_name = (
                    recv.id
                    if isinstance(recv, ast.Name)
                    else recv.attr
                    if isinstance(recv, ast.Attribute)
                    else ""
                )
                file = _RECEIVERS.get(recv_name)
                if file and node.attr.upper() in self.registers[file]:
                    out.add((file, node.attr.upper()))
            if isinstance(node, ast.Call):
                self._of_type_register(node, out)
                target = self.resolve(module, node.func)
                if target:
                    out |= self.reads(*target, stack)
        self.cache[key] = out
        return out

    def _of_type_register(self, call: ast.Call, out: set[Token]) -> None:
        """``dadger.data.of_type(AR)``: a register read through its model class."""
        if not self.registers or not isinstance(call.func, ast.Attribute):
            return
        if call.func.attr != "of_type" or not call.args:
            return
        file = _RECEIVERS.get(_root_name(call.func.value))
        arg = call.args[0]
        if file and isinstance(arg, ast.Name) and arg.id[:2] in self.registers[file]:
            out.add((file, arg.id[:2]))


def _root_name(node: ast.AST) -> str:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else ""


def _imports(mod: ast.Module) -> dict[str, str]:
    out: dict[str, str] = {}
    for node in ast.walk(mod):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                out[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out[alias.asname or alias.name] = alias.name
    return out


def _load(track: str) -> _Package:
    modules: dict[str, ast.Module] = {}
    for path in sorted((SRC / track).rglob("*.py")):
        rel = path.relative_to(SRC.parent).with_suffix("").as_posix()
        name = rel.replace("/", ".").removesuffix(".__init__")
        modules[name] = ast.parse(path.read_text(encoding="utf-8"))
    return _Package(track, modules)


def _key_of(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    return None


def trace(track: str) -> dict[str, set[Token]]:
    """``{output path: tokens}`` for every write site the trace could attribute.

    A write whose value could not be followed back to a converter is absent
    from the result (never mapped to an empty set).
    """
    pkg = _load(track)
    pipeline = f"novomodelo_bridge.{track}.pipeline"
    mod = pkg.modules[pipeline]
    producers: dict[str, set[tuple[str, str]] | None] = {}

    def flows(expr: ast.AST) -> set[tuple[str, str]] | None:
        """Converter functions whose results reach *expr*; ``None`` when a
        value on the way came out of a tuple unpack and cannot be attributed."""
        found: set[tuple[str, str]] = set()
        for node in ast.walk(expr):
            if isinstance(node, ast.Call):
                target = pkg.resolve(pipeline, node.func)
                if target:
                    found.add(target)
            key = _key_of(node)
            if key in producers:
                known = producers[key]
                if known is None:
                    return None
                found |= known
        return found

    writes: dict[str, set[tuple[str, str]] | None] = {}
    for fn in mod.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign | ast.AnnAssign) and node.value is not None:
                value = flows(node.value)
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    if isinstance(target, ast.Tuple):
                        for leaf in target.elts:
                            key = _key_of(leaf)
                            if key:
                                producers[key] = None
                        continue
                    key = _key_of(target)
                    if key:
                        producers[key] = value
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("write_json", "write_parquet")
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Constant)
            ):
                writes[str(node.args[0].value)] = flows(node.args[1])

    result: dict[str, set[Token]] = {}
    for path, funcs in writes.items():
        if not funcs:
            continue
        tokens: set[Token] = set()
        for module, fn in funcs:
            tokens |= pkg.reads(module, fn)
        if tokens:
            result[path] = tokens
    return result


def write_sites(track: str) -> set[str]:
    """Every constant output path a track's pipeline writes."""
    mod = ast.parse(
        (SRC / track / "pipeline.py").read_text(encoding="utf-8"),
    )
    return {
        str(node.args[0].value)
        for node in ast.walk(mod)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("write_json", "write_parquet")
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }


def pipeline_path(track: str) -> Path:
    return SRC / track / "pipeline.py"
