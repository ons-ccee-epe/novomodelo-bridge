"""Gates for the data-map pages (``docs/<track>-data-map.md``).

The pages are rendered from ``docs/lineage/<track>.toml``. Four things keep
that TOML true: it validates against the files dataclasses and the idecomp
register set; the committed page equals a fresh render; every leaf field a
real conversion of the mini deck emits has an entry, and every entry names an
emitted field; and every deck file or register the pipeline's code reads for
an output is listed in that output's ``reads``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from scripts.lineage import model, render, trace

_TRACKS = model.TRACKS


def _leaf_paths(obj: object, prefix: str = "") -> set[str]:
    if isinstance(obj, dict):
        out: set[str] = set()
        for key, value in obj.items():
            if key == "$schema":
                continue
            out |= _leaf_paths(value, f"{prefix}.{key}" if prefix else key)
        return out
    if isinstance(obj, list):
        if not obj:
            return {prefix}
        out = set()
        for value in obj:
            out |= _leaf_paths(value, prefix + "[]")
        return out
    return {prefix}


@pytest.fixture(scope="session")
def converted(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Both mini decks converted once per session (the boundary import is a CLI step, so none runs here)."""
    from novomodelo_bridge.decomp.pipeline import convert_decomp_case
    from novomodelo_bridge.newave.pipeline import convert_newave_case

    decks = Path(__file__).parent / "decks"
    out = {t: tmp_path_factory.mktemp(f"lineage_{t}") for t in _TRACKS}
    convert_newave_case(decks / "newave_mini", out["newave"])
    convert_decomp_case(decks / "decomp_mini", out["decomp"], force=True)
    return out


@pytest.mark.parametrize("track", _TRACKS)
def test_label_table_matches_files_dataclass(track: str) -> None:
    assert model.label_table_matches_files_dataclass(track) == []


@pytest.mark.parametrize("track", _TRACKS)
def test_lineage_validates(track: str) -> None:
    lineage = model.load(track)
    assert model.validate(lineage) == []


@pytest.mark.parametrize("track", _TRACKS)
def test_data_map_page_is_fresh(track: str) -> None:
    page = render.page_path(track)
    assert page.is_file(), f"{page} missing; run scripts/gen-lineage-docs.py"
    assert page.read_text(encoding="utf-8") == render.render(model.load(track)), (
        f"{page.name} is stale; run scripts/gen-lineage-docs.py"
    )


@pytest.mark.parametrize("track", _TRACKS)
def test_every_emitted_field_is_documented(
    track: str, converted: dict[str, Path]
) -> None:
    lineage = model.load(track)
    case_dir = converted[track]
    documented = {out.path: out for out in lineage.outputs}
    problems: list[str] = []
    emitted: dict[str, set[str]] = {}
    for path in sorted(case_dir.rglob("*")):
        if path.is_dir() or path.name == "conversion_manifest.json":
            continue
        rel = path.relative_to(case_dir).as_posix()
        if path.suffix == ".json":
            emitted[rel] = _leaf_paths(json.loads(path.read_text(encoding="utf-8")))
        elif path.suffix == ".parquet":
            emitted[rel] = set(pq.read_schema(path).names)
        else:
            continue
        if rel not in documented:
            problems.append(f"{rel}: emitted but has no [[output]] entry")
    for out in lineage.outputs:
        if out.kind == "checkpoint":
            continue
        if out.path not in emitted:
            if not out.conditional:
                problems.append(
                    f"{out.path}: documented as always written but not emitted"
                )
            continue
        seen = emitted[out.path]
        listed = {f.path for f in out.fields}
        for missing in sorted(seen - listed):
            problems.append(f"{out.path}: emitted field {missing!r} is undocumented")
        for extra in sorted(listed - seen):
            field = next(f for f in out.fields if f.path == extra)
            if not field.when:
                problems.append(
                    f"{out.path}: documented field {extra!r} is not emitted "
                    "(add `when` if it is conditional)"
                )
    assert problems == []


@pytest.mark.parametrize("track", _TRACKS)
def test_code_reads_are_documented(track: str) -> None:
    """documented ⊇ traced, per output; and no file or whole register the code
    reads is listed as `unread`."""
    lineage = model.load(track)
    traced = trace.trace(track)
    assert traced, "the tracer attributed no output; its pipeline walk is broken"
    documented = {out.path: {ref.token for ref in out.reads} for out in lineage.outputs}
    for path in trace.write_sites(track):
        assert path in documented, f"pipeline writes {path} but it is undocumented"
    problems: list[str] = []
    all_tokens: set[tuple[str, str | None]] = set()
    for path, tokens in traced.items():
        reads = documented[path]
        files_read = {t[0] for t in reads}
        all_tokens |= tokens
        for token in sorted(tokens, key=str):
            covered = token in reads or (token[1] is None and token[0] in files_read)
            if not covered:
                problems.append(f"{path}: code reads {token}, lineage does not list it")
    unread_files = {s.file for s in lineage.sources if s.status == "unread"}
    unread_items = {
        (s.file, it.register)
        for s in lineage.sources
        for it in s.items
        if it.status == "unread" and it.register and not it.item
    }
    for token in sorted(all_tokens, key=str):
        if token[0] in unread_files:
            problems.append(f"{token[0]} is listed as unread but the code reads it")
        if token in unread_items:
            problems.append(
                f"register {token} is listed as unread but the code reads it"
            )
    assert problems == []
