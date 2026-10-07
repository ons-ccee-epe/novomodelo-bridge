#!/usr/bin/env python3
"""gen-lineage-docs.py — render docs/<track>-data-map.md from docs/lineage/.

Usage:
    scripts/gen-lineage-docs.py            # validate, render, write both pages
    scripts/gen-lineage-docs.py --check    # validate, freshness, code trace; exit 1
    scripts/gen-lineage-docs.py --track decomp

The pages are pt-BR and never hand-edited. ``--check`` is what the pre-commit
hook runs; ``tests/test_lineage.py`` runs the same checks plus the
emission-coverage gate, which needs a real conversion of the mini decks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lineage import model, render, trace  # noqa: E402 (path set above)


def check(track: str) -> list[str]:
    problems = model.label_table_matches_files_dataclass(track)
    try:
        lineage = model.load(track)
    except (OSError, model.LineageError) as exc:
        return [*problems, f"{track}: {exc}"]
    problems += [f"{track}: {p}" for p in model.validate(lineage)]
    page = render.page_path(track)
    if not page.is_file() or page.read_text(encoding="utf-8") != render.render(lineage):
        problems.append(f"{track}: {page.relative_to(model.REPO_ROOT)} is stale")
    documented = {out.path: {ref.token for ref in out.reads} for out in lineage.outputs}
    unread_files = {s.file for s in lineage.sources if s.status == "unread"}
    unread_registers = {
        (s.file, it.register)
        for s in lineage.sources
        for it in s.items
        if it.status == "unread" and it.register and not it.item
    }
    for path, tokens in trace.trace(track).items():
        if path not in documented:
            problems.append(f"{track}: pipeline writes {path} but it is undocumented")
            continue
        reads = documented[path]
        files_read = {t[0] for t in reads}
        for token in sorted(tokens, key=str):
            covered = token in reads or (token[1] is None and token[0] in files_read)
            if not covered:
                problems.append(
                    f"{track}: {path} reads {token} (from the code) but the lineage "
                    "does not list it"
                )
            if token[0] in unread_files or token in unread_registers:
                problems.append(
                    f"{track}: {token} is listed as unread but {path} reads it"
                )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--track", choices=model.TRACKS)
    args = parser.parse_args()
    tracks = (args.track,) if args.track else model.TRACKS

    if args.check:
        problems = [p for t in tracks for p in check(t)]
        if problems:
            print("FAIL: data-map lineage is inconsistent or stale.\n")
            print("\n".join(f"  {p}" for p in problems))
            print(
                "\nFix docs/lineage/<track>.toml, then run scripts/gen-lineage-docs.py"
            )
            return 1
        print("OK: data-map pages are consistent with the lineage and the code.")
        return 0

    for track in tracks:
        lineage = model.load(track)
        problems = model.validate(lineage)
        if problems:
            print(f"FAIL: docs/lineage/{track}.toml has problems:\n")
            print("\n".join(f"  {p}" for p in problems))
            return 1
        page = render.page_path(track)
        page.write_text(render.render(lineage), encoding="utf-8")
        print(f"Regenerated {page.relative_to(model.REPO_ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
