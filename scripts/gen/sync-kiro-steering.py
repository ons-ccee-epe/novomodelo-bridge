#!/usr/bin/env python3
"""Project the Claude rule set onto Kiro steering files.

The `.claude/` tree is the single source of truth for agent rules. Kiro reads
`.kiro/steering/*.md` instead, with a different front-matter dialect, so this
generator translates each Claude rule file into its Kiro steering equivalent.

Source -> generated mapping:
  CLAUDE.md                 -> .kiro/steering/00-hard-rules.md     (inclusion: always)
  .claude/rules/<name>.md   -> .kiro/steering/<name>.md           (inclusion: fileMatch | always)
  .claude/architecture-rules.md -> .kiro/steering/architecture-rules.md (inclusion: manual)

Front-matter translation:
  Claude `paths: [glob, ...]`  ->  Kiro `inclusion: fileMatch` + `fileMatchPattern: '<glob>'`
  (no `paths:` key)            ->  Kiro `inclusion: always`

Run it to (re)write the steering files after editing any `.claude/` rule.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Generated-file banner tells a human reader where to edit instead — the Kiro
# steering files are generated, so edits belong in the .claude/ source.
BANNER = (
    "<!-- GENERATED FILE — DO NOT EDIT.\n"
    "     Source of truth: {source}\n"
    "     Regenerate: python scripts/gen/sync-kiro-steering.py -->\n"
)

# Default name the CLAUDE.md projection takes in .kiro/steering/. The novomodelo repo
# uses this; novomodelo-bridge's established steering calls it agent-guidelines.md,
# so the name is overridable via --hard-rules-name.
DEFAULT_HARD_RULES_NAME = "00-hard-rules.md"


def build_overrides(hard_rules_name: str) -> dict[str, tuple[str, str]]:
    """Source -> (steering filename, forced inclusion) overrides.

    Rule files not listed here are mapped by their path front-matter (fileMatch)
    or default to `always` when they carry no `paths:` key.
    """
    return {
        "CLAUDE.md": (hard_rules_name, "always"),
        ".claude/architecture-rules.md": ("architecture-rules.md", "manual"),
    }


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_front_matter(text: str) -> tuple[dict, str]:
    """Return (front_matter_dict, body). Only the keys this tool needs are parsed."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    raw = text[3:end].strip("\n")
    body = text[end + 4 :].lstrip("\n")
    fm: dict = {}
    current_list_key: str | None = None
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            if current_list_key is not None:
                fm.setdefault(current_list_key, []).append(
                    stripped[2:].strip().strip("\"'")
                )
            continue
        if ":" in line and not line.startswith(" "):
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if val == "":
                fm[key] = []
                current_list_key = key
            else:
                fm[key] = val.strip("\"'")
                current_list_key = None
    return fm, body


def kiro_front_matter(source_rel: str, claude_fm: dict, overrides: dict) -> str:
    override = overrides.get(source_rel)
    forced = override[1] if override else None

    paths = claude_fm.get("paths")
    if isinstance(paths, str):
        paths = [paths]

    if forced == "always":
        return "---\ninclusion: always\n---\n"
    if forced == "manual":
        return "---\ninclusion: manual\n---\n"

    if paths:
        # Kiro's fileMatchPattern takes one pattern; join multiple globs with the
        # brace-alternation form Kiro accepts, else emit the single glob.
        pattern = paths[0] if len(paths) == 1 else "{" + ",".join(paths) + "}"
        return f"---\ninclusion: fileMatch\nfileMatchPattern: '{pattern}'\n---\n"
    return "---\ninclusion: always\n---\n"


def steering_name(source_rel: str, source_path: Path, overrides: dict) -> str:
    override = overrides.get(source_rel)
    if override:
        return override[0]
    return source_path.name


def build_outputs(root: Path, overrides: dict) -> dict[Path, str]:
    """Map each target steering path to its generated content."""
    outputs: dict[Path, str] = {}
    steering_dir = root / ".kiro" / "steering"

    sources: list[tuple[str, Path]] = [("CLAUDE.md", root / "CLAUDE.md")]
    rules_dir = root / ".claude" / "rules"
    if rules_dir.is_dir():
        for p in sorted(rules_dir.glob("*.md")):
            sources.append((f".claude/rules/{p.name}", p))
    arch = root / ".claude" / "architecture-rules.md"
    if arch.is_file():
        sources.append((".claude/architecture-rules.md", arch))

    for source_rel, source_path in sources:
        if not source_path.is_file():
            continue
        text = source_path.read_text(encoding="utf-8")
        claude_fm, body = parse_front_matter(text)
        fm = kiro_front_matter(source_rel, claude_fm, overrides)
        banner = BANNER.format(source=source_rel)
        content = f"{fm}\n{banner}\n{body}"
        if not content.endswith("\n"):
            content += "\n"
        outputs[steering_dir / steering_name(source_rel, source_path, overrides)] = (
            content
        )
    return outputs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=repo_root())
    ap.add_argument(
        "--hard-rules-name",
        default=DEFAULT_HARD_RULES_NAME,
        help=(
            "steering filename for the CLAUDE.md projection "
            f"(default: {DEFAULT_HARD_RULES_NAME}; novomodelo-bridge uses "
            "agent-guidelines.md)"
        ),
    )
    args = ap.parse_args()

    overrides = build_overrides(args.hard_rules_name)
    outputs = build_outputs(args.root, overrides)

    steering_dir = args.root / ".kiro" / "steering"
    steering_dir.mkdir(parents=True, exist_ok=True)
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path.relative_to(args.root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
