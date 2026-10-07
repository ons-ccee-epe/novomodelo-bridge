#!/usr/bin/env python3
"""check_comment_bloat.py — comment-bloat advisory (never fails the build).

Surfaces the mechanically detectable bloat shapes as candidates for a
comment-skeptic pass (`.claude/rules/comments.md` §1); the nuanced
restatement / drift-copy / contract-vs-bloat calls are left to that pass.

Flags, over ``#`` comments in src/:

1. LONG COMMENT BLOCK — >= BLOCK_THRESHOLD (default 8) consecutive comment
   lines. Almost always verbose prose wanting a cut to the load-bearing
   clause, or a re-derived contract that should be a pointer.
2. REPEATED COMMENT LINE — the same normalized comment text (>= 20 chars)
   appearing >= REPEAT_THRESHOLD (default 3) times in one file: the "stated
   once, echoed across siblings" shape — hoist and delete the copies.

Environment: COMMENT_BLOCK_THRESHOLD / COMMENT_REPEAT_THRESHOLD override the
defaults. Exit code: always 0 (advisory).
"""

from __future__ import annotations

import os
import sys
from collections import Counter

from _scan import SRC_ROOT, iter_comments, iter_py_files, rel

BLOCK_THRESHOLD = int(os.environ.get("COMMENT_BLOCK_THRESHOLD", "8"))
REPEAT_THRESHOLD = int(os.environ.get("COMMENT_REPEAT_THRESHOLD", "3"))


def normalize(comment: str) -> str:
    return comment.lstrip("#! ").rstrip()


def main() -> int:
    hits: list[str] = []
    for path in iter_py_files(SRC_ROOT):
        comments = list(iter_comments(path))
        repeats: Counter[str] = Counter()

        run_start = 0
        prev_line = -2
        run_len = 0
        for lineno, text in comments:
            norm = normalize(text)
            if len(norm) >= 20 and any(ch.isalpha() for ch in norm):
                repeats[norm] += 1
            if lineno == prev_line + 1:
                run_len += 1
            else:
                if run_len >= BLOCK_THRESHOLD:
                    hits.append(
                        f"{rel(path)}:{run_start}-{prev_line}: {run_len}-line "
                        "comment block (cut to the load-bearing clause)"
                    )
                run_start = lineno
                run_len = 1
            prev_line = lineno
        if run_len >= BLOCK_THRESHOLD:
            hits.append(
                f"{rel(path)}:{run_start}-{prev_line}: {run_len}-line "
                "comment block (cut to the load-bearing clause)"
            )

        for norm, count in repeats.items():
            if count >= REPEAT_THRESHOLD:
                hits.append(f'{rel(path)}: comment repeated {count}x (hoist): "{norm}"')

    print(
        f"ADVISORY: comment-bloat candidates — block>={BLOCK_THRESHOLD} lines, "
        f"repeat>={REPEAT_THRESHOLD}x"
    )
    print("-" * 72)
    if hits:
        print("\n".join(hits))
        print("-" * 72)
        print(
            f"{len(hits)} candidate(s). Advisory only — apply the Deletion Test; "
            "see .claude/rules/comments.md."
        )
    else:
        print("No comment-bloat candidates found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
