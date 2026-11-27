"""Labels for the two sides of a version comparison.

Both labels reach the report unescaped: the baseline label through the shared
chart code's ``reference_label``, which interpolates it into single-quoted
JavaScript strings and HTML cells, and the candidate label in place of the
charts' literal ``Novomodelo``. A label is therefore limited to word characters
and `` .+-``: anything else is cleaned to ``_``, never escaped.
"""

from __future__ import annotations

import re
from pathlib import Path

SIDES: tuple[str, str] = ("baseline", "candidate")

_UNSAFE = re.compile(r"[^\w .+-]")


def _clean(text: str | None) -> str:
    return "" if text is None else _UNSAFE.sub("_", text).strip()


def _case_name(output_dir: Path) -> str:
    resolved = output_dir.resolve()
    return resolved.parent.name if resolved.name == "output" else resolved.name


def side_labels(
    versions: tuple[str | None, str | None],
    output_dirs: tuple[Path, Path],
    chosen: tuple[str | None, str | None] = (None, None),
) -> tuple[str, str]:
    """The baseline and candidate labels, each pair ordered as :data:`SIDES`.

    Per side, the first non-empty after cleaning of: the *chosen* label;
    ``Novomodelo <version>`` when both versions are known and differ; the case
    directory name (*output_dir*'s, or its parent's when it is literally
    ``output``); ``Output 1`` or ``Output 2``. Labels that end up equal get
    `` (1)`` and `` (2)``. A version made only of blanks counts as absent.
    """
    known = [_clean(version) for version in versions]
    by_version = all(known) and known[0] != known[1]
    labels = []
    for number, (chosen_label, version, output_dir) in enumerate(
        zip(chosen, known, output_dirs, strict=True), start=1
    ):
        label = _clean(chosen_label)
        if not label and by_version:
            label = f"Novomodelo {version}"
        label = label or _clean(_case_name(output_dir)) or f"Output {number}"
        labels.append(label)
    if labels[0] == labels[1]:
        labels = [f"{label} ({number})" for number, label in enumerate(labels, 1)]
    return labels[0], labels[1]
