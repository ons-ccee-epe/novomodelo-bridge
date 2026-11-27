"""Alignment of the entities and stages two Novomodelo runs have in common.

Entities pair by :func:`name_key` and never by id: the same plant carries
different ids in two runs. A name key held by more than one entry on either
side is ambiguous, so it is reported in ``duplicates`` and appears in no other
field. Both alignments are pure; they emit no Diagnostic and log nothing, and
the caller reports what they return.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

# Stripped spelling and id of each entry sharing a name key, in file order.
_Index = dict[str, list[tuple[str, int]]]


@dataclass(frozen=True)
class AlignedPair:
    """One entity found on both sides: the baseline's spelling and both ids."""

    name: str
    baseline_id: int
    candidate_id: int


@dataclass(frozen=True)
class DuplicateName:
    """A name key held by several entries; every stripped spelling, in file order."""

    name_key: str
    baseline_names: tuple[str, ...]
    candidate_names: tuple[str, ...]


@dataclass(frozen=True)
class NameAlignment:
    """The alignment of one entity type.

    Attributes:
        entity_type: The caller's label for the type, as passed in.
        pairs: Entities on both sides, in baseline file order.
        only_in_baseline: Stripped baseline spellings, sorted by name key.
        only_in_candidate: Stripped candidate spellings, sorted by name key.
        duplicates: Name keys excluded from the other fields, sorted by key.
    """

    entity_type: str
    pairs: tuple[AlignedPair, ...]
    only_in_baseline: tuple[str, ...]
    only_in_candidate: tuple[str, ...]
    duplicates: tuple[DuplicateName, ...]


@dataclass(frozen=True)
class StageAlignment:
    """Stage ids (0-based) present on both sides and on one side only, sorted."""

    common: tuple[int, ...]
    only_in_baseline: tuple[int, ...]
    only_in_candidate: tuple[int, ...]


def name_key(name: str) -> str:
    """The key two spellings of one entity name share."""
    return name.strip().upper()


def _index(
    entity_type: str, side: str, entries: Sequence[Mapping[str, object]]
) -> _Index:
    index: _Index = {}
    for position, entry in enumerate(entries, start=1):
        entry_id = entry.get("id")
        name = entry.get("name")
        if type(entry_id) is not int or not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"{entity_type}: {side} entry {position} has no integer id "
                "or no non-empty name"
            )
        index.setdefault(name_key(name), []).append((name.strip(), entry_id))
    return index


def _only_in(side: _Index, other: _Index, duplicated: set[str]) -> tuple[str, ...]:
    return tuple(
        side[key][0][0]
        for key in sorted(side)
        if key not in other and key not in duplicated
    )


def _spellings(side: _Index, key: str) -> tuple[str, ...]:
    return tuple(name for name, _ in side.get(key, ()))


def align_by_name(
    entity_type: str,
    baseline: Sequence[Mapping[str, object]],
    candidate: Sequence[Mapping[str, object]],
) -> NameAlignment:
    """Pair the entries of two runs by :func:`name_key`.

    Entries are mappings with an integer ``id`` and a string ``name``; other
    keys are ignored. A ``bool`` is not an integer id.

    Raises:
        ValueError: an entry of either list has no integer ``id`` or no
            non-empty ``name``; the message names *entity_type*. Both lists are
            checked before anything is paired.
    """
    base = _index(entity_type, "baseline", baseline)
    cand = _index(entity_type, "candidate", candidate)
    duplicated = {
        key for index in (base, cand) for key, held in index.items() if len(held) > 1
    }
    pairs = tuple(
        AlignedPair(held[0][0], held[0][1], cand[key][0][1])
        for key, held in base.items()
        if key in cand and key not in duplicated
    )
    duplicates = tuple(
        DuplicateName(key, _spellings(base, key), _spellings(cand, key))
        for key in sorted(duplicated)
    )
    return NameAlignment(
        entity_type,
        pairs,
        _only_in(base, cand, duplicated),
        _only_in(cand, base, duplicated),
        duplicates,
    )


def align_stages(baseline: Iterable[int], candidate: Iterable[int]) -> StageAlignment:
    """Split two runs' stage ids into those both have and those one side has."""
    base, cand = set(baseline), set(candidate)
    return StageAlignment(
        tuple(sorted(base & cand)),
        tuple(sorted(base - cand)),
        tuple(sorted(cand - base)),
    )
