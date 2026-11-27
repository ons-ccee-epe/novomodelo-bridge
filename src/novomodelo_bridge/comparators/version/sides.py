"""Reading one side of a version comparison.

One function reads a side and serves both, so the tool carries no
baseline/candidate asymmetry of its own. The failure order for one side is
part of the exit-code contract, first failure wins: the directory is missing
(:class:`~novomodelo_bridge.core.errors.SourceFileError`), it is not a
Novomodelo output (:class:`~novomodelo_bridge.core.errors.NovomodeloOutputError`),
``hydros.json`` then ``buses.json`` is absent or malformed (``SourceFileError``
or :class:`~novomodelo_bridge.core.errors.FieldParseError`), the probe cannot
read a present file
(:class:`~novomodelo_bridge.novomodelo.readers.NovomodeloReadError`).
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from novomodelo_bridge.core import diagnostics as dx
from novomodelo_bridge.core.errors import (
    FieldParseError,
    NovomodeloOutputError,
    SourceFileError,
)
from novomodelo_bridge.novomodelo.probe import is_novomodelo_output, probe_output
from novomodelo_bridge.novomodelo.readers import (
    novomodelo_software_version,
    read_novomodelo_required_entities,
    read_novomodelo_stage_risk_aversion,
    read_novomodelo_training_metadata,
)

_LOG = logging.getLogger(__name__)

_PREFIX = {"baseline": "Baseline output: ", "candidate": "Candidate output: "}

_THERMALS_EFFECT = "the comparison leaves out the thermal plants of this output."
_LINES_EFFECT = "the comparison leaves out the lines of this output."


@dataclass(frozen=True)
class SideInputs:
    """What one side of a version comparison holds.

    Attributes:
        thermals: ``None`` when ``system/thermals.json`` is unreadable, so the
            entity type is unavailable on this side; ``[]`` when the case has none.
        lines: As *thermals*, for ``system/lines.json``.
        risk_aversion: What ``read_novomodelo_stage_risk_aversion`` returned:
            ``{}`` reads as risk-neutral, ``None`` as risk measure unknown.
        partitions: Columns of each probed partition, ``None`` when absent.
    """

    side: str
    output_dir: Path
    version: str | None
    hydros: list[dict]
    buses: list[dict]
    thermals: list[dict] | None
    lines: list[dict] | None
    risk_aversion: dict[int, float] | None
    partitions: dict[str, frozenset[str] | None]

    @property
    def usable_entity_lists(self) -> dict[str, bool]:
        """Whether each optional entity type, by the name ``align_by_name`` takes,
        has a list to align. An empty list is usable; only ``None`` is not."""
        return {"thermal": self.thermals is not None, "line": self.lines is not None}


def require_output_dir(output_dir: Path, side: str) -> None:
    """Raise :class:`SourceFileError` when *output_dir* is not a directory."""
    if not output_dir.is_dir():
        raise SourceFileError(
            f"The {side} output directory was not found: {output_dir}.",
            path=str(output_dir),
        )


@contextmanager
def side_tagged(side: str) -> Generator[None]:
    """Prefix the summary of every Diagnostic emitted inside the block with the
    side's name, and re-emit it to the enclosing sink when the block ends for
    any reason."""
    prefix = _PREFIX[side]
    sink: list[dx.Diagnostic] = []
    try:
        with dx.collect() as sink:
            yield
    finally:
        # After the inner collect() exits: inside it, the emit would append to
        # the inner sink and never reach the enclosing one.
        for diagnostic in sink:
            dx.emit(
                replace(diagnostic, summary=prefix + diagnostic.summary), logger=_LOG
            )


def _optional_entities(
    output_dir: Path, filename: str, effect: str
) -> list[dict] | None:
    """``system/<filename>``'s entries: ``[]`` when the file is absent (a case
    without that entity type), ``None`` with a warning when it is unreadable."""
    try:
        return read_novomodelo_required_entities(output_dir, filename)
    except SourceFileError:
        return []
    except FieldParseError as exc:
        assert exc.path is not None  # read_novomodelo_required_entities always sets it
        dx.emit(
            dx.report_file_unreadable_diagnostic(
                f"system/{filename}", Path(exc.path), effect, exc
            ),
            logger=_LOG,
        )
        return None


def read_side(output_dir: Path, side: str) -> SideInputs:
    """Read one side of a version comparison, ``side`` being a member of
    :data:`~novomodelo_bridge.comparators.version.labels.SIDES`.

    Reader Diagnostics reach the enclosing sink tagged with the side. Fails in
    the order the module documents.
    """
    require_output_dir(output_dir, side)
    if not is_novomodelo_output(output_dir):
        raise NovomodeloOutputError(
            f"The {side} directory {output_dir} holds neither training/ nor "
            "simulation/, so it is not a Novomodelo output.",
            path=str(output_dir),
        )
    hydros = read_novomodelo_required_entities(output_dir, "hydros.json")
    buses = read_novomodelo_required_entities(output_dir, "buses.json")
    with side_tagged(side):
        thermals = _optional_entities(output_dir, "thermals.json", _THERMALS_EFFECT)
        lines = _optional_entities(output_dir, "lines.json", _LINES_EFFECT)
        version = novomodelo_software_version(
            read_novomodelo_training_metadata(output_dir)
        )
        risk_aversion = read_novomodelo_stage_risk_aversion(output_dir)
        partitions = probe_output(output_dir)
    return SideInputs(
        side=side,
        output_dir=output_dir,
        version=version,
        hydros=hydros,
        buses=buses,
        thermals=thermals,
        lines=lines,
        risk_aversion=risk_aversion,
        partitions=partitions,
    )
