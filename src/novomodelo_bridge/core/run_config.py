"""``config.json`` run settings both conversion tracks pin to the same value."""

from __future__ import annotations

from typing import Final

SIMULATION_CUT_SET: Final = "active"
"""``simulation.cut_set``: simulate with the cuts active when training ended.

``all`` would also load every cut the selection deactivated, making the
simulation LPs larger than the training ones. Where a conversion turns on no
selection every cut stays active, so the two sets coincide.
"""
