"""``config.json`` run settings both conversion tracks pin to the same value."""

from __future__ import annotations

from typing import Final

SIMULATION_CUT_SET: Final = "all"
"""``simulation.cut_set``: simulate with every cut training generated.

The source models' cut selection only speeds up the LP solves; the final
simulation evaluates the policy of all the cuts. novomodelo's ``active`` set would
simulate only the cuts its selection kept at the last iteration instead.
"""
