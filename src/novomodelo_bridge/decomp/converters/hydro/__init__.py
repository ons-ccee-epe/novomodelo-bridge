"""Hydro conversion for DECOMP-like decks (registry ``hidr.dat`` + ``UH``).

Thin facade — re-exports the public API from :mod:`.entity`, :mod:`.bounds`,
and :mod:`.productivity`.

Converted scope: faithful registry, cascade,
capability, initial storage, and head/productivity, including head-aware
engolimento — with everything whose faithful treatment is still gated on
later features left unconverted: travel time (``VI`` — a separate dadger
register, not an ``AC`` class) and the evaporation/tailrace-polynomial
models (``COTVAZ``/``COTARE``/``COFEVA`` — ``AC`` classes with no DECOMP
consumer). The ``AC`` families' own per-deck coverage (which of them a
given deck actually declares) is reported by ``check decomp``
(:mod:`cobre_bridge.decomp.preflight`); ``VI``, not being an ``AC``
register, is documented only here.
"""

from __future__ import annotations

from cobre_bridge.decomp.converters.hydro.bounds import (
    _ITAIPU_CODE as _ITAIPU_CODE,
)
from cobre_bridge.decomp.converters.hydro.bounds import (
    _build_split_unit_groups as _build_split_unit_groups,
)
from cobre_bridge.decomp.converters.hydro.bounds import (
    _rated_envelope as _rated_envelope,
)
from cobre_bridge.decomp.converters.hydro.bounds import (
    convert_hydro_group_availability as convert_hydro_group_availability,
)
from cobre_bridge.decomp.converters.hydro.bounds import (
    convert_itaipu_frequency_min_generation as convert_itaipu_frequency_min_generation,
)
from cobre_bridge.decomp.converters.hydro.entity import (
    _downstream_operated as _downstream_operated,
)
from cobre_bridge.decomp.converters.hydro.entity import (
    _evaporation_coefficients_mm as _evaporation_coefficients_mm,
)
from cobre_bridge.decomp.converters.hydro.entity import (
    _evaporation_flag_codes as _evaporation_flag_codes,
)
from cobre_bridge.decomp.converters.hydro.entity import (
    _operated_initial_volumes as _operated_initial_volumes,
)
from cobre_bridge.decomp.converters.hydro.entity import (
    convert_hydros as convert_hydros,
)
from cobre_bridge.decomp.converters.hydro.entity import (
    convert_initial_storage as convert_initial_storage,
)
from cobre_bridge.decomp.converters.hydro.entity import (
    read_hidr as read_hidr,
)
from cobre_bridge.decomp.converters.hydro.productivity import (
    convert_energy_productivity as convert_energy_productivity,
)
from cobre_bridge.decomp.converters.hydro.productivity import (
    convert_production_models as convert_production_models,
)
