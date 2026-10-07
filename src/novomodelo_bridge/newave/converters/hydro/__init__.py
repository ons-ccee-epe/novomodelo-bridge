"""Hydro entity conversion: the source model hydro plant data to Cobre's
``hydros.json``, ``hydro_bounds.parquet``, and ``hydro_production_models.json``.

Thin facade — defines nothing itself. Re-exports the package's public API from
:mod:`.entity` (the ``convert_hydros`` builder and its local leaf helpers),
:mod:`.bounds` (plant-capacity caps and the per-stage bounds converters),
:mod:`.geometry` (VHA geometry and FPHA eligibility), :mod:`.overrides`
(MODIF.DAT/GHMIN.DAT/PENALID.DAT readers), and :mod:`.productivity`
(equivalent-productivity machinery and the production-model JSON).
"""

from __future__ import annotations

from cobre_bridge.newave.converters.hydro.bounds import (
    _compute_max_turbined_head_corrected as _compute_max_turbined_head_corrected,
)
from cobre_bridge.newave.converters.hydro.bounds import (
    _per_stage_turbined_envelope as _per_stage_turbined_envelope,
)
from cobre_bridge.newave.converters.hydro.bounds import (
    _reduced_caps as _reduced_caps,
)
from cobre_bridge.newave.converters.hydro.bounds import (
    convert_storage_bounds as convert_storage_bounds,
)
from cobre_bridge.newave.converters.hydro.bounds import (
    convert_turbined_bounds_head_corrected as convert_turbined_bounds_head_corrected,
)
from cobre_bridge.newave.converters.hydro.bounds import (
    convert_water_withdrawal as convert_water_withdrawal,
)
from cobre_bridge.newave.converters.hydro.entity import (
    convert_hydros as convert_hydros,
)
from cobre_bridge.newave.converters.hydro.geometry import (
    _is_fpha_eligible as _is_fpha_eligible,
)
from cobre_bridge.newave.converters.hydro.geometry import (
    fpha_eligible_codes as fpha_eligible_codes,
)
from cobre_bridge.newave.converters.hydro.geometry import (
    generate_hydro_geometry as generate_hydro_geometry,
)
from cobre_bridge.newave.converters.hydro.overrides import (
    _apply_permanent_overrides as _apply_permanent_overrides,
)
from cobre_bridge.newave.converters.hydro.overrides import (
    _extract_temporal_overrides as _extract_temporal_overrides,
)
from cobre_bridge.newave.converters.hydro.overrides import (
    _read_ghmin_per_stage as _read_ghmin_per_stage,
)
from cobre_bridge.newave.converters.hydro.overrides import (
    read_cadastro as read_cadastro,
)
from cobre_bridge.newave.converters.hydro.productivity import (
    _compute_integrated_productivity as _compute_integrated_productivity,
)
from cobre_bridge.newave.converters.hydro.productivity import (
    _compute_productivity as _compute_productivity,
)
from cobre_bridge.newave.converters.hydro.productivity import (
    _equivalent_productivity as _equivalent_productivity,
)
from cobre_bridge.newave.converters.hydro.productivity import (
    _fpha_computed_config as _fpha_computed_config,
)
from cobre_bridge.newave.converters.hydro.productivity import (
    _parse_fpha_plane_reduction as _parse_fpha_plane_reduction,
)
from cobre_bridge.newave.converters.hydro.productivity import (
    _per_stage_productivities as _per_stage_productivities,
)
from cobre_bridge.newave.converters.hydro.productivity import (
    compute_base_productivities as compute_base_productivities,
)
from cobre_bridge.newave.converters.hydro.productivity import (
    # Self-alias re-export (constraints.py imports it): the name itself, doubled
    # by the mandatory re-export idiom, exceeds line-length.
    compute_per_stage_own_integrated_productivities as compute_per_stage_own_integrated_productivities,  # noqa: E501
)
from cobre_bridge.newave.converters.hydro.productivity import (
    compute_per_stage_prodt_sin_mean as compute_per_stage_prodt_sin_mean,
)
from cobre_bridge.newave.converters.hydro.productivity import (
    compute_prodt_sin_mean as compute_prodt_sin_mean,
)
from cobre_bridge.newave.converters.hydro.productivity import (
    convert_hydro_energy_productivity as convert_hydro_energy_productivity,
)
from cobre_bridge.newave.converters.hydro.productivity import (
    convert_production_models as convert_production_models,
)
