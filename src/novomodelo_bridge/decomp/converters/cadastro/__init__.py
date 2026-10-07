"""AC-override ingestion and the per-stage-effective cadastro view.

Thin facade — defines nothing itself; re-exports the public API from
:mod:`.stage_resolution`, :mod:`.overrides`, :mod:`.effective`.
"""

from __future__ import annotations

from cobre_bridge.decomp.converters.cadastro.effective import (
    CadastroResolutionReport as CadastroResolutionReport,
)
from cobre_bridge.decomp.converters.cadastro.effective import (
    EffectiveCadastro as EffectiveCadastro,
)
from cobre_bridge.decomp.converters.cadastro.effective import (
    build_effective_cadastro as build_effective_cadastro,
)
from cobre_bridge.decomp.converters.cadastro.effective import (
    effective_storage_range as effective_storage_range,
)
from cobre_bridge.decomp.converters.cadastro.effective import (
    is_reservoir as is_reservoir,
)
from cobre_bridge.decomp.converters.cadastro.effective import (
    storage_envelope as storage_envelope,
)
from cobre_bridge.decomp.converters.cadastro.effective import (
    unregulated_runofriver_codes as unregulated_runofriver_codes,
)
from cobre_bridge.decomp.converters.cadastro.overrides import (
    _SCALAR_AC_SPECS as _SCALAR_AC_SPECS,
)
from cobre_bridge.decomp.converters.cadastro.overrides import (
    APPLIED_AC_CLASSES as APPLIED_AC_CLASSES,
)
from cobre_bridge.decomp.converters.cadastro.overrides import (
    UNINGESTABLE_AC_CLASSES as UNINGESTABLE_AC_CLASSES,
)
from cobre_bridge.decomp.converters.cadastro.overrides import (
    DiversionChannel as DiversionChannel,
)
from cobre_bridge.decomp.converters.cadastro.overrides import (
    MachineSet as MachineSet,
)
from cobre_bridge.decomp.converters.cadastro.overrides import (
    OutOfHorizon as OutOfHorizon,
)
from cobre_bridge.decomp.converters.cadastro.overrides import (
    _forward_fill_series as _forward_fill_series,
)
from cobre_bridge.decomp.converters.cadastro.overrides import (
    _read_machine_set_overrides as _read_machine_set_overrides,
)
from cobre_bridge.decomp.converters.cadastro.overrides import (
    _read_polynomial_overrides as _read_polynomial_overrides,
)
from cobre_bridge.decomp.converters.cadastro.overrides import (
    _read_scalar_overrides as _read_scalar_overrides,
)
from cobre_bridge.decomp.converters.cadastro.stage_resolution import (
    resolve_effective_stage as resolve_effective_stage,
)
