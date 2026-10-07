"""Dimensional conversion constants shared across the conversion tracks."""

from __future__ import annotations

# The source model convention (manual §3.24, used in C_M3S2HM3)
MONTH_HOURS: float = 730.0
C_M3S2HM3: float = MONTH_HOURS * 3600.0 / 1e6  # = 2.628 hm³ / (m³/s · month)
# HM3 × ρ_MW_per_m3s → MWh conversion (purely volumetric; 730 cancels here).
HM3_TO_MWH_PER_RHO: float = 1e6 / 3600.0  # ≈ 277.78
