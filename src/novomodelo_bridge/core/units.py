"""Dimensional conversion constants shared across the conversion tracks."""

from __future__ import annotations

# The source model's mean month, 365.25 × 24 / 12 h: the unit its future-cost
# cuts are written in (DECOMP reference manual §4.2, "mês médio ... igual a
# 730.5 horas, de modo a manter a coerência com as unidades dos cortes").
MEAN_MONTH_HOURS: float = 365.25 * 24.0 / 12.0
C_M3S2HM3: float = MEAN_MONTH_HOURS * 3600.0 / 1e6  # hm³ per (m³/s · mean month)
# HM3 × ρ_MW_per_m3s → MWh conversion (purely volumetric; the month cancels here).
HM3_TO_MWH_PER_RHO: float = 1e6 / 3600.0  # ≈ 277.78
