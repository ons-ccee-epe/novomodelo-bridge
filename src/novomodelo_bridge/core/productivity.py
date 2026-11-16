"""The source model hydro productivity functions (presentation-free domain module).

:func:`equivalent_productivity_from_coeffs` is the single settled home of the
equivalent-productivity physics (``ρ_eq = ρ_esp · h_net``, the MW/(m³/s)
coefficient the LP uses for ``generation = ρ_eq · turbined_flow``): mean or
point cota, minus tailrace and hydraulic losses, clamped to a non-negative net
head. It is built from three coefficient-keyed primitives —
:func:`evaluate_cota`, :func:`mean_cota`, :func:`apply_hydraulic_loss` — that
also back the row-keyed wrappers below. Three ways the source model turns a
plant's volume→cota polynomial and tailrace level into a MW/(m³/s)
productivity, all **pure functions over a ``Hidr.cadastro`` row** (``hreg``):

- :func:`compute_productivity` — point ρ at a single reference volume (the LP's
  ``gen = ρ·Q`` coefficient); the source model ``produtibilidade_altura_65``.
- :func:`equivalent_productivity` — PRODT, the mean head over ``[vmin, vmax]``
  (used to convert PENALID R$/MWh penalties to the flow domain); a thin
  coeffs adapter over :func:`equivalent_productivity_from_coeffs`.
- :func:`integrated_productivity` — the volume-integrated ρ over the useful
  range (the source model's stored-energy / EARM convention, used by VminOP).

These were private helpers in ``hydro.py`` that the constraint and FICT-cascade
converters reached into across module boundaries; hosting them here gives them
public names and one home. ``hydro.py`` keeps ``_compute_productivity`` /
``_equivalent_productivity`` / ``_compute_integrated_productivity`` aliases for
its own callers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd

from novomodelo_bridge.core.diagnostics import Diagnostic, DiagnosticTable, Severity
from novomodelo_bridge.core.incremental_inflow import is_artificial_gauge

# --- FPHA turbine efficiency ------------------------------------------------
#
# novomodelo's FPHA production function is ``phi = K · eta · q · h_net`` (MW), with ``K = g /
# 1000`` and ``eta`` the dimensionless turbine efficiency in (0, 1]. The source model
# instead carries the *specific* productivity ``rho_esp`` (MW/((m³/s)·m)), which already
# folds in ``K · eta``. So the efficiency novomodelo needs is ``eta = rho_esp / K`` — the
# value that makes novomodelo's ``phi`` reproduce the source model's ``rho_esp · q · h_liq``.
_GRAVITY_MW_FACTOR = 9.81e-3


def fpha_efficiency(rho_esp: float) -> float:
    """Dimensionless turbine efficiency ``eta = rho_esp / K`` for novomodelo's FPHA,
    capped at ``1.0`` where the source ``rho_esp`` implies an unphysical ``eta > 1``
    (:func:`fpha_efficiency_capped_row` reports it)."""
    return min(rho_esp / _GRAVITY_MW_FACTOR, 1.0)


def fpha_efficiency_capped_row(
    code: int, name: str, rho_esp: float
) -> list[object] | None:
    """The :func:`fpha_efficiency_capped_diagnostic` row of a plant whose
    ``rho_esp`` implies an efficiency above 1, else ``None``."""
    eta = rho_esp / _GRAVITY_MW_FACTOR
    return [code, name, round(eta, 4)] if eta > 1.0 else None


def fpha_efficiency_capped_diagnostic(rows: Sequence[Sequence[object]]) -> Diagnostic:
    """WARNING listing the FPHA plants whose efficiency :func:`fpha_efficiency` capped."""
    return Diagnostic(
        code="fpha-efficiency-capped",
        severity=Severity.WARNING,
        category="Hydro production model",
        title=f"FPHA plants with a turbine efficiency above 1 capped ({len(rows)})",
        summary=(
            "The specific productivity of the listed FPHA plants implies a "
            "turbine efficiency above 1, which Novomodelo's production function "
            "cannot take, so the conversion caps it at 1, and these plants "
            "generate less than in the source model for the same flow and head."
        ),
        table=DiagnosticTable(
            columns=["Code", "Name", "Implied efficiency"],
            rows=rows,
            justify=["right", "left", "right"],
        ),
        remediation="Correct the specific productivity of each listed plant.",
    )


def has_forebay_curve(coeffs: Sequence[float]) -> bool:
    """Whether the volume-elevation polynomial ``a0..a4`` has a nonzero term;
    an all-zero one is missing registry data, and every productivity built
    on it is zero."""
    return any(c != 0.0 for c in coeffs)


def forebay_curve_missing_diagnostic(rows: Sequence[Sequence[object]]) -> Diagnostic:
    """WARNING listing the ``(code, name, stages)`` plants that fail
    :func:`has_forebay_curve`."""
    return Diagnostic(
        code="hydro-forebay-curve-missing",
        severity=Severity.WARNING,
        category="Hydro production model",
        title=f"Plants without a forebay curve ({len(rows)})",
        summary=(
            "Every coefficient of the listed plants' volume-elevation "
            "polynomial is zero, so the case gives them zero productivity on "
            "the listed stages, and they generate nothing there."
        ),
        table=DiagnosticTable(
            columns=["Code", "Name", "Stages"],
            rows=rows,
            justify=["right", "left", "left"],
            caption="Stage ids are 0-based.",
        ),
        remediation="Give each listed plant its volume-elevation polynomial in hidr.dat.",
    )


def fpha_spillage_fit_max_m3s(
    posto: int, monthly_mean_m3s: Sequence[float]
) -> float | None:
    """``fpha_config.spillage_fit_max_m3s`` for a plant gauged at *posto*.

    The source models fit the FPHA spillage coefficient over ``[0, 2·MLT]``, MLT
    the gauge's largest monthly long-term mean natural inflow (*monthly_mean_m3s*,
    one per calendar month). ``None`` on the artificial gauge or a gauge without a
    positive mean: the field is then left out, and novomodelo, like the source models,
    fits over twice the plant's maximum turbined flow.
    """
    if is_artificial_gauge(posto):
        return None
    largest = max((float(m) for m in monthly_mean_m3s), default=0.0)
    return 2.0 * largest if largest > 0.0 else None


# Turbine type code -> kturb exponent used in the head-correction formula. The source
# model codes: 1 = Francis, 2 = Kaplan, 3 = Pelton.  Francis and Pelton share the kturb
# = 0.5 exponent (square-root flow/head response); Kaplan uses 0.2 (gentler response
# thanks to adjustable blades).  Code 0 (= not specified in hidr.dat) falls back to
# Francis.
KTURB_BY_TIPO_TURBINA: dict[int, float] = {0: 0.5, 1: 0.5, 2: 0.2, 3: 0.5}


def evaluate_cota(coeffs: Sequence[float], x: float) -> float:
    """Evaluate the degree-4 cota polynomial ``a0 + a1·x + a2·x² + a3·x³ + a4·x⁴``.

    ``coeffs`` is the 5-element ``a0..a4`` sequence. Generic in ``x``: backs
    both the volume→cota polynomial (hm³ → m) and the structurally identical
    cota→area quartic. No clamp — callers that need a non-negative elevation
    clamp at their own call site.
    """
    a0, a1, a2, a3, a4 = coeffs
    return a0 + a1 * x + a2 * x * x + a3 * x**3 + a4 * x**4


def mean_cota(coeffs: Sequence[float], v_lo: float, v_hi: float) -> float:
    """Return the volume-averaged cota over ``[v_lo, v_hi]`` (hm³ in, m out).

    Computed analytically from the antiderivative of the quartic polynomial —
    the same shape the source model uses to derive ``prodt_eq`` for reservoir
    plants. ``v_hi <= v_lo`` (run-of-river singleton collapse) falls back to
    the point cota at ``v_lo``.

    Each power's difference quotient is expanded,
    ``(v_hi^(k+1) - v_lo^(k+1)) / (v_hi - v_lo) = Σ_j v_hi^j · v_lo^(k-j)``,
    so a range only a rounding error wide (a reservoir drawn to its floor)
    still averages to the point cota rather than cancelling to noise.
    """
    if v_hi <= v_lo:
        return evaluate_cota(coeffs, v_lo)
    return sum(
        a_k / (k + 1) * sum(v_hi**j * v_lo ** (k - j) for j in range(k + 1))
        for k, a_k in enumerate(coeffs)
    )


def apply_hydraulic_loss(h_gross: float, tipo_perda: int, perdas: float) -> float:
    """Return net head after hidr.dat's hydraulic-loss model.

    ``tipo_perda == 1`` -> percentage loss applied to gross head.
    ``tipo_perda == 2`` -> constant head loss in metres.
    ``tipo_perda == 0`` (or any unknown) -> no loss.
    """
    if math.isnan(perdas) or perdas <= 0.0:
        return h_gross
    if tipo_perda == 1:
        return h_gross * (1.0 - perdas / 100.0)
    if tipo_perda == 2:
        return h_gross - perdas
    return h_gross


def hydraulic_generation_mw(
    rho_esp: float, turbined_m3s: float, net_head_m: float, max_generation_mw: float
) -> float:
    """The exact production function ``min(ρ_esp · Q · h_net, G_max)`` (MW).

    A non-positive net head generates nothing.
    """
    return min(rho_esp * turbined_m3s * max(net_head_m, 0.0), max_generation_mw)


def equivalent_productivity_from_coeffs(
    coeffs: Sequence[float],
    *,
    volume_min_hm3: float,
    volume_max_hm3: float,
    rho_esp: float,
    canal_fuga_m: float,
    tipo_perda: int,
    perdas: float,
    reference_volume_hm3: float | None = None,
) -> float:
    """The settled equivalent-productivity physics: ``ρ_eq = ρ_esp · h_net``.

    ``coeffs`` (``a0..a4``) and all scalar inputs are in hidr.dat units: hm³
    for volume, m for cota/tailrace/head. The forebay level anchors at the
    mean cota over ``[volume_min_hm3, volume_max_hm3]`` (:func:`mean_cota`)
    when ``reference_volume_hm3`` is ``None``, or at the point cota there
    (:func:`evaluate_cota`) when a caller supplies one (the point/generation
    anchor DECOMP passes as the plant's initial volume).

    All-zero ``coeffs`` (no forebay curve, :func:`has_forebay_curve`) give
    ``0.0`` before any anchor is computed.

    Net head cannot go negative — a tailrace at or above the forebay level
    means the plant physically cannot generate — so the result is clamped to
    ``0.0`` rather than returning a nonsense negative ``ρ_eq``.
    """
    if not has_forebay_curve(coeffs):
        return 0.0

    if reference_volume_hm3 is None:
        cota = mean_cota(coeffs, volume_min_hm3, volume_max_hm3)
    else:
        cota = evaluate_cota(coeffs, reference_volume_hm3)

    h_gross = cota - canal_fuga_m
    h_net = max(apply_hydraulic_loss(h_gross, tipo_perda, perdas), 0.0)
    return rho_esp * h_net


def compute_productivity(
    hreg: pd.Series,
    *,
    canal_fuga_override: float | None = None,
    cmont_override: float | None = None,
    useful_volume_override: float | None = None,
) -> float:
    """Compute constant point productivity in MW/(m^3/s) for a hydro plant.

    Reads polynomial coefficients ``a0_volume_cota`` through ``a4_volume_cota``
    from the plant's cadastro row to map storage volume (hm3) to upstream height
    (m).  Subtracts the tailrace level to obtain gross drop, applies the loss
    model defined by ``tipo_perda`` and ``perdas``, then multiplies by
    ``produtibilidade_especifica``.

    Reference-volume selection (in priority order):

    1. ``useful_volume_override`` — explicit useful volume (hm³ above
       ``volume_minimo``): ``V = volume_minimo + useful_volume_override``.
    2. Monthly-regulated plants (``tipo_regulacao == "M"``) → 65% of useful
       storage (``V = vmin + 0.65 × (vmax − vmin)``); matches the source model's
       ``produtibilidade_altura_65`` convention.
    3. All other plant types → ``volume_referencia``.

    ``cmont_override`` short-circuits the upstream polynomial entirely.
    """
    coeffs = [float(hreg[f"a{i}_volume_cota"]) for i in range(5)]

    canal_fuga = (
        canal_fuga_override
        if canal_fuga_override is not None
        else float(hreg["canal_fuga_medio"])
    )

    if cmont_override is not None:
        net_drop = cmont_override - canal_fuga
    else:
        if not has_forebay_curve(coeffs):
            return 0.0

        vol_min = float(hreg["volume_minimo"])

        if useful_volume_override is not None:
            net_drop = (
                evaluate_cota(coeffs, vol_min + useful_volume_override) - canal_fuga
            )
        else:
            tipo_regulacao = str(hreg["tipo_regulacao"]).strip()
            vol_max = float(hreg["volume_maximo"])
            if tipo_regulacao == "M":
                v_65 = vol_min + 0.65 * (vol_max - vol_min)
                net_drop = evaluate_cota(coeffs, v_65) - canal_fuga
            else:
                vol_ref = float(hreg["volume_referencia"])
                net_drop = evaluate_cota(coeffs, vol_ref) - canal_fuga

    # No h_net >= 0 clamp: this function has no DECOMP twin, so its clamp
    # behavior is out of scope here — adding one would silently shift the
    # NEWAVE VminOP / FICT-cascade / penalty numbers that consume it.
    tipo_perda = int(hreg["tipo_perda"])
    perdas = float(hreg["perdas"])
    produtibilidade = float(hreg["produtibilidade_especifica"])
    return produtibilidade * apply_hydraulic_loss(net_drop, tipo_perda, perdas)


def equivalent_productivity(
    hreg: pd.Series,
    *,
    canal_fuga_override: float | None = None,
    cmont_override: float | None = None,
) -> float:
    """Compute the source model's PRODT (``produtibilidade_equivalente_volmin_volmax``).

    PRODT is the *equivalent* productivity from the minimum to the maximum
    operative volume (pmo.dat: "PROD. EQUIVALENTE (DO VOL. MINIMO AO VOL.
    MAXIMO)"). Unlike :func:`compute_productivity`, which evaluates the head at a
    single reference volume, PRODT uses the **mean upstream head over the
    useful-volume range**. A thin row→coeffs adapter over
    :func:`equivalent_productivity_from_coeffs`, which owns the settled
    physics (mean-cota anchor, hydraulic loss, and the non-negative net-head
    clamp).

    Run-of-river plants (``Vmax == Vmin``) fall back to the point head at ``Vmin``;
    all-zero polynomials log a WARNING naming the plant and return 0.

    ``canal_fuga_override`` / ``cmont_override`` carry MODIF.DAT CFUGA / CMONT
    temporal overrides: CFUGA replaces the mean tailrace; CMONT pins the upstream
    level to a constant (so the head is ``cmont − cfuga``, no integral) — this
    short-circuit clamps too, for consistency with the coeffs path.
    """
    canal_fuga = (
        canal_fuga_override
        if canal_fuga_override is not None
        else float(hreg["canal_fuga_medio"])
    )
    tipo_perda = int(hreg["tipo_perda"])
    perdas = float(hreg["perdas"])

    if cmont_override is not None:
        h_gross = cmont_override - canal_fuga
        h_net = max(apply_hydraulic_loss(h_gross, tipo_perda, perdas), 0.0)
        return float(hreg["produtibilidade_especifica"]) * h_net

    coeffs = [float(hreg[f"a{i}_volume_cota"]) for i in range(5)]
    return equivalent_productivity_from_coeffs(
        coeffs,
        volume_min_hm3=float(hreg["volume_minimo"]),
        volume_max_hm3=float(hreg["volume_maximo"]),
        rho_esp=float(hreg["produtibilidade_especifica"]),
        canal_fuga_m=canal_fuga,
        tipo_perda=tipo_perda,
        perdas=perdas,
        reference_volume_hm3=None,
    )


def integrated_productivity(
    hreg: pd.Series,
    *,
    canal_fuga_override: float | None = None,
    cmont_override: float | None = None,
) -> float:
    """ρ_esp × ((1/useful) × ∫_vmin^vmax h(V) dV − cf − perdas).

    Mirrors the source model's ``produtibilidade_equivalente_volmin_volmax``: the
    productivity averaged over the full useful storage range, used by the source model
    to convert reservoir volume to stored energy (EARM) and to evaluate VminOP
    constraints.  Different from the point productivity at v_65 that
    :func:`compute_productivity` returns and that the LP uses as the ``gen = ρ·Q``
    coefficient.

    Uses :func:`mean_cota` for the volume-averaged upstream cota. With
    ``cmont_override`` the upstream level is held constant, so the integrated
    drop collapses to ``cmont − cf``.

    Run-of-river plants (vmax == vmin) evaluate the polynomial at the single
    operating point — equivalent to the point productivity.
    """
    if cmont_override is not None:
        cf = (
            canal_fuga_override
            if canal_fuga_override is not None
            else float(hreg["canal_fuga_medio"])
        )
        net_drop = cmont_override - cf
    else:
        coeffs = [float(hreg[f"a{i}_volume_cota"]) for i in range(5)]
        if not has_forebay_curve(coeffs):
            return 0.0

        vmin = float(hreg["volume_minimo"])
        vmax = float(hreg["volume_maximo"])
        cf = (
            canal_fuga_override
            if canal_fuga_override is not None
            else float(hreg["canal_fuga_medio"])
        )

        avg_h = mean_cota(coeffs, vmin, vmax)
        net_drop = avg_h - cf

    # No h_net >= 0 clamp: see compute_productivity — same out-of-scope rule.
    tipo_perda = int(hreg["tipo_perda"])
    perdas = float(hreg["perdas"])
    produtibilidade = float(hreg["produtibilidade_especifica"])
    return produtibilidade * apply_hydraulic_loss(net_drop, tipo_perda, perdas)


def stored_energy_productivity(
    hreg: pd.Series,
    *,
    canal_fuga_override: float | None = None,
    cmont_override: float | None = None,
) -> float:
    """The source model's ``produtibilidade_equivalente_volmin_volmax`` (stored-energy
    ρ).

    The upstream-level reference depends on the regulation type, matching the
    values pmo.dat prints in its ``produtibilidades_equivalentes`` table:

    - **monthly-regulating reservoirs** (``tipo_regulacao == "M"``) use the
      volume-integrated mean head over ``[vmin, vmax]``
      (:func:`integrated_productivity`);
    - **run-of-river / special-regime plants** (``"D"`` / ``"S"``) use the
      **point** productivity at ``volume_referencia`` (:func:`compute_productivity`) —
      the source model does *not* integrate the operative range for them, so the
      volmin→volmax integral over- or under-shoots by a few percent on plants with head
      swing.

    CFUGA / CMONT MODIF overrides are threaded through to whichever primitive
    applies (a CMONT override pins the upstream level in both, collapsing the
    integral/point distinction to ``cmont − canal_fuga``).
    """
    if str(hreg["tipo_regulacao"]).strip() == "M":
        return integrated_productivity(
            hreg,
            canal_fuga_override=canal_fuga_override,
            cmont_override=cmont_override,
        )
    return compute_productivity(
        hreg,
        canal_fuga_override=canal_fuga_override,
        cmont_override=cmont_override,
    )
