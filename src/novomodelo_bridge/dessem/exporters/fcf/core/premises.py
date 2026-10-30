"""Conversion premises for Novomodelo → DECOMP mapcut/cortdeco export.

PREMISES is the single source of truth for the v1 premises. Documentation
quotes it rather than restating it, so code and prose cannot drift apart.

A premise that another premise has taken off the code path is marked DORMANT
in its own text rather than deleted. A manifest that asserts a rule with no
effect on the output pair is an auditability defect, and so is one that drops
a rule the code still carries: naming the dormancy is what distinguishes
the two.
"""

from __future__ import annotations

P1 = (
    "P1: mapcut record indices 4-17 emitted as zeros; WARNING logged; "
    "physical hydro data deferred"
)

P2 = (
    "P2: the intercept, pi_varm and pi_qdefp are divided by 1000 (DECOMP is in "
    "10^3 R$); verified to 1 ulp on a matched pair; not configurable. pi_gnl is "
    "not divided: it prices energy and is written in R$/MWh"
)

P3 = (
    "P3: each hydro with a travel time to its downstream plant is a mapcut "
    "travel-time plant, numbered in the order of the source deck's VI records; "
    "the export is refused when such a plant has no VI record, its travel time "
    "is not a whole number of hours, or it has no downstream plant. Reg 6 holds "
    "each stage's last release lag per plant, stage-major with plants inner, "
    "then the reg 7/8 index where each plant's stage begins. Regs 7/8 hold, per "
    "plant, stage and lag, the share of that lag's release arriving within the "
    "stage, for a release spread evenly over its stage and arriving that many "
    "hours later over an equally long span; periods before the study are as "
    "long as its first stage. cortdeco's pi_qdefp prices each plant's release "
    "over max_lag lags, lag-major with plants inner, in R$ per hm3 released: "
    "the downstream plant's HydroTransitBucket coefficients, each weighted by "
    "the share of the release that is still in transit at the stage's end and "
    "arrives within the bucket's window. The export is refused when a pool's "
    "buckets do not cover that water or belong to no travel-time plant's "
    "downstream plant. Regs 6-8 match the reference DECOMP's own mapcut byte "
    "for byte, and pi_qdefp its cortdeco to 0.2%"
)

P4 = (
    "P4: GNL coefficients are negated once, when the cut is serialized "
    "(Novomodelo negative, DECOMP positive), in one named, tested, documented "
    "function, negate_gnl_array"
)

P5 = (
    "P5: the converted GNL value is written unchanged on every load block of "
    "the delivery, because Novomodelo prices one value per delivery; it is "
    "never split by block hours"
)

P6 = "P6: submarket = bus_id + 1; bus 5 (IV) excluded; n_submercados = 5"

P7 = (
    "P7: only trunk pools converted; terminal pool (last pool_id, the "
    "warm-start fan) excluded"
)

P8 = (
    "P8: inflow-lag coefficients dropped with a counted audit report, "
    "never a silent truncation"
)

P9 = (
    "P9: discount rate duration-proportional (1+r)^(-cumulative_days/365.25); "
    "the day-count basis is 365.25, solved from the reference series and "
    "exact to 3.7e-13, not 365"
)

P10 = (
    "P10: every populated cut past the warm-start slots emitted; is_active "
    "recorded as an ECO column, not used as a filter"
)

P11 = (
    "P11: mapcut reg 10 emits zeros for parcela_custo_geracao_termica_minima, "
    "parcela_custo_contrato_importacao_minimo, "
    "parcela_custo_contrato_exportacao_minimo, "
    "geracao_termica_minima_sinalizada_gnl and "
    "geracao_termica_minima_gerada_gnl; they are DECOMP operational quantities "
    "from the deck's own data, non-zero in the reference, and are not "
    "derivable from a Novomodelo policy checkpoint, whose cut intercept already "
    "embeds the constant term. Only taxa_desconto is filled."
)

P12 = (
    "P12: mapcut reg 9's trailing float64 block holds the delivery stage's "
    "block hours from stages.json, patamar-major with GNL entries inner: "
    "ngnl*npat values, each patamar's hours once per GNL plant, then record "
    "padding. The delivery stage is the one whose delivery P14 prices."
)

P13 = (
    "P13: cortdeco holds numero_cortes + 1 records, and the extra one "
    "duplicates the last cut-building node's last cut. The reference deck "
    "holds the next backward-pass cut there, written but not yet exposed in "
    "the head table, which a checkpoint with a whole number of completed "
    "iterations cannot supply. Nothing points to that record, since the "
    "chains only step backwards from the heads. A duplicate is mathematically "
    "inert, because a repeated hyperplane adds nothing to an FCF, while a "
    "zero-filled record would fabricate theta >= 0."
)

P14 = (
    "P14: for each GNL plant in mapcut reg 9 order, the first cut-building "
    "node's cuts carry the coefficient the first cut-building pool prices on "
    "the delivery that starts with the case's second stage, found by its "
    "delivery interval, divided by that stage's block-hours sum, at the first "
    "position of the pi_gnl block's middle axis; every other position and "
    "every later node's cut is 0.0. The first cut-building pool's other priced "
    "deliveries and every GNL coefficient of the later cut-building pools are "
    "dropped without being folded into the cut intercept, with a counted audit "
    "report per pool, never a silent truncation, so the exported coefficient "
    "is not the whole of the policy's GNL value. A plant with no such delivery "
    "is written as 0.0 with a warning, and an exact zero is written unchanged "
    "with a warning. The export is refused when a plant has two slots on that "
    "delivery, when a single-stage case has GNL plants, or when a slot's hours "
    "disagree with the stage's"
)

P15 = (
    "P15: mapcut reg 9's lag_meses_gnl is each GNL plant's dispatch-anticipation "
    "lag in whole months, read from the source deck's dadgnl NL record. The "
    "case's lead_time_hours is not converted back to months: it is measured from "
    "the deck's own calendar, so no single hours-to-months rule recovers the "
    "deck's lag. The deck's GNL plants pair with the case's anticipated thermals "
    "in ascending code order, and the export is refused when their count or "
    "names differ or a plant has no NL record."
)


PREMISES: tuple[str, ...] = (
    P1,
    P2,
    P3,
    P4,
    P5,
    P6,
    P7,
    P8,
    P9,
    P10,
    P11,
    P12,
    P13,
    P14,
    P15,
)
