"""Conversion premises for Novomodelo → DECOMP mapcut/cortdeco export.

PREMISES is the single source of truth for the v1 premises. Documentation
quotes it rather than restating it, so code and prose cannot drift apart.

A premise that another premise has taken off the code path is marked DORMANT
in its own text rather than deleted. A manifest that asserts a rule with no
effect on the output pair is an auditability defect, and so is one that drops
a rule the code still carries: naming the dormancy is what distinguishes
the two.

Premises P4 and P5 are DORMANT while P14 holds (pi_gnl block emitted as zeros).
"""

from __future__ import annotations

P1 = (
    "P1: mapcut record indices 4-17 emitted as zeros; WARNING logged; "
    "physical hydro data deferred"
)

P2 = (
    "P2: intercept and every coefficient divided by 1000 (DECOMP is in "
    "10^3 R$); verified to 1 ulp on a matched pair; not configurable"
)

P3 = (
    "P3: n_utv = 0, so no mapcut regs 7/8 and NCOEF omits the pi_qdefp block; "
    "Novomodelo carried no HydroTransitBucket state; travel-time coefficients "
    "not yet supported"
)

P4 = (
    "P4: GNL coefficients negated (Novomodelo negative, DECOMP positive), isolated "
    "in one named, tested, documented function. DORMANT while P14 holds: "
    "cortdeco emits its whole pi_gnl block as zeros, so no real coefficient "
    "reaches this rule and it changes no byte of either output file. "
    "negate_gnl and negate_gnl_array are retained, tested and reachable "
    "for when the GNL mapping is resolved"
)

P5 = (
    "P5: GNL disaggregated across the 3 load blocks weighted by each stage's "
    "own hours. The weights are per-stage, not one triple for the study: "
    "stage 0 is 24/65/79, stages 1-4 are 15/64/89, stage 5 is 12/61/95 and "
    "stage 6 is 51/226/323 over 600 hours rather than 168. DORMANT while "
    "P14 holds: with the pi_gnl block emitted as zeros there is no coefficient "
    "to spread across load blocks, so this rule changes no byte of either "
    "output file. gnl_block_weights is retained and tested for when the "
    "GNL mapping is resolved"
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
    "P12: mapcut reg 9's trailing float64 block emitted as zeros. "
    "The reference populates it with the GNL lag month's hours split across "
    "load blocks, three values per submarket summing to 730.5 = 365.25*24/12, "
    "one average month on the same day-count basis as P9. Two independent "
    "reasons, the first decisive: the values are not derivable from a Novomodelo "
    "case, because that split is a monthly load-block structure belonging to "
    "the DECOMP deck while Novomodelo supplies weekly stage blocks, and no "
    "aggregation of this case's own blocks reproduces the reference triple. "
    "Second, the block's axis is itself unsettled: the populated width is "
    "ngnl*npat while idecomp's reader consumes ngnl*n_estagios. Populating "
    "this block needs a DECOMP-side input, not better Novomodelo parsing."
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
    "P14: cortdeco's pi_gnl coefficients emitted as zeros. Only the values "
    "are zero: the block stays dimensioned by the NCOEF formula and occupies "
    "its full n_sbm_gnl*n_estagios*n_patamares span, and n_sbm_gnl, "
    "codigos_submercados_gnl (int32 on disk despite idecomp surfacing floats), "
    "NCOEF, the record size and the file size are all unchanged. Reducing "
    "Novomodelo's anticipated-thermal ring positions to DECOMP's (submarket, stage, "
    "block) address is unresolved: ring positions sharing a delivery month "
    "carry different coefficients, so they are distinct state variables rather "
    "than copies, and the reference deck comes from an independent run whose "
    "GNL magnitudes span seven orders of magnitude against the oracle's 15% "
    "band and therefore cannot arbitrate between candidate reductions. Every "
    "candidate writes a structurally valid file that DESSEM reads without "
    "complaint, so a wrong one would corrupt the cut's GNL slope invisibly - "
    "unlike P1, P11 and P12, whose divergence is a declared zero. P4 and P5 "
    "are DORMANT while this premise holds."
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
