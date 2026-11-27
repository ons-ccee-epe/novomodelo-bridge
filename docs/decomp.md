# The DECOMP track

`convert decomp`, `check decomp`, and `compare decomp` work on a DECOMP deck
revision. This page covers what the converter reads, what it leaves out and
how it tells you, the boundary cost-to-go import that runs by default, and the
one extra rule for running the converted case. Flags and their help text are
in the [CLI reference](cli.md).

## Inputs

`SRC` is the deck directory. `caso.dat` names the revision (`rv0`, `rv1`, and
so on) and the revision's index file names that revision's data files:
`dadger.rvN` and `vazoes.rvN`, the shared `hidr.dat` registry, the
long-term mean inflows `mlt.dat` when the index lists them, `dadgnl.rvN`
when the deck has fuel-constrained thermals, and the cut files when the deck
declares them. The deck is parsed once and shared by every converter.

The CSV inputs of the LIBS family are found through the LIBS index that the
`FA` register of `dadger.rvN` names. Each line of the index names a record key
and the file that holds it, and the converter reads each LIBS record from the
file listed for its key: the `PARQUE-EOLICO-EQUIVALENTE-*` records (the
renewable plants, which must all be in one file),
`HIDRELETRICA-CADASTRO-RESERVATORIO-CURVAJUSANTE` (the tailrace curves) and
`RESTRICAO-ELETRICA-ESPECIAL` (the electrical constraints). Lines starting with
`&` are comments. The `FJ` register can name the tailrace-curve file too, with
or without an index; when both name one, they must agree. Apart from that file,
a deck without `FA` has no LIBS records. An index key listed without a file
name, or a declared file that is not in the deck directory, makes the deck fail
to convert. A file with a usual LIBS name, such as `renovaveis.csv`, that
nothing declares is ignored, and `check decomp` and `convert decomp` report it
in a `libs-unindexed-file` note (`-v` prints it in full). The index is read as
UTF-8 when it is valid UTF-8 and as Latin-1 otherwise. In the
electrical-constraint file a card name is recognized only in upper case,
as the manual writes it, and `check decomp` and `convert decomp` warn about
every line that is not a card it reads (`libs-unread-lines`).

`check decomp` validates the deck without writing anything and also lists what
the conversion will leave behind for this particular deck. Exit 0 means ready,
1 ready with warnings, 2 the deck will not convert.

`dadger.rvN` and `dadgnl.rvN` are read by fixed columns, so a tab where the
layout expects spaces shifts the fields after it, and a register line that
starts with one is not read at all. `check decomp` and `convert decomp` warn
about every tab outside a comment line (one whose first non-blank character
is `&`) and name the file and lines.

The register-by-register map from the deck to the converted case, including
what the converter does not convert yet, is the
[DECOMP data map](decomp-data-map.md) (in Portuguese).

### Electrical constraints

The electrical-constraint file is translated into novomodelo's vocabulary and
entity ids and written as generic constraints. When the file declares any
expression, alias, restriction or activation rule, `convert decomp` writes:

- in `constraints/generic_constraints.json`, after the constraints of the `RE`,
  `HQ`, `HV`, `HE` and `RI` registers, the constraints of the restrictions, by
  id, and, among its expressions, those the constraints read that are more than
  arithmetic over data, followed by the formulas `re(R)` reads;
- in `constraints/generic_constraint_bounds.parquet`, after the rows of the
  other constraints, the rows that say where each of these constraints holds
  and its limits there;
- entries of `constraints/generic_parameters.json`, after the parameters the
  other conversions declare, when the constraints read an alias, a data term
  or an expression of arithmetic over data.

The short-form record names (`RE`, `RE-*`, `ALIAS-ELET*`) are read as the
long-form cards they stand for. The cards that give a horizon or limits by date
(`RESTRICAO-ELETRICA-HORIZONTE-DATA`,
`RESTRICAO-ELETRICA-LIMITES-FORMULA-DATA-PATAMAR`, and their short forms
`RE-HORIZ-DATA` and `RE-LIM-FORM-DATA-PAT`) are read as well: a date becomes the
stage that contains it, and a range runs from the stage that contains its start
date to the stage that contains its end date, the end date included. A date
outside the case's stages makes the deck fail to convert, with an error that
names the restriction or alias code and the date. A range of stages or dates
that ends before it starts also fails, with an error that names the card and
the restriction or alias code. The formula cards by period and by date
(`RESTRICAO-ELETRICA-FORMULA-PERIODO-PATAMAR`,
`RESTRICAO-ELETRICA-FORMULA-DATA-PATAMAR`, and their short forms `RE-PER-PAT`
and `RE-DATA-PAT`) are read as formula overrides: each row gives a formula that
replaces the restriction's formula over the stages and the load block of the
row, with the same date mapping and the same errors. A restriction declared
only by those rows is a formula restriction. Rows that do not fit the
restriction's card shape, such as formula or limit rows on an inequality
restriction, are not applied, and the `decomp-electrical-period-rows-unused`
warning reports them.

Each text of the file (a formula, an inequality side, an override, an
expression, an activation-rule condition) is parsed and written again with
every term replaced as the table shows, one space around each binary operator,
and each `+` or `-` sign written against its operand. Nothing is inlined,
folded or evaluated: the author's parentheses, signs and number spellings stay,
and novomodelo evaluates the conditionals and the activation-rule conditions.
Only entity codes are translated; a block argument `k` is copied as written.

| Deck term | Converted term | Codes translated through |
| --- | --- | --- |
| `ger_usih(c[, k])` | `hydro_generation(id[, k])` | hydro plants |
| `ger_usit(c[, k])` | `thermal_generation(id[, k])` | thermal plants |
| `ger_pee(c[, k])` | `non_controllable_generation(id[, k])` | renewable parks |
| `disp_usih(c[, k])` | `hydro_availability(id[, k])` | hydro plants |
| `val_demanda(s[, k])` | `bus_load_value(id[, k])` | submarkets |
| `val_demanda_sin([k])`, with or without parentheses | the sum of `bus_load_value(id[, k])` over every bus of the submarket table, in parentheses, in ascending id order | none |
| `demanda(s[, k])`, inside a condition | `bus_load_value(id[, k])` | submarkets |
| `demanda_sin([k])`, with or without parentheses, inside a condition | the same sum as `val_demanda_sin([k])` | none |
| `ener_comerc(c[, k])` | `contract_import(id[, k])` or `contract_export(id[, k])`, by the contract's declared kind | contracts |
| `ener_interc(x, y)` | `line_flow(source_bus=id(x), target_bus=id(y))` | submarkets |
| `ger_conjh(c, n[, k])` | `hydro_generation(id(c)[, k], unit_group=g)` | hydro plants, then the plant's machine sets |
| `se(c, a, b)` | `if(c, a, b)`, the condition and both branches translated | none |
| `re(R)` | `@restriction_<R>_formula` | none |
| an expression name | `@<name>` | none |
| an alias name | `@<name>` | none |
| `carga_ande` or `carga_ande()` | `@carga_ande` | none |
| `peq_<series>_<submarket>`, with or without `()` | `@peq_<series>_<submarket>` | none |

novomodelo reads the load only as data, `bus_load_value`, and has no term for
the system load: the conversion writes it in place as the sum of one
`bus_load_value` call per bus of the submarket table, each bus once, with the
block argument copied into every call. With one bus the sum is that call alone,
without parentheses. In DECOMP, `demanda` and `demanda_sin` are the load as an
LP term, so they convert only inside a condition, where they read the load
value. A condition is the text of an activation rule and the first argument of
a `se(...)`, with everything written inside it; the branches of a `se` are
inside a condition only when the `se` itself is. Anywhere else, `demanda` or
`demanda_sin` makes the deck fail to convert, with an error that names where it
is written and the term; write `val_demanda` or `val_demanda_sin`, which read
the same load as data, in its place. An expression that writes `demanda` or
`demanda_sin` outside a `se` condition of its own fails the conversion, even
when only an activation rule reads it.

Each entity code becomes the id the conversion gave that entity. The hydro
table holds every converted plant. The thermal table holds the `CT` plants and
the `dadgnl` GNL plants; a GNL code that is also a `CT` code fails the
conversion. The renewable-park table holds the parks of the
`PARQUE-EOLICO-EQUIVALENTE-*` records. The submarket table holds the `SB`
subsystems and the transshipment bus under code 6, the code the formulas cite
it by, unless a subsystem already has code 6. Machine set `n` of a plant is its
`n`-th unit group by ascending group id. A contract number becomes that
contract's id; a number that more than one contract declares is left out of the
table, and the `decomp-libs-electrical-contract-code-ambiguous` info diagnostic
lists it. In a text the case holds, a code with no entity in the converted
case, a machine set the plant does not have, an entity code that is not a
non-negative integer (`66.0` included), a block argument that is not a number,
or a call with the wrong arguments makes the deck fail to convert, with an
error that names where the call is written (`restriction <code>` for the
relation of one of its constraints, `restriction <code> formula` for the
formula `re(<code>)` reads, `expression <name>` or `activation rule <code>`)
and the call. A name or function the table does not list, and a data term
written with arguments, also fail the conversion, with an error that names the
restriction, expression or rule and shows the text. A text of a restriction, an
override, an expression or an activation rule, assigned or not, that does not
parse fails it too, with an error that shows the text and the position where it
stops parsing.

Each restriction becomes the generic constraint `constraint_<code>`, whose id
is the restriction code. Its relation is the restriction's formula or
inequality, or, in a cell (a stage and a block, each patamar becoming a 0-based
block id) that an override covers, the override's. Its slack follows the
violation treatment of the restriction, as below, and the condition of the
activation rule the restriction is assigned is its `active_when` condition,
which novomodelo evaluates. A restriction whose relation or violation treatment
is not the same in every cell where it holds a row is written as several
constraints `constraint_<code>_<k>`, `k` from 1, one per relation and treatment
in the order of their first cell, each holding only at its own cells,
contiguous or not. The relations are compared as the deck writes them, so two
spellings of one relation are two constraints. These constraints take ids above
every id of the other generic constraints and every restriction code of the
file, in code order and then `k` order, so the restriction code is the id of
none of them; a restriction written as one constraint whose code another
generic constraint of the case already has as id takes such an id too, and
keeps its name. The `decomp-electrical-restrictions-split` info diagnostic
lists each restriction not written as one constraint under its own code, with
the id, the name and the cells of each of its constraints.

`re(R)` reads the expression `restriction_<R>_formula`, which holds the
translated formula of restriction `R`, or its left side when `R` is an
inequality, and never its overrides; `R` must be a declared restriction with a
formula or an inequality of its own, not one declared only by per-period rows,
or the deck fails to convert. Expressions, aliases and data terms keep the
deck's names, which `@<name>` cites. A name is read first as an expression,
then as an alias, then as a term of the table. An alias that an expression of
the same name hides is not converted, and the
`decomp-electrical-alias-shadowed` info diagnostic lists it. A data term the
formulas cite that the deck also declares as an alias or expression reads that
declaration, and the `decomp-electrical-data-alias-shadowed` info diagnostic
lists it. A name that is not a novomodelo identifier, or that a parameter of
another conversion or a generated `restriction_<R>_formula` already takes, is
changed: each character other than a letter, a digit or `_` becomes `_`, a `_`
is put in front when the result does not start with a letter or `_`, and the
smallest free suffix `_2`, `_3`, … is added on a collision. The names that need
no change are kept first; the others are changed in the order expressions,
aliases, data terms, each sorted by name. The
`decomp-electrical-name-sanitized` info diagnostic lists each changed name with
the name it is emitted as, and every text cites it under that name. A generated
`restriction_<R>_formula` that another conversion already declares fails the
conversion.

Only what a constraint of the case reads is converted: the aliases,
expressions, data terms and `re(R)` formulas that the relation of one of its
cells, or the condition of its restriction's assigned rule, cites, directly or
through other expressions. The declared aliases and expressions that no
restriction, override or assigned rule cites, and the activation rules no
restriction is assigned, are left out, and the
`decomp-electrical-unread-omitted` info diagnostic lists them. Anything cited
only by a restriction that holds no row, or only by an override that governs
none of the cells where its restriction holds a row, is left out too, and no
diagnostic lists it.

Each reached alias and data term is a `per_stage_block` parameter with a value
at every stage and block of the study; an alias's value for one block wins over
its value for every block of the stage. The data terms take their values from
the deck: `carga_ande` from the ANDE load of the `RI` register, which also
stays added to the load of the Itaipu submarket's bus, and
`peq_<series>_<submarket>` from the generation of the `PQ` series, with
`<submarket>` the subsystem's name in the `SB` register. Two `PQ` series that
give the same name fail the conversion. A reached expression of arithmetic
(`+ - * /`, signs, parentheses and numbers) over aliases, data terms and other
such expressions is an `expression` parameter. Every other reached expression,
such as one that reads a load or a generation, is a record of the `expressions`
of `constraints/generic_constraints.json`, followed there by the
`restriction_<R>_formula` expressions. A cycle among the reached expressions
fails the conversion, including one through the formula an `re(R)` reads, as
when the formula of `R` cites `re(R)` itself; the error lists the cycle, each
such formula by its `restriction_<R>_formula` name.

An alias or a data term may have values for only part of the study. A cell, a
stage and a block, reads a name when a restriction holds a row there whose
governing text cites it, directly or through expressions and `re(R)`. A
restriction holds rows only in the stages of its horizon. In each cell the
overrides for that block govern, else the overrides for every block, else the
restriction's own text, and a formula restriction holds a row only where its
limit for the block, else its limit for every block, sets a side. The condition
of the assigned rule is read wherever the restriction holds a row, and both
branches of an `if` are read. A missing value at a cell that reads the name
fails the conversion, with an error that names the alias or data term and those
cells. A missing value at a cell nothing reads is written as 0.0, and the
`decomp-electrical-unread-cells-filled` info diagnostic lists each such
parameter with its filled cells.

A constraint holds only at the cells where
`constraints/generic_constraint_bounds.parquet` has a row for it: those where
its relation and treatment govern and its restriction holds a row. In a stage
where it holds at every block, it has one row for every block, with the block
empty, when it is an inequality, or a formula whose only limit row in that
stage is the one for every block; otherwise it has one row for each cell, with
its block. An inequality carries its limit in its relation, so its rows have no
side. A formula's row has the two sides of its limit for the block, else of its
limit for every block of the stage, from
`RESTRICAO-ELETRICA-LIMITES-FORMULA-PERIODO-PATAMAR` and
`RESTRICAO-ELETRICA-LIMITES-FORMULA-DATA-PATAMAR`, as the deck gives them. A
side of magnitude 1E+30 or more, such as the deck's `1E+31` "no limit" value,
is empty, and a cell whose limit has neither side holds no row: a block whose
own limit sets no side stays unbounded rather than take the stage's limit for
every block. A restriction that holds no row, such as a formula with no limit
at any cell, is not written, and `decomp-electrical-restrictions-split` lists
it with no cells. A patamar outside the blocks of a stage it covers, in a
limit, override or reached alias row, fails the conversion.

The violation treatment of a stage is that of the
`RESTRICAO-ELETRICA-TRATAMENTO-VIOLACAO-PERIODO` row that covers it, else that
of the `RESTRICAO-ELETRICA-TRATAMENTO-VIOLACAO` row for the whole horizon, else
`soft` without a cost. A `hard` treatment allows no violation: the constraint
has no slack. A `soft` treatment is a slack charged per hour at the treatment's
cost or, without a cost or without a treatment row, at 10 times the largest
deficit cost of the `CD` registers, the penalty of the `RE` constraints. The
cost a `hard` treatment row gives is not used, and the
`decomp-electrical-hard-treatment-cost-ignored` info diagnostic lists those
rows.

A restriction also makes the deck fail to convert, with an error that names it,
when two or more overrides of the same precedence (for the cell's block, or for
every block) cover one cell of its horizon, when more than one treatment row is
for its whole horizon, or when two
`RESTRICAO-ELETRICA-TRATAMENTO-VIOLACAO-PERIODO` rows cover one stage. Once it
is written, it also fails the conversion when one of its constraints is `soft`
with a cost that is not above 0, or needs the default cost in a deck whose `CD`
registers give no deficit cost above 0, and when it is assigned an activation
rule the file does not declare.

The `decomp-libs-electrical-translated` info diagnostic counts the constraints,
expressions, parameters and bound rows written from the file, and the
constraints with an `active_when` condition. The
[DECOMP data map](decomp-data-map.md) gives the origin of each field.

## What is converted, and what is deferred

The converter emits buses with their deficit curves plus the transshipment
node, exchange lines, thermals (including the fuel-constrained anticipated
dispatch declared in `dadgnl`, with its post-horizon commitments), hydros with
the deck's registry overrides applied, small plants as must-run
non-controllable sources, the explicit scenario tree (inflow, load, and
non-controllable generation per node), pumping stations, energy contracts,
generic constraints from the deck's restriction registers, and head-dependent
production planes where the deck uses them.

A `UH` row with an initial volume is an operated plant. A row marked `NW`
(`configuracao_newave`) is a plant that enters after the study: the case holds
it as a hydro that enters after the last stage, with its minimum volume as its
initial storage. It does not operate over the study, and its inflow passes to
the plants below it. The hydro ids number the operated and the `NW` plants
together, in ascending plant code order. The conversion refuses an `NW`
reservoir whose `UH` row gives a volume above its minimum, an `NW` plant that
`hidr.dat` lacks, an `NW` plant that a `VI` record names, and an operated plant
whose `VI` travel time drains into an `NW` plant. An operated plant with no
travel time that drains into an `NW` plant converts, its water passing through
the `NW` plant; `export dessem` refuses such a case.

The committed weeks of a fuel-constrained thermal (the `GL` register) belong
to the study or to a month after it by their position in the `GS` register, not
by their dates. From the second revision of a month on, the first committed
week of the month after the study starts a few days before the study ends; it
is held from the end of the study. As in DECOMP, its days before that count
in no study stage: the last week `GS` books to the study covers them, at its own
MW per patamar. The weeks after the study declare no patamar durations, so their
patamares are weighted by their month's shares in `cortesh.dat`. When that file
is absent or does not cover the month, the last study stage's patamar hours are
used instead, with a warning; a week that commits different MW per patamar is
also reported.

The `VL`, `VU` and `VA` registers become `tailrace_flow` in
`system/hydros.json` on FPHA plants, so the tailrace level rises with the
lateral flows DECOMP adds. A `VA` gauge maps to the plant whose `AC NUMPOS`
gauge it is in the first stage. Duplicate `VU` or `VA` pairs are summed, a
`VU` naming the plant itself is added to the plant's own outflow factor, and a
`VA` whose gauge is the plant's own stays as a self entry. A duplicate `VL`, an
unmapped or ambiguous gauge, a plant whose spill does not reach the tailrace
(`influencia_vertimento_canal_fuga` 0 in `hidr.dat`; `AC VERTJU` is not read)
and a plant converted at constant productivity each drop the term with a
warning. A `VU` whose influencing plant is not in the case is dropped
without a warning. A `VL` polynomial replaces the constant tailrace only where
no `polinjus` curve applies. The warnings carry `tailrace-*` codes in the
`--json` output of `check decomp` and `convert decomp`.

A plant's diverted flow (`QDES`) reaches its diversion target, the plant named
by `AC DESVIO` or else by the `desvio` field of `hidr.dat`, through a Novomodelo
diversion channel. The channel carries the `AC DESVIO` flow limit; a plant
with no such limit, or with the deck's "no limit" value of 9 999 999 m³/s
or more, gets a channel without one, as in DECOMP, and the `QDES` bounds
still apply stage by stage. When that target is not in the case, a diversion
ceiling is closed and reported, so the water stays in the plant's own river;
a positive diversion floor cannot be closed, and the conversion fails.

A restriction register on one plant variable, such as an `HQ` on `QDEF`,
becomes a bound on that plant. Two such `HQ` restrictions on the same plant
variable that no value meets at once, such as a 0 m³/s ceiling and a 6 m³/s floor, become generic constraints
instead, each with its own slack at the same penalty. DECOMP prices every
`RHQ` violation per unit and violates the cheaper restriction, whatever their
order or numbers, so Novomodelo then makes the same choice. `check decomp` and
`convert decomp` list these pairs in one warning.

A term that is zero in DECOMP too is left out of the sum with a note, and the
rest of the restriction is kept. A plant whose `UH` record gives no initial
volume is kept by DECOMP only for the coupling with NEWAVE and is not operated,
so it generates nothing: its `FU` term in an `RE` constraint is left out, and
an `RE` left with one term becomes a plant bound. An `FI` term of an `RE` on a
pair no `IA` record joins is left out too, since no flow runs between them. Any
other term on an entity the case lacks, such as an `HQ` or `HV` term on a plant
DECOMP does not operate, an `FU` term on a plant with no `UH` record, or an
`FI` term on a submarket no `SB` record declares, leaves the whole restriction
out with a warning, and so does a restriction with no term left. The terms of
the electrical-constraint file are translated as
[Electrical constraints](#electrical-constraints) describes.

An `FT` term in an `RE` constraint on a fuel-constrained thermal of
`dadgnl.rvN` is handled as DECOMP handles it: the generation its `GL` records
command at each stage and patamar, held within the plant's `TG` minimum and
capacity there as DECOMP holds it, times the term's coefficient, is taken off
the limits, and the term leaves the sum. The case holds the plant to its
commitment for the stage, not to each patamar's `GL` value, so its dispatch
in a patamar can differ from the generation taken off the limits there;
`convert decomp` lists the stages where that generation differs by patamar in
a note, and `check decomp` lists them for an `RE` constraint. An `RE` left
with one plant term becomes a plant bound, and one left with no other term
bounds nothing in the case and is left out with a note. A plant with no `GL`
week in a stage where its term applies stops the conversion, as it stops
DECOMP, and `check decomp` reports it for an `RE` constraint. So does an `FT`
term on a GNL plant whose number a `CT` record also gives: as a malformed
field when the `CT` record is in the plant's own subsystem, which DECOMP does
not allow, and as unsupported when it is in another one, since DECOMP tells
the two plants apart by subsystem and the conversion tells thermals apart by
number alone.

DECOMP takes the ANDE load of the deck's `RI` register out of Itaipu's 50 Hz
generation before that generation reaches the submarket, so the 50 Hz units
alone serve the ANDE load. The case adds the load to the demand of Itaipu's
submarket, where the 50 Hz units sit, and the generic constraint `RI_ANDE`
holds their generation at or above it in each stage and patamar, at the
penalty of the `RE` constraints. Without it, the submarket's other plants
could serve the ANDE load wherever the `RI` 50 Hz minimum is below it.

Incremental inflows are converted as the inflow file gives them, negative
values included. DECOMP reports a negative incremental inflow as zero in its
operation output; Novomodelo's water balance removes that water unless its priced
inflow slack adds it back. The conversion lists every negative value as a
warning.

A plant's turbined flow is capped stage by stage at the limit DECOMP computes
before the optimization. The rule:

- The units' rated flow is corrected for the head at the stage's initial
  storage, relative to the machine sets' effective heads (`hidr.dat`, with
  `AC ALTEFE` overrides from their stage on). The tailrace level rises with
  the flow, so the flow is iterated until head and flow agree.
- The result is scaled by the `MP` maintenance and `FD` availability factors.
  It never exceeds the flow that produces the plant's available power.
- A plant that cannot hold back its stage inflow turbines up to that inflow
  when it exceeds the head-limited flow. This covers a run-of-river plant and
  a reservoir without enough free room. The inflow is the deck's, negative
  values included; the terminal stage takes the probability-weighted mean of
  its scenarios.
- The tailrace curves come from `polinjus`. For a deck without that file they
  come from the `hidr.dat` downstream polynomials. That path follows the same
  rule but has not been checked against a DECOMP run.
- Itaipu's limit is plant-wide: its two frequency groups share the
  head-limited flow, and each group is also held to its own available power.
- DECOMP raises the tailrace level by the lateral inflows of `VL` and `VU`
  registers; the turbined-flow limit leaves them out, although the FPHA model
  carries them. A plant with such a register, such as Itaipu or Belo Monte,
  may therefore turbine slightly more than DECOMP allows.

Novomodelo fits the planes of the hydro production function (FPHA) itself. These
inputs follow DECOMP:

- Each stage is fitted over DECOMP's volume window: the `UH` initial storage,
  as a percentage of the stage's useful volume, minus a reduction and plus an
  increase, within the stage's storage limits. Both are 20 % of useful volume
  on a weekly stage and 50 % on a monthly one unless `FP` records set them,
  in % of useful volume or in hm³, from their stage on. A run-of-river plant
  is fitted at its reference volume. The turbined-flow window and point
  counts of `FP` are not converted; the conversion warns when a record sets
  them away from DECOMP's defaults.
- The turbined-flow range of the fit runs up to the stage's turbined limit
  computed without the `MP` and `FD` factors, the range DECOMP fits over: a
  plant under maintenance keeps the planes of its full machine set, and the
  factors cap only its turbined flow. Itaipu's range is its plant-wide
  limit, derated as in the turbined cap.
- The spillage range of the fit is twice the largest monthly long-term mean
  inflow in `mlt.dat` of the plant's station (`AC NUMPOS` included). A plant
  on the artificial station 300, or on a station with a zero mean, is fitted
  over twice its maximum turbined flow, as in DECOMP. A deck without `mlt.dat`
  gets that fallback for every plant, and the conversion warns.
- Where `hidr.dat` keeps a plant's spilled flow out of its tailrace
  (`influencia_vertimento_canal_fuga` 0), DECOMP's planes carry no spillage
  term. Novomodelo's fit always raises the tailrace with the spilled flow, so
  such a plant generates less than in DECOMP whenever it spills, and
  `convert decomp` warns, naming these plants; an `AC VERTJU` override is not
  read.
- A stage where the plant's turbined limit is zero, for example with every
  machine under maintenance, gets a zero constant productivity in place of
  the production function. DECOMP builds no planes for such a stage.

The calendars Novomodelo keeps past the study horizon go to
`post_study_stages.json`. A deck with `GS` deliveries in `dadgnl` gets the
thermal calendar there (`thermal_stages` and `thermal_bounds`). A GNL plant's
capacity, minimum and cost in a post-study week come from the `TG` record of
the month `GS` books the week to, as in DECOMP, even when the week ends in the
next month or starts after the month it is booked to. DECOMP gives each `GS`
interval of a month an equal share of a 730.5-hour month, while the case holds
each post-study stage over its calendar hours; `convert decomp` warns with
`decomp-gnl-post-study-interval-hours`, listing both per stage. A deck with
`VI` travel times gets `water_bucket_stages`: one stage from the end of the
last operative stage to the end of that month, so a horizon that ends on the
first of a month gives the whole month. A `VI` travel time longer than that
stage fails the conversion, naming the plant: a water calendar of more than
one stage is not converted yet. A deck with neither writes no
`post_study_stages.json`.

Evaporation takes the monthly coefficients of `hidr.dat` with the deck's
`AC COFEVA` overrides, linearized at the reference volume the `EV` record
selects: the initial storage for `INI`, the middle of the useful volume for
`MED`. A deck without the `EV` linear model warns with
`decomp-evaporation-outside-lp`, since Novomodelo has no evaporation computed
outside the LP. Novomodelo applies to each stage the coefficient of the
calendar month the stage starts in, over that month's hours, where DECOMP
applies the coefficient of the stage's operative month over a 730.5-hour
month. The conversion leaves this convention difference in place, and
`compare decomp` reports it with the `decomp-evaporation-calendar-convention`
warning.

Some deck features have no Novomodelo counterpart yet, or are held back behind a
novomodelo issue. None is dropped silently: `check decomp` lists them per deck,
`convert decomp` reports each kind once, as a warning with a table of the
affected records, and the same diagnostics land in `--diagnostics-json` and
the `--json` verdict.
Examples at the time of writing are the `RHA` inflow constraints
(`HA`/`LA`/`CA`) and a few restriction-register term types; the
[DECOMP data map](decomp-data-map.md)
lists every register and field the converter leaves out. A warning carries a
remediation line only when a change to the deck removes it; the conversion
proceeds either way.

A deck value the conversion does not convert stops `convert decomp` with a
`source-unsupported` error that names the file and the record field and says
what to change. Its message says the value "is not converted yet" when the
conversion, not Novomodelo, is what is missing. `check decomp` fails with the
same error on the ones it can find without converting the deck.

## The boundary cost-to-go function

A DECOMP deck ends at a horizon where a longer-term cost-to-go function takes
over, delivered as cut files. By default `convert decomp` imports those cuts
into a Novomodelo policy checkpoint under the case's `boundary/` directory and
points `config.json` at it, so Novomodelo prices its terminal stage the way DECOMP
did.

- The import runs after the case is written and needs `novomodelo-python`, which
  the bridge installs as a dependency. It bootstraps the terminal cut layout
  with a short in-process novomodelo pass, so it is the slowest part of the
  conversion.
- Before importing, the bridge checks that the installed `novomodelo-python` can
  write and reload the checkpoint format by doing a real round trip, not by
  reading a version string. If the probe fails, the error says so;
  reinstalling or upgrading `novomodelo-python` resolves it.
- `--no-fcf` skips the import for a quick conversion. `--dry-run` always
  skips it.

### How the cuts attach to the converted plants

The cuts come from NEWAVE, which represents some plants differently from
DECOMP. The import couples them plant by plant, as DECOMP does:

- A plant with its own row in the cuts takes it. A plant inside a complex
  that a `CX` register of `dadger.rvN` names takes the complex's row. Any
  other plant takes the row of the first plant downstream that has one;
  DECOMP reports this rule in its output, but its manual does not describe
  it. The import lists every plant that takes another plant's row. A plant
  that finds no row is reported as an error, because the boundary gives its
  water no value.
- Only a reservoir (`tipo_regulacao` `M` or `S` in `hidr.dat`) takes a
  storage term, through its own row or its complex's, and the term prices
  the storage above the plant's minimum volume.
- The inflow-lag terms are read on each plant's coupling gauge: the one an
  `AC NPOSNW` register names, otherwise the `hidr.dat` gauge, with the
  artificial gauges computed. Novomodelo's lag state follows the inflow the plant
  operates on. Over the stages before the last one, the difference between
  the two is known at conversion and is folded into each cut's constant
  term, as are the lag terms of cut plants that no converted plant takes.
  The pre-study observations that seed Novomodelo's lag state are read on the
  coupling gauges too.
- An `NW` plant takes its own row, and its inflow series is split from the first
  operated plant downstream as NEWAVE splits it: the `NW` plant carries its
  coupling-gauge inflow and that operated plant carries the rest of its
  operation inflow. The split applies to a group only when, on every row, the
  operation inflow matches the sum of the group's coupling inflows within
  1 m³/s and each `NW` plant's own operation inflow is within 1 m³/s of zero.
  Otherwise the group keeps the deck's operation inflows and the conversion
  warns with `decomp-nw-inflow-split-fallback`, naming the plants.
- The cut values are scaled by the mean month of the DECOMP manual
  (365.25 days × 24 h / 12), whatever the length of the last stage.
- For a deck with fuel-constrained thermals, each anticipated delivery is
  priced in the month the `GS` register of `dadgnl.rvN` books its week to
  (in a deck of monthly stages only, its own month), with that month's
  patamar shares from `cortesh.dat`. A delivery the deck
  has already committed is folded into the constant term.
- Water still in transit toward a plant when the study ends (the `VI`
  register's travel time) is valued with that plant's storage term when it
  takes one, otherwise with its first inflow-lag term. This is the in-channel
  rule CEPEL documents for the cost-to-go function DECOMP hands to DESSEM.
  DECOMP gives that water no value at the NEWAVE boundary, so the two
  boundary costs differ by its value; the import warns, naming the plants
  concerned.

In the last stage, the coupling inflow's difference from the operated inflow
varies by scenario, and so does the inflow of cut plants outside the case.
The case carries each of these on an auxiliary coupling plant: a hydro with
no storage, turbining or generation and free spillage, whose last-stage
inflow is the difference (or the absent plant's coupling inflow) raised by a
constant that keeps it non-negative, a constant the cuts take back. The
auxiliary takes its plant's cut row, so the cuts price that inflow as DECOMP
does. Its id is 1000 plus the plant's deck code and its name is
`ACOPL-<code> <name>`; a deck with a plant coded 1000 or above that needs one
is refused, and converts with `--no-fcf`. `compare`, `dashboard` and
`export dessem` leave the auxiliaries out, and `convert decomp` lists them,
with each one's largest gap, in the note `boundary-fcf-coupling-auxiliary-plants`.

One part of the coupling cannot be represented yet. Novomodelo holds one GNL
commitment per delivery window rather than one per patamar, so the boundary
cost differs from DECOMP's when the commitment would differ by patamar; the
import warns and reports how far the per-patamar GNL prices spread.

The [DECOMP data map](decomp-data-map.md) traces each term to the deck
record it is read from.

### Running a case with an imported boundary

The boundary checkpoint path resolves relative to the case directory, so a
case with an imported boundary runs like any other:

```bash
novomodelo run <case_dir>
```

`convert decomp` confirms the imported boundary and this command after a
successful import.

The checkpoint records the version of the `novomodelo-python` that wrote it, and
novomodelo loads it only when that version equals its own. Run the case with the
`novomodelo` release whose version matches the installed `novomodelo-python`
(`pip show novomodelo-python`); a different novomodelo refuses the boundary and names
both versions. To use another novomodelo version, install the novomodelo-bridge
release paired with it, which pins the matching `novomodelo-python`, and convert
the deck again.

## Comparing results

`compare decomp DECOMP_DIR NOVOMODELO_OUTPUT_DIR` reads the deck plus the
`dec_oper_*.csv` operation tables directly in `DECOMP_DIR`, and Novomodelo's
`output/` directory. Percentile bands are omitted, or labelled low-N, when the
deterministic tree has too few scenarios to report a spread without
synthesizing it.

The case must have been converted from `DECOMP_DIR`: when its hydro plants or
their names differ from the deck's, the command exits 2 and says how they
differ. Plants marked `NW` are not compared, since DECOMP does not operate
them.

Each variable is judged by its gap as `compare newave` judges it, and the
console ends with each model's final lower bound and simulated mean cost. The
lower bound includes the future cost at the end of the study, which the
simulated mean cost leaves out, so the two rows are far apart on both models.

Hydro storage is compared on useful volume, as DECOMP prints it: Novomodelo's
means and P10–P90 bands leave out each plant's minimum storage. Outflow is
turbined plus spilled flow on both sides. DECOMP's defluent flow in
`dec_oper_usih` also counts the diverted flow, which leaves the river above the
plant and which Novomodelo reports apart.

DECOMP prints the same spilled flow in every patamar of a stage. Novomodelo
spills per block, and the conversion does not tie a stage's blocks together.
A spilling plant can therefore split its turbined and spilled flow between the
blocks differently from DECOMP, and the spillage, turbined flow and generation
gaps include that difference.

Each submarket's net import compares with the net interchange of
`dec_oper_sist`; the fictitious subsystem has no row. Each line compares with
the `dec_oper_interc` interchange of its `IA` pair, the pairs with the `IV`
node included. The lines on a loop whose exchange costs tie, such as a direct
line and its route through the fictitious subsystem, are left out of the
headline as on the NEWAVE report
(see [the NEWAVE guide](newave.md#comparing-results)).

Itaipu has units on its submarket's bus and on the `IV` node, and each model
can place its output on either one. The submarket's own net import and the
flow on the line between the two buses follow that placement, so the
comparison judges the two as one group, named after both buses (`SE+IV`).
The group's net import is the flow on the lines that leave it. The
submarket's own net import and the flow between the two buses show outside
the headline and the `--json` status, and the `buses-share-plant` warning
names the group.

The REE stored and inflow energies (EARM, ENA) compare with DECOMP's
`dec_oper_ree`. Novomodelo has no REE, so each plant's simulated storage and
incremental inflow are valued with DECOMP's own definition. A plant's water
counts at every plant of its energy chain (the `AC JUSENA` plant, else the next
plant down the river), each at the head of its current filling. A plant with an
`EZ` record counts in its own REE until its chain leaves it; the REE below
counts at most the `EZ` share of the plant's useful range. Inflow energy moves
to the REE of the first plant past a submarket boundary and, inside a
submarket, to a plant of another REE until the chain passes a monthly
reservoir. Both energies are in MWmês of 730.5 hours, the mean month DECOMP
uses.

Each scenario of the stochastic stage weighs by its probability on both sides:
DECOMP's by the tree's probabilities, Novomodelo's by the probability of each
simulated leaf path in `simulation/scenario_summary.parquet`. The `dashboard`
takes its means, spreads and P10–P90 bands across scenarios the same way.

Itaipu's 60 Hz units sit at the transshipment bus, which has no load, and its
50 Hz units at its submarket's bus. The per-bus charts of `compare decomp`
count the whole plant at the submarket's bus, where DECOMP reports it, and so
do the `dashboard`'s per-bus storage, stored-energy and inflow charts. The
`dashboard`'s generation by bus counts each bus's own share, and the 60 Hz
output reaches the submarket's bus as an import over the transshipment line.

The overview's cost breakdown is at present value on both sides. DECOMP's
cost report prints each stage's parcels at their nominal value and the stage
total at both nominal and present value, so each parcel is discounted by that
stage's ratio of the two. The report prints no deficit parcel: the deficit is
priced from `dec_oper_sist` and the deck's `CD` costs. What is left of the
stage total after the printed parcels and the deficit is the penalty on the
constraint slacks the report does not itemize, such as `RHQ`, `RHV`, `RHE`,
evaporation and irrigation; it is shown as one unitemized-slack category. The
two turbining parcels are the price DECOMP puts on turbined water and compare
with Novomodelo's turbined cost. DECOMP's stage total counts the take-or-pay
minimum of an export contract at its price as a cost, and only the energy
above it as revenue. Novomodelo counts every exported MWh as revenue, so on a deck
with an export minimum the two contract costs differ by twice the minimum's
value; the comparison reports that amount as the
`decomp-export-contract-cost-convention` warning.

On a deck with GNL plants (`dadgnl.rvN`), Novomodelo books a GNL plant's fuel
at the stage that commits the generation and charges nothing on delivery.
DECOMP's cost report books it on delivery, but DECOMP states its own operating
cost on the commitment basis: the stage total less the GNL delivered in the
stage (`dec_oper_gnl`) plus the GNL the stage commits for later delivery
(`relgnl.rvN`, without the weeks committed before the study). The comparison
puts DECOMP's costs on that basis. The cost breakdown splits the thermal cost
into Thermal excl. GNL and GNL committed in study, the per-stage immediate and
thermal costs move GNL to the stage that commits it, and a row under the total
shows each model's GNL delivered in the study at its price, which neither total
counts. Without either GNL file, DECOMP's thermal cost stays one row on the
delivery basis and the comparison notes the missing file. `compare newave`
compares NEWAVE's thermal cost as NEWAVE reports it.

DECOMP's `Zsup` is the lowest upper bound reached so far, and the relato
prints no upper bound per iteration, so the convergence chart sets it against
Novomodelo's running minimum.

Below the convergence chart, a table gives each model's final lower bound with
and without its expected constraint-violation penalties. One slack priced far
above any real cost, such as a pair of conflicting constraints, can dominate a
model's bound; the bounds net of penalties compare the operation both models
price. The table names the categories it counts and any cost category it
cannot classify. `compare newave` shows the same table.

On the Energy Balance tab, the ANDE load of the deck's `RI` register is added
to the demand of Itaipu's submarket, as the conversion adds it to that bus:
DECOMP's `dec_oper_sist` demand leaves it out, while its hydro generation
includes the Itaipu output that serves it. DECOMP prints the system's natural
inflow energy (ENA) as energy over the stage, in MWmês of 730.5 hours; the
chart converts it to the stage's mean MW, the unit of Novomodelo's series.

The Productivity tab compares the two models at the operating points each one
reached, per plant, stage, scenario and block. It reads DECOMP's fitted planes
(`avl_cortesfpha_dec.rvN`) and fit windows (`eco_fpha_.rvN`). For each plant
and stage it splits the gap between the two realized productivities into a
part from the fitted planes, Novomodelo's planes evaluated at DECOMP's operating
points, and a part from where Novomodelo operates. A chart draws generation against
turbined flow at DECOMP's mean operating point for both models' planes and the
exact production function, beside each model's fit inputs and head components.
The exact function follows DECOMP's conventions: the forebay at the stage's
mean storage, the tailrace at the plant's outflow plus the lateral flows of its
`VU` and `VA` registers, read between `polinjus` families at the downstream
plant's forebay, and the losses of `hidr.dat`. The `VA` share is the factor
times the mapped plant's incremental inflow in the same stage and scenario.
Novomodelo's planes are evaluated at the plant's own spill times its own factor,
plus the lateral flow its converted declaration gives, using Novomodelo's own
flows and its effective incremental inflow; a gauge link whose influencing
incremental is missing counts as zero and is logged. When the run wrote
`dec_desvfpha.rvN`, the table also gives the largest gap between that function
and DECOMP's own printed value.

A point where the plant's energy or stored water is worth nothing in either
model is a tie: DECOMP's marginal cost is zero, Novomodelo's spot price is zero at
every bus the plant feeds, or Novomodelo's water value is zero, all at DECOMP's
two-decimal print resolution. Either model may then turbine or spill at no
cost. Ties are left out of the split and of the per-plant ranking of
generation, turbined flow, productivity, spillage and outflow on the Hydro
Plant Details tab, and the tab reports their share per stage. The console
table gives the same five variables' gaps without the ties in an Excl. ties
column beside the gap, and the exported summary and the `--json` summary
carry them too. The agreement band still judges the gap over every point.
DECOMP prints no water value, so on its side a tie is detected through the
marginal cost only.
