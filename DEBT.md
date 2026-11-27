# DEBT register

Ids are zero-padded: `DEBT-001`, `DEBT-002`, and so on. Allocate the next id
by scanning this register for the highest id present and taking `max+1`. An id
is unique within this register and is never reused, because a `DEBT(` breadcrumb
in source cites it permanently.

An entry's state is the section it sits in. Paying an entry moves it verbatim
from `## Open` to `## Paid` with its `paid:` line appended. An entry is never
deleted and never renumbered.

## Entry Format

```
### DEBT-<nnn>
what:      one-line description of the shortcut/smell and where (file:symbol)
why-now:   the deliberate-prudent reason it was taken (or "inadvertent — found")
quadrant:  deliberate|inadvertent x prudent|reckless
interest:  what it slows / risks, and whether it compounds
payoff:    the concrete remediation
trigger:   the event that forces payoff
owner:     who closes this entry
blast:     files & consumers affected; tests that must pass
origin:    the plan and ticket that minted this entry
paid:      the ticket that paid it off (written only on retirement)
```

## Open

### DEBT-001
what:      the vendored novomodelo contract fixtures (tests/fixtures/novomodelo_contracts/) and schemas (tests/fixtures/novomodelo_schemas/) are cp copies of the novomodelo sources, and nothing detects when those sources change; tests/decomp/test_generic_constraints_contract.py proves the emitter against the copies only
why-now:   deliberate — a test that reads a sibling novomodelo checkout breaks the local-data policy (tests/test_local_data_policy.py: no test reaches outside the repository), so the contract tests keep to the vendored copies
quadrant:  deliberate x prudent
interest:  a novomodelo change to a vendored fixture or schema leaves the bridge proving an outdated format while its tests stay green, until someone copies the files again by hand; the generic_parameters.schema.json copy had drifted this way; it compounds with each vendored file added
payoff:    a new script, scripts/ci/check_vendored_novomodelo.py --novomodelo PATH, run by hand, that compares each file under tests/fixtures/novomodelo_contracts/ with its namesake in crates/novomodelo-io/tests/fixtures/ and each file under tests/fixtures/novomodelo_schemas/ with its namesake in schemas/ of that checkout, byte for byte, and exits non-zero on a difference or a missing source; documented in CONTRIBUTING.md
trigger:   the next novomodelo change to a contract fixture or schema the bridge vendors, or the next bridge release that pairs with a new novomodelo minor version
owner:     novomodelo-bridge maintainers
blast:     tests/fixtures/novomodelo_contracts/, tests/fixtures/novomodelo_schemas/, CONTRIBUTING.md and the new script; every test that reads those fixtures (tests/decomp/test_generic_constraints_contract.py among them) must pass
origin:    electrical-into-generic-constraints plan, epic-04 ticket-029; the drift check is a follow-up outside that plan

## Paid
