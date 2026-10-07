# Architecture

This page is the map for someone about to change the code: how a command's
data flows from source files to output, how the package tree is layered, which
direction imports may flow, and how the two conversion tracks mirror each
other. Every structural claim below is enforced by a named test, so the page
cannot drift far from the tree without a failing build. All paths are under
`src/novomodelo_bridge/`.

## A command, end to end

`novomodelo-bridge convert newave SRC DST` crosses four layers, top to bottom:

1. **`cli/`** parses the command line into a frozen args dataclass
   (`cli/args.py`) and calls one handler per command (`cli/convert.py`,
   `cli/check.py`, `cli/compare.py`, `cli/dashboard.py`). The handler owns
   what the user sees: exit codes, the `--json` envelope (`cli/verdict.py`),
   `--force` and `--validate`, and the provenance manifest
   (`cli/conversion_manifest.py`).
2. **`newave/`** is the track. `files.py` discovers the source files,
   `case.py` parses each one once and caches it, `id_map.py` fixes the
   deterministic 1-based to 0-based id mapping, `preflight.py` implements
   `check`, and `pipeline.py` calls the converters under `converters/` in
   order, reporting coarse phases to the progress bar.
3. **`core/`** holds the track-independent building blocks: the `Diagnostic`
   model and its `collect()`/`emit()` sink, the typed `BridgeError`
   hierarchy, hydro physics and productivity, calendar and stage weighting,
   the generic-constraint builder, bound-table merging, and the emission
   self-checks run on output tables before they are written.
4. **`novomodelo/`** is the Novomodelo side: `CaseWriter`, the single funnel every JSON
   and Parquet file passes through (it owns `--dry-run`); the `$schema` URL
   registry; the minimum-novomodelo-version policy; and readers for Novomodelo's
   `output/` tree.

Converters are functions from the parsed case to tables and dicts. They do
not write files, print, or format messages. A degraded input becomes a
`Diagnostic` (code, severity, title, summary, an optional per-entity table,
a remediation hint) emitted into the sink the CLI opened around the run.
Afterwards the CLI renders the collected diagnostics through `ui/console.py`
or serialises them into the `--json` verdict. Rich is imported nowhere else.

`compare` has the same shape with `comparators/` in the track's place:
readers for the source model (`comparators/newave/readers.py`,
`comparators/decomp/readers.py`) and for Novomodelo (`novomodelo/readers.py`), an
alignment step that reuses the track's id mapping, one `ComparisonDataset`
model, exporters, and the HTML report assembled from `comparators/charts/`.
`dashboard/` loads a solved case's `output/` into one `DashboardData` object
and renders one module per tab under `dashboard/tabs/`. Both reuse
`ui/theme.py` and `ui/html/` and neither touches the Rich console.

## Package map

| Package       | Role                                                                                                                                                      |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `core`        | Shared foundation for both tracks: diagnostics, errors, units, physics, calendar, generic-constraint format and builder. Imports only itself.               |
| `novomodelo`       | Novomodelo-side I/O: the case writer, the schema registry, version compatibility, output readers.                                                              |
| `newave`      | The NEWAVE track: spine modules plus `converters/`.                                                                                                       |
| `decomp`      | The DECOMP track: spine modules plus `converters/` and `fcf/` (boundary cost-to-go import).                                                               |
| `comparators` | Source-vs-Novomodelo comparison, twin-tracked under `comparators/newave/` and `comparators/decomp/`, with shared chart code under `comparators/charts/`.       |
| `ui`          | Rich terminal rendering (`ui/console.py`) and the HTML theme and document helpers (`ui/theme.py`, `ui/html/`). The only package that imports Rich.       |
| `dashboard`   | The interactive Plotly dashboard: a data layer plus one module per tab.                                                                                  |
| `cli`         | Typer wiring and the command handlers. Top of the stack; nothing imports it.                                                                             |

There are no loose modules at the package root, and no module shadows a
standard-library name (`test_no_loose_root_modules`,
`test_no_stdlib_shadowing_modules`).

## Import direction

A package may import itself and the packages below it, never sideways or up:

| Package       | May import                                                                          |
| ------------- | ----------------------------------------------------------------------------------- |
| `core`        | `core`                                                                              |
| `novomodelo`       | `core`, `novomodelo`                                                                     |
| `ui`          | `core`, `novomodelo`, `ui`                                                               |
| `newave`      | `core`, `novomodelo`, `newave`                                                           |
| `decomp`      | `core`, `novomodelo`, `decomp`                                                           |
| `comparators` | `core`, `novomodelo`, `comparators`, plus `ui.theme` and `ui.html` (not the Rich console) |
| `dashboard`   | `core`, `novomodelo`, `dashboard`, plus `ui.theme` and `ui.html`                         |
| `cli`         | everything                                                                          |

Two refinements: `comparators/newave/` may additionally import `newave`, and
`comparators/decomp/` may additionally import `decomp`; modules directly under
`comparators/` import neither track. `ui/theme.py` is a leaf that depends only
on `core`, so presentation-consuming packages share colours and layout
without pulling in Rich.

The table is a readability aid. The enforced rule is the `_ALLOWED` map in
`tests/test_package_boundaries.py`: `test_import_direction` fails on any edge
the map does not permit, `test_no_private_cross_package_imports` forbids
importing an underscore-private name across any package boundary, and
`test_nothing_imports_cli` holds with no allowlist at all. Imports made only
under `TYPE_CHECKING` are exempt from the direction rule but pinned as an
exact census in the same file, so adding one is a visible diff.

## Twin-track spine

`newave/` and `decomp/` are two conversions of the same shape. Each exposes
the same five spine modules at its top (`case`, `files`, `id_map`,
`pipeline`, `preflight`) and each carries a `converters/` subpackage;
`decomp/` also has `fcf/`. `test_track_packages_are_structurally_symmetric`
enforces this. Other loose modules at a track's top are that track's shared
infrastructure; converter-role code belongs under `converters/`.

The rule the spine serves: a behaviour added to one track (a CLI flag, an
exit-code rule, an emission check, a preflight policy) lands on both tracks in
the same change, and physics or calendar math both tracks need has one
implementation in `core/`. Two near-identical implementations that differ at
the edges is the bug class `compare` exists to catch, in our own tool.
Track-private helpers are never imported across the boundary; when both
tracks need one, it moves to `core/`.

## Extension points

- **A new converter** is a module under a track's `converters/` that takes
  the parsed case and the id map and returns tables or dicts. Wire it into
  that track's `pipeline.py`, write its output through the `CaseWriter`, and
  for a new output file register the `$schema` URL in `novomodelo/schemas.py`.
  Emit a `Diagnostic` for anything degraded. Describe the output's fields and
  their deck sources in the track's TOML under `docs/lineage/` and regenerate
  the data map; the lineage tests fail on an emitted field with no entry.
- **A shared rule** (physics, calendar, a constraint expression) goes in
  `core/`, with both tracks calling it.
- **A new CLI flag** is declared in `cli/app.py` on both tracks' commands,
  threaded through `cli/args.py`, and handled in the handler module.
  Regenerate `docs/cli.md` afterwards.
- **A novomodelo limitation** the bridge has to work around is marked at the site
  with a `TRACKED NOVOMODELO-GAP` comment, surfaced as a log line or diagnostic,
  and recorded with its removal condition in the novomodelo repository. It is never
  adopted silently, and the comment is never removed in a cleanup pass.
- **User-facing text** is self-contained: no repository paths, internal
  symbols, or build tooling in a CLI message, because most users have no
  checkout.

Testing conventions, quality gates, and the contribution workflow are in
`CONTRIBUTING.md`.
