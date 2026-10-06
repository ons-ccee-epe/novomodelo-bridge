# Provenance of the vendored Novomodelo policy readers

These 12 modules are **generated code, vendored verbatim**. They are never edited: not to satisfy
`ruff`, not to satisfy `mypy`, not to add annotations. `pyproject.toml` excludes this directory from
both tools for exactly that reason. To change anything here, regenerate from the schema and update
the digests below.

## Upstream

| Property | Value |
|----------|-------|
| Schema | `crates/novomodelo-io/schemas/policy.fbs` |
| Namespace | `Novomodelo.IO.Policy` |
| FlatBuffers file identifier | `CBVF` |
| Checkpoint `format_version` | 1 |
| Source case | `/home/carlosribeiro/git/DEC_ONS_052026_RV0_VE_CONVERTIDO` |
| Source subdirectory | `Novomodelo/` |
| Copied | 2026-09-04 |

The import path is `Novomodelo.IO.Policy.<Module>`. It is deliberately **not** nested under
`conversor_fcf`: both the generated modules and the reference case rely on that exact path.

## Modules

Paths are relative to this directory. Every digest was verified identical to its counterpart in the
source case at the time of copying, and re-verified after the tree was relocated from the repository
root to `src/`.

| Module | sha256 |
|--------|--------|
| `IO/Policy/AffinePiece.py` | `f7a1362a627befd78a2e1ab0e99672bd705ae3d89a4d5c0dcbc11b80ac57c400` |
| `IO/Policy/CheckpointManifest.py` | `cf0c319bda97294fbf03d85d13fe7eff11244c1a01fa730d4c14959a7d20f808` |
| `IO/Policy/EntitySlot.py` | `6c28a7bc4e9f9c8991052b6884b21a4ac8845263bbfc4621e77cae8fd53f51d9` |
| `IO/Policy/EntityType.py` | `d8a03b3b6dcf2aa1c5a45b24f3048694053e600ecc57a9f515f2d5df388fe8e5` |
| `IO/Policy/ManifestEdge.py` | `fbecb26cfb85d9732a269a14d539c4ca6d7f778e956dfc90ba041b602c863c9e` |
| `IO/Policy/ManifestNode.py` | `3674f3117478485d80aaaee07c0f10e4bb54ca23205cb5858d378c198a1eba0f` |
| `IO/Policy/StageBasis.py` | `99cd8b449926b34a6dfce1639f19a6697525fb3227a0db8af80162d6d4628ca4` |
| `IO/Policy/StageCuts.py` | `bb413d8f68737e9ef9899e6bf49fdc982b66b10978100055072bcad79a86408f` |
| `IO/Policy/StageStates.py` | `95aed5a7c5dc7c30dbc8991007be6429b5a6af53be503bcd2188ef0996562c62` |
| `IO/Policy/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `IO/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

The three `__init__.py` files are empty, which is why they share the sha256 of the empty input.

## Consumed by

`src/conversor_fcf/novomodelo/policy_reader.py` materializes `CheckpointManifest` and `StageCuts` into
frozen dataclasses at the reader boundary; no live FlatBuffers table travels deeper into the
package. `StageBasis` and `StageStates` are vendored for completeness and are not consumed by the
conversion.
