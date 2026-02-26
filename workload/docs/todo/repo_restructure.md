# Repo Restructure — Clean Up Root, Clarify Workload vs Deployment Boundary

## Context

The workload is designed as a generic XRPL traffic generator that happens to live inside an Antithesis testing repo. Long-term it should be its own repo. For now, clean up the root so the boundary is clear and a future split is easy.

The root currently has: legacy dirs (`prepare-workload/`, `.devcontainer/`, `simple/`), loose scratch files, duplicated docs, and Antithesis-specific deployment configs mixed with the workload.

## Guiding Principle

**Two concerns in one repo:**
1. `workload/` — the generic XRPL traffic generator (standalone, no Antithesis dependencies)
2. Everything else — Antithesis deployment harness (Dockerfiles, sidecar, test_composer, network configs)

## Changes

### 1. Delete legacy/dead directories and files

| Delete | Reason |
|--------|--------|
| `prepare-workload/` | Superseded by `generate_ledger` package |
| `.devcontainer/` | Devcontainer removed, running locally now |
| `simple/` | Empty directory |
| `SEQUENCE_BUG_ANALYSIS.md` | Historical debug session notes |
| `SESSION_STATE.md` | Stale session notes from branch `huge-refactor` |
| `get_fee.py` | Scratch script, functionality in workload |
| `test_encode_nftoken_id.py` | One-off test script |
| `test_start_experiment_flow.md` | Outdated experiment flow (superseded by README quick start) |
| `Dockerfile.config` | Built from `prepare-workload/` which is gone |

### 2. Consolidate root `docs/` into `workload/docs/`

The root `docs/` contains rippled source analysis files (TxQ.cpp, Transactor.cpp, etc.) and workload-specific docs (FeeEscalation.md, error_codes.md). Move the workload-relevant ones into `workload/docs/reference/` and delete the rest (rippled source dumps don't belong in this repo).

### 3. Clean root to deployment-only concerns

After cleanup, the root should contain only:

```
rippled-workload/
├── README.md              # Quick start, points to workload/README.md
├── CLAUDE.md              # AI assistant instructions
├── .gitignore
├── Dockerfile             # Antithesis workload image
├── Dockerfile.rippled     # Antithesis rippled image
├── docker-compose.yml     # Testnet + workload compose
├── workload/              # The generic traffic generator (its own project)
├── sidecar/               # Antithesis monitoring sidecar
├── test_composer/          # Antithesis test scenarios
├── scripts/               # Standalone diagnostic tools
└── specs/                 # Feature specs
```

### 4. Move workload-level loose docs into `workload/docs/`

These files in `workload/` should move into `workload/docs/`:

| File | Move to |
|------|---------|
| `Architecture.md` | `workload/docs/architecture.md` |
| `MPToken.md` | `workload/docs/reference/mptoken.md` |
| `XRPL_RELIABLE_SUBMISSION.md` | `workload/docs/reference/reliable-submission.md` |
| `ws-architecture.md` | `workload/docs/ws-architecture.md` |
| `ws-architecture.excalidraw` | `workload/docs/ws-architecture.excalidraw` |

`FAQ.md` and `README.md` stay in `workload/` (they're entry points).

## Future: Repo Split

When ready, the workload becomes its own repo and this repo imports it as a dependency (editable install, like `generate_ledger` is today). The Antithesis-specific Dockerfiles, sidecar, and test_composer stay here.
