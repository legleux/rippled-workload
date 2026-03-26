# Refactoring Log

Tracks complexity metrics across refactoring efforts. Updated each time a refactoring branch lands.

## How to regenerate metrics

```bash
cd workload

# Set FILES for the code being measured
FILES="src/workload/app.py"  # adjust per branch

uv run --with radon radon cc $FILES -s -a     # Cyclomatic Complexity
uv run --with radon radon mi $FILES -s         # Maintainability Index
uv run --with radon radon raw $FILES -s        # Raw metrics (LOC, SLOC, etc.)
uv run --with radon radon hal $FILES           # Halstead metrics
uv run --with complexipy complexipy $FILES     # Cognitive Complexity
```

---

## Entry 2: Extract validation hooks + remove init_participants (2026-03-26)

**What changed:** Two extractions from `workload_core.py` (3,142 → 1,664 lines):

1. **Validation hooks** — 27 `_on_*` methods extracted to `validation_hooks.py` as standalone functions. Single dispatch point via `dispatch_validation_hooks()`. No behavior change.
2. **`init_participants` removed** — 8-phase organic init (CC 94, cognitive 145) plus 4 helper methods (~930 lines). Broken since `generate_ledger` handles pre-genesis provisioning. Design doc at `workload/docs/todo/reimplement_init_participants.md`. Bootstrap now errors if no genesis state found.

### Key Metrics — `workload_core.py`

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| **LOC** | 3,142 | 1,664 | **-1,478 (-47%)** |
| **MI** | C (0.00) | C (0.00) | — (still large) |
| **CC worst** | F (94) `init_participants` | D (28) `submit_pending` | **F → D** |
| **CC functions ≥ C** | 12 | 5 | **-7** |
| **Complexipy failures** | 11 | 5 (in workload_core) + 2 (in validation_hooks) = 7 | **-4** |
| **Halstead effort** | — | 68,967 | — |
| **Halstead est. bugs** | — | 1.77 | — |

### New file: `validation_hooks.py`

| Metric | Value |
|--------|-------|
| MI | **B (14.63)** |
| LOC | 531 |
| CC worst | C (15) `on_payment_validated` |
| Complexipy failures | 2 (`on_payment_validated` 23, `on_nft_offer_accepted` 16) |

### Remaining complexipy failures (`workload_core.py`)

| Function | Score | Notes |
|----------|-------|-------|
| `submit_pending` | 51 | Hot path — next refactor target |
| `_cascade_expire_account` | 33 | Sequence recovery |
| `poll_dex_metrics` | 21 | AMM pool polling |
| `build_sign_and_track` | 19 | Hot path |
| `InMemoryStore.mark` | 16 | State machine |

### Analysis

The MI stays at C (0.00) because `workload_core.py` is still 1,664 lines with high Halstead metrics. But function-level complexity improved dramatically — the F (94) outlier is gone, and complexipy failures dropped from 11 to 7 across both files. The next wins come from decomposing the submission pipeline (`submit_pending` + `build_sign_and_track`).

---

## Entry 1: `app_refactor` — Split app.py into routers (2026-03-25)

**What changed:** Split monolithic `app.py` (2,220 lines) into 14 files: `app.py` (entrypoint), `bootstrap.py` (startup), `schemas.py`, `dependencies.py`, and 10 router modules under `routers/`. Mechanical moves only — no logic changes.

### Key Metrics Comparison

| Metric | Baseline (main) | After (app_refactor) | Delta |
|--------|-----------------|----------------------|-------|
| **Maintainability Index** | **C (8.11)** | **All A (38–100)** | C → A |
| CC average | A (2.28) | A (2.27) | — (same functions) |
| CC worst function | C (16) `state_type_page` | C (16) `state_type_page` | — (unchanged) |
| Halstead effort | 23,701 | 2,182 max per file | **-91% peak** |
| Halstead est. bugs | 0.70 | 0.13 max per file | **-81% peak** |
| Halstead vocabulary | 159 | 45 max per file | **-72% peak** |
| Complexipy failures | 3 | 3 | — (same functions) |
| LOC (total) | 2,220 | 2,308 | +88 (import overhead) |
| SLOC | 1,785 | 1,842 | +57 |
| Files | 1 | 14 | +13 |

### Maintainability Index Detail

| | main | app_refactor |
|---|---|---|
| `app.py` | C (8.11) | A (100.00) |
| `bootstrap.py` | — | A (46.90) |
| `schemas.py` | — | A (61.07) |
| `routers/state_pages.py` | — | A (38.74) |
| `routers/workload.py` | — | A (50.08) |
| `routers/state_api.py` | — | A (50.34) |
| All other new files | — | A (53–100) |

### Halstead Detail

| | main (single file) | app_refactor (worst file) |
|---|---|---|
| Vocabulary | 159 | 45 (`state_pages.py`) |
| Volume | 2,106 | 401 (`state_pages.py`) |
| Difficulty | 11.3 | 7.0 (`workload.py`) |
| Effort | 23,701 | 2,182 (`bootstrap.py`) |
| Est. bugs | 0.70 | 0.13 (`state_pages.py`) |

### Complexipy Failures (unchanged)

| Function | Score | Location (main) | Location (refactored) |
|----------|-------|-----------------|----------------------|
| `wait_for_ledgers` | 25 | `app.py` | `bootstrap.py` |
| `lifespan` | 22 | `app.py` | `bootstrap.py` |
| `state_type_page` | 16 | `app.py` | `routers/state_pages.py` |

### Analysis

The refactor didn't change any function-level complexity (CC and complexipy are identical) because it was a mechanical extraction — same code, different files. The wins are entirely at the **file level**:

- **MI jumped from C to A** — the single biggest signal. A 2,220-line file with mixed responsibilities scores poorly on maintainability regardless of individual function simplicity. Splitting by responsibility lets each file score on its own merits.
- **Halstead effort dropped 91%** — the monolith's 159-symbol vocabulary and 23,701 effort reflected a developer needing to hold the entire file in their head. Now the most complex file (`bootstrap.py`) has effort 2,182.
- **Est. bugs dropped from 0.70 to 0.13** — Halstead bug estimation is proportional to volume. Smaller files = lower predicted bug surface per file.

The 3 complexipy failures (`wait_for_ledgers`, `lifespan`, `state_type_page`) are candidates for future refactoring but were explicitly out of scope for this pass.

---

## Baseline: `main` — Monolithic app.py (2026-03-25)

**Snapshot of `app.py` before any refactoring.**

| Metric | Value |
|--------|-------|
| Maintainability Index | **C (8.11)** |
| CC functions | 81 functions + 8 classes |
| CC average | A (2.28) |
| CC worst | C (16) `state_type_page` |
| Halstead vocabulary | 159 |
| Halstead volume | 2,106 |
| Halstead difficulty | 11.3 |
| Halstead effort | 23,701 |
| Halstead est. bugs | 0.70 |
| Complexipy pass/fail | 78 pass / 3 fail |
| LOC | 2,220 |
| SLOC | 1,785 |
| LLOC | 831 |
| Comments | 25 |
