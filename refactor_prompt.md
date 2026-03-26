# Refactoring Prompt v3

Battle-tested against the rippled-workload codebase. Incorporates lessons from the app.py split (2026-03-25) and workload_core.py extraction (2026-03-26).

---

## Objective

Improve a Python codebase's **maintainability, readability, and modularity** through incremental, measurable refactoring — one target at a time.

Bias toward **clarity, low cognitive load, and predictable structure** over cleverness.

---

## Decision Hierarchy (Strict Priority)

1. **Preserve behavior** — zero functional changes
2. **Preserve or improve readability** — code should be easier to scan after, never harder
3. **Structural moves only** on first pass — extraction, not redesign
4. **Deduplication is a separate pass** — move verbatim first, consolidate later
5. **No speculative abstraction** — no new base classes, generic helpers, or "this could be useful later"
6. **No unrelated churn** — don't touch formatting, naming, type hints, or comments in moved code

---

## Process

### 1. Identify the target

Use complexity metrics to find the worst offender:

```bash
cd workload

# Maintainability Index — find grade C files (worst)
uv run --with radon radon mi src/workload/ -s | sort -t'(' -k2 -n

# Cyclomatic Complexity — find D/E/F functions
uv run --with radon radon cc src/workload/<target>.py -s -a -n C

# Cognitive Complexity — find failures (score > 15)
uv run --with complexipy complexipy src/workload/<target>.py -f -s desc
```

Pick the file with the worst MI score. Within that file, identify clusters of related functions that can move together.

### 2. Classify what you're extracting

Before touching code, categorize every function/block in the target file:

| Category | Action | Example |
|----------|--------|---------|
| **Dead code** | Delete outright. No stubs, no comments. Track in TODO if reimplementation needed. | `init_participants` (broken, replaced by generate_ledger) |
| **Leaf nodes** | Extract first — schemas, models, helpers with no business logic | Pydantic models → `schemas.py` |
| **Naturally clustered groups** | Extract as a module — functions that share a responsibility and are called from one dispatch point | Validation hooks → `validation_hooks.py` |
| **Tightly coupled core** | Extract last — the most connected code, after everything else is pulled out | Lifespan/startup → `bootstrap.py` |
| **Leave alone** | Functions that are fine where they are | Simple one-liners, well-scoped methods |

### 3. Plan the extraction order

**Leaves first, roots last.** The safest sequence:

1. Delete dead code (instant win, no risk)
2. Extract leaf nodes (schemas, models, constants, standalone helpers)
3. Extract naturally clustered groups (called from one place, can move as a unit)
4. Extract the remaining coupled code (startup, state management)
5. Slim the original file to an entrypoint/coordinator

Each step should be independently verifiable — the app runs after each extraction.

### 4. Execute mechanically

For each extraction:

- **Copy verbatim** — same code, different file
- **Fix imports** — the new file imports what it needs, the old file imports from the new one
- **Verify** — import check, endpoint check, full run
- **Stage** — one logical extraction per commit

**Do NOT:**
- Rename functions during extraction
- Add type hints to moved code
- Deduplicate patterns while moving
- Redesign interfaces
- Create abstraction layers, base classes, or registries

### 5. Measure the result

Run the same metrics before and after. Update `refactoring_log.md`:

```bash
uv run --with radon radon mi src/workload/<files> -s
uv run --with radon radon cc src/workload/<files> -s -a -n C
uv run --with complexipy complexipy src/workload/<files> -f -s desc
uv run --with radon radon hal src/workload/<files>
```

Key metrics to track:
- **MI grade change** (C → B → A is the goal)
- **CC worst function** (should decrease or stay same, never increase)
- **Complexipy failures** (count should decrease)
- **Halstead effort** (should distribute across files, not concentrate)
- **LOC** (modest increase from import overhead is fine)

---

## What NOT to do (learned the hard way)

### From refactor_suggestion.md — rejected ideas

| Suggestion | Why rejected |
|---|---|
| `utils/` directory | God module. Group by responsibility, not "utility" |
| `lifespan/startup.py` + `lifespan/shutdown.py` | Over-split. Startup and shutdown are tightly coupled — one `bootstrap.py` suffices |
| `config/env.py` + configuration manager class | Abstraction for 3 env vars. The existing `config.toml` + `os.environ` is fine |
| `from __future__ import annotations` | Unnecessary on Python 3.13+ |
| Centralized error handler module | Error handling is domain-specific (tem/tef/ter codes). It belongs where the domain logic is |
| `routes/` directory name | FastAPI convention is `routers/` |

### From refactor_prompt_v2.md — what survived and what didn't

**Kept:**
- Decision hierarchy (preserve behavior > readability > modularity)
- Anti-patterns list (speculative generality, shotgun abstraction, indirection without benefit, over-modularization)
- Separate structural moves from deduplication
- "Call out uncertainty explicitly"

**Dropped:**
- 3-agent topology (Agent 1 modularization / Agent 2 deduplication / Agent 3 adjudicator) — overkill for mechanical extraction where the target shape is already known. The multi-agent approach makes sense when the architecture is genuinely ambiguous. For known-shape extractions, sequential is safer.
- Branch-based parallel refactoring — editing the same file from two branches creates merge hell. Do it sequentially.

### From rippled_workload_refactor_plan.md — what worked

**Worked great:**
- Split by responsibility, not by line count
- "Boundary clarity first, cleverness later"
- Leaf-to-root extraction order
- The 5-step sequence (schemas → easy routers → state split → transactions → startup)

**Adjusted:**
- Dropped `services/` directory — network reset stayed inline in its router, startup stayed in `bootstrap.py`. Don't create directories for one file.
- Dashboard HTML moved as-is (giant inline string in `state_pages.py`) — restructuring it is a separate job
- Acknowledged existing routers in app.py — the extraction was "move handlers to files" not "design new routers"

---

## Scope discipline

Each refactoring session should have a clear, bounded scope:

- **"Split file X by responsibility"** — mechanical moves, fix imports, verify
- **"Extract function cluster Y to its own module"** — move functions, update dispatch, verify
- **"Remove dead code Z"** — delete, update call sites, add TODO if needed, verify

Do NOT combine these. Do NOT also deduplicate, modernize, or redesign during an extraction pass. That's the "now that it's clean, let's get fancy" phase — and it's a separate commit.

---

## Verification checklist

After each extraction:

```bash
# 1. Import check (catches broken references)
uv run python -c "from workload.app import app; print(f'routes: {len(app.routes)}')"

# 2. Lint (catches unused imports, syntax issues)
uv run --group lint ruff check src/workload/<modified_files>

# 3. Metrics (catches regressions)
uv run --with radon radon mi src/workload/<files> -s

# 4. Live test (catches runtime failures)
docker compose up -d --build workload
sleep 30
curl -s http://localhost:8000/health
curl -s http://localhost:8000/version
curl -s http://localhost:8000/state/summary
```

---

## Philosophy

Do not optimize for elegance in isolation.

Optimize for:
- **Fast human comprehension** — can a new developer find what they need?
- **Low cognitive load** — does each file have one clear purpose?
- **Predictable structure** — do similar things live in similar places?
- **Explicit behavior** — can you trace a request from endpoint to effect?

If a change makes the code "cleaner" but harder to follow, it is wrong.

Three similar lines of code is better than a premature abstraction. A 200-line file with one responsibility is better than a 50-line file that imports from 6 places. Duplicate code that's obvious beats deduplicated code that's clever.

Measure twice, cut once. Run the metrics. Update the log.
