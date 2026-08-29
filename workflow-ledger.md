# Workflow Ledger

## 2026-08-28 — Slice P0-A: Security & Contract Hardening

**Plan:** `docs/plans/2026-08-28-phase0-security-contract-slice.md`  
**Wayfinder:** `.wayfinder/phase0-security-contract/MAP.md`

### Changes
- `memory_handlers.py`: PathValidator on `compile_knowledge` output_dir when write_files
- `file_sync_handlers.py`: PathValidator before refresh read
- `schemas_prompts_ace.py`, `schemas_model_experiment.py`: SCOPE_PROPERTIES on all tools
- Tests: `test_mcp_scope_properties.py`, traversal tests in knowledge + file sync

### Verification
```bash
pytest tests/test_mcp_scope_properties.py tests/test_knowledge_handlers.py \
  tests/test_file_sync_handlers.py::TestHandleRefreshDocument -q --no-cov
```

### Deferred
Doc truth (P0-B), file splits (P1), conftest parallel lock (P2)

---

## 2026-08-28 — Slices P0-B through P3 (thinktank-ordered)

**Thinktank verdict:** CHANGES_REQUIRED on monolithic P1 — executed as ordered slices (docs → tests → handler splits → compressor split).

### Changes
- P0-B: `MCP_TOOL_COUNTS.md`, 128-tool doc alignment, `.env.example` profile comment
- P1: splits for `help_handlers`, `compression_handlers`, `schemas_compression`, `semantic_compressor`
- P2: `test_all_tools_have_handlers`, conftest lock, routing test updates
- P3: `TOKEN_ECONOMY.md`, `PROXY.md`, `FILTER_RULES_DSL.md`, `MCP_ROUTING.md`
- CI: `scripts/audit_env_example.py` in quality-gate
- Removed generated `f11_gac_fixture_harness_receipt.json` (gitignored)

### Verification
```bash
pytest tests/test_all_tools_have_handlers.py tests/test_mcp_routing.py \
  tests/test_compression_handlers.py tests/test_f11_gated_fusion.py -q --no-cov
python scripts/audit_env_example.py
```

---

## 2026-08-28 — Slice P4: Validation + Guides Closeout

**Wayfinder:** `.wayfinder/validation-guides-closeout/MAP.md`  
**Plan:** `docs/plans/2026-08-28-validation-guides-closeout.md`

### Changes
- Fixed `_SMALL_INPUT_TOKEN_THRESHOLD` import in `compression_handlers_ingest.py`
- Regenerated `claude.md` folder guides (5 files delta vs HEAD)
- `artifacts/validation-run-2026-08-28.md` — VAL areas 1–4 matrix

### Verification
```bash
pytest tests/test_async_operations.py tests/test_compression_handlers.py \
  tests/test_f11_gated_fusion.py tests/test_all_tools_have_handlers.py -q --no-cov
python scripts/audit_env_example.py
```

---

## 2026-08-28 — Slice BL-NPR: NumPy PageRank fallback

**Wayfinder:** `.wayfinder/numpy-pagerank-fallback/MAP.md`  
**Plan:** `docs/plans/2026-08-28-numpy-pagerank-fallback-slice.md`  
**Commit:** `b70f2bb`

### Changes
- Route `code_compressor`, `adaptive_rate_allocator`, `multimodal_compressor` through `compute_pagerank`
- `tests/test_pagerank_numpy_fallback.py`

### Verification
```bash
pytest tests/test_pagerank_numpy_fallback.py tests/test_code_compressor.py \
  tests/test_adaptive_rate_allocator_coverage.py tests/test_multimodal_compressor_coverage.py -q --no-cov
```

---

## 2026-08-29 — Session closeout

**HEAD:** `b031441` (BL-NPR + ledger) · **CI:** run `33230864165` **FAIL** — `tests/claude.md` drift after new test file.

### Closeout actions
- Regenerated `tests/claude.md` (folder guide sync)
- Committed CEO update, validation receipt honesty, growth VAL defer brief
- Deleted ephemeral `benchmarks/_saas_tmp/`, `benchmarks/_saas_results/`
- Secret sweep: CLEAN on tracked `src/` `tests/` `scripts/`
- Worktrees: none · Open PRs: none

### Verification (closeout)
```bash
python scripts/check_claude_folder_guides_sync.py
pytest tests/test_pagerank_numpy_fallback.py -q --no-cov
rg 'nx\.pagerank' src/code_compressor.py src/adaptive_rate_allocator.py src/multimodal_compressor.py
# expect no matches
```

### Open for next session
- CEO-gate Growth VAL-* (see `docs/strategy/2026-08-28-growth-val-defer-brief.md`)
- CI must be green on HEAD after closeout push
