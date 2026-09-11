# Phase 0 Security & Contract Hardening (Slice P0-A) Implementation Plan

> **For agentic workers:** TDD per task. Steps use checkbox syntax.

**Goal:** Close audit findings H-001, H-002, M-001 — security write path, tenant schema scope, file refresh TOCTOU.

**Architecture:** Use existing `PathValidator` from `HandlerContext` (or cwd+home default in tests). Inject `SCOPE_PROPERTIES` into ACE/prompt/model/experiment schemas matching compression schemas.

**Tech Stack:** Python 3.10+, pytest, MCP Tool schemas

---

### Task 1: compile_knowledge output_dir validation

**Files:**
- Modify: `src/handlers/memory_handlers.py`
- Test: `tests/test_knowledge_handlers.py`

- [ ] **Step 1: Write failing test** — reject `output_dir` with `../` traversal when `write_files=True`
- [ ] **Step 2:** `pytest tests/test_knowledge_handlers.py -k traversal_output_dir -v` → FAIL
- [ ] **Step 3:** Validate `output_dir` via `path_validator.validate()` before `KnowledgeCompiler`
- [ ] **Step 4:** pytest → PASS

### Task 2: SCOPE_PROPERTIES on prompts/ACE + model/experiment schemas

**Files:**
- Modify: `src/handlers/mcp_core/schemas_prompts_ace.py`
- Modify: `src/handlers/mcp_core/schemas_model_experiment.py`
- Test: `tests/test_mcp_scope_properties.py` (new)

- [ ] **Step 1:** Failing test — all tools in both modules expose workspace_id in inputSchema.properties
- [ ] **Step 2:** Add import + `**SCOPE_PROPERTIES` to each tool properties dict
- [ ] **Step 3:** pytest → PASS

### Task 3: refresh_document path re-validation

**Files:**
- Modify: `src/handlers/file_sync_handlers.py`
- Test: `tests/test_file_sync_handlers.py`

- [ ] **Step 1:** Failing test — refresh rejects path outside whitelist (mock metadata with bad path)
- [ ] **Step 2:** Validate `metadata.file_path` before `open()`
- [ ] **Step 3:** pytest → PASS

### Task 4: Gates

- [ ] `pytest tests/test_knowledge_handlers.py tests/test_file_sync_handlers.py tests/test_mcp_scope_properties.py -q --no-cov`
- [ ] `ruff check` + `black` on touched files
