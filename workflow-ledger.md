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
