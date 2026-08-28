# MAP — Phase 0 Security & Contract Hardening (Slice P0-A)

## Destination

Close audit H-001, H-002, M-001: validated write paths, tenant scope on all ACE/prompt/model/experiment tool schemas, and path re-validation on file refresh — without file-size splits or doc overhaul.

## Open questions

(none — audit provided evidence)

## Not yet specified

- Doc single-source-of-truth (deferred to P0-B)
- Oversized module splits (deferred to P1)

## Out of scope

- Splitting `semantic_compressor.py` / `compression_handlers.py`
- Full MCP_TOOLS_GUIDE rewrite
- conftest parallel-test locking (separate slice)

## Answers

### compile_knowledge output_dir must not write outside whitelist

**Answer:** Validate `output_dir` with `PathValidator` when `write_files` is true.

**Why:** CWE-22 write path; audit H-001.

**Check:** `pytest tests/test_knowledge_handlers.py -k compile_knowledge_rejects_traversal_output_dir` → FAIL before fix, PASS after; traversal path raises ValueError.
**Judged by:** run it
**Reference:** —

### ACE/prompt/model/experiment schemas advertise tenant scope

**Answer:** Every `Tool` in `schemas_prompts_ace.py` and `schemas_model_experiment.py` includes `SCOPE_PROPERTIES`.

**Why:** Audit H-002; 29 tools missing tenant fields.

**Check:** `python -c "import re; ..."` grep each schema file — zero `inputSchema` property blocks without workspace_id key.
**Judged by:** run it
**Reference:** —

### refresh_document re-validates stored path before read

**Answer:** `handle_refresh_document` runs `path_validator.validate(metadata.file_path)` before `open()`.

**Why:** TOCTOU after ingest; audit M-001.

**Check:** `pytest tests/test_file_sync_handlers.py -k refresh_rejects_traversal` → PASS after fix.
**Judged by:** run it
**Reference:** —
