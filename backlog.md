# Token Saver 5000 — Engineering Backlog

> Updated: 2026-08-28 after commit-and-docker-smoke closeout.

## Done

### P0-A — Security & contract
- [x] H-001: `compile_knowledge` validates `output_dir` via `PathValidator`
- [x] H-002: `SCOPE_PROPERTIES` on ACE/prompt/model/experiment tool schemas
- [x] M-001: `refresh_document` re-validates stored path before read

### P0-B — Documentation single source of truth
- [x] `docs/reference/MCP_TOOL_COUNTS.md` — 128 tools, profiles, verification
- [x] Pin tool count **128**, version 0.11.0, profiles in CLAUDE.md + copilot instructions
- [x] `MCP_TOOLS_GUIDE.md` updated; token-optimization section (20 tools in `schemas_token_optimization.py`)
- [x] `.env.example` profile comment aligned with `full` / `core_stable`
- [x] Routing docstrings point at `src/handlers/mcp_core/` package (not flat `mcp_core.py`)

### P1 — File size compliance
- [x] Split `help_handlers.py` → `help_tool_registry.py` (~312 + ~1470 lines)
- [x] Split `compression_handlers.py` → common / ingest / extended + facade
- [x] Split `schemas_compression.py` → core / batch + shim
- [x] Split `semantic_compressor.py` → types / ingest mixin / retrieval mixin / core (~906 lines)
- [x] Remove generated `tests/f11_gac_fixture_harness_receipt.json` (gitignored; harness writes on `--gac`)

### P2 — Test hardening
- [x] `conftest.py` threading lock around `_reset_shared_state`
- [x] `tests/test_all_tools_have_handlers.py` — setup tools == router keys
- [x] `test_mcp_routing.py` uses router key extraction vs hardcoded count only

### P3 — Docs rebuild pack
- [x] `docs/reference/TOKEN_ECONOMY.md`
- [x] `docs/reference/PROXY.md`
- [x] `docs/guides/FILTER_RULES_DSL.md`
- [x] `docs/reference/MCP_ROUTING.md`

### Validation
- [x] CI gate: `python scripts/audit_env_example.py` (VAL-DOCKER-002)
- [x] `scripts/audit_env_example.py`, `scripts/count_mcp_tools.py`
- [x] VAL matrix executed — `artifacts/validation-run-2026-08-28.md`
- [x] Regenerate `**/claude.md` folder guides
- [x] Commit P0-B through P4 + CI green on `main`
- [x] Docker smoke: VAL-DOCKER-001/002/004 **pass**; VAL-DOCKER-003 build+ingest **pass**, size **fail** (3.7GB after CPU-torch; was 9.2GB)

## Open / follow-up

- [ ] **Docker image size** — get under 600MB (ONNX-only runtime stage; drop torch from final image)
- [ ] Growth VAL-* (`validation-contract-growth.md`) when SaaS features land in repo

## Thinktank verdict (2026-08-28)

**CHANGES_REQUIRED** on monolithic P1 — executed as ordered slices (docs → tests → handler splits → compressor split). Do not merge further file splits without per-slice test green.
