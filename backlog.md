# Token Saver 5000 — Engineering Backlog

> Updated: 2026-08-28 after Slice P0-A (security & contract hardening).

## Done (P0-A)

- [x] H-001: `compile_knowledge` validates `output_dir` via `PathValidator`
- [x] H-002: `SCOPE_PROPERTIES` on ACE/prompt/model/experiment tool schemas (29 tools)
- [x] M-001: `refresh_document` re-validates stored path before read

## P0-B — Documentation single source of truth (next slice)

- [ ] Replace stale `mcp_core.py` references with `src/handlers/mcp_core/` package map
- [ ] Pin tool count (138), version (0.11.0), profiles (`full` / `core_stable`) in README + CLAUDE.md
- [ ] Expand `MCP_TOOLS_GUIDE.md` for token-optimization tools (21 tools)
- [ ] Fix `.env.example` profile comment (`core_stable` not `core`)

## P1 — File size compliance

- [ ] Split `src/semantic_compressor.py` (2870 lines)
- [ ] Split `src/handlers/compression_handlers.py` (2935 lines)
- [ ] Split `src/handlers/help_handlers.py` (1748 lines)
- [ ] Split `schemas_compression.py` (730 lines)
- [ ] Compress or split `tests/f11_gac_fixture_harness_receipt.json` (7990 lines)

## P2 — Test hardening

- [ ] conftest `_reset_shared_state` threading lock for parallel pytest
- [ ] `test_all_tools_have_handlers` for 138 tools
- [ ] Replace hardcoded tool count in `test_mcp_routing.py`

## P3 — Docs rebuild pack

- [ ] `docs/reference/TOKEN_ECONOMY.md`
- [ ] `docs/reference/PROXY.md`
- [ ] `docs/guides/FILTER_RULES_DSL.md`
- [ ] `docs/reference/MCP_ROUTING.md` (post-split)

## Validation contracts (execution verify)

- [ ] Run VAL-* matrix from `artifacts/validation-contract-areas-1-4.md`
- [ ] CI gate for `scripts/audit_env_example.py` (VAL-DOCKER-002)
