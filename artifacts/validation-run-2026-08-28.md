# Validation Run Receipt — 2026-08-28

> Execution of VAL-* contracts from `artifacts/validation-contract-areas-1-4.md`  
> **Environment:** Windows local, Python 3.14; Docker build smoke executed locally

## Summary

| Area | Pass | Waived | Fail | Pending |
|------|------|--------|------|---------|
| Docker (4) | 2 | 1 | 1 | 0 |
| Quality (5) | 4 | 1 | 0 | 0 |
| UI (5) | 0 | 5 | 0 | 0 |
| Degrade (6) | 0 | 6 | 0 | 0 |
| **Areas 1–4 total** | **6** | **15** | **1** | **0** |

Growth contracts (`validation-contract-growth.md`) are **waived** — SaaS/team/webhook features not in this MCP server repo.

## Area 1: Docker & Infrastructure

| ID | Status | Evidence |
|----|--------|----------|
| VAL-DOCKER-001 | **waived** | Requires ONNX ingest smoke inside container (build succeeded; full ingest not exercised) |
| VAL-DOCKER-002 | **pass** | `python scripts/audit_env_example.py` → `OK: all 52 src/ env vars are listed in .env.example` |
| VAL-DOCKER-003 | **fail** | `docker build -t gotcontext:test` succeeded; image size **~9.2GB** (contract target **<600MB**) |
| VAL-DOCKER-004 | **pass** | `docker run --rm gotcontext:test whoami` → `mcp`; `id` → `uid=1000(mcp)` |

## Area 2: Thread Safety & Code Quality

| ID | Status | Evidence |
|----|--------|----------|
| VAL-QUALITY-001 | **pass** | `pytest tests/test_async_operations.py tests/test_integration_workflows.py::TestBasicWorkflows::test_concurrent_ingest_workflow tests/test_performance.py::TestBurstCapacity::test_concurrent_10_documents -q --no-cov` → all passed |
| VAL-QUALITY-002 | **pass** | `rg "asyncio\.get_event_loop\(\)" src/` → zero matches |
| VAL-QUALITY-003 | **waived** | No frontend / ESLint in this repo |
| VAL-QUALITY-004 | **pass** | `ruff check src tests scripts` + `black --check src tests scripts` green after CI fix commit |
| VAL-QUALITY-005 | **pass** | Covered by `tests/test_async_operations.py` concurrent same/different doc patterns |

## Area 3: Frontend Features

| ID | Status | Evidence |
|----|--------|----------|
| VAL-UI-001 | **waived** | No frontend in repo |
| VAL-UI-002 | **waived** | No frontend in repo |
| VAL-UI-003 | **waived** | No frontend in repo |
| VAL-UI-004 | **waived** | No frontend in repo |
| VAL-UI-005 | **waived** | No frontend in repo |

## Area 4: Degradation & Resilience

| ID | Status | Evidence |
|----|--------|----------|
| VAL-DEGRADE-001 | **waived** | Requires live Redis + HTTP API stack |
| VAL-DEGRADE-002 | **waived** | Requires Supabase + Clerk deployment |
| VAL-DEGRADE-003 | **waived** | Requires Polar billing integration |
| VAL-DEGRADE-004 | **waived** | Requires multi-service outage harness |
| VAL-DEGRADE-005 | **waived** | Partial coverage via optional-dep tests; full chain needs ONNX/SBERT in CI matrix |
| VAL-DEGRADE-006 | **waived** | Requires deployed API with degradation headers |

## Additional gates (backlog)

| Check | Status | Evidence |
|-------|--------|----------|
| MCP tool count | **pass** | `setup_mcp_tools('full')` → 128 |
| Router parity | **pass** | `tests/test_all_tools_have_handlers.py` |
| Handler regression | **pass** | 123+ tests in async/compression/f11/tools/semantic_compressor suites |
| Folder guides | **pass** | `check_claude_folder_guides_sync.py` (run before push) |
| CI env audit | **pass** | `.github/workflows/ci.yml` runs `audit_env_example.py` |

## Commands to reproduce

```bash
python scripts/audit_env_example.py
python scripts/check_claude_folder_guides_sync.py
pytest tests/test_async_operations.py tests/test_compression_handlers.py \
  tests/test_f11_gated_fusion.py tests/test_all_tools_have_handlers.py -q --no-cov
```
