# Validation Run Receipt — 2026-08-28

> Execution of VAL-* contracts from `artifacts/validation-contract-areas-1-4.md`  
> **Environment:** Windows local, Python 3.14; Docker Desktop rebuild after CPU-torch Dockerfile fix

## Summary

| Area | Pass | Waived | Fail | Pending |
|------|------|--------|------|---------|
| Docker (4) | 3 | 0 | 1 | 0 |
| Quality (5) | 4 | 1 | 0 | 0 |
| UI (5) | 0 | 5 | 0 | 0 |
| Degrade (6) | 0 | 6 | 0 | 0 |
| **Areas 1–4 total** | **7** | **12** | **1** | **0** |

Growth contracts (`validation-contract-growth.md`) are **waived** — SaaS/team/webhook features not in this MCP server repo.

## Area 1: Docker & Infrastructure

| ID | Status | Evidence |
|----|--------|----------|
| VAL-DOCKER-001 | **pass** | `HF_HOME=/home/mcp/.cache/huggingface` set; `ONNXEmbeddingManager().encode(['test'])` → shape `(1, 384)` as uid 1000; ingest via `SemanticCompressor` reports `EmbeddingTier.STANDARD` (not TFIDF). ONNX cache pre-exported in builder under `/home/mcp/.cache/token-saver-5000`. |
| VAL-DOCKER-002 | **pass** | `python scripts/audit_env_example.py` → `OK: all 52 src/ env vars are listed in .env.example` |
| VAL-DOCKER-003 | **fail** (size only) | `docker build -t gotcontext:test` exits 0; ingest round-trip succeeds (160 tokens → ratio 4.1). Image size **3.7GB** after CPU-torch fix (was **9.2GB** with CUDA wheels). Contract target **&lt;600MB** still missed — follow-up: ONNX-only runtime (drop torch from final image). |
| VAL-DOCKER-004 | **pass** | `docker run --rm gotcontext:test whoami` → `mcp`; `id` → `uid=1000(mcp)`; `/home/mcp/.cache` owned by `mcp`. |

## Area 2: Thread Safety & Code Quality

| ID | Status | Evidence |
|----|--------|----------|
| VAL-QUALITY-001 | **pass** | Concurrent async ingest / performance tests green |
| VAL-QUALITY-002 | **pass** | `rg "asyncio\.get_event_loop\(\)" src/` → zero matches |
| VAL-QUALITY-003 | **waived** | No frontend / ESLint in this repo |
| VAL-QUALITY-004 | **pass** | `ruff` + `black --check` green; CI Full Validation green on `eb96b98`+ |
| VAL-QUALITY-005 | **pass** | Covered by `tests/test_async_operations.py` |

## Area 3: Frontend Features

| ID | Status | Evidence |
|----|--------|----------|
| VAL-UI-001..005 | **waived** | No frontend in repo |

## Area 4: Degradation & Resilience

| ID | Status | Evidence |
|----|--------|----------|
| VAL-DEGRADE-001..006 | **waived** | Require live SaaS / Redis / billing stack |

## Additional gates (backlog)

| Check | Status | Evidence |
|-------|--------|----------|
| MCP tool count | **pass** | `setup_mcp_tools('full')` → 128 |
| Router parity | **pass** | `tests/test_all_tools_have_handlers.py` |
| CI | **pass** | https://github.com/oimiragieo/token-saver-5000/actions/runs/33208968025 |
| Folder guides | **pass** | `check_claude_folder_guides_sync.py` |

## Commands to reproduce

```bash
python scripts/audit_env_example.py
docker build -t gotcontext:test .
docker images gotcontext:test --format "{{.Size}}"
docker run --rm gotcontext:test id
docker run --rm gotcontext:test python -c "from src.embeddings_onnx import ONNXEmbeddingManager; print(ONNXEmbeddingManager().encode(['test']).shape)"
docker run --rm gotcontext:test python -c "from src.embeddings import EmbeddingManager; from src.semantic_compressor import SemanticCompressor; print(EmbeddingManager()._tier); print(SemanticCompressor().ingest_file(('x '*200), 'd').compression_ratio)"
```
