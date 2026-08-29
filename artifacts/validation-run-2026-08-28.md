# Validation Run Receipt — 2026-08-28

> Execution of VAL-* contracts from `artifacts/validation-contract-areas-1-4.md`  
> **Environment:** Windows local; Docker Desktop ONNX-only runtime image

## Summary

| Area | Pass | Waived | Fail | Pending |
|------|------|--------|------|---------|
| Docker (4) | 4 | 0 | 0 | 0 |
| Quality (5) | 4 | 1 | 0 | 0 |
| UI (5) | 0 | 5 | 0 | 0 |
| Degrade (6) | 0 | 6 | 0 | 0 |
| **Areas 1–4 total** | **8** | **12** | **0** | **0** |

Growth contracts (`validation-contract-growth.md`) remain **open under CEO-gate: SaaS-not-in-repo** (enumerated in `backlog.md` / `artifacts/ceo-update-2026-08-28.md`). Do not treat “waived” as closed — especially VAL-CROSS-002 (partial ONNX smoke ≠ full license/health contract).

## Area 1: Docker & Infrastructure

| ID | Status | Evidence |
|----|--------|----------|
| VAL-DOCKER-001 | **pass** | `EMBEDDING_TIER=onnx`; `HF_HOME` set; `ONNXEmbeddingManager().encode` → `(1, 384)` as uid 1000; no torch in image; tier=`onnx` (not TFIDF). Quantized ONNX model pre-exported in builder. |
| VAL-DOCKER-002 | **pass** | `python scripts/audit_env_example.py` → OK (includes `EMBEDDING_TIER`) |
| VAL-DOCKER-003 | **pass** | `docker build -t gotcontext:test` exits 0; ingest round-trip succeeds; **image size 519MB** (&lt;600MB). Path: 9.2GB CUDA → 3.7GB CPU-torch → 519MB ONNX-only. |
| VAL-DOCKER-004 | **pass** | `whoami`/`id` → `mcp` / `uid=1000(mcp)` |

## Area 2–4

Unchanged from prior receipt (quality pass; UI/degrade waived).

## Docker architecture (closeout)

- **Builder:** CPU torch + optimum export + dynamic INT8 quantize of `BAAI/bge-small-en-v1.5`
- **Runtime:** `requirements-docker-runtime.txt` — no torch / transformers / sentence-transformers / sklearn / scipy
- **Fallbacks:** `src.similarity` (NumPy cosine), `src.pagerank_numpy` (NumPy PageRank when SciPy absent)
- **Env:** `EMBEDDING_TIER=onnx` honored by `EmbeddingManager`

## Commands to reproduce

```bash
docker build -t gotcontext:test .
docker images gotcontext:test --format "{{.Size}}"   # expect <600MB
docker run --rm gotcontext:test id
docker run --rm gotcontext:test python -c "from src.embeddings_onnx import ONNXEmbeddingManager; from src.embeddings import EmbeddingManager; EmbeddingManager.reset_for_testing(); print(ONNXEmbeddingManager().encode(['t']).shape, EmbeddingManager().get_tier().value)"
```
