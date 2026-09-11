# Plan — NumPy PageRank fallback slice (BL-NPR-001/002/003)

**Wayfinder:** `.wayfinder/numpy-pagerank-fallback/MAP.md`  
**Thinktank:** mechanical engine seam — APPROVED for solo implementation (no auth/money/public surface).

## Premise checks (verify-plan-against-code)

- [x] `src/pagerank_numpy.py` exists with `compute_pagerank` used by `semantic_compressor.py`
- [x] Docker `gotcontext:test` — `nx.pagerank` fails with `ModuleNotFoundError: scipy`
- [x] `code_compressor.py`, `adaptive_rate_allocator.py`, `multimodal_compressor.py` still call `nx.pagerank` directly

## Tasks

1. **Fix `compute_pagerank`** — wrap `nx.pagerank` call in same `except ImportError` as fallback (scipy missing).
2. **BL-NPR-001** — `code_compressor.py`: `from .pagerank_numpy import compute_pagerank`; use on undirected graph.
3. **BL-NPR-002** — `adaptive_rate_allocator.py`: import + replace in `_compute_graph_complexity` (or equivalent).
4. **BL-NPR-003** — `multimodal_compressor.py`: import + replace in project ingest.
5. **Tests** — `tests/test_pagerank_numpy_fallback.py`: numpy fallback when `nx.pagerank` raises ImportError; optional docker parity command in workflow-ledger.
6. **Verify** — `rg 'nx\.pagerank' src/code_compressor.py src/adaptive_rate_allocator.py src/multimodal_compressor.py` → 0; fast pytest subset + targeted tests.
7. **Docs** — `backlog.md`, `workflow-ledger.md`, `.orchestrator/state.json`.

## Depth

Solo (3 mechanical edits + 1 seam fix + tests). No UI, no fan-out tree required.

## Definition of done

- No direct `nx.pagerank` in the three modules (pagerank_numpy internal use OK).
- `compute_pagerank` works in Docker without scipy.
- BL-NPR-001/002/003 checked off in backlog.
- Tests green locally on changed paths.
