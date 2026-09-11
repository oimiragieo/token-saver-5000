# MAP — NumPy PageRank fallback (BL-NPR-001/002/003)

## Destination

All compression graph PageRank paths use `pagerank_numpy.compute_pagerank` so Docker ONNX runtime (networkx without scipy) does not crash on `ModuleNotFoundError: scipy`.

## Open questions

(none)

## Out of scope

- SaaS Growth VAL-* (CEO-gate)
- New PageRank algorithm — reuse existing `pagerank_numpy` / `compute_pagerank`
- torch-free adaptive_rate_allocator work (already done)

## Answers

### compute_pagerank falls back when scipy missing

**Answer:** `compute_pagerank` catches `ImportError` from `nx.pagerank` (not only from `import networkx`) and calls `pagerank_numpy`.

**Why:** Docker run proved `nx.pagerank` raises `ModuleNotFoundError: scipy` while networkx imports fine.

**Check:** `pytest tests/test_pagerank_numpy_fallback.py -q --no-cov` → PASS; docker `python -c "from src.pagerank_numpy import compute_pagerank; import networkx as nx; compute_pagerank(nx.path_graph(3))"` → no raise.
**Judged by:** run it

### code_compressor uses compute_pagerank

**Answer:** Replace `nx.pagerank(undirected)` with `compute_pagerank(undirected)` in `ingest_file` importance scoring.

**Check:** `rg 'nx\.pagerank' src/code_compressor.py` → 0 matches.
**Judged by:** run it

### adaptive_rate_allocator uses compute_pagerank

**Answer:** Replace `nx.pagerank(graph)` in complexity metric with `compute_pagerank(graph)`.

**Check:** `rg 'nx\.pagerank' src/adaptive_rate_allocator.py` → 0 matches.
**Judged by:** run it

### multimodal_compressor uses compute_pagerank

**Answer:** Replace `nx.pagerank(graph)` in project ingest with `compute_pagerank(graph)`.

**Check:** `rg 'nx\.pagerank' src/multimodal_compressor.py` → 0 matches.
**Judged by:** run it
