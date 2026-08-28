# Token Saver 5000 — Engineering Backlog

> Updated: 2026-08-28 after docker-onnx-only-size-under-600mb.

## Done

### P0–P4 pipeline
- [x] P0-A security / P0-B docs SSOT / P1 splits / P2 tests / P3 reference docs / P4 validation receipt
- [x] CI green on `main` after split-module export repairs
- [x] Docker smoke closeout: VAL-DOCKER-001/002/003/004 **all pass**
- [x] Image size **519MB** (&lt;600MB) via ONNX-only runtime (was 9.2GB CUDA → 3.7GB CPU-torch)

## Open / follow-up

- [ ] Growth VAL-* (`validation-contract-growth.md`) when SaaS features land in repo
- [ ] Optional: extend NumPy PageRank fallback to `code_compressor` / `adaptive_rate_allocator` / `multimodal_compressor` (still call `nx.pagerank` directly)

## Thinktank verdict (2026-08-28)

**CHANGES_REQUIRED** on monolithic P1 — executed as ordered slices. Do not merge further file splits without per-slice test green.
