# Token Saver 5000 — Engineering Backlog

> Updated: 2026-08-29 growth-val-park-relocate. Growth VAL park → gotcontext-main (23 IDs).

## Done

### P0–P4 pipeline
- [x] P0-A security / P0-B docs SSOT / P1 splits / P2 tests / P3 reference docs / P4 validation receipt
- [x] CI green on `f7f479f` — run **`33217369479`**
- [x] Docker smoke: VAL-DOCKER-001..004 pass; image **519MB** ONNX-only (`f7f479f`)

### BL-NPR NumPy PageRank fallback (2026-08-28)
- [x] BL-NPR-001/002/003 — `compute_pagerank` in code_compressor, adaptive_rate_allocator, multimodal_compressor — `b70f2bb`
- [x] Tests: `tests/test_pagerank_numpy_fallback.py`
- [x] Wayfinder: `.wayfinder/numpy-pagerank-fallback/MAP.md`

### Session closeout (2026-08-29)
- [x] `tests/claude.md` sync — `59cd255` · CI **`33234361179`**
- [x] CI green receipt on HEAD — `e8fc124` · CI **`33254371815`**
- [x] Secret sweep CLEAN (tracked src/tests/scripts)

## Open / follow-up

### Engine AI-doable
*(none)*

### SaaS Growth VAL (relocated)
Parked validation IDs **VAL-TEAM / VAL-WEBHOOK / VAL-ENTERPRISE / VAL-CROSS** live in the platform repo:
- **Owner list:** `C:\dev\projects\gotcontext-main\docs\growth-val-park.md` (also `docs/growth-val-park.md` inside gotcontext-main)
- **Contracts (engine artifact, kept):** `artifacts/validation-contract-growth.md`
- **Defer brief:** `docs/strategy/2026-08-28-growth-val-defer-brief.md`
Do **not** implement these in token-saver-5000.

### Pipeline gaps (honest state, not work)
- verify-feature / feature-verify / review-router / audit-until-clear / dogfood — not run on BL-NPR

## Thinktank verdict (2026-08-28)

**CHANGES_REQUIRED** on monolithic P1 — executed as ordered slices. Do not merge further file splits without per-slice test green.
