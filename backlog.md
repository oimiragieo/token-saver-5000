# Token Saver 5000 — Engineering Backlog

> Updated: 2026-08-28 after numpy-pagerank-fallback slice (BL-NPR-001/002/003).

## Done

### P0–P4 pipeline
- [x] P0-A security / P0-B docs SSOT / P1 splits / P2 tests / P3 reference docs / P4 validation receipt
- [x] CI green on `main` at `f7f479f` — run **`33217369479`** (Full Validation success)
- [x] Docker smoke: VAL-DOCKER-001..004 pass; image **519MB** ONNX-only (`f7f479f`)

### BL-NPR NumPy PageRank fallback (2026-08-28)
- [x] BL-NPR-001 — `code_compressor.py` → `compute_pagerank`
- [x] BL-NPR-002 — `adaptive_rate_allocator.py` → `compute_pagerank`
- [x] BL-NPR-003 — `multimodal_compressor.py` → `compute_pagerank`
- [x] Tests: `tests/test_pagerank_numpy_fallback.py`; Docker `compute_pagerank` without scipy

## Open / follow-up

### CEO-gate: SaaS-not-in-repo (`artifacts/validation-contract-growth.md`)
- [ ] VAL-TEAM-001 — Team Creation
- [ ] VAL-TEAM-002 — Invite Member by Email
- [ ] VAL-TEAM-003 — Remove Member
- [ ] VAL-TEAM-004 — Transfer Ownership
- [ ] VAL-TEAM-005 — Team-Scoped Usage Aggregation
- [ ] VAL-TEAM-006 — Team Management UI in Dashboard
- [ ] VAL-WEBHOOK-001 — Create Webhook Endpoint
- [ ] VAL-WEBHOOK-002 — Test Ping Delivery
- [ ] VAL-WEBHOOK-003 — Webhook Fires on Compression Event
- [ ] VAL-WEBHOOK-004 — Retry on Failure with Exponential Backoff
- [ ] VAL-WEBHOOK-005 — HMAC-SHA256 Payload Signature
- [ ] VAL-WEBHOOK-006 — Delivery Log in Dashboard
- [ ] VAL-ENTERPRISE-001 — License Key Generation
- [ ] VAL-ENTERPRISE-002 — Docker Image License Check on Startup (**also:** LICENSE_KEY path not in MCP image)
- [ ] VAL-ENTERPRISE-003 — Usage Metering Phone-Home
- [ ] VAL-ENTERPRISE-004 — Enterprise Contact Form
- [ ] VAL-ENTERPRISE-005 — License Expiry Grace Period
- [ ] VAL-CROSS-001 — Team Webhook End-to-End
- [ ] VAL-CROSS-002 — Self-Hosted Docker with ONNX Model (partial ONNX/size smoke only — full contract CEO-gate)
- [ ] VAL-CROSS-003 — Redis-Down Degradation During Team Usage Aggregation
- [ ] VAL-CROSS-004 — New User Onboarding into Team
- [ ] VAL-CROSS-005 — Fidelity Profile Compression with Usage Tracking and Webhook
- [ ] VAL-CROSS-006 — Self-Hosted Metering Reflects Team Usage

## Thinktank verdict (2026-08-28)

**CHANGES_REQUIRED** on monolithic P1 — executed as ordered slices. Do not merge further file splits without per-slice test green.
