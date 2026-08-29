# Token Saver 5000 — Engineering Backlog

> Updated: 2026-08-29 CEO update. HEAD `e8fc124` — CI green run **`33254371815`**. Evidence: `artifacts/ceo-update-2026-08-29.md`.

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

### CEO-gate: SaaS → gotcontext-main (parked contracts, not engine work)

Defer brief: `docs/strategy/2026-08-28-growth-val-defer-brief.md`  
Contracts: `artifacts/validation-contract-growth.md`  
Platform backlog: `gotcontext-main/docs/BACKLOG.md`

| ID | Item | Blocker |
|----|------|---------|
| VAL-TEAM-001 | Team Creation | **CEO-gate: SaaS → gotcontext-main** |
| VAL-TEAM-002 | Invite Member by Email | **CEO-gate: SaaS → gotcontext-main** |
| VAL-TEAM-003 | Remove Member | **CEO-gate: SaaS → gotcontext-main** |
| VAL-TEAM-004 | Transfer Ownership | **CEO-gate: SaaS → gotcontext-main** |
| VAL-TEAM-005 | Team-Scoped Usage Aggregation | **CEO-gate: SaaS → gotcontext-main** |
| VAL-TEAM-006 | Team Management UI in Dashboard | **CEO-gate: SaaS → gotcontext-main** |
| VAL-WEBHOOK-001 | Create Webhook Endpoint | **CEO-gate: SaaS → gotcontext-main** |
| VAL-WEBHOOK-002 | Test Ping Delivery | **CEO-gate: SaaS → gotcontext-main** |
| VAL-WEBHOOK-003 | Webhook Fires on Compression Event | **CEO-gate: SaaS → gotcontext-main** |
| VAL-WEBHOOK-004 | Retry on Failure with Exponential Backoff | **CEO-gate: SaaS → gotcontext-main** |
| VAL-WEBHOOK-005 | HMAC-SHA256 Payload Signature | **CEO-gate: SaaS → gotcontext-main** |
| VAL-WEBHOOK-006 | Delivery Log in Dashboard | **CEO-gate: SaaS → gotcontext-main** |
| VAL-ENTERPRISE-001 | License Key Generation | **CEO-gate: SaaS → gotcontext-main** |
| VAL-ENTERPRISE-002 | Docker License Check on Startup | **CEO-gate: SaaS → gotcontext-main** + no LICENSE_KEY in MCP image |
| VAL-ENTERPRISE-003 | Usage Metering Phone-Home | **CEO-gate: SaaS → gotcontext-main** |
| VAL-ENTERPRISE-004 | Enterprise Contact Form | **CEO-gate: SaaS → gotcontext-main** |
| VAL-ENTERPRISE-005 | License Expiry Grace Period | **CEO-gate: SaaS → gotcontext-main** |
| VAL-CROSS-001 | Team Webhook End-to-End | **CEO-gate: SaaS → gotcontext-main** |
| VAL-CROSS-002 | Self-Hosted Docker ONNX | **CEO-gate** — partial engine smoke; full contract SaaS/self-host |
| VAL-CROSS-003 | Redis-Down Team Usage Aggregation | **CEO-gate: SaaS → gotcontext-main** |
| VAL-CROSS-004 | New User Onboarding into Team | **CEO-gate: SaaS → gotcontext-main** |
| VAL-CROSS-005 | Fidelity + Usage + Webhook | **CEO-gate: SaaS → gotcontext-main** |
| VAL-CROSS-006 | Self-Hosted Metering Reflects Team Usage | **CEO-gate: SaaS → gotcontext-main** |

### Pipeline gaps (honest state, not work)
- verify-feature / feature-verify / review-router / audit-until-clear / dogfood — not run on BL-NPR

## Thinktank verdict (2026-08-28)

**CHANGES_REQUIRED** on monolithic P1 — executed as ordered slices. Do not merge further file splits without per-slice test green.
