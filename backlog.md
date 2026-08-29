# Token Saver 5000 — Engineering Backlog

> Updated: 2026-08-29 session closeout. HEAD `b031441` (pending CI re-green after `tests/claude.md` sync).

## Done

### P0–P4 pipeline
- [x] P0-A security / P0-B docs SSOT / P1 splits / P2 tests / P3 reference docs / P4 validation receipt
- [x] CI green on `f7f479f` — run **`33217369479`**
- [x] Docker smoke: VAL-DOCKER-001..004 pass; image **519MB** ONNX-only (`f7f479f`)

### BL-NPR NumPy PageRank fallback (2026-08-28)
- [x] BL-NPR-001/002/003 — `compute_pagerank` in code_compressor, adaptive_rate_allocator, multimodal_compressor
- [x] Tests: `tests/test_pagerank_numpy_fallback.py` (wayfinder checks verified locally)
- [x] Wayfinder: `.wayfinder/numpy-pagerank-fallback/MAP.md` — all Checks pass locally

## Open / follow-up

### CI / hygiene (session closeout)
- [ ] **CI re-green on HEAD** — run **`33230864165`** **FAILED** (Quality Gate: `tests/claude.md` out of sync). Fix staged: regenerate + commit `tests/claude.md` in closeout commit **`pending`**. Blocker: closeout push → new run.

### CEO-gate: SaaS-not-in-repo (`artifacts/validation-contract-growth.md`)
Defer brief: `docs/strategy/2026-08-28-growth-val-defer-brief.md` (**BLOCK** — do not build here)

| ID | Item | Blocker |
|----|------|---------|
| VAL-TEAM-001 | Team Creation | **CEO-gate: SaaS-not-in-repo** |
| VAL-TEAM-002 | Invite Member by Email | **CEO-gate: SaaS-not-in-repo** |
| VAL-TEAM-003 | Remove Member | **CEO-gate: SaaS-not-in-repo** |
| VAL-TEAM-004 | Transfer Ownership | **CEO-gate: SaaS-not-in-repo** |
| VAL-TEAM-005 | Team-Scoped Usage Aggregation | **CEO-gate: SaaS-not-in-repo** |
| VAL-TEAM-006 | Team Management UI in Dashboard | **CEO-gate: SaaS-not-in-repo** |
| VAL-WEBHOOK-001 | Create Webhook Endpoint | **CEO-gate: SaaS-not-in-repo** |
| VAL-WEBHOOK-002 | Test Ping Delivery | **CEO-gate: SaaS-not-in-repo** |
| VAL-WEBHOOK-003 | Webhook Fires on Compression Event | **CEO-gate: SaaS-not-in-repo** |
| VAL-WEBHOOK-004 | Retry on Failure with Exponential Backoff | **CEO-gate: SaaS-not-in-repo** |
| VAL-WEBHOOK-005 | HMAC-SHA256 Payload Signature | **CEO-gate: SaaS-not-in-repo** |
| VAL-WEBHOOK-006 | Delivery Log in Dashboard | **CEO-gate: SaaS-not-in-repo** |
| VAL-ENTERPRISE-001 | License Key Generation | **CEO-gate: SaaS-not-in-repo** |
| VAL-ENTERPRISE-002 | Docker License Check on Startup | **CEO-gate: SaaS-not-in-repo** + no LICENSE_KEY in MCP image |
| VAL-ENTERPRISE-003 | Usage Metering Phone-Home | **CEO-gate: SaaS-not-in-repo** |
| VAL-ENTERPRISE-004 | Enterprise Contact Form | **CEO-gate: SaaS-not-in-repo** |
| VAL-ENTERPRISE-005 | License Expiry Grace Period | **CEO-gate: SaaS-not-in-repo** |
| VAL-CROSS-001 | Team Webhook End-to-End | **CEO-gate: SaaS-not-in-repo** |
| VAL-CROSS-002 | Self-Hosted Docker ONNX | **CEO-gate** — partial smoke only; full license/health not measured |
| VAL-CROSS-003 | Redis-Down Team Usage Aggregation | **CEO-gate: SaaS-not-in-repo** |
| VAL-CROSS-004 | New User Onboarding into Team | **CEO-gate: SaaS-not-in-repo** |
| VAL-CROSS-005 | Fidelity + Usage + Webhook | **CEO-gate: SaaS-not-in-repo** |
| VAL-CROSS-006 | Self-Hosted Metering Reflects Team Usage | **CEO-gate: SaaS-not-in-repo** |

### Pipeline gaps (not backlog work — honest state)
- `verify-feature` / `feature-verify` / `review-router` / `audit-until-clear` / `dogfood` — **not run** on BL-NPR SHA; no user-visible feature this slice.

## Thinktank verdict (2026-08-28)

**CHANGES_REQUIRED** on monolithic P1 — executed as ordered slices. Do not merge further file splits without per-slice test green.
