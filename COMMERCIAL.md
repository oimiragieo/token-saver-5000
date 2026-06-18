# Commercial licensing — token-saver-5000

token-saver-5000 is **source-available** under the Business Source License 1.1
(see `LICENSE.draft`). Most use is free. This file explains the one case that
needs a paid license, and how to get it.

## Free — no license needed
- Individuals, students, researchers, hobbyists.
- Evaluation, development, testing, internal tooling.
- **Running it inside your own company**, including for commercial work — as
  long as you are not reselling the engine itself to third parties as a service.

## Requires a commercial license
You need a commercial agreement with gotcontext.ai if you want to:
- Offer the engine (or a substantially similar token-compression capability
  derived from it) to third parties as a **hosted, managed, or embedded
  service** that competes with gotcontext.ai; or
- Redistribute it inside a product where you need terms different from the BSL
  (e.g. an OEM/embedding agreement, an indemnity, or support/SLA guarantees).

## How to get one
Two existing paths already serve this — no new billing system required:
1. **Self-hosted / enterprise license** — the Ed25519-signed self-hosted
   license the product already issues (Docker image + license key). This is the
   "pay to run it on your own infra with support" path.
2. **Custom commercial agreement** — for OEM/embedding or reseller terms.

Contact: https://gotcontext.ai/pricing  (enterprise / self-hosted section).

## Why this license
- **The engine stays open and forkable** — we never cripple the source code.
  The only restriction is *reselling it as a competing service*; everything a
  normal user or company wants to do is free.
- **It auto-opens.** Under BSL, each version converts to Apache 2.0 on its
  Change Date (4 years), so this is not a permanent lock.
- **The moat is the hosted product**, not the algorithm — the managed API, the
  MCP gateway, reliability, and the data, not the compression code.
