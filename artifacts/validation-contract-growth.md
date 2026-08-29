# Validation Contract — Growth Features & Cross-Area Flows

> gotcontext.ai SaaS · Post-Launch Growth Mission

---

## Area 5: Teams (`VAL-TEAM-xxx`)

### VAL-TEAM-001 — Team Creation

**Title:** Create a new team with a display name

**Behavioral description:**
An authenticated user submits a team name (1–64 characters, trimmed, non-empty) via `POST /v1/teams`. The API returns `201` with `{ id, name, owner_id, created_at }`. The creating user is recorded as `owner` in the `team_members` table. The team appears in the user's dashboard under "My Teams."

**Pass condition:** Response is `201`, returned `name` matches input (trimmed), `owner_id` equals the requesting user, and the team row plus owner membership row exist in the database.
**Fail condition:** Team is created without an owner membership, `name` is empty/whitespace-only and accepted, or duplicate team names for the same owner are silently allowed without clear policy.

**Evidence requirements:**
- API response payload
- Database query showing `teams` and `team_members` rows
- Dashboard screenshot showing the new team listed

---

### VAL-TEAM-002 — Invite Member by Email

**Title:** Invite a user to a team by email address

**Behavioral description:**
A team owner or admin calls `POST /v1/teams/{team_id}/invites` with `{ email }`. The API returns `201` with an invite record. An invitation email is sent (via Resend) containing a unique, time-limited accept link. If the email belongs to an existing user, they see the invite in their dashboard. If not, they receive the email and can sign up to accept.

**Pass condition:** API returns `201`, invitation row is persisted with `status=invited` and an expiry ≥ 24 hours in the future, and the Resend API is called with the correct recipient and a valid accept URL.
**Fail condition:** Invite is created but no email is sent, invite link has no expiry, or a non-owner/non-admin can send invites.

**Evidence requirements:**
- API response payload with invite ID and status
- Resend API call log or mock assertion
- Database row for the invite (status, expires_at)
- Accepting the invite transitions status to `accepted` and creates a `team_members` row

---

### VAL-TEAM-003 — Remove Member

**Title:** Remove a member from a team

**Behavioral description:**
A team owner calls `DELETE /v1/teams/{team_id}/members/{user_id}`. The member is removed from the `team_members` table. The removed user no longer sees the team in their dashboard and loses access to team-scoped resources. The owner cannot remove themselves (must transfer ownership first).

**Pass condition:** API returns `200`, the membership row is deleted, subsequent requests by the removed user to team endpoints return `403`, and attempting to remove the owner returns `400`/`409` with a descriptive error.
**Fail condition:** Membership row remains after deletion, removed user can still access team resources, or owner can self-remove leaving an ownerless team.

**Evidence requirements:**
- API response for successful removal
- API error response when attempting owner self-removal
- Database state before and after removal
- Removed user receives `403` on team-scoped API calls

---

### VAL-TEAM-004 — Transfer Ownership

**Title:** Transfer team ownership to another member

**Behavioral description:**
The current owner calls `POST /v1/teams/{team_id}/transfer` with `{ new_owner_id }`. The target must be an existing member. On success the `owner_id` on the team row updates, the previous owner is demoted to `admin`, and the new owner gains full control. An email notification is sent to the new owner.

**Pass condition:** API returns `200`, `teams.owner_id` equals `new_owner_id`, previous owner's role is `admin`, and the new owner can perform owner-only actions (e.g., delete team).
**Fail condition:** Transfer succeeds to a non-member, both users end up as owner, or the previous owner retains owner privileges.

**Evidence requirements:**
- API response confirming transfer
- Database rows showing updated `owner_id` and role change
- New owner can invoke owner-only endpoints; old owner cannot

---

### VAL-TEAM-005 — Team-Scoped Usage Aggregation

**Title:** Aggregate compression usage across all team members

**Behavioral description:**
`GET /v1/teams/{team_id}/usage?period=current_month` returns `{ total_tokens_in, total_tokens_out, total_compressions, by_member: [...] }` aggregated from all members' activity in the current billing period. Values update within 60 seconds of a new compression event. Only team owners/admins can access this endpoint.

**Pass condition:** Returned totals equal the sum of individual member usage for the period, a new compression by any member is reflected within 60 s on refresh, and a regular member calling this endpoint receives `403`.
**Fail condition:** Totals are stale beyond 60 s, individual member breakdowns don't sum to the total, or non-admin members can view team usage.

**Evidence requirements:**
- API response with aggregated and per-member breakdown
- Comparison against individual `GET /v1/usage` responses for each member
- Timing evidence (compression event timestamp vs. aggregation query timestamp)
- `403` response for unauthorized member

---

### VAL-TEAM-006 — Team Management UI in Dashboard

**Title:** Dashboard displays team management interface

**Behavioral description:**
When a team owner navigates to `/dashboard/teams/{team_id}`, the page renders: team name (editable), member list with roles, invite button, remove button per member (except self), ownership transfer option, and a usage summary chart. Non-owner members see a read-only view without invite/remove/transfer controls.

**Pass condition:** All UI elements render without console errors, owner sees all management controls, a regular member sees the member list and usage but no mutation controls, and actions (invite, remove, transfer) call the correct API endpoints.
**Fail condition:** Management controls appear for non-owners, usage chart fails to load, or UI actions target incorrect API routes.

**Evidence requirements:**
- Screenshot/DOM snapshot as owner showing all controls
- Screenshot/DOM snapshot as regular member showing read-only view
- Network tab evidence that invite/remove/transfer buttons invoke correct endpoints
- No JavaScript console errors on page load

---

## Area 6: Webhooks (`VAL-WEBHOOK-xxx`)

### VAL-WEBHOOK-001 — Create Webhook Endpoint

**Title:** Register a new webhook endpoint with URL and event subscriptions

**Behavioral description:**
An authenticated user calls `POST /v1/webhooks` with `{ url, events: ["compression.completed", ...] }`. The API validates the URL (HTTPS required in production, HTTP allowed in dev), stores the webhook with a generated `secret` for HMAC signing, and returns `201` with `{ id, url, events, secret, created_at }`. The secret is shown once at creation time.

**Pass condition:** API returns `201`, the webhook row is persisted with the correct URL and events, the returned `secret` is a 32+ character hex string, and subsequent `GET /v1/webhooks` lists the new webhook (with secret redacted).
**Fail condition:** HTTP URLs are accepted in production, secret is not returned on creation, or the webhook is created with no event subscriptions.

**Evidence requirements:**
- API response with `id`, `url`, `events`, `secret`
- Database row confirmation
- `GET /v1/webhooks` listing with `secret` masked/redacted

---

### VAL-WEBHOOK-002 — Test Ping Delivery

**Title:** Send a test ping to a webhook endpoint

**Behavioral description:**
User calls `POST /v1/webhooks/{webhook_id}/test`. The system delivers a `ping` event to the configured URL with a standard payload `{ event: "ping", webhook_id, timestamp }` and an `X-GotContext-Signature` header. The API returns the delivery result (HTTP status from the target, latency).

**Pass condition:** The target URL receives the POST with correct headers and payload, the `X-GotContext-Signature` header contains a valid HMAC-SHA256 of the body using the webhook secret, and the API response includes the target's HTTP status code and round-trip latency.
**Fail condition:** Ping is sent without signature header, payload is missing required fields, or delivery result is not reported back to the caller.

**Evidence requirements:**
- Request received at target URL (use a request-bin style endpoint)
- `X-GotContext-Signature` header value verified against manual HMAC computation
- API response showing delivery status and latency

---

### VAL-WEBHOOK-003 — Webhook Fires on Compression Event

**Title:** Webhook delivers payload when a compression completes

**Behavioral description:**
After a user compresses a document via `POST /v1/compress`, and the user has a webhook subscribed to `compression.completed`, the system asynchronously delivers a POST to the webhook URL with `{ event: "compression.completed", data: { document_id, tokens_in, tokens_out, ratio, timestamp } }` and a valid HMAC signature. Delivery occurs within 5 seconds of compression completion.

**Pass condition:** Webhook POST arrives at the target within 5 s, payload contains all specified fields with correct values matching the compression result, and the HMAC signature is valid.
**Fail condition:** Webhook does not fire, payload is missing fields, data doesn't match the actual compression result, or delivery exceeds 5 s under normal conditions.

**Evidence requirements:**
- Compression API response (document_id, tokens_in, tokens_out)
- Webhook delivery payload received at target
- Field-by-field comparison between compression result and webhook data
- Timestamp delta ≤ 5 s
- HMAC signature verification

---

### VAL-WEBHOOK-004 — Retry on Failure with Exponential Backoff

**Title:** Failed webhook deliveries are retried with exponential backoff

**Behavioral description:**
When a webhook delivery receives a 5xx response or times out, the system retries up to 5 times with exponential backoff (initial delay ~10 s, doubling each attempt, with jitter). Each attempt is logged in the delivery log. After all retries are exhausted, the delivery is marked as `failed`. If the webhook endpoint returns 4xx, no retry is attempted (permanent failure).

**Pass condition:** A 500-returning endpoint receives exactly 6 requests (1 initial + 5 retries), retry intervals approximate exponential growth (10 s, 20 s, 40 s, 80 s, 160 s ± jitter), final delivery status is `failed`, and a 400-returning endpoint receives exactly 1 request with immediate `failed` status.
**Fail condition:** Retries exceed 5, retry intervals are fixed (not exponential), 4xx triggers retries, or failed deliveries are silently dropped without logging.

**Evidence requirements:**
- Request timestamps at the target endpoint showing backoff pattern
- Delivery log entries for each attempt (status, timestamp, response code)
- Final delivery status `failed` after exhaustion
- Separate test with 4xx showing single attempt and immediate failure

---

### VAL-WEBHOOK-005 — HMAC-SHA256 Payload Signature

**Title:** All webhook payloads are signed with HMAC-SHA256

**Behavioral description:**
Every webhook delivery includes an `X-GotContext-Signature` header computed as `sha256=<hex(HMAC-SHA256(secret, raw_body))>`. The raw body is the exact JSON bytes sent in the POST body. Recipients can verify authenticity by computing the same HMAC using their stored secret.

**Pass condition:** `X-GotContext-Signature` header is present on every delivery, the value matches `sha256=` + the HMAC-SHA256 hex digest computed from the webhook secret and the raw request body, and altering one byte of the body produces a different signature.
**Fail condition:** Header is missing, uses a different algorithm, computed over a different representation (e.g., pretty-printed vs. compact JSON), or the same payload always produces the same signature regardless of secret.

**Evidence requirements:**
- Raw request body bytes captured at target
- `X-GotContext-Signature` header value
- Independent HMAC-SHA256 computation matching the header
- Negative test: modified body produces a non-matching signature

---

### VAL-WEBHOOK-006 — Delivery Log in Dashboard

**Title:** Dashboard shows webhook delivery history with status and details

**Behavioral description:**
`GET /v1/webhooks/{webhook_id}/deliveries` returns a paginated list of delivery attempts. Each entry includes `{ event, status (success|failed|pending), attempts, last_attempt_at, response_code, latency_ms }`. The dashboard UI at `/dashboard/webhooks/{id}` renders this as a table with status badges and an expandable detail view per delivery.

**Pass condition:** API returns deliveries in reverse-chronological order, each entry contains all specified fields, pagination works with `?page=&per_page=` parameters, and the UI renders status badges (green=success, red=failed, yellow=pending) correctly.
**Fail condition:** Deliveries are missing from the log, pagination returns duplicates or skips entries, or the UI does not distinguish delivery statuses visually.

**Evidence requirements:**
- API response with ≥ 3 deliveries covering success, failed, and retrying states
- Pagination test (page 1 and page 2 return distinct, contiguous results)
- Dashboard screenshot showing delivery log table with correct status badges

---

## Area 7: Self-Hosted & Enterprise (`VAL-ENTERPRISE-xxx`)

### VAL-ENTERPRISE-001 — License Key Generation

**Title:** Generate a valid enterprise license key

**Behavioral description:**
An admin (or automated flow after Stripe enterprise purchase) calls an internal endpoint to generate a license key. The key encodes: `org_id`, `tier` (enterprise), `seats`, `expires_at`, and is signed with an asymmetric key (Ed25519 or RSA). The key is a base64-encoded JWT or similar self-contained token. It is delivered to the customer via email (Resend).

**Pass condition:** Generated key decodes to reveal correct `org_id`, `tier`, `seats`, and `expires_at`; the signature is verifiable with the public key; and the key is emailed to the customer.
**Fail condition:** Key is unsigned or uses symmetric signing (secret could leak in Docker image), key omits expiry, or key is not delivered to the customer.

**Evidence requirements:**
- Decoded key payload showing all fields
- Signature verification using public key
- Resend API call log confirming email delivery

---

### VAL-ENTERPRISE-002 — Docker Image License Check on Startup

**Title:** Self-hosted Docker container validates license on startup

**Behavioral description:**
The gotcontext Docker image reads `LICENSE_KEY` from environment variable on startup. It verifies the signature using an embedded public key, checks `expires_at > now`, and validates `seats` against usage. If valid, the container starts normally and logs `License valid — org={org_id}, expires={date}`. If invalid or expired, the container logs a clear error and exits with code 1 within 10 seconds.

**Pass condition:** Valid key → container starts, health endpoint (`/health`) returns `200`. Expired key → container exits with code 1 and stderr contains `"license expired"`. Missing key → exits code 1 with `"LICENSE_KEY not set"`. Tampered key → exits code 1 with `"invalid license signature"`.
**Fail condition:** Container starts without a license, expired license is accepted, or error messages are ambiguous/missing.

**Evidence requirements:**
- `docker run` with valid key → health check passes
- `docker run` with expired key → exit code 1 + stderr captured
- `docker run` without key → exit code 1 + stderr captured
- `docker run` with tampered key → exit code 1 + stderr captured

---

### VAL-ENTERPRISE-003 — Usage Metering Phone-Home

**Title:** Self-hosted instances report usage metrics to gotcontext.ai

**Behavioral description:**
Every 24 hours (± 1 hour jitter), the self-hosted instance sends a metering payload to `https://api.gotcontext.ai/metering` containing `{ license_key, org_id, period, total_compressions, total_tokens_in, total_tokens_out, active_users }`. The payload is signed with the license key. The SaaS backend records this in a `metering_reports` table. If the phone-home fails, the instance retries 3 times with backoff and continues operating (no hard failure on network issues).

**Pass condition:** Metering payload arrives at the SaaS endpoint within the 24h window, all fields are present and non-negative, the signature is valid, and the data is persisted. On network failure, the instance continues operating and retries.
**Fail condition:** Phone-home never fires, missing fields, instance stops working when phone-home fails, or metering data is not persisted on the SaaS side.

**Evidence requirements:**
- Captured metering payload at the SaaS endpoint
- Database row in `metering_reports`
- Network-failure simulation: instance remains operational, retry attempts logged
- Metering interval within expected range

---

### VAL-ENTERPRISE-004 — Enterprise Contact Form

**Title:** Pricing page enterprise contact form submits lead to sales

**Behavioral description:**
On `/pricing`, an "Contact Sales" form collects `{ name, email, company, message }`. On submit, `POST /v1/leads` validates fields (email format, non-empty name/company), persists a row in the `leads` table with `source=pricing_enterprise`, and sends a notification email to `sales@gotcontext.ai` via Resend containing the lead details. The user sees a "Thank you" confirmation.

**Pass condition:** API returns `201`, lead row exists in DB with all fields and `source=pricing_enterprise`, Resend sends email to `sales@gotcontext.ai` with lead details, and the UI shows confirmation. Invalid email → `422` with field-level error.
**Fail condition:** Lead is persisted but email is not sent, email goes to wrong recipient, or invalid emails are accepted.

**Evidence requirements:**
- API response for valid submission
- Database `leads` row
- Resend API call showing recipient `sales@gotcontext.ai` and lead content
- API `422` response for invalid email
- UI confirmation state (screenshot or DOM snapshot)

---

### VAL-ENTERPRISE-005 — License Expiry Grace Period

**Title:** Expired license enters grace period before hard shutdown

**Behavioral description:**
When a self-hosted license expires, the instance enters a 7-day grace period. During grace, the instance operates normally but logs a daily warning: `"License expired — grace period ends {date}"`. After grace expires, the instance refuses new compression requests (returns `503`) but continues serving health checks. This prevents abrupt disruption while encouraging renewal.

**Pass condition:** Expired license (within 7 days) → instance runs, warning logged, compressions succeed. Expired license (beyond 7 days) → compressions return `503`, health returns `200`, log contains `"grace period ended"`.
**Fail condition:** Instance shuts down immediately on expiry, grace period is not enforced, or no warnings are logged during grace.

**Evidence requirements:**
- License with `expires_at` = yesterday → instance runs, warning in logs, compression succeeds
- License with `expires_at` = 8 days ago → compression returns `503`, health returns `200`
- Log output showing grace period warnings

---

## Cross-Area Flows (`VAL-CROSS-xxx`)

### VAL-CROSS-001 — Team Webhook End-to-End

**Title:** Team member's compression triggers webhook delivery to team-configured endpoint

**Behavioral description:**
A team admin configures a webhook on the team's account subscribed to `compression.completed`. A different team member performs a compression via `POST /v1/compress`. The webhook fires with the compression data and is delivered to the team's endpoint. The delivery is visible in the team's webhook delivery log, and the compression is counted in the team's usage aggregation.

**Pass condition:** Webhook fires within 5 s of compression, payload attributes the event to the correct team and member, the delivery appears in the team webhook log, and team usage aggregation reflects the new compression.
**Fail condition:** Webhook does not fire for non-admin member compressions, delivery is missing from team log, or usage is not aggregated under the team.

**Evidence requirements:**
- Team webhook configuration (team-scoped, not user-scoped)
- Compression by non-admin member → webhook payload at target
- Team delivery log includes the event
- Team usage endpoint reflects updated totals

**Areas spanned:** Teams (Area 5) · Webhooks (Area 6)

---

### VAL-CROSS-002 — Self-Hosted Docker with ONNX Model

**Title:** Self-hosted Docker image correctly loads ONNX embedding model

**Behavioral description:**
The gotcontext Docker image, started with a valid `LICENSE_KEY`, initializes the ONNX embedding backend (not SBERT, to reduce image size). A compression request via the MCP or HTTP API produces valid embeddings and a correct skeleton output. The embedding dimension matches the expected ONNX model output (e.g., 384-d for all-MiniLM).

**Pass condition:** Container starts, `/health` returns `200` with `embedding_backend: "onnx"` in status, a compression request returns a valid skeleton with non-zero similarity scores, and embedding vectors have the correct dimensionality.
**Fail condition:** Container falls back to TF-IDF silently, ONNX model file is missing from the image, or compression produces zero-vector embeddings.

**Evidence requirements:**
- `docker run` with valid license → health check showing ONNX backend
- Compression API response with valid skeleton
- Embedding vector sample showing correct dimensionality
- Docker image layer inspection confirming ONNX model file presence

**Areas spanned:** Self-Hosted (Area 7) · Embeddings/Docker infrastructure

---

### VAL-CROSS-003 — Redis-Down Degradation During Team Usage Aggregation

**Title:** Team usage aggregation degrades gracefully when Redis is unavailable

**Behavioral description:**
With Redis down (simulated via network partition or stopped container), a team usage aggregation request (`GET /v1/teams/{team_id}/usage`) falls back to a direct database query. The response is still correct but may have higher latency. A warning is logged: `"Redis unavailable — falling back to DB for usage aggregation"`. Individual compressions continue to work (usage is written to DB regardless of Redis). When Redis recovers, the cache is repopulated on the next request.

**Pass condition:** Usage endpoint returns correct data within 10 s (degraded latency acceptable), warning is logged, compressions are unaffected, and cache repopulates after Redis recovery.
**Fail condition:** Usage endpoint returns `500`, compressions fail due to Redis being down, or stale data is served after recovery without repopulation.

**Evidence requirements:**
- Redis stopped → usage endpoint returns correct data (compare with DB query)
- Log output showing fallback warning
- Compression request succeeds during Redis outage
- Redis restarted → subsequent usage request repopulates cache (Redis key exists)

**Areas spanned:** Teams (Area 5) · Infrastructure/Degradation

---

### VAL-CROSS-004 — New User Onboarding into Team

**Title:** New user signs up, joins a team via invite, and sees team usage

**Behavioral description:**
A team admin sends an invite to `newuser@example.com`. The user signs up (OAuth or email/password), which lands them on the dashboard. They see a pending team invite notification. They accept the invite, which adds them to the team. Navigating to the team page shows the team's aggregated usage and member list including themselves.

**Pass condition:** Signup succeeds, invite notification is visible on dashboard within first load, accepting the invite transitions their membership to `active`, team page shows them in the member list, and team usage data is visible.
**Fail condition:** Invite is lost during signup flow, user must manually navigate to find the invite, acceptance fails silently, or team usage is hidden from new members.

**Evidence requirements:**
- Invite sent → user signs up → dashboard shows invite notification
- Accept invite → `team_members` row with `status=active`
- Team page shows new member in list
- Team usage endpoint accessible by new member (read-only)

**Areas spanned:** Auth/Signup · Teams (Area 5) · Dashboard

---

### VAL-CROSS-005 — Fidelity Profile Compression with Usage Tracking and Webhook

**Title:** Compression using a fidelity profile triggers correct usage tracking and webhook delivery

**Behavioral description:**
A user selects a high-fidelity compression profile (e.g., `"code_review"` which preserves more structure). They compress a document. The usage tracker records the compression with the profile metadata (`profile_name`, `fidelity_level`). A webhook subscribed to `compression.completed` fires with the profile information included in the payload. The usage dashboard shows the compression tagged with the profile used.

**Pass condition:** Compression uses the specified profile (output preserves more tokens than default), usage record includes `profile_name` and `fidelity_level`, webhook payload contains profile metadata, and the usage dashboard displays the profile tag.
**Fail condition:** Profile is ignored and default compression is applied, usage record omits profile metadata, or webhook payload doesn't include profile information.

**Evidence requirements:**
- Compression request with `profile=code_review` → response showing higher-fidelity output
- Usage API showing the record with profile metadata
- Webhook payload at target including `profile_name` and `fidelity_level`
- Dashboard usage view showing profile tag on the entry

**Areas spanned:** Compression Profiles · Usage Tracking · Webhooks (Area 6)

---

### VAL-CROSS-006 — Self-Hosted Metering Reflects Team Usage

**Title:** Self-hosted instance phone-home metering accurately reports team-aggregated usage

**Behavioral description:**
On a self-hosted instance, multiple team members perform compressions throughout the day. The 24-hour metering phone-home payload includes `active_users` count and `total_compressions` that correctly reflect all team members' activity — not just the instance operator's. The SaaS backend correlates this with the team's license and persists it for billing/audit.

**Pass condition:** Metering payload `active_users` matches the distinct users who compressed in the period, `total_compressions` matches the sum across all users, and the SaaS `metering_reports` row is linked to the correct `org_id` from the license.
**Fail condition:** Metering only counts the admin user, totals are undercounted, or the SaaS cannot correlate the report with the correct organization.

**Evidence requirements:**
- Multiple users compress on self-hosted instance → metering payload captured
- `active_users` and `total_compressions` match instance-local records
- SaaS `metering_reports` row with correct `org_id` linkage
- Billing/audit query returns accurate per-org totals

**Areas spanned:** Self-Hosted (Area 7) · Teams (Area 5) · Usage Tracking
