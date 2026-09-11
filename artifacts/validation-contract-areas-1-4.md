# Validation Contract — gotcontext.ai Mission (Areas 1–4)

> **Generated:** 2026-04-10
> **Scope:** Docker & Infrastructure, Thread Safety & Code Quality, Frontend Features, Degradation & Resilience
> **Status:** DRAFT — requires review before execution

---

## Area 1: Docker & Infrastructure

### VAL-DOCKER-001 — ONNX Model Loads as appuser (No TF-IDF Fallback)

**Title:** ONNX embedding tier loads successfully inside container as non-root user

**Behavioral Description:**
When the Docker container starts and a compression request is made that triggers the ONNX embedding tier, the model must load from the pre-cached `HF_HOME` directory owned by `appuser` (uid 1000, user `mcp`). The embedding tier reported in the response or logs must be `ONNX`, **not** `TFIDF`.

**Pass Condition:**
- `docker run` the built image → invoke `ingest_context` with a 500+ token document → response metadata or structured logs show `embedding_tier: "ONNX"` (or `"STANDARD"` if SBERT is preferred; key point: **not** `"TFIDF"`).
- No `embedding_degraded` or `embedding_tier_failed` warning for ONNX in container logs.
- `HF_HOME` env var is explicitly set in the Dockerfile (e.g., `ENV HF_HOME=/home/mcp/.cache/huggingface`) **or** the ONNX model files are pre-downloaded in the builder stage and `COPY --chown=mcp:mcp` into the runtime image.

**Fail Condition:**
- Logs contain `embedding_degraded … actual_tier=TFIDF` or `onnx … PermissionError`.
- ONNX model download is attempted at runtime (network not available in air-gapped deploys) and fails, causing silent fallback to TF-IDF.

**Evidence Requirements:**
1. `docker build` succeeds without errors.
2. `docker run -i <image> python -c "from src.embeddings_onnx import ONNXEmbeddingManager; m = ONNXEmbeddingManager(); print(m.encode(['test']).shape)"` completes without `PermissionError`.
3. Container logs (`docker logs`) for a full ingest cycle show no ONNX fallback warnings.

---

### VAL-DOCKER-002 — .env.example Documents All Runtime Env Vars

**Title:** Every `os.getenv` / `os.environ` key used in `src/` is listed in `.env.example`

**Behavioral Description:**
A scripted audit extracts all unique env var names referenced via `os.getenv(...)` or `os.environ.get(...)` or `os.environ[...]` across every Python file under `src/`. Each extracted key must have a corresponding entry (commented or uncommented) in `.env.example`.

**Pass Condition:**
- Running the audit script produces an empty diff (zero missing vars).
- Specifically, the following env vars currently missing from `.env.example` are added:
  - `TEE_MODE`, `TEE_COMPRESSION_THRESHOLD`, `TEE_MAX_ENTRIES`, `TEE_MAX_SIZE_MB`
  - `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_ENABLE_OTLP`
  - `TOKEN_BUDGET_SESSION`, `TOKEN_BUDGET_DAILY`, `TOKEN_BUDGET_MONTHLY`
  - `TOOL_RESULT_SOFT_LIMIT`, `TOOL_RESULT_HARD_LIMIT`, `TOOL_RESULT_PREVIEW`
  - `MIG_DEFAULT_LAMBDA`, `MIG_MIN_CORPUS_TOKENS`
  - `QUALITY_ENTITY_WEIGHT`, `QUALITY_COVERAGE_WEIGHT`, `QUALITY_RELEVANCE_WEIGHT`

**Fail Condition:**
- Any `os.getenv`/`os.environ` key found in `src/**/*.py` that is not present in `.env.example`.

**Evidence Requirements:**
1. Audit script output (e.g., `grep -rhoP "os\.(getenv|environ).*?['\"]([A-Z_]+)['\"]" src/ | sort -u`) compared against `.env.example` entries shows zero delta.
2. PR diff shows all added entries with documented defaults and descriptions.

---

### VAL-DOCKER-003 — Dockerfile Multi-Stage Build Produces Working Image

**Title:** Docker image builds, starts, and serves MCP requests end-to-end

**Behavioral Description:**
The multi-stage Dockerfile builds without errors on `python:3.12-slim`, the resulting image starts the MCP stdio server, and an end-to-end compression round-trip succeeds.

**Pass Condition:**
- `docker build -t gotcontext:test .` exits 0.
- `echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"ingest_context","arguments":{"text":"Hello world test document for validation.","doc_id":"val-test"}}}' | docker run -i gotcontext:test` returns a valid JSON-RPC response with `compressed_text`.
- Image size < 600 MB.

**Fail Condition:**
- Build fails, runtime import errors, or compression returns an error response.

**Evidence Requirements:**
1. CI pipeline log showing successful build + smoke test.
2. `docker images gotcontext:test --format "{{.Size}}"` output.

---

### VAL-DOCKER-004 — Non-Root User Enforcement

**Title:** Container process runs as non-root (`mcp` user, uid 1000)

**Behavioral Description:**
The running container's primary process must execute as user `mcp` (uid 1000), not `root`. All writable paths (`/data`, `/home/mcp/.cache`) must be owned by this user.

**Pass Condition:**
- `docker run --rm gotcontext:test whoami` outputs `mcp`.
- `docker run --rm gotcontext:test id` shows `uid=1000(mcp)`.
- No runtime `PermissionError` when writing to `/data` or reading cached models.

**Fail Condition:**
- Process runs as root, or file permission errors occur at runtime.

**Evidence Requirements:**
1. `docker run --rm gotcontext:test id` output.
2. `docker run --rm gotcontext:test ls -la /home/mcp/.cache/` showing `mcp` ownership.

---

## Area 2: Thread Safety & Code Quality

### VAL-QUALITY-001 — SemanticCompressor Concurrent Compression Safety

**Title:** Concurrent compressions do not corrupt shared state

**Behavioral Description:**
When multiple compression requests arrive simultaneously (simulating concurrent MCP tool calls), `SemanticCompressor`'s internal state (graph, chunks, embeddings) must not be corrupted. The existing `_sync_lock` (`threading.Lock`) and `_async_lock` (`asyncio.Lock`) in `semantic_compressor.py` (lines 180–182) must correctly serialize access to mutable state.

**Pass Condition:**
- Launch 10 concurrent `ingest_context` requests (via `asyncio.gather` or parallel `curl` to HTTP endpoint) with distinct `doc_id` values and 500+ token documents.
- All 10 return valid compressed output with no exceptions.
- No two responses contain mixed/interleaved content from other documents.
- Each response's `doc_id` matches its request.

**Fail Condition:**
- Any request raises an unhandled exception (`RuntimeError`, `KeyError`, data corruption).
- Response content from document A appears in document B's result.
- Deadlock (request hangs beyond 120s timeout).

**Evidence Requirements:**
1. Test script that fires 10 concurrent requests and asserts per-doc correctness.
2. Test passes in CI (pytest marker: `@pytest.mark.integration`).
3. No `threading` or `asyncio` warnings in logs during concurrent run.

---

### VAL-QUALITY-002 — asyncio.get_event_loop() Deprecation Eliminated

**Title:** No calls to deprecated `asyncio.get_event_loop()` remain in source

**Behavioral Description:**
Python 3.10+ deprecates `asyncio.get_event_loop()` when no running loop exists. All occurrences must be replaced with `asyncio.get_running_loop()` (inside async context) or `asyncio.new_event_loop()` (when creating a loop explicitly).

**Pass Condition:**
- `rg "asyncio\.get_event_loop\(\)" src/` returns zero matches.
- No `DeprecationWarning` about event loops appears in `pytest -W error::DeprecationWarning` output.

**Fail Condition:**
- Any remaining `asyncio.get_event_loop()` call in `src/`.
- DeprecationWarning raised during test suite execution.

**Evidence Requirements:**
1. `rg` output showing zero matches.
2. `pytest tests/ -W error::DeprecationWarning --no-header -q` exits 0 (or only unrelated warnings).

---

### VAL-QUALITY-003 — ESLint Zero Errors

**Title:** Frontend codebase passes ESLint with zero errors

**Behavioral Description:**
Running `npx eslint .` (or the project's configured lint command) in the Next.js frontend directory produces zero errors. Warnings are acceptable but errors must be zero.

**Pass Condition:**
- `npx eslint . --format json` in the frontend directory reports `"errorCount": 0` in every file entry.
- CI lint step exits with code 0.

**Fail Condition:**
- Any file reports `errorCount > 0`.
- CI lint step exits non-zero.

**Evidence Requirements:**
1. ESLint JSON output showing `errorCount: 0` aggregate.
2. CI pipeline log for the lint step.

---

### VAL-QUALITY-004 — No Uncommitted Changes in Repository

**Title:** Working tree is clean after all fixes are applied

**Behavioral Description:**
After all bug fixes and feature additions are committed, `git status --porcelain` on the deployment branch shows no untracked, modified, or staged files (excluding `.gitignore`d paths like `benchmarks/_saas_results/`).

**Pass Condition:**
- `git status --porcelain` returns empty output (or only intentionally-ignored paths).
- `git diff HEAD` produces no output.

**Fail Condition:**
- Any tracked file shows as modified, deleted, or untracked.

**Evidence Requirements:**
1. `git status --porcelain` output in CI log.
2. `git stash list` is empty (no accidentally stashed work).

---

### VAL-QUALITY-005 — Per-Document Lock Isolation

**Title:** Per-document async locks prevent cross-document interference

**Behavioral Description:**
The `_doc_locks: Dict[str, asyncio.Lock]` in `SemanticCompressor` (line 182) ensures that operations on the same `doc_id` are serialized while operations on different `doc_id` values proceed concurrently. This prevents corruption when the same document is ingested twice simultaneously.

**Pass Condition:**
- Two concurrent `ingest_context` calls with the **same** `doc_id` and different content: the second call waits for the first to finish (verified by timing — total time ≈ 2× single call, not 1×).
- Two concurrent calls with **different** `doc_id` values: total time ≈ 1× single call (parallel execution).

**Fail Condition:**
- Same-doc concurrent calls produce corrupted output.
- Different-doc calls are serialized unnecessarily (performance regression).

**Evidence Requirements:**
1. Timing-based integration test demonstrating serialization for same-doc and parallelism for different-doc.

---

## Area 3: Frontend Features

### VAL-UI-001 — Usage Historical Graph Renders Real Data

**Title:** Dashboard usage graph displays data from `/v1/usage/history` endpoint

**Behavioral Description:**
When a logged-in user navigates to the usage/dashboard page, an interactive chart (built with `recharts`) renders showing historical token usage data. The chart fetches data from the `/v1/usage/history` API endpoint and displays it as a time-series line or bar chart.

**Pass Condition:**
- Navigate to `/dashboard` (or `/usage`) while authenticated → chart component is visible in the DOM.
- Browser DevTools Network tab shows a successful `GET /v1/usage/history` request returning JSON with `data` array containing `{date, tokens_saved, requests}` entries.
- Chart renders at least one data point when usage history exists.
- Chart shows an empty state message (not a crash) when no history exists.
- Hovering over data points shows a tooltip with exact values.

**Fail Condition:**
- Chart does not render, or shows a loading spinner indefinitely.
- API returns 500 or chart shows hardcoded/mock data in production.
- Console shows React errors or recharts import failures.

**Evidence Requirements:**
1. Screenshot of the rendered chart with real data points.
2. Network capture showing `/v1/usage/history` response payload.
3. Screenshot of empty state when no data exists.
4. `recharts` listed in `package.json` dependencies.

---

### VAL-UI-002 — OG Image Social Sharing Preview

**Title:** Social share preview renders correct OG image and metadata

**Behavioral Description:**
When a link to `gotcontext.ai` is shared on social platforms (Twitter/X, LinkedIn, Slack), the preview card displays a branded OG image with correct title, description, and image. The `<meta>` OG tags are present in the page's `<head>`.

**Pass Condition:**
- `curl -s https://gotcontext.ai | grep 'og:image'` returns a valid `<meta property="og:image" content="https://gotcontext.ai/og-image.png" />` tag (or equivalent).
- The OG image URL returns HTTP 200 with `Content-Type: image/png` (or `image/jpeg`).
- Image dimensions are ≥ 1200×630px (recommended OG dimensions).
- `og:title`, `og:description`, and `og:url` tags are also present and non-empty.
- Validating with [opengraph.xyz](https://opengraph.xyz) or Twitter Card Validator shows the preview correctly.

**Fail Condition:**
- OG image tag is missing, points to a 404, or shows a generic/placeholder image.
- OG metadata references "nextjs-boilerplate" or default Next.js starter text.

**Evidence Requirements:**
1. `curl -s <url> | grep -i 'og:'` output showing all 4 OG tags.
2. Screenshot from OG validator tool showing rendered preview card.
3. Direct HTTP request to the OG image URL returning 200.

---

### VAL-UI-003 — Fidelity Profiles CRUD from Dashboard

**Title:** Users can create, list, and delete compression fidelity profiles via the UI

**Behavioral Description:**
The dashboard provides a Fidelity Profiles management page where users can:
1. **Create** a new profile with a name, skeleton_ratio, fidelity level, and chunk_size.
2. **List** all existing profiles in a table/grid.
3. **Delete** a profile with a confirmation prompt.

Profiles persist across page reloads (stored via API, not just local state).

**Pass Condition:**
- Navigate to Fidelity Profiles page → click "Create" → fill form → submit → new profile appears in list.
- Reload page → profile is still listed (fetched from backend API).
- Click "Delete" on a profile → confirmation dialog appears → confirm → profile removed from list.
- API calls observed: `POST /v1/profiles`, `GET /v1/profiles`, `DELETE /v1/profiles/{id}`.

**Fail Condition:**
- Create form submits but profile doesn't appear.
- Profile disappears on page reload (client-only state).
- Delete has no confirmation (accidental deletion risk).
- API returns 404/500 for any CRUD operation.

**Evidence Requirements:**
1. Screenshots of: empty state, create form, populated list, delete confirmation.
2. Network tab showing successful API round-trips for each CRUD operation.
3. Page reload demonstrating persistence.

---

### VAL-UI-004 — Sentry DSN Configured for Correct Project

**Title:** Frontend Sentry errors route to the gotcontext project (not nextjs-boilerplate)

**Behavioral Description:**
The Sentry SDK initialization in the Next.js app must use a DSN that routes to the correct Sentry project for gotcontext.ai. Errors must not be sent to a boilerplate/template project.

**Pass Condition:**
- `Sentry.init({ dsn: "..." })` in the frontend code uses a DSN containing the gotcontext project slug (not `nextjs-boilerplate` or a template DSN).
- Triggering a deliberate test error (e.g., `/api/debug-sentry`) results in an event appearing in the correct Sentry project dashboard.
- `NEXT_PUBLIC_SENTRY_DSN` or equivalent env var is set in deployment config.

**Fail Condition:**
- DSN contains `nextjs-boilerplate` in the project slug.
- Sentry DSN is empty/undefined in production build.
- Test error does not appear in any Sentry project.

**Evidence Requirements:**
1. Grep of Sentry config showing the DSN project slug.
2. Screenshot of test error event in correct Sentry project dashboard.
3. Deployment env config showing `SENTRY_DSN` is set.

---

### VAL-UI-005 — Usage Graph Interactivity

**Title:** Chart supports hover tooltips, axis labels, and responsive resizing

**Behavioral Description:**
The recharts-based usage graph must be interactive: hovering shows data tooltips, X-axis shows date labels, Y-axis shows token counts, and the chart resizes responsively on different viewport widths.

**Pass Condition:**
- Hover over a bar/point → tooltip shows date + token count.
- X-axis displays readable date labels (not raw timestamps).
- Y-axis displays formatted numbers (e.g., "12.5K" not "12500").
- Resize browser to mobile width (375px) → chart reflows without horizontal scroll or clipping.

**Fail Condition:**
- No tooltips on hover.
- Axes show raw/unformatted data.
- Chart overflows or clips at mobile viewport widths.

**Evidence Requirements:**
1. Screenshot at desktop width with tooltip visible.
2. Screenshot at 375px mobile width showing responsive layout.

---

## Area 4: Degradation & Resilience

### VAL-DEGRADE-001 — Redis-Down: API Continues Serving Requests

**Title:** API responds to compression requests when Redis is unavailable

**Behavioral Description:**
When Redis is unreachable (connection refused, timeout), the API must continue processing compression requests by falling back to Postgres for state that normally lives in Redis. Usage tracking that would normally go to Redis is queued in-memory or written directly to Postgres. The user experiences slightly higher latency but no errors.

**Pass Condition:**
- Stop Redis (`docker stop redis`) → send `POST /v1/compress` with valid payload → receive 200 response with compressed output.
- Response includes a header or field indicating degraded mode (e.g., `X-Degraded: redis`), or structured log emits `redis_fallback` event.
- Usage data is not lost: after Redis recovers, queued usage records are eventually flushed.
- Latency increase < 2× compared to Redis-up baseline.

**Fail Condition:**
- API returns 500/503 when Redis is down.
- Usage tracking data is permanently lost during outage.
- API hangs waiting for Redis connection (no timeout).

**Evidence Requirements:**
1. Test script: stop Redis → send 5 requests → assert all return 200 → start Redis → verify usage data appears.
2. Structured logs showing fallback activation.
3. Latency comparison (with/without Redis).

---

### VAL-DEGRADE-002 — Supabase-Down: Stateless Compression Works

**Title:** Core compression works without Supabase; auth falls back to Clerk JWT verification

**Behavioral Description:**
When Supabase is unreachable, the compression pipeline (which is stateless — text in, compressed text out) must continue functioning. Authentication falls back to verifying Clerk JWTs directly (without Supabase session lookup). Non-critical features that depend on Supabase (profile storage, usage persistence) degrade gracefully with clear error messages.

**Pass Condition:**
- Block Supabase DNS/IP → send authenticated `POST /v1/compress` with valid Clerk JWT → receive 200 with compressed output.
- Auth middleware successfully verifies the Clerk JWT without Supabase.
- Profile/usage endpoints return 503 with `{"error": "service_degraded", "detail": "Database temporarily unavailable"}` (not a stack trace).
- Logs emit `supabase_unreachable` warning with retry schedule.

**Fail Condition:**
- Compression endpoint returns 401 or 500 when Supabase is down.
- Auth completely fails (user cannot access any endpoint).
- Error responses leak internal stack traces or connection strings.

**Evidence Requirements:**
1. Integration test simulating Supabase outage (mock DNS failure or connection timeout).
2. Successful compression response during outage.
3. Graceful 503 responses for database-dependent endpoints.

---

### VAL-DEGRADE-003 — Polar-Down: Existing Subscriptions Cached

**Title:** Billing provider (Polar) outage does not block API access for subscribed users

**Behavioral Description:**
When the Polar billing API is unreachable, existing subscription status must be served from cache (Redis or Postgres). Users with active subscriptions continue to have full access. New subscription purchases are queued or deferred with a user-facing message. Requests must not be blocked waiting for Polar API responses.

**Pass Condition:**
- Block Polar API → authenticated user with active subscription sends `POST /v1/compress` → receives 200 (subscription validated from cache).
- Subscription check completes in < 100ms (cache hit, no network call to Polar).
- New subscription attempt returns a friendly error: `{"error": "billing_unavailable", "message": "Billing system is temporarily unavailable. Please try again later."}`.
- After Polar recovers, cached subscription data is refreshed on next check.

**Fail Condition:**
- Subscribed user gets 402/403 during Polar outage.
- API hangs waiting for Polar response (no circuit breaker timeout).
- Subscription cache has no TTL (stale data served indefinitely).

**Evidence Requirements:**
1. Test: pre-populate subscription cache → block Polar → verify API access.
2. Cache TTL configuration documented (e.g., 24h default).
3. Logs showing `polar_circuit_open` event with automatic recovery.

---

### VAL-DEGRADE-004 — Circuit Breaker Pattern Implementation

**Title:** Circuit breakers protect against cascading failures for all external dependencies

**Behavioral Description:**
The `CircuitBreaker` class in `src/reliability.py` implements the standard three-state pattern (CLOSED → OPEN → HALF_OPEN → CLOSED). Each external dependency (Redis, Supabase, Polar, embedding model downloads) must be wrapped with a circuit breaker instance configured with appropriate thresholds.

**Pass Condition:**
- Circuit breaker transitions: CLOSED → (N failures) → OPEN → (timeout) → HALF_OPEN → (success) → CLOSED.
- `failure_threshold` is configurable per dependency (default: 5).
- `timeout` (recovery window) is configurable (default: 60s).
- OPEN state immediately rejects calls with `CircuitBreakerOpenError` (no network attempt).
- HALF_OPEN allows exactly `half_open_max_calls` probe requests.
- Existing tests in `test_chaos_engineering.py` pass: `test_circuit_breaker_closed_state`, `test_circuit_breaker_opens_after_threshold`, `test_circuit_breaker_rejects_when_open`, `test_circuit_breaker_half_open_after_timeout`, `test_circuit_breaker_reopens_on_half_open_failure`, `test_circuit_breaker_reset`.

**Fail Condition:**
- Any circuit breaker test fails.
- External calls are made when circuit is OPEN (wastes resources, increases latency).
- No circuit breaker configured for a critical external dependency.

**Evidence Requirements:**
1. `pytest tests/test_chaos_engineering.py -v -k circuit_breaker` passes all 6 tests.
2. Code review showing each external dependency wrapped with `CircuitBreaker`.
3. Configuration values documented in `.env.example` or `constants.py`.

---

### VAL-DEGRADE-005 — Embedding Tier Fallback Chain

**Title:** Embedding failures cascade through SBERT → ONNX → TF-IDF without data loss

**Behavioral Description:**
The `GracefulDegradation.embed_with_fallback()` in `src/graceful_degradation.py` implements a 3-tier fallback: STANDARD (SBERT/PyTorch) → ONNX → TFIDF. If the preferred tier fails, the next tier is attempted automatically. The user receives embeddings (potentially lower quality) rather than an error.

**Pass Condition:**
- Mock STANDARD tier failure → ONNX tier is attempted → embeddings returned with `embedding_degraded` log warning.
- Mock both STANDARD and ONNX failure → TFIDF tier succeeds → embeddings returned.
- All three tiers fail → exception is raised (not silently swallowed).
- Each fallback transition is logged with `embedding_tier_failed` structured event.

**Fail Condition:**
- Failure in STANDARD tier causes immediate error without trying ONNX/TFIDF.
- Silent fallback with no logging (ops cannot detect degradation).
- All-tiers-failed scenario is swallowed silently.

**Evidence Requirements:**
1. `pytest tests/test_chaos_engineering.py -v -k graceful_degradation` passes.
2. Log output showing tier transition warnings.

---

### VAL-DEGRADE-006 — User-Visible Degradation Indicators

**Title:** API responses indicate when operating in degraded mode

**Behavioral Description:**
When any component is in degraded/fallback mode, API responses must include a machine-readable indicator so that clients (and monitoring) can detect the degradation. This could be a response header (`X-Degraded-Components: redis,polar`), a response body field (`"warnings": [...]`), or both.

**Pass Condition:**
- During Redis outage: response includes degradation indicator mentioning Redis.
- During embedding fallback: response includes indicator mentioning the actual tier used.
- Healthy state: no degradation indicator present (no false positives).
- Monitoring/alerting can key off the indicator (structured format, not free text).

**Fail Condition:**
- Degraded operation produces identical responses to healthy operation (undetectable).
- Degradation indicator is present during normal operation (false positive).

**Evidence Requirements:**
1. API response samples in degraded vs healthy mode showing the indicator.
2. Monitoring query/alert rule that keys off the indicator.

---

*End of Validation Contract — Areas 1–4*
