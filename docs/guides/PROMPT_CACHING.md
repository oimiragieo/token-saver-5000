# Prompt Caching Best Practices

This guide explains how to structure prompts when using Token Saver 5000's compressed output to maximize LLM prompt cache hit rates, potentially cutting input costs by up to 90%.

## How Prompt Caching Works

LLM providers (Claude, OpenAI, Gemini) cache the processed Key-Value (KV) states from previous queries. Caching requires **exact, byte-level prefix matching** — even a single changed token early in your prompt invalidates the cache for everything that follows.

## The Stability Hierarchy

Order your prompt elements from most stable to most volatile:

```
1. Tool Definitions         ← Most stable (locked at start)
2. System Instructions      ← Core persona, rarely changes
3. Compressed Context (RAG) ← Token Saver 5000 output goes here
4. Few-Shot Examples         ← Static output formats
5. Chat History              ← Previous conversation turns
6. Dynamic Metadata          ← Timestamps, UUIDs, session info
7. User Query                ← Most volatile (always changes)
```

### Why This Order Matters

If your user query comes *before* your RAG context, every new query invalidates the cache for the entire RAG payload — the most expensive part of your prompt. By placing stable content first, you create a large cacheable prefix that persists across requests.

## Using Token Saver 5000 Output for Cache Efficiency

### Skeleton Output

The `read_skeleton` tool produces compressed document views. These are designed to be cache-friendly:

- **Static node content** (ANCHOR/HIDDEN lines) appears first
- **Query metadata** (when using query-guided selection) appears at the end

This means the node structure forms a stable prefix that caches well across different queries on the same document.

### ACE Playbook Output

ACE tool responses (`ace_get_playbook`, `ace_generate`, etc.) exclude volatile fields:

- No UUIDs (`bullet_id`, `context_id`)
- No timestamps (`created_at`, `updated_at`)
- No ephemeral tracking data in `delta_history`

This ensures that playbook content remains cache-stable across calls.

### AFM Context

The `afm_build_context` tool returns messages ordered as:

1. System preamble (static) — cacheable prefix
2. Historical messages (semi-stable) — partially cacheable
3. Current turn context (volatile) — varies per request

Place the system preamble content in your system message, and the rest as assistant/user turn content near the end of your prompt.

## Framework Pitfalls

### The Silent Cache Killer

Orchestration frameworks like LangChain or LlamaIndex may inject dynamic unique IDs (UUIDs) into message headers. Even if your prompt text looks identical, a hidden shifting ID guarantees a cache miss every time.

**Token Saver 5000 avoids this** — it uses the MCP protocol directly with no hidden dynamic injection.

### Detecting Cache Misses

Providers don't throw errors for cache misses — your bills silently inflate. Monitor these API response fields:

| Provider | Field to Monitor | Healthy Value |
|----------|-----------------|---------------|
| **Claude** | `cache_read_input_tokens` | > 0 for repeated prefixes |
| **OpenAI** | `usage.prompt_tokens_details.cached_tokens` | > 0 for repeated prefixes |
| **Gemini** | `cachedContentTokenCount` | > 0 for repeated prefixes |

If these fields are consistently zero when you expect cache hits, something is invalidating your prefix.

### Quick Diagnostic

```python
# Claude (Anthropic SDK)
response = client.messages.create(...)
cached = response.usage.cache_read_input_tokens
if cached == 0:
    print("WARNING: Prompt cache miss — check for dynamic prefix content")

# OpenAI
response = client.chat.completions.create(...)
cached = response.usage.prompt_tokens_details.cached_tokens
if cached == 0:
    print("WARNING: Prompt cache miss — check for dynamic prefix content")
```

## Monitoring With Token Saver 5000

The repo now includes prompt-cache observability tools so you can validate real provider behavior instead of inferring it from cost drift.

If you are integrating through Gemini CLI, Claude Code, or Codex, also read `docs/guides/PROVIDER_CACHE_COMPATIBILITY.md` for provider-versus-harness guidance.

Recommended workflow:

1. Use `render_prompt_template` to generate a cache-friendly prompt and capture a `prompt_id`.
2. Send the rendered prompt to your provider.
3. Pass the raw provider response into `capture_cache_telemetry`.
4. If reuse underperforms or misses unexpectedly, call `diagnose_cache_miss` with the exact `actual_rendered_prefix`.

Key outputs to watch:

- `telemetry.cache_hit_detected`: whether the provider reported any cache reuse
- `telemetry.validation.prefix_integrity`: whether the actual stable prefix changed byte-for-byte
- `telemetry.validation.diagnostic`: likely miss cause, including section interleaving and semantic-equivalence drift
- `telemetry.validation.cache_creation_churn`: repeated cache creation on the same stable prefix
- `telemetry.session_metrics`: multi-turn cache hit ratio and cached-token totals across a workflow session
- `assess_cache_compatibility`: whether your Gemini CLI / Claude Code / Codex surface exposes enough telemetry to trust automated cache monitoring
- `optimize_for_model.cache_thresholds`: whether your reusable prefix is large enough to qualify for provider-side cache accounting

### Example: End-to-End Cache Validation

```json
{
  "tool": "capture_cache_telemetry",
  "args": {
    "model": "gpt-5.4",
    "prompt_id": "prompt-cache-abc123",
    "session_id": "review-session-42",
    "actual_rendered_prefix": "[system_instructions]\nBe accurate.\n[rag_context]\n...",
    "api_response": {
      "usage": {
        "prompt_tokens": 500,
        "completion_tokens": 100,
        "prompt_tokens_details": {
          "cached_tokens": 300
        }
      }
    }
  }
}
```

If `cached_tokens` is unexpectedly low or zero, inspect:

- `validation.warning`
- `validation.diagnostic.probable_cause`
- `validation.diagnostic.partial_reuse`
- `validation.cache_creation_churn`

### Provider Telemetry Notes

- **Claude / Anthropic**: monitor both `cache_read_input_tokens` and `cache_creation_input_tokens`
- **OpenAI**: monitor `usage.prompt_tokens_details.cached_tokens`
- **OpenAI / Codex**: when supported by your integration, use a stable `prompt_cache_key` to improve routing stickiness for repeated workflows
- **Gemini**: monitor `cachedContentTokenCount`; SDK-style payloads may expose the same data as `usage_metadata.cached_content_token_count`
- **Gemini CLI**: exported stats may use camelCase counters like `inputTokens`, `outputTokens`, and `cachedTokens`

## Additional Token-Saving Techniques Now Reflected In The Repo

- **Extractive compression baseline**: prefer sentence selection over abstractive summarization when you need low-latency, high-fidelity trimming.
- **Segment-level compression caching**: cache repeated chunk-compression results for shared or recurring context blocks.
- **History compaction**: summarize older turns into a stable prefix while keeping the most recent turns verbatim at the tail.

Repeated positive `cache_creation_input_tokens` with a supposedly stable prefix is a warning sign: the provider is rebuilding cache state instead of reusing it.

## Checklist

- [ ] Tool definitions are at the very start of your prompt
- [ ] System instructions come before any dynamic content
- [ ] Token Saver 5000 compressed output is placed before chat history
- [ ] No timestamps, UUIDs, or request IDs appear before your RAG context
- [ ] User query is at the absolute end
- [ ] You are monitoring provider cache hit metrics
- [ ] You are monitoring provider cache creation metrics for churn, not just hit/miss
- [ ] Your framework is not injecting hidden dynamic IDs
