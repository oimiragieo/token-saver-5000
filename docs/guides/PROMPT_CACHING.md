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

## Checklist

- [ ] Tool definitions are at the very start of your prompt
- [ ] System instructions come before any dynamic content
- [ ] Token Saver 5000 compressed output is placed before chat history
- [ ] No timestamps, UUIDs, or request IDs appear before your RAG context
- [ ] User query is at the absolute end
- [ ] You are monitoring provider cache hit metrics
- [ ] Your framework is not injecting hidden dynamic IDs
