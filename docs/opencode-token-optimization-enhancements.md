# OpenCode Token Optimization Enhancements for Token Saver 5000

**Date:** 2026-04-01
**Source:** OpenCode CLI at `opencode-main/` (Go, 120K+ GitHub stars)
**Scope:** Optimize Token Saver for OpenCode's multi-provider architecture

---

## Executive Summary

OpenCode is the biggest opportunity for gotcontext.ai because it supports **12+ providers
and 50+ models**. Every model has different caching behavior, context limits, and pricing.
Our tools need to handle all of them.

## OpenCode vs Other CLIs

| Aspect | Claude Code | Codex | Gemini CLI | **OpenCode** |
|--------|------------|-------|------------|-------------|
| Language | TypeScript | Rust | TypeScript | **Go** |
| Models | Claude only | GPT/O-series only | Gemini only | **12+ providers, 50+ models** |
| Context windows | 200K / 1M | 200K | 1M | **128K to 1M+ (model-dependent)** |
| Prompt caching | Ephemeral (server) | prefix_cache_key | Explicit + Implicit | **Provider-dependent** |
| Compression trigger | ~93% of window | 80% (soft cap) | 50% of window | **95% of window** |
| Tool result limit | 50K chars | config-based | 40K tokens | **30K chars (bash)** |
| MCP support | Native | Native | Native | **Native (stdio + SSE)** |
| Stars | N/A (closed) | N/A (closed) | N/A (closed) | **120K+** |
| Users | Enterprise | Enterprise | Enterprise | **5M+/month** |

## Key Findings from Source Analysis

### 1. Multi-Provider Token Tracking

OpenCode tracks `TokenUsage` per request:
```go
type TokenUsage struct {
    InputTokens         int64
    OutputTokens        int64
    CacheCreationTokens int64  // Anthropic-specific
    CacheReadTokens     int64  // Anthropic-specific
}
```

**Only Anthropic gets explicit cache token tracking.** OpenAI, Gemini, and Groq models
have automatic/implicit caching that's invisible to the client. Our savings tracker
must handle both explicit and implicit cache awareness.

### 2. Provider-Specific Caching Strategies

| Provider | Cache Type | Min Prefix | Discount | TTL | Client Action |
|----------|-----------|------------|----------|-----|---------------|
| **Anthropic** | Explicit ephemeral | 1024 tokens | 90% read, +25% write | 5min (1h for subscribers) | `cache_control: ephemeral` on blocks |
| **OpenAI** | Automatic | 1024 tokens | 50% read | 5-60min | None (automatic) |
| **Gemini 2.5+** | Implicit (auto) | Varies | 90% read | Automatic | None |
| **Gemini 2.0** | Explicit only | 4096+ tokens | 75% read | 1hr default | `cachedContent.create()` API |
| **Groq** | Prompt caching | Varies | Varies | Unknown | None (automatic for some models) |
| **Bedrock** | Varies by model | Varies | Varies | Varies | Provider-dependent |
| **Local/Ollama** | None | N/A | N/A | N/A | N/A |
| **OpenRouter** | Passthrough | Depends on underlying | Depends | Depends | Depends on route |

### 3. Auto-Compact at 95% Context Window

OpenCode triggers summarization at 95% of context window (vs Claude's 93%, Gemini's 50%,
Codex's 80%). This means:
- OpenCode users have the MOST context available before compression fires
- But when it fires, it's more aggressive (full LLM summarization)
- Our compression should target the 0-95% range to prevent auto-compact from triggering

### 4. Tool Result Limits

- Bash output: **30,000 characters** (hard truncation)
- File reads: paginated via offset/limit (no hard limit per chunk)
- MCP tool results: no explicit limit (full content passed through)
- Our proxy interceptor should target 30K as the effective limit for OpenCode

### 5. Model Cost Ranges

OpenCode's model definitions include exact pricing:

| Model | Input $/MTok | Output $/MTok | Cache Read $/MTok | Context |
|-------|-------------|--------------|------------------|---------|
| Claude 4 Opus | $15.00 | $75.00 | $1.50 | 200K |
| Claude 4 Sonnet | $3.00 | $15.00 | $0.30 | 200K |
| GPT-4.1 | $2.00 | $8.00 | N/A (auto) | 1M |
| GPT-4.1 mini | $0.40 | $1.60 | N/A (auto) | 200K |
| O3 | $2.00 | $8.00 | N/A | 200K |
| O4-mini | $1.10 | $4.40 | N/A | 200K |
| Gemini 2.5 Flash | $0.15 | $0.60 | $0.02 (implicit) | 1M |
| Gemini 2.5 Pro | $1.25 | $10.00 | $0.31 (implicit) | 1M |
| Groq Llama 4 Scout | $0.11 | $0.34 | N/A | 512K |
| DeepSeek R1 (Groq) | $0.75 | $0.99 | N/A | 128K |
| Grok 3 (XAI) | $3.00 | $15.00 | N/A | 131K |
| Local (Ollama) | $0.00 | $0.00 | N/A | Varies |

### 6. Session Architecture

OpenCode uses SQLite for session persistence with parent/child session linking for
summarization. Our `SessionJournal` maps naturally to this -- we should use the same
`session_id` scoping that OpenCode provides via MCP tool calls.

---

## Enhancements for Token Saver

### Enhancement 1: Expanded Model Database

Add ALL OpenCode models to our `KNOWN_MODEL_CONTEXT_WINDOWS` and
`KNOWN_MODEL_COMPRESSION_TRIGGERS`:

```python
# OpenCode models to add
"claude-4-opus": 200_000,
"claude-4-sonnet": 200_000,
"claude-4.5-sonnet": 200_000,
"claude-3.7-sonnet": 200_000,
"gpt-4.1": 1_047_576,
"gpt-4.1-mini": 200_000,
"gpt-4.1-nano": 200_000,
"o1-pro": 200_000,
"gemini-2.0-flash": 1_000_000,
"groq-llama-4-scout": 512_000,
"groq-llama-4-maverick": 512_000,
"groq-deepseek-r1": 128_000,
"groq-qwq": 128_000,
"grok-3": 131_072,
"grok-3-mini": 131_072,

# Compression triggers
"opencode-*": 0.95,  # all OpenCode models trigger at 95%
```

### Enhancement 2: Provider-Aware Cache Strategy Advisor

New MCP tool: `advise_cache_strategy` that returns the optimal caching approach for the
configured model:

- Anthropic: "Use ephemeral cache markers. Stable prefix first. Last 2 messages cached."
- OpenAI: "Automatic caching. Keep 1024+ token prefix stable. No client action needed."
- Gemini 2.5: "Implicit caching automatic. Explicit cache for >4096 token static content."
- Groq: "Limited caching support. Focus on small prompts, fast inference."
- Local: "No caching. Minimize prompt size for fastest inference."

### Enhancement 3: OpenCode Benchmark Provider

Add OpenCode to our benchmark harness as a fourth provider. OpenCode runs headless via:
```bash
opencode -p "prompt" -f json
```

The JSON output includes session data that we can parse for token counts.

---

## Research Sources

- [OpenAI Prompt Caching](https://developers.openai.com/api/docs/guides/prompt-caching) - Automatic, 1024 token min, 50% discount
- [Gemini Context Caching](https://ai.google.dev/gemini-api/docs/caching) - Explicit + Implicit, 75-90% discount
- [Gemini 2.5 Implicit Caching](https://developers.googleblog.com/en/gemini-2-5-models-now-support-implicit-caching/)
- [OpenCode GitHub](https://github.com/opencode-ai/opencode) - 120K+ stars, 5M+ users
- [OpenCode Docs](https://opencode.ai/docs/)
