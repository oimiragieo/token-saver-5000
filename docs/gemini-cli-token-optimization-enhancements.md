# Gemini CLI Token Optimization Enhancements for Token Saver 5000

**Date:** 2026-03-31
**Version:** v0.11.0 (additive to Claude Code enhancements)
**Author:** Auto-generated from Gemini CLI source analysis
**Scope:** Enhancements to Token Saver 5000 that optimize token usage when used with
Gemini CLI, while remaining model-agnostic

---

## Executive Summary

Analysis of Gemini CLI's source (`gemini-cli-main/packages/core/src/`) reveals a
fundamentally different token management philosophy compared to Claude Code. Key differences:

| Aspect | Claude Code | Gemini CLI |
|--------|------------|------------|
| **Token counting** | `len/4` bytes (fast) or tiktoken (accurate) | ASCII: 0.25/char, non-ASCII: 1.3/char, long text: len/4 |
| **Tool result limit** | 50K chars hard cap, disk persistence | 40K tokens soft cap with proportional truncation |
| **Truncation strategy** | Replace with 2KB preview + disk ref | Keep 20% head + tail with ellipsis |
| **Context compression** | Microcompact (server-side clearing) + full summarization | 3-phase: truncation -> LLM summarization -> verification |
| **Compression trigger** | ~93% of window (200K - 13K buffer) | 50% of token limit (configurable) |
| **Tool output masking** | N/A (clears old results entirely) | Backward-scan masking, protects recent 50K tokens |
| **Context window** | 200K or 1M tokens | 1,048,576 tokens (1M default for all models) |
| **Cache mechanism** | Prompt prefix caching (server-side) | Explicit context caching (client-side, API-managed) |

### Key Findings from Gemini CLI Source

1. **Proportional truncation**: Gemini CLI truncates large tool outputs to 20% head +
   80% tail with an ellipsis marker. This preserves both the beginning (often headers/
   structure) and end (often the most recent/relevant output). Our response formatter
   should support this strategy alongside pagination.

2. **Token estimation divergence**: Gemini CLI uses a character-class-aware heuristic
   (0.25 tok/ASCII char, 1.3 tok/non-ASCII char) vs Claude Code's flat `len/4`. Our
   dual estimation module should add a Gemini-compatible estimation method.

3. **Aggressive early compression**: Gemini CLI triggers compression at 50% of the
   context window (vs Claude Code's ~93%). This means our compressed skeletons need to
   be even more efficient for Gemini -- smaller outputs are more valuable since the
   client is more aggressive about truncating.

4. **Tool output masking**: Gemini CLI replaces old tool outputs with a masking tag
   while protecting the most recent 50K tokens. Our tool responses should include
   structured headers/summaries at the top so the preserved "head" after truncation
   contains the most important information.

5. **Three-phase compression with verification**: Gemini CLI's compression pipeline
   includes an LLM verification pass that checks if critical info was lost. Our
   compressed skeletons are ideal input for this -- they're pre-structured with
   importance rankings, making the verification pass more effective.

6. **Explicit context caching**: Unlike Claude Code's automatic prompt prefix caching,
   Gemini API requires explicit cache creation via `cachedContents.create()`. Our tool
   schemas should be designed for manual caching -- stable, deterministic, and packaged
   as cacheable content blocks.

---

## Enhancement 1: Gemini-Compatible Token Estimation

### Problem
Our `TokenEstimator` provides tiktoken-accurate and `len/4` fast estimation, but
Gemini CLI uses a different formula: 0.25 tokens per ASCII char, 1.3 tokens per
non-ASCII char (with a fast fallback to `len/4` for text > 100K chars). When our
tool reports token counts, Gemini CLI may disagree significantly on non-ASCII content.

### Solution
Add `estimate_gemini(text: str) -> int` to `TokenEstimator` and include
`gemini_estimated` in the `estimate_all()` response dict.

Formula:
```python
def estimate_gemini(text: str) -> int:
    if len(text) > 100_000:
        return len(text) // 4
    tokens = 0.0
    for ch in text:
        tokens += 0.25 if ord(ch) <= 127 else 1.3
    return int(tokens)
```

### Why Model-Agnostic
This adds a new estimation method alongside existing ones. Clients pick the one
that matches their tokenizer. The method is useful for any system that uses
character-class-based estimation.

---

## Enhancement 2: Proportional Truncation Support in ResponseFormatter

### Problem
Our current `ResponseFormatter` paginates at the hard limit with a continuation
token. Gemini CLI expects a different pattern: proportional head/tail truncation
with an ellipsis, preserving JSON structure keys. When Gemini CLI receives our
paginated response, it may re-truncate it in a way that loses the continuation
instructions.

### Solution
Add a `truncation_strategy` parameter to `ResponseFormatter`:
- `"paginate"` (default, existing behavior): continuation token + preview
- `"proportional"`: 20% head + ellipsis + 80% tail (matches Gemini CLI's pattern)
- `"head"`: keep first N chars only (simple truncation)

When `truncation_strategy="proportional"`:
1. Serialize the response to JSON
2. If it exceeds the limit, apply proportional truncation to the serialized string
3. Prepend a `[Truncated by Token Saver]` prefix with token stats
4. Set `_truncated=True` but `_continuation_token=None` (no pagination)

### Why Model-Agnostic
Proportional truncation is useful for any client that wants to see both the
beginning and end of a response. The strategy parameter lets each client choose
the truncation behavior that matches its expectations.

---

## Enhancement 3: Structured Response Headers for Masking Resilience

### Problem
Gemini CLI's tool output masking service replaces old tool outputs with a masking
tag, but protects recent outputs. When it truncates, it uses proportional head/tail
preservation (20% head). If our response has important metadata at the end (as is
common in JSON), the most useful information may be in the truncated middle.

### Solution
Add a `response_header` to all tool responses -- a compact, human-readable summary
at the top of the response that survives any truncation strategy:

```json
{
    "_header": "Token Saver | ingest_context | doc=main.py | 485->61 tokens | ratio=7.9x",
    "_token_estimates": {...},
    "status": "success",
    ...
}
```

The `_header` field:
- Is always the first key in the JSON output (using OrderedDict or insertion order)
- Contains: tool name, key parameters, compression stats in a single line
- Is < 200 chars so it survives even aggressive truncation
- Uses `|` separators for easy parsing by LLMs

### Why Model-Agnostic
A compact header that summarizes the response is universally useful. Even if a
client doesn't truncate, the header provides a quick overview. It's especially
valuable for context-compressed conversations where old tool results may be
summarized.

---

## Enhancement 4: Compression Urgency Hints

### Problem
Gemini CLI triggers compression at 50% of the context window -- much earlier than
Claude Code's ~93%. When a Gemini CLI user ingests a large document, they may only
have ~500K tokens before compression fires. Our compression should be more aggressive
when the client signals urgency.

### Solution
Add an `urgency` parameter to compression tools (`ingest_context`, `read_skeleton`):
- `"normal"` (default): use profile-based or auto ratio
- `"compact"`: force skeleton_ratio to min(current_ratio, 0.15) for more aggressive compression
- `"emergency"`: force skeleton_ratio to 0.05 and fidelity to ABSTRACT

This can also be set via `configure_for_client` when model_id contains "gemini":
```python
if "gemini" in model_id.lower():
    # Gemini compresses early, default to more aggressive ratios
    recommended_ratio *= 0.75  # 25% more compression
```

### Why Model-Agnostic
The `urgency` parameter is useful for any client in a tight context budget. The
model-specific auto-tuning in `configure_for_client` is based on the model's known
compression trigger threshold, not any Gemini-specific feature.

---

## Enhancement 5: Gemini Model Database & Context Window Expansion

### Problem
Our `KNOWN_MODEL_CONTEXT_WINDOWS` only covers Claude, GPT, and basic Gemini models.
Gemini CLI supports multiple model variants with different capabilities, and all
Gemini 2.5+ models have a 1,048,576 token context window.

### Solution
Expand the model database with comprehensive Gemini model coverage and add a
`compression_trigger_ratio` field that indicates when the client typically starts
compressing:

```python
KNOWN_MODEL_PROFILES = {
    # Gemini models (all 1M context, compress at 50%)
    "gemini-2.5-pro": {"context_window": 1_048_576, "compression_trigger": 0.50},
    "gemini-2.5-flash": {"context_window": 1_048_576, "compression_trigger": 0.50},
    "gemini-3.1-pro": {"context_window": 1_048_576, "compression_trigger": 0.50},
    "gemini-3.1-flash": {"context_window": 1_048_576, "compression_trigger": 0.50},
    # Claude models (200K/1M context, compress at ~93%)
    "claude-opus-4-6": {"context_window": 200_000, "compression_trigger": 0.93},
    "claude-opus-4-6[1m]": {"context_window": 1_000_000, "compression_trigger": 0.93},
    # GPT models (no built-in compression)
    "gpt-4o": {"context_window": 128_000, "compression_trigger": 1.0},
}
```

The `compression_trigger` ratio is used by `get_recommended_ratio()` to tune the
skeleton ratio. Models that compress early (Gemini, 0.50) get more aggressive
default compression than models that compress late (Claude, 0.93).

### Why Model-Agnostic
The profile database covers all major providers. The `compression_trigger` field
is a universal concept -- it describes when the client starts managing context,
not any provider-specific behavior.

---

## TDD Implementation Plan

### Phase 1: Gemini Token Estimation (Enhancement 1)

#### Tests (`tests/test_token_estimation.py` -- extend existing)

```
test_gemini_estimate_ascii_only
test_gemini_estimate_non_ascii_only
test_gemini_estimate_mixed_content
test_gemini_estimate_long_text_fallback
test_gemini_estimate_empty_string
test_gemini_estimate_cjk_text
test_estimate_all_includes_gemini_key
```

#### Implementation
- Add `estimate_gemini(text: str) -> int` to `TokenEstimator`
- Add `"gemini_estimated"` key to `estimate_all()` return dict
- Update `ResponseFormatter` to include gemini estimate in `_token_estimates`

---

### Phase 2: Proportional Truncation (Enhancement 2)

#### Tests (`tests/test_response_formatter.py` -- extend existing)

```
test_proportional_truncation_preserves_head_and_tail
test_proportional_truncation_adds_prefix
test_proportional_truncation_no_continuation_token
test_proportional_strategy_via_constructor
test_default_strategy_is_paginate
test_head_only_truncation
test_proportional_truncation_small_response_no_change
```

#### Implementation
- Add `truncation_strategy` parameter to `ResponseFormatter.__init__`
- Implement `_truncate_proportional()` method
- Implement `_truncate_head()` method
- Route to appropriate truncation in `format_response()`

---

### Phase 3: Response Headers (Enhancement 3)

#### Tests (`tests/test_response_formatter.py` -- extend existing)

```
test_header_added_to_response
test_header_is_first_key
test_header_under_200_chars
test_header_contains_tool_name
test_header_survives_proportional_truncation
test_header_optional_via_parameter
```

#### Implementation
- Add `tool_name` parameter to `format_response()`
- Generate `_header` string from response metadata
- Insert as first key using dict insertion order

---

### Phase 4: Compression Urgency (Enhancement 4)

#### Tests (`tests/test_compression_profiles.py` -- extend existing)

```
test_urgency_normal_no_override
test_urgency_compact_caps_ratio
test_urgency_emergency_forces_minimal
test_apply_urgency_to_params
test_gemini_model_gets_more_aggressive_ratio
```

#### Implementation
- Add `apply_urgency(params: dict, urgency: str) -> dict` to `compression_profiles.py`
- Modify `get_recommended_ratio()` in `client_config.py` to factor in `compression_trigger`

---

### Phase 5: Model Database Expansion (Enhancement 5)

#### Tests (`tests/test_client_config.py` -- extend existing)

```
test_gemini_2_5_pro_context_window
test_gemini_3_1_flash_context_window
test_gemini_model_compression_trigger
test_claude_model_compression_trigger
test_gpt_model_no_compression_trigger
test_compression_trigger_affects_ratio
test_default_compression_trigger
```

#### Implementation
- Expand `KNOWN_MODEL_CONTEXT_WINDOWS` -> `KNOWN_MODEL_PROFILES` in `constants.py`
- Update `ClientConfig.from_model()` to read profile dict
- Update `get_recommended_ratio()` to factor compression trigger

---

## Implementation Order

```
Phase 1 (Gemini Estimation) ──> no dependencies
Phase 5 (Model Database)    ──> no dependencies, parallel with Phase 1
Phase 2 (Proportional Truncation) ──> depends on Phase 1 (uses gemini estimates)
Phase 3 (Response Headers) ──> depends on Phase 2 (headers interact with truncation)
Phase 4 (Urgency) ──> depends on Phase 5 (uses compression triggers from model DB)
```

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Gemini estimation formula changes | Formula is from public source; add env var override |
| Proportional truncation breaks JSON parsing | Only apply to serialized string, not structured data |
| Response headers add overhead | Headers are < 200 chars, negligible vs typical response |
| Model database becomes stale | Always fall back to explicit `context_window_tokens` parameter |
| Urgency override surprises users | Urgency only applies when explicitly set, never auto-detected |

---

## Research Sources

- [Gemini CLI MCP Server Docs](https://geminicli.com/docs/tools/mcp-server/) - Official MCP integration
- [Gemini CLI Configuration](https://geminicli.com/docs/reference/configuration/) - Token thresholds
- [Gemini API Context Caching](https://ai.google.dev/gemini-api/docs/caching) - Explicit cache API
- [Gemini Context Caching Cost Guide](https://www.aifreeapi.com/en/posts/gemini-api-context-caching-reduce-cost) - 75-90% savings
- [MCP Context Window Problem](https://www.apideck.com/blog/mcp-server-eating-context-window-cli-alternative) - MCP tool bloat
- [Leveraging 1M Token Window](https://inventivehq.com/knowledge-base/gemini/how-to-leverage-1m-token-context) - Large context strategies
- Gemini CLI source: `gemini-cli-main/packages/core/src/` (token calculation, truncation, compression, masking, distillation)
