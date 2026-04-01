# Claude Code Token Optimization Enhancements for Token Saver 5000

**Date:** 2026-03-31
**Version:** v0.11.0 proposal
**Author:** Auto-generated from Claude Code source analysis + feature review
**Scope:** Enhancements to Token Saver 5000 that optimize token usage when used with Claude Code, while remaining model-agnostic

---

## Executive Summary

Analysis of Claude Code's source (`claude-code-main/src/`), the feature review
(`feature-review2026.md`), and current MCP token optimization research reveals
**5 high-impact enhancements** that Token Saver 5000 can implement to dramatically
reduce token consumption when used as an MCP server with Claude Code (or any LLM
client). Each enhancement is designed to be **model-agnostic** -- it helps any MCP
consumer, but is especially effective with Claude Code's specific patterns.

### Key Findings from Claude Code Source

1. **Tool result budget**: Claude Code caps tool results at 50K chars/tool, 200K chars/message.
   Results exceeding the cap are persisted to disk with a 2KB preview. Our tool responses
   should be structured to stay under these limits and front-load critical information.

2. **Prompt cache sensitivity**: Claude Code sorts tools alphabetically and uses static/dynamic
   boundaries for cache stability. Our MCP tool schemas should be cache-friendly (stable
   descriptions, deterministic ordering).

3. **Microcompact**: Claude Code silently clears old tool results server-side when context
   exceeds ~180K tokens. Our tool responses should be self-contained (not reference prior
   tool results by content).

4. **Token estimation**: Claude Code uses `content.length / 4` as the default bytes-per-token
   ratio. Our token counting uses tiktoken (`cl100k_base`). We should expose both estimates
   so clients can reconcile.

5. **Context compaction**: When Claude Code compacts, it summarizes old turns into 9 structured
   sections. Our compressed skeletons are ideal pre-compaction artifacts -- they're already
   structured summaries that survive compaction well.

---

## Enhancement 1: Tool Result Size-Aware Response Formatting

### Problem
Claude Code enforces a 50K character limit per tool result and 200K aggregate per message.
When Token Saver returns large skeletons, graph exports, or batch results, they may be
silently truncated or persisted to disk with only a 2KB preview visible to the model.
The model then works with incomplete data without knowing it.

### Solution
Add a `response_formatter` module that:
- Measures response size before returning
- If response exceeds a configurable threshold (default: 40K chars, under the 50K limit),
  applies progressive summarization:
  1. First: truncate verbose metadata (processing times, cache stats)
  2. Second: compress node details to summary form
  3. Third: paginate with continuation token
- Adds a `truncated` flag and `continuation_token` when pagination is needed
- Includes `token_estimate` in every response (both tiktoken-accurate and `len/4` rough)

### Why Model-Agnostic
Any MCP client benefits from predictably-sized responses. The thresholds are configurable
via environment variables, so non-Claude clients can set their own limits.

### Constants
```python
# New constants for response formatting
TOOL_RESULT_SOFT_LIMIT_CHARS = int(os.getenv("TOOL_RESULT_SOFT_LIMIT", "40000"))
TOOL_RESULT_HARD_LIMIT_CHARS = int(os.getenv("TOOL_RESULT_HARD_LIMIT", "49000"))
TOOL_RESULT_PREVIEW_CHARS = int(os.getenv("TOOL_RESULT_PREVIEW", "2000"))
```

---

## Enhancement 2: Dual Token Estimation (Accurate + Fast)

### Problem
Token Saver uses tiktoken (`cl100k_base`) for accurate token counting. Claude Code uses
`content.length / 4` for fast estimation (or `/2` for JSON). When Claude Code estimates
how much context a tool result will consume, it uses the fast formula. This means our
reported `compression_ratio` may not match what Claude Code actually sees.

### Solution
Add dual token estimation to all responses:
- `tokens_accurate`: tiktoken count (our existing method)
- `tokens_estimated`: `len(content) / 4` (matches Claude Code's estimation)
- `tokens_json_estimated`: `len(content) / 2` (for JSON-heavy responses)
- `bytes`: raw byte count of the response

This lets the client (Claude Code or otherwise) use whichever estimate matches its
internal accounting. Also expose this as a standalone MCP tool: `estimate_tokens`.

### Why Model-Agnostic
Different LLM providers use different tokenizers. By providing multiple estimates,
any client can pick the one that matches. The raw byte count is universally useful.

---

## Enhancement 3: Context-Window-Aware Adaptive Compression

### Problem
Our `MAX_CONTEXT_TOKENS` is hardcoded at 100K. Claude Code supports 200K and 1M context
windows. When a Claude Code user has a 1M context window, our compression is overly
aggressive. When they have a smaller window, we may not compress enough.

### Solution
Enhance the `adapt_to_context_window` tool and add a new `configure_for_client` tool:
- Accept `context_window_tokens` parameter (detected from client or user-specified)
- Accept `model_id` parameter (optional) to auto-detect window size from known models
- Dynamically adjust `skeleton_ratio` based on available context:
  - 1M window: ratio 0.4-0.6 (preserve more detail)
  - 200K window: ratio 0.2-0.3 (balanced, current default)
  - 100K window: ratio 0.1-0.15 (aggressive compression)
  - 50K window: ratio 0.05-0.1 (maximum compression)
- Store client config per session for consistent behavior across tool calls

### Model Database
```python
KNOWN_MODEL_CONTEXT_WINDOWS = {
    # Claude models
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5": 200_000,
    # 1M variants
    "claude-opus-4-6[1m]": 1_000_000,
    "claude-sonnet-4-6[1m]": 1_000_000,
    # GPT models
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    # Gemini
    "gemini-2.0-pro": 2_000_000,
    "gemini-2.0-flash": 1_000_000,
    # Defaults
    "default": 100_000,
}
```

### Why Model-Agnostic
The model database supports all major providers. Unknown models fall back to 100K.
The `context_window_tokens` parameter allows any client to specify its own window.

---

## Enhancement 4: Cache-Friendly Tool Schema Design

### Problem
Claude Code caches the prompt prefix, which includes all MCP tool schemas. Any change
to tool descriptions between sessions busts the cache. Our tool schemas include dynamic
content (version numbers, document counts) that can change between calls.

### Solution
1. **Stabilize tool descriptions**: Remove any dynamic content from tool schema descriptions.
   Move version info, counts, and runtime state to tool *responses*, not schemas.
2. **Add `tool_profile` metadata**: Include a `schema_version` in tool annotations so
   Claude Code's deferred tool loading can track changes efficiently.
3. **Deterministic schema ordering**: Ensure `setup_mcp_tools()` returns tools in a
   stable alphabetical order (already mostly the case, but audit for consistency).

### Implementation
- Audit all 49+ tool schemas in `mcp_core.py` for dynamic content
- Extract any runtime values from descriptions
- Add `schema_version` annotation to tool metadata

### Why Model-Agnostic
Stable tool schemas benefit any MCP client that caches prompts. This is a general
MCP best practice, not Claude-specific.

---

## Enhancement 5: Compressed Response Presets (Fidelity Profiles)

### Problem
Claude Code's microcompact silently clears old tool results. When a user reads a skeleton,
then later asks about it, the skeleton content may have been cleared. The model needs to
re-read, wasting tokens. Additionally, different use cases need different compression levels
but users must manually tune `skeleton_ratio`, `fidelity_level`, etc.

### Solution
Add named **fidelity profiles** that bundle multiple parameters into a single preset:

| Profile | skeleton_ratio | fidelity | chunk_size | Use Case |
|---------|---------------|----------|------------|----------|
| `minimal` | 0.05 | ABSTRACT | 256 | Maximum compression, navigation only |
| `summary` | 0.15 | OUTLINE | 512 | Quick overview, fits in compacted context |
| `balanced` | 0.25 | STRUCTURE | 512 | Default, good for most tasks |
| `detailed` | 0.50 | DETAILED | 1024 | Code review, deep analysis |
| `full` | 0.80 | RAW | 2048 | Near-original, minimal compression |

- New tool: `set_compression_profile` -- sets the active profile for the session
- All compression tools respect the active profile unless explicitly overridden
- Profiles are stored per-session (not global) via `session_id` scoping
- The `minimal` and `summary` profiles are designed to produce results that survive
  Claude Code's microcompact (small enough to not be cleared, self-contained)

### Why Model-Agnostic
Named presets simplify the UX for any MCP client. The profiles are based on
universal compression trade-offs, not Claude-specific behavior.

---

## TDD Implementation Plan

### Phase 1: Response Formatter (Enhancement 1)

#### Tests First (`tests/test_response_formatter.py`)

```
test_format_response_under_limit_passes_through
test_format_response_at_soft_limit_strips_metadata
test_format_response_at_hard_limit_paginates
test_pagination_continuation_token_is_deterministic
test_truncated_flag_set_when_paginated
test_token_estimates_included_in_every_response
test_custom_limits_via_env_vars
test_empty_response_handling
test_binary_content_handling
test_nested_json_response_measurement
```

#### Implementation Files
1. `src/response_formatter.py` -- core formatter module
2. `src/constants.py` -- add new constants
3. `src/handlers/mcp_core.py` -- wrap `route_tool_call` return with formatter
4. `tests/test_response_formatter.py` -- tests

#### Approach
- Write all tests first against the `ResponseFormatter` interface
- Implement `ResponseFormatter` class with `format_response(data, limits)` method
- Integrate into `route_tool_call()` as a post-processing step
- All existing handler return values pass through unchanged if under limit

---

### Phase 2: Dual Token Estimation (Enhancement 2)

#### Tests First (`tests/test_token_estimation.py`)

```
test_accurate_token_count_matches_tiktoken
test_fast_estimate_uses_len_div_4
test_json_estimate_uses_len_div_2
test_byte_count_is_exact
test_estimate_tokens_tool_returns_all_methods
test_empty_string_returns_zero_for_all
test_unicode_content_byte_vs_char_difference
test_code_content_estimation_accuracy
test_mixed_content_estimation
test_estimation_added_to_compression_responses
```

#### Implementation Files
1. `src/token_estimation.py` -- dual estimation module
2. `src/handlers/mcp_core.py` -- add `estimate_tokens` tool schema
3. `src/handlers/compression_handlers.py` -- add dual estimates to responses
4. `tests/test_token_estimation.py` -- tests

#### Approach
- Write tests against `TokenEstimator` interface
- Implement with tiktoken (existing) + fast estimation (new)
- Add `estimate_tokens` MCP tool
- Inject dual estimates into all compression handler responses

---

### Phase 3: Context-Window-Aware Compression (Enhancement 3)

#### Tests First (`tests/test_client_config.py`)

```
test_configure_for_known_claude_model
test_configure_for_known_gpt_model
test_configure_for_unknown_model_uses_default
test_configure_with_explicit_context_window
test_explicit_window_overrides_model_lookup
test_skeleton_ratio_scales_with_window_size
test_1m_window_produces_less_compression
test_50k_window_produces_more_compression
test_session_config_persists_across_calls
test_session_config_isolated_between_sessions
test_configure_for_client_tool_returns_config
test_adapt_to_context_window_uses_client_config
test_model_database_covers_major_providers
test_invalid_context_window_rejected
```

#### Implementation Files
1. `src/client_config.py` -- client configuration and model database
2. `src/constants.py` -- add `KNOWN_MODEL_CONTEXT_WINDOWS`
3. `src/handlers/mcp_core.py` -- add `configure_for_client` tool schema
4. `src/handlers/compression_handlers.py` -- read session config for ratio tuning
5. `tests/test_client_config.py` -- tests

#### Approach
- Write tests for `ClientConfig` class and model database
- Implement session-scoped configuration storage
- Add `configure_for_client` MCP tool
- Modify compression handlers to consult session config for ratio defaults

---

### Phase 4: Cache-Friendly Schemas (Enhancement 4)

#### Tests First (`tests/test_schema_stability.py`)

```
test_tool_schemas_are_alphabetically_ordered
test_tool_descriptions_contain_no_dynamic_content
test_tool_descriptions_no_version_numbers
test_tool_descriptions_no_document_counts
test_schema_version_annotation_present
test_schema_output_is_deterministic_across_calls
test_schema_hash_stable_between_invocations
```

#### Implementation Files
1. `src/handlers/mcp_core.py` -- audit and stabilize schemas
2. `tests/test_schema_stability.py` -- tests

#### Approach
- Write stability tests that hash schema output and assert consistency
- Audit all tool descriptions for dynamic content
- Move any dynamic content from descriptions to response metadata
- Add `schema_version` to tool annotations

---

### Phase 5: Compression Profiles (Enhancement 5)

#### Tests First (`tests/test_compression_profiles.py`)

```
test_minimal_profile_values
test_summary_profile_values
test_balanced_profile_values
test_detailed_profile_values
test_full_profile_values
test_set_profile_tool_stores_in_session
test_profile_applies_to_subsequent_ingest
test_profile_applies_to_subsequent_read_skeleton
test_explicit_params_override_profile
test_unknown_profile_name_rejected
test_profile_isolated_per_session
test_default_profile_is_balanced
test_profile_persists_across_tool_calls
test_get_active_profile_tool
```

#### Implementation Files
1. `src/compression_profiles.py` -- profile definitions and session management
2. `src/handlers/mcp_core.py` -- add `set_compression_profile` and `get_compression_profile` tool schemas
3. `src/handlers/compression_handlers.py` -- consult active profile for defaults
4. `tests/test_compression_profiles.py` -- tests

#### Approach
- Write tests for profile definitions and session behavior
- Implement `CompressionProfile` dataclass and `ProfileManager` session store
- Add two new MCP tools
- Modify compression handlers to use active profile as defaults

---

## Implementation Order & Dependencies

```
Phase 1 (Response Formatter) ──> no dependencies, start first
Phase 2 (Token Estimation)   ──> no dependencies, can parallelize with Phase 1
Phase 3 (Client Config)      ──> depends on Phase 2 (uses token estimation)
Phase 4 (Schema Stability)   ──> no dependencies, can parallelize
Phase 5 (Profiles)           ──> depends on Phase 3 (uses client config for auto-tuning)
```

**Recommended execution order:** Phase 1 + 2 (parallel) -> Phase 3 + 4 (parallel) -> Phase 5

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Response formatting adds latency | Format only when response exceeds soft limit (fast path for small responses) |
| Dual estimation diverges from Claude's actual counting | Document that `tokens_estimated` is approximate; provide `bytes` for exact |
| Model database becomes stale | Use `default` fallback; allow explicit `context_window_tokens` override |
| Schema changes bust existing caches | Bump `schema_version` and document in changelog |
| Profiles add complexity to API | Profiles are optional; all existing parameters still work without profiles |
| Regression in existing compression | All tests run against existing behavior first; new code is additive |

---

## Non-Goals (Explicitly Out of Scope)

- **Claude-only features**: No code paths that only work with Claude. Every enhancement
  benefits any MCP client.
- **Prompt caching control**: We can't control Claude Code's cache markers from the MCP
  server side. We can only make our schemas cache-friendly.
- **Microcompact integration**: We can't detect or respond to Claude Code's server-side
  context editing. We can only make our responses robust to it.
- **Fork/swarm integration**: These are Claude Code internal features not exposed via MCP.

---

## Research Sources

- [Token Optimizer MCP Server](https://mcpmarket.com/server/token-optimizer) - MCP token optimization patterns
- [MCP Token Optimization Guide (BSWEN)](https://docs.bswen.com/blog/2026-03-23-mcp-token-optimization-claude-code/) - Practical optimization approaches
- [Claude Code Tool Search 46.9% Reduction](https://medium.com/@joe.njenga/claude-code-just-cut-mcp-context-bloat-by-46-9-51k-tokens-down-to-8-5k-with-new-tool-search-ddf9e905f734) - Deferred tool loading impact
- [Prompt Caching Best Practices](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) - Official Claude prompt caching docs
- [Prompt Caching in Claude Code (Camp)](https://www.claudecodecamp.com/p/how-prompt-caching-actually-works-in-claude-code) - Cache architecture deep dive
- [Optimising MCP Context Usage](https://scottspence.com/posts/optimising-mcp-server-context-usage-in-claude-code) - Practical MCP optimization
- Claude Code source: `claude-code-main/src/` (token counting, compaction, tool result storage, prompt caching)
- Feature review: `claude-code-main/feature-review2026.md` (30-section gap analysis)
