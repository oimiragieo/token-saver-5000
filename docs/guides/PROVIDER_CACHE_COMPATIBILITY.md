# Provider and Harness Cache Compatibility

This guide answers a practical question:

How do you make Token Saver 5000 work predictably with prompt caching in:

1. Gemini CLI / Gemini API
2. Claude / Claude Code / Anthropic API
3. Codex / OpenAI API

The short answer is: you need to understand **both** the provider and the harness.

- The **provider** decides whether prompt caching exists, what counts as a cache hit, how long the cache lives, and which telemetry fields prove reuse happened.
- The **harness** decides whether your supposedly stable prefix actually stays stable after message assembly, tool wiring, config injection, and conversation growth.

If you only deep dive one side, you can still miss the real failure mode.

## Expected Compression and Savings

From the repository’s current heuristics and docs, the compression side of Token Saver 5000 usually lands in these ranges:

| Context size | Typical compression ratio | Approx token reduction |
| --- | --- | --- |
| Small | 2x to 4x | 50% to 75% |
| Medium | 5x to 10x | 80% to 90% |
| Large | 15x to 20x | 93% to 95% |

That is the **compression layer** only.

On top of that, when prompt caching works correctly, the **reused prefix** can often be billed at a sharply reduced rate:

- Claude prompt-cache reads are documented at roughly 90% cheaper than normal input tokens.
- OpenAI cached prompt tokens are billed at a reduced rate and exposed as `cached_tokens`.
- Gemini implicit/explicit cache reuse is reflected in cached token counts and reduced billing.

In repeated workflows, the combined effect of:

1. compressing the payload, and
2. caching the stable prefix

often produces real-world savings in the broad **80% to 95%+** range, depending on:

- how stable the prefix is,
- how much of the prompt is reused,
- how often the same context is hit again,
- and whether the harness preserves prefix identity.

## The Rule: Deep Dive Both

If your goal is reliable compatibility in Gemini CLI, Claude, and Codex:

1. Deep dive the **provider model docs first**
2. Deep dive the **harness or CLI prompt assembly second**

Why this order:

- Provider docs tell you the ground truth for cache semantics, TTLs, thresholds, and telemetry fields.
- Harness docs tell you whether those provider semantics survive contact with the actual tool you are using.

## Compatibility Matrix

| Surface | Cache mode | Primary telemetry fields | Harness risk | Token Saver status |
| --- | --- | --- | --- | --- |
| Anthropic API | Automatic and explicit prompt caching | `cache_read_input_tokens`, `cache_creation_input_tokens` | Frameworks or wrappers can still inject dynamic content into `tools`, `system`, or message prefixes | Supported |
| Claude Code | Anthropic prompt caching underneath when the assembled prompt is cacheable | Depends on exposed usage payloads or logs; underlying provider fields are still the Anthropic ones | High: tool ordering, conversation growth, CLAUDE.md patterns, and injected metadata can break byte-identical prefixes | Supported when provider usage is available |
| OpenAI API | Automatic prompt caching on eligible prefixes | `usage.prompt_tokens_details.cached_tokens` | Prefix drift from system/tool reordering or hidden config can erase reuse | Supported |
| Codex CLI | OpenAI prompt caching underneath when prompt assembly stays stable | Underlying OpenAI `cached_tokens`; CLI or logs may expose only a subset unless verbose stats are enabled | High: session assembly, model changes, config drift, and thread compaction can change the prefix | Supported when raw usage or verbose cache stats are available |
| Gemini API | Implicit and explicit context caching | `cachedContentTokenCount` or SDK-style `usage_metadata.cached_content_token_count` | Prefix changes and authentication mode differences can reduce or hide reuse | Supported |
| Gemini CLI | Gemini caching underneath, plus CLI stats surfaces | `/stats`, CLI token caching telemetry, and provider-style cached token counts when exposed | High: CLI prompt assembly, auth mode, and internal message shaping can alter reuse behavior | Supported when cache stats or usage metadata are available |

## What Token Saver 5000 Already Supports

Today the repo is strongest in these areas:

1. **Stable-prefix analysis**
   - `audit_prompt_cacheability`
   - `render_prompt_template`
   - stable-section ordering and volatility checks

2. **Provider + harness compatibility assessment**
   - `assess_cache_compatibility`
   - raw-usage versus CLI-stats visibility checks
   - surface-specific harness risk guidance for Gemini CLI, Claude Code, and Codex

3. **Provider telemetry normalization**
   - Anthropic-style `cache_read_input_tokens`
   - OpenAI-style `cached_tokens`
   - Gemini-style `cachedContentTokenCount`
   - Gemini SDK-style snake_case `usage_metadata.cached_content_token_count`
   - Gemini CLI-style camelCase stats exports such as `inputTokens`, `outputTokens`, and `cachedTokens`

4. **Cache miss and degraded reuse diagnostics**
   - section interleaving
   - semantic-equivalence drift
   - partial cache reuse underperformance
   - cache creation churn
   - session-level cache aggregation

## Where the Real Risk Still Lives

The biggest remaining risk is usually **not** the provider API itself.

It is the harness layer:

1. the CLI or agent framework may reorder sections,
2. inject IDs, timestamps, or dynamic config,
3. expand tool definitions in a different order,
4. compact history differently across turns,
5. or hide the raw provider usage payload you need for telemetry normalization.

That means the next deep dives should answer:

### Gemini CLI

- Does the CLI expose raw `usage_metadata`, or only summarized cache stats?
- Which auth modes preserve token caching?
- Does the CLI prepend dynamic metadata before your stable context?
- Can `/stats` or telemetry export be captured programmatically?

### Claude / Claude Code

- Are `cache_control` breakpoints or automatic caching under your direct control?
- Does Claude Code preserve tool ordering and system preamble deterministically?
- Can you reliably access `cache_read_input_tokens` and `cache_creation_input_tokens` for each turn?
- Does any project-level automation inject changing text into the front of the prompt?

### Codex

- Does the Codex surface give you raw OpenAI usage payloads or only summary reporting?
- Which model IDs are actually used in practice: `gpt-5.4`, `gpt-5.3-codex`, or another Codex family model?
- Does thread resume, compaction, or model switching rewrite the stable prefix?
- Can you pin tool and system sections so only the user tail changes?

## Practical Recommendation Order

If your goal is production-grade cache reliability, do this in order:

1. **Provider docs**
   - Anthropic prompt caching docs
   - OpenAI prompt caching docs
   - Gemini context caching docs

2. **Harness docs**
   - Claude Code prompt assembly / usage visibility
   - Codex CLI model and session behavior
   - Gemini CLI token caching and `/stats` behavior

3. **Instrument the real surface**
   - capture raw provider responses when possible
   - normalize them with `capture_cache_telemetry`
   - compare expected and actual prefixes with `diagnose_cache_miss`

4. **Verify session behavior**
   - do not trust a single cache hit
   - monitor session-level hit ratios
   - watch for repeated cache creation churn
   - check for partial reuse that looks healthy but is actually underperforming

## Current Recommendations by Surface

### Best fit for Claude

Use Token Saver’s prompt-cache tools with Anthropic usage payloads directly whenever possible. If you are operating through Claude Code, make sure you can still observe `cache_read_input_tokens` and `cache_creation_input_tokens` or an equivalent surfaced log.

### Best fit for Codex

If you are running Codex with `gpt-5.4`, Token Saver’s current OpenAI cache telemetry path is the cleanest fit. If you use a Codex-specific model ID, confirm whether your integration still surfaces the standard OpenAI `cached_tokens` usage block before relying on automated cache accounting.

### Best fit for Gemini CLI

Gemini CLI is promising because it explicitly talks about token caching and stats, but you still need to verify whether you get raw provider-style usage metadata or only summarized CLI counters. Token Saver can already normalize Gemini cached token fields once they are available in the payload or exported stats. For current model naming, prefer the Gemini 3.x family such as `gemini-3.1-pro-preview`.

## References

- Anthropic prompt caching docs: `https://platform.claude.com/docs/en/build-with-claude/prompt-caching`
- OpenAI prompt caching docs: `https://developers.openai.com/api/docs/guides/prompt-caching/`
- OpenAI Codex model docs: `https://developers.openai.com/codex/models`
- OpenAI Codex CLI reference: `https://developers.openai.com/codex/cli/reference`
- Gemini context caching docs: `https://ai.google.dev/gemini-api/docs/caching`
- Gemini CLI token caching docs: `https://geminicli.com/docs/cli/token-caching/`
