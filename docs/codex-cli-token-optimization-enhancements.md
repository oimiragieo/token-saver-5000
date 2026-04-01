# Codex CLI Token Optimization Enhancements for Token Saver 5000

**Date:** 2026-03-31
**Version:** v0.11.0 (additive to Claude Code and Gemini CLI enhancements)
**Author:** Auto-generated from Codex CLI source analysis
**Scope:** Enhancements to Token Saver 5000 that optimize token usage when used with
Codex CLI, while remaining model-agnostic

---

## Executive Summary

Analysis of Codex CLI's source (`codex-main/`) reveals a token management approach that
shares foundations with Claude Code (byte-based estimation) but implements its own
context compaction strategy. Key differences from Claude Code and Gemini CLI:

| Aspect | Claude Code | Gemini CLI | Codex CLI |
|--------|------------|------------|-----------|
| **Token counting** | `len/4` bytes (fast) | ASCII: 0.25/char, non-ASCII: 1.3/char | `len/4` bytes (same as Claude Code) |
| **Tool result limit** | 50K chars hard cap | 40K tokens soft cap | `config.tool_output_token_limit` (configurable) |
| **Truncation strategy** | Replace with 2KB preview + disk ref | 20% head + 80% tail with ellipsis | Middle-truncation: head + `"…[N] tokens truncated…"` + tail |
| **Context compression** | Microcompact + full summarization | 3-phase: truncate → LLM → verify | Context compaction: replaces conversation with compact summary |
| **Compression trigger** | ~93% of window (200K - 13K buffer) | 50% of token limit | ~80% of context window (`HISTORY_SOFT_CAP_RATIO = 0.8`) |
| **Tool output masking** | Clears old results (microcompact) | Backward-scan masking, protects 50K | Middle-truncation with preserved head+tail |
| **Context window** | 200K or 1M tokens | 1,048,576 tokens (1M default) | Model-specific via `ModelInfo.context_window` |
| **Cache mechanism** | Prompt prefix caching (automatic) | Explicit context caching (API-managed) | `prompt_cache_key` (conversation ID), 1024+ token prefix |
| **Primary models** | Claude Sonnet/Opus | Gemini 2.5 Pro/Flash | gpt-5.1-codex, o3, o4-mini |
| **Request compression** | N/A | N/A | ZSTD body compression (feature-flagged) |
| **Headless mode** | `claude -p --output-format json` | `gemini --output-format json` | `codex exec --json` (JSONL stream, not single JSON) |

### Key Findings from Codex CLI Source

1. **JSONL output (not single JSON)**: `codex exec --json` streams multiple JSON lines
   (one per event), not a single JSON object. The `turn.completed` event contains usage
   data; the `item.completed` event with `agent_message` contains the response text.
   Our benchmark parser must iterate all lines, not call `json.loads()` once.

2. **Middle-truncation preserves context edges**: Codex CLI truncates tool results by
   keeping the head and tail with a `"…[N] tokens truncated…"` marker in the middle.
   This is a hybrid of Claude's head-only (disk ref) and Gemini's head+tail approach.
   Our `ResponseFormatter` should offer a `middle` truncation strategy to match.

3. **Token estimation parity with Claude Code**: Both Codex CLI and Claude Code use the
   same `len/4` bytes heuristic. Token Saver 5000's existing estimation is already
   compatible — no divergence compensation needed for Codex.

4. **Prompt caching via conversation ID**: Codex uses `prompt_cache_key` (the
   conversation ID) for prefix-based caching. Cache hits require exact prefix match,
   work for 1024+ tokens in 128-token increments. Our stable tool schemas benefit
   this caching since unchanged preamble stays in cache between turns.

5. **Compression trigger at 80%**: Codex CLI compresses at 80% of context window
   (`HISTORY_SOFT_CAP_RATIO = 0.8`), between Claude Code (93%) and Gemini CLI (50%).
   Compressed skeletons should be sized to fit comfortably within the 80% threshold.

6. **ZSTD request compression**: Codex CLI optionally compresses entire request bodies
   with ZSTD. This is transparent to our server — we serve plain JSON, and the
   compression is applied at the HTTP transport layer by Codex.

---

## Enhancement 1: Codex Model Database Expansion

### Problem
Token Saver 5000's `KNOWN_MODEL_CONTEXT_WINDOWS` and `KNOWN_MODEL_COMPRESSION_TRIGGERS`
in `src/constants.py` lacked entries for Codex-native models (`gpt-5.1-codex`,
`codex-mini`) and the o3/o4-mini models used by Codex workflows.

### Solution
Add Codex/OpenAI model entries to both dictionaries:

```python
# In KNOWN_MODEL_CONTEXT_WINDOWS:
"gpt-5.1-codex": 200_000,
"codex-mini": 200_000,

# In KNOWN_MODEL_COMPRESSION_TRIGGERS:
"gpt-5.1-codex": 0.80,   # HISTORY_SOFT_CAP_RATIO = 0.8
"codex-mini": 0.80,
"o3": 0.80,
"o4-mini": 0.80,
```

### Impact
- `configure_for_client` tool can auto-detect appropriate skeleton ratios for Codex workflows
- 80% trigger (vs 50% for Gemini) means moderately aggressive compression targeting

---

## Enhancement 2: Benchmark Provider Support for Codex CLI

### Problem
The Token Saver 5000 benchmark suite (`src/cli_benchmark/`) supported Claude Code and
Gemini CLI but had no Codex provider. This prevented direct measurement of token savings
when Token Saver is used in Codex-based workflows.

### Solution
Add `codex` as a third provider in the benchmark infrastructure:

**`providers.py`:**
- `is_available("codex")`: checks for `codex` binary on PATH
- `_build_command("codex", model)`: `codex exec --json --dangerously-bypass-approvals-and-sandbox`
  (prompt via stdin)
- `_parse_codex_result(lines, raw)`: parses JSONL stream, extracts `turn.completed` usage
  and last `item.completed` agent message text

**`pricing.py`:**
```python
"gpt-5.1-codex": {"input": 2.50, "output": 10.0, "cache_read": 0.625},
"codex-mini": {"input": 1.50, "output": 6.0, "cache_read": 0.375},
```

**`runner.py`:**
- `model_codex` parameter added to `run_benchmark()`
- Model routing: `provider == "codex"` maps to `model_codex`

**`benchmark_token_savings.py`:**
- `--model-codex` CLI argument
- `--providers` choices now include `"codex"`

### JSONL Parsing Strategy
Codex outputs one JSON object per line. Our parser:
1. Splits stdout on newlines
2. Attempts `json.loads()` on each non-empty line
3. Searches for `{"type": "turn.completed", "usage": {...}}` for token counts
4. Searches for the last `item.completed` event with `agent_message` for response text
5. Returns gracefully with zeros if `turn.completed` is not found

---

## Enhancement 3: Pricing Data for Codex Models

### Problem
No pricing data existed for `gpt-5.1-codex` or `codex-mini` in `pricing.py`, so
`compute_cost()` fell back to the generic default rates, producing inaccurate cost
comparisons.

### Solution
Add per-model pricing rates matching OpenAI's published Codex API pricing:

| Model | Input ($/M) | Output ($/M) | Cache Read ($/M) |
|-------|-------------|--------------|------------------|
| gpt-5.1-codex | $2.50 | $10.00 | $0.625 |
| codex-mini | $1.50 | $6.00 | $0.375 |

Cache read is priced at 25% of input rate (same ratio as OpenAI's standard models).

---

## Research Sources

1. **Codex CLI source** (`codex-main/`): `HISTORY_SOFT_CAP_RATIO`, `ModelInfo`,
   `prompt_cache_key`, ZSTD compression flag, `codex exec --json` JSONL format
2. **OpenAI Codex API pricing** (2026-Q1): gpt-5.1-codex and codex-mini rates
3. **Claude Code source analysis** (prior): `len/4` byte heuristic, 50K char cap
4. **Gemini CLI source analysis** (prior): 50% trigger, head+tail truncation

---

## Implementation Checklist

- [x] `src/constants.py`: Added `gpt-5.1-codex`, `codex-mini` to context windows dict
- [x] `src/constants.py`: Added `gpt-5.1-codex`, `codex-mini`, `o3`, `o4-mini` to triggers dict
- [x] `src/cli_benchmark/providers.py`: Codex `is_available()`, `_build_command()`, `_parse_codex_result()`
- [x] `src/cli_benchmark/providers.py`: `run_prompt()` dispatches to `_parse_codex_result` for "codex"
- [x] `src/cli_benchmark/pricing.py`: `gpt-5.1-codex` and `codex-mini` pricing entries
- [x] `src/cli_benchmark/runner.py`: `model_codex` parameter, codex model routing
- [x] `scripts/benchmark_token_savings.py`: `--model-codex` arg, codex in `--providers` choices
- [x] `tests/test_codex_enhancements.py`: 11 tests covering all new behavior
