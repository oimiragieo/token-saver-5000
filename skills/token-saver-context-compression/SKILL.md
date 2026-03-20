---
name: token-saver-context
description: Compress large context before reasoning with exact local script commands from this repo. Use this whenever the user asks for the command or command pattern for `compress_context.py`, `run_skill_workflow.py`, `validate_evidence.py`, `--input-adapter`, query-guided compression, evidence-aware compression, LangChain-style JSON adaptation, prompt-caching-friendly context trimming, or cheaper large-context handling. Prefer this skill over generic repo or MCP guidance when the user wants a local script command.
compatibility: Python 3.10+
---

# Token Saver Context Compression

Use this skill when the problem is mostly "too much context" rather than "not enough capability."

This skill is a self-contained local package. It does not require the MCP server. Prefer it when you need quick token profiling, local compression, evidence checks, or a reproducible compression workflow inside the repository.

## Ground-truth policy

Treat the bundled script commands in this skill as the source of truth.

- Prefer the local scripts in `skills\token-saver-context-compression\scripts\`.
- Do **not** fall back to MCP tools, `python -m src.server`, or generic repo guidance unless the user explicitly asks for MCP mode.
- Do **not** invent script names like `compress.py` or tool names like `read_context`.
- If the user asks for an exact command, answer with one of the exact command forms from this skill before adding any explanation.
- If filesystem tools are unavailable, still use the commands documented in this skill rather than claiming you cannot determine them.

## What this skill does

Use the bundled Python scripts to:

1. measure raw versus compressed token usage
2. compress large text or adapted JSON payloads
3. preserve query-relevant evidence instead of doing naive summarization
4. check whether compressed output still contains enough evidence to answer safely
5. support cache-friendly prompt assembly by keeping stable context early and volatile query text late

## When to trigger

Reach for this skill when the user is asking for any of the following:

- compress this file, transcript, document, or code dump
- reduce token cost before sending context to a model
- make a prompt or RAG payload more cache-friendly
- compact multi-turn history while keeping recent turns intact
- compare semantic compression to a lighter extractive baseline
- validate whether a compressed context still has enough evidence

## Quick workflow

Default to this sequence:

1. profile first
2. run `query_guided` when there is a specific question
3. run `evidence_aware` when correctness matters and you need a sufficiency check
4. if evidence is weak, reduce compression aggressiveness or increase retrieval breadth
5. report savings, risks, and the next safest action

If the user only wants one command, use `run_skill_workflow.py`.

## Exact command map

When the user asks for the exact command, use one of these verbatim patterns and then adapt only the file path, query text, or payload path:

### Exact query-targeted command

```bash
python skills\token-saver-context-compression\scripts\compress_context.py --file tests\fixtures\skill_context_sample.txt --mode query_guided --query "what are the retry rules?" --output-format auto
```

### Exact high-confidence workflow command

```bash
python skills\token-saver-context-compression\scripts\run_skill_workflow.py --file tests\fixtures\skill_context_sample.txt --mode evidence_aware --query "what are the retry rules?" --output-format auto --fail-on-insufficient-evidence
```

### Exact evidence-only command

```bash
python skills\token-saver-context-compression\scripts\validate_evidence.py --file tests\fixtures\skill_context_sample.txt --query "what are the retry rules?" --min-similarity 0.4 --output-format json
```

### Exact LangChain-style JSON command

```bash
python skills\token-saver-context-compression\scripts\compress_context.py --json-file <payload.json> --input-adapter auto --mode query_guided --query "<question>" --output-format auto
```

## Response rules

When answering from this skill:

1. lead with the exact command if the user asks for a command or command pattern
2. name the selected mode explicitly: `baseline`, `query_guided`, or `evidence_aware`
3. say whether the result is safe to answer from
4. if the answer is not safe enough, recommend `run_skill_workflow.py` or `validate_evidence.py`

## Anti-patterns

Do not do any of these unless the user explicitly asks for them:

- recommend `python -m src.server`
- answer with MCP tool sequences instead of the bundled scripts
- cite broad `CLAUDE.md` repo context when the skill already provides an exact local script command
- claim the command is unknown when it is documented here

## Commands

Run from the repository root.

### 1. Profile token usage

```bash
python skills\token-saver-context-compression\scripts\profile_tokens.py --file <path> --output-format auto
```

### 2. Compress context

```bash
python skills\token-saver-context-compression\scripts\compress_context.py --file <path> --mode baseline --output-format auto
python skills\token-saver-context-compression\scripts\compress_context.py --file <path> --mode query_guided --query "<question>" --output-format auto
python skills\token-saver-context-compression\scripts\compress_context.py --file <path> --mode evidence_aware --query "<question>" --min-similarity 0.4 --output-format auto
```

### 3. Adapt framework JSON and compress it

```bash
python skills\token-saver-context-compression\scripts\compress_context.py --json-file <payload.json> --input-adapter auto --mode query_guided --query "<question>" --output-format auto
```

### 4. Run the full workflow

```bash
python skills\token-saver-context-compression\scripts\run_skill_workflow.py --file <path> --mode evidence_aware --query "<question>" --output-format auto --fail-on-insufficient-evidence
```

### 5. Validate evidence only

```bash
python skills\token-saver-context-compression\scripts\validate_evidence.py --file <path> --query "<question>" --min-similarity 0.4 --output-format json
```

### 6. Run the TOON vs JSON guard

```bash
python skills\token-saver-context-compression\scripts\benchmark_toon_vs_json.py
```

## How to choose a mode

- `baseline`: quick general compression when there is no concrete question yet
- `query_guided`: best default for QA, review, or targeted extraction tasks
- `evidence_aware`: use for high-stakes answers, audits, or when you need an explicit sufficiency signal

If the user asks whether the result is safe to answer from, prefer `evidence_aware` or `run_skill_workflow.py` over `query_guided`.

## Output expectations

When using this skill, summarize results in plain language:

1. original size versus compressed size
2. estimated token savings or compression ratio
3. whether the output is query-targeted or generic
4. whether evidence looked sufficient
5. any risk that the answer could miss important detail

If the scripts return insufficient evidence, do not bluff. Say the compressed context is not yet safe enough and recommend a broader pass.

## Bundled references

Read these only when they are relevant:

- `references\workflow-guide.md`: command selection, mode choice, and example flows
- `references\prompt-caching.md`: stable-prefix ordering, cache telemetry, and cache-safe prompt structure
- `references\evaluation.md`: how to benchmark the skill and interpret results
- `references\skill-creator-reference\`: the downloaded Anthropic reference skill kept for local comparison

## Eval scaffolding

Starter prompts live in `evals\evals.json`. Use them when iterating on the skill or when you want a small repeatable benchmark set.
