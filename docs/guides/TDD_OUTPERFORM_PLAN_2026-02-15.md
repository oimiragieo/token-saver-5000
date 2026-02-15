# TDD Outperform Plan (2026-02-15)

Goal: implement and validate improvements that outperform comparable context-compression codebases on quality, usability, and reliability.

Reference comparison:

- `docs/research/COMPETITOR_CODEBASE_COMPARISON_2026-02-15.md`

## Success Criteria

1. Maintain or improve current token-savings baselines.
2. Add measurable retrieval quality metrics (precision/recall/F1).
3. Improve UX with composable compressor pipelines and structured controls.
4. Preserve reliability gates: lint, format, targeted tests, full regression.

## TDD Execution Rules

For every slice:

1. Red: write failing tests first.
2. Green: implement smallest possible passing behavior.
3. Refactor: remove complexity, keep behavior stable.
4. Verify: lint + format + targeted tests + regression ring.
5. Document: README/docs/changelog update in same change set.

## Slice 1: Composable Compression Pipeline

Objective:

1. Introduce pipeline abstraction for stacking compressor passes.

Design:

1. Add app-layer pipeline interfaces:
   - `compressor passes` (filter/extractor/transform)
   - deterministic execution order
2. Expose pipeline in skill script modes with explicit selection.

Red tests:

1. Pipeline executes stages in order.
2. Pipeline supports sync/async pass functions.
3. Pipeline failure surfaces stage-local error context.

Green implementation:

1. Add minimal pipeline engine module.
2. Wire baseline + query_guided as pipeline profiles.

Refactor:

1. Remove duplicated selection logic in scripts/handlers by delegating to pipeline.

## Slice 2: Structured Segment Controls

Objective:

1. Allow user-controlled per-segment compression policy.

Design:

1. Accept structured markers for preserve/compress priorities.
2. Preserve compatibility for plain-text inputs.

Red tests:

1. Marked segments are preserved when requested.
2. Segment policy overrides default ratio only for tagged regions.
3. Unbalanced/invalid markers fail with clear validation error.

Green implementation:

1. Parse segment markers to region policy map.
2. Apply policy during segment scoring/selection.

Refactor:

1. Centralize marker parsing and policy normalization.

## Slice 3: Benchmark Harness Upgrade (Quality Metrics)

Objective:

1. Extend benchmark output beyond token savings.

Design:

1. Add retrieval precision/recall/F1 metrics.
2. Keep deterministic fixture-driven benchmark corpus.
3. Include pass/fail guard thresholds for quality.

Red tests:

1. Summary schema includes precision/recall/F1.
2. Guard fails when compression improves but quality drops below threshold.
3. Per-case metrics are deterministic on fixture corpus.

Green implementation:

1. Add metric calculator and summary fields.
2. Update benchmark scripts/artifacts.

Refactor:

1. Remove duplicate metric math and centralize in harness.

## Slice 4: Explainability Payloads

Objective:

1. Show why segments were kept/dropped.

Design:

1. Emit contribution breakdown:
   - relevance
   - redundancy penalty
   - position/importance prior
2. Return concise reason codes in outputs.

Red tests:

1. Explain payload present and schema-validated.
2. Score decomposition sums to expected final score range.

Green implementation:

1. Add explanation fields to compression outputs and optional MCP responses.

Refactor:

1. Move explanation formatting into utility module.

## Slice 5: External Adapter Layer

Objective:

1. Make Token Saver easier to plug into popular retrieval stacks.

Design:

1. Add lightweight adapter wrappers for external retriever pipelines.
2. Keep adapters optional and dependency-light.

Red tests:

1. Adapter contract tests using mocked external docs/query objects.
2. Adapter round-trip output compatibility checks.

Green implementation:

1. Implement minimal wrappers and usage examples.

Refactor:

1. Keep adapter modules isolated from core compressor internals.

## Checkpoint Commands

Per slice:

1. `python -m pytest --no-cov <new-and-affected-tests> -q`
2. `python -m ruff check src tests scripts skills`
3. `python -m black --check src tests scripts skills`
4. `python -m pytest --no-cov tests/test_token_savings.py tests/test_query_guided_compression.py tests/test_benchmark_harness.py tests/test_skill_scripts.py -q`

Release gate:

1. `python -m pytest --no-cov -q`

## Initial Backlog (Execution Order)

1. Slice 3 (benchmark quality metrics) - fastest path to objective proof.
2. Slice 1 (pipeline abstraction) - highest architecture ROI.
3. Slice 2 (structured controls) - user-facing control improvements.
4. Slice 4 (explainability) - trust and debugging.
5. Slice 5 (adapters) - ecosystem adoption.
