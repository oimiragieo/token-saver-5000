# TDD Modernization Plan (v0.10.x)

This document is the execution plan for simplifying the architecture while preserving or improving measured token savings.

## Objectives

1. Preserve or improve observed savings:
   - Medium sample: `122 -> 49` tokens (~`59.8%`)
   - Large sample: `6346 -> 316` tokens (~`95.0%`)
2. Reduce accidental complexity and drift.
3. Introduce query-aware and evidence-aware compression upgrades with measurable gains.
4. Package a production-ready Claude Skill for token-saving workflows.

## Success Metrics

1. Compression effectiveness:
   - `compression_ratio`
   - `token_savings_percent`
2. Quality proxies:
   - Retrieval hit quality (`top-k` relevance consistency)
   - Evidence coverage rate (answer-supporting spans retained)
3. Runtime:
   - Ingest latency p50/p95
   - Search and modulation latency p50/p95
4. Stability:
   - Core test suite pass rate
   - Stress test pass rate under concurrent ingest/search/modulate

## Research Inputs (Primary Sources)

1. Query-guided compression:
   - https://arxiv.org/abs/2406.02376
2. Redundancy-aware coarse-to-fine selection (MIG):
   - https://arxiv.org/html/2602.01719v1
3. Evidence-guided compression loops for RAG:
   - https://arxiv.org/abs/2506.05167
4. Adaptive/Selective RAG compression:
   - https://arxiv.org/abs/2507.05633
5. Official Claude Skill docs:
   - https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview
   - https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices

## Delivery Strategy

Follow strict Red-Green-Refactor in each milestone:

1. Red: add or tighten tests first.
2. Green: implement minimal change to pass.
3. Refactor: reduce complexity while preserving behavior.
4. Verify: run format/lint/tests and stress checks.
5. Document: update README/changelog/guide docs in same PR.
6. Commit: atomic commit with clear scope.

## Milestone Plan

### Week 1: Stabilize and De-Drift

Scope:

1. Version truth:
   - Align `pyproject.toml`, `src/constants.py`, `src/__init__.py`, README.
2. Tool-count truth:
   - Align docs and tests with routed tool count.
3. Node identity normalization:
   - Centralize node ID parsing in shared utility.
   - Update handler hotspots to use shared parser.
4. Comment/doc drift cleanup in tests.

TDD tasks:

1. Add/update tests for:
   - Node ID parsing (`_n`, `::`, fallback).
   - Tool count assertions.
2. Validate no behavior regression in:
   - `ingest_context`, `modulate_region`, `list_documents`.

Checkpoint commands:

```bash
python -m pytest tests/test_mcp_routing.py -q
python -m pytest tests/test_experimental_handlers.py -q
python -m pytest tests/test_token_savings.py -q
```

Quality gates:

```bash
python -m ruff check src tests
python -m black --check src tests
git status --short
```

Commit pattern:

```bash
git add -A
git commit -m "chore(core): de-drift versions/tool counts and normalize node identity parsing"
```

### Week 2: Benchmark Harness + Golden Metrics

Scope:

1. Add fixed benchmark corpus:
   - Small, medium, large, noisy, multi-topic, code-heavy.
2. Add benchmark runner with reproducible output:
   - JSON + markdown report.
3. Add golden thresholds:
   - Minimum compression and quality proxy thresholds.

TDD tasks:

1. Add tests for benchmark schema validity and deterministic output shape.
2. Add regression tests for key KPI thresholds.

Checkpoint commands:

```bash
python scripts/benchmarks/run_benchmarks.py --output artifacts/benchmarks/latest.json
python -m pytest tests/test_benchmark_harness.py -q
python -m pytest tests/test_token_savings.py -q
```

Quality gates:

```bash
python -m ruff check src tests scripts
python -m black --check src tests scripts
git status --short
```

Commit pattern:

```bash
git add -A
git commit -m "feat(bench): add fixed corpus and golden token-savings benchmark harness"
```

### Week 3: Query-Guided Compressor v1

Scope:

1. Add query-conditioned scoring:
   - relevance(query, node) + graph importance - redundancy penalty.
2. Add optional API path:
   - `read_skeleton_query_aware` or query-aware mode flag.
3. Keep baseline PageRank-only mode for A/B.

TDD tasks:

1. Unit tests for score components:
   - relevance monotonicity
   - redundancy penalty behavior
2. Integration test:
   - query-aware mode should reduce tokens at equal/better retrieval utility.

Checkpoint commands:

```bash
python -m pytest tests/test_query_guided_compression.py -q
python -m pytest tests/test_semantic_fidelity.py -q
python scripts/benchmarks/run_benchmarks.py --compare baseline,query_guided
```

Quality gates:

```bash
python -m ruff check src tests scripts
python -m black --check src tests scripts
git status --short
```

Commit pattern:

```bash
git add -A
git commit -m "feat(compression): add query-guided skeleton allocation with redundancy penalty"
```

### Week 4: Evidence-Aware Compression Loop

Scope:

1. Add evidence classifier/scorer per node relative to question + draft answer.
2. Add insufficiency loop:
   - If evidence confidence below threshold, retrieve additional nodes.
3. Integrate with blind-spot detection path.

TDD tasks:

1. Unit tests:
   - evidence score bounds, thresholding.
2. Integration tests:
   - insufficient evidence triggers additional retrieval.
   - evidence recall improves over baseline in synthetic QA set.

Checkpoint commands:

```bash
python -m pytest tests/test_evidence_aware_loop.py -q
python -m pytest tests/test_detection_handlers.py -q
python scripts/benchmarks/run_benchmarks.py --compare query_guided,evidence_aware
```

Quality gates and commit:

```bash
python -m ruff check src tests scripts
python -m black --check src tests scripts
git status --short
git add -A
git commit -m "feat(quality): add evidence-aware compression loop and insufficiency-triggered retrieval"
```

### Week 5: Claude Skill Packaging

Scope:

1. Create custom skill package:
   - `SKILL.md`
   - scripts:
     - `compress_context.py`
     - `profile_tokens.py`
     - `validate_evidence.py`
2. Define fallback policy:
   - compress-first, escalate to RAW when confidence/evidence is low.
3. Add usage examples and integration tests.

TDD tasks:

1. Script unit tests + contract tests for output schema.
2. End-to-end skill invocation tests (local).

Checkpoint commands:

```bash
python -m pytest tests/test_skill_scripts.py -q
python scripts/skills/profile_tokens.py --help
python scripts/skills/compress_context.py --help
python scripts/skills/validate_evidence.py --help
```

Quality gates and commit:

```bash
python -m ruff check src tests scripts skills
python -m black --check src tests scripts skills
git status --short
git add -A
git commit -m "feat(skill): package token-saving workflow as Claude Skill with evidence-aware fallback"
```

### Week 6: R&D Branch Decision (Soft/Latent Compression)

Scope:

1. Evaluate soft compression prototype in isolated experimental module.
2. Compare accuracy/cost vs query+evidence pipeline.
3. Decide production path based on benchmark deltas.

Go/No-Go Criteria:

1. Go if:
   - >= 10% additional token reduction at non-inferior quality proxies.
   - No major latency regressions beyond defined budget.
2. No-Go if:
   - Gains are marginal or unstable.

## Stress Test Plan (Continuous)

Run at end of each milestone:

1. Concurrent ingest/search/modulate mixed load.
2. Repeated refresh/delete/version operations.
3. Memory and latency tracking for p50/p95.

Recommended command suite:

```bash
python -m pytest tests/test_async_operations.py -q
python -m pytest tests/test_performance.py -q
python -m pytest tests/test_chaos_engineering.py -q
python -m pytest tests/test_integration_workflows.py -q
```

## Documentation Update Rules

In every milestone commit:

1. Update affected doc sections in `README.md`.
2. Update architecture references if behavior changed.
3. Update changelog with scope + measurable impacts.
4. Keep tool count/version references synchronized.

## Branching and Commit Discipline

1. One milestone per branch.
2. One coherent concern per commit where practical.
3. Every commit must include:
   - tests updated/passing for changed behavior
   - docs updated for user-visible changes
   - lint/format clean

## Risks and Mitigations

1. Risk: Optimization regresses quality.
   - Mitigation: golden benchmarks + evidence recall thresholds.
2. Risk: Tool-surface simplification breaks compatibility.
   - Mitigation: stable/advanced profile flags and deprecation period.
3. Risk: Benchmark overfitting.
   - Mitigation: maintain mixed corpus and hidden holdout set.

## Immediate Next Actions

1. Complete Week 1 code/test/doc updates.
2. Add benchmark harness scaffold (Week 2 starter).
3. Run stress sanity pass and publish baseline benchmark artifact.
