# TDD Execution Spec (2026-02-15)

This spec defines the enforceable Red-Green-Refactor workflow for the remaining modernization work in this repository.

## Research Basis

Primary references used:

1. Martin Fowler, Test-Driven Development (updated 2023-12-11): https://martinfowler.com/bliki/TestDrivenDevelopment.html
2. Kent Beck, Canon TDD (2023-12-11): https://tidyfirst.substack.com/p/canon-tdd
3. Rafique and Misic, meta-analysis on TDD quality/productivity (IEEE TSE, 2013): https://doi.org/10.1109/TSE.2012.28
4. LLM4TDD (arXiv:2312.04687): https://arxiv.org/abs/2312.04687
5. Tests as Prompt benchmark (arXiv:2505.09027): https://arxiv.org/abs/2505.09027
6. Class-level test-driven generation study (arXiv:2602.03557): https://arxiv.org/abs/2602.03557

Practical interpretation for this repo:

1. TDD remains design-first, not test-after.
2. Keep loops tiny: one behavior slice per test cycle.
3. Refactor is mandatory in every cycle.
4. Preserve measurable outcomes (token savings + correctness), not just pass tests.

## Global Gates (Every Commit)

1. `Red`: add failing tests first.
2. `Green`: implement minimal behavior to pass.
3. `Refactor`: simplify without behavior drift.
4. `Quality Gate`: run lint + format + targeted tests + regression ring.
5. `Docs Gate`: update docs/changelog in the same commit.
6. `SCM Gate`: clean `git status`, atomic commit message, push.

## Feature Backlog and TDD Slices

## Slice A: App-layer contract hardening (in progress)

Goal:

1. Remove implicit dict-shape coupling in app service boundaries.
2. Make contract drift fail-fast with clear mismatch messages.

Scope:

1. `src/semantic_modulator/app/lifecycle_service.py`
2. `src/semantic_modulator/app/progress_service.py`
3. `src/semantic_modulator/app/persistence_orchestration_service.py`
4. `src/semantic_modulator/app/tool_profile_service.py`
5. `src/semantic_modulator/app/router_binding.py`

Required tests:

1. Keyset declaration contract for each request envelope.
2. Negative validation tests (extra key / missing key).
3. Existing behavior regression tests stay green.

Acceptance criteria:

1. No server behavior regression in lifecycle, profile bootstrap, routing, persistence, or progress helpers.
2. Contract errors use canonical key mismatch format.

## Slice B: Benchmark and quality guard tightening

Goal:

1. Ensure token-saving claims are reproducible and regression-safe.

Scope:

1. Keep and extend fixed benchmark corpus checks.
2. Add round-trip structure checks for JSON and TOON output paths.
3. Promote TOON only when guard thresholds pass.

Required tests:

1. Golden metric schema and threshold checks.
2. Structure correctness on JSON/TOON/auto selection.
3. Retrieval-quality proxy non-regression tests.

Acceptance criteria:

1. Benchmark report artifacts produced deterministically.
2. Guard fails loudly when savings improve but correctness degrades.

## Slice C: Skill parity and portability guard

Goal:

1. Keep `skills/token-saver-context-compression` self-contained and transportable.

Scope:

1. No runtime dependency on files outside the skill directory.
2. CLI scripts support `--output-format {json,toon,auto}` consistently.

Required tests:

1. Script import/runtime isolation checks.
2. CLI contract tests for format selection and fallback.

Acceptance criteria:

1. Skill can be copied into another project and run without MCP.

## Command Checkpoints

Run in order per slice:

1. `python -m pytest --no-cov <targeted-test-files> -q`
2. `python -m ruff check src tests scripts skills`
3. `python -m black --check src tests scripts skills`
4. `python -m pytest --no-cov tests/test_server_unit.py tests/test_server_runtime_wrapper.py tests/test_server_wiring_contract.py tests/test_mcp_tooling_gateway.py tests/test_context_service.py tests/test_lifecycle_service.py tests/test_progress_service.py tests/test_persistence_orchestration_service.py tests/test_tool_profile_service.py tests/test_router_binding.py -q`
5. `python -m pytest --no-cov tests/test_token_savings.py tests/test_query_guided_compression.py tests/test_benchmark_harness.py tests/test_skill_scripts.py -q`

Optional stress ring:

1. `python -m pytest --no-cov tests/test_async_operations.py tests/test_performance.py tests/test_chaos_engineering.py tests/test_integration_workflows.py -q`

## Commit and Push Policy

1. Commit message format:
   - `refactor(app): <service contract scope>`
   - `test(app): <contract coverage scope>`
   - `docs(tdd): update execution spec and checkpoints`
2. Push branch directly to `main` only after all gates pass.
