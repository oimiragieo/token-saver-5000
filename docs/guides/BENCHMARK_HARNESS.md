# Benchmark Harness

This guide documents the fixed-corpus benchmark harness used to track token-savings regressions.

## Goals

- Keep a stable benchmark corpus in version control.
- Track compression ratio and token savings on every run.
- Fail fast when results fall below agreed golden thresholds.

## Files

- Corpus fixture: `tests/fixtures/benchmark_corpus.json`
- Harness logic: `src/benchmark_harness.py`
- Runner script: `scripts/benchmarks/run_benchmarks.py`
- Guard script: `scripts/benchmarks/check_benchmark_guard.py`
- Tests: `tests/test_benchmark_harness.py`

## Run

```bash
python scripts/benchmarks/run_benchmarks.py
python scripts/benchmarks/run_benchmarks.py --mode query_guided
python scripts/benchmarks/run_benchmarks.py --compare baseline,query_guided,evidence_aware
python scripts/benchmarks/check_benchmark_guard.py
```

Run a single case:

```bash
python scripts/benchmarks/run_benchmarks.py --case medium_architecture
```

Allow failures but still produce report:

```bash
python scripts/benchmarks/run_benchmarks.py --allow-failures
```

## Output

Default report path:

`artifacts/benchmarks/latest.json`

The report includes:

- per-case token counts
- per-case compression ratio and savings
- pass/fail target checks
- aggregate averages
- benchmark mode (`baseline`, `query_guided`, `evidence_aware`)
- CI guard thresholds from `artifacts/benchmarks/golden_thresholds.json`

## TDD Workflow

1. Add or adjust corpus case thresholds in `tests/fixtures/benchmark_corpus.json`.
2. Add failing test in `tests/test_benchmark_harness.py`.
3. Implement code changes.
4. Run:
   - `python -m pytest tests/test_benchmark_harness.py -q -o addopts=""`
   - `python scripts/benchmarks/run_benchmarks.py`
5. Commit code + docs + updated benchmark artifact decision.
