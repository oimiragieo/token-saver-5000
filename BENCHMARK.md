# Compression Benchmark: Reproducibility Spec

This document describes how to reproduce the compression numbers we cite for
`token-saver-5000` (the engine behind gotcontext.ai). Anyone who clones this
repository can run the same command against the same corpus and get the same
numbers we report here.

## TL;DR

```bash
git clone https://github.com/<org>/token-saver-5000.git
cd token-saver-5000
pip install -r requirements.txt
pip install -e .
PYTHONPATH=. python scripts/run_public_benchmark.py
```

Expected output (measured on 2026-07-12, package version 0.11.0):

```
Case        Orig tokens  Skeleton tokens    Ratio   Savings
------------------------------------------------------------
small               554              220    2.52x    60.29%
medium             2348              426    5.51x    81.86%
large              7173              513   13.98x    92.85%
code               3390              239   14.18x    92.95%
```

## Methodology

- **What is measured:** each corpus file is ingested with
  `SemanticCompressor.ingest_file(text, file_id)`, then compressed with
  `SemanticCompressor.read_skeleton(file_id)` at default settings (no query,
  no fidelity override). Compression ratio is `original_tokens / skeleton_tokens`,
  where "tokens" is a simple whitespace word count (`len(text.split())`). This
  is the same proxy metric used by our internal GTM regression-lock suite
  (`tests/test_gtm_benchmarks.py`); it needs no external tokenizer dependency
  and is trivially reproducible on any machine.
- **Embedding tier:** the default text encoder is `BAAI/bge-small-en-v1.5`
  (SBERT/STANDARD tier), resolved via `DEFAULT_TEXT_MODEL` in `src/constants.py`.
  This is a roughly 130MB model downloaded once from Hugging Face on first
  run (see "First-run cost" below). Setting `EMBEDDING_TIER=onnx` or
  `EMBEDDING_TIER=tfidf` switches to a lighter-weight tier and will change the
  numbers (typically a lower ratio). That is a real accuracy/speed tradeoff,
  not a bug.
- **A fresh `SemanticCompressor()` instance is used per document** so no
  document's PageRank graph or importance ranking can leak into another's.
- **What is NOT measured here:** semantic fidelity, meaning whether the
  compressed skeleton still answers questions about the source correctly.
  That is a separate, harder claim covered by our quality-gate suite
  (`tests/test_quality_gate.py`; see the `compression-quality-eval` skill in
  the parent gotcontext.ai monorepo). A high ratio alone does not prove the
  output is useful. Pair any ratio claim with a fidelity check before citing
  it as a quality win.

## Determinism (reproducible = same numbers every run)

**Confirmed deterministic.** We ran the benchmark 3 times in separate process
invocations, and again under `PYTHONHASHSEED=0`, `PYTHONHASHSEED=1`, and
`PYTHONHASHSEED=42`. Every run produced byte-identical output:

```
avg ratio=6.39x | avg savings=83.5%
medium_architecture: ratio=4.97x savings=79.9%
large_repetitive_tech: ratio=7.80x savings=87.2%
```

(This is the output of the separate internal CI harness,
`scripts/benchmarks/run_benchmarks.py`. See "Two benchmark surfaces" below.
It is quoted here purely as determinism evidence, not as the headline number.)

This determinism is not incidental. An earlier version of the engine used
`list(set(entities))` when extracting key entities for a compression summary,
which iterates a Python `set` in hash-order: a value that varies per-process
under `PYTHONHASHSEED` randomization (Python's default since 3.3). That made
the "Key entities:" line in the skeleton output non-reproducible across
processes. It was fixed by switching to `list(dict.fromkeys(...))`, which
preserves first-seen order deterministically. The fix is regression-locked in
`tests/test_output_determinism.py` (`test_skeleton_deterministic_across_hashseed`),
which spawns a subprocess with a different `PYTHONHASHSEED` and diffs the
output. If you ever see non-reproducible output from this engine, that test
file is the first place to look.

## Corpus and provenance

The corpus lives at `benchmarks/corpus/` and is committed to this repository.
There is no external download and no proprietary or customer data: every
file is synthetic documentation authored specifically for this benchmark.

| File | Size (words) | Description |
|------|--------------:|-------------|
| `small.txt` | 554 | Short configuration guide. Minimal internal redundancy: this is the honest worst case. Small documents can show weak compression, or even expansion, because the skeleton format has fixed per-document overhead (headers, node IDs, metadata). |
| `medium.txt` | 2,348 | A synthetic architecture specification ("Distributed Task Queue System") with headings, a table of contents, and moderate structural redundancy. |
| `large.txt` | 7,173 | A synthetic REST API reference ("TaskFlow API") with repetitive per-endpoint sections (CRUD patterns repeat across Users, Projects, Tasks, Comments). This is the kind of document that compresses best, because repeated structure is exactly what semantic compression collapses. |
| `code/*.py` | 3,390 (first 5 files) | Synthetic Python source files (`api.py`, `auth.py`, `cache.py`, `config.py`, `database.py`) representing a small backend service, run through the AST-aware code compressor. |

Manifest and per-file descriptions: `benchmarks/corpus/manifest.json`.

## One-command reproduction

```bash
# from the repo root
cd token-saver-5000
pip install -r requirements.txt
pip install -e .

# run the benchmark
PYTHONPATH=. python scripts/run_public_benchmark.py

# optionally write results to JSON
PYTHONPATH=. python scripts/run_public_benchmark.py --output results.json
```

First run only: the `BAAI/bge-small-en-v1.5` embedding model (roughly 130MB)
downloads automatically from Hugging Face on first use and is cached locally
(`~/.cache/huggingface` by default, or `$HF_HOME` if set). Subsequent runs
are fast and fully offline.

## Results (measured 2026-07-12)

| Doc | Original tokens | Skeleton tokens | Ratio | Token savings |
|-----|----------------:|-----------------:|------:|---------------:|
| small | 554 | 220 | 2.52x | 60.29% |
| medium | 2,348 | 426 | 5.51x | 81.86% |
| large | 7,173 | 513 | 13.98x | 92.85% |
| code (5 files) | 3,390 | 239 | 14.18x | 92.95% |

**Ratio depends on document size, structural redundancy, and embedding
fidelity tier. Never quote a single number as universal.** The honest
summary from this run:

- Small, low-redundancy documents compress the least (2.52x here). Below
  roughly 100 to 200 tokens, the skeleton format's fixed overhead can make
  output larger than input. This is a known, by-design limit, not a bug (see
  "Compression Behavior" in the main `CLAUDE.md`).
- Medium documents (low thousands of tokens) with real structure compress
  meaningfully (5.51x here).
- Large, repetitive documents compress the most (13.98x to 14.18x here). This
  is the regime our GTM materials cite when they say roughly 9.5x on large
  docs. That specific figure came from an earlier engine version measured on
  this same `large.txt` file (see `tests/test_gtm_benchmarks.py` for the
  locked regression floor of 9.0x, set a margin below the measured value so
  future regressions are caught without the guard being brittle). The engine
  has since improved (13.98x measured today). Floors are a lower bound, not a
  ceiling, and we do not walk them up every time the number improves, so the
  regression lock stays stable.

## Two benchmark surfaces (know which one you're citing)

This repository actually has two related but distinct benchmark surfaces.
Both are legitimate. They answer slightly different questions and use
different token-counting methods, so do not mix numbers between them.

1. **This public spec** (`scripts/run_public_benchmark.py` over
   `benchmarks/corpus/`): word-count based, matches the GTM claim suite
   (`tests/test_gtm_benchmarks.py`), answers "how much does compression save
   on a small, medium, large, or code document."
2. **The CI regression-lock harness** (`scripts/benchmarks/run_benchmarks.py`
   over `tests/fixtures/benchmark_corpus.json`, gated by
   `scripts/benchmarks/check_benchmark_guard.py` against
   `artifacts/benchmarks/golden_thresholds.json`): uses the engine's own
   internal token accounting (`SkeletonResponse.total_tokens` and
   `.skeleton_tokens`), runs in three modes (`baseline`, `query_guided`,
   `evidence_aware`) with quality metrics (precision, recall, F1 at k) for
   the query-aware modes, and exists to catch regressions in CI, not to
   produce a marketing number. Its current baseline: avg ratio 6.39x, avg
   savings 83.5% across its 2 synthetic fixture cases (`medium_architecture`,
   `large_repetitive_tech`).

If you want the number we use in customer-facing materials, use surface 1
(this document). If you are contributing to the engine and want to check you
have not regressed compression quality, use surface 2 and run
`scripts/benchmarks/check_benchmark_guard.py` before opening a PR.

## How to cite this benchmark

"Measured with `token-saver-5000` v0.11.0 on the committed
`benchmarks/corpus/` corpus via `scripts/run_public_benchmark.py`, run
locally on `<date>`. Ratio depends on document size and structure. See
`BENCHMARK.md` in the repository for full methodology and reproduction
steps." Link to this file rather than repeating a bare ratio out of context.

## Limitations and honest caveats

- This benchmark measures compression ratio only, not answer quality or
  semantic fidelity. A skeleton that compresses 14x but drops an answer a
  downstream query needed is not a win. Pair this with the fidelity gate
  (`tests/test_quality_gate.py`) before making a quality claim.
- The corpus is small (4 documents) and synthetic. It is representative of
  common document shapes (config guide, architecture spec, API reference,
  small codebase) but is not a large-scale, crowd-sourced benchmark. Numbers
  will vary on your own real documents. Run this same script against your
  own corpus for a number specific to your workload.
- Results depend on the embedding tier active in your environment
  (`EMBEDDING_TIER` env var, or plan-based tier mapping in hosted
  deployments). The numbers above are the STANDARD/SBERT tier (the default
  for a local, unauthenticated run of this open package). A `tfidf` or
  `onnx` tier will produce different (typically lower-fidelity, faster)
  results.
- We did not benchmark competing tools (LLMLingua, Selective-Context, etc.)
  in this document. See `docs/audits/` in the parent gotcontext.ai monorepo
  for competitor analysis and comparison pages, kept separate so this spec
  stays a pure, single-tool reproduction guide.

## What's out of scope here

- Surfacing these numbers on the public `/benchmarks` page or `llms.txt` on
  gotcontext.ai. That is a follow-up; the web surface already exists as a
  skeleton per the product roadmap, and wiring this corpus into it is
  separate work.
- A large-scale, community-submitted benchmark repository ("UserBenchmark for
  AI"). That is Product Line #2 in the gotcontext.ai roadmap, a separate and
  much larger effort than this single-engine reproducibility spec.
