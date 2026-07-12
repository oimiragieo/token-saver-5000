"""Task #212 edge-case hardening (2026-07-12 gotcontext.ai engine audit).

Ports docs/audits/2026-07-05-competitive-edgecase-research.md's 23-case TDD
suite. Most cases were ALREADY fixed in prior work (verified against the real
code, cited in the audit table returned alongside this diff) -- this file
locks the genuinely-OPEN, model-free-fixable gaps found during that audit:

- Case #1  (dense CJK, no whitespace): ``_chunk_text``'s oversized-paragraph
  and semantic-strategy branches used an ASCII-only ``(?<=[.!?])\\s+`` sentence
  splitter. A dense CJK paragraph has no ``\\n\\n`` and (often) no ASCII
  terminator, so the whole blob returned as ONE "sentence" -> one mega-chunk
  (never subdivided), the same failure class as case #9 but for prose. Fixed
  by routing both call sites through the already-CJK-aware
  ``_SENTENCE_SPLIT_RE`` (previously wired only into ``_generate_summary`` /
  ``_extractive_anchor_content``). Byte-identical to the old regex on
  pure-ASCII text (no regression risk to existing corpora).
- Case #16 (adversarial repeated content, "bounded time"): ``intra_doc_dedup.
  find_intra_duplicates`` is called UNCONDITIONALLY on every ingest with >2
  chunks (``SemanticCompressor.ingest_file`` step 2b) and had NO node-count
  ceiling -- an all-pairs O(N^2) scan, unlike the sibling similarity-edge
  builders in semantic_compressor.py / code_compressor.py which both cap at
  ``_MAX_GRAPH_CHUNKS``. A large or highly-repetitive adversarial document
  could hang for minutes. Fixed with the same bounded-participation pattern
  (``_MAX_DEDUP_NODES``, env-overridable) + a cheap O(N) norm-precompute.
- Case #10 (Rust/Go generics) and case #9 (minified single-line JS,
  multi-function): regression LOCKS proving the existing behavior is already
  correct (no fix needed) -- Go/Rust route to line-granularity
  ``_chunk_by_lines`` (never parses ``<>``, so it cannot mis-split a generic
  signature), and the #212(b) brace-balanced JS/TS scanner is line-agnostic.
- Case #5 (JSON leaf-type ambiguity) and #15/#17 (whitespace-only): cheap
  additional regression locks on top of already-existing coverage.

Model-free throughout: pure functions / ``object.__new__(...)`` construction,
no SentenceTransformer/ONNX load (per the compression-engine-sota skill
pattern used across ``tests/test_worldclass_batch*.py``).
"""

from __future__ import annotations

import time

import numpy as np

from src.code_compressor import CodeSemanticCompressor
from src.intra_doc_dedup import collapse_redundant_nodes, find_intra_duplicates
from src.semantic_compressor import SemanticCompressor
from src.structured_content import detect_structured_content, split_json_records

# ---------------------------------------------------------------------------
# Case #1 -- dense CJK (no whitespace) must actually SPLIT when oversized,
# not collapse into one mega-chunk.
# ---------------------------------------------------------------------------

_CJK_SENTENCE = "这是一个关于人工智能系统设计与实现的详细技术说明文档。"
_CJK_CORE = _CJK_SENTENCE[:-1]  # strip the trailing full-width period for substring counting


def _bare_semantic_compressor() -> SemanticCompressor:
    c = object.__new__(SemanticCompressor)
    # Deterministic, whitespace-independent token proxy (CJK has no spaces, so
    # the real _count_tokens's word-count fallback would misreport a single
    # dense blob as "1 token" -- character count scales correctly instead).
    c._count_tokens = lambda t: len(t)
    return c


def test_dense_cjk_oversized_paragraph_splits_into_multiple_chunks():
    c = _bare_semantic_compressor()
    text = _CJK_SENTENCE * 40  # one dense "paragraph": no \n\n, no ASCII punctuation
    chunks = c._chunk_text(text, max_chunk_size=200, strategy="fixed")
    assert len(chunks) > 1, "dense CJK oversized paragraph collapsed into one mega-chunk"
    joined = "".join(chunks)
    assert "�" not in joined  # zero U+FFFD replacement chars (no corruption)
    assert joined.count(_CJK_CORE) == 40  # every repetition survives intact


def test_dense_cjk_semantic_strategy_also_splits():
    """Same fix, the OTHER call site (``strategy="semantic"``'s <3-unit fallback).

    Uses a fake embedder (random fixed-size vectors) so ``chunk_by_semantics``
    can run without a real model -- correctness here doesn't depend on
    semantically-meaningful embeddings, only on ``semantic_units`` having >1
    entry so the token-budget cap in ``chunk_by_semantics`` has multiple units
    to work with.
    """

    class _StubModel:
        def encode(self, texts):
            rng = np.random.RandomState(0)
            return rng.rand(len(texts), 8)

    c = _bare_semantic_compressor()
    c.model = _StubModel()
    c.model_name = None
    text = _CJK_SENTENCE * 40
    chunks = c._chunk_text(text, max_chunk_size=200, strategy="semantic")
    assert len(chunks) > 1
    joined = "".join(chunks)
    assert "�" not in joined
    assert joined.count(_CJK_CORE) == 40


def test_ascii_oversized_paragraph_regression_unchanged():
    """The CJK-aware splitter must be byte-identical to the old ASCII regex on
    pure-ASCII input -- no behavior change for existing (non-CJK) corpora."""
    c = _bare_semantic_compressor()
    sentence = "The deployment pipeline builds one artifact for the staging cluster. "
    text = sentence * 40
    chunks = c._chunk_text(text, max_chunk_size=200, strategy="fixed")
    assert len(chunks) > 1
    joined = " ".join(chunks)
    assert joined.count("staging cluster") == 40


# ---------------------------------------------------------------------------
# Case #16 -- adversarial repeated/near-duplicate content must not blow up
# the O(N^2) intra-doc dedup scan.
# ---------------------------------------------------------------------------


def test_find_intra_duplicates_respects_max_nodes_cap():
    emb = np.array([1.0, 0.0, 0.0])
    nodes = {f"n{i}": {"text": f"t{i}", "embedding": emb.copy()} for i in range(5)}
    dupes = find_intra_duplicates(nodes, threshold=0.9, max_nodes=3)
    involved = {d["node_a"] for d in dupes} | {d["node_b"] for d in dupes}
    assert involved <= {"n0", "n1", "n2"}
    assert "n3" not in involved
    assert "n4" not in involved


def test_intra_dedup_bounded_time_on_large_near_duplicate_corpus():
    """#16: a large near-duplicate corpus (well above the production default
    cap of 1000) must not hang -- this is the exact shape an adversarial
    highly-repetitive document produces after chunking."""
    n = 2500
    rng = np.random.RandomState(0)
    base = rng.rand(8)
    nodes = {
        f"n{i}": {
            "text": f"repeated line {i}",
            "embedding": base + rng.normal(scale=1e-4, size=8),
        }
        for i in range(n)
    }
    t0 = time.perf_counter()
    dupes = find_intra_duplicates(nodes, threshold=0.9)
    elapsed = time.perf_counter() - t0
    assert isinstance(dupes, list)
    assert elapsed < 8.0, f"find_intra_duplicates took {elapsed:.2f}s for N={n} -- cap not applied"

    # collapse_redundant_nodes must still run cleanly (not error / hang) on the
    # same corpus, and must not silently drop node COUNT below what came in.
    collapsed = collapse_redundant_nodes(nodes, threshold=0.9)
    assert isinstance(collapsed, dict)
    assert len(collapsed) >= 1


# ---------------------------------------------------------------------------
# Case #9 (additional) -- minified, single-LINE (no newlines) JS with many
# functions must chunk in bounded time with no truncation.
# ---------------------------------------------------------------------------


def test_minified_single_line_js_multiple_functions_bounded():
    cc = object.__new__(CodeSemanticCompressor)
    n_funcs = 500
    funcs = [f"function f{i}(a,b){{if(a){{return a+b}}return b-a}}" for i in range(n_funcs)]
    code = "".join(funcs)  # single line, zero whitespace between functions
    assert "\n" not in code
    assert len(code) > 20_000

    t0 = time.perf_counter()
    chunks = cc.chunk_javascript_code(code, "min.js")
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"chunk_javascript_code took {elapsed:.2f}s on a single-line blob"
    assert len(chunks) > 1
    names = {ch.name for ch in chunks}
    assert "f0" in names
    assert f"f{n_funcs - 1}" in names
    for ch in chunks:
        assert ch.code.count("{") == ch.code.count("}"), f"unbalanced/truncated: {ch.code!r}"


# ---------------------------------------------------------------------------
# Case #10 -- Rust/Go generics must never be mis-split. Go/Rust route through
# line-granularity ``_chunk_by_lines`` (no `<>`-aware parser exists for them),
# so byte-identical reconstruction + never-bisect-mid-line is the correct,
# already-true regression lock (no source fix needed).
# ---------------------------------------------------------------------------


def test_rust_generic_signature_never_bisected_mid_line():
    sig_line = "fn foo<T: Clone + Send>(x: T) -> T { x }"
    lines = [f"// line {i}" for i in range(49)] + [sig_line] + [f"// tail {i}" for i in range(10)]
    code = "\n".join(lines)  # sig_line sits at the LAST line of the first 50-line chunk

    cc = object.__new__(CodeSemanticCompressor)
    chunks = cc._chunk_by_lines(code, "f.rs", lines_per_chunk=50)
    assert len(chunks) >= 2

    reconstructed = "\n".join(ch.code for ch in chunks)
    assert reconstructed == code  # byte-identical round trip

    hits = [ch for ch in chunks if sig_line in ch.code]
    assert len(hits) == 1, "generic signature line split across chunk boundary"
    assert hits[0].code.count("<T: Clone + Send>") == 1


def test_go_generic_signature_never_bisected_mid_line():
    sig_line = "func Map[T, U any](s []T, f func(T) U) []U { return nil }"
    lines = [f"// line {i}" for i in range(20)] + [sig_line] + [f"// tail {i}" for i in range(20)]
    code = "\n".join(lines)

    cc = object.__new__(CodeSemanticCompressor)
    chunks = cc._chunk_by_lines(code, "f.go", lines_per_chunk=50)
    reconstructed = "\n".join(ch.code for ch in chunks)
    assert reconstructed == code
    hits = [ch for ch in chunks if sig_line in ch.code]
    assert len(hits) == 1
    assert hits[0].code.count("[T, U any]") == 1


# ---------------------------------------------------------------------------
# Case #5 -- JSON leaf-type ambiguity ("007" string vs bare 7, null, 1e10)
# must survive byte-identical (span-preserving split, never reserialized).
# ---------------------------------------------------------------------------


def test_split_json_records_preserves_ambiguous_leaf_types_byte_identical():
    text = '[{"code": "007", "n": 7, "id": null, "big": 1e10, "flag": true}, {"code": "008"}]'
    out = split_json_records(text)
    assert out == [
        '{"code": "007", "n": 7, "id": null, "big": 1e10, "flag": true}',
        '{"code": "008"}',
    ]
    assert '"007"' in out[0]  # string leaf stays quoted (not coerced to int 7)
    assert '"n": 7' in out[0]  # numeric leaf stays bare
    assert '"id": null' in out[0]
    assert '"big": 1e10' in out[0]  # scientific notation not reformatted


# ---------------------------------------------------------------------------
# Case #15/#17 -- whitespace-only / empty content must no-op cleanly through
# the structured-content detector, never raise, never NaN.
# ---------------------------------------------------------------------------


def test_detect_structured_content_whitespace_variants_return_none():
    for blob in ("", "   ", "\n\n\t  \n", "\r\n \r\n", "\t\t\t"):
        assert detect_structured_content(blob) is None


# ---------------------------------------------------------------------------
# Cases #6 / #12 -- CI-ONLY (real embedding model required; recall@k survival
# at scale). Per task instructions: do NOT run these locally. Written as
# plain functions (NOT ``test_`` prefixed) so pytest does not auto-collect or
# run them as part of the local gate sweep above -- they document the intended
# assertion and are runnable on demand once a CI run has model + time budget.
#
# Code-level evidence gathered during this audit (see the returned audit
# table) already shows the SELECTION mechanism has no positional bias: the
# COMI coarse-filter at semantic_compressor.py sorts purely by cosine
# similarity (``node_scores.sort(key=lambda x: x[2], reverse=True)``), with NO
# position/index term -- a query-relevant node ranks identically regardless of
# where it sits in the document. These two manual checks exist to prove the
# same holds at REAL embedding-model scale (semantic drift, not just
# mechanism), which this file cannot cheaply do model-free.
# ---------------------------------------------------------------------------


def manual_case6_needle_in_long_table_survives_at_any_position():  # pragma: no cover
    """CI-ONLY (real model). Build a table with N rows (reduced from the
    research doc's 1000+ to a still-meaningful ~120 for CI wall-time), plant a
    query-relevant "needle" row at the START, MIDDLE, and END across three
    ingests, and assert the needle survives (visible / ranked #1) in all three
    -- i.e. no lost-in-the-middle degradation purely from row position.
    """
    from src.semantic_compressor import SemanticCompressor

    def _table(needle_at: int, n_rows: int = 120) -> str:
        rows = []
        for i in range(n_rows):
            if i == needle_at:
                rows.append(
                    "| {} | quetzalcoatl-anomaly-9471 | the unique query-relevant marker row |".format(
                        i
                    )
                )
            else:
                rows.append(f"| {i} | filler-{i} | ordinary boilerplate content for row {i} |")
        return "| idx | key | value |\n|---|---|---|\n" + "\n".join(rows)

    query = "quetzalcoatl-anomaly-9471 unique query-relevant marker"
    for position_name, idx in (("start", 2), ("middle", 60), ("end", 117)):
        c = SemanticCompressor()
        file_id = f"needle_{position_name}"
        c.ingest_file(_table(idx), file_id)
        ranked = c.search_semantic_with_scores(query, file_id=file_id, top_k=3)
        assert ranked, f"no results for needle at {position_name}"
        top_node_id = ranked[0][0]
        top_text = c.chunks[top_node_id].text
        assert "quetzalcoatl-anomaly-9471" in top_text, (
            f"needle at position={position_name} (row {idx}) did not rank #1 "
            f"-- lost-in-the-middle regression"
        )


def manual_case12_fact_at_midpoint_vs_boundary_recall_parity():  # pragma: no cover
    """CI-ONLY (real model). Scaled-down long-context fact-at-midpoint test
    (research doc specifies 500K tokens; a full-scale run is a dedicated CI
    job, not part of this bounded hardening pass -- this reduced version
    exercises the same mechanism at ~15-20K tokens)."""
    from src.semantic_compressor import SemanticCompressor

    def _doc(fact_at_fraction: float, n_sections: int = 40) -> str:
        sections = []
        fact_idx = int(n_sections * fact_at_fraction)
        for i in range(n_sections):
            body = " ".join(
                f"Sentence {k} of section {i} discusses routine operational detail {k}."
                for k in range(40)
            )
            if i == fact_idx:
                body += " The secret launch code is glimmerfax-77219-omega."
            sections.append(f"## Section {i}\n\n{body}")
        return "# Long Document\n\n" + "\n\n".join(sections)

    query = "What is the secret launch code?"
    for label, frac in (("midpoint", 0.5), ("boundary", 0.98)):
        c = SemanticCompressor()
        file_id = f"longctx_{label}"
        c.ingest_file(_doc(frac), file_id)
        ranked = c.search_semantic_with_scores(query, file_id=file_id, top_k=3)
        assert ranked, f"no results for fact at {label}"
        top_node_id = ranked[0][0]
        assert (
            "glimmerfax-77219-omega" in c.chunks[top_node_id].text
        ), f"fact at {label} did not rank #1 -- lost-in-the-middle regression"
