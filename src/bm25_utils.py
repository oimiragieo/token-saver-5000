"""
BM25 utility functions for semantic search.

Extracted from src/handlers/docs_handlers.py to allow reuse by
both docs_handlers and semantic_compressor without duplication.

BM25Okapi parameters (k1=1.5, b=0.75) match the docs_handlers defaults
that have been validated on the gotcontext documentation corpus.

v1.34.35 (F11 Path C): added as shared helper for BM25+RRF hybrid retrieval.
"""

from __future__ import annotations

import math
import re

# Minimum token length for prefix-match stemming (v1.34.33 hotfix; audit P2-5).
# Terms shorter than this require exact match to avoid false positives.
#
# Audit P2-5: a 6-char threshold made "python" (6) prefix-match "pythonic" — a
# different word, not a morphological variant — inflating BM25 scores. Raising
# the bar to 8 keeps useful long-stem matches ("authentication" → variants) while
# making 6-7 char common words ("python", "import", "return", "config") require
# exact match. "auth" (4) → exact only; "authentication" (14) → prefix matches.
_BM25_PREFIX_MATCH_MIN_LEN = 8

# BM25Okapi default hyperparameters (Cormack et al. 2009).
_BM25_K1 = 1.5
_BM25_B = 0.75


def bm25_tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric + underscore tokenizer.

    Identical to docs_handlers._tokenize — extracted here for shared use.
    """
    return re.findall(r"[a-z0-9_]+", text.lower())


def bm25_idf(term: str, tokenized_docs: list[list[str]]) -> float:
    """Smoothed inverse document frequency (BM25Okapi variant).

    idf(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
    where N = total docs, df(t) = docs containing term t.
    Smoothing (+0.5) prevents zero IDF for universal terms.
    """
    docs_with_term = sum(1 for doc in tokenized_docs if term in doc)
    return math.log(1 + (len(tokenized_docs) - docs_with_term + 0.5) / (docs_with_term + 0.5))


def bm25_term_freq_with_stemming(term: str, doc: list[str]) -> int:
    """Count occurrences of ``term`` in ``doc``, with prefix-match stemming.

    For terms >= _BM25_PREFIX_MATCH_MIN_LEN chars, any token sharing the
    first _BM25_PREFIX_MATCH_MIN_LEN characters is counted as a match.
    Short terms require exact match to avoid false positives.
    """
    exact = doc.count(term)
    if len(term) < _BM25_PREFIX_MATCH_MIN_LEN:
        return exact
    prefix = term[:_BM25_PREFIX_MATCH_MIN_LEN]
    return sum(1 for tok in doc if tok == term or tok.startswith(prefix))


_GATE_DIGIT_TOKEN_RE = re.compile(r"[A-Za-z0-9_]*\d[A-Za-z0-9_]*")
_GATE_CAMELCASE_RE = re.compile(r"[a-z][A-Z]")

# A quoted-phrase signal = a BALANCED PAIR of DOUBLE quotes (straight " or smart
# “ ”) wrapping >=1 word char. DOUBLE QUOTES ONLY -- by design.
#
# Single quotes / apostrophes are DELIBERATELY EXCLUDED (blocker-1 final fix,
# 2026-07-08 codex re-gate, 3rd pass). A single quote is inherently ambiguous
# with a contraction/possessive apostrophe: '...' spanning an internal
# apostrophe ("'don't retry'") and a lone smart apostrophe with no close both
# defeat any boundary heuristic. Rather than patch another edge, we ELIMINATE
# the class: only double quotes signal a phrase (the standard search-engine
# convention). A contraction (what's / don't / it's) can therefore NEVER open
# the gate via the quoted-phrase path -- which is exactly the paraphrase
# regression the gate exists to prevent. Legit single-quoted lexical signal is
# still covered by the identifier / numeric predicates in
# ``query_has_lexical_shape`` and by ``bm25_top1_is_discriminative``, so nothing
# real is lost -- only the ambiguity.
_GATE_QUOTED_PHRASE_RES = (
    re.compile(r'"[^"]*\w[^"]*"'),  # straight double: "..."
    re.compile("“[^”]*\\w[^”]*”"),  # smart double: “...”
)


def _query_has_quoted_phrase(query: str) -> bool:
    """True only when ``query`` contains a balanced DOUBLE-quote pair wrapping
    >=1 word char. Single quotes / apostrophes NEVER count (too ambiguous with
    contractions -- see ``_GATE_QUOTED_PHRASE_RES``)."""
    return any(rx.search(query) for rx in _GATE_QUOTED_PHRASE_RES)


def query_has_lexical_shape(query: str) -> bool:
    """F11 Path G gate predicate #1 (design memo idea #1, 'gated fusion',
    ``docs/audits/2026-07-08-f11-retrieval-fusion-ideas.md`` section 2).

    True when the raw query text contains a digit-bearing token, an
    underscore_identifier / camelCase-shaped token, or a quoted phrase (a
    BALANCED quote-delimiter pair -- a contraction apostrophe does NOT count).
    These are exactly the query shapes the 2026-07-08 gate-data measurement
    showed BM25 wins or ties-at-rank-1 on (bare_numeric, identifier,
    quoted_phrase classes) -- the classes gated fusion should always fuse for.

    Pure string predicate: no model, no candidate-set dependency, O(len(query)).
    """
    if _query_has_quoted_phrase(query):
        return True
    for token in re.findall(r"[A-Za-z0-9_]+", query):
        if _GATE_DIGIT_TOKEN_RE.fullmatch(token):
            return True
        if "_" in token:
            return True
        if _GATE_CAMELCASE_RE.search(token):
            return True
    return False


def bm25_top1_is_discriminative(
    query: str,
    top1_text: str,
    all_candidate_texts: list[str],
    *,
    top1_score: float,
    idf_tau: float,
    score_floor: float,
) -> bool:
    """F11 Path G gate predicate #2 (design memo idea #1): did BM25's top-1
    candidate match at least one query term whose IDF *within the candidate
    set* clears ``idf_tau`` (i.e. the term does NOT appear in most sections
    -- kills the design memo's "system"/"calling" false-signal case, where
    IDF ~= 0), OR is the raw BM25 top-1 score itself already >= ``score_floor``
    (an unambiguously strong lexical hit regardless of IDF spread)?

    Args:
        query: the raw search query.
        top1_text: raw text of the BM25-ranked top-1 candidate node.
        all_candidate_texts: raw texts of every candidate in the file_id
            scope (the SAME set BM25 was scored against -- IDF must be
            computed over this set, never the whole corpus; council patch P1).
        top1_score: the top-1 candidate's raw BM25 score.
        idf_tau: minimum smoothed IDF (see ``bm25_idf``) for a matched term
            to count as "discriminative".
        score_floor: absolute BM25 score that short-circuits the IDF check.

    Returns:
        True if the gate's match-quality predicate is satisfied.
    """
    if top1_score >= score_floor:
        return True
    query_terms = bm25_tokenize(query)
    if not query_terms:
        return False
    tokenized_docs = [bm25_tokenize(t) for t in all_candidate_texts]
    top1_tokens = bm25_tokenize(top1_text)
    # Blocker-2 fix (2026-07-08 codex review): judge "did top-1 match a query
    # term" with the SAME prefix-stem matching the BM25 scorer uses
    # (``bm25_term_freq_with_stemming``, see ``bm25_scores``). An exact
    # token-membership check would MISS a real BM25 hit -- e.g. query
    # ``authentication`` scoring against a doc's ``authenticate ...`` via the
    # >=8-char prefix stem -- and wrongly CLOSE the gate on a valid lexical win.
    return any(
        bm25_term_freq_with_stemming(term, top1_tokens) > 0
        and bm25_idf(term, tokenized_docs) >= idf_tau
        for term in query_terms
    )


def bm25_scores(
    query: str,
    texts: list[str],
    k1: float = _BM25_K1,
    b: float = _BM25_B,
) -> list[float]:
    """Compute BM25Okapi scores for a query against a list of text strings.

    Args:
        query: Search query string.
        texts: List of document text strings to score.
        k1: Term-saturation parameter (default 1.5, Okapi BM25 standard).
        b: Length normalization parameter (default 0.75).

    Returns:
        List of BM25 scores parallel to ``texts``. Scores are non-negative.
        Zero score means the document has no query-term overlap.
    """
    query_terms = bm25_tokenize(query)
    if not query_terms or not texts:
        return [0.0] * len(texts)

    tokenized_docs = [bm25_tokenize(t) for t in texts]
    avgdl = sum(len(doc) for doc in tokenized_docs) / max(1, len(tokenized_docs))
    scores: list[float] = []

    for doc in tokenized_docs:
        if not doc:
            scores.append(0.0)
            continue
        score = 0.0
        doc_len = len(doc)
        for term in query_terms:
            freq = bm25_term_freq_with_stemming(term, doc)
            if freq == 0:
                continue
            numerator = freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * doc_len / max(avgdl, 1))
            score += bm25_idf(term, tokenized_docs) * numerator / denominator
        scores.append(score)

    return scores
