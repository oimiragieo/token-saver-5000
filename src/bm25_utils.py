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
