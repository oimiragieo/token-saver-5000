"""Query-aware extractive compression baseline."""

from __future__ import annotations

import re
from typing import Any

import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer

from .segment_compression_cache import SegmentCompressionCache


class ExtractiveCompressor:
    """Simple extractive compressor with optional query-aware ranking."""

    def __init__(self, *, cache: SegmentCompressionCache | None = None) -> None:
        self._cache = cache
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def _split_sentences(self, text: str) -> list[str]:
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
        return sentences or [text.strip()]

    def _compress_uncached(
        self,
        *,
        text: str,
        query: str | None,
        target_tokens: int | None,
    ) -> dict[str, Any]:
        sentences = self._split_sentences(text)
        original_tokens = self._count_tokens(text)
        effective_target = target_tokens or max(24, int(original_tokens * 0.35))
        if len(sentences) == 1:
            compressed_text = sentences[0]
            return {
                "compressed_text": compressed_text,
                "original_tokens": original_tokens,
                "compressed_tokens": self._count_tokens(compressed_text),
                "selected_sentence_count": 1,
            }

        corpus = list(sentences)
        if query:
            corpus.append(query)
        matrix = TfidfVectorizer(stop_words="english").fit_transform(corpus)
        sentence_vectors = matrix[: len(sentences)]
        query_vector = matrix[-1] if query else None

        scores: list[tuple[int, float]] = []
        for index, sentence in enumerate(sentences):
            score = 1.0 - (index / max(len(sentences), 1)) * 0.15
            sentence_tokens = self._count_tokens(sentence)
            if 6 <= sentence_tokens <= 40:
                score += 0.2
            if query_vector is not None:
                similarity = float(sentence_vectors[index].multiply(query_vector).sum())
                score += similarity * 2.0
            scores.append((index, score))

        selected: list[int] = []
        running_tokens = 0
        for index, _score in sorted(scores, key=lambda item: item[1], reverse=True):
            sentence = sentences[index]
            sentence_tokens = self._count_tokens(sentence)
            if selected and running_tokens + sentence_tokens > effective_target:
                continue
            selected.append(index)
            running_tokens += sentence_tokens
            if running_tokens >= effective_target:
                break
        if not selected:
            selected = [sorted(scores, key=lambda item: item[1], reverse=True)[0][0]]
        if query and len(selected) == 1:
            max_expanded_budget = max(effective_target + 12, effective_target * 2)
            for index, score in sorted(scores, key=lambda item: item[1], reverse=True):
                if index in selected or score <= 0.2:
                    continue
                sentence_tokens = self._count_tokens(sentences[index])
                if running_tokens + sentence_tokens <= max_expanded_budget:
                    selected.append(index)
                    running_tokens += sentence_tokens
                    break

        compressed_text = " ".join(sentences[index] for index in sorted(selected))
        return {
            "compressed_text": compressed_text,
            "original_tokens": original_tokens,
            "compressed_tokens": self._count_tokens(compressed_text),
            "selected_sentence_count": len(selected),
        }

    def compress_text(
        self,
        text: str,
        *,
        query: str | None = None,
        target_tokens: int | None = None,
    ) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("compress_text requires non-empty 'text'")

        def compute() -> dict[str, Any]:
            return self._compress_uncached(text=text, query=query, target_tokens=target_tokens)

        if self._cache is None:
            return compute()
        return self._cache.get_or_compute(
            segment_text=text,
            method="extractive",
            query=query,
            compute=compute,
        )
