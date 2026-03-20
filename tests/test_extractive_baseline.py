from src.extractive_baseline import ExtractiveCompressor


def test_extractive_compressor_prioritizes_query_relevant_sentences():
    text = (
        "Prompt caching reduces input cost when the stable prefix is reused exactly. "
        "Bananas are yellow and unrelated to this topic. "
        "Exact prefix matching is required for cache hits across repeated requests. "
        "Another irrelevant sentence talks about vacation weather."
    )

    result = ExtractiveCompressor().compress_text(
        text,
        query="How do cache hits work with stable prefixes?",
        target_tokens=18,
    )

    compressed = result["compressed_text"].lower()
    assert "stable prefix" in compressed
    assert "cache hits" in compressed or "cache hit" in compressed
    assert result["compressed_tokens"] < result["original_tokens"]
    assert result["selected_sentence_count"] >= 1
