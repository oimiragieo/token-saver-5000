from src.segment_compression_cache import SegmentCompressionCache


def test_segment_cache_reuses_cached_result_for_same_query_and_method():
    cache = SegmentCompressionCache()
    calls = {"count": 0}

    def compute():
        calls["count"] += 1
        return {"compressed_text": "cached result"}

    first = cache.get_or_compute(
        segment_text="Stable prefix guidance",
        method="extractive",
        query="cache hits",
        compute=compute,
    )
    second = cache.get_or_compute(
        segment_text="Stable prefix guidance",
        method="extractive",
        query="cache hits",
        compute=compute,
    )

    assert first == second
    assert calls["count"] == 1
    assert cache.stats()["hits"] == 1
    assert cache.stats()["entries"] == 1
