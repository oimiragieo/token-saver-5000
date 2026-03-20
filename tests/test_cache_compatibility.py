from src.cache_compatibility import assess_cache_compatibility


def test_assess_cache_compatibility_for_gemini_cli_accepts_stats_visibility():
    assessment = assess_cache_compatibility(
        model="gemini-3.1-pro-preview",
        harness="gemini_cli",
        raw_usage_available=False,
        cli_stats_available=True,
    )

    assert assessment["provider"] == "google"
    assert assessment["support_level"] == "supported"
    assert "/stats" in assessment["telemetry"]["acceptable_sources"]
    assert "cachedContentTokenCount" in assessment["telemetry"]["primary_fields"]


def test_assess_cache_compatibility_for_codex_flags_missing_visibility():
    assessment = assess_cache_compatibility(
        model="gpt-5.3-codex",
        harness="codex_cli",
        raw_usage_available=False,
        cli_stats_available=False,
    )

    assert assessment["provider"] == "openai"
    assert assessment["support_level"] == "conditional"
    assert assessment["telemetry"]["visibility_status"] == "missing"
    assert any("prompt_cache_key" in item for item in assessment["recommendations"])
