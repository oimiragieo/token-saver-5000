import pytest

from src.provider_profiles import get_provider_profile, list_provider_profiles


def test_get_provider_profile_returns_known_model():
    profile = get_provider_profile("claude-sonnet-4.6")

    assert profile.model == "claude-sonnet-4.6"
    assert profile.provider == "anthropic"
    assert profile.cache_read_field == "cache_read_input_tokens"
    assert profile.input_cost_per_million > 0


def test_provider_profile_supports_alias_lookup():
    profile = get_provider_profile("gpt-5.4")

    assert profile.provider == "openai"
    assert profile.cache_read_field == "cached_tokens"


def test_provider_profile_supports_gemini_3_alias_lookup():
    profile = get_provider_profile("gemini-3")

    assert profile.provider == "google"
    assert profile.model == "gemini-3.1-pro-preview"
    assert profile.cache_read_field == "cachedContentTokenCount"


def test_provider_profile_supports_codex_alias_lookup():
    profile = get_provider_profile("gpt-5.3-codex")

    assert profile.provider == "openai"
    assert profile.model == "gpt-5.4"
    assert profile.cache_read_field == "cached_tokens"
    assert profile.supports_prompt_cache_key is True


def test_provider_profile_exposes_cache_threshold_guidance():
    profile = get_provider_profile("gpt-5.4")

    assert profile.minimum_cacheable_tokens >= 1024
    assert profile.cache_token_increment > 0


def test_unknown_model_raises_value_error():
    with pytest.raises(ValueError, match="Unknown model"):
        get_provider_profile("totally-unknown-model")


def test_list_provider_profiles_includes_multiple_providers():
    profiles = list_provider_profiles()
    providers = {profile["provider"] for profile in profiles}

    assert {"anthropic", "openai", "google"} <= providers
