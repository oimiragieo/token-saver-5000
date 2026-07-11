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


# ---------------------------------------------------------------------------
# v1.7.2 follow-up — mini and flash variants added for per-model benchmarking.
# Pricing tiers let the dashboard show that compression saves more dollars on
# expensive models even when the token count saved is identical.
# ---------------------------------------------------------------------------


def test_gemini_flash_is_cheaper_than_pro():
    pro = get_provider_profile("gemini-3.1-pro-preview")
    flash = get_provider_profile("gemini-3.1-flash")
    assert flash.provider == "google"
    assert flash.input_cost_per_million < pro.input_cost_per_million


def test_gemini_auto_priced_between_pro_and_flash():
    pro = get_provider_profile("gemini-3.1-pro-preview")
    flash = get_provider_profile("gemini-3.1-flash")
    auto = get_provider_profile("gemini-auto")
    assert auto.provider == "google"
    assert flash.input_cost_per_million <= auto.input_cost_per_million <= pro.input_cost_per_million


def test_gemini_flash_alias_resolves():
    # Current flash flagship is Gemini 3.5 Flash (2026-05-19).
    assert get_provider_profile("gemini-flash").model == "gemini-3.5-flash"
    assert get_provider_profile("gemini-3.5").model == "gemini-3.5-flash"
    # gemini-2.5-flash is its own profile; the old "gemini-3.1-flash" id was
    # really 2.5-Flash-priced and is kept resolvable for back-compat.
    assert get_provider_profile("gemini-2.5-flash").model == "gemini-2.5-flash"
    assert get_provider_profile("gemini-3.1-flash").model == "gemini-2.5-flash"


def test_gpt5_mini_cheaper_than_full():
    full = get_provider_profile("gpt-5.4")
    mini = get_provider_profile("gpt-5.4-mini")
    assert mini.provider == "openai"
    assert mini.input_cost_per_million < full.input_cost_per_million
    assert mini.supports_prompt_cache_key is True


def test_gpt41_mini_cheaper_than_full():
    full = get_provider_profile("gpt-4.1")
    mini = get_provider_profile("gpt-4.1-mini")
    assert mini.provider == "openai"
    assert mini.input_cost_per_million < full.input_cost_per_million


def test_mini_aliases_resolve():
    assert get_provider_profile("gpt-5-mini").model == "gpt-5.4-mini"
    assert get_provider_profile("gpt-4o-mini").model == "gpt-4.1-mini"


# ---------------------------------------------------------------------------
# Claude Opus 4.7 — flagship model, 1M-token context.
# ---------------------------------------------------------------------------


def test_opus_4_7_is_registered():
    profile = get_provider_profile("claude-opus-4.7")
    assert profile.provider == "anthropic"
    assert profile.model == "claude-opus-4.7"
    # Opus dropped to $5/$25 at the 4.5 release; the 4.7 profile carries the
    # current rate, not the legacy $15/$75.
    assert profile.input_cost_per_million == 5.0
    assert profile.output_cost_per_million == 25.0
    assert profile.context_window == 1_000_000


def test_opus_4_7_aliases_resolve():
    # Unversioned + unversioned-major resolve to the CURRENT Opus (4.8).
    assert get_provider_profile("claude-opus").model == "claude-opus-4.8"
    assert get_provider_profile("claude-opus-4").model == "claude-opus-4.8"
    # Hyphenated 4.7 (MCP _meta.model convention in some clients) stays pinned.
    assert get_provider_profile("claude-opus-4-7").model == "claude-opus-4.7"


def test_opus_4_6_still_resolves_with_hyphen_alias():
    """Pinned 4.6 users keep working."""
    assert get_provider_profile("claude-opus-4-6").model == "claude-opus-4.6"
    assert get_provider_profile("claude-opus-4.6").model == "claude-opus-4.6"


# ---------------------------------------------------------------------------
# GPT-5.6 family (Sol/Terra/Luna) — OpenAI GA 2026-07-09 (#278).
# ---------------------------------------------------------------------------


def test_gpt_5_6_family_registered():
    sol = get_provider_profile("gpt-5.6-sol")
    terra = get_provider_profile("gpt-5.6-terra")
    luna = get_provider_profile("gpt-5.6-luna")
    for p in (sol, terra, luna):
        assert p.provider == "openai"
        assert p.cache_read_field == "cached_tokens"
        assert p.context_window == 1_050_000
        assert p.supports_prompt_cache_key is True
    # Verified short-context standard rates (openai.com 2026-07-11).
    assert (sol.input_cost_per_million, sol.output_cost_per_million) == (5.0, 30.0)
    assert (terra.input_cost_per_million, terra.output_cost_per_million) == (2.5, 15.0)
    assert (luna.input_cost_per_million, luna.output_cost_per_million) == (1.0, 6.0)
    # Tier ordering: Sol > Terra > Luna on both input and output price.
    assert luna.input_cost_per_million < terra.input_cost_per_million < sol.input_cost_per_million
    assert (
        luna.output_cost_per_million < terra.output_cost_per_million < sol.output_cost_per_million
    )


def test_gpt_5_6_alias_resolves_to_sol():
    # OpenAI's own alias: unversioned gpt-5.6 routes to Sol.
    assert get_provider_profile("gpt-5.6").model == "gpt-5.6-sol"
    assert get_provider_profile("gpt-5-6").model == "gpt-5.6-sol"
    assert get_provider_profile("gpt-5-6-terra").model == "gpt-5.6-terra"
    assert get_provider_profile("gpt-5-6-luna").model == "gpt-5.6-luna"
