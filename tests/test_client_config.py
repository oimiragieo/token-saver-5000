"""
Tests for src/client_config.py (ClientConfig class, SessionConfigStore).

TDD test suite for v0.11.0 token optimization.

ClientConfig:
- from_model(model_id, context_window_tokens=None) factory method
- Stores model_id, context_window_tokens, recommended_skeleton_ratio
- Computes recommended ratio scaled to context window size
- Explicit context_window_tokens overrides model lookup

SessionConfigStore:
- Dict-based store keyed by session_id
- set_config(session_id, config) / get_config(session_id) -> ClientConfig
- Unset sessions return the default (balanced) config
- Isolated per session_id
"""

import pytest

from src.client_config import ClientConfig, SessionConfigStore
from src.constants import KNOWN_MODEL_CONTEXT_WINDOWS


# ============================================================================
# ClientConfig.from_model — known models
# ============================================================================


def test_configure_for_known_claude_model():
    """claude-opus-4-6 is in KNOWN_MODEL_CONTEXT_WINDOWS -> 200K tokens."""
    cfg = ClientConfig.from_model("claude-opus-4-6")
    assert cfg.model_id == "claude-opus-4-6"
    assert cfg.context_window_tokens == 200_000


def test_configure_for_known_gpt_model():
    """gpt-4o is in KNOWN_MODEL_CONTEXT_WINDOWS -> 128K tokens."""
    cfg = ClientConfig.from_model("gpt-4o")
    assert cfg.model_id == "gpt-4o"
    assert cfg.context_window_tokens == 128_000


def test_configure_for_unknown_model_uses_default():
    """Unknown model falls back to KNOWN_MODEL_CONTEXT_WINDOWS['default'] = 100K."""
    cfg = ClientConfig.from_model("some-future-model-xyz")
    assert cfg.context_window_tokens == KNOWN_MODEL_CONTEXT_WINDOWS["default"]


# ============================================================================
# Explicit context_window_tokens override
# ============================================================================


def test_configure_with_explicit_context_window():
    """Explicit context_window_tokens is stored as-is."""
    cfg = ClientConfig.from_model("my-model", context_window_tokens=500_000)
    assert cfg.context_window_tokens == 500_000


def test_explicit_window_overrides_model_lookup():
    """Explicit context_window_tokens wins over the model lookup table."""
    cfg = ClientConfig.from_model("claude-opus-4-6", context_window_tokens=50_000)
    assert cfg.context_window_tokens == 50_000  # Not 200K


# ============================================================================
# Recommended skeleton ratio scaling
# ============================================================================


def test_skeleton_ratio_is_in_valid_range():
    """Recommended ratio is always in [0.0, 1.0]."""
    for model_id in ["claude-opus-4-6", "gpt-4o", "gpt-4", "gemini-2.0-pro"]:
        cfg = ClientConfig.from_model(model_id)
        assert (
            0.0 <= cfg.recommended_skeleton_ratio <= 1.0
        ), f"ratio out of range for {model_id}: {cfg.recommended_skeleton_ratio}"


def test_skeleton_ratio_scales_with_window_size():
    """Larger context window -> higher (or equal) recommended ratio than smaller window."""
    small_cfg = ClientConfig.from_model("gpt-4")  # 8K
    large_cfg = ClientConfig.from_model("gemini-2.0-pro")  # 2M
    assert large_cfg.recommended_skeleton_ratio >= small_cfg.recommended_skeleton_ratio


def test_1m_window_ratio():
    """1M context window should recommend a ratio in 0.4-0.6 range."""
    cfg = ClientConfig.from_model("claude-opus-4-6[1m]")
    assert 0.4 <= cfg.recommended_skeleton_ratio <= 0.6


def test_50k_window_ratio():
    """50K explicit context window should recommend a low ratio (0.05-0.15)."""
    cfg = ClientConfig.from_model("some-model", context_window_tokens=50_000)
    assert 0.05 <= cfg.recommended_skeleton_ratio <= 0.15


def test_get_recommended_ratio():
    """get_recommended_ratio() returns the same float as .recommended_skeleton_ratio."""
    cfg = ClientConfig.from_model("claude-sonnet-4-6")
    assert cfg.get_recommended_ratio() == cfg.recommended_skeleton_ratio
    assert isinstance(cfg.get_recommended_ratio(), float)


# ============================================================================
# Validation
# ============================================================================


def test_invalid_context_window_zero_rejected():
    """context_window_tokens=0 raises ValueError."""
    with pytest.raises(ValueError):
        ClientConfig.from_model("my-model", context_window_tokens=0)


def test_invalid_context_window_negative_rejected():
    """context_window_tokens=-1 raises ValueError."""
    with pytest.raises(ValueError):
        ClientConfig.from_model("my-model", context_window_tokens=-1)


# ============================================================================
# KNOWN_MODEL_CONTEXT_WINDOWS database
# ============================================================================


def test_model_database_has_defaults():
    """KNOWN_MODEL_CONTEXT_WINDOWS contains a 'default' key."""
    assert "default" in KNOWN_MODEL_CONTEXT_WINDOWS


def test_model_database_default_is_100k():
    """The 'default' entry is 100K tokens."""
    assert KNOWN_MODEL_CONTEXT_WINDOWS["default"] == 100_000


# ============================================================================
# SessionConfigStore
# ============================================================================


def test_session_config_persists():
    """Config set for a session can be retrieved."""
    store = SessionConfigStore()
    cfg = ClientConfig.from_model("gpt-4o")
    store.set_config("session-abc", cfg)
    retrieved = store.get_config("session-abc")
    assert retrieved.model_id == "gpt-4o"
    assert retrieved.context_window_tokens == 128_000


def test_session_config_isolated():
    """Different session_ids have different configs and don't interfere."""
    store = SessionConfigStore()
    cfg_a = ClientConfig.from_model("gpt-4o")
    cfg_b = ClientConfig.from_model("claude-opus-4-6")

    store.set_config("session-a", cfg_a)
    store.set_config("session-b", cfg_b)

    assert store.get_config("session-a").model_id == "gpt-4o"
    assert store.get_config("session-b").model_id == "claude-opus-4-6"


def test_get_config_returns_default_when_unset():
    """Retrieving config for an unknown session returns a default config."""
    store = SessionConfigStore()
    cfg = store.get_config("never-set-session")
    assert cfg is not None
    assert isinstance(cfg, ClientConfig)


def test_session_config_overwrite():
    """Setting a new config for an existing session overwrites the old one."""
    store = SessionConfigStore()
    store.set_config("s1", ClientConfig.from_model("gpt-4o"))
    store.set_config("s1", ClientConfig.from_model("claude-opus-4-6"))
    assert store.get_config("s1").model_id == "claude-opus-4-6"


def test_session_store_isolated_across_instances():
    """Two different SessionConfigStore instances do not share state."""
    store_a = SessionConfigStore()
    store_b = SessionConfigStore()

    store_a.set_config("shared", ClientConfig.from_model("gpt-4o"))
    cfg_b = store_b.get_config("shared")

    # store_b doesn't have this session — should return default, not gpt-4o
    assert cfg_b.model_id != "gpt-4o" or store_b.get_config("shared") is not store_a.get_config(
        "shared"
    )
