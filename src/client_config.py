"""
Client and model configuration for context-window-aware compression.

Maps model IDs to context window sizes and computes recommended skeleton
ratios scaled to the available context window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from src.constants import KNOWN_MODEL_COMPRESSION_TRIGGERS, KNOWN_MODEL_CONTEXT_WINDOWS


def _detect_provider(model_id: str) -> str:
    """Infer the AI provider from a model identifier string.

    Args:
        model_id: Model identifier, e.g. ``"claude-opus-4-6"`` or ``"gpt-4o"``.

    Returns:
        One of ``"anthropic"``, ``"google"``, ``"openai"``, or ``"unknown"``.
    """
    model_lower = model_id.lower()
    if model_lower.startswith("claude") or model_lower.startswith("haiku"):
        return "anthropic"
    if model_lower.startswith("gemini"):
        return "google"
    if any(model_lower.startswith(p) for p in ("gpt", "o1", "o3", "o4", "codex")):
        return "openai"
    if model_lower.startswith("groq"):
        return "groq"
    if model_lower.startswith("grok"):
        return "xai"
    if any(model_lower.startswith(p) for p in ("local", "ollama")):
        return "local"
    return "unknown"


def get_recommended_ratio(context_window_tokens: int, compression_trigger: float = 0.80) -> float:
    """Compute a recommended skeleton ratio for a given context window size.

    Scaling intent:
      - Small windows (~8K-50K)    -> low ratio  (0.05 – 0.15) to maximise compression
      - Medium windows (~100K-200K) -> mid ratio (0.19 – 0.30)
      - Large windows  (~1M-2M)    -> high ratio (0.40 – 0.60) to preserve detail

    Uses a log-linear scale anchored at 50K→0.10 and 1M→0.50.
    When compression_trigger < 0.80 (more aggressive client like Gemini CLI),
    the ratio is reduced proportionally so more aggressive clients receive
    a more compressed skeleton.

    Args:
        context_window_tokens: Context window size in tokens.
        compression_trigger: Fraction of context window at which the client
            starts compressing. Lower values = more aggressive client.
            Default 0.80 (reference baseline).

    Returns:
        Float in [0.01, 0.90] representing the recommended skeleton ratio.
    """
    # Log-linear formula derived from two anchors:
    #   50_000 tokens -> 0.10  (low-end: strip aggressively)
    #   1_000_000 tokens -> 0.50  (1M: preserve most detail)
    _A = 0.40 / (math.log(1_000_000) - math.log(50_000))
    _B = 0.10 - _A * math.log(50_000)

    tokens = max(1, context_window_tokens)
    ratio = _A * math.log(tokens) + _B

    # Apply compression trigger adjustment: more aggressive clients get lower ratios
    if compression_trigger < 0.80:
        ratio = ratio * (compression_trigger / 0.80)

    return round(max(0.01, min(0.90, ratio)), 4)


@dataclass
class ClientConfig:
    """Configuration for a specific client / model pairing.

    Attributes:
        model_id: Identifier string for the model.
        context_window_tokens: Available context window in tokens.
        recommended_skeleton_ratio: Suggested skeleton ratio for this window size.
        compression_trigger: Fraction of context window at which the client starts
            compressing (0.50 for Gemini, 0.93 for Claude Code, 1.0 for GPT models).
        provider: Detected provider string — one of ``"anthropic"``, ``"google"``,
            ``"openai"``, or ``"unknown"``.
        preferred_output_format: Preferred serialisation format hint
            (``"toon"`` or ``"json"``).
        truncation_strategy: Default truncation mode for this provider
            (``"paginate"``, ``"proportional"``, or ``"head"``).
        cache_stable_ordering: Whether to apply cache-stable key ordering to
            responses destined for this client.
    """

    model_id: str
    context_window_tokens: int
    recommended_skeleton_ratio: float
    compression_trigger: float = 0.80
    provider: str = "unknown"
    preferred_output_format: str = "json"
    truncation_strategy: str = "paginate"
    cache_stable_ordering: bool = True

    @classmethod
    def from_model(
        cls,
        model_id: str,
        context_window_tokens: Optional[int] = None,
    ) -> "ClientConfig":
        """Create a ClientConfig for *model_id*, optionally overriding window size.

        Args:
            model_id: Model identifier (e.g. "claude-opus-4-6").
            context_window_tokens: Explicit window size. When provided and > 0, this
                overrides the lookup table. Raises ValueError if <= 0.

        Returns:
            ClientConfig instance with computed recommended_skeleton_ratio.

        Raises:
            ValueError: If context_window_tokens is provided but <= 0.
        """
        if context_window_tokens is not None:
            if context_window_tokens <= 0:
                raise ValueError(f"context_window_tokens must be > 0, got {context_window_tokens}")
            window = context_window_tokens
        else:
            window = KNOWN_MODEL_CONTEXT_WINDOWS.get(
                model_id, KNOWN_MODEL_CONTEXT_WINDOWS["default"]
            )

        trigger = KNOWN_MODEL_COMPRESSION_TRIGGERS.get(
            model_id, KNOWN_MODEL_COMPRESSION_TRIGGERS["default"]
        )
        ratio = get_recommended_ratio(window, compression_trigger=trigger)

        provider = _detect_provider(model_id)

        # Set provider-specific format and truncation hints.
        if provider == "anthropic":
            preferred_output_format = "toon"
            truncation_strategy = "paginate"
        elif provider == "google":
            preferred_output_format = "json"
            truncation_strategy = "proportional"
        elif provider == "openai":
            preferred_output_format = "toon"
            truncation_strategy = "head"
        else:
            preferred_output_format = "json"
            truncation_strategy = "paginate"

        return cls(
            model_id=model_id,
            context_window_tokens=window,
            recommended_skeleton_ratio=ratio,
            compression_trigger=trigger,
            provider=provider,
            preferred_output_format=preferred_output_format,
            truncation_strategy=truncation_strategy,
            cache_stable_ordering=True,
        )

    def get_recommended_ratio(self) -> float:
        """Return the recommended skeleton ratio for this configuration.

        Returns:
            Float in [0.01, 0.90].
        """
        return self.recommended_skeleton_ratio


class SessionConfigStore:
    """Dict-backed store mapping session IDs to ClientConfig instances.

    Each instance maintains its own isolated state — no shared class-level storage.
    Unknown sessions return a default ClientConfig rather than None, so callers
    never need to guard against None.
    """

    def __init__(self) -> None:
        self._store: dict[str, ClientConfig] = {}

    def set_config(self, session_id: str, config: ClientConfig) -> None:
        """Associate *config* with *session_id*, overwriting any previous value.

        Args:
            session_id: Unique session identifier.
            config: ClientConfig to store.
        """
        self._store[session_id] = config

    def get_config(self, session_id: str) -> ClientConfig:
        """Retrieve the config for *session_id*, or a default config if not set.

        Args:
            session_id: Unique session identifier.

        Returns:
            Stored ClientConfig or a default ClientConfig if the session is unknown.
        """
        if session_id in self._store:
            return self._store[session_id]
        return ClientConfig.from_model("default")
