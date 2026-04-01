"""
Token Optimization Handlers (v0.11.0)

MCP tool handlers for token estimation, client configuration,
and compression profile management. These tools help MCP clients
optimize token usage by configuring compression parameters for
their specific model and context window.

All handlers are model-agnostic -- they benefit any MCP client,
not just Claude Code.
"""

import json
from typing import Any, Dict

from ..token_estimation import TokenEstimator
from ..client_config import ClientConfig, SessionConfigStore
from ..compression_profiles import (
    ProfileManager,
    get_profile,
    list_profiles,
)


# Module-level singletons for session state
_session_config_store = SessionConfigStore()
_profile_manager = ProfileManager()
_token_estimator = TokenEstimator()


def get_session_config_store() -> SessionConfigStore:
    """Get the module-level session config store."""
    return _session_config_store


def get_profile_manager() -> ProfileManager:
    """Get the module-level profile manager."""
    return _profile_manager


async def handle_estimate_tokens(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    """Handle estimate_tokens tool call.

    Returns multi-method token estimates for the given text.
    """
    text = args.get("text", "")
    if not isinstance(text, str):
        return json.dumps({"error": "text must be a string"})

    estimates = _token_estimator.estimate_all(text)
    return json.dumps(
        {
            "status": "success",
            "text_length": len(text),
            "estimates": estimates,
            "methods": {
                "accurate": "tiktoken cl100k_base (or fallback to bytes/4)",
                "estimated": "len(text) // 4 (matches Claude Code fast estimation)",
                "json_estimated": "len(text) // 2 (for JSON-heavy content)",
                "bytes": "exact UTF-8 byte count",
            },
        }
    )


async def handle_configure_for_client(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    """Handle configure_for_client tool call.

    Configures compression parameters for a specific LLM model.
    """
    model_id = args.get("model_id", "default")
    context_window_tokens = args.get("context_window_tokens")
    session_id = args.get("session_id", "default")

    try:
        config = ClientConfig.from_model(
            model_id=model_id,
            context_window_tokens=context_window_tokens,
        )
    except ValueError as e:
        return json.dumps({"error": str(e)})

    _session_config_store.set_config(session_id, config)

    return json.dumps(
        {
            "status": "success",
            "configuration": {
                "model_id": config.model_id,
                "context_window_tokens": config.context_window_tokens,
                "recommended_skeleton_ratio": round(config.recommended_skeleton_ratio, 4),
                "provider": config.provider,
                "preferred_output_format": config.preferred_output_format,
                "truncation_strategy": config.truncation_strategy,
                "cache_stable_ordering": config.cache_stable_ordering,
            },
            "message": (
                f"Configured for {config.model_id} "
                f"({config.context_window_tokens:,} token context window). "
                f"Recommended skeleton ratio: {config.recommended_skeleton_ratio:.2%}."
            ),
        }
    )


async def handle_set_compression_profile(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    """Handle set_compression_profile tool call.

    Sets a named compression profile for the session.
    """
    profile_name = args.get("profile_name")
    session_id = args.get("session_id", "default")

    if not profile_name:
        return json.dumps(
            {
                "error": "profile_name is required",
                "available_profiles": list_profiles(),
            }
        )

    try:
        _profile_manager.set_profile(session_id, profile_name)
        profile = get_profile(profile_name)
    except ValueError as e:
        return json.dumps({"error": str(e), "available_profiles": list_profiles()})

    return json.dumps(
        {
            "status": "success",
            "active_profile": {
                "name": profile.name,
                "skeleton_ratio": profile.skeleton_ratio,
                "fidelity": profile.fidelity,
                "chunk_size": profile.chunk_size,
                "description": profile.description,
            },
            "message": (
                f"Compression profile set to '{profile.name}': {profile.description}. "
                f"Explicit parameters in subsequent tool calls will override these defaults."
            ),
        }
    )


async def handle_get_compression_profile(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    """Handle get_compression_profile tool call.

    Returns the active compression profile for the session.
    """
    session_id = args.get("session_id", "default")
    profile = _profile_manager.get_profile(session_id)

    return json.dumps(
        {
            "status": "success",
            "active_profile": {
                "name": profile.name,
                "skeleton_ratio": profile.skeleton_ratio,
                "fidelity": profile.fidelity,
                "chunk_size": profile.chunk_size,
                "description": profile.description,
            },
            "available_profiles": list_profiles(),
        }
    )
