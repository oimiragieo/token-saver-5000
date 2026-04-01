"""
Named compression profile presets for Semantic Modulator.

Profiles bundle skeleton_ratio, fidelity level, and chunk size into a single
named choice. Explicit per-call parameters always override profile defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Optional

from src.constants import COMPRESSION_PROFILES, DEFAULT_COMPRESSION_PROFILE


@dataclass
class CompressionProfile:
    """A named preset bundling compression parameters.

    Attributes:
        name: Profile identifier (e.g. "balanced").
        skeleton_ratio: Fraction of graph nodes to retain.
        fidelity: Fidelity level string (e.g. "STRUCTURE").
        chunk_size: Maximum tokens per semantic chunk.
        description: Human-readable description of the profile's use-case.
    """

    name: str
    skeleton_ratio: float
    fidelity: str
    chunk_size: int
    description: str


def _build_profile(name: str) -> CompressionProfile:
    """Construct a CompressionProfile from the constants registry.

    Args:
        name: Profile name key in COMPRESSION_PROFILES.

    Returns:
        CompressionProfile instance.
    """
    spec = COMPRESSION_PROFILES[name]
    return CompressionProfile(
        name=name,
        skeleton_ratio=spec["skeleton_ratio"],
        fidelity=spec["fidelity"],
        chunk_size=spec["chunk_size"],
        description=spec["description"],
    )


def get_profile(name: str) -> CompressionProfile:
    """Return the CompressionProfile for *name*.

    Args:
        name: Profile name (e.g. "minimal", "balanced", "full").

    Returns:
        Matching CompressionProfile.

    Raises:
        ValueError: If *name* is not a known profile, with a message listing
            valid names.
    """
    if name not in COMPRESSION_PROFILES:
        valid = sorted(COMPRESSION_PROFILES.keys())
        raise ValueError(
            f"Unknown compression profile {name!r}. " f"Available profiles: {', '.join(valid)}"
        )
    return _build_profile(name)


def get_default_profile() -> CompressionProfile:
    """Return the default compression profile (balanced).

    Returns:
        CompressionProfile for DEFAULT_COMPRESSION_PROFILE.
    """
    return _build_profile(DEFAULT_COMPRESSION_PROFILE)


def list_profiles() -> list[str]:
    """Return a sorted list of all available profile names.

    Returns:
        Sorted list of profile name strings.
    """
    return sorted(COMPRESSION_PROFILES.keys())


class ProfileManager:
    """Dict-backed store mapping session IDs to compression profile names.

    Each instance maintains its own isolated state. Unknown sessions fall back
    to the default profile rather than raising an error.
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def set_profile(self, session_id: str, profile_name: str) -> None:
        """Associate *profile_name* with *session_id*.

        Args:
            session_id: Unique session identifier.
            profile_name: Name of the compression profile to use.

        Raises:
            ValueError: If *profile_name* is not a known profile.
        """
        # Validate — will raise ValueError for unknown names
        get_profile(profile_name)
        self._store[session_id] = profile_name

    def get_profile(self, session_id: str) -> CompressionProfile:
        """Return the compression profile for *session_id*.

        Args:
            session_id: Unique session identifier.

        Returns:
            Stored CompressionProfile or the default profile if not set.
        """
        name = self._store.get(session_id, DEFAULT_COMPRESSION_PROFILE)
        return get_profile(name)


_FIDELITY_COMPACTNESS_ORDER = ["RAW", "DETAILED", "STRUCTURE", "OUTLINE", "ABSTRACT"]
"""Fidelity levels ordered from least to most compact (index 0 = least compressed)."""


def apply_urgency(params: dict, urgency: str) -> dict:
    """Apply urgency overrides to compression parameters.

    Args:
        params: Existing parameters (not mutated).
        urgency: "normal" (no change), "compact" (cap ratio at 0.15, fidelity at OUTLINE),
                 "emergency" (force ratio 0.05, fidelity ABSTRACT).

    Returns:
        New dict with urgency applied.

    Raises:
        ValueError: If urgency is not one of normal/compact/emergency.
    """
    valid = {"normal", "compact", "emergency"}
    if urgency not in valid:
        raise ValueError(f"Invalid urgency {urgency!r}. Valid values: {', '.join(sorted(valid))}")

    result = dict(params)

    if urgency == "normal":
        return result

    if urgency == "emergency":
        result["skeleton_ratio"] = 0.05
        result["fidelity"] = "ABSTRACT"
        result["chunk_size"] = 256
        return result

    # urgency == "compact"
    current_ratio = result.get("skeleton_ratio", 1.0)
    result["skeleton_ratio"] = min(current_ratio, 0.15)

    current_fidelity = result.get("fidelity", "RAW")
    try:
        current_idx = _FIDELITY_COMPACTNESS_ORDER.index(current_fidelity)
        outline_idx = _FIDELITY_COMPACTNESS_ORDER.index("OUTLINE")
        # Only change if current fidelity is less compact than OUTLINE
        if current_idx < outline_idx:
            result["fidelity"] = "OUTLINE"
    except ValueError:
        result["fidelity"] = "OUTLINE"

    return result


def apply_profile(params: dict, profile: CompressionProfile) -> dict:
    """Fill missing keys in *params* from *profile*, returning a new dict.

    Explicit values in *params* are never overridden. Only the three core
    parameters (skeleton_ratio, fidelity, chunk_size) are filled from the
    profile.

    Args:
        params: Caller-supplied parameter overrides (not mutated).
        profile: Profile supplying default values.

    Returns:
        New dict combining explicit params with profile defaults.
    """
    result = dict(params)
    if "skeleton_ratio" not in result:
        result["skeleton_ratio"] = profile.skeleton_ratio
    if "fidelity" not in result:
        result["fidelity"] = profile.fidelity
    if "chunk_size" not in result:
        result["chunk_size"] = profile.chunk_size
    return result


def auto_select_profile(
    text: str,
    quality_floor: float = 0.7,
    query: Optional[str] = None,
) -> str:
    """Select the most compressed profile that meets *quality_floor*.

    Simulates compression for each profile (ordered from most to least
    compressed) and returns the first one whose predicted quality meets
    *quality_floor*.  Falls back to ``'full'`` when nothing qualifies.

    Args:
        text: Original text to evaluate.
        quality_floor: Minimum acceptable quality score (0.0–1.0).
        query: Optional query for relevance-aware quality prediction.

    Returns:
        Profile name string (e.g. ``'minimal'``, ``'balanced'``, ``'full'``).
    """
    # Import here to avoid circular dependency
    from src.quality_predictor import QualityPredictor

    predictor = QualityPredictor()

    # Order from most compressed to least — return first that meets floor
    ordered = ["minimal", "summary", "balanced", "detailed", "full"]

    for profile_name in ordered:
        if profile_name not in COMPRESSION_PROFILES:
            continue
        simulated = predictor.simulate_compression(text, profile_name)
        score = predictor.predict_quality(text, simulated, query)
        if score >= quality_floor:
            return profile_name

    # If nothing meets the floor, return the least-compressed option
    return ordered[-1]
