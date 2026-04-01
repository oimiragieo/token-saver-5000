"""
Tests for src/compression_profiles.py module.

TDD test suite for v0.11.0 named compression profiles.

CompressionProfile (dataclass):
- name, skeleton_ratio, fidelity, chunk_size, description

Module-level functions:
- get_profile(name) -> CompressionProfile
- get_default_profile() -> CompressionProfile  (returns "balanced")
- list_profiles() -> list[str]

ProfileManager:
- set_profile(session_id, name)
- get_profile(session_id) -> CompressionProfile
- apply_profile(params, profile) -> dict  (fills missing params from profile)

Explicit params always override profile defaults.

Uses COMPRESSION_PROFILES and DEFAULT_COMPRESSION_PROFILE from src.constants.
"""

import pytest

from src.compression_profiles import (
    CompressionProfile,
    ProfileManager,
    apply_profile,
    get_default_profile,
    get_profile,
    list_profiles,
)
from src.constants import COMPRESSION_PROFILES, DEFAULT_COMPRESSION_PROFILE


# ============================================================================
# Individual profile value tests
# ============================================================================


def test_minimal_profile_values():
    """minimal profile has skeleton_ratio=0.05, fidelity=ABSTRACT, chunk_size=256."""
    p = get_profile("minimal")
    assert p.skeleton_ratio == pytest.approx(0.05)
    assert p.fidelity == "ABSTRACT"
    assert p.chunk_size == 256


def test_summary_profile_values():
    """summary profile has skeleton_ratio=0.15, fidelity=OUTLINE, chunk_size=512."""
    p = get_profile("summary")
    assert p.skeleton_ratio == pytest.approx(0.15)
    assert p.fidelity == "OUTLINE"
    assert p.chunk_size == 512


def test_balanced_profile_values():
    """balanced profile has skeleton_ratio=0.25, fidelity=STRUCTURE, chunk_size=512."""
    p = get_profile("balanced")
    assert p.skeleton_ratio == pytest.approx(0.25)
    assert p.fidelity == "STRUCTURE"
    assert p.chunk_size == 512


def test_detailed_profile_values():
    """detailed profile has skeleton_ratio=0.50, fidelity=DETAILED, chunk_size=1024."""
    p = get_profile("detailed")
    assert p.skeleton_ratio == pytest.approx(0.50)
    assert p.fidelity == "DETAILED"
    assert p.chunk_size == 1024


def test_full_profile_values():
    """full profile has skeleton_ratio=0.80, fidelity=RAW, chunk_size=2048."""
    p = get_profile("full")
    assert p.skeleton_ratio == pytest.approx(0.80)
    assert p.fidelity == "RAW"
    assert p.chunk_size == 2048


# ============================================================================
# get_profile
# ============================================================================


def test_get_profile_by_name():
    """get_profile('minimal') returns a CompressionProfile with name='minimal'."""
    p = get_profile("minimal")
    assert isinstance(p, CompressionProfile)
    assert p.name == "minimal"


def test_unknown_profile_raises():
    """get_profile('nonexistent') raises ValueError."""
    with pytest.raises(ValueError, match="nonexistent"):
        get_profile("nonexistent")


def test_unknown_profile_error_mentions_valid_names():
    """ValueError from get_profile lists available profile names."""
    with pytest.raises(ValueError) as exc_info:
        get_profile("does_not_exist")
    error_msg = str(exc_info.value)
    # Should mention at least one valid profile name in the error
    assert any(name in error_msg for name in ["minimal", "balanced", "full", "summary", "detailed"])


# ============================================================================
# get_default_profile
# ============================================================================


def test_default_profile_is_balanced():
    """get_default_profile() returns the 'balanced' profile."""
    p = get_default_profile()
    assert p.name == DEFAULT_COMPRESSION_PROFILE
    assert p.name == "balanced"


def test_default_profile_matches_constants():
    """get_default_profile() matches COMPRESSION_PROFILES[DEFAULT_COMPRESSION_PROFILE]."""
    p = get_default_profile()
    expected = COMPRESSION_PROFILES[DEFAULT_COMPRESSION_PROFILE]
    assert p.skeleton_ratio == pytest.approx(expected["skeleton_ratio"])
    assert p.fidelity == expected["fidelity"]
    assert p.chunk_size == expected["chunk_size"]


# ============================================================================
# list_profiles
# ============================================================================


def test_list_profiles():
    """list_profiles() returns all 5 profile names."""
    names = list_profiles()
    assert isinstance(names, list)
    assert len(names) == 5
    assert set(names) == {"minimal", "summary", "balanced", "detailed", "full"}


def test_list_profiles_no_duplicates():
    """list_profiles() has no duplicate names."""
    names = list_profiles()
    assert len(names) == len(set(names))


# ============================================================================
# Profile value range validation
# ============================================================================


def test_all_skeleton_ratios_in_range():
    """All profiles have skeleton_ratio in (0, 1]."""
    for name in list_profiles():
        p = get_profile(name)
        assert (
            0 < p.skeleton_ratio <= 1.0
        ), f"Profile '{name}' skeleton_ratio={p.skeleton_ratio} out of (0, 1]"


def test_all_chunk_sizes_positive():
    """All profiles have chunk_size > 0."""
    for name in list_profiles():
        p = get_profile(name)
        assert p.chunk_size > 0, f"Profile '{name}' chunk_size={p.chunk_size} is not positive"


def test_skeleton_ratios_increase_with_profile_intensity():
    """Profiles have increasing skeleton_ratio: minimal < summary < balanced < detailed < full."""
    ratios = [
        get_profile(n).skeleton_ratio
        for n in ["minimal", "summary", "balanced", "detailed", "full"]
    ]
    assert ratios == sorted(ratios), f"Skeleton ratios are not monotonically increasing: {ratios}"


# ============================================================================
# ProfileManager — set/get
# ============================================================================


def test_profile_manager_set_and_get():
    """ProfileManager.set_profile + get_profile round-trip."""
    mgr = ProfileManager()
    mgr.set_profile("session-1", "detailed")
    p = mgr.get_profile("session-1")
    assert p.name == "detailed"


def test_profile_manager_isolated_sessions():
    """Different session_ids have independent profile configurations."""
    mgr = ProfileManager()
    mgr.set_profile("s1", "minimal")
    mgr.set_profile("s2", "full")

    assert mgr.get_profile("s1").name == "minimal"
    assert mgr.get_profile("s2").name == "full"


def test_profile_manager_default_when_unset():
    """Unset session returns the balanced (default) profile."""
    mgr = ProfileManager()
    p = mgr.get_profile("never-configured-session")
    assert p.name == DEFAULT_COMPRESSION_PROFILE


def test_profile_manager_overwrite():
    """Re-setting a session's profile overwrites the previous value."""
    mgr = ProfileManager()
    mgr.set_profile("s1", "minimal")
    mgr.set_profile("s1", "full")
    assert mgr.get_profile("s1").name == "full"


def test_profile_manager_rejects_unknown_profile():
    """set_profile raises ValueError for unknown profile names."""
    mgr = ProfileManager()
    with pytest.raises(ValueError):
        mgr.set_profile("s1", "not-a-real-profile")


def test_profile_manager_isolated_across_instances():
    """Two ProfileManager instances do not share session state."""
    mgr_a = ProfileManager()
    mgr_b = ProfileManager()

    mgr_a.set_profile("shared-session", "minimal")
    # mgr_b has no knowledge of mgr_a's sessions
    p = mgr_b.get_profile("shared-session")
    assert p.name == DEFAULT_COMPRESSION_PROFILE  # returns default, not "minimal"


# ============================================================================
# apply_profile — filling missing params
# ============================================================================


def test_apply_profile_to_params():
    """apply_profile fills in missing params (skeleton_ratio, fidelity, chunk_size) from profile."""
    profile = get_profile("minimal")
    params = {}  # No explicit params
    result = apply_profile(params, profile)

    assert result["skeleton_ratio"] == pytest.approx(0.05)
    assert result["fidelity"] == "ABSTRACT"
    assert result["chunk_size"] == 256


def test_explicit_params_override_profile():
    """Explicit params take priority over profile defaults."""
    profile = get_profile("minimal")
    params = {"skeleton_ratio": 0.99, "fidelity": "RAW"}
    result = apply_profile(params, profile)

    # Explicit values win
    assert result["skeleton_ratio"] == pytest.approx(0.99)
    assert result["fidelity"] == "RAW"
    # chunk_size was not specified, so profile default applies
    assert result["chunk_size"] == 256


def test_apply_profile_partial_override():
    """Only provided params override; unprovided params come from profile."""
    profile = get_profile("balanced")
    params = {"chunk_size": 2048}  # Only override chunk_size
    result = apply_profile(params, profile)

    # chunk_size is overridden
    assert result["chunk_size"] == 2048
    # Others come from balanced profile
    assert result["skeleton_ratio"] == pytest.approx(0.25)
    assert result["fidelity"] == "STRUCTURE"


def test_apply_profile_does_not_mutate_params():
    """apply_profile does not modify the original params dict."""
    profile = get_profile("full")
    params = {"skeleton_ratio": 0.5}
    original_keys = set(params.keys())
    original_val = params["skeleton_ratio"]

    apply_profile(params, profile)

    assert set(params.keys()) == original_keys
    assert params["skeleton_ratio"] == original_val


def test_apply_profile_returns_new_dict():
    """apply_profile returns a new dict, not the same object as params."""
    profile = get_profile("summary")
    params = {}
    result = apply_profile(params, profile)
    assert result is not params


# ============================================================================
# CompressionProfile dataclass
# ============================================================================


def test_compression_profile_has_description():
    """All profiles have a non-empty description string."""
    for name in list_profiles():
        p = get_profile(name)
        assert (
            p.description and len(p.description.strip()) > 0
        ), f"Profile '{name}' has empty description"


def test_compression_profile_name_matches_key():
    """Profile's .name attribute matches the key used to retrieve it."""
    for name in list_profiles():
        p = get_profile(name)
        assert p.name == name
