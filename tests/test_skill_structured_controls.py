"""Tests for structured segment controls in skill compression engine."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS = ROOT / "skills" / "token-saver-context-compression" / "scripts"
if str(SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SKILL_SCRIPTS))

from _compression_engine import compress_text  # noqa: E402


def test_structured_preserve_tag_keeps_segment():
    text = (
        "alpha one. alpha two. alpha three. alpha four. alpha five. "
        "<llmlingua, compress=False>must keep this critical sentence.</llmlingua>"
    )
    result = compress_text(text=text, mode="baseline", skeleton_ratio=0.1)

    selected_texts = [row["text"] for row in result.segments if row["selected"]]
    assert any("must keep this critical sentence" in item for item in selected_texts)


def test_structured_rate_tag_overrides_global_ratio_for_tagged_block():
    text = (
        "outside one. outside two. outside three. "
        "<llmlingua, compress=True, rate=0.75>"
        "inside a. inside b. inside c. inside d."
        "</llmlingua>"
        " outside four. outside five."
    )
    result = compress_text(text=text, mode="baseline", skeleton_ratio=0.1)

    inside_selected = [
        row for row in result.segments if row["selected"] and row["text"].startswith("inside ")
    ]
    assert len(inside_selected) >= 3


def test_structured_invalid_marker_raises_clear_error():
    text = "<llmlingua, compress=False>broken marker with no close"
    with pytest.raises(ValueError, match="Unclosed <llmlingua> tag"):
        compress_text(text=text, mode="baseline", skeleton_ratio=0.2)
