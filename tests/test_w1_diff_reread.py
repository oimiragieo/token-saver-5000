"""Unit tests for SessionDiffTracker (CL 3b)."""

from __future__ import annotations

import hashlib

from src.diff_reread import SessionDiffTracker, reconstruct_from_diff


def test_first_read_returns_full_and_reread_unchanged_returns_marker():
    tracker = SessionDiffTracker(max_entries=16)
    content = "\n".join(f"line {i}" for i in range(20))

    first = tracker.read("s1", "src/a.py", content)
    assert first["mode"] == "full"
    assert first["content"] == content

    second = tracker.read("s1", "src/a.py", content)
    assert second["mode"] == "unchanged"
    assert second["content"].startswith("UNCHANGED (sha256=")
    assert "20 lines)" in second["content"]


def test_small_edit_returns_unified_diff():
    tracker = SessionDiffTracker(max_entries=16)
    lines = [f"line {i}" for i in range(30)]
    tracker.read("s1", "src/a.py", "\n".join(lines))

    edited = list(lines)
    edited[10] = "line 10 EDITED"
    reread = tracker.read("s1", "src/a.py", "\n".join(edited))
    assert reread["mode"] == "diff"
    assert "-line 10" in reread["content"]
    assert "+line 10 EDITED" in reread["content"]


def test_large_edit_returns_full_content():
    tracker = SessionDiffTracker(max_entries=16)
    tracker.read("s1", "src/a.py", "a\nb\nc\nd\n")

    reread = tracker.read("s1", "src/a.py", "x\ny\nz\nw\n")
    assert reread["mode"] == "full"
    assert reread["content"] == "x\ny\nz\nw\n"


def test_force_full_bypasses_unchanged_and_diff():
    tracker = SessionDiffTracker(max_entries=16)
    content = "hello\nworld\n"
    tracker.read("s1", "src/a.py", content)

    forced = tracker.read("s1", "src/a.py", content, force_full=True)
    assert forced["mode"] == "full"
    assert forced["content"] == content


def test_lru_eviction_at_max_entries():
    tracker = SessionDiffTracker(max_entries=2)
    tracker.read("s1", "a.py", "1")
    tracker.read("s1", "b.py", "2")
    tracker.read("s1", "c.py", "3")
    assert len(tracker) == 2

    # a.py was evicted, so re-reading it returns full
    reread_a = tracker.read("s1", "a.py", "1")
    assert reread_a["mode"] == "full"


def test_adding_final_newline_reconstructs_byte_exact():
    tracker = SessionDiffTracker(max_entries=16)
    base = "\n".join(f"line {i}" for i in range(30))  # no trailing newline
    updated = base + "\n"
    tracker.read("s1", "src/nl.py", base)

    reread = tracker.read("s1", "src/nl.py", updated)
    assert reread["mode"] == "diff"
    reconstructed = reconstruct_from_diff(base, reread["content"])
    assert reconstructed == updated
    assert (
        hashlib.sha256(reconstructed.encode("utf-8")).hexdigest()
        == reread["sha256"]
        == hashlib.sha256(updated.encode("utf-8")).hexdigest()
    )


def test_lf_to_crlf_conversion_preserves_exact_sha256():
    tracker = SessionDiffTracker(max_entries=16)
    lf_content = "\n".join(f"line {i}" for i in range(20)) + "\n"
    crlf_content = lf_content.replace("\n", "\r\n")
    tracker.read("s1", "src/eol.py", lf_content)

    reread = tracker.read("s1", "src/eol.py", crlf_content)
    expected_sha = hashlib.sha256(crlf_content.encode("utf-8")).hexdigest()
    assert reread["sha256"] == expected_sha
    if reread["mode"] == "diff":
        assert reconstruct_from_diff(lf_content, reread["content"]) == crlf_content
    else:
        assert reread["mode"] == "full"
        assert reread["content"] == crlf_content


def test_long_single_line_patch_falls_back_to_full():
    tracker = SessionDiffTracker(max_entries=16)
    base = "a" * 500
    updated = "a" * 490 + "b" * 10
    tracker.read("s1", "src/single.txt", base)

    reread = tracker.read("s1", "src/single.txt", updated)
    assert reread["mode"] == "full"
    assert reread["content"] == updated


def test_diff_mode_reconstruction_matches_bytes_and_sha256():
    tracker = SessionDiffTracker(max_entries=16)
    lines = [f"item {i}: value={i * 2}\n" for i in range(40)]
    base = "".join(lines)
    edited_lines = list(lines)
    edited_lines[15] = "item 15: value=UPDATED\n"
    updated = "".join(edited_lines)

    tracker.read("s1", "src/data.txt", base)
    reread = tracker.read("s1", "src/data.txt", updated)
    assert reread["mode"] == "diff"
    reconstructed = reconstruct_from_diff(base, reread["content"])
    assert reconstructed == updated
    assert hashlib.sha256(reconstructed.encode("utf-8")).hexdigest() == reread["sha256"]
