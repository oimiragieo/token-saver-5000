"""Tests for the #137 generic conservative fallback strategy in
CLIOutputOptimizer (TDD — written before implementation).

Covers:
- The 7 ordered sub-rules (a)-(g) of `_generic_conservative` in isolation
- `filter(..., fallback_strategy=...)` wiring: only fires on "unknown",
  `command_detected` stays truthful, `strategy_applied` reflects the
  fallback, STRATEGY_MAP registration, default-None leaves other callers
  (MCP tool / /v1/filter-cli / the wrap proxy) unaffected
- A ~400-line synthetic "retry storm" fixture: >=30% savings with every
  high-value identifier surviving
- A genuinely unique-prose fixture: near-zero (<5%) savings — the honest
  floor this strategy must respect, not manufacture fake compression on
- The MUST-NOT guardrails: no global dedup, JSON/YAML structural lines
  protected, frame elision never fires on <=8-frame blocks, blank runs
  never collapse to zero, a distinct URL/UUID/error-code in a run exempts it

codex adversarial-gate round 2 (2026-07-21) findings, all covered below:
  1. `_mask_line` masks ONLY provable volatility (timestamps/UUIDs/hex) —
     NEVER plain integers/decimals — so a run of DISTINCT numeric data
     (`processed batch 1..50`) never collapses (`TestDistinctNumericData`).
  2. Structured JSON/YAML lines bypass generic dedup/collapse entirely,
     regardless of length (`TestStructuredContentBypass`).
  3. CR-resolution only fires on genuine progress/overwrite shape; a
     `\\r`-separated DATA line survives whole (`TestCrResolve`).
  4. Progress-noise drop never eats a spinner/percentage line that carries
     real content — only the leading glyph is stripped (`TestProgressNoiseDrop`).
  5. `extract_critical_identifiers` round-robins across pattern classes so
     neither `numeric_literal` nor `symbol` can crowd the other out of the
     reinjection footer (`test_identifier_preservation.py`).
"""

from __future__ import annotations

from src.cli_output_optimizer import CLIOutputOptimizer, STRATEGY_MAP
from src.identifier_preservation import apply_identifier_guard, extract_critical_identifiers

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _retry_storm_lines(n: int = 300) -> list[str]:
    """`n` timestamped retry-log lines that mask to the IDENTICAL shape.

    codex adversarial-gate finding 1 (2026-07-21): the ONLY thing that
    varies line-to-line is the TIMESTAMP — a provably-volatile field that
    is safe to mask. Earlier drafts embedded a plain incrementing "attempt
    {i}" counter in the message text; since #137's fix deliberately never
    masks plain integers (a counter might be MEANINGFUL, distinct data),
    that shape no longer collapses — which is the correct, signal-safe
    behavior, not a bug. This fixture represents the (very common)
    real-world case of a service logging the IDENTICAL message repeatedly
    with nothing but a new timestamp each time.
    """
    lines = []
    for i in range(n):
        minute, second = divmod(i, 60)
        ts = f"2026-07-21T10:{minute:02d}:{second:02d}.123Z"
        lines.append(f"{ts} [WARN] retry failed: connection refused to db.internal:5432")
    return lines


def _distinct_batch_lines(n: int = 20) -> list[str]:
    """`n` lines that are STRUCTURALLY similar but carry genuinely DISTINCT
    numeric/text data per line (codex finding 1's canonical example) — must
    NEVER collapse, since the varying content is meaningful, not noise."""
    return [f"processed batch {i}: {i * 137} records, ${i * 42}.50 total" for i in range(n)]


def _python_traceback_lines(n_frames: int = 38) -> list[str]:
    """A Python-style traceback with `n_frames` header+context-line frames."""
    lines = ["Traceback (most recent call last):"]
    for i in range(n_frames):
        lines.append(f'  File "src/module_{i}.py", line {100 + i}, in function_{i}')
        lines.append(f"    do_something_{i}()")
    lines.append("ValueError: something went wrong at the deepest frame")
    return lines


def build_retry_storm_fixture() -> tuple[str, list[str]]:
    """A ~400-line synthetic verbose tool-output blob: progress spam, a
    300-line timestamped retry storm, 10 exact-duplicate npm warnings, a
    38-frame Python traceback, and a handful of genuinely unique data rows
    carrying execution-critical identifiers.

    Returns ``(blob_text, critical_identifier_substrings_that_must_survive)``.
    """
    lines: list[str] = []

    # Progress-bar spam — dropped entirely. Uses dots-only noise (not bare
    # "N%" lines): a literal "%" anywhere in the blob trips
    # CLIOutputOptimizer.detect_command's coarse `\d+%|\r|spinner` heuristic
    # and misclassifies the WHOLE document as "progress_output" rather than
    # "unknown" — which would route through `_progress_strip` instead of
    # exercising the generic fallback this fixture is meant to prove out.
    lines.append("Connecting to upstream service...")
    lines.append("..")
    lines.append("...")
    lines.append("....")
    lines.append("Connected.")
    lines.append("")
    lines.append("")
    lines.append("")  # blank run of 3 -> collapses to 1

    lines.extend(_retry_storm_lines(300))

    lines.extend(["npm warn deprecated inflight@1.0.6 this module is not supported"] * 10)

    lines.extend(_python_traceback_lines(38))

    critical_ids = [
        "ECONNREFUSED",
        "src/api/webhooks/stripe.ts:98",
        "https://api.gotcontext.ai/v1/compress",
        "550e8400-e29b-41d4-a716-446655440000",
        "CLERK_SECRET_KEY",
    ]
    lines.append("Fatal: ECONNREFUSED while calling connectToDB")
    lines.append("  at connectToDB (src/api/webhooks/stripe.ts:98:12)")
    lines.append("  URL: https://api.gotcontext.ai/v1/compress")
    lines.append("  project: 550e8400-e29b-41d4-a716-446655440000")
    lines.append("  env: CLERK_SECRET_KEY missing")

    return "\n".join(lines), critical_ids


UNIQUE_PROSE_FIXTURE = (
    "The quarterly report highlights steady growth across every major region.\n"
    "Local partnerships have accelerated distribution ahead of the original roadmap.\n"
    "Customer feedback remains overwhelmingly positive across every surveyed segment.\n"
    "The support team continues to close tickets faster than the target service level.\n"
    "Leadership expects the next two quarters to test the durability of these gains.\n"
    "Seasonal demand shifts and new competitors may pressure margins in adjacent categories.\n"
    "Engineering shipped four major releases this quarter, each with its own retrospective.\n"
    "Marketing attributes the lift to a redesigned onboarding sequence launched in March.\n"
    "Finance flagged a modest increase in cloud spend tied to the new analytics pipeline.\n"
    "Overall sentiment across every department remains cautiously optimistic heading into Q3.\n"
)


# ---------------------------------------------------------------------------
# STRATEGY_MAP + filter() wiring
# ---------------------------------------------------------------------------


class TestStrategyMapAndFilterWiring:
    def setup_method(self):
        self.opt = CLIOutputOptimizer()

    def test_generic_registered_in_strategy_map(self):
        assert STRATEGY_MAP["generic"] == "_generic_conservative"

    def test_unknown_without_fallback_still_passthrough(self):
        plain = "Hello world.\nThis is a simple sentence.\nNothing special here.\n" * 2
        result = self.opt.filter(plain)
        assert result.command_detected == "unknown"
        assert result.strategy_applied == "passthrough"

    def test_unknown_with_fallback_generic_applies_strategy(self):
        text = "\n".join(f"2026-07-21T10:00:{i:02d}.000Z [WARN] retry to db" for i in range(6))
        result = self.opt.filter(text, fallback_strategy="generic")
        assert result.command_detected == "unknown"  # truthful — still don't know the TYPE
        assert result.strategy_applied == "_generic_conservative"
        assert "similar lines elided" in result.filtered_text

    def test_fallback_strategy_ignored_when_command_type_known(self):
        git_diff = (
            "diff --git a/src/main.py b/src/main.py\n"
            "index abc1234..def5678 100644\n"
            "--- a/src/main.py\n"
            "+++ b/src/main.py\n"
            "@@ -10,6 +10,8 @@ def main():\n"
            '     print("hello")\n'
            '+    print("world")\n'
        )
        result = self.opt.filter(git_diff, fallback_strategy="generic")
        assert result.command_detected == "git_diff"
        assert result.strategy_applied == "_git_diff_stats"

    def test_command_hint_generic_forces_strategy_directly(self):
        text = "some plain unclassified text with no signal " * 10
        result = self.opt.filter(text, command_hint="generic")
        assert result.strategy_applied == "_generic_conservative"


# ---------------------------------------------------------------------------
# (a) CR-resolve
# ---------------------------------------------------------------------------


class TestCrResolve:
    def setup_method(self):
        self.opt = CLIOutputOptimizer()

    def test_cr_resolve_keeps_final_rendered_state_with_percentage(self):
        text = "Downloading 10%\rDownloading 50%\rDownloading 100%\nNext line"
        result = self.opt._generic_conservative(text)
        assert "\r" not in result
        assert "Next line" in result
        # The percentage overwrite frames are themselves bare-percentage-
        # shaped progress lines once split -- dropped by rule (b), not
        # asserted verbatim here (see TestProgressNoiseDrop for that rule).

    def test_cr_resolve_keeps_final_rendered_state_with_spinner(self):
        text = "⠋ loading\r⠙ loading\r⠹ loading\nDone"
        result = self.opt._generic_conservative(text)
        assert "\r" not in result
        assert "Done" in result

    def test_crlf_is_not_mistaken_for_an_overwrite(self):
        text = "line one\r\nline two\r\nline three"
        result = self.opt._generic_conservative(text)
        assert "line one" in result
        assert "line two" in result
        assert "line three" in result

    def test_cr_data_line_without_progress_shape_survives_whole(self):
        """codex finding 3: a `\\r`-separated DATA line (old-Mac record
        separator / `\\r`-delimited payload fields) must NOT be truncated
        to its last segment -- there is no %/spinner/dots signal proving
        this is a progress overwrite, so it stays fully intact."""
        text = "field_one\rfield_two\rfield_three"
        result = self.opt._generic_conservative(text)
        assert result == text
        assert "\r" in result

    def test_cr_old_mac_style_record_survives_whole(self):
        text = "record_one_data\rrecord_two_data\rrecord_three_data"
        result = self.opt._generic_conservative(text)
        assert result == text


# ---------------------------------------------------------------------------
# (b) progress noise
# ---------------------------------------------------------------------------


class TestProgressNoiseDrop:
    def setup_method(self):
        self.opt = CLIOutputOptimizer()

    def test_bare_percentage_lines_dropped(self):
        text = "Downloading\n  10%\n  50%\n  100%\nDone"
        result = self.opt._generic_conservative(text)
        assert "10%" not in result
        assert "50%" not in result
        assert "100%" not in result
        assert "Downloading" in result
        assert "Done" in result

    def test_spinner_only_lines_dropped(self):
        text = "⠋ loading\n⠙ loading\nDone"
        result = self.opt._generic_conservative(text)
        assert "⠋" not in result
        assert "Done" in result

    def test_dots_only_lines_dropped(self):
        text = "Working\n...\n....\nDone"
        result = self.opt._generic_conservative(text)
        assert result == "Working\nDone"

    def test_spinner_with_real_content_keeps_message(self):
        """codex finding 4: a spinner-LED line with real content must NOT
        be dropped wholesale — only the leading glyph is stripped, the
        message survives."""
        text = "⠋ candidate accepted\nDone"
        result = self.opt._generic_conservative(text)
        assert "candidate accepted" in result
        assert "⠋" not in result

    def test_spinner_with_bare_percentage_still_drops(self):
        """A spinner immediately followed by JUST a percentage (no other
        substantive tokens) is still ENTIRELY progress -- drops."""
        text = "Downloading\n⠋ 45%\nDone"
        result = self.opt._generic_conservative(text)
        assert "45%" not in result
        assert "⠋" not in result
        assert "Downloading" in result
        assert "Done" in result

    def test_multiple_leading_spinner_chars_with_message_kept(self):
        text = "⠋⠙ processing request\nDone"
        result = self.opt._generic_conservative(text)
        assert "processing request" in result
        assert "⠋" not in result
        assert "⠙" not in result


# ---------------------------------------------------------------------------
# (c) exact-dup collapse + the JSON/YAML structural-line MUST-NOT
# ---------------------------------------------------------------------------


class TestExactDupCollapse:
    def setup_method(self):
        self.opt = CLIOutputOptimizer()

    def test_exact_dup_run_collapses_with_repeated_count(self):
        text = "\n".join(["npm warn deprecated inflight@1.0.6 unsupported package"] * 6)
        result = self.opt._generic_conservative(text)
        assert "(repeated 6 times)" in result
        assert result.count("npm warn deprecated") == 1

    def test_short_json_structural_lines_never_collapsed_even_if_repeated(self):
        text = "\n".join(["},"] * 5)
        result = self.opt._generic_conservative(text)
        assert result.count("},") == 5
        assert "repeated" not in result

    def test_short_yaml_structural_lines_never_collapsed_even_if_repeated(self):
        text = "\n".join(["- name:"] * 6)
        result = self.opt._generic_conservative(text)
        assert result.count("- name:") == 6
        assert "repeated" not in result

    def test_single_occurrence_is_not_annotated(self):
        text = "unique line one\nunique line two\nunique line three"
        result = self.opt._generic_conservative(text)
        assert "repeated" not in result
        assert result == text


# ---------------------------------------------------------------------------
# (d) masked near-dup collapse + the distinct-identifier exemption
# ---------------------------------------------------------------------------


class TestMaskedNearDupCollapse:
    def setup_method(self):
        self.opt = CLIOutputOptimizer()

    def test_masked_near_dup_run_collapses_to_first_and_last(self):
        # Varies ONLY by timestamp (a provably-volatile field) -- the
        # message is byte-identical across all 8 lines, so they group and
        # collapse. (Varying by a plain counter is covered separately by
        # TestDistinctNumericData -- that must NEVER collapse.)
        lines = [f"2026-07-21T10:00:{i:02d}.000Z [WARN] retry connecting to db" for i in range(8)]
        result = self.opt._generic_conservative("\n".join(lines))
        assert lines[0] in result
        assert lines[-1] in result
        assert "similar lines elided" in result
        for middle in lines[1:-1]:
            assert middle not in result

    def test_masked_run_below_min_length_not_collapsed(self):
        lines = [f"2026-07-21T10:00:{i:02d}.000Z [WARN] retry" for i in range(4)]
        result = self.opt._generic_conservative("\n".join(lines))
        for line in lines:
            assert line in result
        assert "elided" not in result

    def test_masked_run_containing_url_is_not_collapsed(self):
        # The SAME URL on every line (only the timestamp varies -- masked,
        # so this run WOULD group) -- the embedded URL must still exempt
        # it from collapse.
        lines = [
            f"2026-07-21T10:00:{i:02d}.000Z [WARN] health check failed for "
            f"https://svc.internal/api/v2/health"
            for i in range(10)
        ]
        result = self.opt._generic_conservative("\n".join(lines))
        for original_line in lines:
            assert original_line in result
        assert "similar lines elided" not in result

    def test_masked_run_containing_uuid_is_not_collapsed(self):
        lines = [
            f"2026-07-21T10:00:{i:02d}.000Z [INFO] request "
            f"550e8400-e29b-41d4-a716-446655440000 processed"
            for i in range(10)
        ]
        result = self.opt._generic_conservative("\n".join(lines))
        for original_line in lines:
            assert original_line in result
        assert "similar lines elided" not in result

    def test_masked_run_containing_file_path_is_not_collapsed(self):
        lines = [
            f"2026-07-21T10:00:{i:02d}.000Z [INFO] loaded config from "
            f"src/config/settings.yaml successfully"
            for i in range(10)
        ]
        result = self.opt._generic_conservative("\n".join(lines))
        for original_line in lines:
            assert original_line in result
        assert "similar lines elided" not in result

    def test_masked_run_containing_real_error_code_is_not_collapsed(self):
        lines = [
            f"2026-07-21T10:00:{i:02d}.000Z [WARN] upstream call failed ECONNREFUSED retry"
            for i in range(10)
        ]
        result = self.opt._generic_conservative("\n".join(lines))
        for original_line in lines:
            assert original_line in result
        assert "similar lines elided" not in result

    def test_common_log_level_word_alone_does_not_exempt_the_run(self):
        # WARN/ERROR/INFO are common log-level tags, not distinct error
        # codes — a run of otherwise-identical lines must still collapse.
        lines = [f"2026-07-21T10:00:{i:02d}.000Z [WARN] retry to db" for i in range(8)]
        result = self.opt._generic_conservative("\n".join(lines))
        assert "similar lines elided" in result

    def test_masking_never_mutates_kept_line_content(self):
        lines = [f"2026-07-21T10:00:{i:02d}.000Z [INFO] job status ok" for i in range(3)]
        result = self.opt._generic_conservative("\n".join(lines))
        # Below the 5-line minimum -- every original timestamp survives
        # byte-identical (no placeholder text leaks into output).
        for line in lines:
            assert line in result
        assert "\x00" not in result
        assert "TS\x00" not in result

    def test_uuid_masked_as_one_unit_by_mask_line(self):
        """Direct unit check on `_mask_line`: a full UUID masks to ONE
        placeholder (not partially, via the 8-char hex rule alone)."""
        from src.cli_output_optimizer import _mask_line

        a = _mask_line("request 550e8400-e29b-41d4-a716-446655440000 done")
        b = _mask_line("request 6ba7b810-9dad-11d1-80b4-00c04fd430c8 done")
        assert a == b
        assert "\x00UUID\x00" in a

    def test_plain_numbers_never_masked_by_mask_line(self):
        """Direct unit check: `_mask_line` leaves plain integers/decimals
        untouched -- only timestamps/UUIDs/long hex blobs are masked."""
        from src.cli_output_optimizer import _mask_line

        assert _mask_line("processed batch 1: 137 records") != _mask_line(
            "processed batch 2: 274 records"
        )


# ---------------------------------------------------------------------------
# codex adversarial-gate finding 1: distinct numeric data must NEVER collapse
# ---------------------------------------------------------------------------


class TestDistinctNumericData:
    def setup_method(self):
        self.opt = CLIOutputOptimizer()

    def test_distinct_batch_lines_survive_uncollapsed(self):
        """The canonical codex example: 20 `processed batch N: <distinct>`
        lines, each carrying genuinely different record counts and dollar
        amounts, must survive byte-identical -- a plain incrementing
        number is never provable noise."""
        lines = _distinct_batch_lines(20)
        text = "\n".join(lines)
        result = self.opt._generic_conservative(text)
        assert result == text
        assert "similar lines elided" not in result
        assert "repeated" not in result

    def test_distinct_dollar_amounts_survive_uncollapsed(self):
        lines = [
            f"charged customer_{i}: ${100 + i * 7}.50 for plan tier {i % 3}" for i in range(20)
        ]
        text = "\n".join(lines)
        result = self.opt._generic_conservative(text)
        assert result == text

    def test_distinct_port_numbers_survive_uncollapsed(self):
        lines = [f"listening on port {8000 + i} for worker" for i in range(15)]
        text = "\n".join(lines)
        result = self.opt._generic_conservative(text)
        assert result == text

    def test_timestamp_only_varying_run_collapses_but_counter_varying_does_not(self):
        """The codex-mandated pairing test: 20 lines varying ONLY by
        timestamp collapse; 20 lines varying by a distinct counter do not."""
        ts_only = [f"2026-07-21T10:00:{i:02d}.000Z [INFO] heartbeat ok" for i in range(20)]
        ts_result = self.opt._generic_conservative("\n".join(ts_only))
        assert "similar lines elided" in ts_result

        counter_varying = _distinct_batch_lines(20)
        counter_result = self.opt._generic_conservative("\n".join(counter_varying))
        assert counter_result == "\n".join(counter_varying)
        assert "similar lines elided" not in counter_result


# ---------------------------------------------------------------------------
# codex adversarial-gate finding 2: JSON/YAML structured content bypass
# ---------------------------------------------------------------------------


class TestStructuredContentBypass:
    def setup_method(self):
        self.opt = CLIOutputOptimizer()

    def test_jsonl_block_with_varying_ids_passes_through_unchanged(self):
        lines = [f'{{"id": {i}, "name": "user_{i}", "status": "active"}}' for i in range(20)]
        text = "\n".join(lines)
        result = self.opt._generic_conservative(text)
        assert result == text  # every JSONL record survives byte-identical

    def test_jsonl_exact_duplicate_records_not_collapsed(self):
        """Repeated IDENTICAL JSON records (e.g. heartbeat events) must not
        be collapsed to "(repeated N times)" -- that is no longer valid
        JSONL and silently deletes the array's true record count."""
        lines = ['{"event": "heartbeat", "status": "ok"}'] * 10
        text = "\n".join(lines)
        result = self.opt._generic_conservative(text)
        assert result == text
        assert "repeated" not in result

    def test_yaml_flat_list_with_varying_ids_passes_through_unchanged(self):
        lines = [f"- id: {i}, name: user_{i}, active: true" for i in range(20)]
        text = "\n".join(lines)
        result = self.opt._generic_conservative(text)
        assert result == text

    def test_yaml_exact_duplicate_list_items_not_collapsed(self):
        lines = ["- status: ok, checked: true"] * 8
        text = "\n".join(lines)
        result = self.opt._generic_conservative(text)
        assert result == text
        assert "repeated" not in result

    def test_json_structural_bracket_only_lines_survive(self):
        lines = ["{", '  "id": 1,', "},", "{", '  "id": 2,', "},"]
        text = "\n".join(lines)
        result = self.opt._generic_conservative(text)
        assert result == text

    def test_is_structured_line_detector_directly(self):
        from src.cli_output_optimizer import _is_structured_line

        assert _is_structured_line('{"id": 1, "name": "alice"}')
        assert _is_structured_line('{"id": 1, "name": "alice"},')
        assert _is_structured_line("},")
        assert _is_structured_line("- name: alice")
        assert _is_structured_line('"key": "value",')
        assert not _is_structured_line("npm warn deprecated inflight@1.0.6 unsupported package")
        assert not _is_structured_line("2026-07-21T10:00:00.000Z [WARN] retry failed")


# ---------------------------------------------------------------------------
# (e) stack-frame elision
# ---------------------------------------------------------------------------


class TestStackFrameElision:
    def setup_method(self):
        self.opt = CLIOutputOptimizer()

    def test_long_python_traceback_elides_middle_keeps_top5_bottom2(self):
        lines = _python_traceback_lines(38)
        result = self.opt._generic_conservative("\n".join(lines))
        assert "(31 frames elided)" in result
        # Top 5 headers survive
        for i in range(5):
            assert f"function_{i}" in result
        # Bottom 2 headers survive
        for i in (36, 37):
            assert f"function_{i}" in result
        # A solidly-middle frame is gone
        assert "function_20" not in result

    def test_block_of_exactly_8_frames_never_elided(self):
        lines = _python_traceback_lines(8)
        result = self.opt._generic_conservative("\n".join(lines))
        assert "elided" not in result
        for i in range(8):
            assert f"function_{i}" in result

    def test_block_of_9_frames_elides(self):
        lines = _python_traceback_lines(9)
        result = self.opt._generic_conservative("\n".join(lines))
        assert "(2 frames elided)" in result

    def test_caused_by_starts_a_new_block(self):
        first = _python_traceback_lines(9)
        second = _python_traceback_lines(9)
        # Rename the second traceback's functions so we can distinguish them.
        second = [ln.replace("function_", "cause_function_") for ln in second]
        text = "\n".join(first + ["Caused by: ValueError: root cause"] + second)
        result = self.opt._generic_conservative(text)
        # Each 9-frame block elides INDEPENDENTLY (2 elided each), not
        # merged into one 18-frame block (which would elide 11).
        assert result.count("(2 frames elided)") == 2
        assert "Caused by: ValueError: root cause" in result

    def test_js_style_frames_elide_without_context_lines(self):
        lines = [f"    at handler{i} (src/server.js:{100 + i}:5)" for i in range(12)]
        result = self.opt._generic_conservative("\n".join(lines))
        assert "frames elided" in result
        assert "handler0" in result
        assert "handler11" in result
        assert "handler6" not in result


# ---------------------------------------------------------------------------
# (f) blank-run collapse
# ---------------------------------------------------------------------------


class TestBlankRunCollapse:
    def setup_method(self):
        self.opt = CLIOutputOptimizer()

    def test_blank_run_of_three_collapses_to_one(self):
        text = "line one\n\n\n\nline two"
        result = self.opt._generic_conservative(text)
        assert result == "line one\n\nline two"

    def test_blank_run_of_two_is_unchanged(self):
        text = "line one\n\n\nline two"
        result = self.opt._generic_conservative(text)
        assert result == "line one\n\n\nline two"

    def test_blank_run_never_collapses_to_zero(self):
        text = "line one\n\n\n\n\n\nline two"
        result = self.opt._generic_conservative(text)
        lines = result.splitlines()
        idx = lines.index("line one")
        assert lines[idx + 1] == ""
        assert lines[idx + 2] == "line two"


# ---------------------------------------------------------------------------
# (g) timestamp+level-only line drop
# ---------------------------------------------------------------------------


class TestTimestampOnlyLineDrop:
    def setup_method(self):
        self.opt = CLIOutputOptimizer()

    def test_timestamp_and_level_only_lines_are_dropped(self):
        text = "\n".join(
            [
                "2026-07-21T10:00:00.000Z [INFO]",
                "2026-07-21T10:00:01.000Z [INFO] request completed successfully",
                "2026-07-21T10:00:02.000Z [INFO]",
            ]
        )
        result = self.opt._generic_conservative(text)
        assert "request completed successfully" in result
        lines_out = [ln for ln in result.splitlines() if ln.strip()]
        assert len(lines_out) == 1

    def test_message_bearing_line_keeps_its_timestamp_prefix(self):
        text = "2026-07-21T10:00:00.000Z [ERROR] disk full on /data"
        result = self.opt._generic_conservative(text)
        assert result == text  # never strip a timestamp PREFIX off a message


# ---------------------------------------------------------------------------
# Comprehensive fixture: retry storm >=30% savings, identifiers survive
# ---------------------------------------------------------------------------


class TestRetryStormFixture:
    def setup_method(self):
        self.opt = CLIOutputOptimizer()

    def test_retry_storm_compresses_at_least_30_pct_with_identifiers_surviving(self):
        blob, critical_ids = build_retry_storm_fixture()

        # Mirror the production compress_tool_output pipeline: extract
        # BEFORE compression, run the generic fallback strategy, guard AFTER.
        critical_tokens = extract_critical_identifiers(blob)
        result = self.opt.filter(blob, fallback_strategy="generic")
        assert result.command_detected == "unknown"
        assert result.strategy_applied == "_generic_conservative"

        compressed, _reinjected = apply_identifier_guard(result.filtered_text, critical_tokens)

        savings_pct = (1 - len(compressed) / len(blob)) * 100
        assert savings_pct >= 30.0, f"expected >=30% savings, got {savings_pct:.1f}%"

        for token in critical_ids:
            assert token in compressed, f"critical identifier lost: {token!r}"

    def test_retry_storm_middle_of_storm_is_elided(self):
        blob, _critical_ids = build_retry_storm_fixture()
        result = self.opt.filter(blob, fallback_strategy="generic")
        assert "similar lines elided" in result.filtered_text
        assert "(repeated 10 times)" in result.filtered_text
        assert "frames elided" in result.filtered_text


class TestUniqueProseHonestFloor:
    def setup_method(self):
        self.opt = CLIOutputOptimizer()

    def test_unique_prose_fixture_stays_below_5_pct_savings(self):
        result = self.opt.filter(UNIQUE_PROSE_FIXTURE, fallback_strategy="generic")
        assert result.command_detected == "unknown"
        savings_pct = (1 - len(result.filtered_text) / len(UNIQUE_PROSE_FIXTURE)) * 100
        assert savings_pct < 5.0, f"expected <5% savings on unique prose, got {savings_pct:.1f}%"
