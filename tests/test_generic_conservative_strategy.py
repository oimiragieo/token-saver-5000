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
  1. `_mask_line` masks ONLY provable volatility (timestamps/UUIDs) —
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

codex adversarial-gate round 3 (2026-07-22) findings, narrowing round 2:
  1. Hex-masking DROPPED entirely from `_mask_line` (digits are valid hex
     chars, so an 8+ digit id/counter collided with the hex-blob pattern
     just like round 2's bare-number bug) — only timestamp/UUID shapes
     (non-pure-digit) survive masking (`TestDistinctNumericData`).
  2. `_is_structured_line` now detects JSON object/array/string SHAPE
     (first/last char, O(1)) regardless of length, closing the >2000-char
     gap that let long JSONL records lose protection (`TestStructuredContentBypass`).
  3. CR-resolution narrowed further: a single `\\r` needs its ONE
     pre-final segment to be entirely progress noise on its own; a
     substantive segment that merely CONTAINS "%" (`discount 95%\\rprice=10`)
     is no longer enough. Multiple `\\r`'s (>=3 segments) remains
     sufficient signal on its own (`TestCrResolve`).

codex adversarial-gate round 4 (2026-07-22, FINAL — convergence guardrail:
simplify rather than keep special-casing two narrow-edge heuristics):
  A. CR-resolution REMOVED ENTIRELY from the pipeline (not narrowed
     further) — it contributed ~0 measured savings and was a signal-loss
     source in every round (r1/r2/r3). A line containing `\\r` now passes
     through completely unmodified except universal `\\r\\n` -> `\\n`
     normalization (a deterministic encoding fix, never a heuristic
     guess) (`TestCrResolve`, rewritten).
  B. A blanket `_MAX_COLLAPSE_LINE_CHARS = 2000` length ceiling: no line
     longer than this is EVER collapsed or masked, by exact-dup OR
     masked-near-dup, superseding the need to keep perfecting a
     JSON-primitive shape detector for arbitrarily large values
     (`TestLengthCeiling`).
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
    """codex adversarial-gate round 4 (2026-07-22): CR-resolution was
    REMOVED entirely from the pipeline (not just narrowed further). It
    contributed ~0 measured savings and was a signal-loss source in every
    prior round. A line containing `\\r` now passes through completely
    unmodified -- only universal `\\r\\n` -> `\\n` normalization still
    applies (a deterministic encoding fix, never a heuristic guess)."""

    def setup_method(self):
        self.opt = CLIOutputOptimizer()

    def test_crlf_is_normalized_to_plain_newline(self):
        text = "line one\r\nline two\r\nline three"
        result = self.opt._generic_conservative(text)
        assert "line one" in result
        assert "line two" in result
        assert "line three" in result
        assert "\r" not in result  # CRLF normalized away, not a lone \r

    def test_lone_cr_data_line_passes_through_verbatim(self):
        """The exact regression this round closes: a `\\r`-separated DATA
        line must NEVER lose a field, regardless of segment count or
        content shape -- CR is no longer interpreted as an overwrite
        signal AT ALL."""
        text = "field1\rfield2\rfield3"
        result = self.opt._generic_conservative(text)
        assert result == text
        assert "field1" in result
        assert "field2" in result
        assert "field3" in result

    def test_cr_percentage_sequence_no_longer_resolved(self):
        """A genuine multi-frame progress sequence is NO LONGER collapsed
        to its final frame -- the whole heuristic class was removed, so
        this now passes through unmodified too (an intentional, documented
        trade-off: CR-resolution added negligible savings but was a
        repeated signal-loss source)."""
        text = "Downloading 10%\rDownloading 50%\rDownloading 100%\nNext line"
        result = self.opt._generic_conservative(text)
        assert result == text

    def test_cr_single_overwrite_with_real_pre_segment_survives_whole(self):
        """The exact codex-reported bug from round 3, now trivially true
        under blanket removal: a single `\\r` whose pre-segment is real
        data ("discount 95%") survives fully intact."""
        text = "discount 95%\rprice=10"
        result = self.opt._generic_conservative(text)
        assert result == text
        assert "discount 95%" in result
        assert "price=10" in result

    def test_cr_old_mac_style_record_survives_whole(self):
        text = "record_one_data\rrecord_two_data\rrecord_three_data"
        result = self.opt._generic_conservative(text)
        assert result == text

    def test_spinner_sequence_with_lone_cr_passes_through(self):
        """The embedded `\\r`s survive intact (finding 3 removed) -- the
        LEADING spinner glyph on the whole "line" is separately stripped
        by the unrelated, already-confirmed-fixed finding-4 rule, which
        keeps everything after the glyph (including the internal `\\r`s)
        verbatim."""
        text = "⠋ loading\r⠙ loading\r⠹ loading\nDone"
        result = self.opt._generic_conservative(text)
        assert "\r" in result
        assert result.count("\r") == 2
        assert "loading" in result
        assert "Done" in result


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

    def test_distinct_8_digit_ids_survive_uncollapsed(self):
        """codex round 3 finding 1: digits are valid hex characters, so the
        round-2 hex-blob mask (`\\b[0-9a-fA-F]{8,}\\b`) matched an 8+ digit
        plain id/counter just as readily as a real hash -- "batch
        12345678" and "batch 12345679" masked IDENTICALLY and could
        collapse as if they were the same noisy line. Hex-masking is now
        dropped entirely; a run of distinct 8-digit ids must survive."""
        lines = [f"batch {12345678 + i}: processing" for i in range(10)]
        text = "\n".join(lines)
        result = self.opt._generic_conservative(text)
        assert result == text
        assert "similar lines elided" not in result

    def test_mask_line_no_longer_collapses_distinct_hex_looking_digit_runs(self):
        """Direct unit check on `_mask_line`: two DIFFERENT 8-digit runs
        (each individually valid hex) must mask DIFFERENTLY now -- the old
        hex-blob rule made them mask identically."""
        from src.cli_output_optimizer import _mask_line

        a = _mask_line("batch 12345678: processing")
        b = _mask_line("batch 12345679: processing")
        assert a != b


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

    def test_oversized_json_object_line_not_collapsed_when_repeated(self):
        """codex round 3 finding 2: the old length cap made a JSON record
        line LOSE structured-content protection past 2000 chars, so
        repeated (or near-dup) oversized JSONL records could collapse,
        corrupting syntax and deleting cardinality. A >2000-char IDENTICAL
        JSON object line must still be protected from exact-dup collapse."""
        long_line = '{"event": "heartbeat", "payload": "' + ("x" * 2100) + '"}'
        assert len(long_line) > 2000
        lines = [long_line] * 8
        text = "\n".join(lines)
        result = self.opt._generic_conservative(text)
        assert result == text
        assert "repeated" not in result

    def test_oversized_json_array_element_line_not_collapsed(self):
        long_line = "[" + ", ".join(str(i) for i in range(500)) + "],"
        assert len(long_line) > 2000
        lines = [long_line] * 6
        text = "\n".join(lines)
        result = self.opt._generic_conservative(text)
        assert result == text

    def test_is_structured_line_detects_oversized_json_regardless_of_length(self):
        """Direct unit check: the shape-based detector fires on a
        >2000-char JSON object without needing (or risking) a full
        `json.loads` parse of the whole line."""
        from src.cli_output_optimizer import _is_structured_line

        long_obj = '{"id": 1, "payload": "' + ("x" * 2100) + '"}'
        assert len(long_obj) > 2000
        assert _is_structured_line(long_obj)
        assert _is_structured_line(long_obj + ",")

        long_array = "[" + ", ".join(str(i) for i in range(500)) + "]"
        assert len(long_array) > 2000
        assert _is_structured_line(long_array)
        assert _is_structured_line(long_array + ",")

        long_string = '"' + ("y" * 2100) + '"'
        assert len(long_string) > 2000
        assert _is_structured_line(long_string)
        assert _is_structured_line(long_string + ",")


# ---------------------------------------------------------------------------
# codex adversarial-gate round 4, item B: blanket length ceiling
# ---------------------------------------------------------------------------


class TestLengthCeiling:
    """A line longer than `_MAX_COLLAPSE_LINE_CHARS` (2000) is NEVER
    collapsed or masked, by exact-dup OR masked-near-dup -- a blanket
    conservative guard that supersedes the need to perfect a JSON-
    primitive shape detector for arbitrarily large, non-JSON-shaped
    values too (e.g. a long plain-text line, not just JSON)."""

    def setup_method(self):
        self.opt = CLIOutputOptimizer()

    def test_oversized_plain_text_line_repeated_not_collapsed(self):
        """A >2000-char line that is NOT JSON/YAML-shaped (so it would
        have relied purely on the length ceiling, not the structured-
        content bypass) must still never be exact-dup collapsed."""
        long_line = "processing record with payload=" + ("z" * 2000)
        assert len(long_line) > 2000
        lines = [long_line] * 6
        text = "\n".join(lines)
        result = self.opt._generic_conservative(text)
        assert result == text
        assert "repeated" not in result

    def test_oversized_masked_near_dup_run_not_collapsed(self):
        """A >2000-char line that would otherwise mask-group (timestamp-
        only-varying) must still never be collapsed once past the
        length ceiling."""
        long_suffix = "z" * 2000
        lines = [f"2026-07-22T10:00:{i:02d}.000Z [INFO] payload={long_suffix}" for i in range(8)]
        assert all(len(ln) > 2000 for ln in lines)
        text = "\n".join(lines)
        result = self.opt._generic_conservative(text)
        assert result == text
        assert "similar lines elided" not in result

    def test_is_collapsible_content_line_rejects_oversized_lines_directly(self):
        from src.cli_output_optimizer import _is_collapsible_content_line

        short_line = "a normal log line with plenty of content here"
        long_line = "a normal log line with plenty of content here " + ("x" * 2000)
        assert _is_collapsible_content_line(short_line)
        assert not _is_collapsible_content_line(long_line)

    def test_line_at_exactly_the_ceiling_is_still_collapsible(self):
        from src.cli_output_optimizer import _MAX_COLLAPSE_LINE_CHARS, _is_collapsible_content_line

        exactly_at_ceiling = "x" * _MAX_COLLAPSE_LINE_CHARS
        one_over_ceiling = "x" * (_MAX_COLLAPSE_LINE_CHARS + 1)
        assert _is_collapsible_content_line(exactly_at_ceiling)
        assert not _is_collapsible_content_line(one_over_ceiling)


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
