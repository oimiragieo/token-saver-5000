"""Root-cause diagnostics for prompt cache misses."""

from __future__ import annotations

import hashlib
import json
import re
from difflib import unified_diff
from typing import Any

from .model_optimizer import summarize_provider_cache_usage
from .prompt_cache_audit import CANONICAL_SECTION_ORDER
from .provider_profiles import get_provider_profile

_UUID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
)
_TIMESTAMP_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\b")
_SECTION_HEADER_PATTERN = re.compile(r"^\[(?P<name>[^\]\n]+)\]$", re.MULTILINE)
_INPUT_TOKEN_FIELDS = ("input_tokens", "prompt_tokens", "promptTokenCount")
_OUTPUT_TOKEN_FIELDS = ("output_tokens", "completion_tokens", "candidatesTokenCount")
_VOLATILE_SECTIONS = {"metadata", "user_query"}


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _find_numeric_field(payload: Any, field_names: tuple[str, ...]) -> int:
    if isinstance(payload, dict):
        for field_name in field_names:
            value = payload.get(field_name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(value)
        for nested in payload.values():
            nested_value = _find_numeric_field(nested, field_names)
            if nested_value >= 0:
                return nested_value
    elif isinstance(payload, list):
        for item in payload:
            nested_value = _find_numeric_field(item, field_names)
            if nested_value >= 0:
                return nested_value
    return -1


def _diff_preview(expected_hash: str, actual_prefix: str) -> str:
    expected_stub = f"expected_prefix_hash={expected_hash}"
    actual_stub = actual_prefix[:240]
    return "\n".join(
        unified_diff(
            [expected_stub],
            [actual_stub],
            fromfile="expected",
            tofile="actual",
            lineterm="",
        )
    )


def extract_section_order(rendered_prefix: str) -> list[str]:
    if not isinstance(rendered_prefix, str):
        raise ValueError("extract_section_order requires string 'rendered_prefix'")

    normalized_prefix = _normalize_line_endings(rendered_prefix)
    return [
        match.group("name")
        for match in _SECTION_HEADER_PATTERN.finditer(normalized_prefix)
        if match.group("name") in CANONICAL_SECTION_ORDER
    ]


def _extract_sections(rendered_prefix: str) -> list[tuple[str, str]]:
    normalized_prefix = _normalize_line_endings(rendered_prefix)
    matches = list(_SECTION_HEADER_PATTERN.finditer(normalized_prefix))
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        name = match.group("name")
        if name not in CANONICAL_SECTION_ORDER:
            continue
        content_start = match.end()
        if normalized_prefix[content_start : content_start + 1] == "\n":
            content_start += 1
        content_end = (
            matches[index + 1].start() - 2
            if index + 1 < len(matches)
            and normalized_prefix[matches[index + 1].start() - 2 : matches[index + 1].start()]
            == "\n\n"
            else matches[index + 1].start() if index + 1 < len(matches) else len(normalized_prefix)
        )
        content = normalized_prefix[content_start:content_end].strip()
        sections.append((name, content))
    return sections


def _normalize_blank_lines(text: str) -> str:
    collapsed_lines: list[str] = []
    previous_blank = False
    for line in text.split("\n"):
        stripped_line = line.rstrip()
        is_blank = stripped_line == ""
        if is_blank and previous_blank:
            continue
        collapsed_lines.append(stripped_line)
        previous_blank = is_blank
    return "\n".join(collapsed_lines).strip()


def _normalize_json_content(content: str) -> str | None:
    normalized_content = _normalize_blank_lines(_normalize_line_endings(content))
    try:
        parsed = json.loads(normalized_content)
    except json.JSONDecodeError:
        return None
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"))


def _normalize_text_content(content: str) -> str:
    return _normalize_blank_lines(_normalize_line_endings(content))


def detect_semantic_equivalence_drift(
    *,
    expected_prefix: str | None,
    actual_prefix: str,
) -> dict[str, Any]:
    if not isinstance(actual_prefix, str):
        raise ValueError("detect_semantic_equivalence_drift requires string 'actual_prefix'")

    if not isinstance(expected_prefix, str) or not expected_prefix:
        return {
            "semantic_match": False,
            "drift_type": "not_available",
            "whitespace_changed": False,
            "json_serialization_changed": False,
            "warning": None,
        }

    if expected_prefix == actual_prefix:
        return {
            "semantic_match": True,
            "drift_type": "none",
            "whitespace_changed": False,
            "json_serialization_changed": False,
            "warning": None,
        }

    expected_sections = _extract_sections(expected_prefix)
    actual_sections = _extract_sections(actual_prefix)
    if bool(expected_sections) != bool(actual_sections) or [
        name for name, _ in expected_sections
    ] != [name for name, _ in actual_sections]:
        return {
            "semantic_match": False,
            "drift_type": "semantic_difference",
            "whitespace_changed": False,
            "json_serialization_changed": False,
            "warning": None,
        }

    whitespace_changed = False
    json_serialization_changed = False
    semantic_difference = False

    comparable_sections = expected_sections or [("raw_prefix", expected_prefix)]
    actual_comparable_sections = actual_sections or [("raw_prefix", actual_prefix)]
    for (_, expected_content), (_, actual_content) in zip(
        comparable_sections, actual_comparable_sections
    ):
        if expected_content == actual_content:
            continue

        if _normalize_text_content(expected_content) == _normalize_text_content(actual_content):
            whitespace_changed = True
            continue

        expected_json = _normalize_json_content(expected_content)
        actual_json = _normalize_json_content(actual_content)
        if expected_json is not None and expected_json == actual_json:
            json_serialization_changed = True
            continue

        semantic_difference = True
        break

    if semantic_difference:
        return {
            "semantic_match": False,
            "drift_type": "semantic_difference",
            "whitespace_changed": whitespace_changed,
            "json_serialization_changed": json_serialization_changed,
            "warning": None,
        }

    if (
        not whitespace_changed
        and not json_serialization_changed
        and _normalize_text_content(expected_prefix) == _normalize_text_content(actual_prefix)
    ):
        whitespace_changed = True

    drift_type = "mixed"
    if whitespace_changed and not json_serialization_changed:
        drift_type = "whitespace_only"
    elif json_serialization_changed and not whitespace_changed:
        drift_type = "json_serialization"

    warning = (
        "Stable prefix changed at the byte level but remained semantically equivalent after "
        "whitespace or serialization normalization. Canonicalize whitespace, line endings, "
        "and JSON serialization to recover cache reuse."
    )
    return {
        "semantic_match": True,
        "drift_type": drift_type,
        "whitespace_changed": whitespace_changed,
        "json_serialization_changed": json_serialization_changed,
        "warning": warning,
    }


def detect_section_interleaving(
    *,
    expected_section_order: list[str] | tuple[str, ...] | None,
    actual_rendered_prefix: str,
) -> dict[str, Any]:
    actual_order = extract_section_order(actual_rendered_prefix)
    if expected_section_order:
        expected_order = [
            section
            for section in expected_section_order
            if isinstance(section, str) and section in CANONICAL_SECTION_ORDER
        ]
    else:
        expected_order = [section for section in CANONICAL_SECTION_ORDER if section in actual_order]

    expected_stable_order = [
        section for section in expected_order if section not in _VOLATILE_SECTIONS
    ]
    actual_stable_order = [section for section in actual_order if section not in _VOLATILE_SECTIONS]
    shared_stable_sections = [
        section for section in expected_stable_order if section in actual_stable_order
    ]
    actual_shared_order = [
        section for section in actual_stable_order if section in shared_stable_sections
    ]
    sections_reordered = shared_stable_sections != actual_shared_order
    unexpected_stable_sections = [
        section for section in actual_stable_order if section not in expected_stable_order
    ]
    missing_stable_sections = [
        section for section in expected_stable_order if section not in actual_stable_order
    ]
    layout_changed = bool(
        sections_reordered or unexpected_stable_sections or missing_stable_sections
    )

    warning = None
    if layout_changed:
        warning = (
            "Prompt section layout drift detected in the stable prefix. "
            "Keep tool defs, system instructions, docs, and examples in canonical order "
            "and avoid inserting or removing stable sections between otherwise reusable renders."
        )

    return {
        "layout_changed": layout_changed,
        "sections_reordered": sections_reordered,
        "expected_order": expected_order,
        "actual_order": actual_order,
        "expected_stable_order": expected_stable_order,
        "actual_stable_order": actual_stable_order,
        "unexpected_stable_sections": unexpected_stable_sections,
        "missing_stable_sections": missing_stable_sections,
        "warning": warning,
    }


def diagnose_cache_miss(
    *,
    expectation: dict[str, Any],
    actual_rendered_prefix: str,
    model: str,
    api_response: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(expectation, dict):
        raise ValueError("diagnose_cache_miss requires dict 'expectation'")
    if not isinstance(actual_rendered_prefix, str) or not actual_rendered_prefix.strip():
        raise ValueError("diagnose_cache_miss requires non-empty 'actual_rendered_prefix'")
    if not isinstance(api_response, dict):
        raise ValueError("diagnose_cache_miss requires dict 'api_response'")

    profile = get_provider_profile(model)
    try:
        telemetry = summarize_provider_cache_usage(
            model=model,
            api_response=api_response,
            expected_cache_hit=bool(expectation.get("expected_cache_hit", False)),
        )
    except ValueError as exc:
        if "cache read tokens" not in str(exc):
            raise
        total_input_tokens = _find_numeric_field(api_response, _INPUT_TOKEN_FIELDS)
        total_output_tokens = _find_numeric_field(api_response, _OUTPUT_TOKEN_FIELDS)
        if total_input_tokens < 0 or total_output_tokens < 0:
            raise
        telemetry = {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            profile.cache_read_field: 0,
            "cache_hit_detected": False,
            "cache_hit_ratio": 0.0,
        }

    expected_prefix_hash = str(expectation.get("cacheable_prefix_hash") or "")
    expected_prefix = expectation.get("cacheable_prefix")
    actual_prefix_hash = _hash_text(actual_rendered_prefix)
    prefix_changed = expected_prefix_hash != actual_prefix_hash
    uuid_matches = _UUID_PATTERN.findall(actual_rendered_prefix)
    timestamp_matches = _TIMESTAMP_PATTERN.findall(actual_rendered_prefix)
    semantic_equivalence = detect_semantic_equivalence_drift(
        expected_prefix=expected_prefix,
        actual_prefix=actual_rendered_prefix,
    )
    section_interleaving = detect_section_interleaving(
        expected_section_order=expectation.get("expected_section_order"),
        actual_rendered_prefix=actual_rendered_prefix,
    )

    if prefix_changed and semantic_equivalence["semantic_match"]:
        probable_cause = "semantic_equivalence_drift"
        suggested_remediation = (
            "Use canonical whitespace, line endings, and JSON serialization for the stable prefix "
            "so semantically equivalent renders also match byte-for-byte for provider cache reuse."
        )
    elif prefix_changed and section_interleaving["layout_changed"]:
        probable_cause = "section_interleaving"
        suggested_remediation = (
            "Rebuild the prompt with canonical section ordering and keep the set of stable sections "
            "consistent across repeated renders before appending metadata or the immediate query."
        )
    elif prefix_changed and uuid_matches:
        probable_cause = "framework_uuid_injection"
        suggested_remediation = (
            "Disable dynamic ID injection in the cached prompt prefix and move request-specific IDs "
            "to the volatile metadata or user-query tail."
        )
    elif prefix_changed and timestamp_matches:
        probable_cause = "volatile_timestamp_in_prefix"
        suggested_remediation = (
            "Move timestamps out of the cached prompt prefix and into metadata or the immediate "
            "user-query tail."
        )
    elif prefix_changed:
        probable_cause = "prefix_content_changed"
        suggested_remediation = (
            "Re-render against the intended stable prompt version and compare the cached prefix for "
            "unexpected instruction or context changes."
        )
    else:
        probable_cause = "provider_cache_eviction_or_miss"
        suggested_remediation = (
            "The rendered prefix still matches the expected hash. Check provider cache configuration, "
            "cache TTL/eviction behavior, and whether the request was routed to a fresh cache shard."
        )

    actual_cached_tokens = int(telemetry.get(profile.cache_read_field, 0))
    expected_cached_tokens = (
        telemetry["total_input_tokens"] if expectation.get("expected_cache_hit") else 0
    )
    cache_shortfall_tokens = max(expected_cached_tokens - actual_cached_tokens, 0)
    partial_reuse_detected = (
        bool(expectation.get("expected_cache_hit"))
        and actual_cached_tokens > 0
        and cache_shortfall_tokens > 0
    )
    partial_reuse_ratio = (
        round(actual_cached_tokens / expected_cached_tokens, 4)
        if expected_cached_tokens > 0
        else 0.0
    )
    estimated_missed_cache_cost_usd = round(
        cache_shortfall_tokens
        / 1_000_000
        * max(profile.input_cost_per_million - profile.cached_input_cost_per_million, 0.0),
        6,
    )

    if (
        probable_cause == "provider_cache_eviction_or_miss"
        and partial_reuse_detected
        and partial_reuse_ratio < 0.5
    ):
        probable_cause = "partial_cache_reuse_underperformance"
        suggested_remediation = (
            "Partial cache reuse was detected, but cached-token reuse fell materially below the "
            "stable prefix expectation. Check for provider-side shard churn, partial prefix drift, "
            "or cache segmentation that is truncating reuse."
        )

    return {
        "template_name": expectation.get("template_name"),
        "expected_prefix_hash": expected_prefix_hash,
        "actual_prefix_hash": actual_prefix_hash,
        "prefix_changed": prefix_changed,
        "probable_cause": probable_cause,
        "suggested_remediation": suggested_remediation,
        "framework_signature": {
            "has_uuid": bool(uuid_matches),
            "uuid_values": uuid_matches,
            "has_timestamp": bool(timestamp_matches),
            "timestamp_values": timestamp_matches,
        },
        "section_interleaving": section_interleaving,
        "semantic_equivalence": semantic_equivalence,
        "partial_reuse": {
            "partial_reuse_detected": partial_reuse_detected,
            "actual_cached_tokens": actual_cached_tokens,
            "expected_cached_tokens": expected_cached_tokens,
            "cache_shortfall_tokens": cache_shortfall_tokens,
            "partial_reuse_ratio": partial_reuse_ratio,
        },
        "metrics": {
            "total_input_tokens": telemetry["total_input_tokens"],
            "actual_cached_tokens": actual_cached_tokens,
            "expected_cached_tokens": expected_cached_tokens,
            "estimated_missed_cache_cost_usd": estimated_missed_cache_cost_usd,
        },
        "diff_preview": (
            _diff_preview(expected_prefix_hash, actual_rendered_prefix) if prefix_changed else ""
        ),
    }
