"""Bridge rendered prompt expectations to provider cache telemetry validation."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
import sys
from datetime import datetime, timezone

if sys.version_info >= (3, 11):
    from datetime import UTC
else:
    UTC = timezone.utc
from typing import Any
from uuid import uuid4

from .cache_diagnostics import (
    detect_semantic_equivalence_drift,
    detect_section_interleaving,
    diagnose_cache_miss,
    extract_section_order,
)
from .model_optimizer import summarize_provider_cache_usage


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _first_difference_index(expected: str, actual: str) -> int | None:
    for index, (expected_char, actual_char) in enumerate(zip(expected, actual)):
        if expected_char != actual_char:
            return index
    if len(expected) != len(actual):
        return min(len(expected), len(actual))
    return None


def _extract_cached_tokens(telemetry: dict[str, Any]) -> int:
    cache_read_field = telemetry.get("cache_read_field")
    candidate_fields: list[str] = []
    if isinstance(cache_read_field, str) and cache_read_field:
        candidate_fields.append(cache_read_field)
    candidate_fields.extend(
        [
            "cache_read_input_tokens",
            "cached_tokens",
            "cachedContentTokenCount",
            "cached_content_token_count",
            "cache_hit_input_tokens",
        ]
    )
    seen: set[str] = set()
    for field_name in candidate_fields:
        if field_name in seen:
            continue
        seen.add(field_name)
        value = telemetry.get(field_name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return 0


@dataclass
class PromptCacheExpectation:
    """Cache expectations derived from a rendered prompt."""

    prompt_id: str
    template_name: str
    template_version: int | None
    resolved_labels: tuple[str, ...]
    audit_score: int
    is_cache_friendly: bool
    expected_cache_hit: bool
    cacheable_prefix: str
    cacheable_prefix_hash: str
    expected_section_order: tuple[str, ...]
    created_at: str
    is_stale: bool = False
    stale_reason: str | None = None
    stale_label: str | None = None
    stale_previous_version: int | None = None
    stale_replaced_by_version: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["resolved_labels"] = list(self.resolved_labels)
        return payload


class PromptCacheMiddleware:
    """Track prompt cache expectations across render and telemetry steps."""

    _expectations: dict[str, PromptCacheExpectation] = {}
    _sessions: dict[str, dict[str, Any]] = {}
    _prefix_collisions: dict[str, dict[str, Any]] = {}
    _deployment_cache_health: dict[tuple[str, str], dict[str, Any]] = {}
    _prefix_integrity_history: dict[str, dict[str, int]] = {}
    _cache_creation_history: dict[tuple[str, str, str], dict[str, Any]] = {}
    _cache_health_baseline_min_events = 3
    _cache_health_degradation_threshold = 0.3
    _cache_health_coherence_threshold = 0.5
    _prefix_drift_systematic_threshold = 0.5
    _cache_creation_churn_threshold = 2

    @classmethod
    def reset(cls) -> None:
        cls._expectations = {}
        cls._sessions = {}
        cls._prefix_collisions = {}
        cls._deployment_cache_health = {}
        cls._prefix_integrity_history = {}
        cls._cache_creation_history = {}

    @classmethod
    def record_expectation(cls, template_name: str, rendered: dict[str, Any]) -> str:
        if not isinstance(rendered, dict):
            raise ValueError("record_expectation requires dict 'rendered'")

        audit = rendered.get("audit")
        cacheable_prefix = rendered.get("cacheable_prefix")
        volatile_suffix = rendered.get("volatile_suffix")
        resolved_version = rendered.get("resolved_version") or {}
        resolved_labels = rendered.get("resolved_labels") or []
        if not isinstance(audit, dict):
            raise ValueError("record_expectation requires dict 'rendered.audit'")
        if not isinstance(cacheable_prefix, str):
            raise ValueError("record_expectation requires string 'rendered.cacheable_prefix'")
        if volatile_suffix is not None and not isinstance(volatile_suffix, str):
            raise ValueError("record_expectation requires string 'rendered.volatile_suffix'")
        if not isinstance(resolved_version, dict):
            raise ValueError("record_expectation requires dict 'rendered.resolved_version'")
        if not isinstance(resolved_labels, list) or not all(
            isinstance(label, str) for label in resolved_labels
        ):
            raise ValueError("record_expectation requires list 'rendered.resolved_labels'")

        expected_section_order = rendered.get("sections")
        if isinstance(expected_section_order, list):
            normalized_expected_section_order = tuple(
                item["name"]
                for item in expected_section_order
                if isinstance(item, dict) and isinstance(item.get("name"), str)
            )
        else:
            rendered_prefix = cacheable_prefix
            if isinstance(volatile_suffix, str) and volatile_suffix:
                rendered_prefix = f"{cacheable_prefix}\n\n{volatile_suffix}"
            normalized_expected_section_order = tuple(extract_section_order(rendered_prefix))

        prompt_id = f"prompt-cache-{uuid4().hex}"
        expectation = PromptCacheExpectation(
            prompt_id=prompt_id,
            template_name=template_name,
            template_version=(
                int(resolved_version["version"])
                if isinstance(resolved_version.get("version"), int)
                else None
            ),
            resolved_labels=tuple(sorted(resolved_labels)),
            audit_score=int(audit.get("score", 0)),
            is_cache_friendly=bool(audit.get("is_cache_friendly", False)),
            expected_cache_hit=bool(audit.get("is_cache_friendly", False)),
            cacheable_prefix=cacheable_prefix,
            cacheable_prefix_hash=_hash_text(cacheable_prefix),
            expected_section_order=normalized_expected_section_order,
            created_at=_utc_now(),
        )
        cls._expectations[prompt_id] = expectation
        collision_bucket = cls._prefix_collisions.setdefault(
            expectation.cacheable_prefix_hash,
            {"prefix_hash": expectation.cacheable_prefix_hash, "templates": []},
        )
        collision_bucket["templates"] = [
            item for item in collision_bucket["templates"] if item["prompt_id"] != prompt_id
        ]
        collision_bucket["templates"].append(
            {
                "template_name": expectation.template_name,
                "template_version": expectation.template_version,
                "resolved_labels": list(expectation.resolved_labels),
                "prompt_id": prompt_id,
            }
        )
        return prompt_id

    @classmethod
    def get_expectation(cls, prompt_id: str) -> dict[str, Any] | None:
        expectation = cls._expectations.get(prompt_id)
        if expectation is None:
            return None
        return expectation.to_dict()

    @classmethod
    def record_cache_health(
        cls,
        *,
        prompt_id: str,
        telemetry: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not isinstance(telemetry, dict):
            raise ValueError("record_cache_health requires dict 'telemetry'")

        expectation = cls._expectations.get(prompt_id)
        if expectation is None:
            raise ValueError(f"Unknown prompt_id '{prompt_id}'")

        hit = bool(telemetry.get("cache_hit_detected", False))
        snapshots: list[dict[str, Any]] = []
        for label in expectation.resolved_labels:
            key = (expectation.template_name, label)
            health = cls._deployment_cache_health.setdefault(
                key,
                {
                    "template_name": expectation.template_name,
                    "label": label,
                    "prefix_hash": expectation.cacheable_prefix_hash,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "total_events": 0,
                    "baseline_hit_ratio": None,
                    "current_hit_ratio": 0.0,
                    "degraded": False,
                    "degradation_amount": 0.0,
                    "warnings": [],
                    "coherence": {},
                },
            )
            if hit:
                health["cache_hits"] += 1
            else:
                health["cache_misses"] += 1
            health["total_events"] = health["cache_hits"] + health["cache_misses"]
            health["current_hit_ratio"] = round(health["cache_hits"] / health["total_events"], 4)

            if (
                health["baseline_hit_ratio"] is None
                and health["total_events"] >= cls._cache_health_baseline_min_events
            ):
                health["baseline_hit_ratio"] = health["current_hit_ratio"]

            baseline = health["baseline_hit_ratio"]
            degradation_amount = 0.0
            degraded = False
            if baseline is not None:
                degradation_amount = round(max(0.0, baseline - health["current_hit_ratio"]), 4)
                degraded = degradation_amount > cls._cache_health_degradation_threshold

            health["degradation_amount"] = degradation_amount
            health["degraded"] = degraded
            baseline_display = baseline if baseline is not None else health["current_hit_ratio"]
            degradation_warning = (
                f"Cache hit ratio degraded for template '{expectation.template_name}' "
                f"label '{label}' from {baseline_display:.2f} "
                f"to {health['current_hit_ratio']:.2f}."
            )
            health["coherence"] = cls.build_cache_health_coherence(
                template_name=expectation.template_name,
                label=label,
                prefix_hash=expectation.cacheable_prefix_hash,
            )
            warnings: list[str] = []
            if degraded:
                warnings.append(degradation_warning)
            if health["coherence"]["warning"] is not None:
                warnings.append(health["coherence"]["warning"])
            health["warnings"] = warnings

            snapshots.append(cls.get_deployment_cache_health(expectation.template_name, label))

        return snapshots

    @classmethod
    def build_cache_health_coherence(
        cls,
        *,
        template_name: str,
        label: str,
        prefix_hash: str,
    ) -> dict[str, Any]:
        current = cls._deployment_cache_health.get((template_name, label))
        if current is None or current["total_events"] < cls._cache_health_baseline_min_events:
            return {
                "skew_detected": False,
                "peer_labels": [],
                "max_hit_ratio_delta": 0.0,
                "warning": None,
            }

        peer_records = [
            health
            for (peer_template, peer_label), health in cls._deployment_cache_health.items()
            if peer_template == template_name
            and peer_label != label
            and health.get("prefix_hash") == prefix_hash
            and health["total_events"] >= cls._cache_health_baseline_min_events
        ]
        if not peer_records:
            return {
                "skew_detected": False,
                "peer_labels": [],
                "max_hit_ratio_delta": 0.0,
                "warning": None,
            }

        max_hit_ratio_delta = round(
            max(
                abs(current["current_hit_ratio"] - peer["current_hit_ratio"])
                for peer in peer_records
            ),
            4,
        )
        skew_detected = max_hit_ratio_delta >= cls._cache_health_coherence_threshold
        warning = None
        if skew_detected:
            warning = (
                f"Cross-label cache skew detected for template '{template_name}' label '{label}'. "
                f"Labels sharing the same stable prefix differ by up to {max_hit_ratio_delta:.2f} hit ratio."
            )
        return {
            "skew_detected": skew_detected,
            "peer_labels": sorted(peer["label"] for peer in peer_records),
            "max_hit_ratio_delta": max_hit_ratio_delta,
            "warning": warning,
        }

    @classmethod
    def get_deployment_cache_health(cls, template_name: str, label: str) -> dict[str, Any] | None:
        health = cls._deployment_cache_health.get((template_name, label))
        if health is None:
            return None
        return {
            "template_name": health["template_name"],
            "label": health["label"],
            "cache_hits": health["cache_hits"],
            "cache_misses": health["cache_misses"],
            "total_events": health["total_events"],
            "baseline_hit_ratio": health["baseline_hit_ratio"],
            "current_hit_ratio": health["current_hit_ratio"],
            "degraded": health["degraded"],
            "degradation_amount": health["degradation_amount"],
            "warnings": list(health["warnings"]),
            "coherence": dict(health["coherence"]),
        }

    @classmethod
    def build_prefix_integrity(
        cls,
        *,
        expectation: PromptCacheExpectation,
        actual_rendered_prefix: str,
    ) -> dict[str, Any]:
        actual_prefix_hash = _hash_text(actual_rendered_prefix)
        prefix_changed = actual_prefix_hash != expectation.cacheable_prefix_hash
        semantic_equivalence = detect_semantic_equivalence_drift(
            expected_prefix=expectation.cacheable_prefix,
            actual_prefix=actual_rendered_prefix,
        )
        first_difference_index = None
        warning = None
        if prefix_changed:
            first_difference_index = _first_difference_index(
                expectation.cacheable_prefix_hash,
                actual_prefix_hash,
            )
            warning = (
                "Actual rendered prefix diverged from the expected stable prefix. "
                "Cacheability may be reduced even if provider telemetry still reports reuse."
            )
            if semantic_equivalence["warning"] is not None:
                warning = f"{warning} {semantic_equivalence['warning']}"

        return {
            "expected_prefix": expectation.cacheable_prefix,
            "expected_prefix_hash": expectation.cacheable_prefix_hash,
            "actual_prefix_hash": actual_prefix_hash,
            "prefix_changed": prefix_changed,
            "first_difference_index": first_difference_index,
            "semantic_equivalence": semantic_equivalence,
            "warning": warning,
        }

    @classmethod
    def track_prefix_integrity_trend(
        cls,
        *,
        prompt_id: str,
        prefix_changed: bool,
    ) -> dict[str, Any]:
        history = cls._prefix_integrity_history.setdefault(
            prompt_id,
            {
                "total_observations": 0,
                "drift_observations": 0,
            },
        )
        history["total_observations"] += 1
        if prefix_changed:
            history["drift_observations"] += 1

        drift_frequency = round(
            history["drift_observations"] / history["total_observations"],
            4,
        )
        systematic_drift_detected = (
            history["total_observations"] >= 3
            and drift_frequency >= cls._prefix_drift_systematic_threshold
        )
        warning = None
        if systematic_drift_detected:
            warning = (
                "Repeated prefix drift detected for this rendered prompt. "
                "Re-audit the stable prefix and investigate hidden dynamic content."
            )

        return {
            "total_observations": history["total_observations"],
            "drift_observations": history["drift_observations"],
            "drift_frequency": drift_frequency,
            "systematic_drift_detected": systematic_drift_detected,
            "warning": warning,
        }

    @classmethod
    def track_cache_creation_churn(
        cls,
        *,
        expectation: PromptCacheExpectation,
        telemetry: dict[str, Any],
    ) -> dict[str, Any] | None:
        creation_tokens = int(telemetry.get("cache_creation_input_tokens", 0) or 0)
        if not expectation.expected_cache_hit:
            return None

        labels = expectation.resolved_labels or ("unlabeled",)
        snapshots: list[dict[str, Any]] = []
        for label in labels:
            key = (
                expectation.template_name,
                label,
                expectation.cacheable_prefix_hash,
            )
            history = cls._cache_creation_history.setdefault(
                key,
                {
                    "template_name": expectation.template_name,
                    "label": label,
                    "prefix_hash": expectation.cacheable_prefix_hash,
                    "total_observations": 0,
                    "creation_events": 0,
                    "creation_token_total": 0,
                },
            )
            history["total_observations"] += 1
            if creation_tokens > 0:
                history["creation_events"] += 1
                history["creation_token_total"] += creation_tokens
            snapshots.append(dict(history))

        creation_events = max(snapshot["creation_events"] for snapshot in snapshots)
        total_observations = max(snapshot["total_observations"] for snapshot in snapshots)
        creation_token_total = max(snapshot["creation_token_total"] for snapshot in snapshots)
        churn_detected = (
            creation_tokens > 0 and creation_events >= cls._cache_creation_churn_threshold
        )
        warning = None
        if churn_detected:
            labels_display = ", ".join(sorted(snapshot["label"] for snapshot in snapshots))
            warning = (
                f"Repeated cache creation churn detected for template "
                f"'{expectation.template_name}' labels [{labels_display}]. "
                "The provider keeps rebuilding cached prompt state for a stable prefix "
                "instead of reusing it."
            )

        return {
            "churn_detected": churn_detected,
            "template_name": expectation.template_name,
            "labels": [snapshot["label"] for snapshot in snapshots],
            "creation_events": creation_events,
            "total_observations": total_observations,
            "creation_token_total": creation_token_total,
            "latest_creation_input_tokens": creation_tokens,
            "warning": warning,
        }

    @classmethod
    def record_session_telemetry(
        cls,
        *,
        session_id: str,
        prompt_id: str,
        telemetry: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("record_session_telemetry requires non-empty 'session_id'")
        if not isinstance(telemetry, dict):
            raise ValueError("record_session_telemetry requires dict 'telemetry'")

        expectation = cls._expectations.get(prompt_id)
        if expectation is None:
            raise ValueError(f"Unknown prompt_id '{prompt_id}'")

        session_key = session_id.strip()
        session = cls._sessions.setdefault(
            session_key,
            {
                "session_id": session_key,
                "cache_hits": 0,
                "cache_misses": 0,
                "total_cached_tokens": 0,
                "total_input_tokens": 0,
                "estimated_uncached_input_cost_usd": 0.0,
                "estimated_cache_savings_usd": 0.0,
                "templates": {},
                "labels": {},
                "warnings": [],
            },
        )
        cache_health = cls.record_cache_health(prompt_id=prompt_id, telemetry=telemetry)

        hit = bool(telemetry.get("cache_hit_detected", False))
        template_metrics = session["templates"].setdefault(
            expectation.template_name, {"hits": 0, "misses": 0}
        )
        if hit:
            session["cache_hits"] += 1
            template_metrics["hits"] += 1
        else:
            session["cache_misses"] += 1
            template_metrics["misses"] += 1

        cached_tokens = _extract_cached_tokens(telemetry)
        total_input_tokens = int(telemetry.get("total_input_tokens", 0))
        session["total_cached_tokens"] += cached_tokens
        session["total_input_tokens"] += total_input_tokens
        session["estimated_uncached_input_cost_usd"] = round(
            session["estimated_uncached_input_cost_usd"]
            + float(telemetry.get("estimated_uncached_input_cost_usd", 0.0)),
            6,
        )
        session["estimated_cache_savings_usd"] = round(
            session["estimated_cache_savings_usd"]
            + float(telemetry.get("estimated_cache_savings_usd", 0.0)),
            6,
        )

        for label in expectation.resolved_labels:
            label_metrics = session["labels"].setdefault(label, {"hits": 0, "misses": 0})
            if hit:
                label_metrics["hits"] += 1
            else:
                label_metrics["misses"] += 1

        total_events = session["cache_hits"] + session["cache_misses"]
        if total_events:
            session["cache_hit_ratio"] = round(session["cache_hits"] / total_events, 4)
        if template_metrics["misses"] >= 3:
            warning = (
                f"Template '{expectation.template_name}' has {template_metrics['misses']} cache misses "
                f"in session '{session_key}'."
            )
            if warning not in session["warnings"]:
                session["warnings"].append(warning)

        return {
            "session_id": session["session_id"],
            "cache_hits": session["cache_hits"],
            "cache_misses": session["cache_misses"],
            "cache_hit_ratio": session.get("cache_hit_ratio", 0.0),
            "total_cached_tokens": session["total_cached_tokens"],
            "total_input_tokens": session["total_input_tokens"],
            "estimated_uncached_input_cost_usd": session["estimated_uncached_input_cost_usd"],
            "estimated_cache_savings_usd": session["estimated_cache_savings_usd"],
            "templates": session["templates"],
            "labels": session["labels"],
            "warnings": list(session["warnings"]),
            "cache_health": cache_health,
        }

    @classmethod
    def get_session_metrics(cls, session_id: str) -> dict[str, Any] | None:
        session = cls._sessions.get(session_id)
        if session is None:
            return None
        return {
            "session_id": session["session_id"],
            "cache_hits": session["cache_hits"],
            "cache_misses": session["cache_misses"],
            "cache_hit_ratio": session.get("cache_hit_ratio", 0.0),
            "total_cached_tokens": session["total_cached_tokens"],
            "total_input_tokens": session["total_input_tokens"],
            "estimated_uncached_input_cost_usd": session["estimated_uncached_input_cost_usd"],
            "estimated_cache_savings_usd": session["estimated_cache_savings_usd"],
            "templates": {name: dict(values) for name, values in session["templates"].items()},
            "labels": {name: dict(values) for name, values in session["labels"].items()},
            "warnings": list(session["warnings"]),
        }

    @classmethod
    def get_prefix_collision_map(cls) -> dict[str, Any]:
        return {
            prefix_hash: {
                "prefix_hash": payload["prefix_hash"],
                "templates": [dict(item) for item in payload["templates"]],
                "collision_count": len(payload["templates"]),
            }
            for prefix_hash, payload in cls._prefix_collisions.items()
            if len(payload["templates"]) > 1
        }

    @classmethod
    def get_cache_siblings(cls, prompt_id: str) -> list[dict[str, Any]]:
        expectation = cls._expectations.get(prompt_id)
        if expectation is None:
            return []
        collision_bucket = cls._prefix_collisions.get(expectation.cacheable_prefix_hash, {})
        templates = collision_bucket.get("templates", [])
        return [dict(item) for item in templates if item["prompt_id"] != prompt_id]

    @classmethod
    def build_sibling_coherence(cls, prompt_id: str) -> dict[str, Any] | None:
        siblings = cls.get_cache_siblings(prompt_id)
        if not siblings:
            return None

        stale_siblings: list[dict[str, Any]] = []
        for sibling in siblings:
            sibling_expectation = cls._expectations.get(sibling["prompt_id"])
            if sibling_expectation is None or not sibling_expectation.is_stale:
                continue
            stale_siblings.append(
                {
                    "template_name": sibling_expectation.template_name,
                    "label": sibling_expectation.stale_label,
                    "rendered_version": sibling_expectation.stale_previous_version,
                    "current_version": sibling_expectation.stale_replaced_by_version,
                    "reason": sibling_expectation.stale_reason,
                }
            )

        warning = None
        if stale_siblings:
            warning = (
                "Stale sibling renders detected for a shared stable prefix. "
                "Cache-hit attribution may be ambiguous until those sibling templates are re-rendered."
            )

        return {
            "coherence_valid": len(stale_siblings) == 0,
            "stale_siblings": stale_siblings,
            "warning": warning,
        }

    @classmethod
    def invalidate_template_expectations(
        cls,
        *,
        template_name: str,
        label: str,
        previous_version: int | None,
        new_version: int,
    ) -> dict[str, Any]:
        stale_expectations = 0
        if previous_version is None:
            return {
                "template_name": template_name,
                "label": label,
                "previous_version": previous_version,
                "new_version": new_version,
                "stale_expectations": stale_expectations,
            }

        for expectation in cls._expectations.values():
            if expectation.template_name != template_name:
                continue
            if expectation.template_version != previous_version:
                continue
            if label not in expectation.resolved_labels:
                continue
            expectation.is_stale = True
            expectation.stale_reason = "redeployed_label"
            expectation.stale_label = label
            expectation.stale_previous_version = previous_version
            expectation.stale_replaced_by_version = new_version
            stale_expectations += 1

        return {
            "template_name": template_name,
            "label": label,
            "previous_version": previous_version,
            "new_version": new_version,
            "stale_expectations": stale_expectations,
        }

    @classmethod
    def validate_provider_response(
        cls,
        *,
        prompt_id: str,
        model: str,
        api_response: dict[str, Any],
        actual_rendered_prefix: str | None = None,
    ) -> dict[str, Any]:
        expectation = cls._expectations.get(prompt_id)
        if expectation is None:
            return {"status": "missing_expectation", "prompt_id": prompt_id}

        telemetry = summarize_provider_cache_usage(
            model=model,
            api_response=api_response,
            file_id=prompt_id,
            expected_cache_hit=expectation.expected_cache_hit,
        )
        validation = {
            "status": "validated",
            "prompt_id": prompt_id,
            "expectation": expectation.to_dict(),
            "cache_hit_detected": telemetry["cache_hit_detected"],
            "cache_hit_ratio": telemetry["cache_hit_ratio"],
        }
        cache_creation_churn = cls.track_cache_creation_churn(
            expectation=expectation,
            telemetry=telemetry,
        )
        if cache_creation_churn is not None:
            validation["cache_creation_churn"] = cache_creation_churn
        sibling_coherence = cls.build_sibling_coherence(prompt_id)
        if sibling_coherence is not None:
            validation["sibling_coherence"] = sibling_coherence
        if expectation.is_stale:
            validation["status"] = "validated_against_stale_expectation"
            validation["stale_expectation"] = {
                "template_name": expectation.template_name,
                "label": expectation.stale_label,
                "rendered_version": expectation.stale_previous_version,
                "current_version": expectation.stale_replaced_by_version,
                "reason": expectation.stale_reason,
            }
        if actual_rendered_prefix is not None:
            validation["section_interleaving"] = detect_section_interleaving(
                expected_section_order=expectation.expected_section_order,
                actual_rendered_prefix=actual_rendered_prefix,
            )
            validation["prefix_integrity"] = cls.build_prefix_integrity(
                expectation=expectation,
                actual_rendered_prefix=actual_rendered_prefix,
            )
            validation["prefix_integrity"]["trend"] = cls.track_prefix_integrity_trend(
                prompt_id=prompt_id,
                prefix_changed=validation["prefix_integrity"]["prefix_changed"],
            )
            diagnostic = None
            if expectation.expected_cache_hit:
                diagnostic = diagnose_cache_miss(
                    expectation=expectation.to_dict(),
                    actual_rendered_prefix=actual_rendered_prefix,
                    model=model,
                    api_response=api_response,
                )
            if diagnostic is not None and (
                not telemetry["cache_hit_detected"]
                or diagnostic["partial_reuse"]["partial_reuse_detected"]
            ):
                validation["diagnostic"] = diagnostic
        section_interleaving_warning = None
        if "section_interleaving" in validation:
            section_interleaving_warning = validation["section_interleaving"]["warning"]
        semantic_equivalence_warning = None
        if "prefix_integrity" in validation:
            semantic_equivalence_warning = validation["prefix_integrity"]["semantic_equivalence"][
                "warning"
            ]
        partial_reuse_warning = None
        if (
            "diagnostic" in validation
            and validation["diagnostic"]["partial_reuse"]["partial_reuse_detected"]
        ):
            partial_reuse_warning = (
                "Partial cache reuse detected, but cached-token reuse materially underperformed "
                "the stable-prefix expectation."
            )
        cache_creation_churn_warning = None
        if "cache_creation_churn" in validation:
            cache_creation_churn_warning = validation["cache_creation_churn"]["warning"]

        if "warning" in telemetry:
            warning = (
                f"{telemetry['warning']} Stable prefix hash: {expectation.cacheable_prefix_hash}."
            )
            if section_interleaving_warning is not None:
                warning = f"{warning} {section_interleaving_warning}"
            if semantic_equivalence_warning is not None:
                warning = f"{warning} {semantic_equivalence_warning}"
            if partial_reuse_warning is not None:
                warning = f"{warning} {partial_reuse_warning}"
            if cache_creation_churn_warning is not None:
                warning = f"{warning} {cache_creation_churn_warning}"
            validation["warning"] = warning
        elif cache_creation_churn_warning is not None:
            validation["warning"] = cache_creation_churn_warning
        elif partial_reuse_warning is not None:
            validation["warning"] = partial_reuse_warning
        elif semantic_equivalence_warning is not None:
            validation["warning"] = semantic_equivalence_warning
        elif section_interleaving_warning is not None:
            validation["warning"] = validation["section_interleaving"]["warning"]
        elif sibling_coherence is not None and sibling_coherence["warning"] is not None:
            validation["warning"] = sibling_coherence["warning"]
        return validation
