"""Handlers for managed prompt template registry operations."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict

from ..metrics import get_metrics
from ..observability import get_observability
from ..prompt_cache_audit import audit_prompt_cacheability
from ..prompt_cache_middleware import PromptCacheMiddleware
from ..prompt_cache_stability_guard import evaluate_prompt_stability
from ..prompt_registry import PromptRegistry

logger = logging.getLogger("semantic-modulator")


def _registry(context: Dict[str, Any]) -> PromptRegistry:
    return context.get("prompt_registry") or PromptRegistry.get_registry()


def _record_metrics(start_time: float, status: str, error_type: str | None = None) -> None:
    metrics = get_metrics()
    try:
        metrics.record_latency("prompt_registry", time.perf_counter() - start_time, "NONE")
        metrics.increment_documents_processed("prompt_registry", "NONE", status)
        if error_type is not None:
            metrics.increment_errors(error_type, "prompt_registry")
    except Exception as exc:
        logger.warning(f"Prompt registry metrics update failed: {exc}")


def _required_string(args: Dict[str, Any], field: str, tool_name: str) -> str:
    value = args.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{tool_name} requires a non-empty '{field}' field")
    return value.strip()


def _optional_string(args: Dict[str, Any], field: str) -> str | None:
    value = args.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"'{field}' must be a string when provided")
    return value


def _optional_variables(args: Dict[str, Any]) -> list[str] | None:
    variables = args.get("variables")
    if variables is None:
        return None
    if not isinstance(variables, list) or not all(isinstance(item, str) for item in variables):
        raise ValueError("'variables' must be a list of strings")
    return variables


def get_prompt_output_fields() -> list[str]:
    return [
        "status",
        "template.name",
        "template.latest_version",
        "template.deployment_labels",
        "resolved_version.version",
        "resolved_version.system_prompt",
        "resolved_version.user_prompt_template",
        "resolved_version.variables",
        "resolved_labels",
        "stable_prefix_analysis.stable_prefix_changed",
        "stable_prefix_analysis.impact",
        "stable_prefix_analysis.stable_fields_changed",
        "stable_prefix_analysis.volatile_fields_changed",
    ]


def get_prompt_deploy_output_fields() -> list[str]:
    return [
        "status",
        "name",
        "label",
        "version",
        "previous_version",
        "allow_stable_prefix_change",
        "deployment_labels",
        "stable_prefix_analysis.stable_prefix_changed",
        "stable_prefix_analysis.impact",
        "stable_prefix_analysis.version_a_hash",
        "stable_prefix_analysis.version_b_hash",
        "expectation_invalidations.stale_expectations",
        "prefix_collisions[].template_name",
    ]


def get_prompt_compare_output_fields() -> list[str]:
    return [
        "status",
        "name",
        "version_a",
        "version_b",
        "changed_fields",
        "stable_prefix_analysis.stable_prefix_changed",
        "stable_prefix_analysis.impact",
        "stable_prefix_analysis.stable_fields_changed",
        "stable_prefix_analysis.volatile_fields_changed",
        "stable_prefix_analysis.version_a_hash",
        "stable_prefix_analysis.version_b_hash",
        "diff",
        "labels_a",
        "labels_b",
    ]


def get_prompt_audit_output_fields() -> list[str]:
    return [
        "status",
        "audit.score",
        "audit.is_cache_friendly",
        "audit.issues[].code",
        "audit.issues[].message",
        "audit.recommended_order",
        "audit.present_order",
        "audit.cacheable_prefix",
        "audit.volatile_suffix",
        "audit.stability_guard.is_stable",
        "audit.stability_guard.stable_prefix_hash",
        "audit.stability_guard.violations[].code",
    ]


def get_prompt_render_output_fields() -> list[str]:
    return [
        "status",
        "rendered.template.name",
        "rendered.resolved_version.version",
        "rendered.resolved_labels",
        "rendered.prompt_id",
        "rendered.sections[].name",
        "rendered.sections[].content",
        "rendered.rendered_variables",
        "rendered.audit.score",
        "rendered.audit.is_cache_friendly",
        "rendered.stability_guard.is_stable",
        "rendered.stability_guard.stable_prefix_hash",
        "rendered.stability_guard.violations[].code",
        "rendered.cacheable_prefix",
        "rendered.volatile_suffix",
        "rendered.rendered_prompt",
    ]


def get_prefix_collision_output_fields() -> list[str]:
    return [
        "status",
        "collision_count",
        "collisions[].prefix_hash",
        "collisions[].templates[].template_name",
        "collisions[].templates[].template_version",
        "collisions[].templates[].resolved_labels",
    ]


async def handle_create_prompt_template(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    start_time = time.perf_counter()
    registry = _registry(context)
    observe = get_observability()
    name = _required_string(args, "name", "create_prompt_template")
    description = _required_string(args, "description", "create_prompt_template")
    system_prompt = _required_string(args, "system_prompt", "create_prompt_template")
    user_prompt_template = _required_string(args, "user_prompt_template", "create_prompt_template")
    variables = _optional_variables(args) or []
    metadata = args.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("'metadata' must be an object when provided")
    deployment_label = _optional_string(args, "deployment_label")

    try:
        with observe.trace("prompt_registry.create", prompt_template=name):
            record = registry.create_template(
                name=name,
                description=description,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                variables=variables,
                metadata=metadata,
                deployment_label=deployment_label,
            )
            resolved = registry.get_template(name)
            stable_prefix_analysis = registry.get_create_stable_prefix_analysis(name)
            observe.set_attribute("prompt.template.name", name)
            observe.set_attribute("prompt.version", record.latest_version().version)
            _record_metrics(start_time, "success")
            return json.dumps(
                {
                    "status": "success",
                    "template": resolved["template"],
                    "resolved_version": resolved["resolved_version"],
                    "resolved_labels": resolved["resolved_labels"],
                    "stable_prefix_analysis": stable_prefix_analysis,
                    "message": f"Created prompt template '{name}'",
                },
                indent=2,
            )
    except Exception as exc:
        _record_metrics(start_time, "failure", type(exc).__name__)
        raise


async def handle_update_prompt_template(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    start_time = time.perf_counter()
    registry = _registry(context)
    observe = get_observability()
    name = _required_string(args, "name", "update_prompt_template")

    try:
        with observe.trace("prompt_registry.update", prompt_template=name):
            version = registry.update_template(
                name,
                description=_optional_string(args, "description"),
                system_prompt=_optional_string(args, "system_prompt"),
                user_prompt_template=_optional_string(args, "user_prompt_template"),
                variables=_optional_variables(args),
                metadata=args.get("metadata"),
                change_note=_optional_string(args, "change_note") or "",
            )
            resolved = registry.get_template(name, version=version.version)
            stable_prefix_analysis = registry.get_update_stable_prefix_analysis(
                name, version.version
            )
            observe.set_attribute("prompt.template.name", name)
            observe.set_attribute("prompt.version", version.version)
            _record_metrics(start_time, "success")
            return json.dumps(
                {
                    "status": "success",
                    "template": resolved["template"],
                    "resolved_version": resolved["resolved_version"],
                    "resolved_labels": resolved["resolved_labels"],
                    "stable_prefix_analysis": stable_prefix_analysis,
                    "message": f"Created prompt template version {version.version} for '{name}'",
                },
                indent=2,
            )
    except Exception as exc:
        _record_metrics(start_time, "failure", type(exc).__name__)
        raise


async def handle_list_prompt_templates(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    start_time = time.perf_counter()
    registry = _registry(context)
    include_versions = bool(args.get("include_versions", False))

    try:
        templates = registry.list_templates(include_versions=include_versions)
        _record_metrics(start_time, "success")
        return json.dumps(
            {
                "status": "success",
                "total_templates": len(templates),
                "templates": templates,
            },
            indent=2,
        )
    except Exception as exc:
        _record_metrics(start_time, "failure", type(exc).__name__)
        raise


async def handle_get_prompt_template(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    start_time = time.perf_counter()
    registry = _registry(context)
    name = _required_string(args, "name", "get_prompt_template")
    version = args.get("version")
    if version is not None and not isinstance(version, int):
        raise ValueError("'version' must be an integer when provided")
    deployment_label = _optional_string(args, "deployment_label")

    try:
        payload = registry.get_template(name, version=version, deployment_label=deployment_label)
        _record_metrics(start_time, "success")
        return json.dumps({"status": "success", **payload}, indent=2)
    except Exception as exc:
        _record_metrics(start_time, "failure", type(exc).__name__)
        raise


async def handle_deploy_prompt_version(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    start_time = time.perf_counter()
    registry = _registry(context)
    observe = get_observability()
    name = _required_string(args, "name", "deploy_prompt_version")
    label = _required_string(args, "deployment_label", "deploy_prompt_version")
    version = args.get("version")
    allow_stable_prefix_change = args.get("allow_stable_prefix_change", False)
    if not isinstance(version, int):
        raise ValueError("deploy_prompt_version requires an integer 'version'")
    if not isinstance(allow_stable_prefix_change, bool):
        raise ValueError("deploy_prompt_version requires boolean 'allow_stable_prefix_change'")

    try:
        with observe.trace(
            "prompt_registry.deploy",
            prompt_template=name,
            prompt_version=version,
            prompt_deployment_label=label,
        ):
            deployment = registry.deploy_version(
                name,
                version,
                label,
                allow_stable_prefix_change=allow_stable_prefix_change,
            )
            deployment["expectation_invalidations"] = (
                PromptCacheMiddleware.invalidate_template_expectations(
                    template_name=name,
                    label=label,
                    previous_version=deployment["previous_version"],
                    new_version=version,
                )
            )
            deployment["prefix_collisions"] = PromptCacheMiddleware.get_cache_siblings(
                args.get("prompt_id", "")
            )
            if not deployment["prefix_collisions"]:
                collision_map = PromptCacheMiddleware.get_prefix_collision_map()
                deployment["prefix_collisions"] = [
                    item
                    for payload in collision_map.values()
                    for item in payload["templates"]
                    if item["template_name"] != name and label in item["resolved_labels"]
                ]
            _record_metrics(start_time, "success")
            return json.dumps(
                {
                    "status": "success",
                    **deployment,
                    "message": f"Deployed '{name}' version {version} to label '{label}'",
                },
                indent=2,
            )
    except Exception as exc:
        _record_metrics(start_time, "failure", type(exc).__name__)
        raise


async def handle_compare_prompt_versions(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    start_time = time.perf_counter()
    registry = _registry(context)
    name = _required_string(args, "name", "compare_prompt_versions")
    version_a = args.get("version_a")
    version_b = args.get("version_b")
    if not isinstance(version_a, int) or not isinstance(version_b, int):
        raise ValueError("compare_prompt_versions requires integer 'version_a' and 'version_b'")

    try:
        comparison = registry.compare_versions(name, version_a, version_b)
        _record_metrics(start_time, "success")
        return json.dumps({"status": "success", **comparison}, indent=2)
    except Exception as exc:
        _record_metrics(start_time, "failure", type(exc).__name__)
        raise


async def handle_audit_prompt_cacheability(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    start_time = time.perf_counter()
    sections = args.get("sections")
    max_stable_prefix_chars = args.get("max_stable_prefix_chars", 5000)
    if not isinstance(sections, list):
        raise ValueError("audit_prompt_cacheability requires list 'sections'")
    if not isinstance(max_stable_prefix_chars, int) or max_stable_prefix_chars <= 0:
        raise ValueError("audit_prompt_cacheability requires integer 'max_stable_prefix_chars' > 0")

    try:
        audit = audit_prompt_cacheability(sections)
        audit["stability_guard"] = evaluate_prompt_stability(
            sections,
            max_stable_prefix_chars=max_stable_prefix_chars,
        )
        _record_metrics(start_time, "success")
        return json.dumps({"status": "success", "audit": audit}, indent=2)
    except Exception as exc:
        _record_metrics(start_time, "failure", type(exc).__name__)
        raise


async def handle_render_prompt_template(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    start_time = time.perf_counter()
    registry = _registry(context)
    name = _required_string(args, "name", "render_prompt_template")
    variables = args.get("variables") or {}
    version = args.get("version")
    deployment_label = _optional_string(args, "deployment_label")
    metadata = args.get("metadata")
    enforce_stability = args.get("enforce_stability", False)
    max_stable_prefix_chars = args.get("max_stable_prefix_chars", 5000)

    if not isinstance(variables, dict):
        raise ValueError("'variables' must be an object when provided")
    if version is not None and not isinstance(version, int):
        raise ValueError("'version' must be an integer when provided")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("'metadata' must be an object when provided")
    if not isinstance(enforce_stability, bool):
        raise ValueError("'enforce_stability' must be a boolean when provided")
    if not isinstance(max_stable_prefix_chars, int) or max_stable_prefix_chars <= 0:
        raise ValueError("'max_stable_prefix_chars' must be an integer > 0 when provided")

    for list_field in ("few_shot_examples", "chat_history"):
        list_value = args.get(list_field)
        if list_value is not None and (
            not isinstance(list_value, list)
            or not all(isinstance(item, str) for item in list_value)
        ):
            raise ValueError(f"'{list_field}' must be a list of strings when provided")

    try:
        rendered = registry.render_prompt(
            name,
            variables=variables,
            version=version,
            deployment_label=deployment_label,
            tool_definitions=_optional_string(args, "tool_definitions"),
            rag_context=_optional_string(args, "rag_context"),
            few_shot_examples=args.get("few_shot_examples"),
            chat_history=args.get("chat_history"),
            metadata=metadata,
            max_stable_prefix_chars=max_stable_prefix_chars,
        )
        if enforce_stability and not rendered["stability_guard"]["is_stable"]:
            raise ValueError(
                "Prompt stability guard rejected the rendered prompt. "
                f"Violations: {', '.join(item['code'] for item in rendered['stability_guard']['violations'])}"
            )
        prompt_id = PromptCacheMiddleware.record_expectation(name, rendered)
        rendered["prompt_id"] = prompt_id
        _record_metrics(start_time, "success")
        return json.dumps(
            {"status": "success", "prompt_id": prompt_id, "rendered": rendered}, indent=2
        )
    except Exception as exc:
        _record_metrics(start_time, "failure", type(exc).__name__)
        raise


async def handle_list_prefix_collisions(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    start_time = time.perf_counter()

    try:
        collision_map = PromptCacheMiddleware.get_prefix_collision_map()
        collisions = list(collision_map.values())
        _record_metrics(start_time, "success")
        return json.dumps(
            {
                "status": "success",
                "collision_count": len(collisions),
                "collisions": collisions,
            },
            indent=2,
        )
    except Exception as exc:
        _record_metrics(start_time, "failure", type(exc).__name__)
        raise
