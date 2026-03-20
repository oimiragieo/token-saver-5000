"""Versioned prompt template registry with deployment labels."""

from __future__ import annotations

import difflib
import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from typing import Any, Dict, List, Optional

from .compression_presets import list_presets
from .prompt_cache_audit import audit_prompt_cacheability
from .prompt_cache_stability_guard import evaluate_prompt_stability


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _normalize_variables(variables: Optional[List[str]]) -> tuple[str, ...]:
    if not variables:
        return ()

    normalized: List[str] = []
    seen: set[str] = set()
    for variable in variables:
        candidate = str(variable).strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
    return tuple(normalized)


def _stable_prefix_text(version: "PromptVersion") -> str:
    return f"[system_instructions]\n{version.system_prompt}"


def _stable_prefix_hash(version: "PromptVersion") -> str:
    return hashlib.sha256(_stable_prefix_text(version).encode("utf-8")).hexdigest()


def _initial_stable_prefix_analysis(version: "PromptVersion") -> Dict[str, Any]:
    prefix_hash = _stable_prefix_hash(version)
    return {
        "stable_sections": ["system_instructions"],
        "version_a_hash": None,
        "version_b_hash": prefix_hash,
        "stable_prefix_changed": True,
        "stable_fields_changed": ["system_prompt"],
        "volatile_fields_changed": ["user_prompt_template", "variables", "metadata"],
        "impact": "initial_prefix_created",
    }


@dataclass(frozen=True)
class PromptVersion:
    """Immutable prompt version payload."""

    version: int
    system_prompt: str
    user_prompt_template: str
    variables: tuple[str, ...]
    created_at: str
    change_note: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_preset: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "system_prompt": self.system_prompt,
            "user_prompt_template": self.user_prompt_template,
            "variables": list(self.variables),
            "created_at": self.created_at,
            "change_note": self.change_note,
            "metadata": deepcopy(self.metadata),
            "source_preset": self.source_preset,
        }


@dataclass
class PromptTemplateRecord:
    """Mutable prompt template record containing all versions and deployments."""

    name: str
    description: str
    created_at: str
    updated_at: str
    versions: Dict[int, PromptVersion]
    deployment_labels: Dict[str, int] = field(default_factory=dict)

    def latest_version(self) -> PromptVersion:
        return self.versions[max(self.versions)]

    def labels_for_version(self, version: int) -> List[str]:
        return sorted(
            label
            for label, deployed_version in self.deployment_labels.items()
            if deployed_version == version
        )

    def to_dict(self, include_versions: bool = False) -> Dict[str, Any]:
        latest = self.latest_version()
        payload = {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "latest_version": latest.version,
            "version_count": len(self.versions),
            "deployment_labels": dict(sorted(self.deployment_labels.items())),
            "latest_variables": list(latest.variables),
            "source_preset": latest.source_preset,
        }
        if include_versions:
            payload["versions"] = [self.versions[key].to_dict() for key in sorted(self.versions)]
        return payload


class PromptRegistry:
    """In-memory prompt registry seeded from compression presets."""

    _instance: Optional["PromptRegistry"] = None

    def __init__(self, seed_defaults: bool = True):
        self._lock = RLock()
        self._templates: Dict[str, PromptTemplateRecord] = {}
        if seed_defaults:
            self._seed_from_presets()

    @classmethod
    def get_registry(cls) -> "PromptRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_singleton(cls) -> None:
        cls._instance = None

    def _seed_from_presets(self) -> None:
        for preset in list_presets():
            seed = preset.to_prompt_seed()
            if seed["name"] in self._templates:
                continue
            self.create_template(
                name=seed["name"],
                description=seed["description"],
                system_prompt=seed["system_prompt"],
                user_prompt_template=seed["user_prompt_template"],
                variables=seed["variables"],
                metadata=seed["metadata"],
                source_preset=seed["source_preset"],
                deployment_label=seed["deployment_label"],
            )

    def create_template(
        self,
        *,
        name: str,
        description: str,
        system_prompt: str,
        user_prompt_template: str,
        variables: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source_preset: Optional[str] = None,
        deployment_label: Optional[str] = None,
    ) -> PromptTemplateRecord:
        with self._lock:
            if name in self._templates:
                raise ValueError(f"Prompt template '{name}' already exists")

            timestamp = _utc_now()
            version = PromptVersion(
                version=1,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                variables=_normalize_variables(variables),
                created_at=timestamp,
                metadata=deepcopy(metadata or {}),
                source_preset=source_preset,
            )
            record = PromptTemplateRecord(
                name=name,
                description=description,
                created_at=timestamp,
                updated_at=timestamp,
                versions={1: version},
                deployment_labels={deployment_label: 1} if deployment_label else {},
            )
            self._templates[name] = record
            return record

    def update_template(
        self,
        name: str,
        *,
        description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        variables: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        change_note: str = "",
    ) -> PromptVersion:
        with self._lock:
            record = self._require_template(name)
            latest = record.latest_version()
            if all(
                value is None
                for value in (description, system_prompt, user_prompt_template, variables, metadata)
            ):
                raise ValueError("At least one field must change when updating a prompt template")

            merged_metadata = deepcopy(latest.metadata)
            if metadata is not None:
                merged_metadata.update(metadata)

            new_version_number = latest.version + 1
            new_version = PromptVersion(
                version=new_version_number,
                system_prompt=system_prompt if system_prompt is not None else latest.system_prompt,
                user_prompt_template=(
                    user_prompt_template
                    if user_prompt_template is not None
                    else latest.user_prompt_template
                ),
                variables=(
                    _normalize_variables(variables) if variables is not None else latest.variables
                ),
                created_at=_utc_now(),
                change_note=change_note,
                metadata=merged_metadata,
                source_preset=latest.source_preset,
            )

            record.versions[new_version_number] = new_version
            if description is not None:
                record.description = description
            record.updated_at = new_version.created_at
            return new_version

    def list_templates(self, include_versions: bool = False) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                self._templates[name].to_dict(include_versions=include_versions)
                for name in sorted(self._templates)
            ]

    def get_template(
        self,
        name: str,
        *,
        version: Optional[int] = None,
        deployment_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            record = self._require_template(name)
            resolved = self._resolve_version(
                record, version=version, deployment_label=deployment_label
            )
            return {
                "template": record.to_dict(include_versions=False),
                "resolved_version": resolved.to_dict(),
                "resolved_labels": record.labels_for_version(resolved.version),
            }

    def deploy_version(
        self,
        name: str,
        version: int,
        label: str,
        *,
        allow_stable_prefix_change: bool = False,
    ) -> Dict[str, Any]:
        with self._lock:
            record = self._require_template(name)
            if version not in record.versions:
                raise ValueError(f"Version {version} does not exist for prompt template '{name}'")
            previous_version = record.deployment_labels.get(label)
            stable_prefix_analysis = (
                self._build_stable_prefix_analysis(
                    record.versions[previous_version],
                    record.versions[version],
                )
                if previous_version is not None
                else None
            )
            if (
                previous_version is not None
                and stable_prefix_analysis is not None
                and stable_prefix_analysis["stable_prefix_changed"]
                and not allow_stable_prefix_change
            ):
                raise ValueError(
                    "Deploying this version would change the stable prompt prefix for the existing "
                    f"'{label}' label. Re-run with allow_stable_prefix_change=True if this cache "
                    "invalidation is intentional."
                )
            record.deployment_labels[label] = version
            record.updated_at = _utc_now()
            return {
                "name": name,
                "label": label,
                "version": version,
                "previous_version": previous_version,
                "stable_prefix_analysis": stable_prefix_analysis,
                "allow_stable_prefix_change": allow_stable_prefix_change,
                "deployment_labels": dict(sorted(record.deployment_labels.items())),
            }

    def compare_versions(self, name: str, version_a: int, version_b: int) -> Dict[str, Any]:
        with self._lock:
            record = self._require_template(name)
            if version_a not in record.versions or version_b not in record.versions:
                raise ValueError(f"Cannot compare missing versions for prompt template '{name}'")

            left = record.versions[version_a]
            right = record.versions[version_b]
            changed_fields: List[str] = []
            if left.system_prompt != right.system_prompt:
                changed_fields.append("system_prompt")
            if left.user_prompt_template != right.user_prompt_template:
                changed_fields.append("user_prompt_template")
            if list(left.variables) != list(right.variables):
                changed_fields.append("variables")
            if left.metadata != right.metadata:
                changed_fields.append("metadata")

            left_render = self._render_for_diff(left)
            right_render = self._render_for_diff(right)
            diff = "\n".join(
                difflib.unified_diff(
                    left_render.splitlines(),
                    right_render.splitlines(),
                    fromfile=f"{name}@v{version_a}",
                    tofile=f"{name}@v{version_b}",
                    lineterm="",
                )
            )
            return {
                "name": name,
                "version_a": version_a,
                "version_b": version_b,
                "changed_fields": changed_fields,
                "stable_prefix_analysis": self._build_stable_prefix_analysis(left, right),
                "diff": diff,
                "labels_a": record.labels_for_version(version_a),
                "labels_b": record.labels_for_version(version_b),
            }

    def get_create_stable_prefix_analysis(self, name: str) -> Dict[str, Any]:
        with self._lock:
            record = self._require_template(name)
            return _initial_stable_prefix_analysis(record.latest_version())

    def get_update_stable_prefix_analysis(self, name: str, version: int) -> Dict[str, Any]:
        with self._lock:
            record = self._require_template(name)
            if version <= 1 or version not in record.versions:
                raise ValueError(
                    f"Version {version} does not exist or has no prior version for prompt template '{name}'"
                )
            return self._build_stable_prefix_analysis(
                record.versions[version - 1],
                record.versions[version],
            )

    def render_prompt(
        self,
        name: str,
        *,
        variables: Optional[Dict[str, Any]] = None,
        version: Optional[int] = None,
        deployment_label: Optional[str] = None,
        tool_definitions: Optional[str] = None,
        rag_context: Optional[str] = None,
        few_shot_examples: Optional[List[str]] = None,
        chat_history: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_stable_prefix_chars: int = 5000,
    ) -> Dict[str, Any]:
        with self._lock:
            record = self._require_template(name)
            resolved = self._resolve_version(
                record, version=version, deployment_label=deployment_label
            )

            provided_variables = dict(variables or {})
            missing_variables = [
                variable for variable in resolved.variables if variable not in provided_variables
            ]
            if missing_variables:
                raise ValueError(
                    "Missing required prompt variables: " + ", ".join(sorted(missing_variables))
                )

            rendered_user_prompt = resolved.user_prompt_template.format(**provided_variables)
            sections: List[Dict[str, str]] = []

            if tool_definitions:
                sections.append({"name": "tool_definitions", "content": str(tool_definitions)})
            sections.append({"name": "system_instructions", "content": resolved.system_prompt})
            if rag_context:
                sections.append({"name": "rag_context", "content": str(rag_context)})
            if few_shot_examples:
                sections.append(
                    {
                        "name": "few_shot_examples",
                        "content": "\n\n".join(str(example) for example in few_shot_examples),
                    }
                )
            if chat_history:
                sections.append(
                    {
                        "name": "chat_history",
                        "content": "\n".join(str(entry) for entry in chat_history),
                    }
                )
            if metadata:
                sections.append(
                    {
                        "name": "metadata",
                        "content": json.dumps(metadata, sort_keys=True, separators=(",", ":")),
                    }
                )
            sections.append({"name": "user_query", "content": rendered_user_prompt})

            audit = audit_prompt_cacheability(sections)
            stability_guard = evaluate_prompt_stability(
                sections,
                max_stable_prefix_chars=max_stable_prefix_chars,
            )
            rendered_prompt = "\n\n".join(
                f"[{section['name']}]\n{section['content']}" for section in sections
            )
            return {
                "template": record.to_dict(include_versions=False),
                "resolved_version": resolved.to_dict(),
                "resolved_labels": record.labels_for_version(resolved.version),
                "sections": sections,
                "rendered_variables": provided_variables,
                "audit": audit,
                "stability_guard": stability_guard,
                "cacheable_prefix": audit["cacheable_prefix"],
                "volatile_suffix": audit["volatile_suffix"],
                "rendered_prompt": rendered_prompt,
            }

    def _require_template(self, name: str) -> PromptTemplateRecord:
        if name not in self._templates:
            raise ValueError(f"Unknown prompt template '{name}'")
        return self._templates[name]

    def _resolve_version(
        self,
        record: PromptTemplateRecord,
        *,
        version: Optional[int] = None,
        deployment_label: Optional[str] = None,
    ) -> PromptVersion:
        if version is not None and deployment_label is not None:
            raise ValueError("Specify either 'version' or 'deployment_label', not both")
        if deployment_label is not None:
            if deployment_label not in record.deployment_labels:
                raise ValueError(
                    f"Deployment label '{deployment_label}' not found for prompt template '{record.name}'"
                )
            version = record.deployment_labels[deployment_label]
        if version is None:
            version = record.latest_version().version
        if version not in record.versions:
            raise ValueError(
                f"Version {version} does not exist for prompt template '{record.name}'"
            )
        return record.versions[version]

    @staticmethod
    def _render_for_diff(version: PromptVersion) -> str:
        return (
            "[system_prompt]\n"
            f"{version.system_prompt}\n\n"
            "[user_prompt_template]\n"
            f"{version.user_prompt_template}\n\n"
            "[variables]\n"
            f"{', '.join(version.variables)}\n\n"
            "[metadata]\n"
            f"{version.metadata}"
        )

    @staticmethod
    def _build_stable_prefix_analysis(left: PromptVersion, right: PromptVersion) -> Dict[str, Any]:
        stable_fields_changed: List[str] = []
        volatile_fields_changed: List[str] = []

        if left.system_prompt != right.system_prompt:
            stable_fields_changed.append("system_prompt")
        if left.user_prompt_template != right.user_prompt_template:
            volatile_fields_changed.append("user_prompt_template")
        if list(left.variables) != list(right.variables):
            volatile_fields_changed.append("variables")
        if left.metadata != right.metadata:
            volatile_fields_changed.append("metadata")

        stable_prefix_changed = bool(stable_fields_changed)
        return {
            "stable_sections": ["system_instructions"],
            "version_a_hash": _stable_prefix_hash(left),
            "version_b_hash": _stable_prefix_hash(right),
            "stable_prefix_changed": stable_prefix_changed,
            "stable_fields_changed": stable_fields_changed,
            "volatile_fields_changed": volatile_fields_changed,
            "impact": (
                "cache_prefix_changed" if stable_prefix_changed else "stable_prefix_unchanged"
            ),
        }
