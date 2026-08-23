"""Regression test: PromptRegistry._templates is bounded via BoundedDict.

Per docs/plans/2026-08-24-a1-bounded-registries.md -- constructs a fresh
PromptRegistry directly (never via get_registry()) with a small max_templates,
inserts one more than the cap, and asserts the oldest-inserted record is
evicted while the rest survive.
"""

from __future__ import annotations

import pytest

from src.prompt_registry import PromptRegistry


def _make_registry(max_templates: int) -> PromptRegistry:
    # seed_defaults=False so the cap isn't immediately consumed by presets.
    return PromptRegistry(seed_defaults=False, max_templates=max_templates)


def test_prompt_registry_evicts_oldest_template_past_cap():
    registry = _make_registry(max_templates=2)

    for i in range(3):  # max_items + 1
        registry.create_template(
            name=f"template-{i}",
            description=f"desc-{i}",
            system_prompt="system",
            user_prompt_template="user {var}",
            variables=["var"],
        )

    with pytest.raises(ValueError, match="Unknown prompt template 'template-0'"):
        registry.get_template("template-0")

    remaining = {t["name"] for t in registry.list_templates()}
    assert remaining == {"template-1", "template-2"}
