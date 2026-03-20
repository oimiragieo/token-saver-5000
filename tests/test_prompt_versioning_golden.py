"""Golden-ish regression tests for prompt version comparison output."""

from src.prompt_registry import PromptRegistry


def setup_function():
    PromptRegistry.reset_singleton()


def test_prompt_version_comparison_has_stable_shape():
    registry = PromptRegistry(seed_defaults=False)
    registry.create_template(
        name="rewrite-default",
        description="Rewrite prompt",
        system_prompt="Rewrite faithfully.",
        user_prompt_template="Rewrite {content}",
        variables=["content"],
    )
    registry.update_template(
        "rewrite-default",
        system_prompt="Rewrite faithfully and preserve APIs.",
        user_prompt_template="Rewrite {content} for {audience}",
        variables=["content", "audience"],
        change_note="audience-aware rewrite",
    )

    comparison = registry.compare_versions("rewrite-default", 1, 2)

    assert comparison == {
        "name": "rewrite-default",
        "version_a": 1,
        "version_b": 2,
        "changed_fields": ["system_prompt", "user_prompt_template", "variables"],
        "stable_prefix_analysis": {
            "stable_sections": ["system_instructions"],
            "version_a_hash": comparison["stable_prefix_analysis"]["version_a_hash"],
            "version_b_hash": comparison["stable_prefix_analysis"]["version_b_hash"],
            "stable_prefix_changed": True,
            "stable_fields_changed": ["system_prompt"],
            "volatile_fields_changed": ["user_prompt_template", "variables"],
            "impact": "cache_prefix_changed",
        },
        "diff": comparison["diff"],
        "labels_a": [],
        "labels_b": [],
    }
    assert "[system_prompt]" in comparison["diff"]
    assert "[user_prompt_template]" in comparison["diff"]
