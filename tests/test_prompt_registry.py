"""Tests for prompt registry service behavior."""

from src.prompt_registry import PromptRegistry


def setup_function():
    PromptRegistry.reset_singleton()


def test_seeded_templates_exist_from_compression_presets():
    registry = PromptRegistry.get_registry()

    templates = registry.list_templates()
    names = {template["name"] for template in templates}

    assert "compression-chat" in names
    assert "compression-code-review" in names


def test_create_update_get_deploy_prompt_template():
    registry = PromptRegistry(seed_defaults=False)
    registry.create_template(
        name="review-default",
        description="Review prompt",
        system_prompt="You are a reviewer.",
        user_prompt_template="Review {diff}",
        variables=["diff"],
    )

    updated = registry.update_template(
        "review-default",
        system_prompt="You are a reviewer. Prioritize bugs.",
        change_note="bug focus",
    )
    deployment = registry.deploy_version(
        "review-default",
        updated.version,
        "production",
        allow_stable_prefix_change=True,
    )
    resolved = registry.get_template("review-default", deployment_label="production")

    assert updated.version == 2
    assert deployment["deployment_labels"]["production"] == 2
    assert resolved["resolved_version"]["version"] == 2
    assert "bug focus" == resolved["resolved_version"]["change_note"]


def test_update_template_reports_authoring_stable_prefix_analysis():
    registry = PromptRegistry(seed_defaults=False)
    registry.create_template(
        name="review-default",
        description="Review prompt",
        system_prompt="You are a reviewer.",
        user_prompt_template="Review {diff}",
        variables=["diff"],
    )

    updated = registry.update_template(
        "review-default",
        system_prompt="You are a reviewer. Prioritize bugs.",
        change_note="bug focus",
    )
    stable_prefix_analysis = registry.get_update_stable_prefix_analysis(
        "review-default", updated.version
    )

    assert updated.version == 2
    assert stable_prefix_analysis["stable_prefix_changed"] is True
    assert stable_prefix_analysis["stable_fields_changed"] == ["system_prompt"]


def test_create_template_reports_initial_stable_prefix_analysis():
    registry = PromptRegistry(seed_defaults=False)

    record = registry.create_template(
        name="review-default",
        description="Review prompt",
        system_prompt="You are a reviewer.",
        user_prompt_template="Review {diff}",
        variables=["diff"],
    )
    stable_prefix_analysis = registry.get_create_stable_prefix_analysis("review-default")

    assert record.latest_version().version == 1
    assert stable_prefix_analysis["stable_prefix_changed"] is True
    assert stable_prefix_analysis["impact"] == "initial_prefix_created"


def test_compare_versions_returns_changed_fields_and_diff():
    registry = PromptRegistry(seed_defaults=False)
    registry.create_template(
        name="chat-default",
        description="Chat prompt",
        system_prompt="Be concise.",
        user_prompt_template="Answer {question}",
        variables=["question"],
    )
    registry.update_template(
        "chat-default",
        system_prompt="Be concise and cite evidence.",
        user_prompt_template="Answer {question} using {context}",
        variables=["question", "context"],
        metadata={"owner": "product"},
    )

    comparison = registry.compare_versions("chat-default", 1, 2)

    assert comparison["changed_fields"] == [
        "system_prompt",
        "user_prompt_template",
        "variables",
        "metadata",
    ]
    assert comparison["stable_prefix_analysis"]["stable_prefix_changed"] is True
    assert comparison["stable_prefix_analysis"]["stable_fields_changed"] == ["system_prompt"]
    assert comparison["stable_prefix_analysis"]["impact"] == "cache_prefix_changed"
    assert "--- chat-default@v1" in comparison["diff"]
    assert "+++ chat-default@v2" in comparison["diff"]


def test_compare_versions_identifies_volatile_only_changes():
    registry = PromptRegistry(seed_defaults=False)
    registry.create_template(
        name="chat-default",
        description="Chat prompt",
        system_prompt="Be concise.",
        user_prompt_template="Answer {question}",
        variables=["question"],
    )
    registry.update_template(
        "chat-default",
        user_prompt_template="Answer {question} with examples",
        metadata={"owner": "product"},
    )

    comparison = registry.compare_versions("chat-default", 1, 2)

    assert comparison["stable_prefix_analysis"]["stable_prefix_changed"] is False
    assert comparison["stable_prefix_analysis"]["stable_fields_changed"] == []
    assert comparison["stable_prefix_analysis"]["volatile_fields_changed"] == [
        "user_prompt_template",
        "metadata",
    ]
    assert comparison["stable_prefix_analysis"]["impact"] == "stable_prefix_unchanged"


def test_render_prompt_template_builds_cache_friendly_sections():
    registry = PromptRegistry(seed_defaults=False)
    registry.create_template(
        name="chat-default",
        description="Chat prompt",
        system_prompt="Be concise and accurate.",
        user_prompt_template="Answer {question} using {context}",
        variables=["question", "context"],
    )

    rendered = registry.render_prompt(
        "chat-default",
        variables={
            "question": "How do retries work?",
            "context": "Retries use exponential backoff.",
        },
        tool_definitions="tool_a(args)\ntool_b(args)",
        rag_context="Static architecture notes.",
        few_shot_examples=["Q: What is auth?\nA: Identity and access control."],
        chat_history=["User previously asked about billing."],
        metadata={"workspace_id": "acme", "request_id": "req-123"},
    )

    assert rendered["template"]["name"] == "chat-default"
    assert rendered["resolved_version"]["version"] == 1
    assert rendered["audit"]["is_cache_friendly"] is True
    assert rendered["sections"][0]["name"] == "tool_definitions"
    assert rendered["sections"][-1]["name"] == "user_query"
    assert rendered["cacheable_prefix"].startswith("[tool_definitions]")
    assert rendered["volatile_suffix"].startswith("[metadata]")
    assert (
        "Answer How do retries work? using Retries use exponential backoff."
        in rendered["rendered_prompt"]
    )


def test_render_prompt_template_rejects_missing_variables():
    registry = PromptRegistry(seed_defaults=False)
    registry.create_template(
        name="chat-default",
        description="Chat prompt",
        system_prompt="Be concise.",
        user_prompt_template="Answer {question} using {context}",
        variables=["question", "context"],
    )

    try:
        registry.render_prompt("chat-default", variables={"question": "hi"})
        raise AssertionError("Expected ValueError for missing variable")
    except ValueError as exc:
        assert "context" in str(exc)


def test_deploy_version_reports_prefix_change_when_repointing_label():
    registry = PromptRegistry(seed_defaults=False)
    registry.create_template(
        name="review-default",
        description="Review prompt",
        system_prompt="You are a reviewer.",
        user_prompt_template="Review {diff}",
        variables=["diff"],
        deployment_label="production",
    )
    updated = registry.update_template(
        "review-default",
        system_prompt="You are a reviewer. Prioritize correctness.",
    )

    deployment = registry.deploy_version(
        "review-default",
        updated.version,
        "production",
        allow_stable_prefix_change=True,
    )

    assert deployment["previous_version"] == 1
    assert deployment["stable_prefix_analysis"]["stable_prefix_changed"] is True
    assert deployment["stable_prefix_analysis"]["impact"] == "cache_prefix_changed"


def test_deploy_version_rejects_prefix_change_without_explicit_acknowledgement():
    registry = PromptRegistry(seed_defaults=False)
    registry.create_template(
        name="review-default",
        description="Review prompt",
        system_prompt="You are a reviewer.",
        user_prompt_template="Review {diff}",
        variables=["diff"],
        deployment_label="production",
    )
    updated = registry.update_template(
        "review-default",
        system_prompt="You are a reviewer. Prioritize correctness.",
    )

    try:
        registry.deploy_version("review-default", updated.version, "production")
        raise AssertionError("Expected ValueError for unacknowledged stable prefix change")
    except ValueError as exc:
        assert "allow_stable_prefix_change=True" in str(exc)
