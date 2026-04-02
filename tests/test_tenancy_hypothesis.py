"""Property-based tests for scope identity normalization and isolation."""

from hypothesis import given, strategies as st

from src.identity_scope import (
    compose_scoped_file_id,
    display_file_id,
    parse_scoped_file_id,
    scope_matches,
)

safe_text = st.text(
    alphabet=st.characters(
        blacklist_categories=("Cs",),
        blacklist_characters=["\x00", "\n", "\r"],
    ),
    min_size=1,
    max_size=24,
)


@given(file_id=safe_text)
def test_unscoped_round_trip_preserves_raw_file_id(file_id):
    assert compose_scoped_file_id(file_id) == file_id
    assert display_file_id(file_id) == file_id
    assert parse_scoped_file_id(file_id)["file_id"] == file_id
    assert scope_matches(file_id)


@given(
    file_id=safe_text,
    workspace_id=safe_text,
    user_id=safe_text,
    agent_id=safe_text,
    session_id=safe_text,
)
def test_scoped_round_trip_preserves_visible_identity(
    file_id, workspace_id, user_id, agent_id, session_id
):
    scoped_file_id = compose_scoped_file_id(
        file_id,
        workspace_id=workspace_id,
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
    )

    parsed = parse_scoped_file_id(scoped_file_id)

    assert display_file_id(scoped_file_id) == file_id
    assert parsed["file_id"] == file_id
    assert parsed["workspace_id"] == workspace_id
    assert parsed["user_id"] == user_id
    assert parsed["agent_id"] == agent_id
    assert parsed["session_id"] == session_id


@given(file_id=safe_text, left_workspace=safe_text, right_workspace=safe_text)
def test_scope_matching_rejects_other_workspaces(file_id, left_workspace, right_workspace):
    if left_workspace == right_workspace:
        right_workspace = f"{right_workspace}-other"

    scoped_file_id = compose_scoped_file_id(file_id, workspace_id=left_workspace)

    assert scope_matches(scoped_file_id, workspace_id=left_workspace)
    assert not scope_matches(scoped_file_id, workspace_id=right_workspace)
    assert not scope_matches(scoped_file_id)


@given(file_id=safe_text, workspace_id=safe_text, user_id=safe_text)
def test_partial_scope_filters_only_requested_dimensions(file_id, workspace_id, user_id):
    scoped_file_id = compose_scoped_file_id(file_id, workspace_id=workspace_id, user_id=user_id)

    assert scope_matches(scoped_file_id, workspace_id=workspace_id)
    assert scope_matches(scoped_file_id, workspace_id=workspace_id, user_id=user_id)
    assert not scope_matches(scoped_file_id, workspace_id=workspace_id, user_id=f"{user_id}-x")
