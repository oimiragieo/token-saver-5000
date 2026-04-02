"""
Comprehensive tests for afm_handlers.py

Coverage target: 80%+ (currently 40%)
Tests all 6 AFM dialogue memory handlers with success cases, edge cases, and error handling.
"""

import pytest
from unittest.mock import Mock
from dataclasses import dataclass
from src.afm import Message, ImportanceLevel
from src.handlers.afm_handlers import (
    handle_afm_add_message,
    handle_afm_build_context,
    handle_afm_get_stats,
    handle_afm_clear_history,
    handle_afm_export_history,
    handle_afm_import_history,
)

# ============================================================================
# Mock Data Classes
# ============================================================================


@dataclass
class MockPackingStats:
    """Mock packing statistics from AFM"""

    total_messages: int
    full_count: int
    compressed_count: int
    placeholder_count: int
    dropped_count: int
    total_tokens: int
    budget_tokens: int
    compression_ratio: float


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_context():
    """Create mock HandlerContext for AFM handlers"""
    context = {}

    # Mock focus manager
    mock_focus_manager = Mock()
    mock_focus_manager.turn_counter = 5
    mock_focus_manager.messages = []
    context["focus_manager"] = mock_focus_manager

    # Mock persistence
    mock_persistence = Mock()
    context["persistence"] = mock_persistence

    return context


@pytest.fixture
def mock_message():
    """Create a mock Message object"""
    msg = Mock(spec=Message)
    msg.turn_index = 5
    msg.role = "user"
    msg.importance = ImportanceLevel.RELEVANT
    msg.content = "Test message content"
    return msg


# ============================================================================
# Test handle_afm_add_message Handler
# ============================================================================


class TestHandleAfmAddMessage:
    """Test AFM add message handler"""

    @pytest.mark.asyncio
    async def test_add_message_user_role(self, mock_context, mock_message):
        """Test adding a user message"""
        # Setup
        mock_context["focus_manager"].add_message.return_value = mock_message
        mock_context["focus_manager"].messages = [mock_message]

        args = {"role": "user", "content": "Hello, I need help with something"}

        # Execute
        result = await handle_afm_add_message(mock_context, args)

        # Verify
        assert "Message Added to Dialogue History" in result
        assert "Turn: 5" in result
        assert "Role: user" in result
        assert "Importance: RELEVANT" in result
        assert "Hello, I need help" in result
        assert "Total messages: 1" in result
        assert "Current turn: 5" in result
        mock_context["focus_manager"].add_message.assert_called_once_with(
            "user", "Hello, I need help with something"
        )

    @pytest.mark.asyncio
    async def test_add_message_assistant_role(self, mock_context, mock_message):
        """Test adding an assistant message"""
        # Setup
        mock_message.role = "assistant"
        mock_message.importance = ImportanceLevel.RELEVANT
        mock_context["focus_manager"].add_message.return_value = mock_message
        mock_context["focus_manager"].messages = [mock_message]

        args = {"role": "assistant", "content": "I can help you with that!"}

        # Execute
        result = await handle_afm_add_message(mock_context, args)

        # Verify
        assert "Message Added" in result
        assert "Role: assistant" in result
        mock_context["focus_manager"].add_message.assert_called_once_with(
            "assistant", "I can help you with that!"
        )

    @pytest.mark.asyncio
    async def test_add_message_system_role(self, mock_context, mock_message):
        """Test adding a system message"""
        # Setup
        mock_message.role = "system"
        mock_message.importance = ImportanceLevel.CRITICAL
        mock_context["focus_manager"].add_message.return_value = mock_message
        mock_context["focus_manager"].messages = [mock_message]

        args = {"role": "system", "content": "You are a helpful assistant"}

        # Execute
        result = await handle_afm_add_message(mock_context, args)

        # Verify
        assert "Message Added" in result
        assert "Role: system" in result
        assert "Importance: CRITICAL" in result

    @pytest.mark.asyncio
    async def test_add_message_critical_importance(self, mock_context, mock_message):
        """Test adding a message with critical importance (e.g., allergy)"""
        # Setup
        mock_message.importance = ImportanceLevel.CRITICAL
        mock_context["focus_manager"].add_message.return_value = mock_message
        mock_context["focus_manager"].messages = [mock_message]

        args = {"role": "user", "content": "I have a severe peanut allergy"}

        # Execute
        result = await handle_afm_add_message(mock_context, args)

        # Verify
        assert "Importance: CRITICAL" in result

    @pytest.mark.asyncio
    async def test_add_message_truncates_long_content(self, mock_context, mock_message):
        """Test that very long content is truncated in output"""
        # Setup
        long_content = "A" * 200
        mock_context["focus_manager"].add_message.return_value = mock_message
        mock_context["focus_manager"].messages = [mock_message]

        args = {"role": "user", "content": long_content}

        # Execute
        result = await handle_afm_add_message(mock_context, args)

        # Verify - content should be truncated to 100 chars + "..."
        assert "..." in result
        assert "A" * 100 in result
        assert "A" * 200 not in result

    @pytest.mark.asyncio
    async def test_add_message_invalid_role(self, mock_context):
        """Test that invalid role raises ValueError"""
        args = {"role": "invalid_role", "content": "Test content"}

        # Execute & Verify
        with pytest.raises(ValueError) as exc_info:
            await handle_afm_add_message(mock_context, args)

        assert "Invalid role" in str(exc_info.value)
        assert "invalid_role" in str(exc_info.value)
        assert "user" in str(exc_info.value)
        assert "assistant" in str(exc_info.value)
        assert "system" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_add_message_shows_stats(self, mock_context, mock_message):
        """Test that dialogue stats are shown"""
        # Setup
        mock_context["focus_manager"].add_message.return_value = mock_message
        mock_context["focus_manager"].messages = [mock_message] * 10
        mock_context["focus_manager"].turn_counter = 15

        args = {"role": "user", "content": "Test"}

        # Execute
        result = await handle_afm_add_message(mock_context, args)

        # Verify
        assert "Total messages: 10" in result
        assert "Current turn: 15" in result
        assert "afm_build_context" in result  # Shows next step


# ============================================================================
# Test handle_afm_build_context Handler
# ============================================================================


class TestHandleAfmBuildContext:
    """Test AFM build context handler"""

    @pytest.mark.asyncio
    async def test_build_context_success(self, mock_context):
        """Test successful context building"""
        # Setup
        ctx = [
            ("system", "You are a helpful assistant"),
            ("user", "Hello"),
            ("assistant", "Hi there!"),
        ]
        stats = MockPackingStats(
            total_messages=10,
            full_count=3,
            compressed_count=5,
            placeholder_count=2,
            dropped_count=0,
            total_tokens=450,
            budget_tokens=500,
            compression_ratio=0.45,
        )
        mock_context["focus_manager"].build_context.return_value = (ctx, stats)

        args = {
            "current_query": "What is the weather today?",
            "budget_tokens": 500,
        }

        # Execute
        result = await handle_afm_build_context(mock_context, args)

        # Verify
        assert "AFM Context Built Successfully" in result
        assert "Query: What is the weather today?" in result
        assert "Total messages processed: 10" in result
        assert "FULL fidelity:         3" in result
        assert "COMPRESSED:            5" in result
        assert "PLACEHOLDER:           2" in result
        assert "DROPPED:               0" in result
        assert "Tokens used:              450 / 500" in result
        assert "Budget utilization:       45" in result
        assert "Context Messages (3 total)" in result
        assert "[1] system:" in result
        assert "[2] user:" in result
        assert "[3] assistant:" in result
        mock_context["focus_manager"].build_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_build_context_with_system_preamble(self, mock_context):
        """Test building context with custom system preamble"""
        # Setup
        ctx = [("system", "Custom preamble"), ("user", "Hello")]
        stats = MockPackingStats(
            total_messages=5,
            full_count=2,
            compressed_count=3,
            placeholder_count=0,
            dropped_count=0,
            total_tokens=100,
            budget_tokens=200,
            compression_ratio=0.5,
        )
        mock_context["focus_manager"].build_context.return_value = (ctx, stats)

        args = {
            "current_query": "Test query",
            "budget_tokens": 200,
            "system_preamble": "Custom preamble for this task",
        }

        # Execute
        result = await handle_afm_build_context(mock_context, args)

        # Verify
        assert "AFM Context Built Successfully" in result
        mock_context["focus_manager"].build_context.assert_called_once_with(
            current_query="Test query",
            budget_tokens=200,
            system_preamble="Custom preamble for this task",
        )

    @pytest.mark.asyncio
    async def test_build_context_truncates_long_messages(self, mock_context):
        """Test that long messages are truncated in display"""
        # Setup
        long_message = "A" * 200
        ctx = [("user", long_message)]
        stats = MockPackingStats(
            total_messages=1,
            full_count=1,
            compressed_count=0,
            placeholder_count=0,
            dropped_count=0,
            total_tokens=50,
            budget_tokens=100,
            compression_ratio=0.5,
        )
        mock_context["focus_manager"].build_context.return_value = (ctx, stats)

        args = {"current_query": "Test", "budget_tokens": 100}

        # Execute
        result = await handle_afm_build_context(mock_context, args)

        # Verify - content should be truncated to 150 chars + "..."
        assert "..." in result
        assert "A" * 150 in result
        assert "A" * 200 not in result

    @pytest.mark.asyncio
    async def test_build_context_truncates_long_query(self, mock_context):
        """Test that long query is truncated in output"""
        # Setup
        long_query = "Q" * 200
        ctx = []
        stats = MockPackingStats(
            total_messages=0,
            full_count=0,
            compressed_count=0,
            placeholder_count=0,
            dropped_count=0,
            total_tokens=0,
            budget_tokens=100,
            compression_ratio=0.0,
        )
        mock_context["focus_manager"].build_context.return_value = (ctx, stats)

        args = {"current_query": long_query, "budget_tokens": 100}

        # Execute
        result = await handle_afm_build_context(mock_context, args)

        # Verify - query should be truncated to 100 chars + "..."
        assert "..." in result
        assert "Q" * 100 in result
        assert "Q" * 200 not in result

    @pytest.mark.asyncio
    async def test_build_context_negative_budget_raises_error(self, mock_context):
        """Test that negative budget_tokens raises ValueError"""
        args = {"current_query": "Test", "budget_tokens": -100}

        # Execute & Verify
        with pytest.raises(ValueError) as exc_info:
            await handle_afm_build_context(mock_context, args)

        assert "budget_tokens must be positive" in str(exc_info.value)
        assert "-100" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_build_context_zero_budget_raises_error(self, mock_context):
        """Test that zero budget_tokens raises ValueError"""
        args = {"current_query": "Test", "budget_tokens": 0}

        # Execute & Verify
        with pytest.raises(ValueError) as exc_info:
            await handle_afm_build_context(mock_context, args)

        assert "budget_tokens must be positive" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_build_context_shows_compression_stats(self, mock_context):
        """Test that compression ratio and savings are shown"""
        # Setup
        ctx = [("user", "Test")]
        stats = MockPackingStats(
            total_messages=10,
            full_count=3,
            compressed_count=7,
            placeholder_count=0,
            dropped_count=0,
            total_tokens=200,
            budget_tokens=500,
            compression_ratio=0.4,  # 40% of budget used = 60% savings
        )
        mock_context["focus_manager"].build_context.return_value = (ctx, stats)

        args = {"current_query": "Test", "budget_tokens": 500}

        # Execute
        result = await handle_afm_build_context(mock_context, args)

        # Verify
        assert "~60% token savings" in result  # 100 * (1 - 0.4) = 60%
        assert "Ready to send to LLM" in result
        assert "arXiv:2511.12712v1" in result  # Paper reference


# ============================================================================
# Test handle_afm_get_stats Handler
# ============================================================================


class TestHandleAfmGetStats:
    """Test AFM get stats handler"""

    @pytest.mark.asyncio
    async def test_get_stats_basic(self, mock_context):
        """Test getting basic dialogue statistics"""
        # Setup
        mock_context["focus_manager"].get_stats.return_value = {
            "total_messages": 15,
            "current_turn": 20,
            "importance_breakdown": {"critical": 2, "relevant": 10, "trivial": 3},
        }

        args = {}

        # Execute
        result = await handle_afm_get_stats(mock_context, args)

        # Verify
        assert "AFM Dialogue Statistics" in result
        assert "Total messages: 15" in result
        assert "Current turn:   20" in result
        assert "CRITICAL:  2 messages" in result
        assert "RELEVANT:  10 messages" in result
        assert "TRIVIAL:   3 messages" in result
        assert "CRITICAL messages" in result
        assert "always preserved" in result

    @pytest.mark.asyncio
    async def test_get_stats_empty_dialogue(self, mock_context):
        """Test getting stats for empty dialogue"""
        # Setup
        mock_context["focus_manager"].get_stats.return_value = {
            "total_messages": 0,
            "current_turn": 0,
            "importance_breakdown": {"critical": 0, "relevant": 0, "trivial": 0},
        }

        args = {}

        # Execute
        result = await handle_afm_get_stats(mock_context, args)

        # Verify
        assert "Total messages: 0" in result
        assert "Current turn:   0" in result
        assert "CRITICAL:  0 messages" in result

    @pytest.mark.asyncio
    async def test_get_stats_all_critical(self, mock_context):
        """Test stats with all critical messages"""
        # Setup
        mock_context["focus_manager"].get_stats.return_value = {
            "total_messages": 5,
            "current_turn": 5,
            "importance_breakdown": {"critical": 5, "relevant": 0, "trivial": 0},
        }

        args = {}

        # Execute
        result = await handle_afm_get_stats(mock_context, args)

        # Verify
        assert "CRITICAL:  5 messages" in result
        assert "RELEVANT:  0 messages" in result
        assert "TRIVIAL:   0 messages" in result


# ============================================================================
# Test handle_afm_clear_history Handler
# ============================================================================


class TestHandleAfmClearHistory:
    """Test AFM clear history handler"""

    @pytest.mark.asyncio
    async def test_clear_history_with_messages(self, mock_context):
        """Test clearing non-empty dialogue history"""
        # Setup
        mock_context["focus_manager"].messages = [Mock()] * 10
        mock_context["focus_manager"].turn_counter = 15

        args = {}

        # Execute
        result = await handle_afm_clear_history(mock_context, args)

        # Verify
        assert "AFM Dialogue History Cleared" in result
        assert "Previous state:" in result
        assert "Messages: 10" in result
        assert "Turns:    15" in result
        assert "Current state:" in result
        assert "Messages: 0" in result
        assert "Turns:    0" in result
        assert "Ready for new conversation" in result
        mock_context["focus_manager"].clear_history.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_history_already_empty(self, mock_context):
        """Test clearing already empty dialogue"""
        # Setup
        mock_context["focus_manager"].messages = []
        mock_context["focus_manager"].turn_counter = 0

        args = {}

        # Execute
        result = await handle_afm_clear_history(mock_context, args)

        # Verify
        assert "Previous state:" in result
        assert "Messages: 0" in result
        assert "Turns:    0" in result
        mock_context["focus_manager"].clear_history.assert_called_once()


# ============================================================================
# Test handle_afm_export_history Handler
# ============================================================================


class TestHandleAfmExportHistory:
    """Test AFM export history handler"""

    @pytest.mark.asyncio
    async def test_export_history_success(self, mock_context, mock_message):
        """Test successful export of dialogue history"""
        # Setup
        mock_context["focus_manager"].messages = [mock_message] * 5
        mock_context["focus_manager"].turn_counter = 10
        mock_context["persistence"].save_afm_history.return_value = True
        mock_context["persistence"].list_afm_sessions.return_value = [
            "session1",
            "session2",
        ]

        args = {"session_id": "test_session"}

        # Execute
        result = await handle_afm_export_history(mock_context, args)

        # Verify
        assert "AFM Export Complete" in result
        assert "Session ID: test_session" in result
        assert "Messages exported: 5" in result
        assert "Current turn: 10" in result
        assert 'afm_import_history(session_id="test_session")' in result
        assert "Available sessions: session1, session2" in result
        mock_context["persistence"].save_afm_history.assert_called_once()

    @pytest.mark.asyncio
    async def test_export_history_default_session_id(self, mock_context, mock_message):
        """Test export with default session ID"""
        # Setup
        mock_context["focus_manager"].messages = [mock_message]
        mock_context["focus_manager"].turn_counter = 1
        mock_context["persistence"].save_afm_history.return_value = True
        mock_context["persistence"].list_afm_sessions.return_value = []

        args = {}  # No session_id provided

        # Execute
        result = await handle_afm_export_history(mock_context, args)

        # Verify
        assert "Session ID: default" in result
        # Verify that save_afm_history was called with "default"
        call_args = mock_context["persistence"].save_afm_history.call_args
        assert call_args[1]["session_id"] == "default"

    @pytest.mark.asyncio
    async def test_export_history_empty_dialogue(self, mock_context):
        """Test export with no dialogue history"""
        # Setup
        mock_context["focus_manager"].messages = []

        args = {"session_id": "test"}

        # Execute
        result = await handle_afm_export_history(mock_context, args)

        # Verify
        assert "No dialogue history to export" in result
        assert "Add messages with afm_add_message()" in result
        # save_afm_history should NOT be called
        mock_context["persistence"].save_afm_history.assert_not_called()

    @pytest.mark.asyncio
    async def test_export_history_persistence_failure(self, mock_context, mock_message):
        """Test export when persistence fails"""
        # Setup
        mock_context["focus_manager"].messages = [mock_message]
        mock_context["focus_manager"].turn_counter = 1
        mock_context["persistence"].save_afm_history.return_value = False

        args = {"session_id": "test"}

        # Execute
        result = await handle_afm_export_history(mock_context, args)

        # Verify
        assert "AFM Export Failed" in result
        assert "Could not save dialogue history" in result
        assert "Check logs" in result


# ============================================================================
# Test handle_afm_import_history Handler
# ============================================================================


class TestHandleAfmImportHistory:
    """Test AFM import history handler"""

    @pytest.mark.asyncio
    async def test_import_history_success(self, mock_context, mock_message):
        """Test successful import of dialogue history"""
        # Setup
        import_data = {
            "messages": [mock_message] * 3,
            "turn_counter": 5,
            "metadata": {"exported_at": "2024-01-15T10:00:00"},
        }
        mock_context["persistence"].load_afm_history.return_value = import_data

        args = {"session_id": "test_session"}

        # Execute
        result = await handle_afm_import_history(mock_context, args)

        # Verify
        assert "AFM Import Complete" in result
        assert "Session ID: test_session" in result
        assert "Messages restored: 3" in result
        assert "Turn counter restored: 5" in result
        assert "Exported at: 2024-01-15T10:00:00" in result
        assert "Conversation state has been restored" in result
        # Verify that focus manager was updated
        assert mock_context["focus_manager"].messages == [mock_message] * 3
        assert mock_context["focus_manager"].turn_counter == 5

    @pytest.mark.asyncio
    async def test_import_history_default_session_id(self, mock_context, mock_message):
        """Test import with default session ID"""
        # Setup
        import_data = {
            "messages": [mock_message],
            "turn_counter": 1,
            "metadata": {},
        }
        mock_context["persistence"].load_afm_history.return_value = import_data

        args = {}  # No session_id provided

        # Execute
        result = await handle_afm_import_history(mock_context, args)

        # Verify
        assert "Session ID: default" in result
        mock_context["persistence"].load_afm_history.assert_called_once_with("default")

    @pytest.mark.asyncio
    async def test_import_history_session_not_found(self, mock_context):
        """Test import when session doesn't exist"""
        # Setup
        mock_context["persistence"].load_afm_history.return_value = None
        mock_context["persistence"].list_afm_sessions.return_value = [
            "session1",
            "session2",
        ]

        args = {"session_id": "nonexistent"}

        # Execute
        result = await handle_afm_import_history(mock_context, args)

        # Verify
        assert "AFM Import Failed" in result
        assert "Session 'nonexistent' not found" in result
        assert "Available sessions: session1, session2" in result

    @pytest.mark.asyncio
    async def test_import_history_no_available_sessions(self, mock_context):
        """Test import when no sessions exist"""
        # Setup
        mock_context["persistence"].load_afm_history.return_value = None
        mock_context["persistence"].list_afm_sessions.return_value = []

        args = {"session_id": "test"}

        # Execute
        result = await handle_afm_import_history(mock_context, args)

        # Verify
        assert "AFM Import Failed" in result
        assert "Available sessions: (none)" in result

    @pytest.mark.asyncio
    async def test_import_history_missing_messages_field(self, mock_context):
        """Test import with invalid data - missing messages field"""
        # Setup
        import_data = {
            "turn_counter": 5,  # Missing "messages"
        }
        mock_context["persistence"].load_afm_history.return_value = import_data

        args = {"session_id": "test"}

        # Execute
        result = await handle_afm_import_history(mock_context, args)

        # Verify
        assert "AFM Import Validation Failed" in result
        assert "Missing 'messages' field" in result

    @pytest.mark.asyncio
    async def test_import_history_missing_turn_counter_field(self, mock_context):
        """Test import with invalid data - missing turn_counter field"""
        # Setup
        import_data = {
            "messages": [],  # Missing "turn_counter"
        }
        mock_context["persistence"].load_afm_history.return_value = import_data

        args = {"session_id": "test"}

        # Execute
        result = await handle_afm_import_history(mock_context, args)

        # Verify
        assert "AFM Import Validation Failed" in result
        assert "Missing 'turn_counter' field" in result

    @pytest.mark.asyncio
    async def test_import_history_messages_not_list(self, mock_context):
        """Test import with invalid data - messages is not a list"""
        # Setup
        import_data = {
            "messages": "not a list",
            "turn_counter": 5,
        }
        mock_context["persistence"].load_afm_history.return_value = import_data

        args = {"session_id": "test"}

        # Execute
        result = await handle_afm_import_history(mock_context, args)

        # Verify
        assert "AFM Import Validation Failed" in result
        assert "'messages' must be a list" in result

    @pytest.mark.asyncio
    async def test_import_history_message_missing_role(self, mock_context):
        """Test import with invalid message - missing role field"""
        # Setup
        invalid_msg = Mock()
        del invalid_msg.role  # Remove role attribute
        import_data = {
            "messages": [invalid_msg],
            "turn_counter": 1,
        }
        mock_context["persistence"].load_afm_history.return_value = import_data

        args = {"session_id": "test"}

        # Execute
        result = await handle_afm_import_history(mock_context, args)

        # Verify
        assert "AFM Import Validation Failed" in result
        assert "Message 0 missing 'role' field" in result

    @pytest.mark.asyncio
    async def test_import_history_message_missing_content(self, mock_context):
        """Test import with invalid message - missing content field"""
        # Setup
        invalid_msg = Mock()
        invalid_msg.role = "user"
        del invalid_msg.content  # Remove content attribute
        import_data = {
            "messages": [invalid_msg],
            "turn_counter": 1,
        }
        mock_context["persistence"].load_afm_history.return_value = import_data

        args = {"session_id": "test"}

        # Execute
        result = await handle_afm_import_history(mock_context, args)

        # Verify
        assert "AFM Import Validation Failed" in result
        assert "Message 0 missing 'content' field" in result

    @pytest.mark.asyncio
    async def test_import_history_message_invalid_role(self, mock_context):
        """Test import with invalid message - invalid role value"""
        # Setup
        invalid_msg = Mock()
        invalid_msg.role = "invalid_role"
        invalid_msg.content = "Test"
        import_data = {
            "messages": [invalid_msg],
            "turn_counter": 1,
        }
        mock_context["persistence"].load_afm_history.return_value = import_data

        args = {"session_id": "test"}

        # Execute
        result = await handle_afm_import_history(mock_context, args)

        # Verify
        assert "AFM Import Validation Failed" in result
        assert "Message 0 has invalid role: invalid_role" in result

    @pytest.mark.asyncio
    async def test_import_history_turn_counter_not_int(self, mock_context, mock_message):
        """Test import with invalid turn_counter - not an integer"""
        # Setup
        import_data = {
            "messages": [mock_message],
            "turn_counter": "not an int",
        }
        mock_context["persistence"].load_afm_history.return_value = import_data

        args = {"session_id": "test"}

        # Execute
        result = await handle_afm_import_history(mock_context, args)

        # Verify
        assert "AFM Import Validation Failed" in result
        assert "'turn_counter' must be an integer" in result

    @pytest.mark.asyncio
    async def test_import_history_metadata_default_unknown(self, mock_context, mock_message):
        """Test import with missing metadata shows 'unknown' exported time"""
        # Setup
        import_data = {
            "messages": [mock_message],
            "turn_counter": 1,
            # No metadata field
        }
        mock_context["persistence"].load_afm_history.return_value = import_data

        args = {"session_id": "test"}

        # Execute
        result = await handle_afm_import_history(mock_context, args)

        # Verify
        assert "Exported at: unknown" in result
