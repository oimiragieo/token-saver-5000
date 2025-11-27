"""
AFM (Adaptive Focus Memory) Handler Functions

This module contains handler functions for AFM-related MCP tools:
- afm_add_message: Add messages to dialogue history
- afm_build_context: Build optimized context for current query
- afm_get_stats: Get dialogue statistics
- afm_clear_history: Clear dialogue history
- afm_export_history: Export dialogue history to JSON
- afm_import_history: Import dialogue history from JSON

These handlers implement the AFM system (arXiv:2511.12712v1) which manages
multi-turn conversations with adaptive fidelity based on recency, semantic
relevance, and importance classification.
"""

import logging
from typing import Any, Dict

from ..types import HandlerContext  # TypedDict for handler context

logger = logging.getLogger("semantic-modulator")


async def handle_afm_add_message(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Handle afm_add_message tool call.

    Adds a message to the AFM dialogue history with automatic importance
    classification (CRITICAL/RELEVANT/TRIVIAL).

    Args:
        context: Server context containing focus_manager
        args: Tool arguments with 'role' and 'content'

    Returns:
        Formatted confirmation message with dialogue stats

    Raises:
        ValueError: If role is invalid
    """
    role = args["role"]
    content = args["content"]
    focus_manager = context["focus_manager"]

    # Validation
    if role not in ["user", "assistant", "system"]:
        raise ValueError(f"Invalid role: {role}. Must be 'user', 'assistant', or 'system'")

    logger.info(f"AFM: Adding {role} message (turn {focus_manager.turn_counter})")

    # Add message
    msg = focus_manager.add_message(role, content)

    # Return confirmation with importance classification
    return f"""
💬 Message Added to Dialogue History

Turn: {msg.turn_index}
Role: {msg.role}
Importance: {msg.importance.value.upper()}
Content: {content[:100]}{'...' if len(content) > 100 else ''}

Dialogue Stats:
  Total messages: {len(focus_manager.messages)}
  Current turn: {focus_manager.turn_counter}

💡 Use afm_build_context(query, budget_tokens) to build optimized context
"""


async def handle_afm_build_context(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Handle afm_build_context tool call.

    Builds optimized context for current query using semantic similarity,
    recency weighting, and importance classification. Achieves ~66% token
    reduction while preserving safety-critical information.

    Args:
        context: Server context containing focus_manager
        args: Tool arguments with 'current_query', 'budget_tokens', and optional 'system_preamble'

    Returns:
        Formatted context with packing statistics

    Raises:
        ValueError: If budget_tokens is invalid
    """
    current_query = args["current_query"]
    budget_tokens = args["budget_tokens"]
    system_preamble = args.get("system_preamble")
    focus_manager = context["focus_manager"]

    # Validation
    if budget_tokens <= 0:
        raise ValueError(f"budget_tokens must be positive, got {budget_tokens}")

    logger.info(f"AFM: Building context for query (budget: {budget_tokens} tokens)")

    # Build context
    ctx, stats = focus_manager.build_context(
        current_query=current_query,
        budget_tokens=budget_tokens,
        system_preamble=system_preamble,
    )

    # Format context for display
    context_display = []
    for i, (role, content) in enumerate(ctx):
        preview = content[:150] + "..." if len(content) > 150 else content
        context_display.append(f"  [{i + 1}] {role}: {preview}")

    context_display_text = "\n".join(context_display)

    result = f"""
🧠 AFM Context Built Successfully

Query: {current_query[:100]}{'...' if len(current_query) > 100 else ''}

📊 Packing Statistics:
  Total messages processed: {stats.total_messages}
  ├─ FULL fidelity:         {stats.full_count}
  ├─ COMPRESSED:            {stats.compressed_count}
  ├─ PLACEHOLDER:           {stats.placeholder_count}
  └─ DROPPED:               {stats.dropped_count}

  Tokens used:              {stats.total_tokens} / {stats.budget_tokens}
  Budget utilization:       {stats.compression_ratio:.1%}

📝 Context Messages ({len(ctx)} total):
{context_display_text}

✅ Ready to send to LLM
💡 AFM achieved ~{100 * (1 - stats.compression_ratio):.0f}% token savings vs naive replay

---
Paper: Adaptive Focus Memory (arXiv:2511.12712v1)
"""

    return result


async def handle_afm_get_stats(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Handle afm_get_stats tool call.

    Returns dialogue statistics including total messages, current turn,
    and importance breakdown (critical/relevant/trivial counts).

    Args:
        context: Server context containing focus_manager
        args: Tool arguments (unused)

    Returns:
        Formatted statistics report
    """
    focus_manager = context["focus_manager"]
    stats = focus_manager.get_stats()

    return f"""
📊 AFM Dialogue Statistics

Total messages: {stats['total_messages']}
Current turn:   {stats['current_turn']}

Importance Breakdown:
  ⚠️  CRITICAL:  {stats['importance_breakdown']['critical']} messages
  📌 RELEVANT:  {stats['importance_breakdown']['relevant']} messages
  📝 TRIVIAL:   {stats['importance_breakdown']['trivial']} messages

💡 CRITICAL messages (e.g., allergies) are always preserved at full fidelity
"""


async def handle_afm_clear_history(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Handle afm_clear_history tool call.

    Clears all dialogue history and resets turn counter.

    Args:
        context: Server context containing focus_manager
        args: Tool arguments (unused)

    Returns:
        Confirmation message with previous state
    """
    focus_manager = context["focus_manager"]
    prev_count = len(focus_manager.messages)
    prev_turn = focus_manager.turn_counter

    focus_manager.clear_history()

    logger.info(f"AFM: Cleared dialogue history ({prev_count} messages)")

    return f"""
🗑️ AFM Dialogue History Cleared

Previous state:
  Messages: {prev_count}
  Turns:    {prev_turn}

Current state:
  Messages: 0
  Turns:    0

✅ Ready for new conversation
"""


async def handle_afm_export_history(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Handle afm_export_history tool call.

    Exports dialogue history to persistent storage with session ID.
    Saves current conversation state including all messages, turn counter,
    and metadata for later resume.

    Args:
        context: Server context containing focus_manager and persistence
        args: Tool arguments with optional 'session_id'

    Returns:
        Export confirmation or error message
    """
    session_id = args.get("session_id", "default")
    focus_manager = context["focus_manager"]
    persistence = context["persistence"]

    logger.info(f"Exporting AFM history to session: {session_id}")

    # Get current state
    messages = focus_manager.messages
    turn_counter = focus_manager.turn_counter

    if not messages:
        return """
💾 AFM Export

No dialogue history to export.

💡 Tip: Add messages with afm_add_message() first.
"""

    # Save to persistence
    metadata = {
        "exported_at": __import__("datetime").datetime.now().isoformat(),
        "message_count": len(messages),
        "turn_counter": turn_counter,
    }

    success = persistence.save_afm_history(
        session_id=session_id, messages=messages, turn_counter=turn_counter, metadata=metadata
    )

    if success:
        return f"""
💾 AFM Export Complete

Session ID: {session_id}
Messages exported: {len(messages)}
Current turn: {turn_counter}

💡 This conversation can be restored later with:
   afm_import_history(session_id="{session_id}")

Available sessions: {', '.join(persistence.list_afm_sessions())}
"""
    else:
        return """
❌ AFM Export Failed

Could not save dialogue history to persistent storage.

💡 Check logs for details.
"""


async def handle_afm_import_history(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Handle afm_import_history tool call.

    Imports dialogue history from persistent storage and restores
    conversation state. This replaces the current dialogue history.

    Args:
        context: Server context containing focus_manager and persistence
        args: Tool arguments with optional 'session_id'

    Returns:
        Import confirmation or error message
    """
    session_id = args.get("session_id", "default")
    focus_manager = context["focus_manager"]
    persistence = context["persistence"]

    logger.info(f"Importing AFM history from session: {session_id}")

    # Load from persistence
    data = persistence.load_afm_history(session_id)

    if not data:
        available_sessions = persistence.list_afm_sessions()
        return f"""
❌ AFM Import Failed

Session '{session_id}' not found.

Available sessions: {', '.join(available_sessions) if available_sessions else '(none)'}

💡 Tip: Use afm_export_history() to save conversations first.
"""

    # Validate data structure before importing
    try:
        # Required fields
        if "messages" not in data:
            raise ValueError("Missing 'messages' field in AFM export")
        if "turn_counter" not in data:
            raise ValueError("Missing 'turn_counter' field in AFM export")

        # Validate messages structure
        if not isinstance(data["messages"], list):
            raise ValueError("'messages' must be a list")

        # Validate each message has required fields
        for i, msg in enumerate(data["messages"]):
            if not hasattr(msg, "role"):
                raise ValueError(f"Message {i} missing 'role' field")
            if not hasattr(msg, "content"):
                raise ValueError(f"Message {i} missing 'content' field")
            if msg.role not in ["user", "assistant", "system"]:
                raise ValueError(f"Message {i} has invalid role: {msg.role}")

        # Validate turn_counter is a number
        if not isinstance(data["turn_counter"], int):
            raise ValueError("'turn_counter' must be an integer")

    except Exception as e:
        logger.error(f"AFM import validation failed: {e}")
        return f"""
❌ AFM Import Validation Failed

Session '{session_id}' contains invalid data: {str(e)}

This may be due to:
  • Corrupted export file
  • Incompatible version
  • Manual editing of export file

💡 Try re-exporting from the source session.
"""

    # Restore to focus manager
    focus_manager.messages = data["messages"]
    focus_manager.turn_counter = data["turn_counter"]

    metadata = data.get("metadata", {})

    logger.info(f"✅ Imported {len(data['messages'])} messages from {session_id}")

    return f"""
📥 AFM Import Complete

Session ID: {session_id}
Messages restored: {len(data['messages'])}
Turn counter restored: {data['turn_counter']}
Exported at: {metadata.get('exported_at', 'unknown')}

✅ Conversation state has been restored.

💡 Use afm_get_stats() to see current state.
"""
