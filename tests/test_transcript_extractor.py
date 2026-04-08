"""Tests for transcript extraction pipeline (src/transcript_extractor.py)."""

import pytest

from src.memory_api import MemoryAPI
from src.transcript_extractor import (
    ExtractionResult,
    extract_insights,
    ingest_transcript,
    _split_sentences,
    _has_signal,
    _strip_role_prefix,
)


@pytest.fixture(autouse=True)
def reset_memory():
    MemoryAPI.reset_singleton()
    yield
    MemoryAPI.reset_singleton()


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------


class TestSplitSentences:
    def test_splits_on_period(self):
        result = _split_sentences("First sentence. Second sentence.")
        assert len(result) == 2

    def test_splits_on_double_newline(self):
        result = _split_sentences("Paragraph one.\n\nParagraph two.")
        assert len(result) == 2

    def test_empty_string(self):
        assert _split_sentences("") == []

    def test_single_sentence(self):
        result = _split_sentences("Just one sentence here")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------


class TestHasSignal:
    def test_decision_signal(self):
        assert _has_signal("We decided to use PostgreSQL instead of MongoDB")

    def test_lesson_signal(self):
        assert _has_signal("The lesson learned was to always validate inputs")

    def test_gotcha_signal(self):
        assert _has_signal("Watch out for race conditions in async handlers")

    def test_issue_signal(self):
        assert _has_signal("There was a critical bug in the authentication flow")

    def test_pattern_signal(self):
        assert _has_signal("The best practice is to use dependency injection")

    def test_no_signal(self):
        assert not _has_signal("The sky is blue today")

    def test_preference_signal(self):
        assert _has_signal("I prefer using black for code formatting")

    def test_action_signal(self):
        assert _has_signal("Going forward we should run lint before commits")


# ---------------------------------------------------------------------------
# Role prefix stripping
# ---------------------------------------------------------------------------


class TestStripRolePrefix:
    def test_strips_user_prefix(self):
        assert _strip_role_prefix("[user] Hello there") == "Hello there"

    def test_strips_assistant_prefix(self):
        assert _strip_role_prefix("[assistant] Here is the plan") == "Here is the plan"

    def test_strips_human_colon(self):
        assert _strip_role_prefix("Human: What is this?") == "What is this?"

    def test_no_prefix(self):
        assert _strip_role_prefix("Just plain text") == "Just plain text"


# ---------------------------------------------------------------------------
# Insight extraction
# ---------------------------------------------------------------------------


class TestExtractInsights:
    def test_extracts_decision(self):
        transcript = "We decided to use FastAPI for the backend."
        results = extract_insights(transcript)
        assert len(results) >= 1
        assert results[0][1].category == "decision"

    def test_extracts_gotcha(self):
        transcript = "Watch out for the subtle race condition in the cache layer."
        results = extract_insights(transcript)
        assert len(results) >= 1
        assert results[0][1].category == "gotcha"

    def test_extracts_issue(self):
        transcript = "There was a critical bug causing timeouts under load."
        results = extract_insights(transcript)
        assert len(results) >= 1
        assert results[0][1].category == "issue"

    def test_mode_decisions_only(self):
        transcript = (
            "We decided to use Redis. "
            "Watch out for the cache bug. "
            "Going forward we should test more."
        )
        results = extract_insights(transcript, mode="decisions")
        for _, classification in results:
            # Should only match decision/action signals
            assert classification.category in ("decision", "pattern", "general")

    def test_mode_patterns_only(self):
        transcript = (
            "Always use black before commits. "
            "There was a critical bug yesterday. "
            "The best practice is dependency injection."
        )
        results = extract_insights(transcript, mode="patterns")
        assert len(results) >= 1

    def test_deduplicates_identical_sentences(self):
        transcript = "We decided to use Redis. We decided to use Redis."
        results = extract_insights(transcript)
        assert len(results) == 1

    def test_ignores_short_sentences(self):
        transcript = "Bug. Error. We decided to use a comprehensive approach."
        results = extract_insights(transcript)
        # "Bug." and "Error." are too short
        texts = [r[0] for r in results]
        assert not any(len(t) < 20 for t in texts)

    def test_empty_transcript(self):
        assert extract_insights("") == []

    def test_no_signals_returns_empty(self):
        transcript = "The weather is nice today. I had lunch at noon."
        assert extract_insights(transcript) == []


# ---------------------------------------------------------------------------
# Full ingestion pipeline
# ---------------------------------------------------------------------------


class TestIngestTranscript:
    def test_basic_ingestion(self):
        api = MemoryAPI()
        result = ingest_transcript(
            "We decided to use PostgreSQL. Watch out for N+1 queries.",
            memory_api=api,
        )
        assert isinstance(result, ExtractionResult)
        assert result.stored_count >= 1
        assert result.extracted_count == result.stored_count
        assert len(result.insights) == result.stored_count

    def test_insights_stored_in_memory_api(self):
        api = MemoryAPI()
        ingest_transcript(
            "We decided to use PostgreSQL for the database layer.",
            memory_api=api,
        )
        memories = api.list_memories()
        assert len(memories) >= 1
        assert any("PostgreSQL" in m["text"] for m in memories)

    def test_scoping_preserved(self):
        api = MemoryAPI()
        ingest_transcript(
            "Always use black before committing code.",
            memory_api=api,
            workspace_id="acme",
            user_id="alice",
        )
        scoped = api.list_memories(workspace_id="acme", user_id="alice")
        assert len(scoped) >= 1

    def test_source_tag(self):
        api = MemoryAPI()
        result = ingest_transcript(
            "We decided to switch from REST to GraphQL.",
            memory_api=api,
            source="session-hook",
        )
        assert result.source == "session-hook"
        if result.insights:
            assert result.insights[0]["source"] == "session-hook"

    def test_mode_filtering(self):
        api = MemoryAPI()
        result_all = ingest_transcript(
            "We decided X. Watch out for Y. Bug in Z.",
            mode="all",
            memory_api=api,
        )
        api2 = MemoryAPI()
        result_decisions = ingest_transcript(
            "We decided X. Watch out for Y. Bug in Z.",
            mode="decisions",
            memory_api=api2,
        )
        assert result_decisions.stored_count <= result_all.stored_count

    def test_empty_transcript(self):
        api = MemoryAPI()
        result = ingest_transcript("", memory_api=api)
        assert result.stored_count == 0
        assert result.insights == []

    def test_metadata_includes_confidence(self):
        api = MemoryAPI()
        result = ingest_transcript(
            "We decided to use async/await for all handlers.",
            memory_api=api,
        )
        if result.insights:
            meta = result.insights[0].get("metadata", {})
            assert "confidence" in meta
            assert "extraction_mode" in meta

    def test_total_sentences_counted(self):
        api = MemoryAPI()
        result = ingest_transcript(
            "First sentence. Second sentence. Third sentence.",
            memory_api=api,
        )
        assert result.total_sentences == 3
