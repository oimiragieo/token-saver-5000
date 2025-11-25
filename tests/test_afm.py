"""
Tests for Adaptive Focus Memory (AFM) module

Tests the dialogue memory management system described in:
"Adaptive Focus Memory for Language Models" (arXiv:2511.12712v1)
"""

import pytest
from src.afm import (
    FocusManager,
    AFMConfig,
    Message,
    ImportanceLevel,
    TokenCounter,
    HeuristicCompressor,
    ImportanceClassifier,
)


class TestTokenCounter:
    """Test token counting functionality"""

    def test_count_basic(self):
        """Test basic token counting"""
        counter = TokenCounter()
        text = "This is a test sentence."
        tokens = counter.count(text)
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_count_empty(self):
        """Test counting empty string"""
        counter = TokenCounter()
        assert counter.count("") == 0

    def test_count_consistency(self):
        """Test that same text always returns same count"""
        counter = TokenCounter()
        text = "Consistent token counting is important."
        count1 = counter.count(text)
        count2 = counter.count(text)
        assert count1 == count2


class TestMessage:
    """Test Message dataclass"""

    def test_message_creation(self):
        """Test creating a message"""
        msg = Message(role="user", content="Hello world", turn_index=0)
        assert msg.role == "user"
        assert msg.content == "Hello world"
        assert msg.turn_index == 0
        assert msg.message_id == "msg_0"

    def test_message_with_custom_id(self):
        """Test message with custom ID"""
        msg = Message(role="assistant", content="Hi there", turn_index=1, message_id="custom_id")
        assert msg.message_id == "custom_id"


class TestImportanceClassifier:
    """Test importance classification"""

    def test_critical_classification(self):
        """Test that allergy messages are classified as critical"""
        classifier = ImportanceClassifier(use_llm=False)
        msg = Message(
            role="user",
            content="I have a severe peanut allergy that is life-threatening.",
            turn_index=0,
        )
        importance = classifier.classify(msg)
        assert importance == ImportanceLevel.CRITICAL

    def test_relevant_classification(self):
        """Test relevant classification"""
        classifier = ImportanceClassifier(use_llm=False)
        msg = Message(role="user", content="I prefer vegetarian food.", turn_index=0)
        importance = classifier.classify(msg)
        assert importance == ImportanceLevel.RELEVANT

    def test_trivial_classification(self):
        """Test trivial classification"""
        classifier = ImportanceClassifier(use_llm=False)
        msg = Message(role="user", content="ok", turn_index=0)
        importance = classifier.classify(msg)
        assert importance == ImportanceLevel.TRIVIAL

    def test_multiple_critical_keywords(self):
        """Test message with multiple critical keywords"""
        classifier = ImportanceClassifier(use_llm=False)
        critical_phrases = [
            "I am allergic to shellfish",
            "This is a medical emergency",
            "I cannot eat peanuts",
            "Severe allergy warning",
            "Life-threatening condition",
        ]
        for phrase in critical_phrases:
            msg = Message(role="user", content=phrase, turn_index=0)
            importance = classifier.classify(msg)
            assert importance == ImportanceLevel.CRITICAL, f"Failed for: {phrase}"

    def test_critical_typo_resilience(self):
        """Test that typos in critical keywords are still detected via fuzzy matching

        Safety-critical: Typos like "alergy" should still be detected as CRITICAL
        to prevent medical information from being compressed/lost.
        """
        classifier = ImportanceClassifier(use_llm=False)

        # Typos that should still be detected (80%+ similar to real keywords)
        typo_phrases = [
            "I have a severe peanut alergy",  # "alergy" → "allergy" (92.3% similar)
            "I am alergic to shellfish",  # "alergic" → "allergic" (93.3% similar)
            "This is a sevear reaction",  # "sevear" → "severe" (83.3% similar)
            "Life threatning condition",  # "threatning" → "threatening" (95.2% similar)
            "Deadley poison",  # "deadley" → "deadly" (92.3% similar)
        ]

        for phrase in typo_phrases:
            msg = Message(role="user", content=phrase, turn_index=0)
            importance = classifier.classify(msg)
            assert importance == ImportanceLevel.CRITICAL, (
                f"SAFETY FAILURE: Typo not detected as CRITICAL: '{phrase}'\n"
                f"Got: {importance.value}, Expected: CRITICAL\n"
                f"This could lead to medical information being lost!"
            )


class TestHeuristicCompressor:
    """Test heuristic compression"""

    def test_compress_short_text(self):
        """Test compressing short text"""
        counter = TokenCounter()
        compressor = HeuristicCompressor(counter)

        text = "This is a short message."
        compressed = compressor.compress(text, target_tokens=5)

        assert len(compressed) > 0
        assert counter.count(compressed) <= counter.count(text)

    def test_compress_preserves_meaning(self):
        """Test that compression preserves key information"""
        counter = TokenCounter()
        compressor = HeuristicCompressor(counter)

        text = "I have a severe peanut allergy. It is life-threatening. I must avoid all peanut products."
        compressed = compressor.compress(text, target_tokens=10)

        # Should preserve critical keywords
        assert "allergy" in compressed.lower() or "peanut" in compressed.lower()

    def test_compress_long_text(self):
        """Test compressing longer text"""
        counter = TokenCounter()
        compressor = HeuristicCompressor(counter)

        text = (
            "Thailand is a beautiful country with amazing culture. "
            "Bangkok is the capital city with many temples. "
            "Chiang Mai is in the north and has mountains. "
            "Phuket has beautiful beaches. "
            "The food is delicious but can contain peanuts."
        )
        compressed = compressor.compress(text, target_tokens=15)

        assert len(compressed) > 0
        tokens = counter.count(compressed)
        original_tokens = counter.count(text)
        assert tokens < original_tokens


class TestFocusManager:
    """Test FocusManager main functionality"""

    def test_initialization(self):
        """Test FocusManager initialization"""
        config = AFMConfig()
        manager = FocusManager(config)
        assert manager.turn_counter == 0
        assert len(manager.messages) == 0

    def test_add_message(self):
        """Test adding messages"""
        config = AFMConfig()
        manager = FocusManager(config)

        msg = manager.add_message("user", "Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.turn_index == 0
        assert len(manager.messages) == 1
        assert manager.turn_counter == 1

        msg2 = manager.add_message("assistant", "Hi there")
        assert msg2.turn_index == 1
        assert manager.turn_counter == 2

    def test_importance_classification_on_add(self):
        """Test that importance is classified when adding messages"""
        config = AFMConfig()
        manager = FocusManager(config)

        # Add critical message
        critical_msg = manager.add_message("user", "I have a severe peanut allergy")
        assert critical_msg.importance == ImportanceLevel.CRITICAL

        # Add trivial message
        trivial_msg = manager.add_message("user", "ok")
        assert trivial_msg.importance == ImportanceLevel.TRIVIAL

    def test_build_context_simple(self):
        """Test building context for simple conversation"""
        config = AFMConfig()
        manager = FocusManager(config)

        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi there!")

        context, stats = manager.build_context(current_query="How are you?", budget_tokens=100)

        assert len(context) > 0
        assert stats.total_messages == 2
        assert stats.total_tokens <= stats.budget_tokens

    def test_build_context_with_preamble(self):
        """Test building context with system preamble"""
        config = AFMConfig()
        manager = FocusManager(config)

        manager.add_message("user", "Hello")

        context, stats = manager.build_context(
            current_query="Hi", budget_tokens=100, system_preamble="You are a helpful assistant."
        )

        # First message should be system preamble
        assert len(context) > 0
        assert context[0][0] == "system"
        assert "helpful assistant" in context[0][1]

    def test_allergy_retention_short_conversation(self):
        """
        Test that allergy is retained in short conversation
        This is the key safety test from the AFM paper
        """
        config = AFMConfig()
        manager = FocusManager(config)

        # Short conversation with allergy
        manager.add_message("user", "I'm planning a trip to Thailand")
        manager.add_message("assistant", "That's great!")
        manager.add_message("user", "I have a severe peanut allergy that is life-threatening")
        manager.add_message("assistant", "I'll keep that in mind")

        # Ask about food
        context, stats = manager.build_context(
            current_query="What street food should I try?", budget_tokens=500
        )

        # Check that allergy is in context
        context_text = " ".join(content for _, content in context)
        assert (
            "allergy" in context_text.lower() or "peanut" in context_text.lower()
        ), "Allergy information should be retained in context"

    def test_allergy_retention_medium_conversation(self):
        """
        Test that allergy is retained across many intervening turns
        This is the challenging scenario from the AFM paper
        """
        config = AFMConfig(half_life=12)
        manager = FocusManager(config)

        # Add early allergy
        manager.add_message("user", "I'm planning a trip to Thailand")
        manager.add_message("user", "I have a severe peanut allergy that is life-threatening")
        manager.add_message(
            "assistant", "Noted, I'll keep that in mind for all food recommendations"
        )

        # Add many intervening messages (like in the paper)
        intervening = [
            ("user", "What are the best destinations?"),
            ("assistant", "Bangkok, Chiang Mai, Phuket are popular."),
            ("user", "How do I get around?"),
            ("assistant", "Trains, buses, and flights are available."),
            ("user", "I want to try Muay Thai"),
            ("assistant", "You can watch matches or take classes."),
            ("user", "Tell me about temples"),
            ("assistant", "Thailand has beautiful Buddhist temples."),
        ]

        for role, content in intervening:
            manager.add_message(role, content)

        # Now ask about food (should trigger allergy memory)
        context, stats = manager.build_context(
            current_query="What Thai street food should I try?", budget_tokens=800
        )

        # Check that allergy is still in context despite distance
        context_text = " ".join(content for _, content in context)
        assert (
            "allergy" in context_text.lower() or "peanut" in context_text.lower()
        ), "Allergy information should be retained even with intervening turns"

    def test_token_budget_respected(self):
        """Test that token budget is strictly respected"""
        config = AFMConfig()
        manager = FocusManager(config)

        # Add many messages
        for i in range(20):
            manager.add_message("user", f"This is message number {i} with some content")
            manager.add_message("assistant", f"Response to message {i}")

        # Build context with tight budget
        budget = 200
        context, stats = manager.build_context(
            current_query="Tell me everything", budget_tokens=budget
        )

        # Should respect budget
        assert stats.total_tokens <= budget
        assert stats.budget_tokens == budget

    def test_chronological_ordering(self):
        """Test that messages are packed in chronological order"""
        config = AFMConfig()
        manager = FocusManager(config)

        # Add messages
        manager.add_message("user", "First message")
        manager.add_message("assistant", "Second message")
        manager.add_message("user", "Third message")

        context, stats = manager.build_context(current_query="Fourth message", budget_tokens=500)

        # Extract turn indices from context
        # Messages should appear in chronological order
        indices = []
        for role, content in context:
            if "First" in content:
                indices.append(0)
            elif "Second" in content:
                indices.append(1)
            elif "Third" in content:
                indices.append(2)

        # Check chronological order
        assert indices == sorted(indices), "Messages should be in chronological order"

    def test_fidelity_levels_assigned(self):
        """Test that different fidelity levels are assigned appropriately"""
        config = AFMConfig(tau_high=0.7, tau_mid=0.4)
        manager = FocusManager(config)

        # Add a critical message (should get FULL even if not similar)
        manager.add_message("user", "I have a severe peanut allergy")

        # Add a somewhat relevant message
        manager.add_message("user", "I like Thai food")

        # Add a trivial message
        manager.add_message("user", "ok")

        # Query about food (similar to middle message)
        context, stats = manager.build_context(
            current_query="What food should I eat?", budget_tokens=1000
        )

        # Should have different fidelity representations
        assert stats.full_count > 0 or stats.compressed_count > 0 or stats.placeholder_count > 0

    def test_compression_ratio(self):
        """Test that compression achieves significant savings"""
        config = AFMConfig()
        manager = FocusManager(config)

        # Add substantial conversation
        for i in range(10):
            manager.add_message(
                "user",
                f"This is a longer message number {i} with quite a bit of content that needs to be compressed",
            )
            manager.add_message(
                "assistant",
                f"This is a detailed response to message {i} with lots of information that takes up space",
            )

        # Build with moderate budget
        context, stats = manager.build_context(
            current_query="Tell me about everything we discussed", budget_tokens=500
        )

        # Should achieve some compression
        # (not all messages at full fidelity)
        assert stats.compressed_count > 0 or stats.placeholder_count > 0 or stats.dropped_count > 0

    def test_clear_history(self):
        """Test clearing dialogue history"""
        config = AFMConfig()
        manager = FocusManager(config)

        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi")
        assert len(manager.messages) == 2

        manager.clear_history()
        assert len(manager.messages) == 0
        assert manager.turn_counter == 0

    def test_get_stats(self):
        """Test getting dialogue statistics"""
        config = AFMConfig()
        manager = FocusManager(config)

        manager.add_message("user", "I have a severe allergy")  # critical
        manager.add_message("user", "I prefer spicy food")  # relevant
        manager.add_message("user", "ok")  # trivial

        stats = manager.get_stats()
        assert stats["total_messages"] == 3
        assert stats["current_turn"] == 3
        assert stats["importance_breakdown"]["critical"] == 1
        assert stats["importance_breakdown"]["relevant"] == 1
        assert stats["importance_breakdown"]["trivial"] == 1


class TestRecencyWeighting:
    """Test recency weighting calculations"""

    def test_recency_decay(self):
        """Test that recency weight decays over time"""
        config = AFMConfig(half_life=12)
        manager = FocusManager(config)

        # Add old message
        old_msg = manager.add_message("user", "Old message")

        # Add many intervening messages
        for i in range(24):  # 2 half-lives
            manager.add_message("user", f"Message {i}")

        # Calculate recency for old message
        recency = manager._calculate_recency_weight(old_msg, manager.turn_counter)

        # After 24 turns with half_life=12, weight should be ~0.25 (0.5^2)
        expected = 0.25
        assert 0.2 <= recency <= 0.3, f"Expected ~{expected}, got {recency}"

    def test_recent_message_high_weight(self):
        """Test that recent messages have high recency weight"""
        config = AFMConfig(half_life=12)
        manager = FocusManager(config)

        recent_msg = manager.add_message("user", "Recent message")

        # Calculate recency at the message's own turn (k=0, no decay)
        recency = manager._calculate_recency_weight(recent_msg, recent_msg.turn_index)

        # Should be 1.0 (no decay when k=0)
        assert recency == 1.0


class TestScoring:
    """Test relevance scoring"""

    def test_critical_message_max_score(self):
        """Test that critical messages get maximum score regardless of similarity"""
        config = AFMConfig()
        manager = FocusManager(config)

        # Add critical message about completely different topic
        critical_msg = manager.add_message("user", "I have a severe peanut allergy")
        assert critical_msg.importance == ImportanceLevel.CRITICAL

        # Query about unrelated topic
        query = "What is the capital of France?"
        query_embedding = manager.embedder.encode([query])[0]

        # Score should still be 1.0 for critical
        score = manager._calculate_relevance_score(
            critical_msg, query_embedding, manager.turn_counter
        )

        assert score == 1.0, "Critical messages should always get score of 1.0"


# Run tests if called directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
