"""
Adaptive Focus Memory (AFM) for Language Models

Implementation of "Adaptive Focus Memory for Language Models" (arXiv:2511.12712v1)
by Christopher Cruz, Purdue University.

This module provides dialogue-specific memory management that:
- Assigns each message one of three fidelity levels (FULL, COMPRESSED, PLACEHOLDER)
- Uses semantic similarity, recency weighting, and importance classification
- Packs messages chronologically under a strict token budget
- Reduces token usage by ~66% while preserving safety-critical information

Key differences from document compression (semantic_compressor.py):
- Optimized for multi-turn dialogue (not long documents)
- Temporal recency weighting with half-life decay
- Message-level granularity (not paragraph-level)
- Strict chronological ordering (preserves conversation flow)

License: CC BY 4.0 (as specified in paper)
"""

import enum
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from difflib import SequenceMatcher
import time
import math

import numpy as np
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity

# Import EmbeddingManager for shared model caching
from .embeddings import EmbeddingManager

# Try to import sentence_transformers, fall back to hash-based embedder
try:
    from sentence_transformers import SentenceTransformer  # noqa: F401

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("afm")


# ============================================================================
# Enumerations
# ============================================================================


class FidelityLevel(enum.Enum):
    """
    Three fidelity levels as described in the AFM paper (Section 3.3)
    """

    FULL = "full"  # Include message verbatim
    COMPRESSED = "compressed"  # Include compressed summary
    PLACEHOLDER = "placeholder"  # Include short stub only


class ImportanceLevel(enum.Enum):
    """
    Importance classification levels (Section 3.2)
    Used by LLM-based classifier or heuristics
    """

    CRITICAL = "critical"  # Safety-critical, force-elevated score
    RELEVANT = "relevant"  # Semantically relevant
    TRIVIAL = "trivial"  # Low importance


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class Message:
    """
    Represents a single dialogue turn with metadata
    """

    role: str  # "user", "assistant", "system"
    content: str
    turn_index: int  # Position in dialogue (0-indexed)
    message_id: str = None

    # Computed lazily
    embedding: Optional[np.ndarray] = None
    importance: ImportanceLevel = ImportanceLevel.TRIVIAL
    relevance_score: float = 0.0
    intended_fidelity: FidelityLevel = FidelityLevel.PLACEHOLDER

    # Compressed representations (cached)
    compressed_summary: Optional[str] = None
    placeholder_stub: Optional[str] = None

    # Timestamps
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.message_id is None:
            self.message_id = f"msg_{self.turn_index}"


@dataclass
class AFMConfig:
    """
    Configuration for Adaptive Focus Memory
    """

    # Scoring thresholds (Section 3.3)
    tau_high: float = 0.45  # Threshold for FULL fidelity
    tau_mid: float = 0.25  # Threshold for COMPRESSED fidelity

    # Recency weighting (Section 3.2)
    half_life: int = 12  # Turns until weight decays to 50%

    # Token budgeting
    max_stub_tokens: int = 12
    target_compression_ratio: float = 0.33  # Target 1/3 of original for summaries

    # Embedding model
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # LLM-based features (optional)
    use_llm_importance: bool = False
    use_llm_compression: bool = False
    llm_api_key: Optional[str] = None
    llm_model: str = "gpt-4o-mini"


@dataclass
class PackingStats:
    """
    Statistics from context packing
    """

    total_messages: int
    full_count: int
    compressed_count: int
    placeholder_count: int
    dropped_count: int
    total_tokens: int
    budget_tokens: int
    compression_ratio: float


# ============================================================================
# Token Counting
# ============================================================================


class TokenCounter:
    """
    Estimates token counts using tiktoken when available, falls back to word count
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except Exception:
            try:
                self.encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.encoding = None
                logger.warning("tiktoken not available, using word count fallback")

    def count(self, text: str) -> int:
        """Count tokens in text"""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Fallback: approximate as 1.3 tokens per word
            return int(len(text.split()) * 1.3)


# ============================================================================
# Compressors
# ============================================================================


class Compressor:
    """Base class for message compressors"""

    def compress(self, content: str, target_tokens: int) -> str:
        """
        Compress message content to approximately target_tokens

        Args:
            content: Original message content
            target_tokens: Target token count

        Returns:
            Compressed summary
        """
        raise NotImplementedError


class HeuristicCompressor(Compressor):
    """
    Fully local, extractive compressor using simple heuristics
    No external API calls required
    """

    def __init__(self, token_counter: TokenCounter):
        self.token_counter = token_counter

    def compress(self, content: str, target_tokens: int) -> str:
        """
        Extractive compression using sentence ranking

        Strategy:
        1. Split into sentences
        2. Rank by: length (prefer medium), position (prefer early), keywords
        3. Select top sentences until budget exhausted
        4. Preserve chronological order
        """
        # Split into sentences (simple heuristic)
        sentences = []
        for chunk in content.split(". "):
            chunk = chunk.strip()
            if chunk:
                if not chunk.endswith("."):
                    chunk += "."
                sentences.append(chunk)

        if not sentences:
            return content[: target_tokens * 4]  # Fallback: char truncation

        # Rank sentences
        scored_sentences = []
        for i, sent in enumerate(sentences):
            tokens = self.token_counter.count(sent)

            # Prefer medium-length sentences (not too short, not too long)
            length_score = 1.0 - abs(tokens - 15) / 30.0
            length_score = max(0.1, length_score)

            # Prefer early sentences
            position_score = 1.0 - (i / len(sentences)) * 0.3

            # Simple keyword boost
            keyword_boost = (
                1.2
                if any(
                    kw in sent.lower()
                    for kw in [
                        "critical",
                        "important",
                        "allergy",
                        "severe",
                        "warning",
                        "must",
                        "never",
                    ]
                )
                else 1.0
            )

            score = length_score * position_score * keyword_boost
            scored_sentences.append((score, i, sent, tokens))

        # Sort by score (descending)
        scored_sentences.sort(reverse=True, key=lambda x: x[0])

        # Select sentences until budget exhausted
        selected = []
        current_tokens = 0
        for score, idx, sent, tokens in scored_sentences:
            if current_tokens + tokens <= target_tokens:
                selected.append((idx, sent))
                current_tokens += tokens
            if current_tokens >= target_tokens * 0.9:  # Stop at 90% to be safe
                break

        # Sort selected by original position to preserve chronology
        selected.sort(key=lambda x: x[0])

        if not selected:
            # If no sentences fit, take first sentence truncated
            first = sentences[0]
            max_chars = target_tokens * 4
            return first[:max_chars] + "..." if len(first) > max_chars else first

        # Join selected sentences
        summary = " ".join(sent for _, sent in selected)

        # Ensure we didn't exceed budget
        actual_tokens = self.token_counter.count(summary)
        if actual_tokens > target_tokens:
            # Truncate
            max_chars = target_tokens * 4
            summary = summary[:max_chars] + "..."

        return summary


class LLMCompressor(Compressor):
    """
    Abstractive compressor using an LLM
    Requires API key

    Note: Not implemented in this reference (requires OpenAI client)
    Placeholder for future integration
    """

    def __init__(self, token_counter: TokenCounter, api_key: str, model: str = "gpt-4o-mini"):
        self.token_counter = token_counter
        self.api_key = api_key
        self.model = model
        logger.warning("LLMCompressor not fully implemented, falling back to heuristic")

    def compress(self, content: str, target_tokens: int) -> str:
        """
        Use LLM to generate abstractive summary

        For now, falls back to heuristic compression
        """
        # NOTE: LLM-based compression is intentionally not implemented.
        # The heuristic-based approach (extractive summary) provides:
        # 1. Zero latency (no API calls)
        # 2. Zero cost (no external API fees)
        # 3. Predictable token usage
        # 4. Privacy preservation (local-first)
        # 5. Sufficient quality for most use cases
        # If needed, users can implement custom LLM compression via subclassing.

        # For reference implementation, use heuristic
        heuristic = HeuristicCompressor(self.token_counter)
        return heuristic.compress(content, target_tokens)


# ============================================================================
# Embedders
# ============================================================================


class Embedder:
    """Base class for embedding models"""

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    """Use sentence-transformers for embeddings"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not available")
        # Use EmbeddingManager for shared model caching
        embedding_manager = EmbeddingManager()
        self.model = embedding_manager.get_text_embedder(model_name)
        logger.info(f"Using cached SentenceTransformer: {model_name}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into semantic embeddings using SentenceTransformer.

        This is the primary encoding method for AFM, providing true semantic
        similarity based on pre-trained language models. The embeddings are
        used to calculate importance scores and semantic similarity between
        dialogue turns.

        Args:
            texts: List of text strings to encode (e.g., message contents)

        Returns:
            NumPy array of shape (len(texts), embedding_dim) where embedding_dim
            is 384 for all-MiniLM-L6-v2 (default model). Each row is a dense
            semantic embedding vector.

        Note:
            - Uses cached model instance from EmbeddingManager (singleton)
            - Embeddings are deterministic for the same input text
            - Uses CPU by default (no GPU required)
        """
        return self.model.encode(texts, convert_to_numpy=True)


class HashingEmbedder(Embedder):
    """
    Fallback embedder using hash-based vectors
    Not semantic, but provides basic similarity
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        logger.warning("Using HashingEmbedder (not semantic, for offline operation only)")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Create deterministic hash-based embeddings"""
        embeddings = []
        for text in texts:
            # Simple hash-based embedding
            words = text.lower().split()
            vec = np.zeros(self.dim)
            for word in words:
                h = hash(word) % self.dim
                vec[h] += 1.0
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            embeddings.append(vec)
        return np.array(embeddings)


# ============================================================================
# Importance Classifier
# ============================================================================


class ImportanceClassifier:
    """
    Classifies message importance
    Can use LLM or heuristics
    """

    def __init__(self, use_llm: bool = False, api_key: Optional[str] = None):
        self.use_llm = use_llm
        self.api_key = api_key
        if use_llm and not api_key:
            logger.warning(
                "LLM importance classification requested but no API key, using heuristic"
            )
            self.use_llm = False

    def classify(self, message: Message) -> ImportanceLevel:
        """
        Classify message importance

        Returns one of: CRITICAL, RELEVANT, TRIVIAL
        """
        if self.use_llm:
            return self._classify_llm(message)
        else:
            return self._classify_heuristic(message)

    def _fuzzy_match_keywords(
        self, text: str, keywords: List[str], threshold: float = 0.80
    ) -> bool:
        """
        Fuzzy match keywords to handle typos (e.g., "alergy" → "allergy")

        Handles hyphenated keywords like "life-threatening" by splitting them
        into parts and checking each part separately.

        Args:
            text: Text to search in (should be lowercased)
            keywords: List of keywords to match
            threshold: Similarity threshold (0.80 = 80% similar)

        Returns:
            True if any word in text fuzzy-matches a keyword
        """
        words = text.split()
        for word in words:
            # Skip very short words to avoid false positives
            if len(word) < 4:
                continue

            for keyword in keywords:
                # For multi-word keywords (with spaces), check exact match
                if " " in keyword:
                    if keyword in text:
                        return True
                    continue

                # For hyphenated keywords, split and check each part
                # e.g., "life-threatening" → ["life", "threatening"]
                if "-" in keyword:
                    keyword_parts = keyword.split("-")
                    for part in keyword_parts:
                        if len(part) < 4:
                            continue
                        similarity = SequenceMatcher(None, word, part).ratio()
                        if similarity >= threshold:
                            return True
                else:
                    # Single word fuzzy matching
                    similarity = SequenceMatcher(None, word, keyword).ratio()
                    if similarity >= threshold:
                        return True

        return False

    def _classify_heuristic(self, message: Message) -> ImportanceLevel:
        """
        Heuristic classification based on keywords and patterns

        Now includes fuzzy matching to handle typos in critical keywords
        (e.g., "alergy" → "allergy", "sevear" → "severe")
        """
        content_lower = message.content.lower()

        # Critical keywords (safety-sensitive)
        critical_keywords = [
            "allergy",
            "allergic",
            "severe",
            "life-threatening",
            "critical",
            "emergency",
            "danger",
            "fatal",
            "deadly",
            "poisonous",
            "toxic",
            "medical condition",
            "cannot eat",
            "cannot have",
            "avoid",
            "never",
            "must not",
            "forbidden",
        ]

        # Relevant keywords
        relevant_keywords = [
            "important",
            "prefer",
            "like",
            "dislike",
            "want",
            "need",
            "requirement",
            "constraint",
            "limitation",
            "restriction",
        ]

        # Check for critical keywords (exact match first, then fuzzy)
        if any(kw in content_lower for kw in critical_keywords):
            return ImportanceLevel.CRITICAL

        # Fuzzy match for typos (80% similarity threshold)
        # Only check critical keywords to avoid performance impact
        if self._fuzzy_match_keywords(content_lower, critical_keywords, threshold=0.80):
            logger.info(f"Fuzzy matched critical keyword in: '{message.content[:50]}...'")
            return ImportanceLevel.CRITICAL

        # Check for relevant keywords (exact match only - less critical)
        if any(kw in content_lower for kw in relevant_keywords):
            return ImportanceLevel.RELEVANT

        # Check message role
        if message.role == "system":
            return ImportanceLevel.RELEVANT

        # Check length (very short messages often trivial)
        if len(message.content.split()) < 5:
            return ImportanceLevel.TRIVIAL

        # Default
        return ImportanceLevel.TRIVIAL

    def _classify_llm(self, message: Message) -> ImportanceLevel:
        """
        LLM-based classification

        Not implemented in reference - would call OpenAI API
        Falls back to heuristic
        """
        # NOTE: LLM-based importance classification is intentionally not implemented.
        # The heuristic-based approach (safety keywords + length) provides:
        # 1. Zero latency (no API calls)
        # 2. Zero cost (no external API fees)
        # 3. Deterministic results
        # 4. Privacy preservation (local-first)
        # 5. High accuracy for safety-critical detection (allergies, medical, etc.)
        # If needed, users can implement custom LLM integration via subclassing.
        return self._classify_heuristic(message)


# ============================================================================
# Main AFM Class
# ============================================================================


class FocusManager:
    """
    Adaptive Focus Memory Manager

    Manages multi-turn dialogue history with adaptive fidelity.

    Usage:
        manager = FocusManager(config)
        manager.add_message("user", "I have a peanut allergy")
        manager.add_message("assistant", "Noted, I'll keep that in mind")
        ...
        context = manager.build_context(
            current_query="What Thai food should I try?",
            budget_tokens=800
        )
    """

    def __init__(self, config: Optional[AFMConfig] = None):
        self.config = config or AFMConfig()

        # Initialize components
        self.token_counter = TokenCounter(model_name="gpt-4o-mini")

        # Embedder
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embedder = SentenceTransformerEmbedder(self.config.embedding_model_name)
        else:
            self.embedder = HashingEmbedder()

        # Compressor
        if self.config.use_llm_compression and self.config.llm_api_key:
            self.compressor = LLMCompressor(
                self.token_counter, self.config.llm_api_key, self.config.llm_model
            )
        else:
            self.compressor = HeuristicCompressor(self.token_counter)

        # Importance classifier
        self.importance_classifier = ImportanceClassifier(
            use_llm=self.config.use_llm_importance, api_key=self.config.llm_api_key
        )

        # Dialogue history
        self.messages: List[Message] = []
        self.turn_counter = 0

        logger.info(f"FocusManager initialized with config: {self.config}")

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> Message:
        """
        Add a new message to dialogue history

        Args:
            role: "user", "assistant", or "system"
            content: Message content
            metadata: Optional metadata dict

        Returns:
            Created Message object
        """
        # Validate inputs
        if not content or not content.strip():
            raise ValueError("Message content cannot be empty or whitespace-only")
        if role not in ["user", "assistant", "system"]:
            raise ValueError(f"Invalid role: {role}. Must be 'user', 'assistant', or 'system'")

        msg = Message(
            role=role,
            content=content,
            turn_index=self.turn_counter,
            message_id=f"{role}_{self.turn_counter}",
        )

        # Classify importance (done eagerly)
        msg.importance = self.importance_classifier.classify(msg)

        self.messages.append(msg)
        self.turn_counter += 1

        logger.debug(f"Added message {msg.message_id}: {msg.importance.value}")

        return msg

    def _embed_message(self, message: Message) -> np.ndarray:
        """
        Compute embedding for message (lazily, with caching)
        """
        if message.embedding is None:
            message.embedding = self.embedder.encode([message.content])[0]
        return message.embedding

    def _calculate_recency_weight(self, message: Message, current_turn: int) -> float:
        """
        Calculate recency weight with half-life decay

        w_recency = 0.5^(k / h)
        where k = turns since message, h = half_life

        Section 3.2 of AFM paper
        """
        k = current_turn - message.turn_index
        h = self.config.half_life
        weight = math.pow(0.5, k / h)
        return weight

    def _calculate_similarity(
        self, msg_embedding: np.ndarray, query_embedding: np.ndarray
    ) -> float:
        """Calculate cosine similarity between embeddings"""
        sim = cosine_similarity([msg_embedding], [query_embedding])[0][0]
        return float(sim)

    def _calculate_relevance_score(
        self, message: Message, query_embedding: np.ndarray, current_turn: int
    ) -> float:
        """
        Calculate relevance score for message

        Implements the piecewise scoring function from Section 3.2:
        - CRITICAL: score = 1.0 (force-elevated)
        - RELEVANT: score = max(0, sim) * (0.5 + 0.5 * w_recency)
        - TRIVIAL: score = max(0, sim) * (0.25 * w_recency)

        Args:
            message: Message to score
            query_embedding: Embedding of current query
            current_turn: Current turn index

        Returns:
            Relevance score in [0, 1]
        """
        # Get embeddings
        msg_embedding = self._embed_message(message)

        # Calculate similarity
        sim = self._calculate_similarity(msg_embedding, query_embedding)
        sim = max(0.0, sim)  # Clamp negative similarities to 0

        # Calculate recency weight
        w_recency = self._calculate_recency_weight(message, current_turn)

        # Piecewise scoring based on importance
        if message.importance == ImportanceLevel.CRITICAL:
            score = 1.0  # Force-elevated
        elif message.importance == ImportanceLevel.RELEVANT:
            score = sim * (0.5 + 0.5 * w_recency)
        else:  # TRIVIAL
            score = sim * (0.25 * w_recency)

        return score

    def _assign_intended_fidelity(self, message: Message) -> FidelityLevel:
        """
        Assign intended fidelity level based on relevance score

        Section 3.3 of AFM paper:
        - score >= tau_high → FULL
        - score >= tau_mid → COMPRESSED
        - else → PLACEHOLDER
        """
        score = message.relevance_score

        if score >= self.config.tau_high:
            return FidelityLevel.FULL
        elif score >= self.config.tau_mid:
            return FidelityLevel.COMPRESSED
        else:
            return FidelityLevel.PLACEHOLDER

    def _create_placeholder(self, message: Message) -> str:
        """
        Create a short placeholder stub for a message

        Max tokens: config.max_stub_tokens (default 12)
        """
        stub = f"[{message.role} turn {message.turn_index}]"

        # Try to add a hint
        tokens_used = self.token_counter.count(stub)
        tokens_left = self.config.max_stub_tokens - tokens_used

        if tokens_left > 3:
            # Add a very short preview
            preview_chars = tokens_left * 4  # Approximate
            preview = message.content[:preview_chars]
            if len(message.content) > preview_chars:
                preview += "..."
            stub += f" {preview}"

        return stub

    def _create_compressed(self, message: Message) -> str:
        """
        Create compressed summary of message

        Target tokens: original_tokens * config.target_compression_ratio
        """
        # Check cache
        if message.compressed_summary is not None:
            return message.compressed_summary

        # Calculate target tokens
        original_tokens = self.token_counter.count(message.content)
        target_tokens = max(10, int(original_tokens * self.config.target_compression_ratio))

        # Compress
        summary = self.compressor.compress(message.content, target_tokens)

        # Add prefix to indicate compression
        summary = f"[{message.role} summarized]: {summary}"

        # Cache
        message.compressed_summary = summary

        return summary

    def _try_add_system_preamble(
        self,
        preamble: str,
        budget_left: int,
        packed: List[Tuple[str, str]],
    ) -> int:
        """
        Try to add system preamble to packed messages.

        Args:
            preamble: System preamble text
            budget_left: Remaining token budget
            packed: List of packed messages (modified in-place)

        Returns:
            Updated budget_left after adding preamble (or unchanged if doesn't fit)
        """
        preamble_tokens = self.token_counter.count(preamble)
        if preamble_tokens <= budget_left:
            packed.append(("system", preamble))
            return budget_left - preamble_tokens
        else:
            logger.warning("System preamble exceeds budget, skipping")
            return budget_left

    def _try_pack_message_at_fidelity(
        self,
        message: Message,
        fidelity: FidelityLevel,
        budget_left: int,
    ) -> Tuple[Optional[Tuple[str, str]], int]:
        """
        Try to pack message at specified fidelity level.

        Args:
            message: Message to pack
            fidelity: Fidelity level to attempt (FULL, COMPRESSED, or PLACEHOLDER)
            budget_left: Remaining token budget

        Returns:
            (packed_message, tokens_used) if successful, (None, 0) if doesn't fit
            packed_message: (role, content) tuple or None
            tokens_used: Number of tokens consumed (0 if message didn't fit)
        """
        # Generate content at requested fidelity
        if fidelity == FidelityLevel.FULL:
            content = message.content
        elif fidelity == FidelityLevel.COMPRESSED:
            content = self._create_compressed(message)
        else:  # PLACEHOLDER
            content = self._create_placeholder(message)

        # Check if it fits
        tokens = self.token_counter.count(content)
        if tokens <= budget_left:
            return (message.role, content), tokens
        else:
            return None, 0

    def _build_packing_stats(
        self,
        messages_with_scores: List[Tuple[Message, float]],
        budget_tokens: int,
        budget_left: int,
        full_count: int,
        compressed_count: int,
        placeholder_count: int,
        dropped_count: int,
    ) -> PackingStats:
        """
        Build PackingStats object from counters.

        Args:
            messages_with_scores: Original list of (Message, score) tuples
            budget_tokens: Original token budget
            budget_left: Remaining token budget after packing
            full_count: Number of messages packed at FULL fidelity
            compressed_count: Number of messages packed at COMPRESSED fidelity
            placeholder_count: Number of messages packed at PLACEHOLDER fidelity
            dropped_count: Number of messages dropped

        Returns:
            PackingStats with all metrics
        """
        total_tokens_used = budget_tokens - budget_left
        compression_ratio = total_tokens_used / budget_tokens if budget_tokens > 0 else 0.0

        return PackingStats(
            total_messages=len(messages_with_scores),
            full_count=full_count,
            compressed_count=compressed_count,
            placeholder_count=placeholder_count,
            dropped_count=dropped_count,
            total_tokens=total_tokens_used,
            budget_tokens=budget_tokens,
            compression_ratio=compression_ratio,
        )

    def _pack_messages(
        self,
        messages_with_scores: List[Tuple[Message, float]],
        budget_tokens: int,
        system_preamble: Optional[str] = None,
    ) -> Tuple[List[Tuple[str, str]], PackingStats]:
        """
        Pack messages chronologically under token budget (refactored v0.4.3).

        Section 3.3 of AFM paper:
        - Process messages in chronological order (oldest to newest)
        - For each message, try to include at intended fidelity
        - If doesn't fit, try lower fidelity
        - If still doesn't fit, drop

        Args:
            messages_with_scores: List of (Message, score) tuples
            budget_tokens: Maximum token budget
            system_preamble: Optional system message to include first

        Returns:
            (packed_messages, stats)
            packed_messages: List of (role, content) tuples
            stats: PackingStats with metrics

        Note:
            Refactored in v0.4.3 to improve maintainability (105 lines → 45 lines).
            Extracted helpers: _try_add_system_preamble, _try_pack_message_at_fidelity,
            _build_packing_stats.
        """
        packed = []
        budget_left = budget_tokens

        # Counters for stats
        full_count = 0
        compressed_count = 0
        placeholder_count = 0
        dropped_count = 0

        # Sort messages by turn_index to ensure chronological order
        messages_with_scores.sort(key=lambda x: x[0].turn_index)

        # Add system preamble first if provided
        if system_preamble:
            budget_left = self._try_add_system_preamble(system_preamble, budget_left, packed)

        # Pack messages chronologically with fidelity fallback
        for message, score in messages_with_scores:
            intended = message.intended_fidelity

            # Try fidelity levels in order: intended → lower → lowest
            packed_msg, tokens = None, 0

            # Try FULL (if intended or fallback)
            if intended == FidelityLevel.FULL:
                packed_msg, tokens = self._try_pack_message_at_fidelity(
                    message, FidelityLevel.FULL, budget_left
                )
                if packed_msg:
                    packed.append(packed_msg)
                    budget_left -= tokens
                    full_count += 1

            # Try COMPRESSED (if intended or fallback from FULL)
            if not packed_msg and intended in [FidelityLevel.COMPRESSED, FidelityLevel.FULL]:
                packed_msg, tokens = self._try_pack_message_at_fidelity(
                    message, FidelityLevel.COMPRESSED, budget_left
                )
                if packed_msg:
                    packed.append(packed_msg)
                    budget_left -= tokens
                    compressed_count += 1

            # Try PLACEHOLDER (if intended or fallback from COMPRESSED/FULL)
            if not packed_msg:
                packed_msg, tokens = self._try_pack_message_at_fidelity(
                    message, FidelityLevel.PLACEHOLDER, budget_left
                )
                if packed_msg:
                    packed.append(packed_msg)
                    budget_left -= tokens
                    placeholder_count += 1

            # If even placeholder doesn't fit, drop
            if not packed_msg:
                dropped_count += 1
                logger.debug(
                    f"Dropped message {message.message_id} (no space even for placeholder)"
                )

        # Build and return stats
        stats = self._build_packing_stats(
            messages_with_scores,
            budget_tokens,
            budget_left,
            full_count,
            compressed_count,
            placeholder_count,
            dropped_count,
        )

        return packed, stats

    def build_context(
        self, current_query: str, budget_tokens: int, system_preamble: Optional[str] = None
    ) -> Tuple[List[Tuple[str, str]], PackingStats]:
        """
        Build context for current query under token budget

        Main entry point for AFM system.

        Args:
            current_query: The current user query
            budget_tokens: Maximum tokens allowed in context
            system_preamble: Optional system message to include first

        Returns:
            (context_messages, stats)
            context_messages: List of (role, content) tuples ready for LLM
            stats: PackingStats with compression metrics
        """
        # Validate budget
        if budget_tokens <= 0:
            raise ValueError("Token budget must be positive")

        logger.info(f"Building context for query (budget: {budget_tokens} tokens)")

        # Embed current query
        query_embedding = self.embedder.encode([current_query])[0]

        # Score all messages
        current_turn = self.turn_counter
        messages_with_scores = []

        for message in self.messages:
            score = self._calculate_relevance_score(message, query_embedding, current_turn)
            message.relevance_score = score

            # Assign intended fidelity
            message.intended_fidelity = self._assign_intended_fidelity(message)

            messages_with_scores.append((message, score))

            logger.debug(
                f"Message {message.message_id}: "
                f"score={score:.3f}, "
                f"importance={message.importance.value}, "
                f"fidelity={message.intended_fidelity.value}"
            )

        # Pack messages under budget
        context, stats = self._pack_messages(messages_with_scores, budget_tokens, system_preamble)

        logger.info(
            f"Context built: {stats.total_messages} messages → "
            f"{stats.full_count} full, {stats.compressed_count} compressed, "
            f"{stats.placeholder_count} placeholder, {stats.dropped_count} dropped | "
            f"Tokens: {stats.total_tokens}/{stats.budget_tokens} "
            f"({stats.compression_ratio:.1%} of budget used)"
        )

        return context, stats

    def clear_history(self):
        """Clear all dialogue history"""
        self.messages.clear()
        self.turn_counter = 0
        logger.info("Dialogue history cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get current dialogue statistics"""
        return {
            "total_messages": len(self.messages),
            "current_turn": self.turn_counter,
            "importance_breakdown": {
                "critical": sum(
                    1 for m in self.messages if m.importance == ImportanceLevel.CRITICAL
                ),
                "relevant": sum(
                    1 for m in self.messages if m.importance == ImportanceLevel.RELEVANT
                ),
                "trivial": sum(1 for m in self.messages if m.importance == ImportanceLevel.TRIVIAL),
            },
        }
