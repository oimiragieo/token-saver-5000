"""
Experience Synthesis for Adversarial Testing

Implements ASG-SI's experience synthesis concept:
"Synthesized experiences stress-test skill interfaces, probe boundary
conditions, and expand coverage for adversarial scenarios."

Features:
- Generate adversarial test documents
- Probe compression boundary conditions
- Stress-test AFM dialogue memory
- Test ACE framework edge cases
- Synthesize edge cases for all MCP tools

Usage:
    from src.experience_synthesis import ExperienceSynthesizer

    synth = ExperienceSynthesizer()

    # Generate boundary test cases
    test_docs = synth.generate_boundary_cases()

    # Stress test compression
    results = synth.stress_test_compression(compressor)
"""

from __future__ import annotations

import logging
import random
import string
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .semantic_compressor import SemanticCompressor
    from .afm import FocusManager
    from .ace_framework import ACEFramework

logger = logging.getLogger(__name__)


class TestCategory(Enum):
    """Categories of test cases"""
    BOUNDARY = "boundary"
    ADVERSARIAL = "adversarial"
    STRESS = "stress"
    EDGE_CASE = "edge_case"
    REGRESSION = "regression"


@dataclass
class SyntheticDocument:
    """A synthetically generated test document"""
    content: str
    category: TestCategory
    name: str
    description: str
    expected_behavior: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        """Rough token count estimate"""
        return len(self.content.split())


@dataclass
class StressTestResult:
    """Result of a stress test"""
    test_name: str
    passed: bool
    duration_ms: float
    iterations: int
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "duration_ms": self.duration_ms,
            "iterations": self.iterations,
            "errors": self.errors,
            "metrics": self.metrics,
        }


@dataclass
class BoundaryTestSuite:
    """Suite of boundary test cases"""
    documents: List[SyntheticDocument]
    dialogues: List[List[Dict[str, str]]]
    ace_contexts: List[Dict[str, Any]]


class ExperienceSynthesizer:
    """
    Generates synthetic test experiences for adversarial testing.

    Implements ASG-SI's concept of experience synthesis to:
    - Stress-test compression interfaces
    - Probe boundary conditions
    - Expand test coverage
    - Identify edge cases
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize synthesizer.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)

    # =========================================================================
    # Document Generation
    # =========================================================================

    def generate_boundary_cases(self) -> List[SyntheticDocument]:
        """
        Generate boundary test documents.

        Returns:
            List of boundary test documents
        """
        return [
            self._empty_document(),
            self._single_token(),
            self._single_sentence(),
            self._max_token_document(50000),
            self._highly_repetitive(),
            self._no_semantic_structure(),
            self._deeply_nested_structure(),
            self._mixed_language(),
            self._adversarial_unicode(),
            self._code_only(),
            self._whitespace_heavy(),
            self._single_paragraph(),
        ]

    def _empty_document(self) -> SyntheticDocument:
        """Empty document edge case"""
        return SyntheticDocument(
            content="",
            category=TestCategory.BOUNDARY,
            name="empty_document",
            description="Completely empty document",
            expected_behavior="Should reject with precondition failure",
        )

    def _single_token(self) -> SyntheticDocument:
        """Single token document"""
        return SyntheticDocument(
            content="Hello",
            category=TestCategory.BOUNDARY,
            name="single_token",
            description="Document with only one token",
            expected_behavior="Should handle gracefully with minimal compression",
        )

    def _single_sentence(self) -> SyntheticDocument:
        """Single sentence document"""
        return SyntheticDocument(
            content="The quick brown fox jumps over the lazy dog.",
            category=TestCategory.BOUNDARY,
            name="single_sentence",
            description="Document with one pangram sentence",
            expected_behavior="Should compress but may have overhead",
        )

    def _max_token_document(self, target_tokens: int = 50000) -> SyntheticDocument:
        """Large document near token limit"""
        # Generate paragraphs of lorem ipsum-like text
        paragraphs = []
        words_per_paragraph = 100
        current_tokens = 0

        while current_tokens < target_tokens:
            para = self._generate_paragraph(words_per_paragraph)
            paragraphs.append(para)
            current_tokens += words_per_paragraph

        content = "\n\n".join(paragraphs)

        return SyntheticDocument(
            content=content,
            category=TestCategory.STRESS,
            name="max_token_document",
            description=f"Large document with ~{target_tokens} tokens",
            expected_behavior="Should compress with high ratio",
            metadata={"target_tokens": target_tokens}
        )

    def _highly_repetitive(self) -> SyntheticDocument:
        """Document with high repetition"""
        sentence = "This is a repeated sentence that appears many times. "
        content = sentence * 100

        return SyntheticDocument(
            content=content,
            category=TestCategory.ADVERSARIAL,
            name="highly_repetitive",
            description="Document with 100 identical sentences",
            expected_behavior="Should achieve very high compression via deduplication",
        )

    def _no_semantic_structure(self) -> SyntheticDocument:
        """Document with no semantic structure"""
        # Random words without coherent meaning
        words = [
            self._random_word() for _ in range(200)
        ]
        content = " ".join(words)

        return SyntheticDocument(
            content=content,
            category=TestCategory.ADVERSARIAL,
            name="no_semantic_structure",
            description="Random words without semantic coherence",
            expected_behavior="Should still compress but with lower quality score",
        )

    def _deeply_nested_structure(self) -> SyntheticDocument:
        """Document with deeply nested sections"""
        lines = []
        for i in range(1, 6):
            lines.append(f"{'#' * i} Section Level {i}")
            lines.append(f"Content at level {i}. " * 10)

        for i in range(5, 0, -1):
            lines.append(f"{'#' * i} Closing Section Level {i}")
            lines.append(f"More content at level {i}. " * 5)

        return SyntheticDocument(
            content="\n".join(lines),
            category=TestCategory.EDGE_CASE,
            name="deeply_nested_structure",
            description="Document with 5 levels of nested sections",
            expected_behavior="Should preserve hierarchical structure in skeleton",
        )

    def _mixed_language(self) -> SyntheticDocument:
        """Document with mixed languages"""
        content = """
# Introduction (English)
This is the English introduction to our document.

# Einleitung (German)
Dies ist die deutsche Einleitung zu unserem Dokument.

# 简介 (Chinese)
这是我们文档的中文介绍。

# مقدمة (Arabic)
هذه هي المقدمة العربية لوثيقتنا.

# Conclusion
A multilingual document presents unique challenges for compression.
"""
        return SyntheticDocument(
            content=content,
            category=TestCategory.EDGE_CASE,
            name="mixed_language",
            description="Document with English, German, Chinese, and Arabic",
            expected_behavior="Should handle Unicode correctly and preserve meaning",
        )

    def _adversarial_unicode(self) -> SyntheticDocument:
        """Document with adversarial Unicode characters"""
        content = """
Normal text followed by special characters:
- Zero-width space: ​word​joined​
- Right-to-left mark: ‎reversed text‎
- Combining characters: é̃ñ̃ (with multiple combining marks)
- Emoji: 🔥💯🚀 (fire, 100, rocket)
- Mathematical symbols: ∀x∈ℝ, ∃y∈ℂ
- Control characters removed for safety
- Fullwidth: ＦＵＬＬＷＩＤＴＨ
"""
        return SyntheticDocument(
            content=content,
            category=TestCategory.ADVERSARIAL,
            name="adversarial_unicode",
            description="Document with special Unicode characters",
            expected_behavior="Should handle all Unicode without errors",
        )

    def _code_only(self) -> SyntheticDocument:
        """Document containing only code"""
        content = '''
```python
def fibonacci(n):
    """Calculate nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    """Simple calculator class."""

    def __init__(self):
        self.result = 0

    def add(self, x, y):
        """Add two numbers."""
        return x + y

    def multiply(self, x, y):
        """Multiply two numbers."""
        return x * y

# Example usage
calc = Calculator()
print(calc.add(5, 3))
print(fibonacci(10))
```
'''
        return SyntheticDocument(
            content=content,
            category=TestCategory.EDGE_CASE,
            name="code_only",
            description="Document containing only Python code",
            expected_behavior="Should use code compressor with AST awareness",
        )

    def _whitespace_heavy(self) -> SyntheticDocument:
        """Document with excessive whitespace"""
        content = "Word   " + "   ".join(["word"] * 50) + "\n\n\n\n" * 10
        content += "    Indented    text    with    spaces    \n" * 20

        return SyntheticDocument(
            content=content,
            category=TestCategory.ADVERSARIAL,
            name="whitespace_heavy",
            description="Document with excessive whitespace",
            expected_behavior="Should normalize whitespace in compression",
        )

    def _single_paragraph(self) -> SyntheticDocument:
        """Document with one very long paragraph"""
        sentences = [
            f"Sentence number {i} contains some information about topic {i % 5}."
            for i in range(100)
        ]
        content = " ".join(sentences)

        return SyntheticDocument(
            content=content,
            category=TestCategory.EDGE_CASE,
            name="single_paragraph",
            description="100 sentences in one paragraph without breaks",
            expected_behavior="Should identify sentence boundaries for chunking",
        )

    def _generate_paragraph(self, word_count: int) -> str:
        """Generate a paragraph with given word count"""
        words = [
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "semantic", "compression", "reduces", "tokens", "while", "preserving",
            "meaning", "and", "structure", "in", "documents", "for", "efficient",
            "AI", "interactions", "using", "graph", "based", "analysis"
        ]
        paragraph_words = [self.rng.choice(words) for _ in range(word_count)]
        return " ".join(paragraph_words) + "."

    def _random_word(self) -> str:
        """Generate a random word"""
        length = self.rng.randint(3, 10)
        return "".join(self.rng.choices(string.ascii_lowercase, k=length))

    # =========================================================================
    # Dialogue Synthesis
    # =========================================================================

    def generate_dialogue_cases(self) -> List[List[Dict[str, str]]]:
        """
        Generate test dialogues for AFM stress testing.

        Returns:
            List of dialogue turn sequences
        """
        return [
            self._empty_dialogue(),
            self._single_turn(),
            self._long_dialogue(100),
            self._safety_critical_dialogue(),
            self._context_switching_dialogue(),
            self._repetitive_dialogue(),
            self._adversarial_dialogue(),
        ]

    def _empty_dialogue(self) -> List[Dict[str, str]]:
        """Empty dialogue"""
        return []

    def _single_turn(self) -> List[Dict[str, str]]:
        """Single turn dialogue"""
        return [{"role": "user", "content": "Hello!"}]

    def _long_dialogue(self, turns: int) -> List[Dict[str, str]]:
        """Long multi-turn dialogue"""
        dialogue = []
        for i in range(turns):
            role = "user" if i % 2 == 0 else "assistant"
            content = f"Turn {i}: This is {'a question' if role == 'user' else 'a response'}."
            dialogue.append({"role": role, "content": content})
        return dialogue

    def _safety_critical_dialogue(self) -> List[Dict[str, str]]:
        """Dialogue with safety-critical information"""
        return [
            {"role": "user", "content": "I have a severe peanut allergy."},
            {"role": "assistant", "content": "I've noted your peanut allergy."},
            {"role": "user", "content": "Can you suggest a restaurant?"},
            {"role": "assistant", "content": "I'll find peanut-free options."},
            # Many filler turns
            *[{"role": "user" if i % 2 == 0 else "assistant",
               "content": f"Filler turn {i}"} for i in range(50)],
            # Critical test - allergy should still be remembered
            {"role": "user", "content": "What food should I avoid?"},
        ]

    def _context_switching_dialogue(self) -> List[Dict[str, str]]:
        """Dialogue with frequent context switches"""
        topics = ["weather", "coding", "cooking", "travel", "music"]
        dialogue = []
        for i in range(25):
            topic = topics[i % len(topics)]
            dialogue.append({
                "role": "user",
                "content": f"Let's talk about {topic}. What do you think?"
            })
            dialogue.append({
                "role": "assistant",
                "content": f"Sure, {topic} is interesting. Here's my view..."
            })
        return dialogue

    def _repetitive_dialogue(self) -> List[Dict[str, str]]:
        """Dialogue with repetitive content"""
        return [
            {"role": "user", "content": "Tell me about cats."},
            {"role": "assistant", "content": "Cats are wonderful pets."},
        ] * 20

    def _adversarial_dialogue(self) -> List[Dict[str, str]]:
        """Dialogue designed to challenge memory"""
        return [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Nice to meet you, Alice."},
            {"role": "user", "content": "Actually, my name is Bob."},
            {"role": "assistant", "content": "I apologize, Bob."},
            {"role": "user", "content": "No wait, it's actually Charlie."},
            {"role": "assistant", "content": "Got it, Charlie."},
            {"role": "user", "content": "What's my name?"},
        ]

    # =========================================================================
    # ACE Context Synthesis
    # =========================================================================

    def generate_ace_cases(self) -> List[Dict[str, Any]]:
        """
        Generate test cases for ACE framework.

        Returns:
            List of ACE context test cases
        """
        return [
            self._conflicting_principles(),
            self._duplicate_bullets(),
            self._low_confidence_bullets(),
            self._empty_context(),
            self._max_bullets(),
        ]

    def _conflicting_principles(self) -> Dict[str, Any]:
        """ACE context with conflicting principles"""
        return {
            "name": "conflicting_principles",
            "bullets": [
                {"text": "Be concise and brief", "type": "principle"},
                {"text": "Be detailed and thorough", "type": "principle"},
                {"text": "Avoid redundancy", "type": "constraint"},
                {"text": "Include examples for clarity", "type": "strategy"},
            ],
            "expected": "Should identify and handle conflicts",
        }

    def _duplicate_bullets(self) -> Dict[str, Any]:
        """ACE context with semantic duplicates"""
        return {
            "name": "duplicate_bullets",
            "bullets": [
                {"text": "Be concise", "type": "principle"},
                {"text": "Keep it short", "type": "principle"},
                {"text": "Brevity is key", "type": "principle"},
                {"text": "Don't be verbose", "type": "constraint"},
            ],
            "expected": "Should deduplicate to single principle",
        }

    def _low_confidence_bullets(self) -> Dict[str, Any]:
        """ACE context with low confidence bullets"""
        return {
            "name": "low_confidence_bullets",
            "bullets": [
                {"text": "Maybe use emojis", "type": "preference", "confidence": 0.1},
                {"text": "Perhaps format as lists", "type": "preference", "confidence": 0.2},
            ],
            "expected": "Should handle low confidence gracefully",
        }

    def _empty_context(self) -> Dict[str, Any]:
        """Empty ACE context"""
        return {
            "name": "empty_context",
            "bullets": [],
            "expected": "Should generate initial bullets from scratch",
        }

    def _max_bullets(self) -> Dict[str, Any]:
        """ACE context at maximum bullet limit"""
        bullets = [
            {"text": f"Principle number {i}", "type": "principle"}
            for i in range(100)
        ]
        return {
            "name": "max_bullets",
            "bullets": bullets,
            "expected": "Should prune or refuse additional bullets",
        }

    # =========================================================================
    # Stress Testing
    # =========================================================================

    def stress_test_compression(
        self,
        compressor: "SemanticCompressor",
        iterations: int = 100,
    ) -> StressTestResult:
        """
        Stress test compression with synthetic documents.

        Args:
            compressor: Compressor to test
            iterations: Number of iterations

        Returns:
            StressTestResult with metrics
        """
        start = time.time()
        errors = []
        compression_ratios = []
        processing_times = []

        for i in range(iterations):
            # Generate random document
            doc = self._generate_paragraph(self.rng.randint(100, 1000))

            try:
                iter_start = time.time()
                result = compressor.ingest_file(doc, f"stress_doc_{i}")
                iter_time = (time.time() - iter_start) * 1000

                # Get skeleton for compression ratio
                skeleton = compressor.read_skeleton(f"stress_doc_{i}")
                ratio = skeleton.compression_ratio

                compression_ratios.append(ratio)
                processing_times.append(iter_time)

            except Exception as e:
                errors.append(f"Iteration {i}: {str(e)}")

        duration = (time.time() - start) * 1000

        return StressTestResult(
            test_name="compression_stress_test",
            passed=len(errors) == 0,
            duration_ms=duration,
            iterations=iterations,
            errors=errors,
            metrics={
                "avg_compression_ratio": np.mean(compression_ratios) if compression_ratios else 0,
                "min_compression_ratio": min(compression_ratios) if compression_ratios else 0,
                "max_compression_ratio": max(compression_ratios) if compression_ratios else 0,
                "avg_processing_time_ms": np.mean(processing_times) if processing_times else 0,
                "error_rate": len(errors) / iterations,
            }
        )

    def stress_test_afm(
        self,
        manager: "FocusManager",
        turns: int = 1000,
    ) -> StressTestResult:
        """
        Stress test AFM with many dialogue turns.

        Args:
            manager: FocusManager to test
            turns: Number of turns to add

        Returns:
            StressTestResult with metrics
        """
        start = time.time()
        errors = []
        build_times = []

        try:
            # Add safety-critical message
            manager.add_message("user", "I have a severe peanut allergy.")

            # Add many turns
            for i in range(turns):
                role = "user" if i % 2 == 0 else "assistant"
                content = f"Turn {i}: " + self._generate_paragraph(20)
                manager.add_message(role, content)

            # Test context building at intervals
            for budget in [100, 500, 1000, 2000]:
                build_start = time.time()
                context, stats = manager.build_context("What should I eat?", budget_tokens=budget)
                build_time = (time.time() - build_start) * 1000
                build_times.append(build_time)

                # Verify allergy is retained
                if "allergy" not in context.lower() and "peanut" not in context.lower():
                    errors.append(f"Safety info lost at budget {budget}")

        except Exception as e:
            errors.append(f"Stress test failed: {str(e)}")

        duration = (time.time() - start) * 1000

        return StressTestResult(
            test_name="afm_stress_test",
            passed=len(errors) == 0,
            duration_ms=duration,
            iterations=turns,
            errors=errors,
            metrics={
                "avg_build_time_ms": np.mean(build_times) if build_times else 0,
                "max_build_time_ms": max(build_times) if build_times else 0,
                "turns_added": turns,
            }
        )

    def generate_full_test_suite(self) -> BoundaryTestSuite:
        """
        Generate complete boundary test suite.

        Returns:
            BoundaryTestSuite with all test cases
        """
        return BoundaryTestSuite(
            documents=self.generate_boundary_cases(),
            dialogues=self.generate_dialogue_cases(),
            ace_contexts=self.generate_ace_cases(),
        )

    def run_boundary_tests(
        self,
        compressor: "SemanticCompressor",
    ) -> List[Tuple[str, bool, Optional[str]]]:
        """
        Run all boundary tests against compressor.

        Args:
            compressor: Compressor to test

        Returns:
            List of (test_name, passed, error_message)
        """
        results = []
        documents = self.generate_boundary_cases()

        for doc in documents:
            try:
                # Skip empty document (expected to fail)
                if doc.name == "empty_document":
                    try:
                        compressor.ingest_file(doc.content, doc.name)
                        results.append((doc.name, False, "Should have rejected empty doc"))
                    except (ValueError, Exception):
                        results.append((doc.name, True, None))
                    continue

                # Test ingestion
                compressor.ingest_file(doc.content, doc.name)
                skeleton = compressor.read_skeleton(doc.name)

                # Basic validation
                if skeleton.skeleton_tokens > 0:
                    results.append((doc.name, True, None))
                else:
                    results.append((doc.name, False, "Zero skeleton tokens"))

            except Exception as e:
                results.append((doc.name, False, str(e)))

        return results
