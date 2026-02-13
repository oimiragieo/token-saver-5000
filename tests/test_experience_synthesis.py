"""
Tests for Experience Synthesis Module

Tests for:
- SyntheticDocument generation
- Boundary case generation
- Dialogue synthesis
- ACE context synthesis
- Stress testing infrastructure
"""

from src.experience_synthesis import (
    ExperienceSynthesizer,
    SyntheticDocument,
    StressTestResult,
    BoundaryTestSuite,
    TestCategory,
)


class TestSyntheticDocument:
    """Tests for SyntheticDocument dataclass"""

    def test_basic_creation(self):
        """Test creating a synthetic document"""
        doc = SyntheticDocument(
            content="This is test content",
            category=TestCategory.BOUNDARY,
            name="test_doc",
            description="A test document",
            expected_behavior="Should pass",
        )

        assert doc.content == "This is test content"
        assert doc.category == TestCategory.BOUNDARY
        assert doc.name == "test_doc"

    def test_token_estimate(self):
        """Test token count estimation"""
        doc = SyntheticDocument(
            content="One two three four five",
            category=TestCategory.BOUNDARY,
            name="test",
            description="test",
            expected_behavior="test",
        )

        assert doc.token_estimate == 5

    def test_empty_content(self):
        """Test empty content token estimate"""
        doc = SyntheticDocument(
            content="",
            category=TestCategory.BOUNDARY,
            name="empty",
            description="empty",
            expected_behavior="test",
        )

        assert doc.token_estimate == 0

    def test_metadata(self):
        """Test metadata field"""
        doc = SyntheticDocument(
            content="test",
            category=TestCategory.ADVERSARIAL,
            name="test",
            description="test",
            expected_behavior="test",
            metadata={"key": "value"},
        )

        assert doc.metadata["key"] == "value"


class TestStressTestResult:
    """Tests for StressTestResult dataclass"""

    def test_basic_creation(self):
        """Test creating a stress test result"""
        result = StressTestResult(
            test_name="compression_test",
            passed=True,
            duration_ms=1500.5,
            iterations=100,
        )

        assert result.test_name == "compression_test"
        assert result.passed
        assert result.duration_ms == 1500.5
        assert result.iterations == 100

    def test_with_errors(self):
        """Test result with errors"""
        result = StressTestResult(
            test_name="failed_test",
            passed=False,
            duration_ms=500.0,
            iterations=50,
            errors=["Error 1", "Error 2"],
        )

        assert not result.passed
        assert len(result.errors) == 2

    def test_with_metrics(self):
        """Test result with metrics"""
        result = StressTestResult(
            test_name="metrics_test",
            passed=True,
            duration_ms=1000.0,
            iterations=100,
            metrics={
                "avg_compression_ratio": 5.5,
                "error_rate": 0.01,
            },
        )

        assert result.metrics["avg_compression_ratio"] == 5.5

    def test_to_dict(self):
        """Test serialization"""
        result = StressTestResult(
            test_name="test",
            passed=True,
            duration_ms=100.0,
            iterations=10,
            errors=["error1"],
            metrics={"ratio": 5.0},
        )

        d = result.to_dict()
        assert d["test_name"] == "test"
        assert d["passed"] is True
        assert d["errors"] == ["error1"]
        assert d["metrics"]["ratio"] == 5.0


class TestExperienceSynthesizerDocuments:
    """Tests for document generation"""

    def test_generate_boundary_cases(self):
        """Test boundary case generation"""
        synth = ExperienceSynthesizer(seed=42)
        docs = synth.generate_boundary_cases()

        assert len(docs) >= 10  # At least 10 boundary cases
        assert all(isinstance(d, SyntheticDocument) for d in docs)

    def test_empty_document(self):
        """Test empty document is included"""
        synth = ExperienceSynthesizer()
        docs = synth.generate_boundary_cases()

        empty_docs = [d for d in docs if d.name == "empty_document"]
        assert len(empty_docs) == 1
        assert empty_docs[0].content == ""

    def test_single_token(self):
        """Test single token document"""
        synth = ExperienceSynthesizer()
        docs = synth.generate_boundary_cases()

        single_token = [d for d in docs if d.name == "single_token"]
        assert len(single_token) == 1
        assert single_token[0].token_estimate <= 2

    def test_max_token_document(self):
        """Test large document generation"""
        synth = ExperienceSynthesizer(seed=42)
        docs = synth.generate_boundary_cases()

        max_docs = [d for d in docs if d.name == "max_token_document"]
        assert len(max_docs) == 1
        # Should have many tokens
        assert max_docs[0].token_estimate > 1000

    def test_highly_repetitive(self):
        """Test repetitive document"""
        synth = ExperienceSynthesizer()
        docs = synth.generate_boundary_cases()

        repetitive = [d for d in docs if d.name == "highly_repetitive"]
        assert len(repetitive) == 1
        assert repetitive[0].category == TestCategory.ADVERSARIAL

    def test_adversarial_unicode(self):
        """Test adversarial Unicode document"""
        synth = ExperienceSynthesizer()
        docs = synth.generate_boundary_cases()

        unicode_doc = [d for d in docs if d.name == "adversarial_unicode"]
        assert len(unicode_doc) == 1
        # Should contain non-ASCII characters
        assert any(ord(c) > 127 for c in unicode_doc[0].content)

    def test_code_only(self):
        """Test code-only document"""
        synth = ExperienceSynthesizer()
        docs = synth.generate_boundary_cases()

        code_docs = [d for d in docs if d.name == "code_only"]
        assert len(code_docs) == 1
        assert "def " in code_docs[0].content or "class " in code_docs[0].content

    def test_deterministic_with_seed(self):
        """Test that seed produces deterministic output"""
        synth1 = ExperienceSynthesizer(seed=42)
        synth2 = ExperienceSynthesizer(seed=42)

        docs1 = synth1.generate_boundary_cases()
        docs2 = synth2.generate_boundary_cases()

        # Max token document should be the same with same seed
        max1 = [d for d in docs1 if d.name == "max_token_document"][0]
        max2 = [d for d in docs2 if d.name == "max_token_document"][0]
        assert max1.content == max2.content


class TestExperienceSynthesizerDialogues:
    """Tests for dialogue generation"""

    def test_generate_dialogue_cases(self):
        """Test dialogue case generation"""
        synth = ExperienceSynthesizer()
        dialogues = synth.generate_dialogue_cases()

        assert len(dialogues) >= 5
        assert all(isinstance(d, list) for d in dialogues)

    def test_empty_dialogue(self):
        """Test empty dialogue is included"""
        synth = ExperienceSynthesizer()
        dialogues = synth.generate_dialogue_cases()

        empty = [d for d in dialogues if len(d) == 0]
        assert len(empty) == 1

    def test_single_turn(self):
        """Test single turn dialogue"""
        synth = ExperienceSynthesizer()
        dialogues = synth.generate_dialogue_cases()

        single = [d for d in dialogues if len(d) == 1]
        assert len(single) == 1
        assert single[0][0]["role"] == "user"

    def test_long_dialogue(self):
        """Test long dialogue generation"""
        synth = ExperienceSynthesizer()
        dialogues = synth.generate_dialogue_cases()

        long_dialogues = [d for d in dialogues if len(d) >= 50]
        assert len(long_dialogues) >= 1

    def test_safety_critical_dialogue(self):
        """Test safety-critical dialogue includes allergy info"""
        synth = ExperienceSynthesizer()
        dialogues = synth.generate_dialogue_cases()

        # Find dialogue mentioning allergy
        safety_dialogues = [
            d for d in dialogues if any("allergy" in turn.get("content", "").lower() for turn in d)
        ]
        assert len(safety_dialogues) >= 1

    def test_dialogue_structure(self):
        """Test dialogue structure (role, content)"""
        synth = ExperienceSynthesizer()
        dialogues = synth.generate_dialogue_cases()

        for dialogue in dialogues:
            for turn in dialogue:
                assert "role" in turn
                assert "content" in turn
                assert turn["role"] in ["user", "assistant"]


class TestExperienceSynthesizerACE:
    """Tests for ACE context generation"""

    def test_generate_ace_cases(self):
        """Test ACE case generation"""
        synth = ExperienceSynthesizer()
        cases = synth.generate_ace_cases()

        assert len(cases) >= 5
        assert all(isinstance(c, dict) for c in cases)

    def test_conflicting_principles(self):
        """Test conflicting principles case"""
        synth = ExperienceSynthesizer()
        cases = synth.generate_ace_cases()

        conflicting = [c for c in cases if c.get("name") == "conflicting_principles"]
        assert len(conflicting) == 1
        assert len(conflicting[0]["bullets"]) >= 2

    def test_duplicate_bullets(self):
        """Test duplicate bullets case"""
        synth = ExperienceSynthesizer()
        cases = synth.generate_ace_cases()

        duplicates = [c for c in cases if c.get("name") == "duplicate_bullets"]
        assert len(duplicates) == 1

    def test_empty_context(self):
        """Test empty context case"""
        synth = ExperienceSynthesizer()
        cases = synth.generate_ace_cases()

        empty = [c for c in cases if c.get("name") == "empty_context"]
        assert len(empty) == 1
        assert len(empty[0]["bullets"]) == 0

    def test_max_bullets(self):
        """Test max bullets case"""
        synth = ExperienceSynthesizer()
        cases = synth.generate_ace_cases()

        max_case = [c for c in cases if c.get("name") == "max_bullets"]
        assert len(max_case) == 1
        assert len(max_case[0]["bullets"]) >= 50

    def test_case_has_expected_field(self):
        """Test each case has expected behavior field"""
        synth = ExperienceSynthesizer()
        cases = synth.generate_ace_cases()

        for case in cases:
            assert "expected" in case


class TestBoundaryTestSuite:
    """Tests for BoundaryTestSuite"""

    def test_generate_full_suite(self):
        """Test full suite generation"""
        synth = ExperienceSynthesizer()
        suite = synth.generate_full_test_suite()

        assert isinstance(suite, BoundaryTestSuite)
        assert len(suite.documents) > 0
        assert len(suite.dialogues) > 0
        assert len(suite.ace_contexts) > 0

    def test_suite_coverage(self):
        """Test suite covers all categories"""
        synth = ExperienceSynthesizer()
        suite = synth.generate_full_test_suite()

        categories = {doc.category for doc in suite.documents}
        assert TestCategory.BOUNDARY in categories
        assert TestCategory.ADVERSARIAL in categories


class TestRandomGeneration:
    """Tests for random generation utilities"""

    def test_generate_paragraph(self):
        """Test paragraph generation"""
        synth = ExperienceSynthesizer(seed=42)

        # Access private method through instance
        para = synth._generate_paragraph(50)

        assert len(para.split()) == 50
        assert para.endswith(".")

    def test_random_word(self):
        """Test random word generation"""
        synth = ExperienceSynthesizer(seed=42)

        word = synth._random_word()

        assert len(word) >= 3
        assert len(word) <= 10
        assert word.isalpha()
        assert word.islower()


class TestTestCategories:
    """Tests for TestCategory enum"""

    def test_all_categories_exist(self):
        """Test all expected categories exist"""
        assert hasattr(TestCategory, "BOUNDARY")
        assert hasattr(TestCategory, "ADVERSARIAL")
        assert hasattr(TestCategory, "STRESS")
        assert hasattr(TestCategory, "EDGE_CASE")
        assert hasattr(TestCategory, "REGRESSION")

    def test_category_values(self):
        """Test category values"""
        assert TestCategory.BOUNDARY.value == "boundary"
        assert TestCategory.ADVERSARIAL.value == "adversarial"


class TestStressTestInfrastructure:
    """Tests for stress testing infrastructure (without actual compressor)"""

    def test_stress_test_result_defaults(self):
        """Test StressTestResult defaults"""
        result = StressTestResult(
            test_name="test",
            passed=True,
            duration_ms=100.0,
            iterations=10,
        )

        assert result.errors == []
        assert result.metrics == {}

    def test_stress_test_result_serialization(self):
        """Test full serialization"""
        result = StressTestResult(
            test_name="compression_stress_test",
            passed=False,
            duration_ms=5000.0,
            iterations=100,
            errors=["Timeout on iteration 50"],
            metrics={
                "avg_compression_ratio": 7.5,
                "min_compression_ratio": 2.0,
                "max_compression_ratio": 15.0,
                "error_rate": 0.01,
            },
        )

        d = result.to_dict()

        assert d["test_name"] == "compression_stress_test"
        assert d["passed"] is False
        assert d["duration_ms"] == 5000.0
        assert d["iterations"] == 100
        assert len(d["errors"]) == 1
        assert d["metrics"]["avg_compression_ratio"] == 7.5


class TestMixedLanguageDocument:
    """Tests for mixed language document"""

    def test_contains_multiple_languages(self):
        """Test mixed language doc has multiple scripts"""
        synth = ExperienceSynthesizer()
        docs = synth.generate_boundary_cases()

        mixed = [d for d in docs if d.name == "mixed_language"]
        assert len(mixed) == 1

        content = mixed[0].content
        # Should contain CJK characters
        has_cjk = any("\u4e00" <= c <= "\u9fff" for c in content)
        # Should contain Arabic characters
        has_arabic = any("\u0600" <= c <= "\u06ff" for c in content)

        assert has_cjk or has_arabic  # At least one non-Latin script


class TestDeepNestedStructure:
    """Tests for deeply nested document"""

    def test_has_multiple_heading_levels(self):
        """Test deeply nested has multiple heading levels"""
        synth = ExperienceSynthesizer()
        docs = synth.generate_boundary_cases()

        nested = [d for d in docs if d.name == "deeply_nested_structure"]
        assert len(nested) == 1

        content = nested[0].content
        # Should have multiple heading levels
        assert "# " in content
        assert "## " in content
        assert "### " in content
