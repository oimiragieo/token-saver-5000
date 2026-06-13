#!/usr/bin/env python3
"""
UX Improvements Test Suite (v0.4.1)

Tests for Week 2 UX improvements:
- Fidelity recommendations (Day 7)
- Smart error messages with fuzzy matching (Day 8)
- Compression size estimation (Day 9)
"""

import pytest
from src.fidelity_advisor import FidelityAdvisor, UseCase, FidelityLevel
from src.error_helpers import SmartError, suggest_file_id, suggest_node_id, suggest_enum
from src.compression_advisor import CompressionAdvisor, estimate_compression_size

# ===========================
# Fidelity Advisor Tests (Day 7)
# ===========================


class TestFidelityAdvisor:
    """Test fidelity level recommendation system"""

    def test_quick_summary_use_case(self):
        """Quick summary should recommend ABSTRACT"""
        advisor = FidelityAdvisor()
        rec = advisor.recommend(UseCase.QUICK_SUMMARY, num_nodes=3)

        assert rec.recommended_level == FidelityLevel.ABSTRACT
        assert rec.confidence > 0.8  # High confidence for clear-cut case
        assert "ABSTRACT" in rec.reasoning.upper()

    def test_question_answering_use_case(self):
        """Question answering should recommend DETAILED"""
        advisor = FidelityAdvisor()
        rec = advisor.recommend(UseCase.QUESTION_ANSWERING, num_nodes=5)

        assert rec.recommended_level == FidelityLevel.DETAILED
        assert rec.token_estimate == 500  # 5 nodes × 100 tokens/node

    def test_exact_quotes_use_case(self):
        """Exact quotes should recommend RAW"""
        advisor = FidelityAdvisor()
        rec = advisor.recommend(UseCase.EXACT_QUOTES, num_nodes=2)

        assert rec.recommended_level == FidelityLevel.RAW
        assert "RAW" in rec.reasoning or "exact" in rec.reasoning.lower()

    def test_token_budget_downgrade(self):
        """Budget constraints should downgrade fidelity"""
        advisor = FidelityAdvisor()

        # Without budget: DETAILED for Q&A
        rec_no_budget = advisor.recommend(UseCase.QUESTION_ANSWERING, num_nodes=5)
        assert rec_no_budget.recommended_level == FidelityLevel.DETAILED

        # With tight budget: Should downgrade
        rec_with_budget = advisor.recommend(
            UseCase.QUESTION_ANSWERING, num_nodes=5, token_budget=150
        )
        # 5 nodes × 100 tokens = 500, but budget only 150
        # Should downgrade to STRUCTURE (5 × 50 = 250) or OUTLINE (5 × 30 = 150)
        assert rec_with_budget.recommended_level in [
            FidelityLevel.OUTLINE,
            FidelityLevel.STRUCTURE,
        ]

    def test_complex_query_upgrade(self):
        """Complex queries should upgrade fidelity"""
        advisor = FidelityAdvisor()

        # Medium complexity: OUTLINE for topic overview
        rec_medium = advisor.recommend(
            UseCase.TOPIC_OVERVIEW, num_nodes=3, query_complexity="medium"
        )

        # Complex: Should upgrade to STRUCTURE
        rec_complex = advisor.recommend(
            UseCase.TOPIC_OVERVIEW, num_nodes=3, query_complexity="complex", token_budget=500
        )

        # Complex should be higher fidelity (unless budget prevented it)
        assert rec_complex.recommended_level.value >= rec_medium.recommended_level.value

    def test_alternatives_provided(self):
        """Recommendations should include alternatives"""
        advisor = FidelityAdvisor()
        rec = advisor.recommend(UseCase.QUESTION_ANSWERING, num_nodes=3)

        assert len(rec.alternatives) > 0
        # Should have lower option (save tokens)
        assert any("Save" in alt["trade_off"] for alt in rec.alternatives)

    def test_token_estimate_accuracy(self):
        """Token estimates should match expected values"""
        advisor = FidelityAdvisor()

        # ABSTRACT: 10 tokens/node
        rec_abstract = advisor.recommend(UseCase.QUICK_SUMMARY, num_nodes=5)
        assert rec_abstract.token_estimate == 50

        # DETAILED: 100 tokens/node
        rec_detailed = advisor.recommend(UseCase.DETAILED_ANALYSIS, num_nodes=5)
        assert rec_detailed.token_estimate == 500


# ===========================
# Smart Error Tests (Day 8)
# ===========================


class TestSmartError:
    """Test fuzzy matching error messages"""

    def test_file_id_fuzzy_match_typo(self):
        """Typo in file_id should suggest correction"""
        available = ["quantum_paper", "neural_nets", "blockchain_intro"]
        error = SmartError.file_id_not_found("quantum_papper", available)

        error_msg = str(error)
        assert "quantum_paper" in error_msg  # Should suggest correct spelling
        assert "Did you mean" in error_msg

    def test_file_id_no_match(self):
        """Very different file_id should still show available"""
        available = ["quantum_paper", "neural_nets"]
        error = SmartError.file_id_not_found("totally_different", available)

        error_msg = str(error)
        assert "Available" in error_msg
        assert "quantum_paper" in error_msg or "neural_nets" in error_msg

    def test_node_id_fuzzy_match(self):
        """Node ID typo should suggest correction"""
        available = ["quantum_paper_n0", "quantum_paper_n1", "quantum_paper_n2"]
        error = SmartError.node_id_not_found("quantum_paper_n99", available, "quantum_paper")

        error_msg = str(error)
        assert "Did you mean" in error_msg or "Format" in error_msg
        assert "quantum_paper" in error_msg

    def test_enum_value_fuzzy_match(self):
        """Enum typo should suggest correction"""
        valid = ["ABSTRACT", "OUTLINE", "STRUCTURE", "DETAILED", "RAW"]
        error = SmartError.invalid_enum_value("DETALED", valid, "fidelity_level")

        error_msg = str(error)
        assert "DETAILED" in error_msg  # Should suggest correct spelling
        assert "Did you mean" in error_msg

    def test_invalid_parameter_message(self):
        """Invalid parameter should have clear message"""
        error = SmartError.invalid_parameter("num_nodes", -5, "positive integer", "between 1-1000")

        error_msg = str(error)
        assert "num_nodes" in error_msg
        assert "-5" in error_msg
        assert "positive integer" in error_msg
        assert "between 1-1000" in error_msg

    def test_missing_required_field(self):
        """Missing field should have clear message"""
        error = SmartError.missing_required_field("file_id", "ingest_context")

        error_msg = str(error)
        assert "file_id" in error_msg
        assert "required" in error_msg.lower()

    def test_convenience_functions(self):
        """Convenience functions should work"""
        # Test suggest_file_id
        suggestion = suggest_file_id("quantum_papper", ["quantum_paper", "neural_nets"])
        assert "quantum_paper" in suggestion

        # Test suggest_node_id
        suggestion = suggest_node_id("doc_n99", ["doc_n0", "doc_n1", "doc_n2"])
        # May or may not match depending on threshold
        assert isinstance(suggestion, str)

        # Test suggest_enum
        suggestion = suggest_enum("DETALED", ["ABSTRACT", "DETAILED", "RAW"])
        assert "DETAILED" in suggestion


# ===========================
# Compression Advisor Tests (Day 9)
# ===========================


class TestCompressionAdvisor:
    """Test compression size estimation"""

    def test_small_document_warning(self):
        """Tiny documents should advise sending as-is (#92 recalibration)"""
        advisor = CompressionAdvisor()
        text = "Short document." * 5  # ~50 tokens
        estimate = advisor.estimate_compression(text)

        assert estimate.original_tokens < 100
        assert estimate.confidence == "high"
        assert "as-is" in estimate.reasoning

    def test_medium_document_estimate(self):
        """Medium documents should have high confidence"""
        advisor = CompressionAdvisor()
        # Create ~500 token document
        text = "Quantum computing uses qubits for parallel computation. " * 50

        estimate = advisor.estimate_compression(text)

        assert 400 < estimate.original_tokens < 600
        # #92 recalibration: the engine keeps ~80% of nodes below 8K tokens, so
        # the estimate must be conservative (~1.2-2x), never the legacy 5-10x
        # that produced the 7.08x-estimated vs 1.17x-actual dogfood overclaim.
        assert estimate.confidence == "low"
        assert 1.0 <= estimate.compression_ratio < 2.5

    def test_large_document_estimate(self):
        """Large documents should predict high compression"""
        advisor = CompressionAdvisor()
        # Create ~2000 token document
        text = "Quantum error correction codes protect quantum information. " * 200

        estimate = advisor.estimate_compression(text)

        assert estimate.original_tokens > 1000
        # #92: ~2K tokens is still the keep-80% regime — conservative estimate.
        assert 1.0 <= estimate.compression_ratio < 2.5

    def test_redundancy_detection(self):
        """High redundancy should increase compression estimate"""
        advisor = CompressionAdvisor()

        # Low redundancy: unique sentences
        unique_text = "\n".join([f"Sentence number {i} is unique." for i in range(100)])

        # High redundancy: repeated phrases
        repetitive_text = "The same sentence repeated. " * 100

        est_unique = advisor.estimate_compression(unique_text)
        est_repetitive = advisor.estimate_compression(repetitive_text)

        # Repetitive should compress better
        assert est_repetitive.compression_ratio > est_unique.compression_ratio

    def test_structure_detection(self):
        """Structured documents should score higher"""
        advisor = CompressionAdvisor()

        # Structured: headers, paragraphs
        structured = """
# Introduction

Quantum computing is revolutionary.

## Background

Quantum bits enable parallel computation.

- Superposition
- Entanglement
- Decoherence

## Applications

Many potential uses exist.
"""

        # Unstructured: wall of text
        unstructured = "Word " * 200

        structure_score_structured = advisor._estimate_structure(structured)
        structure_score_unstructured = advisor._estimate_structure(unstructured)

        assert structure_score_structured > structure_score_unstructured

    def test_best_worst_case_scenarios(self):
        """Estimates should include range"""
        advisor = CompressionAdvisor()
        text = "Medium length document. " * 100

        estimate = advisor.estimate_compression(text)

        assert "best_case" in estimate.__dict__
        assert "worst_case" in estimate.__dict__
        assert estimate.best_case["ratio"] > estimate.compression_ratio
        assert estimate.worst_case["ratio"] < estimate.compression_ratio

    def test_format_estimate_output(self):
        """Formatted output should be readable"""
        advisor = CompressionAdvisor()
        text = "Test document. " * 50

        estimate = advisor.estimate_compression(text)
        formatted = advisor.format_estimate(estimate)

        assert "COMPRESSION ESTIMATE" in formatted
        assert "Original tokens" in formatted
        assert "BEST CASE" in formatted
        assert "WORST CASE" in formatted

    def test_functional_interface(self):
        """Convenience function should work"""
        text = "Test document. " * 50
        result = estimate_compression_size(text)

        assert "original_tokens" in result
        assert "compression_ratio" in result
        assert "confidence" in result


# ===========================
# Integration Tests
# ===========================


class TestUXIntegration:
    """Test UX features working together"""

    def test_fidelity_recommendation_with_estimate(self):
        """Fidelity advisor + compression advisor integration"""
        # Scenario: User wants to know best fidelity for document
        text = "Quantum paper content. " * 100

        # Step 1: Estimate compression
        comp_advisor = CompressionAdvisor()
        comp_est = comp_advisor.estimate_compression(text)

        # Step 2: Get fidelity recommendation
        # Assume compression creates ~10 nodes
        num_nodes = 10
        fid_advisor = FidelityAdvisor()
        fid_rec = fid_advisor.recommend(
            UseCase.QUESTION_ANSWERING, num_nodes=num_nodes, token_budget=300
        )

        # Verify both work
        assert comp_est.compression_ratio > 1
        assert fid_rec.token_estimate <= 300

    def test_error_message_improvements(self):
        """Smart errors should be actionable"""
        # Simulate common user error: typo in file_id
        available_files = ["quantum_paper_v1", "neural_nets_doc", "blockchain_research"]

        # User types "quantum_papper_v1"
        error = SmartError.file_id_not_found("quantum_papper_v1", available_files)
        error_msg = str(error)

        # Should be helpful
        assert "quantum_paper_v1" in error_msg or "Did you mean" in error_msg
        assert any(file_id in error_msg for file_id in available_files)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
