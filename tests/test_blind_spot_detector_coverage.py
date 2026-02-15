from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pytest

from src.blind_spot_detector import BlindSpotDetector, HaloEffectDetector


@dataclass
class _Node:
    text: str
    embedding: np.ndarray
    importance: float


class _Model:
    def encode(self, texts):
        out = []
        for text in texts:
            t = text.lower()
            if "alpha" in t:
                out.append(np.array([1.0, 0.0, 0.0]))
            elif "beta" in t:
                out.append(np.array([0.0, 1.0, 0.0]))
            else:
                out.append(np.array([0.0, 0.0, 1.0]))
        return np.array(out)


class _Compressor:
    def __init__(self):
        self.model = _Model()
        self.graphs = {"doc": nx.Graph()}
        self.graphs["doc"].add_nodes_from(["doc_n0", "doc_n1", "doc_n2"])
        self.chunks = {
            "doc_n0": _Node("alpha fact", np.array([1.0, 0.0, 0.0]), 0.95),
            "doc_n1": _Node("beta detail", np.array([0.0, 1.0, 0.0]), 0.6),
            "doc_n2": _Node("gamma note", np.array([0.0, 0.0, 1.0]), 0.3),
        }

    def _generate_summary(self, text: str, max_length: int = 60) -> str:
        return text[:max_length]


def test_analyze_response_and_wrappers():
    detector = BlindSpotDetector(_Compressor(), similarity_threshold=0.5, urgency_threshold=0.3)
    report = detector.analyze_response(
        ai_response="This answer talks about alpha",
        file_id="doc",
        retrieved_node_ids=["doc_n1"],
    )
    assert report.total_blind_spots >= 1
    assert isinstance(report.recommendations, list)

    wrapped = detector.check_blind_spots(
        ai_response="This answer talks about alpha",
        file_id="doc",
        retrieved_nodes=["doc_n1"],
    )
    assert "blind_spot_nodes" in wrapped

    ok, correction = detector.validate_response_fidelity(
        ai_response="This answer talks about alpha",
        file_id="doc",
        retrieved_node_ids=["doc_n1"],
    )
    assert ok is False
    assert "critical blind spots" in correction

    formatted = detector.format_report(report)
    assert "BLIND SPOT ANALYSIS REPORT" in formatted


def test_halo_effect_detector_modes_and_missing_file():
    compressor = _Compressor()
    halo = HaloEffectDetector(compressor)
    is_hallu, claims = halo.detect_hallucination("alpha statement", "doc", confidence_threshold=0.2)
    assert is_hallu is False
    assert claims == []

    is_hallu2, claims2 = halo.detect_hallucination(
        "completely unrelated", "doc", confidence_threshold=1.1
    )
    assert is_hallu2 is True
    assert claims2

    detector = BlindSpotDetector(compressor)
    wrapped = detector.detect_hallucination("unrelated", "doc", confidence_threshold=1.1)
    assert wrapped["is_grounded"] is False
    assert wrapped["hallucination_score"] == 1.0

    with pytest.raises(ValueError, match="File missing not found"):
        detector.analyze_response("alpha", "missing", [])
