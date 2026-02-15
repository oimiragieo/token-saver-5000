from __future__ import annotations

import numpy as np

import src.multimodal_compressor as mm
from src.multimodal_compressor import ModalityType, MultiModalCompressor, MultiModalNode


class _DummyEncoder:
    def __init__(self, dim: int = 4):
        self._dim = dim

    def get_sentence_embedding_dimension(self):
        return self._dim

    def encode(self, items, show_progress_bar: bool = True):
        out = []
        for item in items:
            if hasattr(item, "size"):  # PIL-like object path
                seed = 7
            else:
                text = str(item)
                seed = len(text)
            rng = np.random.default_rng(seed)
            out.append(rng.random(self._dim))
        return np.array(out)


class _DummyManager:
    def get_text_embedder(self):
        return _DummyEncoder()

    def get_code_embedder(self):
        return _DummyEncoder()

    def get_image_embedder(self):
        return _DummyEncoder()


def test_multimodal_node_defaults():
    node = MultiModalNode(
        node_id="n1",
        modality=ModalityType.TEXT,
        content="x",
        embedding=np.array([1.0, 0.0]),
    )
    assert node.metadata == {}


def test_ingest_search_get_content_and_summary(monkeypatch):
    monkeypatch.setattr(mm, "EmbeddingManager", _DummyManager)
    compressor = MultiModalCompressor(use_clip_for_images=False, use_codebert_for_code=False)

    # Force image branch without PIL dependency
    monkeypatch.setattr(compressor, "_encode_image", lambda _img: None)
    stats = compressor.ingest_mixed_content(
        [
            {"type": "text", "content": "alpha documentation", "metadata": {"file": "README.md"}},
            {"type": "code", "content": "def train(): pass", "metadata": {"file": "train.py"}},
            {"type": "image", "content": b"\x89PNG", "metadata": {"file": "diagram.png"}},
            {"type": "unknown", "content": "123", "metadata": {"file": "misc.txt"}},
        ],
        project_id="proj",
        similarity_threshold=0.0,
    )
    assert stats["project_id"] == "proj"
    assert stats["total_nodes"] >= 3
    assert "nodes_by_modality" in stats

    all_results = compressor.search_cross_modal(
        "train", query_type="text", project_id="proj", top_k=5
    )
    assert all_results

    code_only = compressor.search_cross_modal(
        "train", query_type="code", project_id="proj", top_k=5, filter_modality="code"
    )
    assert all(r[2] == "code" for r in code_only)

    unknown_query = compressor.search_cross_modal("fallback", query_type="weird", project_id="proj")
    assert unknown_query

    # Insert an image node to exercise base64 content path
    compressor.nodes["proj_img_manual"] = MultiModalNode(
        node_id="proj_img_manual",
        modality=ModalityType.IMAGE,
        content=b"\x89PNG",
        embedding=np.array([0.1, 0.2, 0.3, 0.4]),
        metadata={"file": "manual.png"},
    )
    image_payload = compressor.get_node_content("proj_img_manual")
    assert image_payload["content_type"] == "base64"
    assert image_payload["modality"] == "image"

    text_id = next(iter([nid for nid in compressor.nodes if nid.startswith("proj_n")]))
    text_payload = compressor.get_node_content(text_id)
    assert text_payload["content_type"] == "text"

    missing = compressor.get_node_content("missing")
    assert "error" in missing

    summary = compressor.generate_multimodal_summary("proj")
    assert "MULTI-MODAL PROJECT: proj" in summary

    assert compressor.generate_multimodal_summary("unknown") == "Project unknown not found"


def test_search_image_query_returns_empty_when_encoder_absent(monkeypatch):
    monkeypatch.setattr(mm, "EmbeddingManager", _DummyManager)
    compressor = MultiModalCompressor(use_clip_for_images=False, use_codebert_for_code=False)
    compressor.nodes["proj_n0"] = MultiModalNode(
        node_id="proj_n0",
        modality=ModalityType.TEXT,
        content="hello",
        embedding=np.array([0.1, 0.2, 0.3, 0.4]),
    )
    result = compressor.search_cross_modal(b"img", query_type="image", project_id="proj")
    assert result == []
