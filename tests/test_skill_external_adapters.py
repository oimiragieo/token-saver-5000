"""Tests for portable skill external adapter normalization."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS = ROOT / "skills" / "token-saver-context-compression" / "scripts"
if str(SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SKILL_SCRIPTS))

from _external_adapters import (  # noqa: E402
    adapt_input_to_text,
    infer_adapter,
    normalize_langchain_like,
    normalize_llamaindex_like,
)


class _LangChainDoc:
    def __init__(self, page_content: str, metadata: dict | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class _LlamaNode:
    def __init__(self, text: str, *, node_id: str | None = None, metadata: dict | None = None):
        self._text = text
        self.node_id = node_id
        self.metadata = metadata or {}

    def get_content(self) -> str:
        return self._text


def test_normalize_langchain_like_objects():
    docs = [
        _LangChainDoc("Retry policy summary", {"source": "ops.md"}),
        _LangChainDoc("Circuit breaker defaults", {"source": "arch.md"}),
    ]
    rows = normalize_langchain_like(docs)
    assert len(rows) == 2
    assert rows[0]["text"] == "Retry policy summary"
    assert rows[1]["metadata"]["source"] == "arch.md"


def test_normalize_llamaindex_like_objects():
    nodes = [
        _LlamaNode("Timeout budget is 300ms", node_id="n1", metadata={"section": "latency"}),
        _LlamaNode("Retry up to 3 times", node_id="n2", metadata={"section": "retries"}),
    ]
    rows = normalize_llamaindex_like(nodes)
    assert len(rows) == 2
    assert rows[0]["source_id"] == "n1"
    assert rows[1]["text"] == "Retry up to 3 times"


def test_infer_adapter_for_json_shapes():
    langchain_shape = [{"page_content": "hello", "metadata": {"a": 1}}]
    llama_shape = [{"text": "hello", "metadata": {"a": 1}, "id_": "x"}]
    assert infer_adapter(langchain_shape) == "langchain_json"
    assert infer_adapter(llama_shape) == "llamaindex_json"


def test_adapt_input_to_text_auto_and_metadata():
    payload = [
        {"page_content": "alpha", "metadata": {"source": "a.md"}},
        {"page_content": "beta", "metadata": {"source": "b.md"}},
    ]
    adapted_text, meta = adapt_input_to_text(payload, adapter="auto")
    assert "alpha" in adapted_text
    assert "beta" in adapted_text
    assert meta["adapter"] == "langchain_json"
    assert meta["document_count"] == 2
