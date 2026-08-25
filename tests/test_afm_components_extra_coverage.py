"""afm components — consolidated from coverage_boost1-4/4b (backlog N10 re-filing).

Moved verbatim (byte-for-byte class bodies); nothing reworded.
"""

from unittest.mock import AsyncMock, MagicMock, patch


def _has_pillow() -> bool:
    try:
        import PIL  # noqa: F401

        return True
    except ImportError:
        return False


def _make_mock_context(**overrides):
    """Build a mock HandlerContext dict."""
    compressor = MagicMock()
    compressor.chunks = {}
    compressor.graphs = {}
    compressor.file_metadata = {}

    ctx = {
        "compressor": compressor,
        "persistence": MagicMock(),
        "resource_manager": MagicMock(),
        "sync_manager": MagicMock(),
        "version_manager": MagicMock(),
        "path_validator": MagicMock(),
        "retrieval_history": {},
        "context_window_adapter": MagicMock(),
        "multilevel_encoder": MagicMock(),
    }
    ctx["resource_manager"].unregister_document_async = AsyncMock()
    ctx["version_manager"].delete_versions_async = AsyncMock()
    ctx.update(overrides)
    return ctx


class TestAFMComponents:
    def test_token_counter_double_fallback(self):
        from src.afm import TokenCounter

        with patch("tiktoken.encoding_for_model", side_effect=Exception("no")):
            with patch("tiktoken.get_encoding", side_effect=Exception("no")):
                tc = TokenCounter()
                assert tc.encoding is None
                count = tc.count("hello world test")
                assert count > 0

    def test_token_counter_fallback_to_cl100k(self):
        from src.afm import TokenCounter

        with patch("tiktoken.encoding_for_model", side_effect=Exception("no")):
            tc = TokenCounter()
            assert tc.encoding is not None

    def test_heuristic_compressor_empty_sentences(self):
        from src.afm import HeuristicCompressor, TokenCounter

        tc = TokenCounter()
        comp = HeuristicCompressor(tc)
        result = comp.compress("", 10)
        assert isinstance(result, str)

    def test_heuristic_compressor_truncation_fallback(self):
        from src.afm import HeuristicCompressor, TokenCounter

        tc = TokenCounter()
        comp = HeuristicCompressor(tc)
        result = comp.compress("Word " * 100, 5)
        assert isinstance(result, str)

    def test_llm_compressor_falls_back(self):
        from src.afm import LLMCompressor, TokenCounter

        tc = TokenCounter()
        comp = LLMCompressor(tc, api_key="fake", model="gpt-4o-mini")
        result = comp.compress("This is a test sentence. Another sentence here.", 20)
        assert isinstance(result, str)

    def test_llm_compressor_no_api_key(self):
        from src.afm import LLMCompressor, TokenCounter

        tc = TokenCounter()
        comp = LLMCompressor(tc, api_key=None)
        # LLMCompressor always falls back to heuristic - just verify it works
        result = comp.compress("Test sentence here.", 10)
        assert isinstance(result, str)

    def test_hashing_embedder(self):
        from src.afm import HashingEmbedder

        emb = HashingEmbedder(dim=64)
        result = emb.encode(["hello world", "test"])
        assert result.shape == (2, 64)

    def test_hashing_embedder_empty_text(self):
        from src.afm import HashingEmbedder

        emb = HashingEmbedder(dim=64)
        result = emb.encode([""])
        assert result.shape == (1, 64)

    def test_importance_classifier_llm_mode(self):
        from src.afm import ImportanceClassifier, Message, ImportanceLevel

        clf = ImportanceClassifier(use_llm=True, api_key="fake")
        msg = Message(role="user", content="test", importance=ImportanceLevel.TRIVIAL, turn_index=0)
        level = clf._classify_llm(msg)
        assert level in [
            ImportanceLevel.CRITICAL,
            ImportanceLevel.RELEVANT,
            ImportanceLevel.TRIVIAL,
        ]

    def test_importance_classifier_no_apikey_warning(self):
        from src.afm import ImportanceClassifier

        clf = ImportanceClassifier(use_llm=True, api_key=None)
        assert clf.use_llm is False

    def test_focus_manager_system_preamble_too_large(self):
        from src.afm import FocusManager, AFMConfig

        with patch("src.afm.SENTENCE_TRANSFORMERS_AVAILABLE", False):
            fm = FocusManager(AFMConfig())
            packed = []
            fm._try_add_system_preamble("x" * 10000, 5, packed)
            assert len(packed) == 0  # skipped

    def test_focus_manager_llm_compression_config(self):
        from src.afm import FocusManager, AFMConfig

        cfg = AFMConfig(use_llm_compression=True, llm_api_key="fake")
        with patch("src.afm.SENTENCE_TRANSFORMERS_AVAILABLE", False):
            fm = FocusManager(cfg)
            assert fm.compressor is not None
