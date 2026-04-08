"""Tests for knowledge compiler (src/knowledge_compiler.py)."""

import pytest

from src.memory_api import MemoryAPI
from src.knowledge_compiler import (
    ConceptArticle,
    CompilationResult,
    KnowledgeCompiler,
)


@pytest.fixture(autouse=True)
def reset_memory():
    MemoryAPI.reset_singleton()
    yield
    MemoryAPI.reset_singleton()


def _populate_memories(api: MemoryAPI) -> None:
    """Add a variety of memories for testing compilation."""
    api.add_memory(text="Always use black before commits", category="pattern")
    api.add_memory(text="Watch out for N+1 query problems", category="gotcha")
    api.add_memory(text="Decided to use PostgreSQL over MongoDB", category="decision")
    api.add_memory(text="Best practice: dependency injection everywhere", category="pattern")
    api.add_memory(text="Authentication bug caused session leaks", category="issue")


# ---------------------------------------------------------------------------
# ConceptArticle
# ---------------------------------------------------------------------------


class TestConceptArticle:
    def test_to_markdown(self):
        article = ConceptArticle(
            title="Pattern",
            category="pattern",
            memories=[
                {"text": "Always use black", "confidence": 0.9},
                {"text": "Use dependency injection", "confidence": 0.8},
            ],
        )
        md = article.to_markdown()
        assert "# Pattern" in md
        assert "Always use black" in md
        assert "confidence: 0.9" in md

    def test_to_markdown_with_related(self):
        article = ConceptArticle(
            title="Pattern",
            category="pattern",
            memories=[{"text": "Use DI"}],
            related_titles=["Gotcha", "Decision"],
        )
        md = article.to_markdown()
        assert "## Related concepts" in md
        assert "[Gotcha]" in md
        assert "[Decision]" in md

    def test_empty_article(self):
        article = ConceptArticle(title="Empty", category="general", memories=[])
        md = article.to_markdown()
        assert "# Empty" in md
        assert "Entries:** 0" in md


# ---------------------------------------------------------------------------
# KnowledgeCompiler
# ---------------------------------------------------------------------------


class TestKnowledgeCompiler:
    def test_compile_groups_by_category(self):
        api = MemoryAPI()
        _populate_memories(api)
        compiler = KnowledgeCompiler()
        result = compiler.compile_from_api(memory_api=api)
        categories = {a.category for a in result.articles}
        assert "pattern" in categories
        assert "gotcha" in categories
        assert "decision" in categories
        assert "issue" in categories

    def test_compile_returns_compilation_result(self):
        api = MemoryAPI()
        _populate_memories(api)
        compiler = KnowledgeCompiler()
        result = compiler.compile_from_api(memory_api=api)
        assert isinstance(result, CompilationResult)
        assert result.total_memories == 5
        assert len(result.articles) >= 1
        assert result.index_markdown != ""

    def test_compile_deduplicates(self):
        api = MemoryAPI()
        api.add_memory(text="Always use black before commits", category="pattern")
        api.add_memory(text="Always use black before commits!", category="pattern")
        compiler = KnowledgeCompiler(similarity_threshold=0.65)
        result = compiler.compile_from_api(memory_api=api)
        assert result.deduplicated >= 1

    def test_compile_empty_memories(self):
        api = MemoryAPI()
        compiler = KnowledgeCompiler()
        result = compiler.compile_from_api(memory_api=api)
        assert result.total_memories == 0
        assert len(result.articles) == 0
        assert "Articles:** 0" in result.index_markdown

    def test_index_markdown_lists_articles(self):
        api = MemoryAPI()
        _populate_memories(api)
        compiler = KnowledgeCompiler()
        result = compiler.compile_from_api(memory_api=api)
        for article in result.articles:
            assert article.title in result.index_markdown

    def test_cross_linking(self):
        api = MemoryAPI()
        # Create two categories with overlapping tokens
        api.add_memory(text="Use black formatting tool always", category="pattern")
        api.add_memory(
            text="Black formatting tool has a gotcha with line length", category="gotcha"
        )
        compiler = KnowledgeCompiler()
        result = compiler.compile_from_api(memory_api=api)
        # At least one article should have related titles
        has_related = any(len(a.related_titles) > 0 for a in result.articles)
        assert has_related

    def test_compile_with_scoping(self):
        api = MemoryAPI()
        api.add_memory(
            text="Use PostgreSQL for this workspace", category="decision", workspace_id="acme"
        )
        api.add_memory(
            text="Use MongoDB for this workspace", category="decision", workspace_id="other"
        )
        compiler = KnowledgeCompiler()
        result = compiler.compile_from_api(memory_api=api, workspace_id="acme")
        assert result.total_memories == 1

    def test_compile_write_files(self, tmp_path):
        api = MemoryAPI()
        _populate_memories(api)
        compiler = KnowledgeCompiler(output_dir=tmp_path / "compiled")
        result = compiler.compile_from_api(memory_api=api, write_files=True)
        # Check files were written
        index_path = tmp_path / "compiled" / "index.md"
        assert index_path.exists()
        assert "Knowledge Index" in index_path.read_text(encoding="utf-8")
        # Check article files
        for article in result.articles:
            slug = article.title.lower().replace(" ", "-")
            article_path = tmp_path / "compiled" / f"{slug}.md"
            assert article_path.exists()

    def test_compile_idempotent(self):
        api = MemoryAPI()
        _populate_memories(api)
        compiler = KnowledgeCompiler()
        result1 = compiler.compile_from_api(memory_api=api)
        result2 = compiler.compile_from_api(memory_api=api)
        assert result1.total_memories == result2.total_memories
        assert len(result1.articles) == len(result2.articles)
        assert result1.deduplicated == result2.deduplicated

    def test_compile_direct_memories_list(self):
        memories = [
            {"text": "Use TypeScript", "category": "decision", "memory_id": "m1"},
            {"text": "Always lint first", "category": "pattern", "memory_id": "m2"},
        ]
        compiler = KnowledgeCompiler()
        result = compiler.compile(memories)
        assert result.total_memories == 2
        assert len(result.articles) == 2
