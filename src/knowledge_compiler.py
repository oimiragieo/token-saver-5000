"""
Knowledge compiler — periodic consolidation of flat memories into
cross-linked markdown concept articles with a navigable index.

Designed to be called on-demand or via a scheduled hook.  Idempotent:
rerunning on the same memory set produces the same output.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .memory_api import MemoryAPI

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")
_DEFAULT_OUTPUT_DIR = Path(".semantic_modulator_data") / "compiled"
_SIMILARITY_THRESHOLD = 0.65  # Above this, two memories are considered duplicates


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _WORD_RE.findall(text)}


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ConceptArticle:
    """A compiled concept page grouping related memories."""

    title: str
    category: str
    memories: list[dict[str, Any]] = field(default_factory=list)
    related_titles: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"# {self.title}\n"]
        lines.append(f"**Category:** {self.category}  ")
        lines.append(f"**Entries:** {len(self.memories)}  ")
        lines.append(f"**Compiled:** {_utc_now_iso()}\n")

        for mem in self.memories:
            confidence = mem.get("confidence", "")
            conf_str = f" (confidence: {confidence})" if confidence else ""
            lines.append(f"- {mem['text']}{conf_str}")

        if self.related_titles:
            lines.append("\n## Related concepts\n")
            for title in self.related_titles:
                slug = title.lower().replace(" ", "-")
                lines.append(f"- [{title}]({slug}.md)")

        lines.append("")
        return "\n".join(lines)


@dataclass
class CompilationResult:
    """Result of a full compilation run."""

    articles: list[ConceptArticle] = field(default_factory=list)
    index_markdown: str = ""
    total_memories: int = 0
    deduplicated: int = 0
    output_dir: str = ""


# ---------------------------------------------------------------------------
# Core compiler
# ---------------------------------------------------------------------------


class KnowledgeCompiler:
    """Groups, deduplicates, and compiles memories into concept articles."""

    def __init__(
        self,
        *,
        similarity_threshold: float = _SIMILARITY_THRESHOLD,
        output_dir: Path | str | None = None,
    ):
        self._threshold = similarity_threshold
        self._output_dir = Path(output_dir) if output_dir else _DEFAULT_OUTPUT_DIR

    # -- public API ----------------------------------------------------------

    def compile(
        self,
        memories: list[dict[str, Any]],
        *,
        write_files: bool = False,
    ) -> CompilationResult:
        """Compile a list of memory dicts into concept articles.

        Args:
            memories: List of memory dicts (as returned by ``MemoryAPI.list_memories``).
            write_files: If True, persist markdown files to *output_dir*.

        Returns:
            CompilationResult with articles, index, and stats.
        """
        total = len(memories)

        # 1. Deduplicate
        deduped = self._deduplicate(memories)

        # 2. Group by category
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for mem in deduped:
            groups[mem.get("category", "general")].append(mem)

        # 3. Build articles
        articles: list[ConceptArticle] = []
        for category, mems in sorted(groups.items()):
            title = category.replace("_", " ").title()
            article = ConceptArticle(title=title, category=category, memories=mems)
            articles.append(article)

        # 4. Cross-link: articles that share token overlap
        self._cross_link(articles)

        # 5. Build index
        index_md = self._build_index(articles)

        result = CompilationResult(
            articles=articles,
            index_markdown=index_md,
            total_memories=total,
            deduplicated=total - len(deduped),
            output_dir=str(self._output_dir),
        )

        if write_files:
            self._write(articles, index_md)

        return result

    def compile_from_api(
        self,
        *,
        memory_api: MemoryAPI | None = None,
        workspace_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        write_files: bool = False,
    ) -> CompilationResult:
        """Convenience wrapper that pulls memories from MemoryAPI before compiling."""
        api = memory_api or MemoryAPI.get_api()
        memories = api.list_memories(
            workspace_id=workspace_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        return self.compile(memories, write_files=write_files)

    # -- internals -----------------------------------------------------------

    def _deduplicate(self, memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove near-duplicate memories using pairwise similarity."""
        if not memories:
            return []
        kept: list[dict[str, Any]] = []
        for mem in memories:
            text = str(mem.get("text", ""))
            is_dup = any(_similarity(text, str(k.get("text", ""))) >= self._threshold for k in kept)
            if not is_dup:
                kept.append(mem)
        return kept

    def _cross_link(self, articles: list[ConceptArticle]) -> None:
        """Set related_titles by comparing token overlap between articles."""
        article_tokens: list[set[str]] = []
        for article in articles:
            combined = " ".join(m.get("text", "") for m in article.memories)
            article_tokens.append(_tokenize(combined))

        for i, article in enumerate(articles):
            for j, other in enumerate(articles):
                if i == j:
                    continue
                overlap = len(article_tokens[i] & article_tokens[j])
                union = len(article_tokens[i] | article_tokens[j])
                if union > 0 and overlap / union > 0.1:
                    article.related_titles.append(other.title)

    def _build_index(self, articles: list[ConceptArticle]) -> str:
        """Build an index.md summarizing all articles."""
        lines = ["# Knowledge Index\n"]
        lines.append(f"**Compiled:** {_utc_now_iso()}  ")
        lines.append(f"**Articles:** {len(articles)}\n")
        for article in articles:
            slug = article.title.lower().replace(" ", "-")
            count = len(article.memories)
            lines.append(f"- [{article.title}]({slug}.md) — {count} entries ({article.category})")
        lines.append("")
        return "\n".join(lines)

    def _write(self, articles: list[ConceptArticle], index_md: str) -> None:
        """Persist markdown files to output_dir."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        for article in articles:
            slug = article.title.lower().replace(" ", "-")
            path = self._output_dir / f"{slug}.md"
            path.write_text(article.to_markdown(), encoding="utf-8", newline="\n")
        index_path = self._output_dir / "index.md"
        index_path.write_text(index_md, encoding="utf-8", newline="\n")
