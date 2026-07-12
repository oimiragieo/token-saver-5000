"""
Code-Specific Semantic Compressor

Optimized for compressing source code using:
- AST (Abstract Syntax Tree) parsing for intelligent chunking
- Function/class boundaries as natural semantic units
- Import dependency graphs
- Code-aware similarity metrics
- Documentation extraction

Supports: Python, JavaScript, TypeScript, Java, C++, Go, Rust
"""

import ast
import logging
import os
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import numpy as np
import networkx as nx

from .embeddings import EmbeddingManager

logger = logging.getLogger(__name__)

# Code-graph memory-safety bounds (task #236 rank11). Defined locally rather than
# imported from semantic_compressor so `import code_compressor` does NOT eagerly
# import the heavy semantic_compressor module (which pulls the embedding stack) --
# same env vars + defaults as semantic_compressor._MAX_GRAPH_CHUNKS /
# _SIMILARITY_BLOCK_SIZE, so an operator override affects both paths identically.
_SIMILARITY_BLOCK_SIZE: int = int(os.environ.get("SIMILARITY_BLOCK_SIZE", "256"))
_MAX_GRAPH_CHUNKS: int = int(os.environ.get("MAX_GRAPH_CHUNKS", "2500"))


def _ast_qualname(node: ast.AST) -> str:
    """Full dotted qualname from the ``_tsk_parent`` chain.

    #195 qualified a node by its enclosing class only when the IMMEDIATE parent
    was a ``ClassDef`` (``A.foo``), leaving a bare name otherwise. BUG-3 (droid
    review 2026-07-11): a function nested in a FUNCTION (``def deco(): def
    wrapper(): ...``) then fell through to the bare ``wrapper`` and collided
    across decorators in the ``chunk_id``-keyed dict, silently dropping all but
    the last. Walking the whole ClassDef/FunctionDef parent chain gives every
    nesting site a distinct id (``deco.wrapper``, ``A.outer.inner``) while
    preserving the existing shapes: a method stays ``A.foo`` and a top-level
    def/class stays its bare name (its parent is the Module, not in the chain).
    """
    parts = [getattr(node, "name", "<anon>")]
    parent = getattr(node, "_tsk_parent", None)
    while isinstance(parent, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        parts.append(parent.name)
        parent = getattr(parent, "_tsk_parent", None)
    return ".".join(reversed(parts))


class CodeLanguage(Enum):
    """Supported programming languages"""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"
    UNKNOWN = "unknown"


@dataclass
class CodeChunk:
    """Represents a semantic code unit"""

    chunk_id: str
    chunk_type: str  # "function", "class", "import", "comment", "block"
    code: str
    name: str  # Function/class name
    docstring: Optional[str]
    start_line: int
    end_line: int
    dependencies: List[str]  # Imports, function calls, etc.
    embedding: Optional[np.ndarray] = None
    importance: float = 0.0


class CodeSemanticCompressor:
    """
    Semantic compressor optimized for source code.

    Key differences from text compressor:
    1. AST-based chunking (functions, classes) instead of paragraph splitting
    2. Dependency graph (imports, calls) in addition to semantic similarity
    3. Code-specific importance (public vs private, entry points, etc.)
    4. Documentation extraction and indexing
    """

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",  # Code-specific model
        similarity_threshold: float = 0.70,
    ):
        """
        Initialize code compressor.

        Args:
            model_name: Embedding model optimized for code
                       - "microsoft/codebert-base" (recommended)
                       - "all-MiniLM-L6-v2" (general, faster)
            similarity_threshold: Minimum similarity for edges
        """
        # Use EmbeddingManager for shared model caching (handles fallback internally).
        # In ONNX-only mode, get_code_embedder() raises ImportError —
        # fall back to the manager's encode() with tier fallback.
        from .embeddings import _EmbeddingManagerAdapter

        embedding_manager = EmbeddingManager()
        try:
            self.model = embedding_manager.get_code_embedder(model_name)
        except (ImportError, TypeError):
            self.model = _EmbeddingManagerAdapter(embedding_manager)
        self.similarity_threshold = similarity_threshold

        # Storage
        self.chunks: Dict[str, CodeChunk] = {}
        self.graphs: Dict[str, nx.DiGraph] = {}  # Directed graph for code dependencies
        self.file_metadata: Dict[str, Dict] = {}

    def detect_language(self, filepath: str) -> CodeLanguage:
        """Detect programming language from file extension"""
        ext_map = {
            ".py": CodeLanguage.PYTHON,
            ".js": CodeLanguage.JAVASCRIPT,
            ".jsx": CodeLanguage.JAVASCRIPT,
            ".ts": CodeLanguage.TYPESCRIPT,
            ".tsx": CodeLanguage.TYPESCRIPT,
            ".java": CodeLanguage.JAVA,
            ".cpp": CodeLanguage.CPP,
            ".cc": CodeLanguage.CPP,
            ".cxx": CodeLanguage.CPP,
            ".h": CodeLanguage.CPP,
            ".hpp": CodeLanguage.CPP,
            ".go": CodeLanguage.GO,
            ".rs": CodeLanguage.RUST,
        }

        for ext, lang in ext_map.items():
            if filepath.endswith(ext):
                return lang

        return CodeLanguage.UNKNOWN

    def chunk_python_code(self, code: str, file_id: str) -> List[CodeChunk]:
        """
        Parse Python code using AST to extract semantic chunks.

        Returns chunks for:
        - Import statements (grouped)
        - Functions (with docstrings)
        - Classes (with docstrings and methods)
        - Top-level code blocks
        """
        chunks = []

        try:
            tree = ast.parse(code)
        except (SyntaxError, RecursionError, MemoryError, ValueError) as exc:
            # #237 rank-1 HIGH (DoS): a crafted deeply-nested payload (e.g.
            # chained unary operators or bracket nesting) can blow CPython's
            # parser recursion/stack guards, raising RecursionError or
            # MemoryError instead of SyntaxError -- neither is a parse-time
            # syntax problem, but both mean "can't parse this safely," so we
            # degrade to the same line-based fallback rather than let the
            # exception crash the worker for every tenant sharing this
            # process. ValueError covers null-byte source.
            logger.warning(
                f"Failed to parse {file_id} as Python ({type(exc).__name__}), "
                "falling back to line-based chunking"
            )
            return self._chunk_by_lines(code, file_id)

        # Extract imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")

        if imports:
            import_code = "\n".join([line for line in code.split("\n") if "import" in line])
            chunks.append(
                CodeChunk(
                    chunk_id=f"{file_id}::imports",
                    chunk_type="import",
                    code=import_code,
                    name="imports",
                    docstring=None,
                    start_line=1,
                    end_line=len(import_code.split("\n")),
                    dependencies=imports,
                )
            )

        # #195: ast.walk recurses INTO classes, so a method is chunked as a
        # FunctionDef. Track each node's parent so a method can be qualified by
        # its class — an unqualified f"{file_id}::{method}" collides across
        # classes (and with a same-named top-level function), and self.chunks is
        # a dict keyed by chunk_id, so the collision silently OVERWRITES.
        for _parent in ast.walk(tree):
            for _child in ast.iter_child_nodes(_parent):
                _child._tsk_parent = _parent

        # Extract functions and classes
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Extract function (async def is a sibling type, not a subclass — #194)
                func_code = ast.get_source_segment(code, node)
                # #195 + BUG-3: qualify by the FULL parent chain (A.foo vs B.foo,
                # and deco.wrapper vs another_deco.wrapper for function-nested fns).
                _qualified_name = _ast_qualname(node)
                docstring = ast.get_docstring(node)

                # Find dependencies (function calls)
                deps = []
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name):
                            deps.append(child.func.id)
                        elif isinstance(child.func, ast.Attribute):
                            deps.append(child.func.attr)

                chunks.append(
                    CodeChunk(
                        chunk_id=f"{file_id}::{_qualified_name}",
                        chunk_type="function",
                        code=func_code or "",
                        name=node.name,
                        docstring=docstring,
                        start_line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        dependencies=deps,
                    )
                )

            elif isinstance(node, ast.ClassDef):
                # Extract class
                class_code = ast.get_source_segment(code, node)
                docstring = ast.get_docstring(node)

                # #220 rank 5: qualify a NESTED class by its enclosing class,
                # mirroring the FunctionDef fix above. ast.walk recurses into
                # classes, so two same-named nested classes (e.g. Django's
                # `class Meta` inside two different outer classes) both produced
                # f"{file_id}::Meta" and the second silently OVERWROTE the first
                # in the chunk_id-keyed dict — dropping it from the skeleton.
                # #220 rank5 + BUG-3: full parent chain (A.Meta, deco.LocalClass).
                _qualified_class_name = _ast_qualname(node)

                # Get method names
                methods = [
                    n.name
                    for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]

                chunks.append(
                    CodeChunk(
                        chunk_id=f"{file_id}::{_qualified_class_name}",
                        chunk_type="class",
                        code=class_code or "",
                        name=node.name,
                        docstring=docstring,
                        start_line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        dependencies=methods,
                    )
                )

        return chunks

    def chunk_javascript_code(self, code: str, file_id: str) -> List[CodeChunk]:
        """
        Parse JavaScript/TypeScript code using regex patterns.

        Note: For production, use a proper JS parser like esprima or babel parser.
        This is a simplified regex-based approach.
        """
        chunks = []

        # Extract imports
        import_pattern = r'(?:import|require)\s+.*?(?:from\s+[\'"](.+?)[\'"]|[\'"](.+?)[\'"])'
        imports = re.findall(import_pattern, code)

        if imports:
            import_lines = [
                line for line in code.split("\n") if "import" in line or "require" in line
            ]
            chunks.append(
                CodeChunk(
                    chunk_id=f"{file_id}::imports",
                    chunk_type="import",
                    code="\n".join(import_lines),
                    name="imports",
                    docstring=None,
                    start_line=1,
                    end_line=len(import_lines),
                    dependencies=[imp[0] or imp[1] for imp in imports],
                )
            )

        # Extract functions. Match only the SIGNATURE up to the opening brace, then scan
        # for the matching close brace with a string/comment-aware depth counter. The old
        # `\{([^}]*)\}` stopped at the first inner `}`, truncating every function with a
        # nested block (if/for/object-literal/nested-fn) and SILENTLY dropping its tail
        # from the chunk's `code` (#212b, backlog master-plan Wave 1, 2026-07-12).
        sig_pattern = (
            r"(?:function|const|let|var)\s+(\w+)\s*=?\s*"
            r"(?:\([^)]*\)|async\s*\([^)]*\))\s*(?:=>)?\s*\{"
        )
        last_end = -1
        for match in re.finditer(sig_pattern, code, re.MULTILINE | re.DOTALL):
            if match.start() < last_end:
                continue  # nested inside an already-captured function
            open_idx = match.end() - 1  # index of the opening '{'
            close_idx = self._find_matching_brace(code, open_idx)
            if close_idx is None:
                continue  # unbalanced / truncated source -> skip gracefully
            func_name = match.group(1)
            full = code[match.start() : close_idx + 1]
            last_end = close_idx + 1

            # Extract docstring (JSDoc)
            jsdoc_pattern = r"/\*\*([^*]|\*(?!/))*\*/"
            jsdoc_match = re.search(jsdoc_pattern, code[: match.start()])
            docstring = jsdoc_match.group(0) if jsdoc_match else None

            chunks.append(
                CodeChunk(
                    chunk_id=f"{file_id}::{func_name}",
                    chunk_type="function",
                    code=full,
                    name=func_name,
                    docstring=docstring,
                    start_line=code[: match.start()].count("\n") + 1,
                    end_line=code[: close_idx + 1].count("\n") + 1,
                    dependencies=[],
                )
            )

        return chunks

    # Keywords after which a `/` begins a regex literal, not division (JS lexer rule).
    _REGEX_PRECEDING_KEYWORDS = frozenset(
        {
            "return",
            "typeof",
            "instanceof",
            "in",
            "of",
            "new",
            "delete",
            "void",
            "do",
            "else",
            "yield",
            "await",
            "case",
            "throw",
        }
    )

    @staticmethod
    def _find_matching_brace(code: str, open_idx: int):
        """Index of the '}' matching ``code[open_idx] == '{'``, skipping braces inside JS
        strings ('', "", ``), // and /* */ comments, AND regex literals (whose `{`/`}` and
        `/*`-looking bytes must NOT be read as structural). Returns None if unbalanced.
        Regex-vs-division is resolved with the standard heuristic (previous significant
        token is not a value, or is a regex-preceding keyword). (#212b codex round 2.)

        Known limitation (codex round-2, out of scope for this non-parser chunker): a
        BARE regex in statement position after `)` -- e.g. ``if (x) /[}]/`` -- reads the
        `)` as a value and treats the `/` as division, so a `}` inside such a regex could
        still be miscounted. Fully resolving it needs a real JS tokenizer (see the
        chunk_javascript_code docstring). This is rare code and NOT a regression: the
        pre-fix regex truncated EVERY nested-brace function; this scanner fixes the common
        cases (nested blocks, object literals, strings, comments, and regex literals in
        expression/keyword position)."""
        depth = 0
        i = open_idx
        n = len(code)
        while i < n:
            ch = code[i]
            if ch in ('"', "'", "`"):
                quote = ch
                i += 1
                while i < n:
                    c = code[i]
                    if c == "\\":
                        i += 2
                        continue
                    if c == quote:
                        i += 1
                        break
                    i += 1
                continue
            if ch == "/" and i + 1 < n:
                nxt = code[i + 1]
                if nxt == "/":  # line comment (always)
                    nl = code.find("\n", i)
                    i = n if nl == -1 else nl
                    continue
                if nxt == "*":  # block comment (always)
                    end = code.find("*/", i + 2)
                    i = n if end == -1 else end + 2
                    continue
                # regex vs division: look back to the previous significant char.
                j = i - 1
                while j >= 0 and code[j] in " \t\r\n":
                    j -= 1
                prev = code[j] if j >= 0 else ""
                is_value = prev.isalnum() or prev in ")]_$"
                if is_value and prev.isalnum():
                    k = j
                    while k >= 0 and (code[k].isalnum() or code[k] in "_$"):
                        k -= 1
                    if code[k + 1 : j + 1] in CodeSemanticCompressor._REGEX_PRECEDING_KEYWORDS:
                        is_value = False
                if not is_value:  # regex literal -> skip to the unescaped closing '/'
                    i += 1
                    in_class = False
                    while i < n:
                        c = code[i]
                        if c == "\\":
                            i += 2
                            continue
                        if c == "[":
                            in_class = True
                        elif c == "]":
                            in_class = False
                        elif c == "/" and not in_class:
                            i += 1
                            break
                        elif c == "\n":
                            break  # unterminated regex -> bail
                        i += 1
                    continue
                # else: division operator -> fall through and advance one char
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
            i += 1
        return None

    def _chunk_by_lines(
        self, code: str, file_id: str, lines_per_chunk: int = 50
    ) -> List[CodeChunk]:
        """
        Fallback chunking strategy: split by lines when AST parsing fails.
        """
        lines = code.split("\n")
        chunks = []

        for i in range(0, len(lines), lines_per_chunk):
            chunk_lines = lines[i : i + lines_per_chunk]
            chunk_code = "\n".join(chunk_lines)

            chunks.append(
                CodeChunk(
                    chunk_id=f"{file_id}::block_{i // lines_per_chunk}",
                    chunk_type="block",
                    code=chunk_code,
                    name=f"block_{i}",
                    docstring=None,
                    start_line=i + 1,
                    end_line=min(i + lines_per_chunk, len(lines)),
                    dependencies=[],
                )
            )

        return chunks

    @staticmethod
    def _build_dependency_edges(chunks: List[CodeChunk]) -> List[Tuple[str, str]]:
        """Resolve chunk.dependencies into (src_chunk_id, dst_chunk_id) edges.

        Task #236 rank11 — builds a name -> [chunk_id] index once (O(C)), then
        looks each dependency up in O(1), replacing the previous nested
        ``for dep in chunk.dependencies: for other_chunk in chunks`` scan that
        was O(C x D x C). Behaviour-identical to the nested-loop version:
        every chunk whose ``name`` equals a dependency string gets an edge from
        the depending chunk, self-edges are excluded, and duplicate dependency
        mentions collapse to the same edge set (nx.add_edge was already
        idempotent, so de-duplicating here changes nothing downstream).
        """
        name_index: Dict[str, List[str]] = {}
        for chunk in chunks:
            name_index.setdefault(chunk.name, []).append(chunk.chunk_id)

        edges: List[Tuple[str, str]] = []
        seen: set = set()
        for chunk in chunks:
            for dep in chunk.dependencies:
                for other_id in name_index.get(dep, ()):
                    if other_id == chunk.chunk_id:
                        continue
                    edge = (chunk.chunk_id, other_id)
                    if edge not in seen:
                        seen.add(edge)
                        edges.append(edge)
        return edges

    @staticmethod
    def _build_similarity_edges(
        embeddings: np.ndarray,
        chunk_ids: List[str],
        similarity_threshold: float,
        block_size: int = _SIMILARITY_BLOCK_SIZE,
        max_chunks: int = _MAX_GRAPH_CHUNKS,
    ) -> List[Tuple[str, str, float]]:
        """Build the upper-triangle cosine-similarity edge list for code chunks
        without materialising the full N x N similarity matrix.

        Mirrors SemanticCompressor._build_similarity_edges (task #30 OOM fix),
        but explicitly L2-normalises each row before taking block-wise dot
        products. Code embedding models (e.g. the default
        "microsoft/codebert-base") are NOT guaranteed to ship a built-in
        Normalize pooling layer the way many sentence-transformers text models
        do, so a bare dot product would silently diverge from sklearn's
        cosine_similarity() on non-unit-norm vectors. Normalising here keeps
        output IDENTICAL to the pre-fix ``cosine_similarity(embeddings)`` +
        nested-loop approach (up to float precision) while bounding peak memory
        to O(block_size x min(N, max_chunks)) instead of O(N^2).

        Only the first ``max_chunks`` rows participate in edge building; chunks
        beyond that index still exist but remain unconnected via dense
        similarity edges (uniform PageRank fallback). Returns (i_id, j_id,
        weight) triples for i < j only.
        """
        n = len(chunk_ids)
        edge_count = min(n, max_chunks)
        edges: List[Tuple[str, str, float]] = []
        if edge_count < 2:
            return edges

        # L2-normalise once -- O(edge_count x dim), not O(N^2) -- so a plain
        # dot product reproduces cosine similarity exactly regardless of
        # whether the source embeddings were already unit-norm.
        edge_embeddings = np.asarray(embeddings[:edge_count])
        norms = np.linalg.norm(edge_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # guard zero vectors
        normalised = edge_embeddings / norms

        for block_start in range(0, edge_count, block_size):
            block_end = min(block_start + block_size, edge_count)
            block = normalised[block_start:block_end]  # (bs, dim)
            sim_block = block @ normalised.T  # (bs, edge_count)
            for r in range(block_end - block_start):
                i = block_start + r
                for j in range(i + 1, edge_count):
                    sim = float(sim_block[r, j])
                    if sim > similarity_threshold:
                        edges.append((chunk_ids[i], chunk_ids[j], sim))
        return edges

    def ingest_code_file(
        self,
        code: str,
        file_id: str,
        filepath: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Ingest source code file and create semantic graph.

        Args:
            code: Source code content
            file_id: Unique identifier
            filepath: Optional file path for language detection
            metadata: Optional metadata

        Returns:
            Compression statistics
        """
        logger.info(f"Ingesting code file: {file_id}")

        # Detect language
        if filepath:
            language = self.detect_language(filepath)
        else:
            language = CodeLanguage.UNKNOWN

        logger.info(f"  Language: {language.value}")

        # Chunk code based on language
        if language == CodeLanguage.PYTHON:
            chunks = self.chunk_python_code(code, file_id)
        elif language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT]:
            chunks = self.chunk_javascript_code(code, file_id)
        else:
            # Fallback to line-based chunking
            chunks = self._chunk_by_lines(code, file_id)

        logger.info(f"  Created {len(chunks)} code chunks")

        # Generate embeddings
        logger.info("  Generating embeddings...")
        chunk_texts = []
        for chunk in chunks:
            # Combine code + docstring for better semantic representation
            text = chunk.code
            if chunk.docstring:
                text = f"{chunk.docstring}\n\n{text}"
            chunk_texts.append(text)

        embeddings = self.model.encode(chunk_texts, show_progress_bar=True)

        # Build dependency graph
        logger.info("  Building dependency graph...")
        graph = nx.DiGraph()  # Directed graph for code dependencies

        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i]
            self.chunks[chunk.chunk_id] = chunk

            # Add node
            graph.add_node(
                chunk.chunk_id,
                **{
                    "type": chunk.chunk_type,
                    "name": chunk.name,
                    "lines": chunk.end_line - chunk.start_line,
                },
            )

        # Add dependency edges via a name index — O(C + total_deps), not the
        # former O(C x D x C) repeated list-scan (task #236 rank11; ports the
        # #30 indexing discipline to the dependency-edge pass). A function
        # body's ast.Call walk can emit hundreds of dependency names, so the
        # nested rescan of `chunks` per dependency was quadratic in input size.
        for src_id, dst_id in self._build_dependency_edges(chunks):
            graph.add_edge(src_id, dst_id, type="dependency")

        # Add semantic similarity edges.
        # Memory-safety (task #236 rank11 OOM fix, mirrors the #30 fix in
        # semantic_compressor.py): never materialise the full N x N
        # cosine-similarity matrix, and cap the number of chunks that
        # participate in dense edge-building so a pathologically large source
        # file cannot blow up peak RSS. Chunks beyond the cap still exist
        # (uniform PageRank fallback) but are not densely connected.
        n_chunks = len(chunks)
        if n_chunks > 1:
            if n_chunks > _MAX_GRAPH_CHUNKS:
                logger.warning(
                    f"  Code file '{file_id}' has {n_chunks} chunks, exceeding "
                    f"_MAX_GRAPH_CHUNKS={_MAX_GRAPH_CHUNKS}. Similarity edges will "
                    f"only be built for the first {_MAX_GRAPH_CHUNKS} chunks to bound "
                    f"peak memory; remaining chunks receive uniform PageRank."
                )
            edge_chunk_count = min(n_chunks, _MAX_GRAPH_CHUNKS)
            chunk_ids = [c.chunk_id for c in chunks]
            for src, dst, weight in self._build_similarity_edges(
                embeddings=embeddings,
                chunk_ids=chunk_ids,
                similarity_threshold=self.similarity_threshold,
                block_size=_SIMILARITY_BLOCK_SIZE,
                max_chunks=edge_chunk_count,
            ):
                # Undirected similarity edge -- mirrors the pre-fix behaviour of
                # adding both directions on the DiGraph.
                graph.add_edge(src, dst, type="semantic", weight=weight)
                graph.add_edge(dst, src, type="semantic", weight=weight)

        # Calculate importance using PageRank
        logger.info("  Calculating importance scores...")
        if len(graph.nodes) > 0:
            # Convert to undirected for PageRank
            undirected = graph.to_undirected()
            pagerank = nx.pagerank(undirected)

            for chunk_id, score in pagerank.items():
                if chunk_id in self.chunks:
                    self.chunks[chunk_id].importance = score

        # Store graph and metadata
        self.graphs[file_id] = graph
        self.file_metadata[file_id] = metadata or {}

        # Generate statistics
        total_lines = sum(c.end_line - c.start_line for c in chunks)

        stats = {
            "file_id": file_id,
            "language": language.value,
            "total_chunks": len(chunks),
            "total_lines": total_lines,
            "chunk_types": {
                "imports": sum(1 for c in chunks if c.chunk_type == "import"),
                "functions": sum(1 for c in chunks if c.chunk_type == "function"),
                "classes": sum(1 for c in chunks if c.chunk_type == "class"),
                "blocks": sum(1 for c in chunks if c.chunk_type == "block"),
            },
            "graph_nodes": graph.number_of_nodes(),
            "graph_edges": graph.number_of_edges(),
        }

        logger.info(f"  [OK] Parsed {len(chunks)} chunks:")
        logger.info(f"     Functions: {stats['chunk_types']['functions']}")
        logger.info(f"     Classes: {stats['chunk_types']['classes']}")
        logger.info(f"     Dependencies: {graph.number_of_edges()}")

        return stats

    def generate_code_skeleton(self, file_id: str, show_top_n: int = None) -> str:
        """
        Generate code skeleton showing high-importance functions/classes.

        Similar to semantic skeleton but code-aware:
        - Shows import statements
        - Shows function signatures (not bodies)
        - Shows class definitions (not implementations)
        - Indicates dependencies
        """
        graph = self.graphs.get(file_id)
        if not graph:
            raise ValueError(f"File {file_id} not found")

        # Get chunks sorted by importance
        file_chunks = [(cid, self.chunks[cid]) for cid in graph.nodes() if cid.startswith(file_id)]
        file_chunks.sort(key=lambda x: x[1].importance, reverse=True)

        # Determine how many to show
        if show_top_n is None:
            show_top_n = max(1, len(file_chunks) // 4)  # Top 25%

        skeleton_lines = []
        skeleton_lines.append(f"=== CODE SKELETON: {file_id} ===")
        skeleton_lines.append(f"Total chunks: {len(file_chunks)} | Showing: {show_top_n}")
        skeleton_lines.append("")

        # Always show imports first
        import_chunks = [c for cid, c in file_chunks if c.chunk_type == "import"]
        if import_chunks:
            skeleton_lines.append("[IMPORTS]")
            for chunk in import_chunks:
                skeleton_lines.append(f"   {', '.join(chunk.dependencies[:5])}")
                if len(chunk.dependencies) > 5:
                    skeleton_lines.append(f"   ... and {len(chunk.dependencies) - 5} more")
            skeleton_lines.append("")

        # Show top N important chunks. SELECTION is by importance (top-N of the
        # importance-sorted file_chunks); RENDER order is document/source order
        # (world-class compression audit #2, 2026-07-07 — same fix as the text
        # skeleton in semantic_compressor.py). Rendering code in importance order
        # scrambles structure: a helper can print above the function that uses it,
        # or after an [IMPORTS] block that references it. start_line is set on
        # every chunk at ingest; the (start_line, name) tiebreak is deterministic.
        top_chunks = [c for cid, c in file_chunks[:show_top_n] if c.chunk_type != "import"]
        # (start_line, name, chunk_id) is a TOTAL order — chunk_id breaks any
        # start_line+name tie deterministically instead of falling back to the
        # importance order the fix is replacing (codex 2026-07-07).
        top_chunks.sort(key=lambda c: (c.start_line, c.name, c.chunk_id))

        for chunk in top_chunks:
            if chunk.chunk_type == "function":
                # Extract function signature
                signature = (
                    chunk.code.split("{")[0].strip()
                    if "{" in chunk.code
                    else chunk.code.split("\n")[0]
                )

                skeleton_lines.append(
                    f"[FUNCTION] {chunk.name} (importance: {chunk.importance:.3f})"
                )
                skeleton_lines.append(f"   Signature: {signature}")
                if chunk.docstring:
                    # First line of docstring
                    first_doc_line = chunk.docstring.strip().split("\n")[0].strip()
                    skeleton_lines.append(f"   Doc: {first_doc_line[:80]}...")
                skeleton_lines.append(f"   Lines: {chunk.start_line}-{chunk.end_line}")
                skeleton_lines.append("")

            elif chunk.chunk_type == "class":
                skeleton_lines.append(f"[CLASS] {chunk.name} (importance: {chunk.importance:.3f})")
                if chunk.docstring:
                    first_doc_line = chunk.docstring.strip().split("\n")[0].strip()
                    skeleton_lines.append(f"   Doc: {first_doc_line[:80]}...")
                skeleton_lines.append(f"   Methods: {', '.join(chunk.dependencies[:5])}")
                if len(chunk.dependencies) > 5:
                    skeleton_lines.append(f"   ... and {len(chunk.dependencies) - 5} more methods")
                skeleton_lines.append(f"   Lines: {chunk.start_line}-{chunk.end_line}")
                skeleton_lines.append("")

        # Show hidden chunks
        hidden_count = len(file_chunks) - show_top_n - len(import_chunks)
        if hidden_count > 0:
            skeleton_lines.append(f"[HIDDEN] {hidden_count} additional chunks hidden")
            skeleton_lines.append("   Use search_code() or modulate_code() to retrieve them")

        return "\n".join(skeleton_lines)

    def search_code(
        self, query: str, file_id: Optional[str] = None, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Semantic code search.

        Args:
            query: Natural language query or code snippet
            file_id: Optional file to search within
            top_k: Number of results

        Returns:
            List of (chunk_id, similarity_score) tuples
        """
        query_embedding = self.model.encode([query])[0]

        candidates = []
        for chunk_id, chunk in self.chunks.items():
            if file_id and not chunk_id.startswith(file_id):
                continue

            from sklearn.metrics.pairwise import cosine_similarity

            similarity = cosine_similarity([query_embedding], [chunk.embedding])[0][0]

            candidates.append((chunk_id, float(similarity)))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def get_code_chunk(self, chunk_id: str, include_context: bool = False) -> str:
        """
        Retrieve full code for a specific chunk.

        Args:
            chunk_id: Chunk identifier
            include_context: Include surrounding chunks for context

        Returns:
            Code content
        """
        chunk = self.chunks.get(chunk_id)
        if not chunk:
            return f"Chunk {chunk_id} not found"

        output = []
        output.append(f"=== {chunk.chunk_type.upper()}: {chunk.name} ===")
        output.append(f"Lines: {chunk.start_line}-{chunk.end_line}")

        if chunk.docstring:
            output.append("\nDocumentation:")
            output.append(chunk.docstring)

        output.append("\nCode:")
        output.append(chunk.code)

        if chunk.dependencies:
            output.append(f"\nDependencies: {', '.join(chunk.dependencies[:10])}")

        return "\n".join(output)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("CODE-SPECIFIC SEMANTIC COMPRESSION DEMO")
    print("=" * 70)

    # Sample Python code
    sample_code = '''
import numpy as np
from sklearn.metrics import accuracy_score

def preprocess_data(data, normalize=True):
    """
    Preprocess input data for model training.

    Args:
        data: Input data array
        normalize: Whether to normalize the data

    Returns:
        Preprocessed data
    """
    if normalize:
        data = (data - np.mean(data)) / np.std(data)
    return data

class NeuralNetwork:
    """
    Simple neural network implementation.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        self.weights1 = np.random.randn(input_dim, hidden_dim)
        self.weights2 = np.random.randn(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass through the network"""
        hidden = np.dot(x, self.weights1)
        output = np.dot(hidden, self.weights2)
        return output

    def train(self, x, y, epochs=100):
        """Train the network"""
        for epoch in range(epochs):
            predictions = self.forward(x)
            loss = np.mean((predictions - y) ** 2)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

def evaluate_model(model, test_data, test_labels):
    """Evaluate model performance on test set"""
    predictions = model.forward(test_data)
    accuracy = accuracy_score(test_labels, predictions > 0.5)
    return accuracy
'''

    # Initialize code compressor
    compressor = CodeSemanticCompressor()

    # Ingest code
    stats = compressor.ingest_code_file(
        code=sample_code, file_id="neural_net", filepath="neural_net.py"
    )

    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Generate skeleton
    print("\n" + "=" * 70)
    print("CODE SKELETON")
    print("=" * 70)
    skeleton = compressor.generate_code_skeleton("neural_net")
    print(skeleton)

    # Search code
    print("\n" + "=" * 70)
    print("SEMANTIC CODE SEARCH")
    print("=" * 70)
    query = "How do I train the model?"
    results = compressor.search_code(query, "neural_net", top_k=3)

    print(f"\nQuery: '{query}'")
    print("\nResults:")
    for chunk_id, score in results:
        chunk = compressor.chunks[chunk_id]
        print(f"  {chunk_id} (score: {score:.3f}): {chunk.chunk_type} '{chunk.name}'")
