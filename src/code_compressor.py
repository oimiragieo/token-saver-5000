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
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import numpy as np
import networkx as nx

from .embeddings import EmbeddingManager
from .semantic_compressor import _MAX_GRAPH_CHUNKS, _SIMILARITY_BLOCK_SIZE

logger = logging.getLogger(__name__)


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
        except SyntaxError:
            logger.warning(f"Syntax error in {file_id}, falling back to line-based chunking")
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
                # #195: qualify methods by their enclosing class (A.foo vs B.foo).
                _parent = getattr(node, "_tsk_parent", None)
                _qualified_name = (
                    f"{_parent.name}.{node.name}"
                    if isinstance(_parent, ast.ClassDef)
                    else node.name
                )
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
                _class_parent = getattr(node, "_tsk_parent", None)
                _qualified_class_name = (
                    f"{_class_parent.name}.{node.name}"
                    if isinstance(_class_parent, ast.ClassDef)
                    else node.name
                )

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

        # Extract functions (simplified)
        func_pattern = r"(?:function|const|let|var)\s+(\w+)\s*=?\s*(?:\([^)]*\)|async\s*\([^)]*\))\s*(?:=>)?\s*\{([^}]*)\}"

        for match in re.finditer(func_pattern, code, re.MULTILINE | re.DOTALL):
            func_name = match.group(1)
            match.group(2)

            # Extract docstring (JSDoc)
            jsdoc_pattern = r"/\*\*([^*]|\*(?!/))*\*/"
            jsdoc_match = re.search(jsdoc_pattern, code[: match.start()])
            docstring = jsdoc_match.group(0) if jsdoc_match else None

            chunks.append(
                CodeChunk(
                    chunk_id=f"{file_id}::{func_name}",
                    chunk_type="function",
                    code=match.group(0),
                    name=func_name,
                    docstring=docstring,
                    start_line=code[: match.start()].count("\n") + 1,
                    end_line=code[: match.end()].count("\n") + 1,
                    dependencies=[],
                )
            )

        return chunks

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

        print(f"  Language: {language.value}")

        # Chunk code based on language
        if language == CodeLanguage.PYTHON:
            chunks = self.chunk_python_code(code, file_id)
        elif language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT]:
            chunks = self.chunk_javascript_code(code, file_id)
        else:
            # Fallback to line-based chunking
            chunks = self._chunk_by_lines(code, file_id)

        print(f"  Created {len(chunks)} code chunks")

        # Generate embeddings
        print("  Generating embeddings...")
        chunk_texts = []
        for chunk in chunks:
            # Combine code + docstring for better semantic representation
            text = chunk.code
            if chunk.docstring:
                text = f"{chunk.docstring}\n\n{text}"
            chunk_texts.append(text)

        embeddings = self.model.encode(chunk_texts, show_progress_bar=True)

        # Build dependency graph
        print("  Building dependency graph...")
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

            # Add dependency edges
            for dep in chunk.dependencies:
                # Find chunks with matching names
                for other_chunk in chunks:
                    if other_chunk.name == dep and other_chunk.chunk_id != chunk.chunk_id:
                        graph.add_edge(chunk.chunk_id, other_chunk.chunk_id, type="dependency")

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
        print("  Calculating importance scores...")
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

        print(f"  [OK] Parsed {len(chunks)} chunks:")
        print(f"     Functions: {stats['chunk_types']['functions']}")
        print(f"     Classes: {stats['chunk_types']['classes']}")
        print(f"     Dependencies: {graph.number_of_edges()}")

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

        # Show top N important chunks
        top_chunks = [c for cid, c in file_chunks[:show_top_n] if c.chunk_type != "import"]

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
