"""Split semantic_compressor into types + ingest + retrieval mixins."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SC = ROOT / "src/semantic_compressor.py"
lines = SC.read_text(encoding="utf-8").splitlines(keepends=True)

# Line indices (0-based): ingest starts 1130, retrieval starts 2047
ingest_start, retrieval_start = 1130, 2047

header = lines[:43]  # through logger = ...
types_body = lines[43:237]  # constants through compute_adaptive_ratio
core_body = lines[237:ingest_start]  # class declaration through _encode_async
ingest_body = lines[ingest_start:retrieval_start]
retrieval_body = lines[retrieval_start:]

types_path = ROOT / "src/semantic_compressor_types.py"
types_content = (
    '"""Types, constants, and helpers for semantic compression."""\n\n'
    + "".join(header)
    + "".join(types_body)
)
types_path.write_text(types_content, encoding="utf-8", newline="\n")

ingest_path = ROOT / "src/semantic_compressor_ingest.py"
ingest_content = (
    '"""Ingestion and skeleton generation mixin for SemanticCompressor."""\n\n'
    "from typing import Dict, List, Optional, Tuple\n\n"
    "import numpy as np\n"
    "import networkx as nx\n\n"
    "from .semantic_compressor_types import DiffReingestionResult, FidelityLevel, SkeletonResponse\n\n"
    "class SemanticCompressorIngestMixin:\n"
    + "".join(ingest_body)
)
ingest_path.write_text(ingest_content, encoding="utf-8", newline="\n")

retrieval_path = ROOT / "src/semantic_compressor_retrieval.py"
retrieval_content = (
    '"""Read, search, and evidence retrieval mixin for SemanticCompressor."""\n\n'
    "from typing import Dict, List, Optional, Tuple\n\n"
    "import numpy as np\n\n"
    "from .semantic_compressor_types import DiffReingestionResult, EvidenceResult, FidelityLevel\n\n"
    "class SemanticCompressorRetrievalMixin:\n"
    + "".join(retrieval_body)
)
retrieval_path.write_text(retrieval_content, encoding="utf-8", newline="\n")

main_content = (
    '"""Fidelity-Preserving Semantic Compressor (facade + core graph/chunking)."""\n\n'
    "from .semantic_compressor_types import (\n"
    "    DiffReingestionResult,\n"
    "    EvidenceResult,\n"
    "    FidelityLevel,\n"
    "    SemanticNode,\n"
    "    SkeletonResponse,\n"
    "    compute_adaptive_ratio,\n"
    ")\n"
    "from .semantic_compressor_ingest import SemanticCompressorIngestMixin\n"
    "from .semantic_compressor_retrieval import SemanticCompressorRetrievalMixin\n\n"
    + "".join(core_body).replace(
        "class SemanticCompressor:",
        "class SemanticCompressor(SemanticCompressorIngestMixin, SemanticCompressorRetrievalMixin):",
        1,
    )
)
SC.write_text(main_content, encoding="utf-8", newline="\n")
print(
    "semantic_compressor split:",
    len(lines),
    "types",
    len(types_content.splitlines()),
    "ingest",
    len(ingest_content.splitlines()),
    "retrieval",
    len(retrieval_content.splitlines()),
    "main",
    len(main_content.splitlines()),
)
