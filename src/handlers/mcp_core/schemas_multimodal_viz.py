"""Tool schemas: Multimodal (mmh) + Visualization (vh). Split from mcp_core.py (N2 slice 2).

Each list is verbatim `Tool(...)` schema literals moved from the original
monolithic `setup_mcp_tools`, unchanged.
"""

from mcp.types import Tool

from ._constants import SCOPE_PROPERTIES

VISUALIZATION_TOOLS: list = [
    Tool(
        name="export_graph_json",
        description=(
            "[STATS] Export semantic graph as JSON for programmatic access. "
            "Returns a structured JSON representation of the semantic graph with nodes, edges, "
            "importance scores, and statistics. Perfect for custom analysis or integration with "
            "other tools."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "Document ID to export",
                },
                "max_nodes": {
                    "type": "integer",
                    "description": "Maximum nodes to include (default: 50)",
                    "minimum": 1,
                },
                "min_importance": {
                    "type": "number",
                    "description": "Minimum importance score to include (default: 0.0)",
                    "minimum": 0.0,
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["file_id"],
        },
    ),
    Tool(
        name="visualize_graph_html",
        description=(
            "[VIZ] Generate interactive HTML visualization of the semantic graph. "
            "Creates a beautiful, interactive web page with draggable nodes, zoom/pan, "
            "color-coded importance, and edge weights. Great for exploring and presenting "
            "compression decisions. Requires pyvis library."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "Document ID to visualize",
                },
                "output_path": {
                    "type": "string",
                    "description": "Path to save HTML file (e.g., 'graph.html')",
                },
                "max_nodes": {
                    "type": "integer",
                    "description": "Maximum nodes to visualize (default: 50)",
                    "minimum": 1,
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["file_id", "output_path"],
        },
    ),
    Tool(
        name="export_graph_graphml",
        description=(
            "[VIZ] Export semantic graph as GraphML for analysis tools. "
            "GraphML is a standard XML format supported by Gephi, Cytoscape, igraph, "
            "and NetworkX. Perfect for advanced network analysis, visualization, "
            "and research workflows."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "Document ID to export",
                },
                "output_path": {
                    "type": "string",
                    "description": "Path to save GraphML file (e.g., 'graph.graphml')",
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["file_id", "output_path"],
        },
    ),
    Tool(
        name="explain_compression_decision",
        description=(
            "[ANALYZE] Explain why a specific node was kept or dropped during compression. "
            "Provides detailed analysis including importance score ranking, connectivity, "
            "key entities, and relationships with other nodes. Perfect for understanding "
            "and debugging compression decisions."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "Document ID",
                },
                "node_id": {
                    "type": "string",
                    "description": "Node ID to explain (e.g., 'quantum_paper_n3')",
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["file_id", "node_id"],
        },
    ),
]


MULTIMODAL_TOOLS: list = [
    Tool(
        name="ingest_multimodal",
        description=(
            "Production-grade multimodal ingestion for text, code, images, audio transcripts, "
            "and document-with-images bundles."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "Logical multimodal document identifier",
                },
                "text_content": {"type": "string", "description": "Optional text content"},
                "code_content": {"type": "string", "description": "Optional code content"},
                "code_language": {
                    "type": "string",
                    "description": "Optional code language label",
                },
                "image_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional local image paths validated for security",
                },
                "image_captions": {
                    "type": "object",
                    "description": "Optional mapping from submitted image path to caption text",
                },
                "image_ocr_text": {
                    "type": "object",
                    "description": "Optional mapping from submitted image path to OCR text",
                },
                "audio_items": {
                    "type": "array",
                    "description": "Optional transcript-backed audio payloads",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "path": {"type": "string"},
                            "transcript": {"type": "string"},
                        },
                        "required": ["transcript"],
                    },
                },
                "document_items": {
                    "type": "array",
                    "description": "Optional document-with-images bundles",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "text": {"type": "string"},
                            "image_paths": {"type": "array", "items": {"type": "string"}},
                            "image_captions": {"type": "object"},
                            "image_ocr_text": {"type": "object"},
                        },
                    },
                },
                "video_items": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Currently unsupported and rejected explicitly",
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["doc_id"],
        },
    ),
    Tool(
        name="search_multimodal",
        description="Search a production multimodal project using text, code, or image queries.",
        inputSchema={
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "Logical multimodal document identifier",
                },
                "query": {"type": "string", "description": "Text or code query"},
                "query_type": {
                    "type": "string",
                    "enum": ["text", "code", "image"],
                    "description": "Query modality",
                    "default": "text",
                },
                "image_query_path": {
                    "type": "string",
                    "description": "Required when query_type=image",
                },
                "top_k": {"type": "integer", "description": "Result count", "default": 5},
                "filter_modality": {
                    "type": "string",
                    "enum": ["text", "code", "image"],
                    "description": "Optional result modality filter",
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["doc_id"],
        },
    ),
]
