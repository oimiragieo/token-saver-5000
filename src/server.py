#!/usr/bin/env python3
"""
Semantic Modulator MCP Server

Exposes adaptive semantic fidelity tools to AI agents via MCP protocol.

Architecture:
- Local semantic compression (no external API calls)
- Graph-based structure preservation
- Adaptive fidelity levels
- Blind spot detection for self-correction
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    EmbeddedResource,
)

from .semantic_compressor import SemanticCompressor, FidelityLevel
from .blind_spot_detector import BlindSpotDetector, HaloEffectDetector
from .adaptive_rate_allocator import (
    AdaptiveRateAllocator,
    ContextWindowAdapter,
    MultiLevelSemanticEncoder
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("semantic-modulator")


class SemanticModulatorServer:
    """MCP Server for Semantic Modulation"""

    def __init__(self):
        self.server = Server("semantic-modulator")
        self.compressor = SemanticCompressor(
            model_name="all-MiniLM-L6-v2",
            similarity_threshold=0.75,
            skeleton_ratio=0.2,
        )
        self.blind_spot_detector = BlindSpotDetector(self.compressor)
        self.halo_detector = HaloEffectDetector(self.compressor)

        # JSCCM-inspired adaptive components
        self.context_window_adapter = ContextWindowAdapter(self.compressor)
        self.multilevel_encoder = MultiLevelSemanticEncoder(self.compressor)

        # Context window monitoring (like SNR in JSCCM)
        self.context_window_monitor = {
            'max_tokens': 100000,  # Typical context window size
            'used_tokens': 0,
            'history': []
        }

        # Track what the AI has retrieved (for blind spot detection)
        self.retrieval_history: Dict[str, List[str]] = {}

        self._setup_handlers()

    def _setup_handlers(self):
        """Register MCP tool handlers"""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available semantic modulation tools"""
            return [
                Tool(
                    name="ingest_context",
                    description=(
                        "Ingest and compress a document into a semantic graph. "
                        "This creates a fidelity-preserving encoding that reduces token usage by 80-95%. "
                        "The document is analyzed for structure, relationships, and importance. "
                        "Returns a compressed skeleton view."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The raw document text to ingest"
                            },
                            "file_id": {
                                "type": "string",
                                "description": "Unique identifier for this document (e.g., 'paper_1', 'manual_v2')"
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Optional metadata (author, date, source, etc.)",
                                "properties": {
                                    "author": {"type": "string"},
                                    "date": {"type": "string"},
                                    "source": {"type": "string"},
                                    "tags": {"type": "array", "items": {"type": "string"}},
                                },
                            },
                        },
                        "required": ["text", "file_id"],
                    },
                ),
                Tool(
                    name="read_skeleton",
                    description=(
                        "Read the compressed skeleton view of a previously ingested document. "
                        "Shows high-importance 'anchor' concepts with summaries, and lists "
                        "other sections as expandable nodes. Achieves 80-95% token reduction. "
                        "Use this FIRST before requesting specific details."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_id": {
                                "type": "string",
                                "description": "The document identifier"
                            },
                        },
                        "required": ["file_id"],
                    },
                ),
                Tool(
                    name="modulate_region",
                    description=(
                        "Retrieve specific sections at a chosen fidelity level. "
                        "Use this to 'zoom in' on relevant parts after reading the skeleton. "
                        "Fidelity levels: "
                        "'ABSTRACT' (~10 tokens/node), "
                        "'STRUCTURE' (~50 tokens/node), "
                        "'RAW' (full content). "
                        "This implements adaptive semantic fidelity."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "node_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of node IDs to retrieve (from skeleton)"
                            },
                            "fidelity_level": {
                                "type": "string",
                                "enum": ["ABSTRACT", "STRUCTURE", "RAW"],
                                "description": "Detail level to retrieve",
                                "default": "RAW"
                            },
                        },
                        "required": ["node_ids"],
                    },
                ),
                Tool(
                    name="search_semantic",
                    description=(
                        "Semantic search across ingested documents. "
                        "Uses vector similarity to find relevant sections, "
                        "even if exact keywords don't match. "
                        "Returns ranked node IDs."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language search query"
                            },
                            "file_id": {
                                "type": "string",
                                "description": "Optional: limit search to specific document"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to return",
                                "default": 5
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="check_blind_spots",
                    description=(
                        "🔍 BLIND SPOT DETECTOR: Analyze if your response missed critical context. "
                        "This tool embeds your response and compares it to ALL nodes in the document. "
                        "If relevant content was not retrieved, it alerts you and suggests auto-injection. "
                        "Use AFTER generating a response to ensure fidelity. "
                        "This implements the 'Self-Correcting Context Loop'."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "ai_response": {
                                "type": "string",
                                "description": "The response you generated"
                            },
                            "file_id": {
                                "type": "string",
                                "description": "Which document was being discussed"
                            },
                            "retrieved_nodes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Which node IDs you actually retrieved/viewed"
                            },
                        },
                        "required": ["ai_response", "file_id", "retrieved_nodes"],
                    },
                ),
                Tool(
                    name="detect_hallucination",
                    description=(
                        "🛡️ HALLUCINATION DETECTOR: Check if a response is grounded in source material. "
                        "Compares response embedding to document graph. "
                        "Flags responses with low similarity to all nodes (possible fabrication). "
                        "Use when uncertain about answer accuracy."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "ai_response": {
                                "type": "string",
                                "description": "The response to validate"
                            },
                            "file_id": {
                                "type": "string",
                                "description": "The source document"
                            },
                        },
                        "required": ["ai_response", "file_id"],
                    },
                ),
                Tool(
                    name="get_stats",
                    description=(
                        "Get statistics about ingested documents. "
                        "Shows token counts, compression ratios, and graph structure. "
                        "Useful for understanding the semantic compression efficiency."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_id": {
                                "type": "string",
                                "description": "Optional: specific file ID, or omit for global stats"
                            },
                        },
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> List[TextContent]:
            """Handle tool calls"""
            try:
                if name == "ingest_context":
                    result = self._handle_ingest(arguments)
                elif name == "read_skeleton":
                    result = self._handle_read_skeleton(arguments)
                elif name == "modulate_region":
                    result = self._handle_modulate_region(arguments)
                elif name == "search_semantic":
                    result = self._handle_search_semantic(arguments)
                elif name == "check_blind_spots":
                    result = self._handle_check_blind_spots(arguments)
                elif name == "detect_hallucination":
                    result = self._handle_detect_hallucination(arguments)
                elif name == "get_stats":
                    result = self._handle_get_stats(arguments)
                else:
                    result = f"Unknown tool: {name}"

                return [TextContent(type="text", text=str(result))]

            except Exception as e:
                logger.error(f"Error in {name}: {e}", exc_info=True)
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    def _handle_ingest(self, args: Dict) -> str:
        """Handle ingest_context tool call"""
        text = args["text"]
        file_id = args["file_id"]
        metadata = args.get("metadata")

        logger.info(f"Ingesting document: {file_id}")

        skeleton = self.compressor.ingest_file(text, file_id, metadata)

        # Initialize retrieval history
        self.retrieval_history[file_id] = []

        result = f"""
✅ Document ingested successfully!

File ID: {file_id}
Total nodes: {skeleton.total_nodes}
Original tokens: {skeleton.total_tokens:,}
Skeleton tokens: {skeleton.skeleton_tokens:,}
Compression ratio: {skeleton.compression_ratio:.1f}x

📊 Token savings: {skeleton.total_tokens - skeleton.skeleton_tokens:,} tokens ({(1 - skeleton.skeleton_tokens/skeleton.total_tokens)*100:.1f}%)

Next steps:
1. Use read_skeleton('{file_id}') to view the compressed structure
2. Use modulate_region() to retrieve specific sections
3. Use search_semantic() to find relevant content
4. Use check_blind_spots() after generating responses to ensure completeness

{skeleton.skeleton_text[:500]}...
(Use read_skeleton to see full structure)
"""
        return result

    def _handle_read_skeleton(self, args: Dict) -> str:
        """Handle read_skeleton tool call"""
        file_id = args["file_id"]
        logger.info(f"Reading skeleton: {file_id}")

        skeleton_text = self.compressor.read_skeleton(file_id)
        return skeleton_text

    def _handle_modulate_region(self, args: Dict) -> str:
        """Handle modulate_region tool call"""
        node_ids = args["node_ids"]
        fidelity_str = args.get("fidelity_level", "RAW")

        # Convert string to enum
        fidelity = FidelityLevel[fidelity_str]

        logger.info(f"Modulating {len(node_ids)} nodes at {fidelity_str} fidelity")

        # Track retrieval for blind spot detection
        for node_id in node_ids:
            # Extract file_id from node_id (format: file_id_n123)
            file_id = "_".join(node_id.split("_")[:-1])
            if file_id not in self.retrieval_history:
                self.retrieval_history[file_id] = []
            if node_id not in self.retrieval_history[file_id]:
                self.retrieval_history[file_id].append(node_id)

        result = self.compressor.modulate_region(node_ids, fidelity)
        return result

    def _handle_search_semantic(self, args: Dict) -> str:
        """Handle search_semantic tool call"""
        query = args["query"]
        file_id = args.get("file_id")
        top_k = args.get("top_k", 5)

        logger.info(f"Semantic search: '{query}' in {file_id or 'all files'}")

        node_ids = self.compressor.search_semantic(query, file_id, top_k)

        result_lines = [f"🔍 Semantic Search Results for: '{query}'"]
        result_lines.append(f"Found {len(node_ids)} relevant nodes:\n")

        for i, node_id in enumerate(node_ids, 1):
            node = self.compressor.chunks[node_id]
            summary = self.compressor._generate_summary(node.text, max_length=100)
            result_lines.append(f"{i}. [{node_id}]")
            result_lines.append(f"   Importance: {node.importance:.3f}")
            result_lines.append(f"   Summary: {summary}\n")

        result_lines.append(f"💡 Tip: Use modulate_region({node_ids[:3]}) to retrieve full content")

        return "\n".join(result_lines)

    def _handle_check_blind_spots(self, args: Dict) -> str:
        """Handle check_blind_spots tool call"""
        ai_response = args["ai_response"]
        file_id = args["file_id"]
        retrieved_nodes = args["retrieved_nodes"]

        logger.info(f"Checking blind spots for response about {file_id}")

        report = self.blind_spot_detector.analyze_response(
            ai_response, file_id, retrieved_nodes
        )

        result = self.blind_spot_detector.format_report(report)

        # If critical blind spots found, auto-suggest retrieval
        if report.auto_inject:
            result += f"\n\n🔧 AUTO-CORRECTION SUGGESTED:\n"
            result += f"Retrieve these nodes: {report.auto_inject}\n"
            result += f"Command: modulate_region({report.auto_inject}, 'RAW')"

        return result

    def _handle_detect_hallucination(self, args: Dict) -> str:
        """Handle detect_hallucination tool call"""
        ai_response = args["ai_response"]
        file_id = args["file_id"]

        logger.info(f"Checking for hallucination in response about {file_id}")

        is_hallucinating, warnings = self.halo_detector.detect_hallucination(
            ai_response, file_id
        )

        if is_hallucinating:
            result = "🚨 HALLUCINATION ALERT 🚨\n\n"
            result += "The response may contain fabricated information:\n"
            for warning in warnings:
                result += f"  • {warning}\n"
            result += "\nRecommendation: Re-examine source material and regenerate response."
        else:
            result = "✅ Response appears grounded in source material.\n"
            result += "No hallucination detected."

        return result

    def _handle_get_stats(self, args: Dict) -> str:
        """Handle get_stats tool call"""
        file_id = args.get("file_id")

        stats = self.compressor.get_stats(file_id)

        if file_id:
            result = f"""
📊 Document Statistics: {file_id}

Total nodes: {stats['total_nodes']}
Total edges: {stats['total_edges']}
Original tokens: {stats['total_tokens']:,}
Skeleton tokens: {stats['skeleton_tokens']:,}
Compression ratio: {stats['compression_ratio']:.1f}x

Token savings: {stats['total_tokens'] - stats['skeleton_tokens']:,} ({(1 - stats['skeleton_tokens']/stats['total_tokens'])*100:.1f}%)

Metadata: {json.dumps(stats['metadata'], indent=2)}
"""
        else:
            result = f"""
📊 Global Statistics

Total files ingested: {stats['total_files']}
Total nodes: {stats['total_nodes']}

Files: {', '.join(stats['files'])}
"""

        return result

    async def run(self):
        """Run the MCP server"""
        logger.info("🚀 Starting Semantic Modulator MCP Server")
        logger.info("   Combining Semantic Communication + Fidelity-Preserving Encoding")
        logger.info("   Model: all-MiniLM-L6-v2 (local)")
        logger.info("   Mode: Adaptive Semantic Fidelity\n")

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


def main():
    """Entry point"""
    server = SemanticModulatorServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
