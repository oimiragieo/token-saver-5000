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
    MultiLevelSemanticEncoder,
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
            "max_tokens": 100000,  # Typical context window size
            "used_tokens": 0,
            "history": [],
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
                                "description": "The raw document text to ingest",
                            },
                            "file_id": {
                                "type": "string",
                                "description": "Unique identifier for this document (e.g., 'paper_1', 'manual_v2')",
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Optional metadata (author, date, source, etc.)",
                                "properties": {
                                    "author": {"type": "string"},
                                    "date": {"type": "string"},
                                    "source": {"type": "string"},
                                    "tags": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
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
                                "description": "The document identifier",
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
                        "5 Fidelity levels (JSCCM-inspired adaptive modulation): "
                        "'ABSTRACT' (~10 tokens/node) - Quick summary only, "
                        "'OUTLINE' (~30 tokens/node) - Summary + section context, "
                        "'STRUCTURE' (~50 tokens/node) - Summary + entities + metadata, "
                        "'DETAILED' (~100 tokens/node) - Summary + entities + key excerpts, "
                        "'RAW' (variable tokens) - Full original content. "
                        "This implements adaptive semantic fidelity - choose lower levels when context is tight."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "node_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of node IDs to retrieve (from skeleton)",
                            },
                            "fidelity_level": {
                                "type": "string",
                                "enum": [
                                    "ABSTRACT",
                                    "OUTLINE",
                                    "STRUCTURE",
                                    "DETAILED",
                                    "RAW",
                                ],
                                "description": "Detail level to retrieve (default: RAW for maximum fidelity)",
                                "default": "RAW",
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
                                "description": "Natural language search query",
                            },
                            "file_id": {
                                "type": "string",
                                "description": "Optional: limit search to specific document",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to return",
                                "default": 5,
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
                                "description": "The response you generated",
                            },
                            "file_id": {
                                "type": "string",
                                "description": "Which document was being discussed",
                            },
                            "retrieved_nodes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Which node IDs you actually retrieved/viewed",
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
                                "description": "The response to validate",
                            },
                            "file_id": {
                                "type": "string",
                                "description": "The source document",
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
                                "description": "Optional: specific file ID, or omit for global stats",
                            },
                        },
                    },
                ),
                Tool(
                    name="adapt_to_context_window",
                    description=(
                        "🔧 ADAPTIVE CONTEXT ALLOCATION (JSCCM-inspired): "
                        "Dynamically adjust compression based on available context window. "
                        "Low availability (like low SNR in wireless) → More compression. "
                        "High availability → Less compression, more detail. "
                        "Uses learned rate allocator to determine optimal skeleton ratio. "
                        "Inspired by JSCCM paper's channel adaptation strategy."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_id": {
                                "type": "string",
                                "description": "Document to generate adaptive skeleton for",
                            },
                            "available_tokens": {
                                "type": "integer",
                                "description": "How many tokens are currently available in context window",
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Maximum context window size (default: 100000)",
                                "default": 100000,
                            },
                            "query_priority": {
                                "type": "number",
                                "description": "Query importance (0-1, default: 0.5)",
                                "default": 0.5,
                            },
                        },
                        "required": ["file_id", "available_tokens"],
                    },
                ),
                Tool(
                    name="multilevel_encode",
                    description=(
                        "📊 MULTI-LEVEL ENCODING (JSCCM-inspired): "
                        "Generate skeleton with 3 priority levels: "
                        "• Main branch (top 15%, always included) - critical concepts "
                        "• Auxiliary branch (next 25%, include if space allows) - important details "
                        "• Detail branch (remaining, only if plenty of space) - supplementary content. "
                        "Progressively adds levels based on available context window. "
                        "Inspired by JSCCM's parallel encoder architecture."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_id": {
                                "type": "string",
                                "description": "Document to encode",
                            },
                            "available_tokens": {
                                "type": "integer",
                                "description": "Available context window tokens",
                            },
                        },
                        "required": ["file_id", "available_tokens"],
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
                elif name == "adapt_to_context_window":
                    result = self._handle_adapt_to_context_window(arguments)
                elif name == "multilevel_encode":
                    result = self._handle_multilevel_encode(arguments)
                else:
                    result = f"Unknown tool: {name}"

                return [TextContent(type="text", text=str(result))]

            except Exception as e:
                logger.error(f"Error in {name}: {e}", exc_info=True)
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    def _validate_file_id(self, file_id: str, must_exist: bool = True) -> None:
        """Validate file_id and provide helpful error messages"""
        if not file_id:
            raise ValueError("file_id cannot be empty")

        if must_exist:
            if file_id not in self.compressor.chunks:
                available = list(set([nid.split("_n")[0] for nid in self.compressor.chunks.keys()]))
                raise ValueError(
                    f"Document '{file_id}' not found. "
                    f"Available documents: {available if available else '(none)'}\n"
                    f"💡 Tip: Use ingest_context() to add documents first."
                )

    def _validate_node_ids(self, node_ids: List[str]) -> None:
        """Validate node_ids and provide helpful suggestions"""
        if not node_ids:
            raise ValueError("node_ids list cannot be empty")

        invalid_nodes = [nid for nid in node_ids if nid not in self.compressor.chunks]
        if invalid_nodes:
            # Extract file_id from first node to give better error message
            file_id = node_ids[0].rsplit("_n", 1)[0] if "_n" in node_ids[0] else "unknown"
            valid_nodes = [nid for nid in self.compressor.chunks.keys() if nid.startswith(file_id)]

            raise ValueError(
                f"Invalid node IDs: {invalid_nodes[:3]}\n"
                f"💡 Tip: Use read_skeleton('{file_id}') to see valid node IDs.\n"
                f"   Valid nodes for '{file_id}': {valid_nodes[:5]}..."
                if valid_nodes
                else f"   No nodes found for '{file_id}'. Document may not be ingested."
            )

    def _validate_token_count(self, available_tokens: int, max_tokens: int = None) -> None:
        """Validate token counts"""
        if available_tokens < 0:
            raise ValueError(f"available_tokens must be non-negative, got {available_tokens}")

        if available_tokens == 0:
            raise ValueError(
                "available_tokens is 0 - no space for content!\n"
                "💡 Tip: Provide a positive number (e.g., 10000 for 10k tokens available)"
            )

        if max_tokens is not None and available_tokens > max_tokens:
            raise ValueError(
                f"available_tokens ({available_tokens}) exceeds max_tokens ({max_tokens})\n"
                "💡 Tip: available_tokens should be ≤ max_tokens"
            )

    def _handle_ingest(self, args: Dict) -> str:
        """Handle ingest_context tool call"""
        text = args["text"]
        file_id = args["file_id"]
        metadata = args.get("metadata")

        # Validation
        if not text or len(text.strip()) == 0:
            raise ValueError(
                "text cannot be empty\n"
                "💡 Tip: Provide document content to ingest (minimum ~20 characters recommended)"
            )

        if len(text) < 20:
            raise ValueError(
                f"text is too short ({len(text)} chars)\n"
                "💡 Tip: Provide at least 20 characters for meaningful semantic analysis"
            )

        self._validate_file_id(file_id, must_exist=False)

        logger.info(f"Ingesting document: {file_id} ({len(text)} chars)")

        try:
            skeleton = self.compressor.ingest_file(text, file_id, metadata)
        except Exception as e:
            raise RuntimeError(
                f"Failed to ingest document: {str(e)}\n"
                "💡 Tip: Check that text is valid and file_id contains only alphanumeric and underscores"
            ) from e

        # Initialize retrieval history
        self.retrieval_history[file_id] = []

        result = f"""
✅ Document ingested successfully!

File ID: {file_id} containing {skeleton.total_nodes} semantic nodes
Original tokens: {skeleton.total_tokens:,}
Skeleton tokens: {skeleton.skeleton_tokens:,}
Compression ratio: {skeleton.compression_ratio:.1f}x

📊 Token savings: {skeleton.total_tokens - skeleton.skeleton_tokens:,} tokens ({(1 - skeleton.skeleton_tokens/skeleton.total_tokens)*100:.1f}%)

💡 IMPORTANT: Use read_skeleton('{file_id}') to view the semantic map BEFORE requesting specific details.
   This "map before territory" approach ensures you understand the document structure.

Next steps:
1. read_skeleton('{file_id}') - View the compressed structure (recommended first step)
2. modulate_region() - Retrieve specific sections at chosen fidelity
3. search_semantic() - Find relevant content via vector similarity
4. check_blind_spots() - Verify response completeness after generating answers

{skeleton.skeleton_text[:500]}...
(Use read_skeleton to see full structure)
"""
        return result

    def _handle_read_skeleton(self, args: Dict) -> str:
        """Handle read_skeleton tool call"""
        file_id = args["file_id"]
        self._validate_file_id(file_id, must_exist=True)

        logger.info(f"Reading skeleton: {file_id}")

        try:
            skeleton_text = self.compressor.read_skeleton(file_id)
            return skeleton_text
        except Exception as e:
            raise RuntimeError(
                f"Failed to read skeleton for '{file_id}': {str(e)}\n"
                f"💡 Tip: Verify the document was ingested successfully with get_stats()"
            ) from e

    def _handle_modulate_region(self, args: Dict) -> str:
        """Handle modulate_region tool call"""
        node_ids = args["node_ids"]
        fidelity_str = args.get("fidelity_level", "RAW")

        # Validation
        self._validate_node_ids(node_ids)

        # Convert string to enum with validation
        try:
            fidelity = FidelityLevel[fidelity_str]
        except KeyError:
            valid_levels = [level.name for level in FidelityLevel]
            raise ValueError(
                f"Invalid fidelity_level: '{fidelity_str}'\n"
                f"💡 Valid levels: {valid_levels}\n"
                f"   ABSTRACT: ~10 tokens (summary only)\n"
                f"   OUTLINE: ~30 tokens (summary + section markers)\n"
                f"   STRUCTURE: ~50 tokens (headers + entities)\n"
                f"   DETAILED: ~100 tokens (summary + excerpts)\n"
                f"   RAW: Full original text"
            )

        logger.info(f"Modulating {len(node_ids)} nodes at {fidelity_str} fidelity")

        # Track retrieval for blind spot detection
        for node_id in node_ids:
            # Extract file_id from node_id (format: file_id_n123)
            file_id = "_".join(node_id.split("_")[:-1])
            if file_id not in self.retrieval_history:
                self.retrieval_history[file_id] = []
            if node_id not in self.retrieval_history[file_id]:
                self.retrieval_history[file_id].append(node_id)

        try:
            result = self.compressor.modulate_region(node_ids, fidelity)
            return result
        except Exception as e:
            raise RuntimeError(
                f"Failed to modulate region: {str(e)}\n"
                f"💡 Tip: Verify node IDs are valid with read_skeleton()"
            ) from e

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

        report = self.blind_spot_detector.analyze_response(ai_response, file_id, retrieved_nodes)

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

        is_hallucinating, warnings = self.halo_detector.detect_hallucination(ai_response, file_id)

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

    def _handle_adapt_to_context_window(self, args: Dict) -> str:
        """Handle adapt_to_context_window tool call"""
        file_id = args["file_id"]
        available_tokens = args["available_tokens"]
        max_tokens = args.get("max_tokens", 100000)
        query_priority = args.get("query_priority", 0.5)

        # Validation
        self._validate_file_id(file_id, must_exist=True)
        self._validate_token_count(available_tokens, max_tokens)

        if not 0.0 <= query_priority <= 1.0:
            raise ValueError(
                f"query_priority must be between 0.0 and 1.0, got {query_priority}\n"
                "💡 Tip: 0.0 = low priority, 0.5 = medium, 1.0 = high priority"
            )

        logger.info(
            f"Adapting skeleton for {file_id}: {available_tokens}/{max_tokens} tokens available"
        )

        try:
            result = self.context_window_adapter.adapt_to_context_window(
                file_id=file_id,
                available_tokens=available_tokens,
                max_tokens=max_tokens,
                query_priority=query_priority,
            )
            return result
        except Exception as e:
            raise RuntimeError(
                f"Failed to adapt to context window: {str(e)}\n"
                "💡 Tip: This is a JSCCM-inspired feature. Check that the document exists and token counts are valid."
            ) from e

    def _handle_multilevel_encode(self, args: Dict) -> str:
        """Handle multilevel_encode tool call"""
        file_id = args["file_id"]
        available_tokens = args["available_tokens"]

        # Validation
        self._validate_file_id(file_id, must_exist=True)
        self._validate_token_count(available_tokens)

        logger.info(f"Generating multi-level encoding for {file_id}: {available_tokens} tokens available")

        try:
            result = self.multilevel_encoder.generate_adaptive_skeleton(file_id, available_tokens)
            return result
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate multi-level encoding: {str(e)}\n"
                "💡 Tip: This JSCCM-inspired feature requires Main + Auxiliary + Detail branches.\n"
                "   Try with at least 1000 tokens available for meaningful output."
            ) from e

    async def run(self):
        """Run the MCP server"""
        logger.info("🚀 Starting Semantic Modulator MCP Server")
        logger.info("   Combining Semantic Communication + Fidelity-Preserving Encoding")
        logger.info("   Model: all-MiniLM-L6-v2 (local)")
        logger.info("   Mode: Adaptive Semantic Fidelity\n")

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream, write_stream, self.server.create_initialization_options()
            )


def main():
    """Entry point"""
    server = SemanticModulatorServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
