"""
Visualization Handlers for Graph Visualization MCP Tools (v0.6.0)

Provides 4 MCP tools for visualizing and analyzing semantic compression graphs:
- export_graph_json: Export graph as JSON
- visualize_graph_html: Generate interactive HTML visualization
- export_graph_graphml: Export as GraphML for analysis tools
- explain_compression_decision: Explain why nodes were kept/dropped
"""

import logging
import json
from typing import Any, Dict

from ..graph_visualizer import GraphVisualizer, VisualizationConfig
from ..error_helpers import SmartError
from ..identity_scope import compose_scoped_file_id

logger = logging.getLogger(__name__)


def _scoped_file_id(args: Dict[str, Any]) -> str:
    return compose_scoped_file_id(
        args.get("file_id", ""),
        workspace_id=args.get("workspace_id"),
        user_id=args.get("user_id"),
        agent_id=args.get("agent_id"),
        session_id=args.get("session_id"),
    )


async def handle_export_graph_json(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    """Handle export_graph_json MCP tool (v0.6.0)."""
    # Extract arguments
    file_id = args.get("file_id")
    scoped_file_id = _scoped_file_id(args)
    max_nodes = args.get("max_nodes", 50)
    min_importance = args.get("min_importance", 0.0)

    # Validate file_id
    if not file_id or not isinstance(file_id, str) or not file_id.strip():
        raise SmartError.missing_required_field("file_id", "export_graph_json")

    # Create visualizer
    visualizer = GraphVisualizer(context["compressor"])

    # Create custom config
    config = VisualizationConfig(
        max_nodes=max_nodes,
        min_importance=min_importance,
        show_edge_weights=True,
        highlight_skeleton=True,
    )

    # Export as JSON
    try:
        json_output = visualizer.export_json(scoped_file_id, config)
        parsed = json.loads(json_output)
        parsed["file_id"] = file_id
        return json.dumps(parsed)
    except ValueError:
        raise SmartError.document_not_found(file_id)
    except Exception as e:
        logger.error(f"export_graph_json error: {e}")
        raise


async def handle_visualize_graph_html(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    """Handle visualize_graph_html MCP tool (v0.6.0)."""
    # Extract arguments
    file_id = args.get("file_id")
    scoped_file_id = _scoped_file_id(args)
    output_path = args.get("output_path")
    max_nodes = args.get("max_nodes", 50)

    # Validate arguments
    if not file_id or not isinstance(file_id, str) or not file_id.strip():
        raise SmartError.missing_required_field("file_id", "visualize_graph_html")

    if not output_path or not isinstance(output_path, str) or not output_path.strip():
        raise SmartError.missing_required_field("output_path", "visualize_graph_html")

    # Create visualizer
    visualizer = GraphVisualizer(context["compressor"])

    # Create custom config
    config = VisualizationConfig(
        max_nodes=max_nodes,
        min_importance=0.0,
        show_edge_weights=True,
        highlight_skeleton=True,
    )

    # Generate HTML visualization
    try:
        result = visualizer.visualize_html(scoped_file_id, output_path, config)
        return result
    except ValueError as e:
        if "No graph found" in str(e):
            raise SmartError.document_not_found(file_id)
        raise
    except ImportError:
        raise ValueError(
            "pyvis is required for HTML visualization. " "Install with: pip install pyvis"
        )
    except Exception as e:
        logger.error(f"visualize_graph_html error: {e}")
        raise


async def handle_export_graph_graphml(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    """Handle export_graph_graphml MCP tool (v0.6.0)."""
    # Extract arguments
    file_id = args.get("file_id")
    scoped_file_id = _scoped_file_id(args)
    output_path = args.get("output_path")

    # Validate arguments
    if not file_id or not isinstance(file_id, str) or not file_id.strip():
        raise SmartError.missing_required_field("file_id", "export_graph_graphml")

    if not output_path or not isinstance(output_path, str) or not output_path.strip():
        raise SmartError.missing_required_field("output_path", "export_graph_graphml")

    # Create visualizer
    visualizer = GraphVisualizer(context["compressor"])

    # Export as GraphML
    try:
        result = visualizer.export_graphml(scoped_file_id, output_path)
        return result
    except ValueError as e:
        if "No graph found" in str(e):
            raise SmartError.document_not_found(file_id)
        raise
    except Exception as e:
        logger.error(f"export_graph_graphml error: {e}")
        raise


async def handle_explain_compression_decision(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    """Handle explain_compression_decision MCP tool (v0.6.0)."""
    # Extract arguments
    file_id = args.get("file_id")
    scoped_file_id = _scoped_file_id(args)
    node_id = args.get("node_id")

    # Validate arguments
    if not file_id or not isinstance(file_id, str) or not file_id.strip():
        raise SmartError.missing_required_field("file_id", "explain_compression_decision")

    if not node_id or not isinstance(node_id, str) or not node_id.strip():
        raise SmartError.missing_required_field("node_id", "explain_compression_decision")

    # Create visualizer
    visualizer = GraphVisualizer(context["compressor"])

    # Generate explanation
    try:
        explanation = visualizer.explain_compression_decision(scoped_file_id, node_id)
        return explanation
    except ValueError as e:
        error_msg = str(e)
        if "No graph found" in error_msg:
            raise SmartError.document_not_found(file_id)
        elif "not found in chunks" in error_msg:
            raise ValueError(
                f"Node '{node_id}' not found in document '{file_id}'. "
                f"Check available nodes with export_graph_json."
            )
        raise
    except Exception as e:
        logger.error(f"explain_compression_decision error: {e}")
        raise
