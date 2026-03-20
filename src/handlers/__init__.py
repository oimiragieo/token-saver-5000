"""
Handler Modules for Semantic Modulator MCP Server

This package contains the refactored handler modules that were split
from the monolithic server.py file.

Module Structure:
- mcp_core.py: Core MCP infrastructure (tool schemas + routing)
- compression_handlers.py: Document compression tools
- afm_handlers.py: AFM dialogue management tools
- file_sync_handlers.py: File sync and versioning tools
- resource_handlers.py: Resource health monitoring
- detection_handlers.py: Blind spot and hallucination detection

Version: 0.4.0
"""

# Convenience exports (can be expanded as needed)
__all__ = [
    "mcp_core",
    "compression_handlers",
    "afm_handlers",
    "file_sync_handlers",
    "resource_handlers",
    "detection_handlers",
    "prompt_handlers",
    "memory_handlers",
    "experiment_handlers",
    "model_handlers",
    "bundle_handlers",
    "connector_handlers",
    "temporal_handlers",
    "multimodal_handlers",
]
