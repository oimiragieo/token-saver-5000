"""Bootstrap entrypoints for application wiring."""

from src.server import SemanticModulatorServer


def create_server() -> SemanticModulatorServer:
    """Create and return the canonical MCP server instance."""
    return SemanticModulatorServer()
