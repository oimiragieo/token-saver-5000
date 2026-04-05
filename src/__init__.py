"""
Semantic Modulator - Adaptive Semantic Fidelity MCP Server

Combining Semantic Communication with Fidelity-Preserving Encoding
to achieve context-aware compression for AI interactions.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("semantic-modulator")
except PackageNotFoundError:
    __version__ = "0.11.0"  # fallback for editable/dev installs
