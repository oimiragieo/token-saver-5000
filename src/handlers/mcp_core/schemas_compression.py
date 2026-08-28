"""Compression MCP tool schemas (core + batch modules)."""

from .schemas_compression_core import COMPRESSION_CORE_TOOLS
from .schemas_compression_batch import COMPRESSION_BATCH_TOOLS

COMPRESSION_TOOLS: list = COMPRESSION_CORE_TOOLS + COMPRESSION_BATCH_TOOLS
