"""Tool schemas: Experimental / NOT production-ready (exp). Split from mcp_core.py (N2 slice 2).

Each list is verbatim `Tool(...)` schema literals moved from the original
monolithic `setup_mcp_tools`, unchanged.
"""

from mcp.types import Tool

from ._constants import SCOPE_PROPERTIES

EXPERIMENTAL_TOOLS: list = [
    Tool(
        name="toon_encode",
        description=(
            "[EXPERIMENTAL] Encode data to TOON format (~40% smaller than JSON). "
            "TOON = Token-Oriented Object Notation. Pure Python, always available. "
            "NOT production-ready. Returns experimental flag."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "description": "Data to encode (dict or list)",
                },
            },
            "required": ["data"],
        },
    ),
    Tool(
        name="toon_decode",
        description=(
            "[EXPERIMENTAL] Decode TOON format back to structured data. "
            "TOON is lossy - optimized for LLM consumption, not round-trip serialization. "
            "NOT production-ready. Returns experimental flag."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "toon_input": {
                    "type": "string",
                    "description": "TOON-formatted string to decode",
                },
            },
            "required": ["toon_input"],
        },
    ),
    Tool(
        name="scar_compress",
        description=(
            "[EXPERIMENTAL] Compress embeddings using SCAR (learnable compression). "
            "WARNING: Uses UNTRAINED random weights by default. Requires PyTorch. "
            "NOT production-ready without model training. Returns experimental flag."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "Document ID to compress embeddings for",
                },
                "target_dim": {
                    "type": "integer",
                    "description": "Target embedding dimension (default: 128)",
                    "default": 128,
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["doc_id"],
        },
    ),
    Tool(
        name="scar_get_stats",
        description=(
            "[EXPERIMENTAL] Get SCAR compressor statistics and model state. "
            "Shows PyTorch availability and model training status. "
            "NOT production-ready. Returns experimental flag."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="multimodal_ingest",
        description=(
            "[EXPERIMENTAL] Ingest mixed content (text, code, images). "
            "Requires Pillow for image support. Image paths validated for security. "
            "NOT production-ready. Returns experimental flag."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "Unique document identifier",
                },
                "text_content": {
                    "type": "string",
                    "description": "Text content to ingest",
                },
                "code_content": {
                    "type": "string",
                    "description": "Code content to ingest",
                },
                "code_language": {
                    "type": "string",
                    "description": "Code language (default: python)",
                    "default": "python",
                },
                "image_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Paths to images (validated for security)",
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["doc_id"],
        },
    ),
    Tool(
        name="verify_compression",
        description=(
            "[EXPERIMENTAL] Verify compression operation using ASG-SI contracts. "
            "Checks preconditions (valid input, fidelity level) and postconditions "
            "(compression ratio, skeleton quality). Returns contract violations. "
            "Based on arxiv.org/abs/2512.23760 Audited Skill-Graph Self-Improvement."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "document": {
                    "type": "string",
                    "description": "Original document text",
                },
                "skeleton_text": {
                    "type": "string",
                    "description": "Compressed skeleton output",
                },
                "node_map": {
                    "type": "object",
                    "description": "Node ID to description mapping",
                },
                "original_tokens": {
                    "type": "integer",
                    "description": "Original token count",
                },
                "skeleton_tokens": {
                    "type": "integer",
                    "description": "Skeleton token count",
                },
                "fidelity_level": {
                    "type": "string",
                    "description": "Target fidelity (ABSTRACT, OUTLINE, STRUCTURE, DETAILED, RAW)",
                },
            },
            "required": [
                "document",
                "skeleton_text",
                "original_tokens",
                "skeleton_tokens",
                "fidelity_level",
            ],
        },
    ),
    Tool(
        name="calculate_reward",
        description=(
            "[EXPERIMENTAL] Calculate decomposed compression reward using ASG-SI system. "
            "Computes 5 reward components: Schema (validation), Semantic (meaning preservation), "
            "Fidelity (ratio adherence), Composition (graph integrity), Memory (efficiency). "
            "Based on arxiv.org/abs/2512.23760 Audited Skill-Graph Self-Improvement."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "input_text": {
                    "type": "string",
                    "description": "Original text",
                },
                "output_text": {
                    "type": "string",
                    "description": "Compressed output",
                },
                "input_tokens": {
                    "type": "integer",
                    "description": "Input token count",
                },
                "output_tokens": {
                    "type": "integer",
                    "description": "Output token count",
                },
                "fidelity_level": {
                    "type": "string",
                    "description": "Target fidelity level",
                },
                "ssim_score": {
                    "type": "number",
                    "description": "Pre-calculated SSIM score (optional)",
                },
            },
            "required": [
                "input_text",
                "output_text",
                "input_tokens",
                "output_tokens",
                "fidelity_level",
            ],
        },
    ),
    Tool(
        name="get_evidence_stats",
        description=(
            "[EXPERIMENTAL] Get evidence store statistics for audit trail. "
            "The store maintains a tamper-evident blockchain-style chain of all "
            "compression operations with cryptographic integrity verification. "
            "Based on arxiv.org/abs/2512.23760 Audited Skill-Graph Self-Improvement."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="generate_synthetic_tests",
        description=(
            "[EXPERIMENTAL] Generate synthetic test cases for adversarial testing. "
            "Uses ASG-SI experience synthesis to create boundary cases, adversarial "
            "documents, and stress test scenarios for compression validation. "
            "Based on arxiv.org/abs/2512.23760 Audited Skill-Graph Self-Improvement."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "test_type": {
                    "type": "string",
                    "description": "Test type: boundary, dialogue, ace, or all",
                    "default": "boundary",
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility (optional)",
                },
            },
        },
    ),
]
