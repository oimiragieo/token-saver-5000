"""Tool schemas: Model handlers (moh) + Experiment handlers (eh). Split from mcp_core.py (N2 slice 2).

Each list is verbatim `Tool(...)` schema literals moved from the original
monolithic `setup_mcp_tools`, unchanged.
"""

from mcp.types import Tool

from ._constants import SCOPE_PROPERTIES

MODEL_TOOLS: list = [
    Tool(
        name="get_provider_profile",
        description="Get provider-aware pricing, cache telemetry fields, and prompt-shaping guidance for a model.",
        inputSchema={
            "type": "object",
            "properties": {
                **SCOPE_PROPERTIES,
                "model": {"type": "string", "description": "Model identifier"},
            },
            "required": ["model"],
        },
    ),
    Tool(
        name="estimate_model_cost",
        description="Estimate token cost savings for a model using original and compressed token counts.",
        inputSchema={
            "type": "object",
            "properties": {
                **SCOPE_PROPERTIES,
                "model": {"type": "string", "description": "Model identifier"},
                "original_tokens": {"type": "integer", "description": "Original token count"},
                "compressed_tokens": {
                    "type": "integer",
                    "description": "Compressed token count",
                },
            },
            "required": ["model", "original_tokens", "compressed_tokens"],
        },
    ),
    Tool(
        name="optimize_for_model",
        description="Generate provider-aware cost, fidelity, and prompt-shaping recommendations for a target model.",
        inputSchema={
            "type": "object",
            "properties": {
                **SCOPE_PROPERTIES,
                "model": {"type": "string", "description": "Model identifier"},
                "text": {"type": "string", "description": "Representative source text"},
                "use_case": {
                    "type": "string",
                    "enum": [
                        "quick_summary",
                        "topic_overview",
                        "entity_extraction",
                        "question_answering",
                        "detailed_analysis",
                        "exact_quotes",
                        "code_review",
                        "fact_verification",
                    ],
                },
                "num_nodes": {
                    "type": "integer",
                    "description": "Expected retrieval node count",
                },
                "token_budget": {
                    "type": "integer",
                    "description": "Optional explicit token budget",
                },
                "query_complexity": {
                    "type": "string",
                    "enum": ["simple", "medium", "complex"],
                    "default": "medium",
                },
            },
            "required": ["model", "text", "use_case", "num_nodes"],
        },
    ),
    Tool(
        name="assess_cache_compatibility",
        description="Assess whether a provider and harness combination exposes enough telemetry to validate prompt cache behavior reliably.",
        inputSchema={
            "type": "object",
            "properties": {
                **SCOPE_PROPERTIES,
                "model": {"type": "string", "description": "Model identifier"},
                "harness": {
                    "type": "string",
                    "enum": [
                        "anthropic_api",
                        "claude_code",
                        "openai_api",
                        "codex_cli",
                        "gemini_api",
                        "gemini_cli",
                    ],
                    "description": "Provider or CLI surface used to make the request",
                },
                "raw_usage_available": {
                    "type": "boolean",
                    "description": "Whether raw provider usage payloads are visible",
                    "default": False,
                },
                "cli_stats_available": {
                    "type": "boolean",
                    "description": "Whether CLI-exported cache stats are available",
                    "default": False,
                },
            },
            "required": ["model", "harness"],
        },
    ),
    Tool(
        name="capture_cache_telemetry",
        description="Normalize provider-side prompt cache telemetry from a real model API response and warn on silent cache misses.",
        inputSchema={
            "type": "object",
            "properties": {
                **SCOPE_PROPERTIES,
                "model": {"type": "string", "description": "Model identifier"},
                "api_response": {
                    "type": "object",
                    "description": "Raw provider response object containing usage telemetry",
                },
                "file_id": {
                    "type": "string",
                    "description": "Optional document or prompt identifier tied to this request",
                },
                "prompt_id": {
                    "type": "string",
                    "description": "Optional prompt identifier returned by render_prompt_template",
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional session identifier for aggregating multi-turn cache metrics",
                },
                "actual_rendered_prefix": {
                    "type": "string",
                    "description": "Optional exact prefix string actually sent to the provider for cache miss diagnosis",
                },
                "expected_cache_hit": {
                    "type": "boolean",
                    "description": "Whether this request was expected to reuse a cached prompt prefix",
                    "default": False,
                },
            },
            "required": ["model", "api_response"],
        },
    ),
    Tool(
        name="diagnose_cache_miss",
        description=(
            "Diagnose why an expected provider cache hit missed by comparing the recorded prompt "
            "expectation with the actual rendered prefix that reached the provider."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                **SCOPE_PROPERTIES,
                "prompt_id": {
                    "type": "string",
                    "description": "Prompt identifier returned by render_prompt_template",
                },
                "model": {"type": "string", "description": "Model identifier"},
                "actual_rendered_prefix": {
                    "type": "string",
                    "description": "Exact rendered prompt prefix actually sent to the provider",
                },
                "api_response": {
                    "type": "object",
                    "description": "Raw provider response object containing usage telemetry",
                },
            },
            "required": ["prompt_id", "model", "actual_rendered_prefix", "api_response"],
        },
    ),
]


EXPERIMENT_TOOLS: list = [
    Tool(
        name="create_dataset",
        description=(
            "Create a reusable named benchmark/evaluation dataset from inline cases "
            "or a JSON corpus fixture."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                **SCOPE_PROPERTIES,
                "name": {"type": "string", "description": "Dataset name"},
                "description": {"type": "string", "description": "Dataset description"},
                "cases": {
                    "type": "array",
                    "description": "Optional inline benchmark cases",
                    "items": {
                        "type": "object",
                        "properties": {
                            **SCOPE_PROPERTIES,
                            "case_id": {"type": "string"},
                            "name": {"type": "string"},
                            "text": {"type": "string"},
                            "min_compression_ratio": {"type": "number"},
                            "min_token_savings_pct": {"type": "number"},
                            "query": {"type": "string"},
                        },
                        "required": [
                            "case_id",
                            "name",
                            "text",
                            "min_compression_ratio",
                            "min_token_savings_pct",
                        ],
                    },
                },
                "source_path": {
                    "type": "string",
                    "description": "Optional benchmark corpus JSON path",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional structured dataset metadata",
                },
            },
            "required": ["name", "description"],
        },
    ),
    Tool(
        name="list_datasets",
        description="List named datasets available for experiment runs.",
        inputSchema={"type": "object", "properties": {**SCOPE_PROPERTIES}},
    ),
    Tool(
        name="run_experiment",
        description=(
            "Run a tracked benchmark/evaluation experiment over a named dataset and "
            "store benchmark, verifier, and reward outputs."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                **SCOPE_PROPERTIES,
                "dataset_name": {"type": "string", "description": "Dataset name"},
                "mode": {
                    "type": "string",
                    "enum": ["baseline", "query_guided", "evidence_aware"],
                    "description": "Benchmark mode to execute",
                    "default": "baseline",
                },
                "case_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional subset of case IDs to run",
                },
                "similarity_threshold": {
                    "type": "number",
                    "description": "Optional similarity threshold override",
                    "default": 0.75,
                },
                "skeleton_ratio": {
                    "type": "number",
                    "description": "Optional skeleton ratio override",
                    "default": 0.2,
                },
                "baseline_run_id": {
                    "type": "string",
                    "description": "Optional baseline run identifier",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional structured run metadata",
                },
            },
            "required": ["dataset_name"],
        },
    ),
    Tool(
        name="get_experiment_run",
        description="Fetch a stored experiment run and its per-case evaluation details.",
        inputSchema={
            "type": "object",
            "properties": {
                **SCOPE_PROPERTIES,
                "run_id": {"type": "string", "description": "Experiment run identifier"},
            },
            "required": ["run_id"],
        },
    ),
    Tool(
        name="compare_experiment_runs",
        description=(
            "Compare two experiment runs and report deltas in pass counts, compression, "
            "verification, and reward quality."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                **SCOPE_PROPERTIES,
                "run_id_a": {"type": "string", "description": "Base run identifier"},
                "run_id_b": {"type": "string", "description": "Comparison run identifier"},
            },
            "required": ["run_id_a", "run_id_b"],
        },
    ),
]
