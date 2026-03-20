"""
Help Handler Module

This module provides handler functions for tool documentation and help:
- handle_tool_help: Provides detailed help, examples, and tips for any MCP tool

New in v0.9.0: Part of the Programmer UX Improvement Plan.
"""

import json
import logging
from functools import lru_cache
from typing import Any, Dict

from ..types import HandlerContext
from .compression_handlers import (
    get_ingest_context_output_fields,
    get_read_skeleton_output_fields,
    get_recommend_fidelity_output_fields,
    get_search_semantic_output_fields,
)
from .bundle_handlers import get_bundle_output_fields
from .connector_handlers import get_connector_output_fields
from .experiment_handlers import get_dataset_output_fields, get_experiment_output_fields
from .memory_handlers import get_memory_output_fields, get_user_profile_output_fields
from .model_handlers import (
    get_cache_compatibility_output_fields,
    get_cache_diagnostic_output_fields,
    get_cache_telemetry_output_fields,
    get_model_output_fields,
)
from .multimodal_handlers import get_multimodal_output_fields
from .prompt_handlers import (
    get_prompt_audit_output_fields,
    get_prompt_compare_output_fields,
    get_prompt_deploy_output_fields,
    get_prefix_collision_output_fields,
    get_prompt_output_fields,
    get_prompt_render_output_fields,
)
from .resource_handlers import get_check_environment_output_fields
from .temporal_handlers import get_temporal_output_fields

logger = logging.getLogger("semantic-modulator")

# Tool documentation registry with examples and tips
TOOL_HELP_REGISTRY: Dict[str, Dict[str, Any]] = {
    # === Document Compression Tools ===
    "ingest_context": {
        "category": "Document Compression",
        "description": "Ingest and compress a document into a semantic graph for efficient retrieval.",
        "parameters": {
            "text": "The raw document text to ingest (required)",
            "file_id": "Unique identifier for this document (required)",
            "file_path": "Optional path to source file for sync tracking",
            "metadata": "Optional metadata dict (author, date, source, tags)",
            "chunking_strategy": "Optional: auto (default), fixed, or semantic chunking",
        },
        "output_fields": get_ingest_context_output_fields(),
        "examples": [
            {
                "description": "Basic text ingestion",
                "args": {"text": "Your document text...", "file_id": "doc_1"},
            },
            {
                "description": "With file path for sync tracking",
                "args": {
                    "text": "Code content...",
                    "file_id": "main.py",
                    "file_path": "/path/to/main.py",
                },
            },
        ],
        "tips": [
            "Use meaningful file_ids (e.g., 'auth_service.py' not 'doc1')",
            "Provide file_path to enable staleness detection",
            "Default chunking_strategy=auto upgrades larger structured docs to semantic chunking",
            "Metadata is preserved and returned in read_skeleton",
        ],
        "related_tools": ["read_skeleton", "search_semantic", "modulate_region"],
    },
    "read_skeleton": {
        "category": "Document Compression",
        "description": "Get a compressed skeleton view of an ingested document.",
        "parameters": {
            "file_id": "ID of the document to read (required)",
            "selection_mode": "Optional: baseline, query_guided, or evidence_aware (default: baseline)",
            "query": "Optional query text. Required for query_guided and evidence_aware modes",
            "top_k": "Optional evidence node count for evidence_aware mode (default: 5)",
            "min_similarity": "Optional sufficiency threshold for evidence_aware mode (default: 0.35)",
            "as_of": "Optional ISO-8601 or unix timestamp for temporal filtering",
            "include_invalidated": "Optional: include invalidated facts in output (default: false)",
        },
        "output_fields": get_read_skeleton_output_fields(),
        "examples": [
            {"description": "Read document skeleton", "args": {"file_id": "my_doc"}},
            {
                "description": "Query-guided skeleton",
                "args": {
                    "file_id": "my_doc",
                    "selection_mode": "query_guided",
                    "query": "error handling strategy",
                },
            },
            {
                "description": "Evidence-aware skeleton",
                "args": {
                    "file_id": "my_doc",
                    "selection_mode": "evidence_aware",
                    "query": "authentication flow",
                    "top_k": 5,
                    "min_similarity": 0.4,
                },
            },
        ],
        "tips": [
            "Skeleton shows high-importance nodes only (~20% of content)",
            "Node IDs in skeleton can be used with modulate_region",
            "Use selection_mode=query_guided to bias anchors toward your task",
            "Use selection_mode=evidence_aware to detect insufficient evidence",
            "Returns staleness warning if source file changed",
        ],
        "related_tools": ["ingest_context", "modulate_region", "check_file_sync"],
    },
    "modulate_region": {
        "category": "Document Compression",
        "description": "Retrieve content at specified fidelity level for specific nodes.",
        "parameters": {
            "node_ids": "List of node IDs to retrieve (required)",
            "fidelity_level": "Detail level: ABSTRACT, OUTLINE, STRUCTURE, DETAILED, RAW",
        },
        "examples": [
            {
                "description": "Get detailed content",
                "args": {"node_ids": ["doc_n0", "doc_n1"], "fidelity_level": "DETAILED"},
            },
            {
                "description": "Quick summary",
                "args": {"node_ids": ["doc_n0"], "fidelity_level": "ABSTRACT"},
            },
        ],
        "tips": [
            "Use recommend_fidelity to choose the best fidelity level",
            "Lower fidelity = fewer tokens, higher fidelity = more detail",
            "Combine with search_semantic to find relevant nodes first",
        ],
        "related_tools": ["read_skeleton", "search_semantic", "recommend_fidelity"],
    },
    "search_semantic": {
        "category": "Document Compression",
        "description": "Find relevant content using semantic similarity search.",
        "parameters": {
            "query": "Search query (required)",
            "file_id": "Optional: limit search to specific document",
            "top_k": "Number of results to return (default: 5)",
            "evidence_aware": "Optional: enable insufficiency detection with expanded retrieval (default: false)",
            "min_similarity": "Optional sufficiency threshold for evidence_aware mode (default: 0.35)",
            "as_of": "Optional ISO-8601 or unix timestamp for temporal filtering",
            "include_invalidated": "Optional: include invalidated facts in results (default: false)",
        },
        "output_fields": get_search_semantic_output_fields(),
        "examples": [
            {"description": "Search all docs", "args": {"query": "authentication logic"}},
            {
                "description": "Search specific doc",
                "args": {"query": "error handling", "file_id": "auth.py", "top_k": 10},
            },
            {
                "description": "Evidence-aware search",
                "args": {
                    "query": "token refresh race condition",
                    "file_id": "auth.py",
                    "evidence_aware": True,
                    "min_similarity": 0.4,
                },
            },
        ],
        "tips": [
            "Returns both similarity (query match) and importance (PageRank) scores",
            "Use file_id to focus search on specific documents",
            "Set evidence_aware=true when correctness matters more than speed",
            "Results are ranked by semantic similarity, not keyword match",
        ],
        "related_tools": ["modulate_region", "read_skeleton", "check_blind_spots"],
    },
    "ingest_directory": {
        "category": "Directory Ingestion",
        "description": "Bulk ingest code files from a directory using glob patterns.",
        "parameters": {
            "directory": "Directory path to scan (required)",
            "patterns": "Glob patterns for files (default: ['*.py', '*.js', '*.ts'])",
            "exclude_patterns": "Patterns to exclude (default: node_modules, __pycache__)",
            "max_files": "Maximum files to ingest (default: 50, max: 100)",
            "max_concurrent": "Concurrent ingestions (default: 4, max: 8)",
        },
        "examples": [
            {
                "description": "Ingest Python files",
                "args": {"directory": "./src", "patterns": ["*.py"]},
            },
            {
                "description": "Ingest with custom exclusions",
                "args": {
                    "directory": "./project",
                    "patterns": ["*.py", "*.js"],
                    "exclude_patterns": ["**/test/**", "**/vendor/**"],
                    "max_files": 30,
                },
            },
        ],
        "tips": [
            "Uses PathValidator for security (prevents path traversal)",
            "Files are processed in parallel for 4x throughput",
            "file_id is derived from relative path (e.g., 'src/main.py')",
        ],
        "related_tools": ["ingest_context", "batch_ingest_documents", "list_documents"],
    },
    "ingest_multimodal": {
        "category": "Multimodal",
        "description": "Ingest production multimodal content including images, transcripts, and document bundles.",
        "parameters": {
            "doc_id": "Logical multimodal document identifier (required)",
            "text_content": "Optional text payload",
            "code_content": "Optional code payload",
            "code_language": "Optional code language label",
            "image_paths": "Optional validated image paths",
            "image_captions": "Optional caption text keyed by submitted image path",
            "image_ocr_text": "Optional OCR text keyed by submitted image path",
            "audio_items": "Optional transcript-backed audio items",
            "document_items": "Optional document-with-images bundles",
        },
        "output_fields": get_multimodal_output_fields(),
        "examples": [
            {
                "description": "Ingest a release brief with code",
                "args": {
                    "doc_id": "release_brief",
                    "text_content": "Architecture and rollout notes",
                    "code_content": "def deploy():\n    return 'ok'",
                    "code_language": "python",
                },
            }
        ],
        "tips": [
            "Provide captions or OCR text for images when possible to improve retrieval quality",
            "Audio support expects transcripts instead of raw speech decoding",
            "Unsupported video assets fail explicitly rather than silently no-op",
        ],
        "related_tools": ["search_multimodal", "multimodal_ingest"],
    },
    "search_multimodal": {
        "category": "Multimodal",
        "description": "Search a multimodal project using text, code, or image queries.",
        "parameters": {
            "doc_id": "Logical multimodal document identifier (required)",
            "query": "Required for text or code queries",
            "query_type": "Optional: text, code, or image (default: text)",
            "image_query_path": "Required for image queries",
            "top_k": "Optional number of results (default: 5)",
            "filter_modality": "Optional result modality filter",
        },
        "output_fields": get_multimodal_output_fields(),
        "examples": [
            {
                "description": "Search a multimodal brief",
                "args": {"doc_id": "release_brief", "query": "deployment checklist", "top_k": 3},
            }
        ],
        "tips": [
            "Text queries can still retrieve caption/OCR-enriched image content",
            "Use filter_modality=image to focus on visual assets",
        ],
        "related_tools": ["ingest_multimodal", "search_semantic"],
    },
    "create_handoff_bundle": {
        "category": "Handoff Bundles",
        "description": "Create a structured, auditable handoff bundle from a compressed document.",
        "parameters": {
            "file_id": "Visible document identifier (required)",
            "query": "Optional focus query for evidence-guided distillation",
            "top_k": "Optional focused search result count (default: 5)",
            "metadata": "Optional handoff metadata such as owner or destination",
        },
        "output_fields": get_bundle_output_fields(),
        "examples": [
            {
                "description": "Create an authentication handoff bundle",
                "args": {
                    "file_id": "auth_doc",
                    "query": "authentication flow",
                    "metadata": {"owner": "team-auth"},
                },
            }
        ],
        "tips": [
            "Provide a focus query when the handoff is for a specific task",
            "Bundle outputs include both replay text and a TOON-friendly artifact",
        ],
        "related_tools": [
            "list_handoff_bundles",
            "get_handoff_bundle",
            "replay_handoff_bundle",
        ],
    },
    "list_handoff_bundles": {
        "category": "Handoff Bundles",
        "description": "List structured handoff bundles visible to the current scope.",
        "parameters": {
            "file_id": "Optional visible document identifier filter",
        },
        "output_fields": get_bundle_output_fields(),
        "examples": [
            {"description": "List current workspace bundles", "args": {"workspace_id": "acme"}}
        ],
        "tips": ["Bundle visibility follows the same scope rules as stored documents."],
        "related_tools": ["create_handoff_bundle", "get_handoff_bundle"],
    },
    "get_handoff_bundle": {
        "category": "Handoff Bundles",
        "description": "Fetch one structured handoff bundle including its distilled artifact payloads.",
        "parameters": {
            "bundle_id": "Bundle identifier (required)",
        },
        "output_fields": get_bundle_output_fields(),
        "examples": [{"description": "Fetch a bundle", "args": {"bundle_id": "bundle-1"}}],
        "tips": ["Use replay_handoff_bundle when you want the condensed handoff text directly."],
        "related_tools": ["list_handoff_bundles", "replay_handoff_bundle"],
    },
    "replay_handoff_bundle": {
        "category": "Handoff Bundles",
        "description": "Replay a structured handoff bundle as condensed text plus artifact payloads.",
        "parameters": {
            "bundle_id": "Bundle identifier (required)",
        },
        "output_fields": get_bundle_output_fields(),
        "examples": [{"description": "Replay a bundle", "args": {"bundle_id": "bundle-1"}}],
        "tips": ["Replay output is designed for agent-to-agent or team handoff flows."],
        "related_tools": ["get_handoff_bundle", "create_handoff_bundle"],
    },
    "get_provider_profile": {
        "category": "Model Optimization",
        "description": "Fetch provider-specific pricing, cache fields, and prompt-shaping guidance for a model.",
        "parameters": {"model": "Target model identifier (required)"},
        "output_fields": get_model_output_fields(),
        "examples": [
            {"description": "Inspect Claude Sonnet profile", "args": {"model": "claude-sonnet-4.6"}}
        ],
        "tips": [
            "Use the returned cache field to monitor whether provider-side prompt caching is actually working.",
        ],
        "related_tools": ["estimate_model_cost", "optimize_for_model"],
    },
    "estimate_model_cost": {
        "category": "Model Optimization",
        "description": "Estimate cost savings for a model from original vs compressed token counts.",
        "parameters": {
            "model": "Target model identifier (required)",
            "original_tokens": "Original token count (required)",
            "compressed_tokens": "Compressed token count (required)",
        },
        "output_fields": get_model_output_fields(),
        "examples": [
            {
                "description": "Estimate savings for GPT-5.4",
                "args": {
                    "model": "gpt-5.4",
                    "original_tokens": 100000,
                    "compressed_tokens": 20000,
                },
            }
        ],
        "tips": [
            "Savings estimates are provider-aware and stay aligned with the model profile registry.",
        ],
        "related_tools": ["get_provider_profile", "optimize_for_model"],
    },
    "optimize_for_model": {
        "category": "Model Optimization",
        "description": "Generate model-aware fidelity, output format, caching, and prompt-structure guidance.",
        "parameters": {
            "model": "Target model identifier (required)",
            "text": "Representative source text (required)",
            "use_case": "Target retrieval/use-case mode (required)",
            "num_nodes": "Expected node count (required)",
            "token_budget": "Optional token budget",
            "query_complexity": "Optional simple/medium/complex hint",
        },
        "output_fields": get_model_output_fields(),
        "examples": [
            {
                "description": "Optimize a bundle for a premium model",
                "args": {
                    "model": "claude-opus-4.6",
                    "text": "Architecture context...",
                    "use_case": "topic_overview",
                    "num_nodes": 8,
                },
            }
        ],
        "tips": [
            "High-cost models bias toward stronger compaction and stable cached prefixes.",
            "Large-context models can justify higher-fidelity retrieval when the task is complex.",
        ],
        "related_tools": ["get_provider_profile", "estimate_model_cost", "recommend_fidelity"],
    },
    "assess_cache_compatibility": {
        "category": "Model Optimization",
        "description": "Assess whether a provider+harness surface exposes enough cache telemetry to monitor real prompt reuse reliably.",
        "parameters": {
            "model": "Target model identifier (required)",
            "harness": "Harness or integration surface such as anthropic_api, claude_code, openai_api, codex_cli, gemini_api, or gemini_cli (required)",
            "raw_usage_available": "Whether raw provider usage payloads are available for this surface",
            "cli_stats_available": "Whether CLI-exported cache stats such as Gemini CLI /stats are available",
        },
        "output_fields": get_cache_compatibility_output_fields(),
        "examples": [
            {
                "description": "Check Codex cache observability readiness",
                "args": {
                    "model": "gpt-5.3-codex",
                    "harness": "codex_cli",
                    "raw_usage_available": False,
                    "cli_stats_available": True,
                },
            }
        ],
        "tips": [
            "Deep dive both provider docs and the harness layer; either one can silently destroy cache hit rates.",
            "If raw provider usage is unavailable, verify whether your CLI exposes cache stats before trusting automated monitoring.",
        ],
        "related_tools": [
            "get_provider_profile",
            "optimize_for_model",
            "capture_cache_telemetry",
        ],
    },
    "capture_cache_telemetry": {
        "category": "Model Optimization",
        "description": "Normalize provider-side prompt cache telemetry from a real API response and warn on silent cache misses.",
        "parameters": {
            "model": "Target model identifier (required)",
            "api_response": "Raw provider response object containing usage data (required)",
            "file_id": "Optional related file or document identifier",
            "prompt_id": "Optional prompt render identifier returned by render_prompt_template for expectation-aware validation",
            "session_id": "Optional conversation or workflow session identifier for aggregating multi-turn cache metrics",
            "actual_rendered_prefix": "Optional exact prefix string actually sent to the provider for cache miss diagnosis",
            "expected_cache_hit": "Set true when you expected a cache reuse event and want a warning on miss",
        },
        "output_fields": get_cache_telemetry_output_fields(),
        "examples": [
            {
                "description": "Inspect cache reuse from an OpenAI response",
                "args": {
                    "model": "gpt-5.4",
                    "api_response": {
                        "usage": {
                            "prompt_tokens": 500,
                            "completion_tokens": 100,
                            "prompt_tokens_details": {"cached_tokens": 300},
                        }
                    },
                    "expected_cache_hit": True,
                },
            }
        ],
        "tips": [
            "Use this immediately after a provider call to catch silent cache misses before costs drift upward.",
            "When available, pass the prompt_id from render_prompt_template to validate actual provider behavior against the rendered prompt's cache expectations.",
            "If expected cache hits come back as zero, inspect hidden IDs, timestamps, and prompt ordering ahead of the query.",
            "Gemini CLI exports may surface cache visibility through /stats even when raw provider usage is not captured directly.",
        ],
        "related_tools": [
            "get_provider_profile",
            "optimize_for_model",
            "estimate_model_cost",
            "assess_cache_compatibility",
        ],
    },
    "diagnose_cache_miss": {
        "category": "Model Optimization",
        "description": "Diagnose a provider cache miss by comparing the expected cached prefix with the actual rendered prefix that was sent.",
        "parameters": {
            "prompt_id": "Prompt render identifier returned by render_prompt_template (required)",
            "model": "Target model identifier (required)",
            "actual_rendered_prefix": "Exact rendered prompt prefix actually sent to the provider (required)",
            "api_response": "Raw provider response object containing usage data (required)",
        },
        "output_fields": get_cache_diagnostic_output_fields(),
        "examples": [
            {
                "description": "Diagnose a cache miss caused by hidden UUID injection",
                "args": {
                    "prompt_id": "prompt-cache-abc123",
                    "model": "gpt-5.4",
                    "actual_rendered_prefix": '[system_instructions]\\nBe accurate.\\n{"id":"550e8400-e29b-41d4-a716-446655440000"}',
                    "api_response": {
                        "usage": {
                            "prompt_tokens": 500,
                            "completion_tokens": 100,
                            "prompt_tokens_details": {"cached_tokens": 0},
                        }
                    },
                },
            }
        ],
        "tips": [
            "Use this after a suspected cache miss when you can capture the exact prefix that reached the provider.",
            "UUIDs and timestamps in the stable prefix are common causes of unexpected cache misses.",
        ],
        "related_tools": [
            "render_prompt_template",
            "capture_cache_telemetry",
            "audit_prompt_cacheability",
        ],
    },
    "create_prompt_template": {
        "category": "Prompt Registry",
        "description": "Create a managed prompt template with version 1 and optional deployment label.",
        "parameters": {
            "name": "Unique prompt template name (required)",
            "description": "Human-readable prompt description (required)",
            "system_prompt": "Stable system prompt section (required)",
            "user_prompt_template": "User prompt template with optional {variables} (required)",
            "variables": "Optional list of variable names used by the template",
            "metadata": "Optional structured metadata for rollout and ownership",
            "deployment_label": "Optional initial deployment label such as production or staging",
        },
        "output_fields": get_prompt_output_fields(),
        "examples": [
            {
                "description": "Create a code review prompt",
                "args": {
                    "name": "review-default",
                    "description": "Prompt for high-signal code review",
                    "system_prompt": "You are a senior reviewer.",
                    "user_prompt_template": "Review this diff for {language}: {diff}",
                    "variables": ["language", "diff"],
                    "deployment_label": "production",
                },
            }
        ],
        "tips": [
            "Keep the most stable instructions in system_prompt for better cache behavior",
            "Use deployment labels to separate staging from production prompt versions",
        ],
        "related_tools": [
            "update_prompt_template",
            "deploy_prompt_version",
            "compare_prompt_versions",
        ],
    },
    "update_prompt_template": {
        "category": "Prompt Registry",
        "description": "Create a new version of an existing prompt template.",
        "parameters": {
            "name": "Prompt template name (required)",
            "description": "Optional updated template description",
            "system_prompt": "Optional replacement system prompt",
            "user_prompt_template": "Optional replacement user prompt template",
            "variables": "Optional replacement variable list",
            "metadata": "Optional metadata patch merged into the latest version metadata",
            "change_note": "Optional note explaining the change",
        },
        "output_fields": get_prompt_output_fields(),
        "examples": [
            {
                "description": "Create version 2 with a stronger system prompt",
                "args": {
                    "name": "review-default",
                    "system_prompt": "You are a senior reviewer. Prioritize correctness over style.",
                    "change_note": "Focus reviews on bug risk",
                },
            }
        ],
        "tips": [
            "Every update creates a new version instead of mutating history",
            "Use change_note so experiment and rollout comparisons stay understandable",
        ],
        "related_tools": [
            "get_prompt_template",
            "compare_prompt_versions",
            "deploy_prompt_version",
        ],
    },
    "list_prompt_templates": {
        "category": "Prompt Registry",
        "description": "List all managed prompt templates and their deployment labels.",
        "parameters": {
            "include_versions": "Optional: include all versions for each template (default: false)"
        },
        "examples": [{"description": "List prompt templates", "args": {}}],
        "tips": [
            "Seeded templates are created from compression presets at startup",
            "Use include_versions=true when auditing rollout history",
        ],
        "related_tools": ["get_prompt_template", "deploy_prompt_version"],
    },
    "get_prompt_template": {
        "category": "Prompt Registry",
        "description": "Resolve a prompt template to a concrete version or deployment label.",
        "parameters": {
            "name": "Prompt template name (required)",
            "version": "Optional explicit version number",
            "deployment_label": "Optional deployment label like production or staging",
        },
        "output_fields": get_prompt_output_fields(),
        "examples": [
            {"description": "Get latest prompt", "args": {"name": "review-default"}},
            {
                "description": "Resolve production prompt",
                "args": {"name": "review-default", "deployment_label": "production"},
            },
        ],
        "tips": [
            "Specify either version or deployment_label, not both",
            "Resolved labels show which rollout targets already point at the returned version",
        ],
        "related_tools": ["list_prompt_templates", "compare_prompt_versions"],
    },
    "deploy_prompt_version": {
        "category": "Prompt Registry",
        "description": "Assign a deployment label to a specific prompt version.",
        "parameters": {
            "name": "Prompt template name (required)",
            "version": "Version number to deploy (required)",
            "deployment_label": "Deployment target label such as production, staging, or canary (required)",
            "allow_stable_prefix_change": "Optional boolean acknowledgment required when redeploying a live label across a stable-prefix change",
        },
        "examples": [
            {
                "description": "Promote version 3 to production",
                "args": {
                    "name": "review-default",
                    "version": 3,
                    "deployment_label": "production",
                },
            }
        ],
        "output_fields": get_prompt_deploy_output_fields(),
        "tips": [
            "Deployment labels are mutable pointers to immutable prompt versions",
            "If stable_prefix_analysis reports a cache prefix change for an existing live label, deployments are blocked unless allow_stable_prefix_change=true is explicitly provided.",
            "Check stable_prefix_analysis before moving a live label to see whether the deployment invalidates cached prompt prefixes.",
            "Use compare_prompt_versions before moving production",
        ],
        "related_tools": ["compare_prompt_versions", "get_prompt_template"],
    },
    "compare_prompt_versions": {
        "category": "Prompt Registry",
        "description": "Compare two versions of the same prompt template with a unified diff.",
        "parameters": {
            "name": "Prompt template name (required)",
            "version_a": "Base version number (required)",
            "version_b": "Comparison version number (required)",
        },
        "examples": [
            {
                "description": "Compare prompt v1 and v2",
                "args": {"name": "review-default", "version_a": 1, "version_b": 2},
            }
        ],
        "output_fields": get_prompt_compare_output_fields(),
        "tips": [
            "Changed fields tell you whether the system prompt, user prompt, variables, or metadata moved",
            "Stable prefix analysis tells you whether the cacheable prefix changed or whether only volatile prompt parts moved.",
            "Use this before deploying a new version to production",
        ],
        "related_tools": ["update_prompt_template", "deploy_prompt_version"],
    },
    "audit_prompt_cacheability": {
        "category": "Prompt Registry",
        "description": "Audit a composed prompt for cache-friendly section ordering and volatile metadata leaks.",
        "parameters": {
            "sections": "Ordered list of prompt sections, each with a canonical name and string content (required)",
            "max_stable_prefix_chars": "Optional guard window for scanning the stable prefix for volatile content",
        },
        "output_fields": get_prompt_audit_output_fields(),
        "examples": [
            {
                "description": "Audit a prompt before sending it to a provider",
                "args": {
                    "sections": [
                        {"name": "system_instructions", "content": "You are a reviewer."},
                        {"name": "metadata", "content": '{"workspace_id":"acme"}'},
                        {"name": "user_query", "content": "Review this diff."},
                    ]
                },
            }
        ],
        "tips": [
            "Use canonical section names so the audit can verify ordering against provider cache best practices.",
            "If stable sections contain UUIDs or timestamps, move them to metadata or the immediate user query tail.",
        ],
        "related_tools": [
            "create_prompt_template",
            "get_prompt_template",
            "capture_cache_telemetry",
        ],
    },
    "render_prompt_template": {
        "category": "Prompt Registry",
        "description": "Resolve and render a prompt template into cache-friendly ordered sections for a provider call.",
        "parameters": {
            "name": "Prompt template name (required)",
            "variables": "Object of template variables used to render the prompt body",
            "version": "Optional explicit version number",
            "deployment_label": "Optional deployment label like production or staging",
            "tool_definitions": "Optional serialized tool definitions to pin in the stable prefix",
            "rag_context": "Optional static retrieved context to place before volatile sections",
            "few_shot_examples": "Optional list of few-shot examples",
            "chat_history": "Optional list of prior conversation turns",
            "metadata": "Optional dynamic metadata to place in the volatile tail",
            "enforce_stability": "When true, reject rendered prompts that fail the stability guard",
            "max_stable_prefix_chars": "Optional guard window for scanning the stable prefix for volatile content",
        },
        "output_fields": get_prompt_render_output_fields(),
        "examples": [
            {
                "description": "Render a prompt for a provider call",
                "args": {
                    "name": "review-default",
                    "variables": {"diff": "print('hi')"},
                    "metadata": {"request_id": "req-123"},
                },
            }
        ],
        "tips": [
            "Use this instead of hand-concatenating prompt strings when you want cache-friendly section ordering by default.",
            "Reuse the returned prompt_id when you later call capture_cache_telemetry so the system can validate whether a cache-friendly render actually hit provider cache.",
            "Pair this with audit_prompt_cacheability and capture_cache_telemetry to verify both structure and provider outcomes.",
        ],
        "related_tools": [
            "create_prompt_template",
            "get_prompt_template",
            "audit_prompt_cacheability",
            "capture_cache_telemetry",
        ],
    },
    "list_prefix_collisions": {
        "category": "Prompt Registry",
        "description": "List cacheable prefix collisions across rendered prompts so shared stable prefixes are visible across templates.",
        "parameters": {},
        "output_fields": get_prefix_collision_output_fields(),
        "examples": [{"description": "List detected prefix collisions", "args": {}}],
        "tips": [
            "Collisions mean multiple templates currently share the same provider-cacheable prefix.",
            "Use this before deploys to understand cross-template cache reuse or interference.",
        ],
        "related_tools": [
            "render_prompt_template",
            "deploy_prompt_version",
            "capture_cache_telemetry",
        ],
    },
    "add_memory": {
        "category": "Memory API",
        "description": "Store an explicit user or agent memory independently of document ingestion.",
        "parameters": {
            "text": "Memory text to store (required)",
            "category": "Optional explicit category override",
            "source": "Optional source tag such as manual, hook, or import",
            "file_id": "Optional source document identifier",
            "metadata": "Optional structured metadata",
            "workspace_id": "Optional workspace scope",
            "user_id": "Optional user scope",
            "agent_id": "Optional agent scope",
            "session_id": "Optional session scope",
        },
        "output_fields": get_memory_output_fields(),
        "examples": [
            {
                "description": "Store a coding preference",
                "args": {
                    "text": "Prefer pytest fixtures for state isolation.",
                    "user_id": "alice",
                    "workspace_id": "acme",
                },
            }
        ],
        "tips": [
            "Use explicit memories for preferences and decisions that should survive beyond one prompt",
            "Tenant scope fields keep user and workspace memories isolated",
        ],
        "related_tools": ["search_memory", "list_memories", "get_user_profile"],
    },
    "search_memory": {
        "category": "Memory API",
        "description": "Search explicit memories by semantic-ish lexical overlap within the requested scope.",
        "parameters": {
            "query": "Search query (required)",
            "top_k": "Optional result count limit (default: 5)",
            "category": "Optional category filter",
            "workspace_id": "Optional workspace scope",
            "user_id": "Optional user scope",
            "agent_id": "Optional agent scope",
            "session_id": "Optional session scope",
        },
        "output_fields": get_memory_output_fields(),
        "examples": [
            {
                "description": "Find stored test preferences",
                "args": {"query": "pytest fixtures", "user_id": "alice"},
            }
        ],
        "tips": [
            "Search is scope-aware, so pass user/workspace fields when memories should stay isolated",
            "Use category when you only want decisions, issues, or patterns",
        ],
        "related_tools": ["add_memory", "list_memories", "summarize_user_memory"],
    },
    "list_memories": {
        "category": "Memory API",
        "description": "List explicit memories in the requested scope.",
        "parameters": {
            "category": "Optional category filter",
            "limit": "Optional max number of memories to return",
            "workspace_id": "Optional workspace scope",
            "user_id": "Optional user scope",
            "agent_id": "Optional agent scope",
            "session_id": "Optional session scope",
        },
        "output_fields": get_memory_output_fields(),
        "examples": [{"description": "List one user's memories", "args": {"user_id": "alice"}}],
        "tips": [
            "Without scope fields, only unscoped memories are listed",
            "Use this to audit what will feed profile synthesis",
        ],
        "related_tools": ["search_memory", "delete_memory", "get_user_profile"],
    },
    "delete_memory": {
        "category": "Memory API",
        "description": "Delete an explicit memory by ID within the requested scope.",
        "parameters": {
            "memory_id": "Memory identifier (required)",
            "workspace_id": "Optional workspace scope",
            "user_id": "Optional user scope",
            "agent_id": "Optional agent scope",
            "session_id": "Optional session scope",
        },
        "examples": [
            {"description": "Delete one memory", "args": {"memory_id": "mem_1", "user_id": "alice"}}
        ],
        "tips": [
            "Deletion is scope-aware to avoid cross-tenant accidental removal",
        ],
        "related_tools": ["list_memories", "add_memory"],
    },
    "summarize_user_memory": {
        "category": "Memory API",
        "description": "Summarize one user's explicit memories into reusable preference and topic signals.",
        "parameters": {
            "user_id": "User identifier (required)",
            "workspace_id": "Optional workspace scope",
            "agent_id": "Optional agent scope",
            "session_id": "Optional session scope",
        },
        "output_fields": get_user_profile_output_fields(),
        "examples": [
            {
                "description": "Summarize a user",
                "args": {"user_id": "alice", "workspace_id": "acme"},
            }
        ],
        "tips": [
            "Use this for a compact memory briefing before model invocation",
        ],
        "related_tools": ["get_user_profile", "search_memory"],
    },
    "get_user_profile": {
        "category": "Personalization",
        "description": "Build a deterministic user profile from explicit stored memories.",
        "parameters": {
            "user_id": "User identifier (required)",
            "workspace_id": "Optional workspace scope",
            "agent_id": "Optional agent scope",
            "session_id": "Optional session scope",
        },
        "output_fields": get_user_profile_output_fields(),
        "examples": [
            {"description": "Build a profile", "args": {"user_id": "alice", "workspace_id": "acme"}}
        ],
        "tips": [
            "Profiles are derived from explicit memories, not hidden model state",
            "Store preferences with add_memory first for best results",
        ],
        "related_tools": ["summarize_user_memory", "add_memory", "list_memories"],
    },
    "create_dataset": {
        "category": "Experiments",
        "description": "Create a reusable named dataset from inline cases or a benchmark corpus file.",
        "parameters": {
            "name": "Dataset name (required)",
            "description": "Human-readable dataset description (required)",
            "cases": "Optional inline benchmark cases",
            "source_path": "Optional JSON corpus path to load cases from",
            "metadata": "Optional structured dataset metadata",
        },
        "output_fields": get_dataset_output_fields(),
        "examples": [
            {
                "description": "Create dataset from benchmark fixture",
                "args": {
                    "name": "release-gate",
                    "description": "Release benchmark subset",
                    "source_path": "tests\\fixtures\\benchmark_corpus.json",
                },
            }
        ],
        "tips": [
            "Use named datasets so experiment runs are reproducible and comparable",
        ],
        "related_tools": ["list_datasets", "run_experiment"],
    },
    "list_datasets": {
        "category": "Experiments",
        "description": "List reusable datasets available for experiment runs.",
        "parameters": {},
        "output_fields": get_dataset_output_fields(),
        "examples": [{"description": "List datasets", "args": {}}],
        "tips": [
            "The default benchmark corpus is seeded automatically for immediate experiment runs",
        ],
        "related_tools": ["create_dataset", "run_experiment"],
    },
    "run_experiment": {
        "category": "Experiments",
        "description": "Execute a tracked experiment run over a named dataset and store summary, verification, and reward outputs.",
        "parameters": {
            "dataset_name": "Dataset name to run (required)",
            "mode": "Optional benchmark mode: baseline, query_guided, or evidence_aware",
            "case_ids": "Optional subset of case IDs",
            "similarity_threshold": "Optional similarity threshold override",
            "skeleton_ratio": "Optional skeleton ratio override",
            "baseline_run_id": "Optional baseline run for later comparisons",
            "metadata": "Optional structured run metadata",
        },
        "output_fields": get_experiment_output_fields(),
        "examples": [
            {
                "description": "Run a query-guided experiment",
                "args": {
                    "dataset_name": "benchmark-corpus",
                    "mode": "query_guided",
                    "case_ids": ["medium_architecture"],
                },
            }
        ],
        "tips": [
            "Tracked runs include benchmark summary, contract verification, and reward decomposition",
        ],
        "related_tools": ["get_experiment_run", "compare_experiment_runs", "list_datasets"],
    },
    "get_experiment_run": {
        "category": "Experiments",
        "description": "Fetch a stored experiment run by ID.",
        "parameters": {"run_id": "Experiment run identifier (required)"},
        "output_fields": get_experiment_output_fields(),
        "examples": [{"description": "Fetch run details", "args": {"run_id": "exp_1"}}],
        "tips": [
            "Use this to inspect per-case verification and reward outputs after a run finishes",
        ],
        "related_tools": ["run_experiment", "compare_experiment_runs"],
    },
    "compare_experiment_runs": {
        "category": "Experiments",
        "description": "Compare two stored experiment runs and report deltas in quality and token-savings metrics.",
        "parameters": {
            "run_id_a": "Base run identifier (required)",
            "run_id_b": "Comparison run identifier (required)",
        },
        "output_fields": get_experiment_output_fields(),
        "examples": [
            {"description": "Compare two runs", "args": {"run_id_a": "exp_1", "run_id_b": "exp_2"}}
        ],
        "tips": [
            "Use this to quantify whether a new mode or threshold actually improved results",
        ],
        "related_tools": ["run_experiment", "get_experiment_run"],
    },
    "list_connector_types": {
        "category": "Connectors",
        "description": "List supported managed connector types.",
        "parameters": {},
        "output_fields": get_connector_output_fields(),
        "examples": [{"description": "List connector types", "args": {}}],
        "tips": ["Connector feeds normalize exported source payloads into ingestible documents."],
        "related_tools": ["create_connector_feed", "list_connector_feeds"],
    },
    "create_connector_feed": {
        "category": "Connectors",
        "description": "Create a managed connector feed definition for later sync runs.",
        "parameters": {
            "name": "Feed name (required)",
            "connector_type": "Connector type: web, github, s3, or slack_export (required)",
            "config": "Connector-specific exported payload config (required)",
            "metadata": "Optional feed metadata",
        },
        "output_fields": get_connector_output_fields(),
        "examples": [
            {
                "description": "Create a web feed",
                "args": {
                    "name": "docs-web",
                    "connector_type": "web",
                    "config": {
                        "pages": [
                            {"url": "https://example.com/docs", "content": "Example docs body"}
                        ]
                    },
                },
            }
        ],
        "tips": ["Create feeds once, then resync them with sync_connector_feed."],
        "related_tools": ["list_connector_types", "sync_connector_feed", "get_connector_feed"],
    },
    "list_connector_feeds": {
        "category": "Connectors",
        "description": "List managed connector feeds and their last sync state.",
        "parameters": {},
        "output_fields": get_connector_output_fields(),
        "examples": [{"description": "List feeds", "args": {}}],
        "tips": ["Use get_connector_feed to inspect the full stored feed config."],
        "related_tools": ["create_connector_feed", "get_connector_feed"],
    },
    "get_connector_feed": {
        "category": "Connectors",
        "description": "Fetch one managed connector feed definition.",
        "parameters": {"name": "Feed name (required)"},
        "output_fields": get_connector_output_fields(),
        "examples": [{"description": "Get one feed", "args": {"name": "docs-web"}}],
        "tips": ["Feed configs are stored so syncs can be repeated consistently."],
        "related_tools": ["list_connector_feeds", "sync_connector_feed"],
    },
    "sync_connector_feed": {
        "category": "Connectors",
        "description": "Normalize a managed connector feed and ingest its documents through the standard compression pipeline.",
        "parameters": {
            "name": "Feed name (required)",
            "workspace_id": "Optional workspace scope",
            "user_id": "Optional user scope",
            "agent_id": "Optional agent scope",
            "session_id": "Optional session scope",
        },
        "output_fields": get_connector_output_fields(),
        "examples": [{"description": "Sync a feed", "args": {"name": "docs-web"}}],
        "tips": [
            "Connector syncs respect resource quotas and write sync metadata for external sources.",
            "Use scope fields to isolate connector-ingested documents per tenant.",
        ],
        "related_tools": ["create_connector_feed", "list_connector_feeds", "list_connector_types"],
    },
    "get_context_block": {
        "category": "Temporal Context",
        "description": "Build a lifecycle-aware context block with active facts, recent events, and a cache-stable skeleton prefix.",
        "parameters": {
            "file_id": "Document ID (required)",
            "query": "Optional query to bias the skeleton selection",
            "as_of": "Optional ISO-8601 or unix timestamp reference time",
            "max_facts": "Maximum active facts to include (default: 5)",
            "limit": "Maximum timeline events to include (default: 10)",
            "include_invalidated": "Include invalidated facts/events instead of hiding them",
            "workspace_id": "Optional workspace scope",
            "user_id": "Optional user scope",
            "agent_id": "Optional agent scope",
            "session_id": "Optional session scope",
        },
        "output_fields": get_temporal_output_fields(),
        "examples": [
            {"description": "Build temporal context block", "args": {"file_id": "design_doc"}}
        ],
        "tips": [
            "Use this to assemble recency-aware context before sending a prompt to an LLM.",
        ],
        "related_tools": ["search_timeline", "list_fact_history", "invalidate_fact"],
    },
    "search_timeline": {
        "category": "Temporal Context",
        "description": "Search lifecycle events such as ingests, reads, searches, and invalidations.",
        "parameters": {
            "query": "Optional free-text filter across event summaries and metadata",
            "file_id": "Optional document ID filter",
            "fact_id": "Optional exact fact ID filter",
            "event_types": "Optional list of event types",
            "since": "Optional lower time bound",
            "until": "Optional upper time bound",
            "limit": "Maximum events to return (default: 25)",
            "include_invalidated": "Include invalidation/supersession events",
            "workspace_id": "Optional workspace scope",
            "user_id": "Optional user scope",
            "agent_id": "Optional agent scope",
            "session_id": "Optional session scope",
        },
        "output_fields": get_temporal_output_fields(),
        "examples": [
            {"description": "Search document timeline", "args": {"file_id": "design_doc"}}
        ],
        "tips": [
            "Timeline search returns chronological lifecycle events for documents and facts.",
        ],
        "related_tools": ["get_context_block", "list_fact_history"],
    },
    "list_fact_history": {
        "category": "Temporal Context",
        "description": "List temporal fact versions for a document or exact fact ID.",
        "parameters": {
            "file_id": "Optional document ID filter",
            "fact_id": "Optional exact fact ID filter",
            "as_of": "Optional reference time",
            "include_invalidated": "Include invalidated versions",
            "workspace_id": "Optional workspace scope",
            "user_id": "Optional user scope",
            "agent_id": "Optional agent scope",
            "session_id": "Optional session scope",
        },
        "output_fields": get_temporal_output_fields(),
        "examples": [{"description": "List document facts", "args": {"file_id": "design_doc"}}],
        "tips": ["Returned fact IDs can be passed back to invalidate_fact."],
        "related_tools": ["search_timeline", "invalidate_fact", "get_context_block"],
    },
    "invalidate_fact": {
        "category": "Temporal Context",
        "description": "Invalidate a previously observed fact so it is hidden from temporal retrieval by default.",
        "parameters": {
            "fact_id": "Exact fact ID to invalidate (required)",
            "reason": "Human-readable invalidation reason (required)",
            "timestamp": "Optional timestamp for the invalidation event",
        },
        "output_fields": get_temporal_output_fields(),
        "examples": [
            {
                "description": "Invalidate a stale fact",
                "args": {"fact_id": "design_doc_n0", "reason": "Superseded by newer design"},
            }
        ],
        "tips": ["Invalidated facts remain in history but are excluded from retrieval by default."],
        "related_tools": ["list_fact_history", "search_timeline", "get_context_block"],
    },
    "should_compress": {
        "category": "Resource Management",
        "description": "Estimate file size/token cost before ingestion and recommend whether to compress, read directly, or convert first.",
        "parameters": {
            "file_path": "Absolute or validated file path to inspect (required)",
            "content_type": "Optional hint: auto, prose, or code",
        },
        "examples": [
            {"description": "Preflight a file", "args": {"file_path": "C:\\project\\README.md"}},
            {
                "description": "Hint code-like content",
                "args": {"file_path": "C:\\project\\main.py", "content_type": "code"},
            },
        ],
        "tips": [
            "Use this before ingest_context when reading from disk",
            "Binary files may return CONVERT_THEN_COMPRESS with a suggested conversion workflow",
        ],
        "related_tools": ["ingest_context", "check_environment", "check_context_budget"],
    },
    "check_context_budget": {
        "category": "Context Budget",
        "description": "Check current context window usage and get threshold-based recommendations before you overflow the model window.",
        "parameters": {
            "current_tokens": "Current tokens already in prompt/context (required)",
            "context_limit": "Optional max context window size (default: 200000)",
        },
        "examples": [
            {"description": "Check budget health", "args": {"current_tokens": 120000}},
            {
                "description": "Check against smaller window",
                "args": {"current_tokens": 28000, "context_limit": 32000},
            },
        ],
        "tips": [
            "Use after ingesting several documents or before adding more context",
            "Pair with advise_context for cross-document strategy decisions",
        ],
        "related_tools": ["adapt_to_context_window", "advise_context", "should_compress"],
    },
    "prune_by_relevance": {
        "category": "Document Compression",
        "description": "Keep only the most query-relevant nodes from an ingested document for aggressive task-focused compression.",
        "parameters": {
            "doc_id": "Document ID to prune (required)",
            "query": "Query used to score relevance (required)",
            "keep_ratio": "Fraction of nodes to keep (default: 0.5)",
        },
        "examples": [
            {
                "description": "Prune to most relevant half",
                "args": {"doc_id": "auth_doc", "query": "token refresh flow"},
            },
        ],
        "tips": [
            "Use when you need a task-specific subset instead of a general skeleton",
            "Follow with modulate_region on kept_node_ids for detailed retrieval",
        ],
        "related_tools": ["read_skeleton", "search_semantic", "modulate_region"],
    },
    "get_multi_level_skeleton": {
        "category": "Document Compression",
        "description": "Return 3 fixed-depth skeleton tiers so the client can choose how much detail to consume.",
        "parameters": {
            "doc_id": "Document ID to summarize at multiple levels (required)",
        },
        "examples": [
            {
                "description": "Get headline/summary/full tiers",
                "args": {"doc_id": "system_design_doc"},
            },
        ],
        "tips": [
            "Use when you want deterministic depth tiers instead of query-guided selection",
            "Prefer read_skeleton for task-aware selection modes",
        ],
        "related_tools": ["read_skeleton", "modulate_region", "advise_context"],
    },
    "advise_context": {
        "category": "Context Budget",
        "description": "Analyze all ingested documents and recommend which models, budgets, and compression actions fit the current workload.",
        "parameters": {},
        "examples": [
            {"description": "Get global context strategy", "args": {}},
        ],
        "tips": [
            "Use after multiple documents are ingested to prioritize what should stay in context",
            "Combine with check_context_budget for real-time token pressure monitoring",
        ],
        "related_tools": ["check_context_budget", "adapt_to_context_window", "list_documents"],
    },
    # === AFM Dialogue Tools ===
    "afm_add_message": {
        "category": "AFM Dialogue",
        "description": "Add a message to the dialogue history with importance classification.",
        "parameters": {
            "role": "Message role: 'user' or 'assistant' (required)",
            "content": "Message content (required)",
            "importance_override": "Optional: force HIGH, MEDIUM, or LOW importance",
        },
        "examples": [
            {
                "description": "Add user message",
                "args": {"role": "user", "content": "I have a peanut allergy"},
            },
            {
                "description": "Add assistant response",
                "args": {"role": "assistant", "content": "Noted, I'll remember that."},
            },
        ],
        "tips": [
            "Safety messages (allergies, constraints) are auto-classified as HIGH",
            "Messages decay in importance over time (half-life: 12 turns)",
            "Use afm_build_context to get compressed context for LLM",
        ],
        "related_tools": ["afm_build_context", "afm_get_stats", "afm_clear_history"],
    },
    "afm_build_context": {
        "category": "AFM Dialogue",
        "description": "Build compressed dialogue context within a token budget.",
        "parameters": {
            "query": "Current user query (required)",
            "budget_tokens": "Maximum tokens for context (default: 1000)",
        },
        "examples": [
            {
                "description": "Build context",
                "args": {"query": "What can I eat?", "budget_tokens": 500},
            },
        ],
        "tips": [
            "Safety messages are always preserved regardless of budget",
            "Returns both context text and statistics",
            "Use token savings stats to optimize budget allocation",
        ],
        "related_tools": ["afm_add_message", "afm_get_stats"],
    },
    # === ACE Framework Tools ===
    "ace_generate": {
        "category": "ACE Framework",
        "description": "Generate context bullets from task outcome for playbook evolution.",
        "parameters": {
            "task": "Task description (required)",
            "outcome": "Task outcome/result (required)",
            "success": "Whether task succeeded (required)",
            "context_id": "Optional context ID for organizing bullets",
        },
        "examples": [
            {
                "description": "Generate from successful task",
                "args": {"task": "Fix auth bug", "outcome": "Added null check", "success": True},
            },
        ],
        "tips": [
            "Generated bullets are deduplicated at 0.85 similarity threshold",
            "Use ace_execute_cycle for automated Generate->Reflect->Curate",
            "Bullets are ranked by novelty and usefulness",
        ],
        "related_tools": ["ace_reflect", "ace_curate", "ace_execute_cycle"],
    },
    # === File Sync Tools ===
    "check_file_sync": {
        "category": "File Sync",
        "description": "Check if a tracked file has changed since ingestion.",
        "parameters": {
            "file_id": "ID of the document to check (required)",
        },
        "examples": [
            {"description": "Check sync status", "args": {"file_id": "main.py"}},
        ],
        "tips": [
            "Uses mtime + MD5 checksum for change detection",
            "Returns 'stale' if file changed, 'synced' if unchanged",
            "Use refresh_document to re-ingest stale files",
        ],
        "related_tools": ["refresh_document", "diff_cached_file", "get_version_history"],
    },
    "refresh_document": {
        "category": "File Sync",
        "description": "Re-ingest a document from its source file path.",
        "parameters": {
            "file_id": "ID of the document to refresh (required)",
        },
        "examples": [
            {"description": "Refresh stale document", "args": {"file_id": "config.py"}},
        ],
        "tips": [
            "Requires file_path to have been provided during initial ingest",
            "Creates version history entry for the change",
            "Use after check_file_sync reports 'stale' status",
        ],
        "related_tools": ["check_file_sync", "get_version_history", "ingest_context"],
    },
    # === Detection Tools ===
    "check_blind_spots": {
        "category": "Detection",
        "description": "Detect relevant content that may have been missed in retrieval.",
        "parameters": {
            "file_id": "Document to check (required)",
            "retrieved_nodes": "List of node IDs already retrieved (required)",
            "query": "Optional: user's query for context",
        },
        "examples": [
            {
                "description": "Check for blind spots",
                "args": {
                    "file_id": "manual",
                    "retrieved_nodes": ["manual_n0", "manual_n1"],
                    "query": "How to configure auth?",
                },
            },
        ],
        "tips": [
            "Returns urgency score (0-1) for each potential blind spot",
            "High urgency (>0.5) suggests important content was missed",
            "Use after search_semantic to catch gaps in retrieval",
        ],
        "related_tools": ["search_semantic", "detect_hallucination"],
    },
    # === Resource Management ===
    "check_resource_health": {
        "category": "Resource Management",
        "description": "Check storage, memory, and document count metrics.",
        "parameters": {},
        "examples": [
            {"description": "Check health", "args": {}},
        ],
        "tips": [
            "Shows warnings when approaching resource limits",
            "Includes recommendations for optimization",
            "Storage limit: 1GB, Document limit: 1000",
        ],
        "related_tools": ["check_environment", "list_documents", "delete_document"],
    },
    "check_environment": {
        "category": "Resource Management",
        "description": (
            "Check environment health: models, memory, cache, disk space, and MCP tool profile."
        ),
        "parameters": {},
        "output_fields": get_check_environment_output_fields(),
        "examples": [
            {"description": "Check environment", "args": {}},
        ],
        "tips": [
            "Shows which embedding models are loaded",
            "Reports cache hit ratio for performance tuning",
            "Lists any stale documents that need refresh",
            "Includes tool_profile diagnostics (active profile and enabled_tools list)",
        ],
        "related_tools": ["check_resource_health", "check_file_sync"],
    },
    # === Utility Tools ===
    "recommend_fidelity": {
        "category": "Fidelity Advisor",
        "description": "Get recommendation for optimal fidelity level based on use case.",
        "parameters": {
            "use_case": "What you want to do (e.g., 'quick_summary', 'detailed_analysis')",
            "num_nodes": "Number of nodes you plan to retrieve",
            "token_budget": "Optional maximum tokens available",
            "query_complexity": "Optional: 'simple', 'medium', or 'complex'",
        },
        "output_fields": get_recommend_fidelity_output_fields(),
        "examples": [
            {
                "description": "Get recommendation for summary",
                "args": {"use_case": "quick_summary", "num_nodes": 3},
            },
            {
                "description": "With token budget",
                "args": {"use_case": "question_answering", "num_nodes": 5, "token_budget": 1000},
            },
        ],
        "tips": [
            "Use this BEFORE modulate_region to make informed choices",
            "Returns token estimate for each fidelity level",
            "Considers context to suggest alternatives",
        ],
        "related_tools": ["modulate_region", "search_semantic"],
    },
}


_AUTO_CATEGORY_BY_PREFIX = {
    "ace_": "ACE Framework",
    "afm_": "AFM Dialogue",
}

_AUTO_CATEGORY_BY_TOOL = {
    "adapt_to_context_window": "Document Compression",
    "batch_ingest_documents": "Batch Processing",
    "calculate_reward": "Experimental",
    "check_blind_spots": "Detection",
    "delete_document": "Document Compression",
    "detect_hallucination": "Detection",
    "diff_cached_file": "File Sync",
    "diff_reingest": "File Sync",
    "evict_stale": "Document Compression",
    "explain_compression_decision": "Visualization",
    "export_graph_graphml": "Visualization",
    "export_graph_json": "Visualization",
    "find_duplicates": "Document Compression",
    "generate_rewrite_prompt": "Document Compression",
    "generate_synthetic_tests": "Experimental",
    "get_compression_insights": "Document Compression",
    "get_compression_presets": "Document Compression",
    "get_evidence_stats": "Experimental",
    "get_stats": "Document Compression",
    "get_version_history": "File Sync",
    "ingest_directory": "Directory Ingestion",
    "ingest_multimodal": "Multimodal",
    "list_documents": "Document Compression",
    "multilevel_encode": "Document Compression",
    "multimodal_ingest": "Multimodal",
    "recommend_fidelity": "Fidelity Advisor",
    "scar_compress": "Experimental",
    "scar_get_stats": "Experimental",
    "tool_help": "Help & Documentation",
    "toon_decode": "Experimental",
    "toon_encode": "Experimental",
    "verify_compression": "Experimental",
    "visualize_graph_html": "Visualization",
}

_AUTO_RELATED_TOOLS = {
    "tool_help": ["ingest_context", "read_skeleton", "search_semantic"],
    "list_documents": ["get_stats", "delete_document", "read_skeleton"],
    "delete_document": ["list_documents", "ingest_context", "get_stats"],
    "batch_ingest_documents": ["ingest_context", "ingest_directory", "list_documents"],
    "detect_hallucination": ["search_semantic", "read_skeleton", "verify_compression"],
    "adapt_to_context_window": ["check_context_budget", "advise_context", "recommend_fidelity"],
    "multilevel_encode": ["read_skeleton", "get_multi_level_skeleton", "adapt_to_context_window"],
    "diff_cached_file": ["check_file_sync", "refresh_document", "get_version_history"],
    "diff_reingest": ["diff_cached_file", "refresh_document", "get_version_history"],
    "evict_stale": ["check_file_sync", "refresh_document", "list_documents"],
    "export_graph_json": [
        "visualize_graph_html",
        "export_graph_graphml",
        "explain_compression_decision",
    ],
    "visualize_graph_html": ["export_graph_json", "export_graph_graphml", "read_skeleton"],
    "toon_encode": ["toon_decode", "verify_compression", "create_handoff_bundle"],
    "verify_compression": ["calculate_reward", "get_evidence_stats", "read_skeleton"],
}


def _infer_tool_category(tool_name: str) -> str:
    for prefix, category in _AUTO_CATEGORY_BY_PREFIX.items():
        if tool_name.startswith(prefix):
            return category
    return _AUTO_CATEGORY_BY_TOOL.get(tool_name, "General")


def _placeholder_value(param_name: str, schema: Dict[str, Any]) -> Any:
    explicit_values = {
        "file_id": "doc_1",
        "doc_id": "doc_1",
        "bundle_id": "bundle_1",
        "memory_id": "mem_1",
        "run_id": "run_1",
        "dataset_name": "release-gate",
        "name": "example-name",
        "query": "authentication flow",
        "text": "Representative context text",
        "directory": ".",
        "file_path": "src\\app.py",
        "content_type": "code",
        "model": "claude-sonnet-4.6",
        "workspace_id": "acme",
        "user_id": "alice",
        "agent_id": "assistant",
        "session_id": "session-1",
        "version": 1,
        "top_k": 5,
        "limit": 10,
        "keep_ratio": 0.5,
        "token_budget": 1200,
        "current_tokens": 4000,
        "context_limit": 8000,
        "num_nodes": 5,
        "original_tokens": 100000,
        "compressed_tokens": 20000,
        "max_concurrent": 4,
        "verbose": True,
    }
    if param_name in explicit_values:
        return explicit_values[param_name]

    schema_type = schema.get("type")
    if schema_type == "integer":
        return 1
    if schema_type == "number":
        return 0.5
    if schema_type == "boolean":
        return True
    if schema_type == "array":
        return []
    if schema_type == "object":
        return {}
    return "example"


def _schema_parameters(input_schema: Dict[str, Any]) -> Dict[str, str]:
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))
    parameters: Dict[str, str] = {}
    for name, schema in properties.items():
        description = schema.get("description") or f"{name} parameter"
        if name in required and "(required)" not in description.lower():
            description = f"{description} (required)"
        parameters[name] = description
    return parameters


def _schema_example_args(input_schema: Dict[str, Any]) -> Dict[str, Any]:
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])
    if not required:
        return {}
    return {name: _placeholder_value(name, properties.get(name, {})) for name in required}


def _auto_related_tools(tool_name: str, category: str) -> list[str]:
    related = _AUTO_RELATED_TOOLS.get(tool_name)
    if related:
        return related

    registry = TOOL_HELP_REGISTRY
    same_category = [
        name
        for name, info in registry.items()
        if info.get("category") == category and name != tool_name
    ]
    return same_category[:3]


@lru_cache(maxsize=1)
def get_tool_help_registry() -> Dict[str, Dict[str, Any]]:
    """Return the merged help registry for all registered MCP tools."""
    registry = {name: dict(info) for name, info in TOOL_HELP_REGISTRY.items()}

    from .mcp_core import setup_mcp_tools

    for tool in setup_mcp_tools():
        if tool.name in registry:
            continue

        category = _infer_tool_category(tool.name)
        parameters = _schema_parameters(tool.inputSchema)
        example_args = _schema_example_args(tool.inputSchema)

        registry[tool.name] = {
            "category": category,
            "description": tool.description,
            "parameters": parameters,
            "examples": [
                {
                    "description": f"Basic {tool.name} invocation",
                    "args": example_args,
                }
            ],
            "tips": [
                "Use tool_help with verbose=true to inspect arguments before first use.",
                "Pair this tool with the related tools below when building a longer workflow.",
            ],
            "related_tools": _auto_related_tools(tool.name, category),
        }

    return registry


async def handle_tool_help(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Handle tool_help MCP tool (v0.9.0).

    Provides detailed help, examples, and tips for any Semantic Modulator tool.

    Args:
        context: Server context (unused for help)
        args: Tool arguments:
            - tool_name: Name of tool to get help for (required)
            - verbose: Include full examples (default: False)

    Returns:
        JSON string with structured help information
    """
    tool_name = args.get("tool_name", "")
    verbose = args.get("verbose", False)
    registry = get_tool_help_registry()

    if not tool_name:
        # Return list of all tools with categories
        categories: Dict[str, list] = {}
        for name, info in registry.items():
            cat = info.get("category", "Other")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(
                {
                    "name": name,
                    "description": info.get("description", "")[:80] + "...",
                }
            )

        return json.dumps(
            {
                "status": "tool_list",
                "message": "Specify tool_name to get detailed help",
                "available_tools": categories,
                "total_tools": len(registry),
                "recommended_workflow": {
                    "description": "Optimal tool sequence for maximum token savings",
                    "steps": [
                        {
                            "step": 1,
                            "tool": "should_compress",
                            "purpose": "Check if compression is worthwhile for your content size",
                        },
                        {
                            "step": 2,
                            "tool": "ingest_context",
                            "purpose": "Ingest document into semantic graph (use chunking_strategy='semantic' for best results)",
                        },
                        {
                            "step": 3,
                            "tool": "read_skeleton",
                            "purpose": "Get compressed view (80-95% token reduction). Use anchored_keywords to preserve critical terms",
                        },
                        {
                            "step": 4,
                            "tool": "search_semantic",
                            "purpose": "Find specific information within compressed docs",
                        },
                        {
                            "step": 5,
                            "tool": "modulate_region",
                            "purpose": "Zoom into specific nodes at chosen fidelity level",
                        },
                        {
                            "step": 6,
                            "tool": "advise_context",
                            "purpose": "Get model-specific optimization recommendations",
                        },
                    ],
                },
                "tool_profiles": {
                    "core_stable": {
                        "tools": [
                            "ingest_context",
                            "read_skeleton",
                            "modulate_region",
                            "search_semantic",
                            "get_stats",
                            "list_documents",
                            "delete_document",
                        ],
                        "description": "7 essential tools (~3K tokens). Best for prompt-cache-friendly setups.",
                    },
                    "full": {
                        "tools": f"All {len(registry)} tools",
                        "description": "Complete toolkit (~16K tokens). Use when context budget allows.",
                    },
                },
            },
            indent=2,
        )

    # Look up specific tool
    if tool_name not in registry:
        # Suggest similar tools
        suggestions = [
            name
            for name in registry.keys()
            if tool_name.lower() in name.lower() or name.lower() in tool_name.lower()
        ]

        return json.dumps(
            {
                "status": "not_found",
                "tool_name": tool_name,
                "message": f"Tool '{tool_name}' not found in help registry.",
                "suggestions": suggestions[:5] if suggestions else [],
                "tip": "Use tool_help without tool_name to see all available tools.",
            },
            indent=2,
        )

    # Return help for specific tool
    info = registry[tool_name]
    result = {
        "tool": tool_name,
        "category": info.get("category", "Other"),
        "description": info.get("description", ""),
        "parameters": info.get("parameters", {}),
        "tips": info.get("tips", []),
        "related_tools": info.get("related_tools", []),
    }
    if "output_fields" in info:
        result["output_fields"] = info.get("output_fields", [])

    if verbose:
        result["examples"] = info.get("examples", [])
    else:
        # Just show first example
        examples = info.get("examples", [])
        if examples:
            result["example"] = examples[0]
            if len(examples) > 1:
                result["more_examples"] = f"Use verbose=true to see {len(examples)} examples"

    return json.dumps(result, indent=2)
