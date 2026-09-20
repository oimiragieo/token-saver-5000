"""Typed source of truth for the engine MCP tool catalogue.

``setup_mcp_tools`` and ``route_tool_call`` used to carry separate schema and
router tables.  That made adding a tool a two-edit operation and allowed the
two tables to drift.  The registry keeps the existing ``Tool`` objects and
handler module names, while giving both consumers one typed metadata record.

Handlers are bound through a tiny live proxy rather than capturing the
function object at import time.  This is intentional: the test suite and
hosted reload path patch handler module attributes, and routing must continue
to observe those patches exactly as it did before the registry was introduced.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from mcp.types import Tool

from .. import ace_handlers as ace
from .. import afm_handlers as afm
from .. import bundle_handlers as bh
from .. import compression_handlers as ch
from .. import connector_handlers as coh
from .. import detection_handlers as dh
from .. import docs_handlers as doch
from .. import experiment_handlers as eh
from .. import experimental_handlers as exp
from .. import file_sync_handlers as fs
from .. import help_handlers as hh
from .. import memory_handlers as mh
from .. import model_handlers as moh
from .. import multimodal_handlers as mmh
from .. import prompt_handlers as ph
from .. import resource_handlers as rh
from .. import temporal_handlers as th
from .. import token_optimization_handlers as toh
from .. import visualization_handlers as vh
from ._constants import CORE_STABLE_TOOL_NAMES
from ._profile import _normalize_tool_profile
from .schemas_afm_temporal import AFM_TOOLS, TEMPORAL_TOOLS
from .schemas_compression import COMPRESSION_TOOLS
from .schemas_experimental import EXPERIMENTAL_TOOLS
from .schemas_filesync_bundle import BUNDLE_TOOLS, FILESYNC_TOOLS
from .schemas_memory import MEMORY_TOOLS
from .schemas_misc import CONNECTOR_TOOLS, DETECTION_TOOLS, DOCS_TOOLS, HELP_TOOLS, RESOURCE_TOOLS
from .schemas_model_experiment import EXPERIMENT_TOOLS, MODEL_TOOLS
from .schemas_multimodal_viz import MULTIMODAL_TOOLS, VISUALIZATION_TOOLS
from .schemas_prompts_ace import ACE_TOOLS, PROMPT_TOOLS
from .schemas_token_optimization import TOKEN_OPTIMIZATION_TOOLS

ToolHandler = Callable[[dict[str, Any], dict[str, Any]], Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    """All runtime metadata needed to expose and execute one MCP tool.

    The boolean fields deliberately describe behavior independently from
    profile membership.  A tool may be experimental, state-changing, or
    model-dependent without those facts changing which client profile exposes
    it.  ``timeout`` and ``deps`` are intentionally conservative in this
    tranche; they provide typed homes for later execution policy without
    changing today's call behavior.  Every entry carries a nonempty fixture
    and test policy identifier so future execution policy cannot silently lose
    its characterization contract.
    """

    schema: Tool
    handler: ToolHandler
    profiles: frozenset[str] = frozenset({"full"})
    timeout: float | None = None
    deps: tuple[str, ...] = ()
    fixture_id: str = ""
    test_policy: str = ""
    read_only: bool = False
    mutates_state: bool = False
    destructive: bool = False
    idempotent: bool = False
    open_world: bool = False
    experimental: bool = False
    requires_model: bool = False
    requires_context: bool = True
    requires_auth: bool = False

    def __post_init__(self) -> None:
        if not self.fixture_id.strip():
            raise ValueError("RegisteredTool.fixture_id must be non-empty")
        if not self.test_policy.strip():
            raise ValueError("RegisteredTool.test_policy must be non-empty")
        if self.timeout is not None and self.timeout < 0:
            raise ValueError("RegisteredTool.timeout must be non-negative")
        if any(not dependency.strip() for dependency in self.deps):
            raise ValueError("RegisteredTool.deps cannot contain empty names")
        if self.read_only and self.destructive:
            raise ValueError("RegisteredTool cannot be both read_only and destructive")
        if self.destructive and not self.mutates_state:
            raise ValueError("destructive RegisteredTool entries must mutate state")

    @property
    def name(self) -> str:
        return self.schema.name

    @property
    def tool(self) -> Tool:
        """Compatibility/readability alias for the owned MCP schema."""
        return self.schema

    @property
    def timeout_seconds(self) -> float | None:
        return self.timeout

    @property
    def dependencies(self) -> tuple[str, ...]:
        return self.deps

    @property
    def fixture(self) -> str:
        """Compatibility alias for callers that use the shorter name."""
        return self.fixture_id

    @property
    def is_read_only(self) -> bool:
        return self.read_only

    @property
    def is_mutating(self) -> bool:
        return self.mutates_state

    @property
    def is_destructive(self) -> bool:
        return self.destructive

    @property
    def is_idempotent(self) -> bool:
        return self.idempotent

    @property
    def is_open_world(self) -> bool:
        return self.open_world

    @property
    def is_experimental(self) -> bool:
        return self.experimental


def _live_handler(module: ModuleType, attribute: str) -> ToolHandler:
    """Return an async handler proxy that resolves the module attribute live."""

    async def invoke(context: dict[str, Any], args: dict[str, Any]) -> Any:
        return await getattr(module, attribute)(context, args)

    # Preserve the observable identity used by structured logging and tests.
    invoke.__name__ = attribute
    invoke.__module__ = module.__name__
    return invoke


# This is the former dispatch table, moved here so schema and handler metadata
# cannot be registered independently.  Values are module/attribute pairs so a
# monkeypatch or a hot reload of a handler remains visible to the proxy.
_HANDLER_SPECS: dict[str, tuple[ModuleType, str]] = {
    "gc_read_doc": (doch, "handle_gc_read_doc"),
    "gc_search_docs": (doch, "handle_gc_search_docs"),
    "ingest_context": (ch, "handle_ingest"),
    "read_skeleton": (ch, "handle_read_skeleton"),
    "modulate_region": (ch, "handle_modulate_region"),
    "search_semantic": (ch, "handle_search_semantic"),
    "get_stats": (ch, "handle_get_stats"),
    "list_documents": (ch, "handle_list_documents"),
    "delete_document": (ch, "handle_delete_document"),
    "adapt_to_context_window": (ch, "handle_adapt_to_context_window"),
    "multilevel_encode": (ch, "handle_multilevel_encode"),
    "batch_ingest_documents": (ch, "handle_batch_ingest"),
    "ingest_directory": (ch, "handle_ingest_directory"),
    "recommend_fidelity": (ch, "handle_recommend_fidelity"),
    "check_blind_spots": (dh, "handle_check_blind_spots"),
    "detect_hallucination": (dh, "handle_detect_hallucination"),
    "afm_add_message": (afm, "handle_afm_add_message"),
    "afm_build_context": (afm, "handle_afm_build_context"),
    "afm_get_stats": (afm, "handle_afm_get_stats"),
    "afm_clear_history": (afm, "handle_afm_clear_history"),
    "afm_export_history": (afm, "handle_afm_export_history"),
    "afm_import_history": (afm, "handle_afm_import_history"),
    "check_file_sync": (fs, "handle_check_file_sync"),
    "diff_cached_file": (fs, "handle_diff_cached_file"),
    "refresh_document": (fs, "handle_refresh_document"),
    "get_version_history": (fs, "handle_get_version_history"),
    "check_resource_health": (rh, "handle_check_resource_health"),
    "check_environment": (rh, "handle_check_environment"),
    "should_compress": (rh, "handle_should_compress"),
    "tool_help": (hh, "handle_tool_help"),
    "export_graph_json": (vh, "handle_export_graph_json"),
    "visualize_graph_html": (vh, "handle_visualize_graph_html"),
    "export_graph_graphml": (vh, "handle_export_graph_graphml"),
    "explain_compression_decision": (vh, "handle_explain_compression_decision"),
    "ace_generate": (ace, "handle_ace_generate"),
    "ace_reflect": (ace, "handle_ace_reflect"),
    "ace_curate": (ace, "handle_ace_curate"),
    "ace_grow_context": (ace, "handle_ace_grow_context"),
    "ace_refine_context": (ace, "handle_ace_refine_context"),
    "ace_get_playbook": (ace, "handle_ace_get_playbook"),
    "ace_execute_cycle": (ace, "handle_ace_execute_cycle"),
    "toon_encode": (exp, "handle_toon_encode"),
    "toon_decode": (exp, "handle_toon_decode"),
    "scar_compress": (exp, "handle_scar_compress"),
    "scar_get_stats": (exp, "handle_scar_get_stats"),
    "multimodal_ingest": (exp, "handle_multimodal_ingest"),
    "ingest_multimodal": (mmh, "handle_ingest_multimodal"),
    "search_multimodal": (mmh, "handle_search_multimodal"),
    "create_handoff_bundle": (bh, "handle_create_handoff_bundle"),
    "list_handoff_bundles": (bh, "handle_list_handoff_bundles"),
    "get_handoff_bundle": (bh, "handle_get_handoff_bundle"),
    "replay_handoff_bundle": (bh, "handle_replay_handoff_bundle"),
    "get_provider_profile": (moh, "handle_get_provider_profile"),
    "estimate_model_cost": (moh, "handle_estimate_model_cost"),
    "optimize_for_model": (moh, "handle_optimize_for_model"),
    "assess_cache_compatibility": (moh, "handle_assess_cache_compatibility"),
    "capture_cache_telemetry": (moh, "handle_capture_cache_telemetry"),
    "diagnose_cache_miss": (moh, "handle_diagnose_cache_miss"),
    "verify_compression": (exp, "handle_verify_compression"),
    "calculate_reward": (exp, "handle_calculate_reward"),
    "get_evidence_stats": (exp, "handle_get_evidence_stats"),
    "generate_synthetic_tests": (exp, "handle_generate_synthetic_tests"),
    "diff_reingest": (ch, "handle_diff_reingest"),
    "find_duplicates": (ch, "handle_find_duplicates"),
    "get_compression_presets": (ch, "handle_get_presets"),
    "create_prompt_template": (ph, "handle_create_prompt_template"),
    "update_prompt_template": (ph, "handle_update_prompt_template"),
    "list_prompt_templates": (ph, "handle_list_prompt_templates"),
    "get_prompt_template": (ph, "handle_get_prompt_template"),
    "deploy_prompt_version": (ph, "handle_deploy_prompt_version"),
    "compare_prompt_versions": (ph, "handle_compare_prompt_versions"),
    "render_prompt_template": (ph, "handle_render_prompt_template"),
    "list_prefix_collisions": (ph, "handle_list_prefix_collisions"),
    "audit_prompt_cacheability": (ph, "handle_audit_prompt_cacheability"),
    "add_memory": (mh, "handle_add_memory"),
    "search_memory": (mh, "handle_search_memory"),
    "list_memories": (mh, "handle_list_memories"),
    "delete_memory": (mh, "handle_delete_memory"),
    "summarize_user_memory": (mh, "handle_summarize_user_memory"),
    "get_user_profile": (mh, "handle_get_user_profile"),
    "ingest_transcript": (mh, "handle_ingest_transcript"),
    "compile_knowledge": (mh, "handle_compile_knowledge"),
    "get_knowledge_index": (mh, "handle_get_knowledge_index"),
    "lint_knowledge": (mh, "handle_lint_knowledge"),
    "search_memory_index": (mh, "handle_search_memory_index"),
    "create_dataset": (eh, "handle_create_dataset"),
    "list_datasets": (eh, "handle_list_datasets"),
    "run_experiment": (eh, "handle_run_experiment"),
    "get_experiment_run": (eh, "handle_get_experiment_run"),
    "compare_experiment_runs": (eh, "handle_compare_experiment_runs"),
    "list_connector_types": (coh, "handle_list_connector_types"),
    "create_connector_feed": (coh, "handle_create_connector_feed"),
    "list_connector_feeds": (coh, "handle_list_connector_feeds"),
    "get_connector_feed": (coh, "handle_get_connector_feed"),
    "sync_connector_feed": (coh, "handle_sync_connector_feed"),
    "get_context_block": (th, "handle_get_context_block"),
    "search_timeline": (th, "handle_search_timeline"),
    "list_fact_history": (th, "handle_list_fact_history"),
    "invalidate_fact": (th, "handle_invalidate_fact"),
    "check_context_budget": (ch, "handle_check_context_budget"),
    "prune_by_relevance": (ch, "handle_prune_by_relevance"),
    "get_multi_level_skeleton": (ch, "handle_multi_level_skeleton"),
    "evict_stale": (ch, "handle_evict_stale"),
    "advise_context": (ch, "handle_advise_context"),
    "get_compression_insights": (ch, "handle_get_compression_insights"),
    "generate_rewrite_prompt": (ch, "handle_generate_rewrite_prompt"),
    "estimate_tokens": (toh, "handle_estimate_tokens"),
    "configure_for_client": (toh, "handle_configure_for_client"),
    "set_compression_profile": (toh, "handle_set_compression_profile"),
    "get_compression_profile": (toh, "handle_get_compression_profile"),
    "compress_meta_tokens": (toh, "handle_compress_meta_tokens"),
    "recommend_compression": (toh, "handle_recommend_compression"),
    "recover_session": (toh, "handle_recover_session"),
    "compress_codebase": (ch, "handle_compress_codebase"),
    "search_code": (ch, "handle_search_code"),
    "filter_cli_output": (toh, "handle_filter_cli_output"),
    "get_savings_report": (toh, "handle_get_savings_report"),
    "get_savings_inline": (toh, "handle_get_savings_inline"),
    "advise_cache_strategy": (toh, "handle_advise_cache_strategy"),
    "generate_structural_summary": (toh, "handle_generate_structural_summary"),
    "detect_dead_code": (toh, "handle_detect_dead_code"),
    "get_original_output": (toh, "handle_get_original_output"),
    "list_tee_entries": (toh, "handle_list_tee_entries"),
    "tee_store_stats": (toh, "handle_tee_store_stats"),
    "discover_savings": (toh, "handle_discover_savings"),
    "calculate_roi": (toh, "handle_calculate_roi"),
    "check_budget": (toh, "handle_check_budget"),
    "export_team_data": (toh, "handle_export_team_data"),
}

_SCHEMA_GROUPS = (
    DOCS_TOOLS,
    COMPRESSION_TOOLS,
    DETECTION_TOOLS,
    AFM_TOOLS,
    FILESYNC_TOOLS,
    RESOURCE_TOOLS,
    HELP_TOOLS,
    ACE_TOOLS,
    VISUALIZATION_TOOLS,
    EXPERIMENTAL_TOOLS,
    MULTIMODAL_TOOLS,
    BUNDLE_TOOLS,
    MODEL_TOOLS,
    PROMPT_TOOLS,
    MEMORY_TOOLS,
    EXPERIMENT_TOOLS,
    CONNECTOR_TOOLS,
    TEMPORAL_TOOLS,
    TOKEN_OPTIMIZATION_TOOLS,
)
_EXPERIMENTAL_NAMES = {tool.name for tool in EXPERIMENTAL_TOOLS}
# These are intentionally small, conservative policy tables.  Names not in a
# table are explicitly treated as validation-only/unknown rather than inferred
# from a profile or handler module name.
_DESTRUCTIVE_NAMES = frozenset(
    {
        "delete_document",
        "delete_memory",
        "invalidate_fact",
    }
)
_MUTATING_NAMES = frozenset(
    {
        "ingest_context",
        "delete_document",
        "afm_add_message",
        "afm_clear_history",
        "afm_import_history",
        "refresh_document",
        "ace_curate",
        "ace_grow_context",
        "ace_refine_context",
        "ace_execute_cycle",
        "create_handoff_bundle",
        "replay_handoff_bundle",
        "diff_reingest",
        "create_prompt_template",
        "update_prompt_template",
        "deploy_prompt_version",
        "add_memory",
        "delete_memory",
        "ingest_transcript",
        "compile_knowledge",
        "lint_knowledge",
        "create_dataset",
        "run_experiment",
        "create_connector_feed",
        "sync_connector_feed",
        "invalidate_fact",
        "set_compression_profile",
        "configure_for_client",
        "recover_session",
        "get_original_output",
    }
)
_READ_ONLY_NAMES = frozenset(
    {
        "gc_read_doc",
        "gc_search_docs",
        "read_skeleton",
        "search_semantic",
        "get_stats",
        "list_documents",
        "adapt_to_context_window",
        "recommend_fidelity",
        "check_blind_spots",
        "detect_hallucination",
        "afm_build_context",
        "afm_get_stats",
        "afm_export_history",
        "check_file_sync",
        "diff_cached_file",
        "get_version_history",
        "check_resource_health",
        "check_environment",
        "should_compress",
        "tool_help",
        "get_provider_profile",
        "estimate_model_cost",
        "assess_cache_compatibility",
        "diagnose_cache_miss",
        "list_prompt_templates",
        "get_prompt_template",
        "search_memory",
        "list_memories",
        "summarize_user_memory",
        "get_user_profile",
        "get_knowledge_index",
        "search_memory_index",
        "list_datasets",
        "get_experiment_run",
        "compare_experiment_runs",
        "list_connector_types",
        "list_connector_feeds",
        "get_connector_feed",
        "get_context_block",
        "search_timeline",
        "list_fact_history",
        "get_compression_profile",
        "get_savings_report",
        "get_savings_inline",
        "tee_store_stats",
        "discover_savings",
        "check_budget",
    }
)
_IDEMPOTENT_NAMES = _READ_ONLY_NAMES | {"delete_document", "delete_memory"}
_OPEN_WORLD_NAMES = frozenset({"gc_read_doc", "gc_search_docs", "list_connector_types"})


def _build_registry() -> tuple[RegisteredTool, ...]:
    schemas = [tool for group in _SCHEMA_GROUPS for tool in group]
    schema_by_name = {tool.name: tool for tool in schemas}
    if len(schema_by_name) != len(schemas):
        raise RuntimeError("MCP schema registry contains duplicate tool names")
    schema_names = set(schema_by_name)
    handler_names = set(_HANDLER_SPECS)
    if schema_names != handler_names:
        missing = sorted(schema_names - handler_names)
        orphaned = sorted(handler_names - schema_names)
        raise RuntimeError(f"MCP registry drift: missing={missing}, orphaned={orphaned}")

    unresolved = sorted(
        f"{name}:{module.__name__}.{attribute}"
        for name, (module, attribute) in _HANDLER_SPECS.items()
        if not hasattr(module, attribute)
    )
    if unresolved:
        raise RuntimeError(f"MCP registry contains unresolved handlers: {unresolved}")

    entries = []
    for name, schema in schema_by_name.items():
        module, attribute = _HANDLER_SPECS[name]
        entries.append(
            RegisteredTool(
                schema=schema,
                handler=_live_handler(module, attribute),
                profiles=(
                    frozenset({"full", "core_stable"})
                    if name in CORE_STABLE_TOOL_NAMES
                    else frozenset({"full"})
                ),
                fixture_id=f"mcp-tool:{name}",
                test_policy="registry-characterization",
                read_only=name in _READ_ONLY_NAMES,
                mutates_state=name in _MUTATING_NAMES,
                destructive=name in _DESTRUCTIVE_NAMES,
                idempotent=name in _IDEMPOTENT_NAMES,
                open_world=name in _OPEN_WORLD_NAMES,
                experimental=name in _EXPERIMENTAL_NAMES,
            )
        )
    return tuple(sorted(entries, key=lambda entry: entry.name))


REGISTERED_TOOLS: tuple[RegisteredTool, ...] = _build_registry()
_BY_NAME = {entry.name: entry for entry in REGISTERED_TOOLS}


def registered_tools(profile: str = "full") -> tuple[RegisteredTool, ...]:
    """Return registry entries enabled by ``profile`` in stable name order."""
    normalized = _normalize_tool_profile(profile)
    return tuple(entry for entry in REGISTERED_TOOLS if normalized in entry.profiles)


def registered_tool_names(profile: str = "full") -> frozenset[str]:
    return frozenset(entry.name for entry in registered_tools(profile))


def get_registered_tool(name: str) -> RegisteredTool:
    return _BY_NAME[name]


__all__ = [
    "REGISTERED_TOOLS",
    "RegisteredTool",
    "get_registered_tool",
    "registered_tool_names",
    "registered_tools",
]
