"""One-off splits for backlog P1 (help registry, schemas_compression, compression_handlers)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def split_help_handlers() -> None:
    hh_path = ROOT / "src/handlers/help_handlers.py"
    lines = hh_path.read_text(encoding="utf-8").splitlines(keepends=True)
    header = lines[:45]
    registry_block = lines[46:1483]
    tail = lines[1483:]
    imports = (
        '"""Static tool help registry (examples, tips, output fields)."""\n\n'
        "from typing import Any, Dict\n\n"
        "from .compression_handlers import (\n"
        "    get_ingest_context_output_fields,\n"
        "    get_read_skeleton_output_fields,\n"
        "    get_recommend_fidelity_output_fields,\n"
        "    get_search_semantic_output_fields,\n"
        ")\n"
        "from .bundle_handlers import get_bundle_output_fields\n"
        "from .connector_handlers import get_connector_output_fields\n"
        "from .experiment_handlers import get_dataset_output_fields, get_experiment_output_fields\n"
        "from .memory_handlers import get_memory_output_fields, get_user_profile_output_fields\n"
        "from .model_handlers import (\n"
        "    get_cache_compatibility_output_fields,\n"
        "    get_cache_diagnostic_output_fields,\n"
        "    get_cache_telemetry_output_fields,\n"
        "    get_model_output_fields,\n"
        ")\n"
        "from .multimodal_handlers import get_multimodal_output_fields\n"
        "from .prompt_handlers import (\n"
        "    get_prompt_audit_output_fields,\n"
        "    get_prompt_compare_output_fields,\n"
        "    get_prompt_deploy_output_fields,\n"
        "    get_prefix_collision_output_fields,\n"
        "    get_prompt_output_fields,\n"
        "    get_prompt_render_output_fields,\n"
        ")\n"
        "from .resource_handlers import get_check_environment_output_fields\n"
        "from .temporal_handlers import get_temporal_output_fields\n\n"
    )
    (ROOT / "src/handlers/help_tool_registry.py").write_text(
        imports + "".join(registry_block), encoding="utf-8", newline="\n"
    )
    new_hh = "".join(header) + "from .help_tool_registry import TOOL_HELP_REGISTRY\n\n" + "".join(tail)
    hh_path.write_text(new_hh, encoding="utf-8", newline="\n")
    print(f"help_handlers: {len(lines)} -> {len(new_hh.splitlines())} lines")


def split_schemas_compression() -> None:
    path = ROOT / "src/handlers/mcp_core/schemas_compression.py"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    # Split before batch_ingest (line 368, index 367)
    head = lines[:367]
    tail = lines[367:]
    core_path = ROOT / "src/handlers/mcp_core/schemas_compression_core.py"
    batch_path = ROOT / "src/handlers/mcp_core/schemas_compression_batch.py"
    core_head = head[:10]  # docstring + imports through COMPRESSION_CORE_TOOLS
    core_body = head[10:]
    core_path.write_text(
        "".join(core_head).replace("COMPRESSION_TOOLS", "COMPRESSION_CORE_TOOLS", 1)
        + "".join(core_body).rstrip()
        + "\n]\n",
        encoding="utf-8",
        newline="\n",
    )
    batch_path.write_text(
        '"""Compression tool schemas: batch, directory, and extended ops."""\n\n'
        "from mcp.types import Tool, ToolAnnotations\n\n"
        "from ._constants import SCOPE_PROPERTIES\n\n"
        "COMPRESSION_BATCH_TOOLS: list = [\n"
        + "".join(tail[1:]).replace("COMPRESSION_TOOLS", "COMPRESSION_BATCH_TOOLS", 1),
        encoding="utf-8",
        newline="\n",
    )
    shim = (
        '"""Compression MCP tool schemas (core + batch modules)."""\n\n'
        "from .schemas_compression_core import COMPRESSION_CORE_TOOLS\n"
        "from .schemas_compression_batch import COMPRESSION_BATCH_TOOLS\n\n"
        "COMPRESSION_TOOLS: list = COMPRESSION_CORE_TOOLS + COMPRESSION_BATCH_TOOLS\n"
    )
    path.write_text(shim, encoding="utf-8", newline="\n")
    print("schemas_compression split done")


def split_compression_handlers() -> None:
    path = ROOT / "src/handlers/compression_handlers.py"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    common = lines[:625]  # through helpers before handle_ingest
    ingest_block = lines[625:2012]  # handle_ingest through recommend_fidelity
    extended = lines[2012:]  # batch through end

    common_path = ROOT / "src/handlers/compression_handlers_common.py"
    ingest_path = ROOT / "src/handlers/compression_handlers_ingest.py"
    extended_path = ROOT / "src/handlers/compression_handlers_extended.py"

    common_path.write_text("".join(common), encoding="utf-8", newline="\n")
    ingest_path.write_text(
        '"""Compression handlers: ingest, read, search, manage documents."""\n\n'
        + "".join(ingest_block),
        encoding="utf-8",
        newline="\n",
    )
    extended_path.write_text(
        '"""Compression handlers: batch, directory, codebase, and extended ops."""\n\n'
        + "".join(extended),
        encoding="utf-8",
        newline="\n",
    )

    exports = [
        "handle_ingest",
        "handle_read_skeleton",
        "handle_modulate_region",
        "handle_search_semantic",
        "handle_get_stats",
        "handle_list_documents",
        "handle_delete_document",
        "handle_adapt_to_context_window",
        "handle_multilevel_encode",
        "handle_recommend_fidelity",
        "handle_batch_ingest",
        "handle_ingest_directory",
        "handle_diff_reingest",
        "handle_find_duplicates",
        "handle_get_presets",
        "handle_check_context_budget",
        "handle_prune_by_relevance",
        "handle_multi_level_skeleton",
        "handle_evict_stale",
        "handle_advise_context",
        "handle_get_compression_insights",
        "handle_generate_rewrite_prompt",
        "handle_compress_codebase",
        "handle_search_code",
        "get_ingest_context_output_fields",
        "get_read_skeleton_output_fields",
        "get_recommend_fidelity_output_fields",
        "get_search_semantic_output_fields",
    ]
    shim_lines = [
        '"""Compression MCP handlers (re-export facade)."""\n\n',
        "from .compression_handlers_common import *  # noqa: F403\n",
        "from .compression_handlers_ingest import *  # noqa: F403\n",
        "from .compression_handlers_extended import *  # noqa: F403\n",
    ]
    path.write_text("".join(shim_lines), encoding="utf-8", newline="\n")
    print(
        f"compression_handlers split: common={len(common)} ingest={len(ingest_block)} ext={len(extended)}"
    )


if __name__ == "__main__":
    split_help_handlers()
    split_schemas_compression()
    split_compression_handlers()
