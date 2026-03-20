# Workflow Orchestration Guide

This guide turns the scattered tool-level hints into concrete workflows for real deployments.

## Core Compression Workflow

Use this when you want the normal "compress first, expand only where needed" path:

1. `should_compress` to decide if the payload is large enough to benefit.
2. `ingest_context` to build the semantic graph.
3. `read_skeleton` to get the compact overview.
4. `search_semantic` to find task-relevant nodes.
5. `modulate_region` to expand only the nodes you need.
6. `advise_context` to shape the final prompt for the target model and budget.

## Query-Critical Workflow

Use this when correctness matters more than raw speed:

1. `ingest_context`
2. `read_skeleton` with query-guided or evidence-aware settings
3. `search_semantic`
4. `recommend_fidelity`
5. `modulate_region`
6. `create_handoff_bundle` when another agent or teammate will continue the task

## Prompt and Memory Workflow

Use this when the system is acting like a context platform instead of a one-shot compressor:

1. `add_memory` to store durable user or team preferences.
2. `search_memory` or `get_user_profile` before prompt assembly.
3. `create_prompt_template` or `update_prompt_template` to keep static prompt sections versioned.
4. `deploy_prompt_version` to promote known-good prompts.
5. `optimize_for_model` before final inference when provider-specific costs or cache behavior matter.

## Connector and Refresh Workflow

Use this when upstream systems are feeding Token Saver 5000:

1. `create_connector_feed`
2. `sync_connector_feed`
3. `check_file_sync` before expensive downstream reasoning
4. `diff_cached_file` or `diff_reingest` when source changes are detected
5. `get_version_history` when operators need change visibility

## Multi-Tenant Workflow Guardrails

When requests come from a shared web or SaaS surface:

1. Attach `workspace_id` first.
2. Add `user_id`, `agent_id`, and `session_id` when the workflow needs narrower isolation.
3. Keep those scope fields stable through memory, prompts, connector feeds, temporal queries, and bundle replay.

Read `docs/deployment/SAAS_MULTI_TENANT.md` for the full isolation model.
