"""Tool schemas: Prompt templates (ph) + ACE framework (ace). Split from mcp_core.py (N2 slice 2).

Each list is verbatim `Tool(...)` schema literals moved from the original
monolithic `setup_mcp_tools`, unchanged.
"""

from mcp.types import Tool

ACE_TOOLS: list = [
    Tool(
        name="ace_generate",
        description=(
            "[AFM] ACE GENERATE: Generate reasoning trajectory for a task using ACE playbook. "
            "Produces step-by-step reasoning that applies relevant bullets from the playbook. "
            "Each step includes relevant guidelines, reasoning, and confidence scores. "
            "Use this to guide semantic node selection and compression decisions."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task or query to reason about",
                },
                "context_id": {
                    "type": "string",
                    "description": "ACE context identifier (default: 'default')",
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Maximum trajectory steps (default: 5)",
                },
                "top_k_bullets": {
                    "type": "integer",
                    "description": "Bullets to consider per step (default: 5)",
                },
            },
            "required": ["task"],
        },
    ),
    Tool(
        name="ace_reflect",
        description=(
            "[ANALYZE] ACE REFLECT: Extract insights from a reasoning trajectory. "
            "Analyzes what worked (successes) and what didn't (failures) to formulate new bullets. "
            "Returns insights with confidence scores and reasoning. "
            "Use after completing a task to learn and improve the playbook."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "trajectory": {
                    "type": "array",
                    "description": "Generated reasoning trajectory from ace_generate",
                    "items": {"type": "object"},
                },
                "outcome": {
                    "type": "string",
                    "description": "What actually happened (result description)",
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether the trajectory led to success",
                },
                "context_id": {
                    "type": "string",
                    "description": "ACE context identifier (default: 'default')",
                },
            },
            "required": ["trajectory", "outcome", "success"],
        },
    ),
    Tool(
        name="ace_curate",
        description=(
            "[CURATE] ACE CURATE: Integrate insights into playbook via delta updates. "
            "Applies incremental changes (add/update/remove bullets) with semantic deduplication. "
            "Prevents context collapse through grow-and-refine strategy. "
            "Use after reflecting to evolve the playbook."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "insights": {
                    "type": "array",
                    "description": "Insights from ace_reflect",
                    "items": {"type": "object"},
                },
                "context_id": {
                    "type": "string",
                    "description": "ACE context identifier (default: 'default')",
                },
                "max_bullets": {
                    "type": "integer",
                    "description": "Maximum bullets (triggers pruning if exceeded)",
                },
            },
            "required": ["insights"],
        },
    ),
    Tool(
        name="ace_grow_context",
        description=(
            "[ADD] ACE GROW: Manually add bullets to playbook (grow operation). "
            "Directly insert principles, strategies, tactics, constraints, or preferences. "
            "Use to seed domain-specific knowledge or codify team standards. "
            "Each bullet gets an embedding for semantic operations."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "bullets": {
                    "type": "array",
                    "description": "Bullets to add",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "bullet_type": {
                                "type": "string",
                                "enum": [
                                    "principle",
                                    "strategy",
                                    "tactic",
                                    "constraint",
                                    "preference",
                                    "learned",
                                ],
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                        },
                        "required": ["text", "bullet_type"],
                    },
                },
                "context_id": {
                    "type": "string",
                    "description": "ACE context identifier (default: 'default')",
                },
            },
            "required": ["bullets"],
        },
    ),
    Tool(
        name="ace_refine_context",
        description=(
            "[ACE] ACE REFINE: Update bullet performance based on feedback (refine operation). "
            "Adjusts confidence scores for specific bullets based on success/failure. "
            "Use to reinforce successful patterns or penalize failed approaches. "
            "Enables continuous improvement of the playbook."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "bullet_ids": {
                    "type": "array",
                    "description": "Bullet IDs to update",
                    "items": {"type": "string"},
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether these bullets led to success",
                },
                "confidence_boost": {
                    "type": "number",
                    "description": "Adjustment amount (default: 0.05)",
                },
                "context_id": {
                    "type": "string",
                    "description": "ACE context identifier (default: 'default')",
                },
            },
            "required": ["bullet_ids", "success"],
        },
    ),
    Tool(
        name="ace_get_playbook",
        description=(
            "[ACE] ACE GET PLAYBOOK: Retrieve current ACE playbook state. "
            "Returns all bullets with performance stats, versioning, and delta history. "
            "Supports filtering by confidence, bullet type, or custom criteria. "
            "Use to inspect the evolved playbook and understand learned patterns."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "context_id": {
                    "type": "string",
                    "description": "ACE context identifier (default: 'default')",
                },
                "include_embeddings": {
                    "type": "boolean",
                    "description": "Include bullet embeddings (default: false)",
                },
                "min_confidence": {
                    "type": "number",
                    "description": "Filter bullets below this confidence",
                },
                "bullet_type": {
                    "type": "string",
                    "description": "Filter by bullet type",
                    "enum": [
                        "principle",
                        "strategy",
                        "tactic",
                        "constraint",
                        "preference",
                        "learned",
                    ],
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="ace_execute_cycle",
        description=(
            "[SYNC] ACE EXECUTE CYCLE: Execute complete ACE cycle (Generate -> Reflect -> Curate). "
            "Convenience tool that combines the three-step ACE process into one call. "
            "Generates trajectory, reflects on outcome, and curates insights automatically. "
            "Use for rapid iteration and continuous playbook improvement."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task or query",
                },
                "outcome": {
                    "type": "string",
                    "description": "What actually happened",
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether the task succeeded",
                },
                "context_id": {
                    "type": "string",
                    "description": "ACE context identifier (default: 'default')",
                },
                "max_trajectory_steps": {
                    "type": "integer",
                    "description": "Maximum trajectory steps (default: 5)",
                },
            },
            "required": ["task", "outcome", "success"],
        },
    ),
]


PROMPT_TOOLS: list = [
    Tool(
        name="create_prompt_template",
        description=(
            "Create a managed prompt template with version 1 and optional deployment "
            "label. Use this to make prompts first-class artifacts instead of hard-coded strings."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Unique prompt template name"},
                "description": {
                    "type": "string",
                    "description": "Human-readable prompt description",
                },
                "system_prompt": {
                    "type": "string",
                    "description": "Static system prompt section",
                },
                "user_prompt_template": {
                    "type": "string",
                    "description": "User prompt template with optional {variables}",
                },
                "variables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Template variable names used by the prompt",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional structured prompt metadata",
                },
                "deployment_label": {
                    "type": "string",
                    "description": "Optional initial deployment label (for example: production, staging)",
                },
            },
            "required": ["name", "description", "system_prompt", "user_prompt_template"],
        },
    ),
    Tool(
        name="update_prompt_template",
        description=(
            "Create a new version of an existing prompt template. Supports prompt edits, "
            "variable changes, metadata updates, and change notes."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Existing prompt template name"},
                "description": {
                    "type": "string",
                    "description": "Optional updated template description",
                },
                "system_prompt": {
                    "type": "string",
                    "description": "Optional replacement system prompt",
                },
                "user_prompt_template": {
                    "type": "string",
                    "description": "Optional replacement user prompt template",
                },
                "variables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional replacement variable list",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata patch merged into the latest version metadata",
                },
                "change_note": {
                    "type": "string",
                    "description": "Optional summary of why this version changed",
                },
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="list_prompt_templates",
        description=(
            "List managed prompt templates with their latest version and deployment labels. "
            "Optionally include all versions."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "include_versions": {
                    "type": "boolean",
                    "description": "Include all versions for each template (default: false)",
                    "default": False,
                }
            },
        },
    ),
    Tool(
        name="get_prompt_template",
        description=(
            "Get a prompt template and resolve a specific version or deployment label "
            "to the exact prompt content."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Prompt template name"},
                "version": {
                    "type": "integer",
                    "description": "Optional version number to resolve",
                },
                "deployment_label": {
                    "type": "string",
                    "description": "Optional deployment label to resolve (mutually exclusive with version)",
                },
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="deploy_prompt_version",
        description=(
            "Assign or move a deployment label (production, staging, canary) to a specific "
            "prompt template version."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Prompt template name"},
                "version": {"type": "integer", "description": "Version to deploy"},
                "deployment_label": {
                    "type": "string",
                    "description": "Deployment label to assign",
                },
                "allow_stable_prefix_change": {
                    "type": "boolean",
                    "description": "Acknowledge and allow deployment if the stable cacheable prefix will change",
                    "default": False,
                },
            },
            "required": ["name", "version", "deployment_label"],
        },
    ),
    Tool(
        name="compare_prompt_versions",
        description=(
            "Compare two versions of the same prompt template and return changed fields "
            "plus a unified diff."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Prompt template name"},
                "version_a": {"type": "integer", "description": "Base version"},
                "version_b": {"type": "integer", "description": "Comparison version"},
            },
            "required": ["name", "version_a", "version_b"],
        },
    ),
    Tool(
        name="render_prompt_template",
        description=(
            "Resolve and render a prompt template into cache-friendly ordered sections "
            "for a provider call."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Prompt template name"},
                "variables": {
                    "type": "object",
                    "description": "Template variables used to render the user prompt",
                },
                "version": {
                    "type": "integer",
                    "description": "Optional version number to resolve",
                },
                "deployment_label": {
                    "type": "string",
                    "description": "Optional deployment label to resolve",
                },
                "tool_definitions": {
                    "type": "string",
                    "description": "Optional serialized tool definitions to pin in the stable prefix",
                },
                "rag_context": {
                    "type": "string",
                    "description": "Optional static retrieved context to place before volatile sections",
                },
                "few_shot_examples": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional few-shot examples",
                },
                "chat_history": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional prior conversation turns",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional dynamic metadata to place in the volatile tail",
                },
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="list_prefix_collisions",
        description=(
            "List rendered prompt prefixes that collide across templates so shared "
            "provider-cacheable prefixes are visible."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="audit_prompt_cacheability",
        description=(
            "Audit a composed prompt for cache-friendly section ordering and volatile "
            "metadata that can break provider prefix caching."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "sections": {
                    "type": "array",
                    "description": "Ordered prompt sections using canonical names",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "enum": [
                                    "tool_definitions",
                                    "system_instructions",
                                    "rag_context",
                                    "few_shot_examples",
                                    "chat_history",
                                    "metadata",
                                    "user_query",
                                ],
                            },
                            "content": {"type": "string"},
                        },
                        "required": ["name", "content"],
                    },
                }
            },
            "required": ["sections"],
        },
    ),
]
