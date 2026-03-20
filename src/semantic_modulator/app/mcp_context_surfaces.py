"""Prompt and resource surfaces for the Token Saver MCP server."""

from __future__ import annotations

import json
from urllib.parse import urlparse

from mcp.types import (
    GetPromptResult,
    Prompt,
    PromptArgument,
    PromptMessage,
    Resource,
    ResourceTemplate,
    TextContent,
    TextResourceContents,
)

from ...handlers.help_handlers import handle_tool_help
from ...mcp_install import SERVER_KEY, WORKSPACE_FOLDER_VAR, inspect_mcp_installation


def _prompt_catalog() -> list[dict[str, object]]:
    return [
        {
            "name": "document_compression_workflow",
            "description": "Guide the model through Token Saver's preferred document compression workflow.",
            "arguments": [
                PromptArgument(
                    name="goal",
                    description="What the user wants to accomplish with the document.",
                    required=True,
                ),
                PromptArgument(
                    name="file_id",
                    description="Optional existing document identifier.",
                    required=False,
                ),
                PromptArgument(
                    name="query",
                    description="Optional retrieval question to bias skeleton/search steps.",
                    required=False,
                ),
            ],
        },
        {
            "name": "prompt_cache_review",
            "description": "Review prompt-cache stability, telemetry visibility, and likely cache-miss causes.",
            "arguments": [
                PromptArgument(
                    name="user_prompt",
                    description="Prompt or prompt section set to review.",
                    required=True,
                ),
                PromptArgument(
                    name="model",
                    description="Optional model/provider family, such as claude or gpt-4.1.",
                    required=False,
                ),
                PromptArgument(
                    name="harness",
                    description="Optional harness/client name, such as Claude Code or Codex.",
                    required=False,
                ),
            ],
        },
        {
            "name": "mcp_setup_assistant",
            "description": "Choose the right Token Saver MCP install mode and verify it with doctor output.",
            "arguments": [
                PromptArgument(
                    name="target",
                    description="Setup target: desktop, project, or portable_project.",
                    required=False,
                ),
                PromptArgument(
                    name="environment",
                    description="Optional environment note such as Windows, macOS, team repo, or local dev.",
                    required=False,
                ),
            ],
        },
    ]


def list_prompts() -> list[Prompt]:
    return [
        Prompt(
            name=item["name"],
            description=item["description"],
            arguments=item["arguments"],
        )
        for item in _prompt_catalog()
    ]


def _required_argument(arguments: dict[str, object], name: str, prompt_name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{prompt_name} requires a non-empty '{name}' argument.")
    return value.strip()


def _optional_argument(arguments: dict[str, object], name: str) -> str | None:
    value = arguments.get(name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def get_prompt(name: str, arguments: dict[str, object] | None) -> GetPromptResult:
    args = arguments or {}
    if name == "document_compression_workflow":
        goal = _required_argument(args, "goal", name)
        file_id = _optional_argument(args, "file_id")
        query = _optional_argument(args, "query")
        prompt_text = "\n".join(
            [
                "Use Token Saver 5000 with its preferred compression workflow.",
                "Sequence: ingest_context -> read_skeleton -> search_semantic -> modulate_region -> check_blind_spots.",
                f"Goal: {goal}",
                f"Preferred file_id: {file_id or 'choose a stable file_id if ingestion is needed'}",
                f"Task query: {query or 'none provided; infer retrieval needs from the task'}",
                "Read the skeleton before expanding details, and cite evidence from modulated/search results.",
            ]
        )
        return GetPromptResult(
            description="Token Saver document compression workflow",
            messages=[
                PromptMessage(role="user", content=TextContent(type="text", text=prompt_text))
            ],
        )

    if name == "prompt_cache_review":
        user_prompt = _required_argument(args, "user_prompt", name)
        model = _optional_argument(args, "model") or "unspecified"
        harness = _optional_argument(args, "harness") or "unspecified"
        prompt_text = "\n".join(
            [
                "Audit this prompt for prompt-cache stability and telemetry observability.",
                "Preferred sequence: audit_prompt_cacheability -> render_prompt_template (when relevant) -> assess_cache_compatibility -> capture_cache_telemetry -> diagnose_cache_miss.",
                f"Model: {model}",
                f"Harness: {harness}",
                "Prompt to review:",
                user_prompt,
                "Focus on stable-prefix ordering, hidden volatility, telemetry blind spots, and concrete fixes.",
            ]
        )
        return GetPromptResult(
            description="Token Saver prompt cache review workflow",
            messages=[
                PromptMessage(role="user", content=TextContent(type="text", text=prompt_text))
            ],
        )

    if name == "mcp_setup_assistant":
        target = _optional_argument(args, "target") or "desktop"
        environment = _optional_argument(args, "environment") or "unspecified"
        prompt_text = "\n".join(
            [
                "Help the user set up Token Saver MCP using the supported installer commands.",
                "Start with token-saver-setup for guided onboarding, then fall back to token-saver-install-mcp for advanced modes.",
                "Choose between token-saver-setup, token-saver-install-mcp, --project-config, --portable-project-config, --uninstall, and --print-config.",
                "Always recommend running token-saver-install-mcp --doctor after setup.",
                f"Target: {target}",
                f"Environment: {environment}",
                "Explain why the chosen mode fits the user's environment and keep the setup minimal.",
            ]
        )
        return GetPromptResult(
            description="Token Saver MCP setup assistant",
            messages=[
                PromptMessage(role="user", content=TextContent(type="text", text=prompt_text))
            ],
        )

    raise ValueError(f"Unknown prompt '{name}'.")


def list_resources(tooling: object, profile: str) -> list[Resource]:
    tools = tooling.list_tools(profile=profile)
    return [
        Resource(
            name="tool_catalog",
            uri="token-saver://catalog/tools",
            description="Available Token Saver MCP tools for the active profile.",
            mimeType="application/json",
        ),
        Resource(
            name="prompt_catalog",
            uri="token-saver://catalog/prompts",
            description="Available Token Saver MCP prompts and their arguments.",
            mimeType="application/json",
        ),
        Resource(
            name="server_instructions",
            uri="token-saver://instructions/server",
            description="Preferred tool sequences and usage guidance for Token Saver MCP.",
            mimeType="text/markdown",
        ),
        Resource(
            name="document_workflow",
            uri="token-saver://workflow/document-compression",
            description="Recommended document compression workflow.",
            mimeType="text/markdown",
        ),
        Resource(
            name="prompt_cache_workflow",
            uri="token-saver://workflow/prompt-caching",
            description="Recommended prompt-cache audit workflow.",
            mimeType="text/markdown",
        ),
        Resource(
            name="install_modes",
            uri="token-saver://config/install-modes",
            description="Supported MCP installation and configuration modes.",
            mimeType="text/markdown",
        ),
        Resource(
            name="installation_status",
            uri="token-saver://status/mcp-installation",
            description="Current install health for desktop and project MCP configs.",
            mimeType="application/json",
        ),
        Resource(
            name="tool_count",
            uri="token-saver://status/tool-count",
            description="Active tool count for the current Token Saver MCP profile.",
            mimeType="application/json",
            size=len(tools),
        ),
    ]


def list_resource_templates() -> list[ResourceTemplate]:
    return [
        ResourceTemplate(
            name="tool_help",
            uriTemplate="token-saver://tool/{name}/help",
            description="Detailed help payload for a specific Token Saver tool.",
            mimeType="application/json",
        )
    ]


async def read_resource(
    uri: str, *, tooling: object, profile: str, context: object
) -> list[TextResourceContents]:
    parsed = urlparse(uri)
    tools = tooling.list_tools(profile=profile)
    tool_names = [tool.name for tool in tools]

    if uri == "token-saver://catalog/tools":
        text = json.dumps(
            {
                "profile": profile,
                "total_tools": len(tools),
                "tools": [{"name": tool.name, "description": tool.description} for tool in tools],
            },
            indent=2,
        )
        return [TextResourceContents(uri=uri, mimeType="application/json", text=text)]

    if uri == "token-saver://catalog/prompts":
        text = json.dumps(
            {
                "prompts": [
                    {
                        "name": item["name"],
                        "description": item["description"],
                        "arguments": [
                            {
                                "name": argument.name,
                                "description": argument.description,
                                "required": argument.required,
                            }
                            for argument in item["arguments"]
                        ],
                    }
                    for item in _prompt_catalog()
                ]
            },
            indent=2,
        )
        return [TextResourceContents(uri=uri, mimeType="application/json", text=text)]

    if uri == "token-saver://instructions/server":
        text = "\n".join(
            [
                "# Token Saver server instructions",
                "",
                "- Prefer `read_skeleton` before `modulate_region` when a document is already ingested.",
                "- Use `search_semantic` to narrow evidence before expanding nodes.",
                "- Use `check_blind_spots` before final answers that summarize compressed content.",
                "- For prompt-caching work, start with `audit_prompt_cacheability` and only then inspect telemetry/diagnostics.",
                f"- Active profile: `{profile}` with {len(tool_names)} tools.",
            ]
        )
        return [TextResourceContents(uri=uri, mimeType="text/markdown", text=text)]

    if uri == "token-saver://workflow/document-compression":
        text = "\n".join(
            [
                "# Document compression workflow",
                "",
                "1. `ingest_context` with a stable `file_id`.",
                "2. `read_skeleton` to inspect anchors.",
                "3. `search_semantic` for targeted evidence.",
                "4. `modulate_region` for precise expansion.",
                "5. `check_blind_spots` before the final answer.",
            ]
        )
        return [TextResourceContents(uri=uri, mimeType="text/markdown", text=text)]

    if uri == "token-saver://workflow/prompt-caching":
        text = "\n".join(
            [
                "# Prompt caching workflow",
                "",
                "1. `audit_prompt_cacheability` to check stable-prefix order.",
                "2. `render_prompt_template` when prompts come from the registry.",
                "3. `assess_cache_compatibility` for model and harness visibility.",
                "4. `capture_cache_telemetry` after real provider calls.",
                "5. `diagnose_cache_miss` when expected cached tokens are missing.",
            ]
        )
        return [TextResourceContents(uri=uri, mimeType="text/markdown", text=text)]

    if uri == "token-saver://config/install-modes":
        text = "\n".join(
            [
                "# Token Saver MCP install modes",
                "",
                "- `token-saver-setup`: guided setup command that recommends and applies the best install target.",
                "- `token-saver-setup --auto`: apply the recommended target for the current workspace.",
                "- `token-saver-setup --uninstall --desktop|--project|--portable-project`: remove a specific config target.",
                "- `token-saver-setup --uninstall-all`: remove Token Saver from both desktop and current project configs.",
                "- `token-saver-install-mcp`: install Claude Desktop config.",
                "- `token-saver-install-mcp --project-config`: write project-scoped `.claude/.mcp.json` for the current working directory unless `--project-root` is provided.",
                f"- `token-saver-install-mcp --portable-project-config`: write project-scoped `.claude/.mcp.json` using `{WORKSPACE_FOLDER_VAR}` for the current working directory unless `--project-root` is provided.",
                "- `token-saver-install-mcp --uninstall`: remove Token Saver from Claude Desktop config.",
                "- `token-saver-install-mcp --project-config --uninstall`: remove Token Saver from project config.",
                "- `token-saver-install-mcp --print-config`: print raw JSON instead of writing files.",
                "- `token-saver-install-mcp --doctor`: inspect command and config health.",
                "- `token-saver-install-mcp --doctor --human`: print a concise setup summary with a recommended next command.",
            ]
        )
        return [TextResourceContents(uri=uri, mimeType="text/markdown", text=text)]

    if uri == "token-saver://status/mcp-installation":
        text = json.dumps(inspect_mcp_installation(), indent=2)
        return [TextResourceContents(uri=uri, mimeType="application/json", text=text)]

    if uri == "token-saver://status/tool-count":
        text = json.dumps(
            {"profile": profile, "total_tools": len(tools), "server_key": SERVER_KEY},
            indent=2,
        )
        return [TextResourceContents(uri=uri, mimeType="application/json", text=text)]

    if parsed.scheme == "token-saver" and parsed.netloc == "tool":
        path_parts = [part for part in parsed.path.split("/") if part]
        if len(path_parts) == 2 and path_parts[1] == "help":
            tool_name = path_parts[0]
            help_text = await handle_tool_help(context, {"tool_name": tool_name, "verbose": True})
            return [TextResourceContents(uri=uri, mimeType="application/json", text=help_text)]

    raise ValueError(f"Unknown resource '{uri}'.")
