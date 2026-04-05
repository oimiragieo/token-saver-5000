# Folder guide: `src/semantic_modulator/app/`

Breadcrumb for AI navigation. Master index: [`CLAUDE.md`](../../../CLAUDE.md).

## Contents

### Python modules

#### `__init__.py`

Application layer package.

_No top-level classes or functions (may re-export only)._

#### `ace_context_manager.py`

ACE context storage with LRU eviction semantics.

| Kind | Name |
|------|------|
| `class` | `ACEContextManager` |

#### `bootstrap.py`

Bootstrap entrypoints for application wiring.

| Kind | Name |
|------|------|
| `class` | `_ServerProtocol` |
| `def` | `create_server` |
| `async def` | `async_main` |
| `def` | `main` |

#### `context_service.py`

App-layer service for handler context assembly and input validation.

| Kind | Name |
|------|------|
| `class` | `ServerContextService` |

#### `contract_validation.py`

Shared contract-validation utilities for the app layer.

| Kind | Name |
|------|------|
| `def` | `contract_key_mismatch_message` |
| `def` | `validate_contract_keys` |

#### `lifecycle_service.py`

App-layer lifecycle service for startup and shutdown orchestration.

| Kind | Name |
|------|------|
| `class` | `StartupRequest` |
| `class` | `ShutdownRequest` |
| `class` | `ServerLifecycleService` |

#### `mcp_context_surfaces.py`

Prompt and resource surfaces for the Token Saver MCP server.

| Kind | Name |
|------|------|
| `def` | `_prompt_catalog` |
| `def` | `list_prompts` |
| `def` | `_required_argument` |
| `def` | `_optional_argument` |
| `def` | `get_prompt` |
| `def` | `list_resources` |
| `def` | `list_resource_templates` |
| `async def` | `read_resource` |

#### `persistence_orchestration_service.py`

App-layer persistence/sync orchestration service for server lifecycle.

| Kind | Name |
|------|------|
| `class` | `LoadPersistedRequest` |
| `class` | `LoadSyncRequest` |
| `class` | `SaveSyncRequest` |
| `class` | `PersistenceOrchestrationService` |

#### `progress_service.py`

App-layer progress bar rendering service.

| Kind | Name |
|------|------|
| `class` | `ProgressRequest` |
| `class` | `ProgressRenderService` |

#### `router_binding.py`

MCP router binding helper for server protocol registration.

| Kind | Name |
|------|------|
| `def` | `_get_formatter` |
| `def` | `_format_result` |
| `class` | `BindRequest` |
| `def` | `validate_bind_request_map` |
| `def` | `bind_mcp_handlers` |

#### `runtime_service.py`

App-layer runtime execution service for MCP stdio serving.

| Kind | Name |
|------|------|
| `class` | `RunRequest` |
| `class` | `RuntimeService` |

#### `server_aliases.py`

Helpers for server class alias mapping and wiring overrides.

| Kind | Name |
|------|------|
| `def` | `validate_override_keys` |
| `def` | `build_server_class_overrides` |

#### `server_factory_service.py`

App-layer factory service for SemanticModulatorServer composition wiring.

| Kind | Name |
|------|------|
| `class` | `FactoryClassMap` |
| `class` | `BuildKwargsMap` |
| `class` | `CoreRuntimeArtifacts` |
| `class` | `ServiceLayerArtifacts` |
| `class` | `BuildArtifacts` |
| `class` | `FactoryValidationResult` |
| `class` | `DefaultBuildInputs` |
| `class` | `BuildDefaultRequest` |
| `class` | `BuildRequest` |
| `class` | `ServerFactoryService` |

#### `server_service_adapter.py`

Adapter that centralizes helper delegation for SemanticModulatorServer.

| Kind | Name |
|------|------|
| `class` | `ServerServiceAdapter` |

#### `tool_profile_service.py`

App-layer tool profile bootstrap and diagnostics service.

| Kind | Name |
|------|------|
| `class` | `BootstrapRequest` |
| `class` | `ToolProfileBootstrapService` |

#### `tooling.py`

Application-layer MCP tooling gateway.

| Kind | Name |
|------|------|
| `class` | `ProfileState` |
| `class` | `MCPToolingGateway` |

---

Symbols are **top-level only** (nested methods and inner functions are not listed). Regenerate: `python scripts/generate_claude_folder_guides.py`.
