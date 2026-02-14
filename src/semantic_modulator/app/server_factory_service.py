"""App-layer factory service for SemanticModulatorServer composition wiring."""

from __future__ import annotations

from typing import Any, TypedDict

from .server_aliases import ALLOWED_FACTORY_OVERRIDE_KEYS, validate_override_keys

ALLOWED_OVERRIDE_KEYS: frozenset[str] = ALLOWED_FACTORY_OVERRIDE_KEYS


class CoreRuntimeArtifacts(TypedDict):
    """Foundational runtime collaborators built outside the service layer."""

    focus_manager: Any
    persistence: Any
    resource_manager: Any
    sync_manager: Any
    version_manager: Any
    path_validator: Any
    ace_framework: Any
    ace_contexts: Any


class ServiceLayerArtifacts(TypedDict):
    """Service-layer collaborators and adapter wiring."""

    context_service: Any
    lifecycle_service: Any
    progress_service: Any
    persistence_service: Any
    tool_profile_service: Any
    runtime_service: Any
    service_adapter: Any


class BuildArtifacts(CoreRuntimeArtifacts, ServiceLayerArtifacts):
    """Complete artifact map returned by factory build methods."""

    compressor: Any
    blind_spot_detector: Any
    halo_detector: Any
    context_window_adapter: Any
    multilevel_encoder: Any
    tooling: Any
    context_window_monitor: dict[str, Any]
    retrieval_history: dict[str, Any]


class ServerFactoryService:
    """Builds server collaborators and shared runtime state in one place."""

    @staticmethod
    def default_class_map() -> dict[str, Any]:
        """Build production default class wiring map for build_default()."""
        from ...ace_framework import ACEFramework
        from ...adaptive_rate_allocator import ContextWindowAdapter, MultiLevelSemanticEncoder
        from ...afm import AFMConfig, FocusManager
        from ...blind_spot_detector import BlindSpotDetector, HaloEffectDetector
        from ...code_compression_adapter import CodeCompressionAdapter
        from ...file_sync_manager import FileSyncManager
        from ...path_validator import PathValidator
        from ...persistence import PersistenceManager
        from ...resource_manager import ResourceLimits, ResourceManager
        from ...version_manager import VersionManager
        from .ace_context_manager import ACEContextManager
        from .context_service import ServerContextService
        from .lifecycle_service import ServerLifecycleService
        from .persistence_orchestration_service import PersistenceOrchestrationService
        from .progress_service import ProgressRenderService
        from .runtime_service import RuntimeService
        from .server_service_adapter import ServerServiceAdapter
        from .tool_profile_service import ToolProfileBootstrapService
        from .tooling import MCPToolingGateway

        return {
            "CodeCompressionAdapter": CodeCompressionAdapter,
            "BlindSpotDetector": BlindSpotDetector,
            "HaloEffectDetector": HaloEffectDetector,
            "ContextWindowAdapter": ContextWindowAdapter,
            "MultiLevelSemanticEncoder": MultiLevelSemanticEncoder,
            "AFMConfig": AFMConfig,
            "FocusManager": FocusManager,
            "PersistenceManager": PersistenceManager,
            "ResourceLimits": ResourceLimits,
            "ResourceManager": ResourceManager,
            "FileSyncManager": FileSyncManager,
            "VersionManager": VersionManager,
            "PathValidator": PathValidator,
            "ACEFramework": ACEFramework,
            "ACEContextManager": ACEContextManager,
            "MCPToolingGateway": MCPToolingGateway,
            "ServerContextService": ServerContextService,
            "ServerLifecycleService": ServerLifecycleService,
            "ProgressRenderService": ProgressRenderService,
            "PersistenceOrchestrationService": PersistenceOrchestrationService,
            "ToolProfileBootstrapService": ToolProfileBootstrapService,
            "RuntimeService": RuntimeService,
            "ServerServiceAdapter": ServerServiceAdapter,
        }

    @staticmethod
    def resolve_class_overrides(
        *,
        defaults: dict[str, Any],
        overrides: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Merge class overrides into defaults with strict unknown-key validation."""
        validate_override_keys(overrides=overrides, allowed_keys=ALLOWED_OVERRIDE_KEYS)
        active_overrides = overrides or {}
        return {**defaults, **active_overrides}

    @staticmethod
    def build_kwargs_from_resolved_classes(resolved_classes: dict[str, Any]) -> dict[str, Any]:
        """Translate resolved class alias map into build() keyword arguments."""
        return {
            "code_adapter_cls": resolved_classes["CodeCompressionAdapter"],
            "blind_spot_cls": resolved_classes["BlindSpotDetector"],
            "halo_cls": resolved_classes["HaloEffectDetector"],
            "context_window_adapter_cls": resolved_classes["ContextWindowAdapter"],
            "multilevel_encoder_cls": resolved_classes["MultiLevelSemanticEncoder"],
            "afm_config_cls": resolved_classes["AFMConfig"],
            "focus_manager_cls": resolved_classes["FocusManager"],
            "persistence_cls": resolved_classes["PersistenceManager"],
            "resource_limits_cls": resolved_classes["ResourceLimits"],
            "resource_manager_cls": resolved_classes["ResourceManager"],
            "file_sync_cls": resolved_classes["FileSyncManager"],
            "version_manager_cls": resolved_classes["VersionManager"],
            "path_validator_cls": resolved_classes["PathValidator"],
            "ace_framework_cls": resolved_classes["ACEFramework"],
            "ace_context_manager_cls": resolved_classes["ACEContextManager"],
            "tooling_gateway_cls": resolved_classes["MCPToolingGateway"],
            "context_service_cls": resolved_classes["ServerContextService"],
            "lifecycle_service_cls": resolved_classes["ServerLifecycleService"],
            "progress_service_cls": resolved_classes["ProgressRenderService"],
            "persistence_service_cls": resolved_classes["PersistenceOrchestrationService"],
            "tool_profile_service_cls": resolved_classes["ToolProfileBootstrapService"],
            "runtime_service_cls": resolved_classes["RuntimeService"],
            "server_service_adapter_cls": resolved_classes["ServerServiceAdapter"],
        }

    @staticmethod
    def code_adapter_config(*, preload_code_model: bool) -> dict[str, Any]:
        """Default constructor kwargs for the code compression adapter."""
        return {
            "text_model": "all-MiniLM-L6-v2",
            "code_model": "microsoft/codebert-base",
            "similarity_threshold": 0.75,
            "skeleton_ratio": 0.2,
            "preload_code_model": preload_code_model,
        }

    @staticmethod
    def afm_config_kwargs() -> dict[str, Any]:
        """Default AFM configuration used by production wiring."""
        return {
            "tau_high": 0.45,
            "tau_mid": 0.25,
            "half_life": 12,
            "use_llm_importance": False,
            "use_llm_compression": False,
        }

    @staticmethod
    def resource_limits_kwargs() -> dict[str, Any]:
        """Default resource limits for server runtime."""
        return {
            "max_document_size_mb": 100.0,
            "max_total_storage_mb": 1024.0,
            "max_documents": 1000,
            "max_memory_mb": 2048.0,
        }

    @staticmethod
    def ace_framework_kwargs() -> dict[str, Any]:
        """Default ACE framework configuration for bullet extraction."""
        return {
            "deduplication_threshold": 0.85,
            "max_bullets": 100,
        }

    @staticmethod
    def default_context_window_monitor() -> dict[str, Any]:
        """Default context window monitor shape for session state."""
        return {"max_tokens": 100000, "used_tokens": 0, "history": []}

    @staticmethod
    def file_sync_log_kwargs() -> dict[str, Any]:
        """Structured log fields for file-sync initialization."""
        return {"status": "enabled"}

    @staticmethod
    def path_validator_log_kwargs(*, allowed_base_dirs: list[str]) -> dict[str, Any]:
        """Structured log fields for path-validator initialization."""
        return {
            "allowed_directories_count": len(allowed_base_dirs),
            "security_feature": "CWE-22 path traversal prevention",
        }

    @staticmethod
    def ace_framework_log_kwargs(*, max_ace_contexts: int) -> dict[str, Any]:
        """Structured log fields for ACE framework initialization."""
        return {
            **ServerFactoryService.ace_framework_kwargs(),
            "max_contexts": max_ace_contexts,
        }

    @classmethod
    def build_core_runtime_layer(
        cls,
        *,
        cwd: str,
        home_dir: str,
        max_ace_contexts: int,
        afm_config,
        focus_manager_cls,
        persistence_cls,
        resource_limits_cls,
        resource_manager_cls,
        file_sync_cls,
        version_manager_cls,
        path_validator_cls,
        ace_framework_cls,
        ace_context_manager_cls,
        logger,
    ) -> CoreRuntimeArtifacts:
        """Build foundational runtime collaborators outside the service layer."""
        focus_manager = focus_manager_cls(afm_config)

        persistence = persistence_cls()
        resource_manager = resource_manager_cls(resource_limits_cls(**cls.resource_limits_kwargs()))

        sync_manager = file_sync_cls()
        version_manager = version_manager_cls()
        logger.info("file_sync_initialized", **cls.file_sync_log_kwargs())

        path_validator = path_validator_cls(allowed_base_dirs=[cwd, home_dir])
        logger.info(
            "path_validator_initialized",
            **cls.path_validator_log_kwargs(allowed_base_dirs=[cwd, home_dir]),
        )

        ace_framework = ace_framework_cls(**cls.ace_framework_kwargs())
        ace_contexts = ace_context_manager_cls(max_contexts=max_ace_contexts)
        logger.info(
            "ace_framework_initialized",
            **cls.ace_framework_log_kwargs(max_ace_contexts=max_ace_contexts),
        )

        return {
            "focus_manager": focus_manager,
            "persistence": persistence,
            "resource_manager": resource_manager,
            "sync_manager": sync_manager,
            "version_manager": version_manager,
            "path_validator": path_validator,
            "ace_framework": ace_framework,
            "ace_contexts": ace_contexts,
        }

    @classmethod
    def build_service_layer(
        cls,
        *,
        context_service_cls,
        lifecycle_service_cls,
        progress_service_cls,
        persistence_service_cls,
        tool_profile_service_cls,
        runtime_service_cls,
        server_service_adapter_cls,
        logger,
    ) -> ServiceLayerArtifacts:
        """Build service-layer collaborators and wire the service adapter."""
        context_service = context_service_cls()
        lifecycle_service = lifecycle_service_cls()
        progress_service = progress_service_cls()
        persistence_service = persistence_service_cls()
        tool_profile_service = tool_profile_service_cls()
        runtime_service = runtime_service_cls()
        service_adapter = server_service_adapter_cls(
            persistence_service=persistence_service,
            context_service=context_service,
            progress_service=progress_service,
            logger=logger,
        )

        return {
            "context_service": context_service,
            "lifecycle_service": lifecycle_service,
            "progress_service": progress_service,
            "persistence_service": persistence_service,
            "tool_profile_service": tool_profile_service,
            "runtime_service": runtime_service,
            "service_adapter": service_adapter,
        }

    @classmethod
    def build_default(
        cls,
        *,
        preload_code_model: bool,
        cwd: str,
        home_dir: str,
        max_ace_contexts: int,
        logger,
        class_overrides: dict[str, Any] | None = None,
    ) -> BuildArtifacts:
        resolved_classes = cls.resolve_class_overrides(
            defaults=cls.default_class_map(),
            overrides=class_overrides,
        )

        return cls.build(
            preload_code_model=preload_code_model,
            cwd=cwd,
            home_dir=home_dir,
            max_ace_contexts=max_ace_contexts,
            logger=logger,
            **cls.build_kwargs_from_resolved_classes(resolved_classes),
        )

    @classmethod
    def build(
        cls,
        *,
        preload_code_model: bool,
        cwd: str,
        home_dir: str,
        max_ace_contexts: int,
        code_adapter_cls,
        blind_spot_cls,
        halo_cls,
        context_window_adapter_cls,
        multilevel_encoder_cls,
        afm_config_cls,
        focus_manager_cls,
        persistence_cls,
        resource_limits_cls,
        resource_manager_cls,
        file_sync_cls,
        version_manager_cls,
        path_validator_cls,
        ace_framework_cls,
        ace_context_manager_cls,
        tooling_gateway_cls,
        context_service_cls,
        lifecycle_service_cls,
        progress_service_cls,
        persistence_service_cls,
        tool_profile_service_cls,
        runtime_service_cls,
        server_service_adapter_cls,
        logger,
    ) -> BuildArtifacts:
        compressor = code_adapter_cls(
            **cls.code_adapter_config(preload_code_model=preload_code_model)
        )
        blind_spot_detector = blind_spot_cls(compressor)
        halo_detector = halo_cls(compressor)
        context_window_adapter = context_window_adapter_cls(compressor)
        multilevel_encoder = multilevel_encoder_cls(compressor)

        afm_config = afm_config_cls(**cls.afm_config_kwargs())

        core = cls.build_core_runtime_layer(
            cwd=cwd,
            home_dir=home_dir,
            max_ace_contexts=max_ace_contexts,
            afm_config=afm_config,
            focus_manager_cls=focus_manager_cls,
            persistence_cls=persistence_cls,
            resource_limits_cls=resource_limits_cls,
            resource_manager_cls=resource_manager_cls,
            file_sync_cls=file_sync_cls,
            version_manager_cls=version_manager_cls,
            path_validator_cls=path_validator_cls,
            ace_framework_cls=ace_framework_cls,
            ace_context_manager_cls=ace_context_manager_cls,
            logger=logger,
        )

        services = cls.build_service_layer(
            context_service_cls=context_service_cls,
            lifecycle_service_cls=lifecycle_service_cls,
            progress_service_cls=progress_service_cls,
            persistence_service_cls=persistence_service_cls,
            tool_profile_service_cls=tool_profile_service_cls,
            runtime_service_cls=runtime_service_cls,
            server_service_adapter_cls=server_service_adapter_cls,
            logger=logger,
        )

        return {
            "compressor": compressor,
            "blind_spot_detector": blind_spot_detector,
            "halo_detector": halo_detector,
            "context_window_adapter": context_window_adapter,
            "multilevel_encoder": multilevel_encoder,
            **core,
            "tooling": tooling_gateway_cls(),
            **services,
            "context_window_monitor": cls.default_context_window_monitor(),
            "retrieval_history": {},
        }
