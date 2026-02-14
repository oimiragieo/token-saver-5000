"""App-layer factory service for SemanticModulatorServer composition wiring."""

from __future__ import annotations

from typing import Any

from .server_aliases import SERVER_ALIAS_KEYS

APP_OVERRIDE_KEYS: tuple[str, ...] = (
    "PathValidator",
    "ServerContextService",
    "ServerLifecycleService",
    "ProgressRenderService",
    "PersistenceOrchestrationService",
    "ToolProfileBootstrapService",
    "RuntimeService",
    "ServerServiceAdapter",
)

ALLOWED_OVERRIDE_KEYS: frozenset[str] = frozenset((*SERVER_ALIAS_KEYS, *APP_OVERRIDE_KEYS))


class ServerFactoryService:
    """Builds server collaborators and shared runtime state in one place."""

    @staticmethod
    def resolve_class_overrides(
        *,
        defaults: dict[str, Any],
        overrides: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Merge class overrides into defaults with strict unknown-key validation."""
        active_overrides = overrides or {}
        unknown_keys = sorted(set(active_overrides) - set(ALLOWED_OVERRIDE_KEYS))
        if unknown_keys:
            unknown_csv = ", ".join(unknown_keys)
            raise ValueError(f"Unknown class_overrides keys: {unknown_csv}")
        return {**defaults, **active_overrides}

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
    ) -> dict[str, Any]:
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

        resolved_classes = cls.resolve_class_overrides(
            defaults={
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
            },
            overrides=class_overrides,
        )

        return cls.build(
            preload_code_model=preload_code_model,
            cwd=cwd,
            home_dir=home_dir,
            max_ace_contexts=max_ace_contexts,
            code_adapter_cls=resolved_classes["CodeCompressionAdapter"],
            blind_spot_cls=resolved_classes["BlindSpotDetector"],
            halo_cls=resolved_classes["HaloEffectDetector"],
            context_window_adapter_cls=resolved_classes["ContextWindowAdapter"],
            multilevel_encoder_cls=resolved_classes["MultiLevelSemanticEncoder"],
            afm_config_cls=resolved_classes["AFMConfig"],
            focus_manager_cls=resolved_classes["FocusManager"],
            persistence_cls=resolved_classes["PersistenceManager"],
            resource_limits_cls=resolved_classes["ResourceLimits"],
            resource_manager_cls=resolved_classes["ResourceManager"],
            file_sync_cls=resolved_classes["FileSyncManager"],
            version_manager_cls=resolved_classes["VersionManager"],
            path_validator_cls=resolved_classes["PathValidator"],
            ace_framework_cls=resolved_classes["ACEFramework"],
            ace_context_manager_cls=resolved_classes["ACEContextManager"],
            tooling_gateway_cls=resolved_classes["MCPToolingGateway"],
            context_service_cls=resolved_classes["ServerContextService"],
            lifecycle_service_cls=resolved_classes["ServerLifecycleService"],
            progress_service_cls=resolved_classes["ProgressRenderService"],
            persistence_service_cls=resolved_classes["PersistenceOrchestrationService"],
            tool_profile_service_cls=resolved_classes["ToolProfileBootstrapService"],
            runtime_service_cls=resolved_classes["RuntimeService"],
            server_service_adapter_cls=resolved_classes["ServerServiceAdapter"],
            logger=logger,
        )

    @staticmethod
    def build(
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
    ) -> dict[str, Any]:
        compressor = code_adapter_cls(
            text_model="all-MiniLM-L6-v2",
            code_model="microsoft/codebert-base",
            similarity_threshold=0.75,
            skeleton_ratio=0.2,
            preload_code_model=preload_code_model,
        )
        blind_spot_detector = blind_spot_cls(compressor)
        halo_detector = halo_cls(compressor)
        context_window_adapter = context_window_adapter_cls(compressor)
        multilevel_encoder = multilevel_encoder_cls(compressor)

        afm_config = afm_config_cls(
            tau_high=0.45,
            tau_mid=0.25,
            half_life=12,
            use_llm_importance=False,
            use_llm_compression=False,
        )
        focus_manager = focus_manager_cls(afm_config)

        persistence = persistence_cls()
        resource_manager = resource_manager_cls(
            resource_limits_cls(
                max_document_size_mb=100.0,
                max_total_storage_mb=1024.0,
                max_documents=1000,
                max_memory_mb=2048.0,
            )
        )

        sync_manager = file_sync_cls()
        version_manager = version_manager_cls()
        logger.info("file_sync_initialized", status="enabled")

        path_validator = path_validator_cls(allowed_base_dirs=[cwd, home_dir])
        logger.info(
            "path_validator_initialized",
            allowed_directories_count=2,
            security_feature="CWE-22 path traversal prevention",
        )

        ace_framework = ace_framework_cls(
            deduplication_threshold=0.85,
            max_bullets=100,
        )
        ace_contexts = ace_context_manager_cls(max_contexts=max_ace_contexts)
        logger.info(
            "ace_framework_initialized",
            deduplication_threshold=0.85,
            max_bullets=100,
            max_contexts=max_ace_contexts,
        )

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
            "compressor": compressor,
            "blind_spot_detector": blind_spot_detector,
            "halo_detector": halo_detector,
            "context_window_adapter": context_window_adapter,
            "multilevel_encoder": multilevel_encoder,
            "focus_manager": focus_manager,
            "persistence": persistence,
            "resource_manager": resource_manager,
            "sync_manager": sync_manager,
            "version_manager": version_manager,
            "path_validator": path_validator,
            "ace_framework": ace_framework,
            "ace_contexts": ace_contexts,
            "tooling": tooling_gateway_cls(),
            "context_service": context_service,
            "lifecycle_service": lifecycle_service,
            "progress_service": progress_service,
            "persistence_service": persistence_service,
            "tool_profile_service": tool_profile_service,
            "runtime_service": runtime_service,
            "service_adapter": service_adapter,
            "context_window_monitor": {"max_tokens": 100000, "used_tokens": 0, "history": []},
            "retrieval_history": {},
        }
