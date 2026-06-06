"""App-layer factory service for SemanticModulatorServer composition wiring."""

from __future__ import annotations

from typing import Any, TypedDict, cast

from .contract_validation import (
    contract_key_mismatch_message as _contract_key_mismatch_message,
    validate_contract_keys as _validate_contract_keys,
)
from .server_aliases import ALLOWED_FACTORY_OVERRIDE_KEYS, validate_override_keys

ALLOWED_OVERRIDE_KEYS: frozenset[str] = ALLOWED_FACTORY_OVERRIDE_KEYS


class FactoryClassMap(TypedDict):
    """Class-alias map used by build_default/override resolution."""

    CodeCompressionAdapter: Any
    BlindSpotDetector: Any
    HaloEffectDetector: Any
    ContextWindowAdapter: Any
    MultiLevelSemanticEncoder: Any
    AFMConfig: Any
    FocusManager: Any
    PersistenceManager: Any
    ResourceLimits: Any
    ResourceManager: Any
    FileSyncManager: Any
    VersionManager: Any
    PathValidator: Any
    ACEFramework: Any
    ACEContextManager: Any
    MCPToolingGateway: Any
    ServerContextService: Any
    ServerLifecycleService: Any
    ProgressRenderService: Any
    PersistenceOrchestrationService: Any
    ToolProfileBootstrapService: Any
    RuntimeService: Any
    ServerServiceAdapter: Any


class BuildKwargsMap(TypedDict):
    """Keyword arguments required by ServerFactoryService.build()."""

    code_adapter_cls: Any
    blind_spot_cls: Any
    halo_cls: Any
    context_window_adapter_cls: Any
    multilevel_encoder_cls: Any
    afm_config_cls: Any
    focus_manager_cls: Any
    persistence_cls: Any
    resource_limits_cls: Any
    resource_manager_cls: Any
    file_sync_cls: Any
    version_manager_cls: Any
    path_validator_cls: Any
    ace_framework_cls: Any
    ace_context_manager_cls: Any
    tooling_gateway_cls: Any
    context_service_cls: Any
    lifecycle_service_cls: Any
    progress_service_cls: Any
    persistence_service_cls: Any
    tool_profile_service_cls: Any
    runtime_service_cls: Any
    server_service_adapter_cls: Any


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


class FactoryValidationResult(TypedDict):
    """Validated class and build-kwargs maps used by build_default."""

    resolved_classes: FactoryClassMap
    build_kwargs: BuildKwargsMap


class DefaultBuildInputs(TypedDict):
    """Validated runtime and wiring inputs used by build_default orchestration."""

    request: BuildDefaultRequest
    validation: FactoryValidationResult


class BuildDefaultRequest(TypedDict):
    """Default-build runtime parameters that are independent from class wiring."""

    preload_code_model: bool
    cwd: str
    home_dir: str
    max_ace_contexts: int
    logger: Any


class BuildRequest(BuildKwargsMap, BuildDefaultRequest):
    """Complete build request payload for ServerFactoryService.build(...)."""


BUILD_DEFAULT_REQUEST_KEYS: frozenset[str] = frozenset(BuildDefaultRequest.__annotations__.keys())
FACTORY_VALIDATION_RESULT_KEYS: frozenset[str] = frozenset(
    FactoryValidationResult.__annotations__.keys()
)
DEFAULT_BUILD_INPUTS_KEYS: frozenset[str] = frozenset(DefaultBuildInputs.__annotations__.keys())
BUILD_REQUEST_KEYS: frozenset[str] = frozenset(BuildRequest.__annotations__.keys())
BUILD_KWARGS_KEYS: frozenset[str] = frozenset(BuildKwargsMap.__annotations__.keys())


class ServerFactoryService:
    """Builds server collaborators and shared runtime state in one place."""

    @staticmethod
    def contract_key_mismatch_message(
        *,
        contract_name: str,
        missing: list[str],
        extra: list[str],
    ) -> str:
        """Build a canonical contract-drift message for key mismatches."""
        return _contract_key_mismatch_message(
            contract_name=contract_name, missing=missing, extra=extra
        )

    @classmethod
    def validate_contract_keys(
        cls,
        *,
        contract_name: str,
        payload: dict[str, Any],
        expected_keys: frozenset[str],
    ) -> None:
        """Fail fast when a payload keyset drifts from the expected contract."""
        _validate_contract_keys(
            contract_name=contract_name, payload=payload, expected_keys=expected_keys
        )

    @staticmethod
    def default_class_map() -> FactoryClassMap:
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

    @classmethod
    def validate_default_class_map(cls, default_map: dict[str, Any]) -> FactoryClassMap:
        """Fail fast when default class map keys drift from allowed override aliases."""
        cls.validate_contract_keys(
            contract_name="default_class_map",
            payload=default_map,
            expected_keys=ALLOWED_OVERRIDE_KEYS,
        )

        return cast(FactoryClassMap, default_map)

    @staticmethod
    def resolve_class_overrides(
        *,
        defaults: FactoryClassMap,
        overrides: dict[str, Any] | None,
    ) -> FactoryClassMap:
        """Merge class overrides into defaults with strict unknown-key validation."""
        validate_override_keys(overrides=overrides, allowed_keys=ALLOWED_OVERRIDE_KEYS)
        active_overrides = overrides or {}
        return cast(FactoryClassMap, {**defaults, **active_overrides})

    @staticmethod
    def build_kwargs_from_resolved_classes(resolved_classes: FactoryClassMap) -> BuildKwargsMap:
        """Translate resolved class alias map into build() keyword arguments."""
        return cast(
            BuildKwargsMap,
            {
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
            },
        )

    @classmethod
    def validate_build_kwargs_map(cls, build_kwargs: dict[str, Any]) -> BuildKwargsMap:
        """Fail fast when build-kwargs map drifts from constructor kwargs contract."""
        cls.validate_contract_keys(
            contract_name="build_kwargs_map",
            payload=build_kwargs,
            expected_keys=BUILD_KWARGS_KEYS,
        )

        return cast(BuildKwargsMap, build_kwargs)

    @staticmethod
    def validate_build_request_map(build_request: dict[str, Any]) -> BuildRequest:
        """Fail fast on required runtime-key drift or unknown keys in build request."""
        actual_keys = set(build_request.keys())
        missing = sorted(BUILD_DEFAULT_REQUEST_KEYS - actual_keys)
        extra = sorted(actual_keys - BUILD_REQUEST_KEYS)
        if missing or extra:
            raise ValueError(
                ServerFactoryService.contract_key_mismatch_message(
                    contract_name="build_request_map",
                    missing=missing,
                    extra=extra,
                )
            )

        return cast(BuildRequest, build_request)

    @staticmethod
    def code_adapter_config(*, preload_code_model: bool) -> dict[str, Any]:
        """Default constructor kwargs for the code compression adapter.

        ``skeleton_ratio="auto"`` selects the engine's adaptive curve
        (``compute_adaptive_ratio``): ~0.8 for <8k-token docs, scaling down to
        0.1 for >=100k-token docs.  This keeps small/medium docs faithful
        (a ~5-node prose doc surfaces ~4 nodes) while large docs stay
        aggressively compressed.  The old fixed 0.2 floored any <=5-node doc to
        a single node — an over-aggressive summary rather than "the compressed
        version".  Aggressive 0.2 remains reachable as an explicit caller param.
        """
        return {
            "text_model": "all-MiniLM-L6-v2",
            "code_model": "microsoft/codebert-base",
            "similarity_threshold": 0.75,
            "skeleton_ratio": "auto",
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
    def validate_factory_contracts(
        cls,
        *,
        class_overrides: dict[str, Any] | None,
    ) -> FactoryValidationResult:
        """Run class-map and build-kwargs validations before factory build."""
        defaults = cls.validate_default_class_map(cls.default_class_map())
        resolved_classes = cls.resolve_class_overrides(
            defaults=defaults,
            overrides=class_overrides,
        )
        build_kwargs = cls.validate_build_kwargs_map(
            cls.build_kwargs_from_resolved_classes(resolved_classes)
        )
        return {
            "resolved_classes": resolved_classes,
            "build_kwargs": build_kwargs,
        }

    @staticmethod
    def build_default_request(
        *,
        preload_code_model: bool,
        cwd: str,
        home_dir: str,
        max_ace_contexts: int,
        logger,
    ) -> BuildDefaultRequest:
        """Build a typed request payload for default factory orchestration."""
        return {
            "preload_code_model": preload_code_model,
            "cwd": cwd,
            "home_dir": home_dir,
            "max_ace_contexts": max_ace_contexts,
            "logger": logger,
        }

    @classmethod
    def validate_build_default_request_map(
        cls,
        request: dict[str, Any],
    ) -> BuildDefaultRequest:
        """Fail fast when default runtime request keys drift from contract."""
        cls.validate_contract_keys(
            contract_name="build_default_request_map",
            payload=request,
            expected_keys=BUILD_DEFAULT_REQUEST_KEYS,
        )
        return cast(BuildDefaultRequest, request)

    @classmethod
    def validate_factory_validation_result_map(
        cls,
        validation: dict[str, Any],
    ) -> FactoryValidationResult:
        """Fail fast when factory validation payload drifts from contract."""
        cls.validate_contract_keys(
            contract_name="factory_validation_result_map",
            payload=validation,
            expected_keys=FACTORY_VALIDATION_RESULT_KEYS,
        )
        return cast(FactoryValidationResult, validation)

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
        inputs = cls.validate_default_build_inputs_map(
            cls.validate_default_build_inputs(
                preload_code_model=preload_code_model,
                cwd=cwd,
                home_dir=home_dir,
                max_ace_contexts=max_ace_contexts,
                logger=logger,
                class_overrides=class_overrides,
            )
        )
        return cls.build_default_from_validation(
            request=inputs["request"],
            validation=inputs["validation"],
        )

    @classmethod
    def validate_default_build_inputs_map(
        cls,
        inputs: dict[str, Any],
    ) -> DefaultBuildInputs:
        """Fail fast when DefaultBuildInputs map drifts from contract."""
        cls.validate_contract_keys(
            contract_name="default_build_inputs_map",
            payload=inputs,
            expected_keys=DEFAULT_BUILD_INPUTS_KEYS,
        )
        validated_request = cls.validate_build_default_request_map(
            cast(dict[str, Any], inputs["request"])
        )
        validated_validation = cls.validate_factory_validation_result_map(
            cast(dict[str, Any], inputs["validation"])
        )
        return {"request": validated_request, "validation": validated_validation}

    @classmethod
    def validate_default_build_inputs(
        cls,
        *,
        preload_code_model: bool,
        cwd: str,
        home_dir: str,
        max_ace_contexts: int,
        logger,
        class_overrides: dict[str, Any] | None,
    ) -> DefaultBuildInputs:
        """Validate runtime request and class-wiring contracts for build_default()."""
        request = cls.build_default_request(
            preload_code_model=preload_code_model,
            cwd=cwd,
            home_dir=home_dir,
            max_ace_contexts=max_ace_contexts,
            logger=logger,
        )
        validated_request = cls.validate_build_default_request_map(request)
        validation = cls.validate_factory_contracts(class_overrides=class_overrides)
        return {"request": validated_request, "validation": validation}

    @classmethod
    def build_default_from_validation(
        cls,
        *,
        request: BuildDefaultRequest,
        validation: FactoryValidationResult,
    ) -> BuildArtifacts:
        """Build default runtime artifacts from pre-validated class wiring contracts."""
        validated_request = cls.validate_build_default_request_map(request)
        validated_validation = cls.validate_factory_validation_result_map(validation)
        build_request = cls.build_request_from_default_validation(
            request=validated_request,
            validation=validated_validation,
        )
        return cls.build_from_request(request=build_request)

    @staticmethod
    def build_request_from_default_validation(
        *,
        request: BuildDefaultRequest,
        validation: FactoryValidationResult,
    ) -> BuildRequest:
        """Merge runtime request parameters and validated class wiring into one payload."""
        return cast(BuildRequest, {**request, **validation["build_kwargs"]})

    @classmethod
    def build_from_request(
        cls,
        *,
        request: BuildRequest,
    ) -> BuildArtifacts:
        """Dispatch full typed build request through the canonical build entrypoint."""
        validated_request = cls.validate_build_request_map(request)
        return cls.build(**validated_request)

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
