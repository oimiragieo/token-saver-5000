"""App-layer runtime execution service for MCP stdio serving."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

from src.semantic_modulator.app.contract_validation import (
    contract_key_mismatch_message as _contract_key_mismatch_message,
    validate_contract_keys as _validate_contract_keys,
)


class RunRequest(TypedDict):
    """Runtime service request envelope for stdio execution."""

    server: Any
    logger: Any
    stdio_server_fn: Callable[[], Any]


class RuntimeService:
    """Executes MCP server runtime loop and startup diagnostics."""

    RUN_REQUEST_KEYS: frozenset[str] = frozenset(RunRequest.__annotations__.keys())

    @staticmethod
    def contract_key_mismatch_message(
        *,
        contract_name: str,
        missing: list[str],
        extra: list[str],
    ) -> str:
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
        _validate_contract_keys(
            contract_name=contract_name, payload=payload, expected_keys=expected_keys
        )

    @classmethod
    def validate_run_request_map(cls, request: dict[str, Any]) -> RunRequest:
        cls.validate_contract_keys(
            contract_name="run_request_map",
            payload=request,
            expected_keys=cls.RUN_REQUEST_KEYS,
        )
        return request

    @classmethod
    async def run(
        cls,
        *,
        server: Any,
        logger: Any,
        stdio_server_fn: Callable[[], Any],
    ) -> None:
        request = cls.validate_run_request_map(
            {
                "server": server,
                "logger": logger,
                "stdio_server_fn": stdio_server_fn,
            }
        )
        server = request["server"]
        logger = request["logger"]
        stdio_server_fn = request["stdio_server_fn"]

        if logger is not None:
            logger.info(
                "mcp_server_starting",
                server_name="Semantic Modulator",
                features=["Semantic Communication", "Fidelity-Preserving Encoding"],
                model="all-MiniLM-L6-v2",
                mode="Adaptive Semantic Fidelity",
            )

        async with stdio_server_fn() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
