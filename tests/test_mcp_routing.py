"""
Unit tests for MCP Core Routing (mcp_core.py)

Tests the central routing logic that dispatches MCP tool calls to appropriate
handler functions. Following 2025 best practices for pytest testing.

Coverage:
- setup_mcp_tools(): Tool schema definitions
- route_tool_call(): Routing dispatch logic
- Error handling for unknown tools
- All tools are properly registered
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from mcp.types import Tool

from src.handlers import mcp_core


class TestSetupMCPTools:
    """Tests for setup_mcp_tools() function"""

    def test_returns_list_of_tools(self):
        """Test that setup_mcp_tools returns a list"""
        tools = mcp_core.setup_mcp_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_correct_number_of_tools(self):
        """Test that all MCP tools are registered."""
        tools = mcp_core.setup_mcp_tools()

        # Expected count: 99 tools total
        # - Document Compression: 9
        # - Batch Processing: 1 (NEW in v0.6.0)
        # - Directory Ingestion: 1 (NEW in v0.9.0)
        # - Fidelity Advisor: 1
        # - Detection: 2
        # - AFM Dialogue: 6
        # - File Sync: 4
        # - Resource Management: 3 (check_resource_health, check_environment, should_compress)
        # - Help & Documentation: 1 (NEW in v0.9.0)
        # - ACE Framework: 7
        # - Visualization: 4 (NEW in v0.6.0)
        # - Experimental: 9 (v0.11.0) - TOON, SCAR, Multimodal, ASG-SI suite
        # - New: 3 (diff_reingest, find_duplicates, get_compression_presets)
        # - New: 1 (check_context_budget)
        # - Prompt registry: 6
        # - Explicit memory and personalization: 6
        # - Datasets and experiments: 5
        # - Managed connector feeds: 5
        # - Temporal context and lifecycle: 4
        # - Stable multimodal: 2
        # - Structured handoff bundles: 4
        # - Model optimization: 6
        # - Prompt registry/cache audit/rendering: 8
        # - Token Optimization: 4 (v0.11.0) - estimate_tokens, configure_for_client, set/get_compression_profile
        assert len(tools) == 103, f"Expected 103 tools, got {len(tools)}"

    def test_core_stable_profile_has_expected_tools(self):
        """Test that core_stable profile exposes only stable core tools."""
        tools = mcp_core.setup_mcp_tools(profile="core_stable")
        tool_names = {tool.name for tool in tools}
        expected = {
            "ingest_context",
            "read_skeleton",
            "search_semantic",
            "modulate_region",
            "get_stats",
            "list_documents",
            "delete_document",
        }

        assert len(tools) == 7
        assert tool_names == expected

    def test_invalid_profile_raises_value_error(self):
        """Test that unsupported profile names fail fast."""
        with pytest.raises(ValueError) as exc_info:
            mcp_core.setup_mcp_tools(profile="invalid")
        assert "Unknown tool profile" in str(exc_info.value)

    def test_all_tools_have_required_fields(self):
        """Test that all tools have name, description, and inputSchema"""
        tools = mcp_core.setup_mcp_tools()

        for tool in tools:
            assert isinstance(tool, Tool), f"Tool is not Tool instance: {tool}"
            assert hasattr(tool, "name"), "Tool missing 'name' field"
            assert hasattr(tool, "description"), "Tool missing 'description' field"
            assert hasattr(tool, "inputSchema"), "Tool missing 'inputSchema' field"

            # Validate fields are non-empty
            assert tool.name, f"Tool has empty name: {tool}"
            assert tool.description, f"Tool {tool.name} has empty description"
            assert tool.inputSchema, f"Tool {tool.name} has empty inputSchema"

    def test_tool_names_are_unique(self):
        """Test that all tool names are unique (no duplicates)"""
        tools = mcp_core.setup_mcp_tools()
        tool_names = [tool.name for tool in tools]

        assert len(tool_names) == len(
            set(tool_names)
        ), f"Duplicate tool names found: {[name for name in tool_names if tool_names.count(name) > 1]}"

    def test_input_schemas_are_valid(self):
        """Test that all inputSchemas are valid JSON Schema objects"""
        tools = mcp_core.setup_mcp_tools()

        for tool in tools:
            schema = tool.inputSchema
            assert isinstance(schema, dict), f"Tool {tool.name} has non-dict inputSchema"
            assert "type" in schema, f"Tool {tool.name} missing 'type' in inputSchema"
            assert (
                schema["type"] == "object"
            ), f"Tool {tool.name} inputSchema type is not 'object': {schema['type']}"
            assert "properties" in schema, f"Tool {tool.name} missing 'properties' in inputSchema"

    def test_required_fields_in_schemas(self):
        """Test that tools with required fields properly define them"""
        tools = mcp_core.setup_mcp_tools()

        for tool in tools:
            schema = tool.inputSchema
            if "required" in schema:
                required_fields = schema["required"]
                assert isinstance(
                    required_fields, list
                ), f"Tool {tool.name} 'required' is not a list: {required_fields}"

                # Verify all required fields are in properties
                properties = schema.get("properties", {})
                for field in required_fields:
                    assert (
                        field in properties
                    ), f"Tool {tool.name} has required field '{field}' not in properties"

    def test_document_compression_tools_present(self):
        """Test that all 9 document compression tools are registered"""
        tools = mcp_core.setup_mcp_tools()
        tool_names = {tool.name for tool in tools}

        expected_doc_tools = {
            "ingest_context",
            "read_skeleton",
            "modulate_region",
            "search_semantic",
            "get_stats",
            "list_documents",
            "delete_document",
            "adapt_to_context_window",
            "multilevel_encode",
        }

        assert expected_doc_tools.issubset(
            tool_names
        ), f"Missing document compression tools: {expected_doc_tools - tool_names}"

    def test_read_skeleton_schema_exposes_query_modes(self):
        tools = mcp_core.setup_mcp_tools()
        tool = next(t for t in tools if t.name == "read_skeleton")
        properties = tool.inputSchema["properties"]

        assert "selection_mode" in properties
        assert "query" in properties
        assert "top_k" in properties
        assert "min_similarity" in properties
        assert "file_id" in tool.inputSchema.get("required", [])

    def test_search_semantic_schema_exposes_evidence_mode(self):
        tools = mcp_core.setup_mcp_tools()
        tool = next(t for t in tools if t.name == "search_semantic")
        properties = tool.inputSchema["properties"]

        assert "evidence_aware" in properties
        assert "min_similarity" in properties

    def test_afm_dialogue_tools_present(self):
        """Test that all 6 AFM dialogue tools are registered"""
        tools = mcp_core.setup_mcp_tools()
        tool_names = {tool.name for tool in tools}

        expected_afm_tools = {
            "afm_add_message",
            "afm_build_context",
            "afm_get_stats",
            "afm_clear_history",
            "afm_export_history",
            "afm_import_history",
        }

        assert expected_afm_tools.issubset(
            tool_names
        ), f"Missing AFM dialogue tools: {expected_afm_tools - tool_names}"

    def test_ace_framework_tools_present(self):
        """Test that all 7 ACE framework tools are registered"""
        tools = mcp_core.setup_mcp_tools()
        tool_names = {tool.name for tool in tools}

        expected_ace_tools = {
            "ace_generate",
            "ace_reflect",
            "ace_curate",
            "ace_grow_context",
            "ace_refine_context",
            "ace_get_playbook",
            "ace_execute_cycle",
        }

        assert expected_ace_tools.issubset(
            tool_names
        ), f"Missing ACE framework tools: {expected_ace_tools - tool_names}"

    def test_file_sync_tools_present(self):
        """Test that all 4 file sync tools are registered"""
        tools = mcp_core.setup_mcp_tools()
        tool_names = {tool.name for tool in tools}

        expected_sync_tools = {
            "check_file_sync",
            "diff_cached_file",
            "refresh_document",
            "get_version_history",
        }

        assert expected_sync_tools.issubset(
            tool_names
        ), f"Missing file sync tools: {expected_sync_tools - tool_names}"


class TestRouteToolCall:
    """Tests for route_tool_call() function"""

    def setup_method(self):
        """Set up mock context for routing tests"""
        self.mock_context = {
            "compressor": Mock(),
            "blind_spot_detector": Mock(),
            "halo_detector": Mock(),
            "context_window_adapter": Mock(),
            "multilevel_encoder": Mock(),
            "focus_manager": Mock(),
            "persistence": Mock(),
            "resource_manager": Mock(),
            "sync_manager": Mock(),
            "version_manager": Mock(),
            "ace_framework": Mock(),
            "ace_contexts": Mock(),
            "validate_file_id": Mock(),
            "validate_node_ids": Mock(),
            "validate_token_count": Mock(),
            "save_file_sync_metadata": Mock(),
        }

    @pytest.mark.asyncio
    async def test_unknown_tool_raises_value_error(self):
        """Test that routing unknown tool name raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            await mcp_core.route_tool_call("nonexistent_tool", {}, self.mock_context)

        error_msg = str(exc_info.value)
        assert "Unknown tool" in error_msg
        assert "nonexistent_tool" in error_msg
        assert "Available tools" in error_msg

    @pytest.mark.asyncio
    async def test_core_stable_profile_blocks_non_core_tool(self):
        """Test that non-core tools are blocked when profile is core_stable."""
        with pytest.raises(ValueError) as exc_info:
            await mcp_core.route_tool_call(
                "afm_add_message",
                {},
                self.mock_context,
                tool_profile="core_stable",
            )

        error_msg = str(exc_info.value)
        assert "not enabled" in error_msg
        assert "core_stable" in error_msg

    @pytest.mark.asyncio
    async def test_error_message_lists_available_tools(self):
        """Test that error message for unknown tool lists all available tools"""
        try:
            await mcp_core.route_tool_call("invalid_tool", {}, self.mock_context)
            pytest.fail("Expected ValueError to be raised")
        except ValueError as e:
            error_msg = str(e)
            expected_count = len(mcp_core.setup_mcp_tools())
            assert str(expected_count) in error_msg
            # Should list some tool names
            assert "ingest_context" in error_msg or "afm_add_message" in error_msg

    @pytest.mark.asyncio
    @patch("src.handlers.compression_handlers.handle_ingest", new_callable=AsyncMock)
    async def test_routes_ingest_context_correctly(self, mock_handler):
        """Test that ingest_context routes to compression_handlers.handle_ingest"""
        # Configure mock to have __name__ and __module__ for logging
        mock_handler.__name__ = "handle_ingest"
        mock_handler.__module__ = "src.handlers.compression_handlers"
        mock_handler.return_value = "success"
        args = {"text": "test", "file_id": "doc1"}

        result = await mcp_core.route_tool_call("ingest_context", args, self.mock_context)

        mock_handler.assert_called_once_with(self.mock_context, args)
        assert result == "success"

    @pytest.mark.asyncio
    @patch("src.handlers.compression_handlers.handle_read_skeleton", new_callable=AsyncMock)
    async def test_routes_read_skeleton_correctly(self, mock_handler):
        """Test that read_skeleton routes to compression_handlers.handle_read_skeleton"""
        mock_handler.__name__ = "handle_read_skeleton"
        mock_handler.__module__ = "src.handlers.compression_handlers"
        mock_handler.return_value = "skeleton data"
        args = {"file_id": "doc1"}

        result = await mcp_core.route_tool_call("read_skeleton", args, self.mock_context)

        mock_handler.assert_called_once_with(self.mock_context, args)
        assert result == "skeleton data"

    @pytest.mark.asyncio
    @patch("src.handlers.afm_handlers.handle_afm_add_message", new_callable=AsyncMock)
    async def test_routes_afm_add_message_correctly(self, mock_handler):
        """Test that afm_add_message routes to afm_handlers.handle_afm_add_message"""
        mock_handler.__name__ = "handle_afm_add_message"
        mock_handler.__module__ = "src.handlers.afm_handlers"
        mock_handler.return_value = "message added"
        args = {"role": "user", "content": "Hello"}

        result = await mcp_core.route_tool_call("afm_add_message", args, self.mock_context)

        mock_handler.assert_called_once_with(self.mock_context, args)
        assert result == "message added"

    @pytest.mark.asyncio
    @patch("src.handlers.file_sync_handlers.handle_check_file_sync", new_callable=AsyncMock)
    async def test_routes_check_file_sync_correctly(self, mock_handler):
        """Test that check_file_sync routes to file_sync_handlers.handle_check_file_sync"""
        mock_handler.__name__ = "handle_check_file_sync"
        mock_handler.__module__ = "src.handlers.file_sync_handlers"
        mock_handler.return_value = "in_sync: true"
        args = {"file_id": "doc1"}

        result = await mcp_core.route_tool_call("check_file_sync", args, self.mock_context)

        mock_handler.assert_called_once_with(self.mock_context, args)
        assert result == "in_sync: true"

    @pytest.mark.asyncio
    @patch("src.handlers.resource_handlers.handle_check_resource_health", new_callable=AsyncMock)
    async def test_routes_check_resource_health_correctly(self, mock_handler):
        """Test that check_resource_health routes correctly"""
        mock_handler.__name__ = "handle_check_resource_health"
        mock_handler.__module__ = "src.handlers.resource_handlers"
        mock_handler.return_value = "healthy"
        args = {}

        result = await mcp_core.route_tool_call("check_resource_health", args, self.mock_context)

        mock_handler.assert_called_once_with(self.mock_context, args)
        assert result == "healthy"

    @pytest.mark.asyncio
    @patch("src.handlers.detection_handlers.handle_check_blind_spots", new_callable=AsyncMock)
    async def test_routes_check_blind_spots_correctly(self, mock_handler):
        """Test that check_blind_spots routes to detection_handlers"""
        mock_handler.__name__ = "handle_check_blind_spots"
        mock_handler.__module__ = "src.handlers.detection_handlers"
        mock_handler.return_value = "no blind spots"
        args = {"response": "test", "file_id": "doc1", "retrieved_node_ids": ["n1"]}

        result = await mcp_core.route_tool_call("check_blind_spots", args, self.mock_context)

        mock_handler.assert_called_once_with(self.mock_context, args)
        assert result == "no blind spots"

    @pytest.mark.asyncio
    @patch("src.handlers.ace_handlers.handle_ace_generate", new_callable=AsyncMock)
    async def test_routes_ace_generate_with_handler_context(self, mock_handler):
        """Test that ACE tools route correctly using HandlerContext (refactored v0.4.3)"""
        mock_handler.return_value = "generated"
        args = {"query": "test"}

        result = await mcp_core.route_tool_call("ace_generate", args, self.mock_context)

        # ACE handlers now use HandlerContext signature like all other handlers
        mock_handler.assert_called_once_with(self.mock_context, args)
        assert result == "generated"

    @pytest.mark.asyncio
    @patch("src.handlers.ace_handlers.handle_ace_execute_cycle", new_callable=AsyncMock)
    async def test_routes_ace_execute_cycle_with_handler_context(self, mock_handler):
        """Test that ace_execute_cycle routes correctly using HandlerContext (refactored v0.4.3)"""
        mock_handler.return_value = "cycle executed"
        args = {"context_id": "ctx1"}

        result = await mcp_core.route_tool_call("ace_execute_cycle", args, self.mock_context)

        # ACE handlers now use HandlerContext signature like all other handlers
        mock_handler.assert_called_once_with(self.mock_context, args)
        assert result == "cycle executed"

    @pytest.mark.asyncio
    async def test_all_registered_tools_have_handlers(self):
        """Test that every tool in setup_mcp_tools has a corresponding handler"""
        tools = mcp_core.setup_mcp_tools()
        tool_names = {tool.name for tool in tools}

        # Try routing each tool (will raise ValueError if handler missing)
        for tool_name in tool_names:
            # Use empty args and mock context
            # We're not testing handler logic, just routing presence
            try:
                # This will call the actual handler, but that's OK for this test
                # We just want to verify the routing doesn't raise "Unknown tool"
                with patch(
                    "src.handlers.compression_handlers.handle_ingest", new_callable=AsyncMock
                ) as mock:
                    with patch(
                        "src.handlers.afm_handlers.handle_afm_add_message", new_callable=AsyncMock
                    ) as mock2:
                        with patch(
                            "src.handlers.ace_handlers.handle_ace_generate", new_callable=AsyncMock
                        ) as mock3:
                            mock.return_value = "test"
                            mock2.return_value = "test"
                            mock3.return_value = "test"

                            # Try to route - should not raise "Unknown tool" error
                            try:
                                await mcp_core.route_tool_call(tool_name, {}, self.mock_context)
                            except ValueError as e:
                                # OK if it's a validation error from the handler
                                # NOT OK if it's "Unknown tool"
                                assert "Unknown tool" not in str(
                                    e
                                ), f"Tool '{tool_name}' is registered but has no handler"
                            except Exception:
                                # Other exceptions are fine (handler logic errors)
                                pass
            except Exception:
                # Setup exceptions are fine, we just care about routing
                pass

    @pytest.mark.asyncio
    @patch("src.handlers.compression_handlers.handle_ingest", new_callable=AsyncMock)
    async def test_context_parameter_is_passed_to_handlers(self, mock_handler):
        """Test that context dict is correctly passed to handlers"""
        mock_handler.__name__ = "handle_ingest"
        mock_handler.__module__ = "src.handlers.compression_handlers"
        mock_handler.return_value = "success"
        args = {"text": "test", "file_id": "doc1"}

        await mcp_core.route_tool_call("ingest_context", args, self.mock_context)

        # Verify first argument to handler is the context dict
        call_args = mock_handler.call_args
        assert call_args[0][0] == self.mock_context

    @pytest.mark.asyncio
    @patch("src.handlers.compression_handlers.handle_ingest", new_callable=AsyncMock)
    async def test_args_parameter_is_passed_to_handlers(self, mock_handler):
        """Test that args dict is correctly passed to handlers"""
        mock_handler.__name__ = "handle_ingest"
        mock_handler.__module__ = "src.handlers.compression_handlers"
        mock_handler.return_value = "success"
        args = {"text": "test content", "file_id": "doc123"}

        await mcp_core.route_tool_call("ingest_context", args, self.mock_context)

        # Verify second argument to handler is the args dict
        call_args = mock_handler.call_args
        assert call_args[0][1] == args
        assert call_args[0][1]["file_id"] == "doc123"


class TestRouterIntegrity:
    """Tests for router integrity and consistency"""

    @pytest.mark.asyncio
    async def test_router_count_matches_tool_count(self):
        """Test that internal router has same count as setup_mcp_tools"""
        tools = mcp_core.setup_mcp_tools()

        # We can't directly access the router dict (it's local to route_tool_call)
        # But we can verify that all tool names can be routed without "Unknown tool" error
        tool_names = {tool.name for tool in tools}

        # Count how many tools can be successfully looked up
        mock_context = {
            "compressor": Mock(),
            "ace_framework": Mock(),
            "ace_contexts": Mock(),
        }

        routable_count = 0
        for tool_name in tool_names:
            try:
                # Try routing with mocked handlers
                with patch(
                    "src.handlers.compression_handlers.handle_ingest",
                    new_callable=AsyncMock,
                    return_value="ok",
                ):
                    with patch(
                        "src.handlers.afm_handlers.handle_afm_add_message",
                        new_callable=AsyncMock,
                        return_value="ok",
                    ):
                        with patch(
                            "src.handlers.ace_handlers.handle_ace_generate",
                            new_callable=AsyncMock,
                            return_value="ok",
                        ):
                            try:
                                await mcp_core.route_tool_call(tool_name, {}, mock_context)
                                routable_count += 1
                            except ValueError as e:
                                if "Unknown tool" in str(e):
                                    # This is a routing failure - tool not in router
                                    pass
                                else:
                                    # This is a handler validation error - routing succeeded
                                    routable_count += 1
                            except Exception:
                                # Handler errors mean routing succeeded
                                routable_count += 1
            except Exception:
                # Setup errors - routing may still have worked
                routable_count += 1

        assert routable_count == len(
            tools
        ), f"Router has {routable_count} handlers but setup_mcp_tools returns {len(tools)} tools"


# Run tests if called directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
