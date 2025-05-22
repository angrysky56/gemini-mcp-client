"""Tests for the MCP Client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from gemini_mcp_client.client import MCPClient, MCPClientError, ServerConnectionError


@pytest.fixture
def mock_gemini_agent():
    """Mock Gemini Agent for testing."""
    with patch('gemini_mcp_client.client.GEMINI_AVAILABLE', True):
        with patch('gemini_mcp_client.client.Agent') as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.tools = []
            mock_agent.history = []
            mock_agent.process_query.return_value = {"needs_direct_response": True, "direct_response": "Test response"}
            mock_agent.generate_response.return_value = "Generated response"
            mock_agent_class.return_value = mock_agent
            yield mock_agent


@pytest.fixture
def mock_session():
    """Mock MCP Session for testing."""
    session = AsyncMock()
    session.initialize = AsyncMock()
    session.list_tools = AsyncMock()
    session.list_resources = AsyncMock()
    session.list_prompts = AsyncMock()
    session.call_tool = AsyncMock()
    session.read_resource = AsyncMock()
    return session


class TestMCPClient:
    """Test cases for MCPClient."""

    def test_init_without_api_key(self):
        """Test initialization without API key."""
        client = MCPClient()
        assert client.session is None
        assert not client._connected

    def test_init_with_api_key(self, mock_gemini_agent):
        """Test initialization with API key."""
        client = MCPClient(api_key="test-key")
        assert client.session is None
        assert not client._connected

    @pytest.mark.asyncio
    async def test_connect_to_nonexistent_server(self):
        """Test connecting to a non-existent server."""
        client = MCPClient()
        
        with pytest.raises(ValueError, match="Server script not found"):
            await client.connect_to_server("nonexistent.py")

    @pytest.mark.asyncio
    async def test_connect_to_invalid_server_type(self):
        """Test connecting to an invalid server type."""
        client = MCPClient()
        
        with patch('pathlib.Path.exists', return_value=True):
            with pytest.raises(ValueError, match="Unsupported server script type"):
                await client.connect_to_server("server.txt")

    @pytest.mark.asyncio
    async def test_get_response_not_connected(self):
        """Test getting response when not connected."""
        client = MCPClient()
        
        with pytest.raises(MCPClientError, match="Not connected to server"):
            await client.get_response("test input")

    @pytest.mark.asyncio
    async def test_basic_response_without_agent(self):
        """Test basic response when agent is not available."""
        client = MCPClient()
        client._connected = True
        client.session = AsyncMock()
        client._server_info = {"tools": [{"name": "test_tool", "description": "Test tool"}]}
        
        response = await client._basic_response("what tools are available?")
        assert "test_tool" in response
        assert "Test tool" in response

    @pytest.mark.asyncio
    async def test_close_client(self):
        """Test closing the client."""
        client = MCPClient()
        client._connected = True
        client.exit_stack = AsyncMock()
        
        await client.close()
        assert not client._connected
        assert client.session is None
        client.exit_stack.aclose.assert_called_once()

    def test_repr(self):
        """Test string representation."""
        client = MCPClient()
        repr_str = repr(client)
        assert "MCPClient" in repr_str
        assert "connected=False" in repr_str
