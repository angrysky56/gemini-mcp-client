#!/usr/bin/env python3
"""
Enhanced MCP client components with proper async cleanup.
Provides a more robust wrapper around stdio_client to handle GeneratorExit exceptions.
"""

import asyncio
import uuid
import sys
from typing import Any, AsyncGenerator, Dict, Optional, Tuple

# Import MCP components
try:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
    from mcp.client.stdio import StdioServerParameters
except ImportError:
    print("❌ MCP client libraries not available. Please install with: pip install mcp")
    sys.exit(1)

class EnhancedStdioClient:
    """Enhanced version of stdio_client with proper cleanup handling."""
    
    @staticmethod
    async def create_client(command: str, args: Optional[list] = None, env: Optional[Dict[str, str]] = None) -> AsyncGenerator[Tuple[Any, Any], None]:
        """
        Create an enhanced stdio client with proper cleanup handling.
        
        Args:
            command: The command to execute
            args: Optional command line arguments
            env: Optional environment variables
            
        Yields:
            Same as stdio_client: (read_stream, write_stream) tuple
        """
        client_gen = stdio_client(command, args, env)
        try:
            async for read_stream, write_stream in client_gen:
                try:
                    yield read_stream, write_stream
                except GeneratorExit:
                    print("Note: GeneratorExit caught in EnhancedStdioClient yield")
                    # Proper cleanup will happen in the finally block
                finally:
                    # We'll attempt to close streams properly
                    if hasattr(read_stream, 'aclose'):
                        try:
                            await read_stream.aclose()
                        except Exception as e:
                            print(f"Warning: Could not close read_stream: {e}")
                    
                    if hasattr(write_stream, 'aclose'):
                        try:
                            await write_stream.aclose()
                        except Exception as e:
                            print(f"Warning: Could not close write_stream: {e}")
        except GeneratorExit:
            print("Note: GeneratorExit caught in EnhancedStdioClient loop")
            # No need to raise, just clean exit
        except Exception as e:
            print(f"Error in EnhancedStdioClient: {e}")
            raise
        finally:
            # Try to ensure client_gen is properly closed
            if hasattr(client_gen, 'aclose'):
                try:
                    await client_gen.aclose()
                except Exception as e:
                    print(f"Warning: Could not close client_gen: {e}")

class EnhancedMCPBridge:
    """
    An enhanced MCP bridge that properly handles cleanup.
    This class wraps the standard ClientSession with improved error handling.
    """
    def __init__(self, session_id=None, server_params=None):
        """
        Initialize the enhanced MCP bridge.
        
        Args:
            session_id: Optional session ID (will generate one if not provided)
            server_params: Optional MCP server parameters
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.server_params = server_params
        self.client_session = None
        
    async def __aenter__(self):
        """Enter the context manager with proper initialization."""
        # Use our enhanced stdio_client instead of the original
        if self.server_params:
            self.client_session = ClientSession(
                self.session_id,
                read_stream_factory=lambda: EnhancedStdioClient.create_client(
                    self.server_params.command,
                    self.server_params.args,
                    self.server_params.env
                )
            )
        else:
            # If no server_params, create a default session
            self.client_session = ClientSession(self.session_id)
            
        await self.client_session.__aenter__()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager with proper cleanup."""
        try:
            if self.client_session:
                await self.client_session.__aexit__(exc_type, exc_val, exc_tb)
        except BaseExceptionGroup as e:
            # Filter out GeneratorExit exceptions
            filtered_exceptions = []
            for ex in e.exceptions:
                if not isinstance(ex.__cause__, GeneratorExit) and not isinstance(ex, GeneratorExit):
                    filtered_exceptions.append(ex)
            
            if filtered_exceptions:
                print(f"Filtered exceptions during MCPBridge cleanup: {filtered_exceptions}")
                if len(filtered_exceptions) == 1:
                    raise filtered_exceptions[0]
                else:
                    raise BaseExceptionGroup("filtered errors", filtered_exceptions)
            # If all exceptions were GeneratorExit, suppress them
            return True
        except Exception as e:
            print(f"Error during MCPBridge cleanup: {e}")
            if not isinstance(e, GeneratorExit):
                raise
        return True
    
    async def list_tools(self):
        """List available tools from the MCP server."""
        if self.client_session:
            return await self.client_session.list_tools()
        return []
    
    async def call_tool(self, tool_name, params=None):
        """Call a tool on the MCP server."""
        if self.client_session:
            return await self.client_session.call_tool(tool_name, params or {})
        return None

# Helper functions for common MCP server parameter configurations
def create_stdio_server_params(command, args=None, env=None):
    """Create MCP server parameters for stdio connection."""
    return StdioServerParameters(
        command=command,
        args=args or [],
        env=env or {}
    )

async def run_with_mcp_bridge(server_params, session_id=None, callback=None):
    """
    Run a function with an enhanced MCP bridge.
    
    Args:
        server_params: MCP server parameters
        session_id: Optional session ID
        callback: Function to call with the bridge
        
    Returns:
        Result of the callback function
    """
    async with EnhancedMCPBridge(session_id, server_params) as bridge:
        if callback:
            return await callback(bridge)
        return None
