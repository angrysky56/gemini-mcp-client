#!/usr/bin/env python3
"""
Simple MCP Server for testing without external dependencies
"""

import logging
import sys
from mcp.server.fastmcp import FastMCP

# Configure logging to stderr only
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr
)

# Create the MCP server
mcp = FastMCP("Simple Test Server")


@mcp.tool()
def echo_message(message: str) -> str:
    """Echo back a message."""
    return f"Echo: {message}"


@mcp.tool()
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@mcp.tool()
def get_server_info() -> dict:
    """Get information about this server."""
    return {
        "name": "Simple Test Server",
        "version": "1.0.0",
        "description": "A simple MCP server for testing",
        "tools": ["echo_message", "add_numbers", "get_server_info"]
    }


@mcp.resource("test://greeting/{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting."""
    return f"Hello, {name}! This is from the Simple Test Server."


@mcp.prompt()
def test_prompt(topic: str) -> str:
    """Generate a test prompt."""
    return f"Please tell me about {topic}."


if __name__ == "__main__":
    mcp.run()
