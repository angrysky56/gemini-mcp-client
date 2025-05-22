"""
Simple Echo MCP Server for testing the client
"""

import sys
import logging
from mcp.server.fastmcp import FastMCP

# Configure logging to go to stderr instead of stdout to avoid mixing with MCP protocol
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr  # Use stderr instead of stdout
)

# Create MCP server
mcp = FastMCP("Echo Server")


@mcp.tool()
def echo(message: str) -> str:
    """Echo back the provided message."""
    return f"Echo: {message}"


@mcp.tool()
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@mcp.tool() 
def get_current_time() -> str:
    """Get the current time."""
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting."""
    return f"Hello, {name}! Welcome to the Echo Server."


@mcp.prompt()
def generate_poem(topic: str) -> str:
    """Generate a simple poem about a topic."""
    return f"Please write a short poem about {topic}."


if __name__ == "__main__":
    # Run the server with minimal logging
    mcp.run()
