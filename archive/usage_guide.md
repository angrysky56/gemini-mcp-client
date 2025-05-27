# Gemini MCP Client Usage Guide

## Overview

The Gemini MCP Client is a powerful tool for connecting to Model Context Protocol (MCP) servers and interacting with them using Google's Gemini AI model for intelligent conversation and tool usage.

## Basic Usage

### 1. Setup and Installation

```bash
# Navigate to project directory
cd /home/ty/Repositories/ai_workspace/gemini-mcp-client

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv add .

# Set up environment variables
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 2. Command Line Usage

```bash
# Start interactive chat with a server
gemini-mcp-client chat path/to/server.py

# Get information about a server
gemini-mcp-client info path/to/server.py

# Set logging level
gemini-mcp-client chat server.py --log-level DEBUG
```

### 3. Programmatic Usage

```python
import asyncio
from gemini_mcp_client import MCPClient

async def main():
    client = MCPClient()
    
    try:
        # Connect to server
        await client.connect_to_server("examples/echo_server.py")
        
        # Get server information
        info = await client.get_server_info()
        print(f"Available tools: {[tool['name'] for tool in info['tools']]}")
        
        # Interactive conversation
        response = await client.get_response("What tools are available?")
        print(response)
        
        # Direct tool usage
        result = await client.call_tool_directly("echo", {"message": "Hello!"})
        print(result)
        
        # Start chat loop
        await client.chat_loop()
        
    finally:
        await client.close()

asyncio.run(main())
```

## Features

### AI-Powered Conversations
- Intelligent tool selection and usage
- Context-aware responses
- Conversation history tracking
- Natural language interaction

### Connection Management
- Async/await support for efficient operations
- Proper resource cleanup
- Connection error handling and recovery
- Support for Python and JavaScript servers

### Tool Integration
- Automatic tool discovery
- Intelligent parameter mapping
- Error handling for tool failures
- Direct tool calling capability

### Resource Access
- Read resources from servers
- Handle different MIME types
- Efficient resource caching

## Server Compatibility

The client works with any MCP server that follows the standard protocol:

- **Python servers**: Using `mcp.server` or `FastMCP`
- **JavaScript servers**: Using `@modelcontextprotocol/sdk`
- **Other implementations**: Any server implementing MCP protocol

## Error Handling

The client includes comprehensive error handling:

- **Connection errors**: Automatic retry and graceful failure
- **Tool execution errors**: Clear error messages and recovery
- **Network issues**: Timeout handling and reconnection
- **Resource errors**: Proper cleanup and error reporting

## Logging

Configure logging levels for debugging:

```python
client = MCPClient(log_level="DEBUG")
```

Available levels: DEBUG, INFO, WARNING, ERROR

Log files are written to `mcp_client.log` in the project directory.

## Configuration

### Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini API key
- `LOG_LEVEL`: Default logging level (INFO, DEBUG, etc.)

### MCP Server Configuration

Add servers to Claude Desktop configuration:

```json
{
  "mcpServers": {
    "your-server": {
      "command": "uv",
      "args": [
        "--directory",
        "/home/ty/Repositories/ai_workspace/gemini-mcp-client",
        "run",
        "python",
        "path/to/your/server.py"
      ]
    }
  }
}
```

## Examples

### Chat with Echo Server

```bash
gemini-mcp-client chat examples/echo_server.py
```

### Get Server Information

```bash
gemini-mcp-client info examples/echo_server.py
```

### Custom Tool Usage

```python
# Connect and use tools programmatically
client = MCPClient()
await client.connect_to_server("server.py")

# Call tools directly
result = await client.call_tool_directly("add_numbers", {"a": 5, "b": 3})
print(f"Result: {result}")  # Result: 8

# Use AI to determine which tool to use
response = await client.get_response("Add 10 and 20 together")
print(response)  # AI will automatically use add_numbers tool
```

## Tips for Effective Usage

1. **Clear prompts**: Be specific about what you want to accomplish
2. **Tool awareness**: Ask about available tools to understand capabilities
3. **Context**: The AI maintains conversation history for better responses
4. **Error handling**: Check logs if something goes wrong
5. **Resource management**: Always close the client when done
