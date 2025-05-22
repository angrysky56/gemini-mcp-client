# Gemini MCP Client
- AI generated, untested as of now, try at own risk.

A powerful MCP (Model Context Protocol) client that uses Google's Gemini AI models for intelligent tool usage and conversation handling.

## Features

- **Multiple Gemini Models**: Support for various Gemini models (2.0-flash, 2.5-pro, 1.5-pro, etc.)
- **Flexible Package Support**: Works with gemini-tool-agent, google-generativeai, or google-genai packages
- **Runtime Model Switching**: Change models during conversation
- **Intelligent Tool Usage**: AI-powered tool discovery and execution
- **Async/await Support**: Efficient concurrent operations
- **Comprehensive Error Handling**: Robust error recovery and logging
- **Type Hints**: Modern Python practices with full type safety

## Installation

This project uses `uv` for dependency management. Make sure you have `uv` installed.

### Setting up the environment

```bash
# Create and activate virtual environment
uv venv --python 3.12 --seed
source .venv/bin/activate

# Install base dependencies
uv add .

# Install a Gemini package (choose one or more):
uv add --optional-dependencies gemini-tool-agent
# OR
uv add --optional-dependencies google-generativeai
# OR
uv add --optional-dependencies google-genai
# OR install all options
uv add --optional-dependencies all-gemini
```

### Development setup

```bash
# Install with development dependencies
uv add --dev ".[dev]"

# Install pre-commit hooks
uv run pre-commit install
```

## Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
LOG_LEVEL=INFO
```

## Usage

### Model Selection

The client supports multiple Gemini models:

- `gemini-2.0-flash` - Fast, efficient model (default)
- `gemini-2.5-pro-preview-03-25` - Advanced model for complex tasks
- `gemini-2.5-flash-preview-04-17` - Balanced speed and capability
- `gemini-1.5-pro` - Stable, reliable model
- `gemini-1.5-flash` - Fast, lightweight model
- `gemini-1.5-flash-8b` - Ultra-lightweight model

### Command Line Usage

```bash
# Start chat with default model
gemini-mcp-client chat path/to/server.py

# Start chat with specific model
gemini-mcp-client chat server.py --model gemini-2.5-pro-preview-03-25

# Get server information
gemini-mcp-client info server.py

# List available models
gemini-mcp-client models

# Set logging level
gemini-mcp-client chat server.py --log-level DEBUG
```

### Programmatic Usage

```python
import asyncio
from gemini_mcp_client import MCPClient

async def main():
    # Initialize with specific model
    client = MCPClient(model="gemini-2.5-pro-preview-03-25")

    try:
        # Connect to server
        await client.connect_to_server("examples/echo_server.py")

        # Get server information
        info = await client.get_server_info()
        print(f"Current model: {info['model']}")
        print(f"Available tools: {[tool['name'] for tool in info['tools']]}")

        # Change model during runtime
        client.set_model("gemini-2.0-flash")

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

### Runtime Model Switching

During an interactive chat session, you can change models:

```
💬 You: model gemini-2.5-pro-preview-03-25
✅ Model changed to: gemini-2.5-pro-preview-03-25

💬 You: What's the current model?
🤖 Assistant: I'm currently using gemini-2.5-pro-preview-03-25...
```

## Features

### AI-Powered Conversations
- **Intelligent Tool Selection**: AI determines when and which tools to use
- **Context-Aware Responses**: Maintains conversation history for better responses
- **Natural Language Interaction**: Understands user intent and responds appropriately

### Model Management
- **Multiple Model Support**: Works with different Gemini model versions
- **Runtime Switching**: Change models without restarting the session
- **Package Flexibility**: Supports multiple Gemini Python packages

### Connection Management
- **Async/await Support**: Efficient handling of multiple operations
- **Proper Resource Cleanup**: Automatic cleanup of connections and resources
- **Error Recovery**: Robust error handling and recovery mechanisms
- **Multi-server Support**: Connect to different MCP servers

### Tool Integration
- **Automatic Discovery**: Finds and catalogs available tools from servers
- **Intelligent Parameter Mapping**: AI determines appropriate tool parameters
- **Error Handling**: Graceful handling of tool execution failures
- **Direct Tool Access**: Call tools directly without AI mediation

## Server Compatibility

The client works with any MCP server that follows the standard protocol:

- **Python servers**: Using `mcp.server` or `FastMCP`
- **JavaScript servers**: Using `@modelcontextprotocol/sdk`
- **Other implementations**: Any server implementing MCP protocol

## Package Dependencies

The client supports multiple Gemini packages and will use the first available:

1. **gemini-tool-agent**: Advanced tool-calling capabilities
2. **google-generativeai**: Official Google SDK (legacy)
3. **google-genai**: New official Google SDK

Install at least one of these packages for full functionality.

## Error Handling

The client includes comprehensive error handling:

- **Connection Errors**: Automatic retry and graceful failure handling
- **Tool Execution Errors**: Clear error messages and recovery
- **Network Issues**: Timeout handling and reconnection
- **Resource Errors**: Proper cleanup and error reporting

## Logging

Configure logging levels for debugging:

```python
client = MCPClient(log_level="DEBUG")
```

Available levels: DEBUG, INFO, WARNING, ERROR

Log files are written to `mcp_client.log` in the project directory.

## Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini API key (required)
- `LOG_LEVEL`: Default logging level (INFO, DEBUG, etc.)

## MCP Server Configuration

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
        "gemini-mcp-client",
        "chat",
        "path/to/your/server.py",
        "--model",
        "gemini-2.5-pro-preview-03-25"
      ],
      "env": {
        "GEMINI_API_KEY": "your_gemini_api_key_here"
      }
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

### Model Selection Example

```bash
# List available models
gemini-mcp-client models

# Use specific model
gemini-mcp-client chat server.py --model gemini-2.5-pro-preview-03-25
```

### Custom Tool Usage

```python
# Connect and use tools programmatically
client = MCPClient(model="gemini-2.0-flash")
await client.connect_to_server("server.py")

# Call tools directly
result = await client.call_tool_directly("add_numbers", {"a": 5, "b": 3})
print(f"Result: {result}")  # Result: 8

# Use AI to determine which tool to use
response = await client.get_response("Add 10 and 20 together")
print(response)  # AI will automatically use add_numbers tool
```

## Tips for Effective Usage

1. **Model Selection**:
   - Use `gemini-2.0-flash` for quick responses
   - Use `gemini-2.5-pro-preview-03-25` for complex reasoning
   - Switch models based on task complexity

2. **Clear Prompts**: Be specific about what you want to accomplish

3. **Tool Awareness**: Ask about available tools to understand capabilities

4. **Context Management**: The AI maintains conversation history for better responses

5. **Error Handling**: Check logs if something goes wrong

6. **Resource Management**: Always close the client when done

## Development

### Code Quality

```bash
# Format code
uv run ruff format .

# Check code
uv run ruff check .

# Type checking
uv run pyright

# Run tests
uv run pytest
```

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=gemini_mcp_client

# Run specific test
uv run pytest tests/test_client.py::test_model_selection
```

## Architecture

The client is built with these key components:

- **MCPClient**: Main client class handling server connections and conversations
- **GeminiAgent**: Wrapper for different Gemini package implementations
- **Async Context Management**: Proper resource cleanup and connection handling
- **Model Management**: Runtime model selection and switching
- **Error Handling**: Comprehensive error recovery and logging
- **Type Safety**: Full type hints for better development experience

## Contributing

1. Follow the development guidelines in the codebase
2. Ensure all tests pass
3. Add type hints to all new code
4. Write docstrings for public APIs
5. Keep functions focused and small
6. Test with multiple Gemini packages

## License

MIT License - see LICENSE file for details.
