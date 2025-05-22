# MCP Gemini Client

Credit to and started from- [medium.com/@fruitful2007/building-an-mcp-client-101-lets-build-one-for-a-gemini-chat-agent](https://medium.com/@fruitful2007/building-an-mcp-client-101-lets-build-one-for-a-gemini-chat-agent-37f9308a802b)

[pypi.org/project/gemini-tool-agent](https://pypi.org/project/gemini-tool-agent/)

A (potentially, lol) powerful MCP (Model Context Protocol) client that uses Google's Gemini AI models for intelligent tool usage and conversation handling.

### Based on untested AI gen code by a non-coder use at own risk.

## Features

- **Multiple Gemini Models**: Support for various Gemini models (2.0-flash, 2.5-pro, 1.5-pro, etc.)
- **Flexible Package Support**: Works with gemini-tool-agent, google-generativeai, or google-genai packages
- **Runtime Model Switching**: Change models during conversation
- **Server Configuration Management**: Store and manage MCP server configurations
- **Intelligent Tool Usage**: AI-powered tool discovery and execution
- **Async/await Support**: Efficient concurrent operations
- **Comprehensive Error Handling**: Robust error recovery and logging
- **Type Hints**: Modern Python practices with full type safety


📋 Easy Installation Instructions
Here's the correct way to set it up with Claude- maybe, doesn't throw an error, can't test for a few hours:
Just use this config in Claude Desktop:

```json
{
  "mcpServers": {
    "gemini-ai": {
      "command": "uv",
      "args": [
        "--directory",
        "/home/ty/Repositories/ai_workspace/gemini-mcp-client",
        "run",
        "python",
        "servers/gemini_mcp_server.py"
      ],
      "env": {
        "GEMINI_API_KEY": "your_gemini_api_key_here"
      }
    },
    "echo-server": {
      "command": "uv",
      "args": [
        "--directory",
        "/home/ty/Repositories/ai_workspace/gemini-mcp-client",
        "run",
        "python",
        "examples/echo_server.py"
      ]
    }
  }
}
```

## Installation outside of Claude if someone wants to try it and maybe develop a Standalone UI for Gemeni with MCP Tools!:
- possibly hallucinatory

This project uses `uv` for dependency management. Make sure you have `uv` installed.

### Setting up the environment

```bash
# Create and activate virtual environment
uv venv --python 3.12 --seed
source .venv/bin/activate


```bash
# Install base dependencies- this is wrong it seems.
# uv add .

# maybe
/gemini-mcp-client
uv sync
uv add mcp

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

The client supports multiple Gemini models- not sure if more can be added somehow:

- `gemini-2.0-flash` - Fast, efficient model (default)
- `gemini-2.5-pro-preview-03-25` - Advanced model for complex tasks
- `gemini-2.5-flash-preview-04-17` - Balanced speed and capability
- `gemini-1.5-pro` - Stable, reliable model
- `gemini-1.5-flash` - Fast, lightweight model
- `gemini-1.5-flash-8b` - Ultra-lightweight model

### Command Line Usage

```bash
# Start chat with default model
mcp-gemini-client chat path/to/server.py

# Start chat with specific model
mcp-gemini-client chat server.py --model gemini-2.5-pro-preview-03-25

# Get server information
mcp-gemini-client info server.py

# List available models
mcp-gemini-client models

# Set logging level
mcp-gemini-client chat server.py --log-level DEBUG
```

### Server Management

```bash
# List configured servers
mcp-gemini-client servers list

# Add new server interactively
mcp-gemini-client servers add

# Connect to configured server by name
mcp-gemini-client chat echo-server

# Enable/disable servers
mcp-gemini-client servers enable my-server
mcp-gemini-client servers disable my-server

# Export for Claude Desktop
mcp-gemini-client servers export claude_config.json

# Import from Claude Desktop config
mcp-gemini-client servers import existing_config.json
```

### Configuration Management

```bash
# Show current configuration
mcp-gemini-client config show

# Set default model
mcp-gemini-client config set default_model gemini-2.5-pro-preview-03-25

# Set other settings
mcp-gemini-client config set log_level DEBUG
mcp-gemini-client config set connection_timeout 60.0
```

### Programmatic Usage

```python
import asyncio
from mcp_gemini_client import MCPClient

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

## Server Configuration

### Configuration Files

- `config/servers.json` - MCP server configurations
- `config/client.json` - Client settings (model, timeouts, etc.)

### Example Server Configuration

```json
{
  "my-database": {
    "name": "my-database",
    "description": "Project SQLite database",
    "server_type": "python",
    "command": "uv",
    "args": ["run", "mcp-server-sqlite", "--db-path", "data/project.db"],
    "env": {
      "DB_READONLY": "false"
    },
    "enabled": true,
    "tags": ["database", "sqlite"]
  }
}
```

### Claude Desktop Integration

Export configurations directly for Claude Desktop:

```bash
mcp-gemini-client servers export
```

This creates a `claude_desktop_config.json` file:

```json
{
  "mcpServers": {
    "my-database": {
      "command": "uv",
      "args": ["run", "mcp-server-sqlite", "--db-path", "data/project.db"],
      "env": {
        "DB_READONLY": "false"
      }
    }
  }
}
```

## Quick Start

```bash
# Install and setup
uv add . --optional-dependencies google-generativeai
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# List available models
mcp-gemini-client models

# Test with echo server
mcp-gemini-client chat examples/echo_server.py

# Add a server configuration
mcp-gemini-client servers add

# Use configured server
mcp-gemini-client chat my-server

# Export for Claude Desktop
mcp-gemini-client servers export
```

## Examples

### Chat with Echo Server

```bash
mcp-gemini-client chat examples/echo_server.py
```

### Model Selection Example

```bash
# List available models
mcp-gemini-client models

# Use specific model
mcp-gemini-client chat server.py --model gemini-2.5-pro-preview-03-25
```

### Server Configuration Example

```python
from mcp_gemini_client.config import get_config_manager, ServerConfig, ServerType

config_manager = get_config_manager()

# Add a database server
db_server = ServerConfig(
    name="project-db",
    description="Project database server",
    server_type=ServerType.PYTHON,
    command="uv",
    args=["run", "mcp-server-sqlite", "--db-path", "data/project.db"],
    env={"DB_READONLY": "false"},
    tags=["database", "project"]
)

config_manager.add_server(db_server)

# Export for Claude Desktop
config_manager.export_claude_desktop_config("claude_config.json")
```

## Architecture

The client is built with these key components:

- **MCPClient**: Main client class handling server connections and conversations
- **GeminiAgent**: Wrapper for different Gemini package implementations
- **ConfigManager**: Server and client configuration management
- **Async Context Management**: Proper resource cleanup and connection handling
- **Model Management**: Runtime model selection and switching
- **Error Handling**: Comprehensive error recovery and logging
- **Type Safety**: Full type hints for better development experience

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
uv run pytest --cov=mcp_gemini_client

# Run specific test
uv run pytest tests/test_client.py::test_model_selection
```

## Documentation

- **Usage Guide**: `prompts/usage_guide.md`
- **Server Configuration**: `prompts/server_configuration_guide.md`
- **Troubleshooting**: `prompts/troubleshooting.md`
- **Best Practices**: `prompts/best_practices.md`

## Contributing

1. Follow the development guidelines in the codebase
2. Ensure all tests pass
3. Add type hints to all new code
4. Write docstrings for public APIs
5. Keep functions focused and small
6. Test with multiple Gemini packages

## License

MIT License - see LICENSE file for details.
