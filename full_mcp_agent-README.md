# Full MCP Agent

A dynamic ADK agent that automatically loads all enabled MCP servers from the configuration file, providing comprehensive access to all available tools without hardcoding server definitions.

## Features

- **Dynamic Configuration Loading**: Automatically reads `servers.json` to load all enabled MCP servers
- **No AFC Limits**: Removes the Automatic Function Calling (AFC) limit by not restricting tools
- **Flexible Architecture**: Easily add/remove MCP servers by editing the config file
- **Smart Instruction Generation**: Dynamically generates agent instructions based on available servers
- **Error Handling**: Gracefully handles server loading failures with fallback behavior

## Key Improvements Over multi_mcp_agent

1. **Dynamic vs Hardcoded**: No more hardcoding MCP server configurations in the agent file
2. **Configuration-Driven**: All server settings managed through `config/servers.json`
3. **No Redundancy**: Removed unnecessary test agents (sqlite_agent, docker_agent, etc.)
4. **No AFC Limit**: By setting `tool_filter=None`, all tools are available without the 10-call limit
5. **Better Maintainability**: Add new MCP servers by updating config, no code changes needed

## Usage

Run the agent using ADK Web from the agents directory:

```bash
cd /gemini-mcp-client/agents
adk web
```

Then select `full_mcp_agent` from the dropdown in the UI.

## Configuration

The agent reads from `/config/servers.json`. To add or modify your own MCP servers:

1. Edit `config/servers.json`
2. Set `"enabled": true` for servers you want to use
3. Restart the agent

Example server configuration:
```json
{
  "new-server": {
    "name": "new-server",
    "description": "Description of the new server",
    "server_type": "python",
    "command": "python",
    "args": ["path/to/server.py"],
    "env": {
      "CUSTOM_VAR": "value"
    },
    "enabled": true,
    "tags": ["tag1", "tag2"]
  }
}
```

## Available Models

The agent uses `gemini-2.0-flash-lite` by default, which offers:
- 30 requests per minute
- 1500 requests per day
- Good balance of performance and availability

You can change the model by editing the `model` parameter in `agent.py`.

## Debug Mode

A debug agent (`full_mcp_agent_debug`) is also available for verbose output during troubleshooting.

## Technical Details

- **No Tool Filter**: Setting `tool_filter=None` in MCPToolset ensures all tools from each server are available
- **Dynamic Instructions**: The agent instruction is generated based on enabled servers, providing context-aware guidance
- **Error Recovery**: If config loading fails, the agent falls back to an empty tool list with an appropriate message
