#!/usr/bin/env python3
"""
Configurable Multi-MCP Agent for ADK Web Interface
This agent loads model configuration dynamically and connects to multiple MCP servers.
"""

import os
import json
from pathlib import Path
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool import MCPToolset
from google.adk.tools.mcp_tool.mcp_toolset import StdioServerParameters

# Load configuration
config_path = Path(__file__).parent.parent / "agent_config.json"
if config_path.exists():
    with open(config_path) as f:
        config = json.load(f)
    model_name = config.get("default_model", "gemini-2.0-flash-lite")
else:
    model_name = "gemini-2.0-flash-lite"

# Define all your MCP servers
mcp_tools = []

# 1. SQLite MCP Server
mcp_tools.append(
    MCPToolset(
        connection_params=StdioServerParameters(
            command='uv',
            args=[
                '--directory',
                '/home/ty/Repositories/servers/src/sqlite',
                'run',
                'mcp-server-sqlite',
                '--db-path',
                '/home/ty/Repositories/ai_workspace/algorithm_platform/data/algo.db'
            ],
        ),
        tool_filter=None,  # Include all tools from this server
    )
)

# 2. Docker MCP Server
mcp_tools.append(
    MCPToolset(
        connection_params=StdioServerParameters(
            command='uvx',
            args=['docker-mcp'],
        ),
        tool_filter=None,
    )
)

# 3. Desktop Commander MCP Server
mcp_tools.append(
    MCPToolset(
        connection_params=StdioServerParameters(
            command='npx',
            args=[
                '-y',
                '/home/ty/Repositories/DesktopCommanderMCP/dist/index.js',
                'run',
                'desktop-commander',
                '--config',
                '"{}"'
            ],
        ),
        tool_filter=None,
    )
)

# 4. Chroma Vector Database MCP Server
mcp_tools.append(
    MCPToolset(
        connection_params=StdioServerParameters(
            command='uvx',
            args=[
                'chroma-mcp',
                '--client-type',
                'persistent',
                '--data-dir',
                '/home/ty/Repositories/chroma-db'
            ],
        ),
        tool_filter=None,
    )
)

# 5. ArXiv MCP Server
mcp_tools.append(
    MCPToolset(
        connection_params=StdioServerParameters(
            command='uv',
            args=[
                '--directory',
                '/home/ty/Repositories/arxiv-mcp-server',
                'run',
                'arxiv-mcp-server',
                '--storage-path',
                '/home/ty/Documents/core_bot_instruction_concepts/arxiv-papers'
            ],
        ),
        tool_filter=None,
    )
)

# 6. Package Version MCP Server
mcp_tools.append(
    MCPToolset(
        connection_params=StdioServerParameters(
            command='/home/ty/Repositories/mcp-package-version/bin/mcp-package-version',
            args=[],
        ),
        tool_filter=None,
    )
)

# 7. Code Executor MCP Server
mcp_tools.append(
    MCPToolset(
        connection_params=StdioServerParameters(
            command='node',
            args=['/home/ty/Repositories/mcp_code_executor/build/index.js'],
            env={
                'CODE_STORAGE_DIR': '/home/ty/Repositories/ai_workspace/python_coding_storage/',
                'CONDA_ENV_NAME': 'mcp_code_executor_env'
            }
        ),
        tool_filter=None,
    )
)

# Define the main agent with configurable model
agent = LlmAgent(
    model=model_name,
    name='multi_mcp_assistant',
    instruction=f"""You are a comprehensive AI assistant running on {model_name} with access to multiple powerful tools:

1. **SQLite Database**: Query and manage algorithm platform database
2. **Docker**: Manage containers, images, and deployments
3. **Desktop Commander**: Control desktop applications and system operations
4. **Chroma Vector DB**: Store, search, and retrieve vectorized data and documents
5. **ArXiv Research**: Search, download, and analyze academic papers
6. **Package Versions**: Check latest versions of software packages across ecosystems
7. **Code Executor**: Execute Python code safely in isolated environments

Use these tools intelligently to help users with:
- Data analysis and database operations
- Container and deployment management
- System automation and desktop control
- Document storage and semantic search
- Research paper analysis and retrieval
- Software dependency management
- Code development and testing

Always explain what you're doing and suggest the most appropriate tools for each task.
Current model: {model_name}
""",
    description=f"Multi-MCP assistant with comprehensive tooling capabilities (Model: {model_name})",
    tools=mcp_tools,
)
