#!/usr/bin/env python3
"""
Multi-MCP Agent for ADK Web Interface
This agent connects to multiple MCP servers to provide comprehensive functionality.
"""

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool import MCPToolset
from google.adk.tools.mcp_tool.mcp_toolset import StdioServerParameters

# Define all your MCP servers
mcp_tools = []

# SQLite MCP Server
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

# Docker MCP Server
mcp_tools.append(
    MCPToolset(
        connection_params=StdioServerParameters(
            command='uvx',
            args=['docker-mcp'],
        ),
        tool_filter=None,
    )
)

# Desktop Commander MCP Server
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


# ArXiv MCP Server
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

# Package Version MCP Server
mcp_tools.append(
    MCPToolset(
        connection_params=StdioServerParameters(
            command='/home/ty/Repositories/mcp-package-version/bin/mcp-package-version',
            args=[],
        ),
        tool_filter=None,
    )
)

# Code Executor MCP Server
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

# Define the root agent with all MCP tools
agent = LlmAgent(
    model='gemini-2.0-flash-lite',  # Lightweight model with good rate limits
    name='multi_mcp_agent',
    instruction="""You are a comprehensive AI assistant with access to multiple powerful tools:

 **SQLite Database**: Query and manage algorithm platform database
 **Docker**: Manage containers, images, and deployments
 **Desktop Commander**: Filesystem read/write, edit_block and CLI system operations
 **ArXiv Research**: Search, download, and analyze academic papers
 **Package Versions**: Check latest versions of software packages across ecosystems
 **Code Executor**: Execute Python code safely in isolated environments

Use these tools intelligently to help users with:
- Data analysis and database operations
- Container and deployment management
- CLI automation and file functions
- Document storage and semantic search
- Research paper analysis and retrieval
- Software dependency management
- Code development and testing

Always explain what you're doing and suggest the most appropriate tools for each task.
""",
    description="Multi-MCP assistant with comprehensive tooling capabilities",
    tools=mcp_tools,
)

# DEPRECATED: This agent uses hardcoded MCP server configurations.
# Please use full_mcp_agent instead, which dynamically loads servers from config/servers.json
# See: /agents/full_mcp_agent/

# Note: The individual test agents (sqlite_agent, docker_agent, etc.) have been removed.
# If you need to test individual MCP servers, you can:
# 1. Disable other servers in config/servers.json
# 2. Use the full_mcp_agent with only the desired server enabled
# 3. Create a custom agent using the pattern in full_mcp_agent
