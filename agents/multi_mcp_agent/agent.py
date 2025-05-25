#!/usr/bin/env python3
"""
Multi-MCP Agent for ADK Web Interface
This agent connects to multiple MCP servers to provide comprehensive functionality.
"""

import os
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool import MCPToolset
from google.adk.tools.mcp_tool.mcp_toolset import StdioServerParameters

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

# Define the root agent with all MCP tools
agent = LlmAgent(
    model='gemini-1.5-flash-8b',  # Lightweight model with good rate limits
    name='multi_mcp_assistant',
    instruction="""You are a comprehensive AI assistant with access to multiple powerful tools:

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
""",
    description="Multi-MCP assistant with comprehensive tooling capabilities",
    tools=mcp_tools,
)

# Alternative single-server testing agents for debugging
sqlite_agent = LlmAgent(
    model='gemini-1.5-flash-8b', 
    name='sqlite_test_agent',
    instruction="You help users query and manage SQLite databases.",
    tools=[mcp_tools[0]]  # Just SQLite
)

docker_agent = LlmAgent(
    model='gemini-1.5-flash-8b',
    name='docker_test_agent', 
    instruction="You help users manage Docker containers and images.",
    tools=[mcp_tools[1]]  # Just Docker
)

chroma_agent = LlmAgent(
    model='gemini-1.5-flash-8b',
    name='chroma_test_agent',
    instruction="You help users with vector database operations using Chroma.",
    tools=[mcp_tools[3]]  # Just Chroma
)

arxiv_agent = LlmAgent(
    model='gemini-1.5-flash-8b',
    name='arxiv_test_agent',
    instruction="You help users search and analyze academic papers from ArXiv.",
    tools=[mcp_tools[4]]  # Just ArXiv
)

code_executor_agent = LlmAgent(
    model='gemini-1.5-flash-8b',
    name='code_executor_test_agent',
    instruction="You help users execute and test Python code safely.",
    tools=[mcp_tools[6]]  # Just Code Executor
)
