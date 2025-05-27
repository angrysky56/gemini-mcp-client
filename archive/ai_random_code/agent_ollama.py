#!/usr/bin/env python3
"""
Multi-MCP Agent for ADK Web Interface - LOCAL OLLAMA VERSION
This agent connects to multiple MCP servers using local Ollama models.
"""


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
                'CODE_STORAGE_DIR': '/home/ty/Repositories/ai_workspace/python_coding_storage',
                'CONDA_ENV_NAME': 'mcp_code_executor_env'
            }
        ),
        tool_filter=None,
    )
)

# Define the root agent with all MCP tools - using LOCAL OLLAMA
agent = LlmAgent(
    model='ollama:qwen3:14b',  # Using your local Ollama model
    name='multi_mcp_assistant',
    instruction="""You are a comprehensive AI assistant with access to multiple powerful tools:

 **SQLite Database**: Query and manage algorithm platform database
 **Docker**: Manage containers, images, and deployments
 **Desktop Commander**: Control desktop applications and system operations
 **ArXiv Research**: Search, download, and analyze academic papers
 **Package Versions**: Check latest versions of software packages across ecosystems
 **Code Executor**: Execute Python code safely in isolated environments

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
    description="Multi-MCP assistant with comprehensive tooling capabilities using local Ollama",
    tools=mcp_tools,
)

# Alternative single-server testing agents for debugging - all using local Ollama
sqlite_agent = LlmAgent(
    model='ollama:qwen3:4b',  # Lighter model for simpler tasks
    name='sqlite_test_agent',
    instruction="You help users query and manage SQLite databases.",
    tools=[mcp_tools[0]]  # Just SQLite
)

docker_agent = LlmAgent(
    model='ollama:qwen3:4b',
    name='docker_test_agent',
    instruction="You help users manage Docker containers and images.",
    tools=[mcp_tools[1]]  # Just Docker
)

arxiv_agent = LlmAgent(
    model='ollama:qwen3:4b',
    name='arxiv_test_agent',
    instruction="You help users search and analyze academic papers from ArXiv.",
    tools=[mcp_tools[3]]  # Just ArXiv (fixed index)
)

code_executor_agent = LlmAgent(
    model='ollama:qwen3:4b',
    name='code_executor_test_agent',
    instruction="You help users execute and test Python code safely.",
    tools=[mcp_tools[5]]  # Just Code Executor (fixed index)
)
