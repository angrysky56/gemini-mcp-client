#!/usr/bin/env python3
"""
Simple MCP Test Agent for ADK Web Interface
Start with basic functionality to test the setup.
"""

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool import MCPToolset, StdioServerParameters

# Test with just one MCP server first - SQLite
root_agent = LlmAgent(
    model='gemini-2.0-flash',
    name='sqlite_test_agent',
    instruction="""You are a database assistant with access to an SQLite database.
    
You can help users:
- Query the algorithm platform database
- Explore database structure and contents
- Analyze data and generate insights
- Perform database operations

Be helpful and explain what you're doing when you use database tools.
""",
    description="Database assistant using SQLite MCP server",
    tools=[
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
            # Optional: Filter which tools from the MCP server are exposed
            # tool_filter=['read_query', 'write_query', 'create_table', 'list_tables', 'describe_table']
        )
    ],
)
