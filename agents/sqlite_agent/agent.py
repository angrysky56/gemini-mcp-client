#!/usr/bin/env python3
"""
SQLite Database Agent for ADK Web Interface
Provides database querying and management capabilities.
"""

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool import MCPToolset
from google.adk.tools.mcp_tool.mcp_toolset import StdioServerParameters

# SQLite Database Agent
agent = LlmAgent(
    model='gemini-2.0-flash-lite',  # Lightweight model with good rate limits
    name='sqlite_database_agent',
    instruction="""You are a helpful database assistant with access to an SQLite database containing algorithm platform data.

You can help users:
- Query the database to find information
- Explore database structure (tables, columns, relationships)
- Analyze data and generate insights
- Perform database operations like creating tables, inserting data
- Help with SQL query optimization and debugging

Always explain what you're doing when you use database tools and provide clear, helpful responses.
Be proactive in suggesting useful queries based on the user's needs.
""",
    description="Database assistant with SQLite access via MCP",
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
        )
    ],
)
