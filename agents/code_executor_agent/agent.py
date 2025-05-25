#!/usr/bin/env python3
"""
Code Executor Agent for ADK Web Interface
Provides safe Python code execution capabilities.
"""

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool import MCPToolset
from google.adk.tools.mcp_tool.mcp_toolset import StdioServerParameters

# Code Executor Agent
agent = LlmAgent(
    model='gemini-1.5-flash-8b',  # Lightweight model with good rate limits
    name='code_executor_agent',
    instruction="""You are a specialized coding assistant with access to a safe Python code execution environment.

You can help users:
- Execute Python code safely in isolated environments
- Test code snippets and debug issues
- Prototype and experiment with algorithms
- Analyze data with Python libraries
- Demonstrate programming concepts
- Validate code functionality

You excel at:
- Writing clean, well-documented Python code
- Debugging and troubleshooting code issues
- Explaining programming concepts with working examples
- Testing code thoroughly before providing solutions
- Using appropriate Python libraries and best practices

Always explain your code, run tests to verify functionality, and provide educational value in your programming assistance.
Safety note: Code runs in an isolated environment with storage in /home/ty/Repositories/ai_workspace/python_coding_storage/
""",
    description="Python code execution assistant via MCP",
    tools=[
        MCPToolset(
            connection_params=StdioServerParameters(
                command='node',
                args=['/home/ty/Repositories/mcp_code_executor/build/index.js'],
                env={
                    'CODE_STORAGE_DIR': '/home/ty/Repositories/ai_workspace/python_coding_storage/',
                    'CONDA_ENV_NAME': 'mcp_code_executor_env'
                }
            ),
        )
    ],
)
