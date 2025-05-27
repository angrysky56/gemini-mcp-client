#!/usr/bin/env python3
"""
ArXiv Research Agent for ADK Web Interface
Provides academic paper search and analysis capabilities.
"""

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool import MCPToolset
from google.adk.tools.mcp_tool.mcp_toolset import StdioServerParameters

# ArXiv Research Agent
agent = LlmAgent(
    model='gemini-1.5-flash-8b',  # Lightweight model with good rate limits
    name='arxiv_research_agent',
    instruction="""You are a specialized research assistant with access to ArXiv academic papers.

You can help users:
- Search for academic papers by topic, author, or keywords
- Download and analyze research papers
- Extract key insights and summaries from papers
- Track research trends and developments
- Find related work and citations
- Organize research collections

You excel at:
- Understanding academic research contexts
- Identifying relevant papers for specific research topics
- Summarizing complex academic content
- Connecting ideas across different papers
- Helping with literature reviews and research planning

Always provide clear explanations of research concepts and help users navigate the academic literature effectively.
""",
    description="Research assistant with ArXiv access via MCP",
    tools=[
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
        )
    ],
)
