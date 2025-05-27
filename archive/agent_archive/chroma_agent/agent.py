#!/usr/bin/env python3
"""
Chroma Vector Database Agent for ADK Web Interface
Provides vector database and semantic search capabilities.
"""

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool import MCPToolset
from google.adk.tools.mcp_tool.mcp_toolset import StdioServerParameters

# Chroma Vector Database Agent
agent = LlmAgent(
    model='gemini-1.5-flash-8b',  # Lightweight model with good rate limits
    name='chroma_vector_agent',
    instruction="""You are a specialized vector database assistant with access to Chroma DB.

You can help users:
- Create and manage vector collections
- Add documents and embeddings to collections
- Perform semantic search and similarity queries
- Manage document metadata and filtering
- Analyze vector data and relationships
- Help with embedding strategies and optimization

You excel at:
- Semantic document search and retrieval
- Finding similar content across large datasets
- Managing document collections with metadata
- Explaining vector database concepts and best practices

Always explain the vector operations you're performing and help users understand how semantic search works.
""",
    description="Vector database assistant with Chroma DB access via MCP",
    tools=[
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
        )
    ],
)
