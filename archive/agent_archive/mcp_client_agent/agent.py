# https://google.github.io/adk-docs/tools/mcp-tools/#what-is-model-context-protocol-mcp
# ./adk_agent_samples/mcp_client_agent/agent.py
import os
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool import MCPToolset, StdioServerParameters

# IMPORTANT: Replace this with the ABSOLUTE path to your my_adk_mcp_server.py script
 # /path/to/your/my_adk_mcp_server.py" # <<< REPLACE
PATH_TO_YOUR_MCP_SERVER_SCRIPT = "/home/ty/Repositories/ai_workspace/gemini-mcp-client/servers/gemini_mcp_server.py"

if PATH_TO_YOUR_MCP_SERVER_SCRIPT == "/path/to/your/my_adk_mcp_server.py":
    print("WARNING: PATH_TO_YOUR_MCP_SERVER_SCRIPT is not set. Please update it in agent.py.")
    # Optionally, raise an error if the path is critical

root_agent = LlmAgent(
    model='gemini-2.0-flash',
    name='web_reader_mcp_client_agent',
    instruction="Use the 'load_web_page' tool to fetch content from a URL provided by the user.",
    tools=[
        MCPToolset(
            connection_params=StdioServerParameters(
                command='python3', # Command to run your MCP server script
                args=[PATH_TO_YOUR_MCP_SERVER_SCRIPT], # Argument is the path to the script
            )
            # tool_filter=['load_web_page'] # Optional: ensure only specific tools are loaded
        )
    ],
)