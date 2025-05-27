#!/usr/bin/env python3
"""
Full MCP Agent - Dynamically loads all enabled MCP servers from config.
Properly integrated with Gemini client for optimal performance.
"""

import json
import os
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool import MCPToolset
from google.adk.tools.mcp_tool.mcp_toolset import StdioServerParameters
from google.genai import types

# ========== USER CONFIGURATION ==========
# These can be overridden by environment variables

# Model selection - override with GEMINI_MODEL env var
# Using models from your provided list for best performance
DEFAULT_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-lite-001')  # 30 rpm, 1500 rpd

# Agent configuration
AGENT_NAME = os.getenv('AGENT_NAME', 'full_mcp_agent')
VERBOSE_LOGGING = os.getenv('VERBOSE_LOGGING', 'false').lower() == 'true'

# Temperature for model responses (0.0-1.0)
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.7'))

# Maximum output tokens (adjust based on your needs)
MAX_OUTPUT_TOKENS = int(os.getenv('MAX_OUTPUT_TOKENS', '8192'))

# ========== END USER CONFIGURATION ==========


def get_config_path():
    """Get the configuration directory path."""
    config_dir = os.getenv('MCP_CONFIG_DIR')
    if config_dir:
        return Path(config_dir)
    return Path(__file__).parent.parent.parent / "config"


def load_config_file(filename):
    """Load a JSON configuration file."""
    config_path = get_config_path() / filename

    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def get_model_name():
    """
    Get the model name with fallback options.
    Priority: Environment variable > Client config > Default
    """
    # First check environment variable
    env_model = os.getenv('GEMINI_MODEL')
    if env_model:
        return env_model

    # Then check client config
    client_config = load_config_file('client.json')
    config_model = client_config.get('default_model')
    if config_model:
        return config_model

    # Use default
    return DEFAULT_MODEL


def load_mcp_servers_from_config():
    """
    Dynamically load all enabled MCP servers from servers.json config.

    Returns:
        list: List of MCPToolset instances for all enabled servers
    """
    servers_config = load_config_file('servers.json')

    if not servers_config:
        print("Warning: No servers.json found in config directory")
        return []

    mcp_tools = []
    loaded_servers = []

    for server_name, server_config in servers_config.items():
        # Skip disabled servers
        if not server_config.get('enabled', True):
            continue

        try:
            # Create StdioServerParameters from config
            params = {
                'command': server_config['command'],
                'args': server_config.get('args', []),
            }

            # Add environment variables if specified
            if server_config.get('env'):
                params['env'] = server_config['env']

            # Create MCPToolset without restrictions
            toolset = MCPToolset(
                connection_params=StdioServerParameters(**params),
                # No tool_filter means all tools are available
                tool_filter=None,
            )

            mcp_tools.append(toolset)
            loaded_servers.append(server_name)

        except Exception as e:
            print(f"Warning: Failed to load MCP server '{server_name}': {e}")

    if VERBOSE_LOGGING:
        print(f"Successfully loaded {len(loaded_servers)} MCP servers: {', '.join(loaded_servers)}")

    return mcp_tools


def create_agent_instruction():
    """
    Generate dynamic instruction based on available MCP servers.

    Returns:
        str: Generated instruction text
    """
    servers_config = load_config_file('servers.json')

    if not servers_config:
        return "You are an AI assistant. Note: No MCP servers configured."

    # Build instruction with available tools
    instruction = """You are a comprehensive AI assistant with dynamic access to multiple MCP tools.

Available MCP Servers and their capabilities:
"""

    enabled_count = 0
    for server_name, server_config in servers_config.items():
        if server_config.get('enabled', True):
            enabled_count += 1
            name = server_config.get('name', server_name)
            description = server_config.get('description', 'No description provided')
            tags = server_config.get('tags', [])

            instruction += f"\n• **{name}**: {description}"
            if tags:
                instruction += f" (Tags: {', '.join(tags)})"

    instruction += f"""

Total tools available: {enabled_count} MCP servers

Guidelines:
1. Always explain what you're doing and why
2. Choose the most appropriate tool for each task
3. Handle errors gracefully and suggest alternatives
4. Provide clear and helpful responses
5. You can use multiple tools in sequence or parallel as needed

Note: All tool functions are available without call limits through the MCP protocol.
"""

    return instruction


# Load configuration
model_name = get_model_name()
mcp_tools = load_mcp_servers_from_config()
agent_instruction = create_agent_instruction()
name = AGENT_NAME or "full_mcp_agent"
# Configure generation settings for optimal Gemini integration
generation_config = types.GenerateContentConfig(
    temperature=TEMPERATURE,
    max_output_tokens=MAX_OUTPUT_TOKENS,
    # candidate_count=1,  # Only one response needed
    # stop_sequences=[],  # No custom stop sequences
)

# Create the main agent
agent = LlmAgent(
    model=model_name,
    name=AGENT_NAME,
    instruction=agent_instruction,
    description="Full MCP agent with dynamic tool loading and Gemini optimization",
    tools=mcp_tools,
    generate_content_config=generation_config,
    # Enable all agent capabilities
    disallow_transfer_to_parent=False,
    disallow_transfer_to_peers=False,
)

# Print configuration summary
if VERBOSE_LOGGING:
    print(f"""
Full MCP Agent Configuration:
- Model: {model_name}
- Temperature: {TEMPERATURE}
- Max Output Tokens: {MAX_OUTPUT_TOKENS}
- Loaded MCP Servers: {len(mcp_tools)}
- Agent Name: {AGENT_NAME}
""")
