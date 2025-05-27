"""
Gemini-powered MCP (Model Context Protocol) Client

This module provides a client for connecting to MCP servers and interacting with them
using Google's Gemini AI model for intelligent tool usage and conversation handling.
"""

import logging
import os
import pprint
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import models, types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import AnyUrl

client = genai.Client(
    api_key='GEMINI_API_KEY',
    http_options=types.HttpOptions(api_version='v1alpha')
)
# Try to import different Gemini packages
GEMINI_AVAILABLE = False
GEMINI_PACKAGE: str | None = None

# Try gemini-tool-agent first
try:
    GEMINI_AVAILABLE = True
    GEMINI_PACKAGE = "full_mcp_agent"
except ImportError:
    pass

# Try google-generativeai as fallback
if not GEMINI_AVAILABLE:
    try:
        GEMINI_AVAILABLE = True
        GEMINI_PACKAGE = "google-generativeai"
    except ImportError:
        pass

# Try the new google-genai SDK
if not GEMINI_AVAILABLE:
    try:
        from google import genai as new_genai
        GEMINI_AVAILABLE = True
        GEMINI_PACKAGE = "google-genai"
    except ImportError:
        pass

# Load environment variables
load_dotenv()


class MCPClientError(Exception):
    """Base exception for MCP Client errors."""
    pass


class ServerConnectionError(MCPClientError):
    """Raised when connection to MCP server fails."""
    pass


class ToolExecutionError(MCPClientError):
    """Raised when tool execution fails."""
    pass


class Geminiagent:
    """Wrapper for different Gemini implementations."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        self.api_key = api_key
        self.model = model
        self.tools: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        self.agent: Any | None = None
        self.client: Any | None = None

        if GEMINI_PACKAGE == "full_mcp_agent":
                from agents.full_mcp_agent import agent
                self.agent = agent
        elif GEMINI_PACKAGE == "google-genai":
            self.client = new_genai.Client(api_key=api_key)
        else:
            raise RuntimeError("No Gemini package available")

    def process_query(self, query: str) -> dict[str, Any]:
        """Process a query and determine if tools are needed."""
        if GEMINI_PACKAGE == "gemini-tool-agent" and self.agent:
            return self.agent.process_query(query)
        else:
            # Basic implementation for other packages
            # Check if query might need tools
            tool_keywords = ["use", "call", "execute", "run", "apply"]
            needs_tool = any(keyword in query.lower() for keyword in tool_keywords) and self.tools

            if needs_tool:
                # Try to identify which tool to use
                for tool in self.tools:
                    if tool["name"].lower() in query.lower():
                        return {
                            "needs_tool": True,
                            "tool_name": tool["name"]
                        }

                # If no specific tool mentioned, use first available tool
                if self.tools:
                    return {
                        "needs_tool": True,
                        "tool_name": self.tools[0]["name"]
                    }

            return {"needs_direct_response": True, "direct_response": None}

    def process_use_tool(self, tool_name: str) -> dict[str, Any]:
        """Process tool usage request."""
        if GEMINI_PACKAGE == "full_mcp_agent" and self.agent:
            return self.agent.process_use_tool(tool_name)
        else:
            # Basic implementation
            tool = next((t for t in self.tools if t["name"] == tool_name), None)
            if tool:
                # Extract required parameters from tool schema
                params: dict[str, Any] = {}
                if "input_schema" in tool and "properties" in tool["input_schema"]:
                    for param_name, param_info in tool["input_schema"]["properties"].items():
                        # Use example or default values
                        if "example" in param_info:
                            params[param_name] = param_info["example"]
                        elif param_info.get("type") == "string":
                            params[param_name] = "sample_value"
                        elif param_info.get("type") == "integer":
                            params[param_name] = 42
                        elif param_info.get("type") == "boolean":
                            params[param_name] = True

                return {
                    "tool_name": tool_name,
                    "input": params
                }

            return {"tool_name": tool_name, "input": {}}

    def generate_response(self, prompt: str) -> str:
        """Generate a response using Gemini."""
        try:
            if GEMINI_PACKAGE == "gemini-tool-agent" and self.agent:
                return self.agent.generate_response(prompt)
            elif GEMINI_PACKAGE == "google-generativeai" and self.client:
                response = self.client.generate_content(prompt)
                return response.text
            elif GEMINI_PACKAGE == "google-genai" and self.client:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt
                )
                return response.text
            else:
                return "Unable to generate response - no Gemini package available"
        except Exception as e:
            return f"Error generating response: {str(e)}"


class MCPClient:
    """
    MCP Client that connects to MCP servers and provides intelligent conversation
    using Google's Gemini AI model.

    This client handles:
    - Connection management to MCP servers
    - Tool discovery and execution
    - AI-powered conversation handling
    - Proper resource cleanup
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
        log_level: str = "INFO"
    ) -> None:
        """
        Initialize the MCP Client.

        Args:
            api_key: Gemini API key. If None, will try to load from environment.
            model: Gemini model to use (e.g., "gemini-2.0-flash-lite", "gemini-2.5-pro-preview-05-06")
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        # Set up logging
        self._setup_logging(log_level)
        self.logger = logging.getLogger(__name__)

        # Initialize connection state
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self._connected = False
        self._server_info: dict[str, Any] = {}
        self.model = model
        self.api_key: str | None = None
        self.agent: Geminiagent | None = None

        # Initialize Gemini agent if available
        if GEMINI_AVAILABLE:
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                self.logger.warning(
                    "No Gemini API key provided. AI features will be limited."
                )
                self.agent = None
            else:
                try:
                    self.agent = Geminiagent(self.api_key, model)
                    self.logger.info(f"Gemini agent initialized with model: {model} using {GEMINI_PACKAGE}")
                except Exception as e:
                    self.logger.error(f"Failed to initialize Gemini agent: {e}")
                    self.agent = None
        else:
            self.logger.warning(
                "No Gemini package available. Install google-generativeai, google-genai, or gemini-tool-agent for AI features."
            )
            self.agent = None
    def _setup_logging(self, log_level: str) -> None:
        """Set up logging configuration."""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("mcp_client.log")
            ]
        )

    def set_model(self, model: str) -> None:
        """
        Change the Gemini model being used.

        Args:
            model: Model name (e.g., "gemini-2.0-flash", "gemini-1.5-pro", "gemini-2.5-pro-preview-03-25")
        """
        self.model = model
        if self.agent:
            self.agent.model = model
            self.logger.info(f"Model changed to: {model}")

            # Reinitialize agent with new model if needed
            if GEMINI_PACKAGE == "google-generativeai" and self.agent.client:
                import google.genai as genai
                self.agent.client = genai

    def get_available_models(self) -> list[str]:
        """
        Get list of available Gemini models.
        First tries to fetch from API, falls back to hardcoded list.

        Returns:
            list of available model names
        """
        try:
            # Try to get models dynamically from the API
            if GEMINI_PACKAGE == "google-generativeai" and self.api_key:
                import google.generativeai as genai

                model_names = []
                for model in genai.models:
                    pprint.pprint(models)

                    # Extract model name from the full name (e.g. "models/gemini-2.0-flash" -> "gemini-2.0-flash")
                    model_name = model.name.split('/')[-1] if '/' in model.name else model.name
                    # Only include generative models that support generateContent
                    if hasattr(model, 'supported_generation_methods') and 'generateContent' in model.supported_generation_methods:
                        model_names.append(model_name)

                if model_names:
                    return sorted(model_names)
        except Exception as e:
            self.logger.warning(f"Could not fetch models dynamically: {e}")

        # Fallback to hardcoded list
        return [
            "gemini-2.0-flash",
            "gemini-2.5-pro-preview-05-06",
            "gemini-2.5-flash-preview-05-20",
            "gemini-2.5-flash-preview-04-17",
        ]

    async def connect_to_server(self, server_script_path: str) -> None:
        """
        Connect to an MCP server.

        Args:
            server_script_path: Path to the server script (.py or .js)

        Raises:
            ServerConnectionError: If connection fails
            ValueError: If server script is not supported
        """
        try:
            self.logger.info(f"Connecting to server: {server_script_path}")

            # Validate server script
            script_path = Path(server_script_path)
            if not script_path.exists():
                raise ValueError(f"Server script not found: {server_script_path}")

            is_python = script_path.suffix == '.py'
            is_js = script_path.suffix == '.js'

            if not (is_python or is_js):
                raise ValueError(
                    f"Unsupported server script type: {script_path.suffix}. "
                    "Only .py and .js files are supported."
                )

            # Determine command
            command = "python" if is_python else "node"

            # Create server connection
            server_params = StdioServerParameters(
                command=command,
                args=[str(script_path)],
                env=None,
            )

            # Establish connection using stdio_client with proper context management
            @asynccontextmanager
            async def stdio_client_context():
                async with stdio_client(server_params) as client:
                    yield client

            read_stream, write_stream = await self.exit_stack.enter_async_context(
                stdio_client_context()
            )

            # Create session
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            # Initialize session
            await self.session.initialize()
            self._connected = True

            # Discover tools and update agent
            await self._discover_tools()

            self.logger.info("Successfully connected to MCP server")

        except Exception as e:
            self.logger.error(f"Failed to connect to server: {e}")
            raise ServerConnectionError(f"Connection failed: {e}") from e

    async def _discover_tools(self) -> None:
        """Discover available tools from the server and configure the agent."""
        if not self.session:
            return

        try:
            # list available tools
            tools_response = await self.session.list_tools()
            tools = []

            for tool in tools_response.tools:
                tool_info = {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                }
                tools.append(tool_info)

            self._server_info["tools"] = tools

            # Update agent with tools
            if self.agent:
                self.agent.tools = tools
                self.logger.info(f"agent updated with {len(tools)} tools")

            # list available resources
            try:
                resources_response = await self.session.list_resources()
                resources = [
                    {
                        "uri": resource.uri,
                        "name": resource.name,
                        "description": resource.description,
                        "mimeType": resource.mimeType
                    }
                    for resource in resources_response.resources
                ]
                self._server_info["resources"] = resources
                self.logger.info(f"Discovered {len(resources)} resources")
            except Exception as e:
                self.logger.warning(f"Could not list resources: {e}")
                self._server_info["resources"] = []

            # list available prompts
            try:
                prompts_response = await self.session.list_prompts()
                prompts = [
                    {
                        "name": prompt.name,
                        "description": prompt.description,
                        "arguments": [
                            {
                                "name": arg.name,
                                "description": arg.description,
                                "required": arg.required
                            }
                            for arg in (prompt.arguments or [])
                        ]
                    }
                    for prompt in prompts_response.prompts
                ]
                self._server_info["prompts"] = prompts
                self.logger.info(f"Discovered {len(prompts)} prompts")
            except Exception as e:
                self.logger.warning(f"Could not list prompts: {e}")
                self._server_info["prompts"] = []

            tool_names = [tool["name"] for tool in tools]
            self.logger.info(f"Connected to server with tools: {tool_names}")

        except Exception as e:
            self.logger.error(f"Failed to discover tools: {e}")
            raise ServerConnectionError(f"Tool discovery failed: {e}") from e

    async def get_response(self, user_input: str) -> str:
        """
        Get a response to user input, potentially using tools.

        Args:
            user_input: The user's input/question

        Returns:
            AI-generated response

        Raises:
            MCPClientError: If response generation fails
        """
        if not self._connected or not self.session:
            raise MCPClientError("Not connected to server")

        try:
            self.logger.debug(f"Processing user input: {user_input}")

            # If no agent available, provide basic response
            if not self.agent:
                return await self._basic_response(user_input)

            # Process query with agent
            response = self.agent.process_query(user_input)
            self.agent.history.append({"role": "user", "content": user_input})

            # Handle tool usage
            if isinstance(response, dict) and response.get("needs_tool", False):
                return await self._handle_tool_usage(response, user_input)

            # Handle direct response
            if isinstance(response, dict) and response.get("needs_direct_response", False):
                if response.get("direct_response"):
                    direct_response = response["direct_response"]
                    self.agent.history.append({"role": "direct_response", "content": direct_response})
                    return direct_response
                else:
                    # Generate contextual response
                    return await self._generate_contextual_response(user_input)

            # Generate contextual response
            return await self._generate_contextual_response(user_input)

        except Exception as e:
            self.logger.error(f"Error processing response: {e}")
            return f"An error occurred while processing your request: {str(e)}"

    async def _handle_tool_usage(self, response: dict[str, Any], user_input: str) -> str:
        """Handle tool usage workflow."""
        tool_name = response.get("tool_name")
        if not tool_name:
            return "Tool usage requested but no tool name provided."

        try:
            if not self.agent or not self.session:
                return "agent or session not available for tool usage."

            # Get tool usage context
            tool_response = self.agent.process_use_tool(tool_name)
            self.agent.history.append({"role": "assistant", "content": tool_response})

            # Get tool call parameters
            tool = tool_response["tool_name"]
            call_tool = self.agent.process_use_tool(tool)
            self.agent.history.append({"role": "process_tool_call", "content": call_tool})

            # Execute tool
            result = await self.session.call_tool(tool, call_tool["input"])
            self.agent.history.append({"role": "tool_call_result", "content": result})

            # Generate response based on tool result
            context = self.agent.history[-3:]  # Last 3 interactions
            response_text = self.agent.generate_response(f"""
            Based on the tool execution result, provide a helpful response to: {user_input}

            Tool Result: {result}
            Context: {context}
            """)

            self.agent.history.append({"role": "assistant", "content": response_text})
            return response_text

        except Exception as e:
            self.logger.error(f"Tool execution failed: {e}")
            raise ToolExecutionError(f"Failed to execute tool {tool_name}: {e}") from e

    async def _generate_contextual_response(self, user_input: str) -> str:
        """Generate a contextual response using conversation history."""
        if not self.agent:
            return await self._basic_response(user_input)

        conversation_context = (
            self.agent.history[-5:] if len(self.agent.history) >= 5
            else self.agent.history
        )

        response_text = self.agent.generate_response(f"""
        You are a helpful assistant responding to the following query:
        QUERY: {user_input}

        CONVERSATION HISTORY: {conversation_context}

        Available Tools: {[tool['name'] for tool in self._server_info.get('tools', [])]}
        Available Resources: {[res['name'] for res in self._server_info.get('resources', [])]}

        Please provide a comprehensive and accurate response that considers the conversation history.
        """)

        self.agent.history.append({"role": "assistant", "content": response_text})
        return response_text

    async def _basic_response(self, user_input: str) -> str:
        """Provide basic response when AI agent is not available."""
        if "tools" in user_input.lower():
            tools = self._server_info.get("tools", [])
            if tools:
                tool_list = "\n".join([f"- {tool['name']}: {tool['description']}" for tool in tools])
                return f"Available tools:\n{tool_list}"
            else:
                return "No tools are currently available."

        if "model" in user_input.lower():
            return f"Current model: {self.model}\nAvailable models: {', '.join(self.get_available_models())}"

        if "help" in user_input.lower():
            return (
                "Available commands:\n"
                "- Ask about 'tools' to see available tools\n"
                "- Ask about 'model' to see current model and available models\n"
                "- Ask about 'resources' to see available resources\n"
                "- Ask about 'prompts' to see available prompts\n"
                "- Type 'exit' to quit"
            )

        return "I'm a basic MCP client. Ask about tools, resources, prompts, or model to get started."

    async def chat_loop(self) -> None:
        """Start an interactive chat session."""
        if not self._connected:
            raise MCPClientError("Not connected to server")

        print("🚀 MCP Client Chat Session Started")
        print(f"🤖 Using model: {self.model}")
        print("Type 'exit' to quit, 'help' for commands, or 'model <name>' to change model\n")

        while True:
            try:
                user_input = input("\n💬 You: ").strip()

                if user_input.lower() in ['exit', 'quit']:
                    print("👋 Ending chat session...")
                    break

                # Handle model change command
                if user_input.lower().startswith('model '):
                    new_model = user_input[6:].strip()
                    if new_model in self.get_available_models():
                        self.set_model(new_model)
                        print(f"✅ Model changed to: {new_model}")
                    else:
                        print(f"❌ Unknown model: {new_model}")
                        print(f"Available models: {', '.join(self.get_available_models())}")
                    continue

                if not user_input:
                    continue

                print("🤖 Assistant: ", end="")
                response = await self.get_response(user_input)
                print(response)

            except KeyboardInterrupt:
                print("\n👋 Chat session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                self.logger.error(f"Chat loop error: {e}")

    async def get_server_info(self) -> dict[str, Any]:
        """Get information about the connected server."""
        if not self._connected:
            raise MCPClientError("Not connected to server")

        return {
            "connected": self._connected,
            "model": self.model,
            "available_models": self.get_available_models(),
            "gemini_package": GEMINI_PACKAGE,
            "tools": self._server_info.get("tools", []),
            "resources": self._server_info.get("resources", []),
            "prompts": self._server_info.get("prompts", []),
            "agent_available": self.agent is not None
        }

    async def call_tool_directly(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        Call a tool directly with provided arguments.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result
        """
        if not self.session:
            raise MCPClientError("Not connected to server")

        try:
            result = await self.session.call_tool(tool_name, arguments)
            self.logger.info(f"Tool {tool_name} executed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Failed to call tool {tool_name}: {e}")
            raise ToolExecutionError(f"Tool call failed: {e}") from e

    async def read_resource(self, uri: AnyUrl) -> tuple[str, str | None]:
        """
        Read a resource from the server.

        Args:
            uri: Resource URI to read

        Returns:
            Tuple of (content, mime_type)
        """
        if not self.session:
            raise MCPClientError("Not connected to server")

        try:
            result = await self.session.read_resource(uri)
            # Convert contents to string if it's a list of text contents
            content = str(result.contents[0]) if result.contents else ""
            mime_type = getattr(result, 'mimetype', None)
            return content, mime_type
        except Exception as e:
            self.logger.error(f"Failed to read resource {uri}: {e}")
            raise MCPClientError(f"Resource read failed: {e}") from e

    async def close(self) -> None:
        """Close the client and clean up resources."""
        try:
            self.logger.info("Closing MCP client...")
            await self.exit_stack.aclose()
            self._connected = False
            self.session = None
            self.logger.info("MCP client closed successfully")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def __repr__(self) -> str:
        """String representation of the client."""
        return f"MCPClient(connected={self._connected}, model={self.model}, agent_available={self.agent is not None})"
