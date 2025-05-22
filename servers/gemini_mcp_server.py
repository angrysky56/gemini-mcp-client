#!/usr/bin/env python3
"""
MCP Server that provides Gemini AI capabilities

This server exposes Gemini AI as MCP tools and resources, allowing Claude Desktop
to interact with Gemini models through the MCP protocol.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the project root to Python path so we can import our modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mcp.server.fastmcp import FastMCP, Context
from mcp_gemini_client.client import MCPClient

# Configure logging to stderr only (never stdout which is used for MCP protocol)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr
)

# Create the MCP server
mcp = FastMCP("Gemini AI Server")

# Global Gemini client instance
gemini_client = None


async def get_gemini_client() -> MCPClient:
    """Get or create the Gemini client instance."""
    global gemini_client
    if gemini_client is None:
        from dotenv import load_dotenv
        load_dotenv()
        
        # Initialize with environment variables
        api_key = os.getenv("GEMINI_API_KEY")
        default_model = os.getenv("DEFAULT_MODEL", "gemini-2.0-flash")
        
        gemini_client = MCPClient(
            api_key=api_key,
            model=default_model,
            log_level="WARNING"  # Reduce logging
        )
    
    return gemini_client


@mcp.tool()
async def chat_with_gemini(prompt: str, model: str = "gemini-2.0-flash") -> str:
    """
    Chat with Gemini AI model.
    
    Args:
        prompt: The message to send to Gemini
        model: The Gemini model to use (default: gemini-2.0-flash)
    
    Returns:
        Gemini's response
    """
    try:
        client = await get_gemini_client()
        
        # Set the model if different from current
        if model != client.model:
            client.set_model(model)
        
        # For this tool, we'll use Gemini directly without MCP server connection
        if client.agent:
            response = client.agent.generate_response(prompt)
            return response
        else:
            return "Gemini agent not available. Please check your API key configuration."
            
    except Exception as e:
        return f"Error communicating with Gemini: {str(e)}"


@mcp.tool()
async def list_gemini_models() -> list[str]:
    """
    List available Gemini models.
    
    Returns:
        List of available Gemini model names
    """
    try:
        client = await get_gemini_client()
        return client.get_available_models()
    except Exception as e:
        return [f"Error: {str(e)}"]


@mcp.tool()
async def set_gemini_model(model: str) -> str:
    """
    Set the default Gemini model to use.
    
    Args:
        model: The model name to set as default
    
    Returns:
        Confirmation message
    """
    try:
        client = await get_gemini_client()
        available_models = client.get_available_models()
        
        if model not in available_models:
            return f"Invalid model '{model}'. Available models: {', '.join(available_models)}"
        
        client.set_model(model)
        return f"Default model set to: {model}"
        
    except Exception as e:
        return f"Error setting model: {str(e)}"


@mcp.tool()
async def gemini_model_comparison(prompt: str, models: list[str] = None) -> dict:
    """
    Compare responses from multiple Gemini models.
    
    Args:
        prompt: The prompt to send to all models
        models: List of models to compare (default: ["gemini-2.0-flash", "gemini-1.5-pro"])
    
    Returns:
        Dictionary with model names as keys and responses as values
    """
    if models is None:
        models = ["gemini-2.0-flash", "gemini-1.5-pro"]
    
    try:
        client = await get_gemini_client()
        results = {}
        
        for model in models:
            try:
                client.set_model(model)
                if client.agent:
                    response = client.agent.generate_response(prompt)
                    results[model] = response
                else:
                    results[model] = "Agent not available"
            except Exception as e:
                results[model] = f"Error: {str(e)}"
        
        return results
        
    except Exception as e:
        return {"error": f"Error in model comparison: {str(e)}"}


@mcp.resource("gemini://models")
async def list_models_resource() -> str:
    """Resource containing available Gemini models."""
    try:
        client = await get_gemini_client()
        models = client.get_available_models()
        
        model_info = []
        for model in models:
            if "2.0-flash" in model:
                description = "Fast, efficient model for most tasks"
            elif "2.5-pro" in model:
                description = "Advanced model for complex reasoning"
            elif "1.5-pro" in model:
                description = "Stable, reliable model"
            elif "1.5-flash" in model:
                description = "Lightweight, fast model"
            else:
                description = "Gemini model"
            
            model_info.append(f"• {model}: {description}")
        
        return "Available Gemini Models:\n" + "\n".join(model_info)
        
    except Exception as e:
        return f"Error listing models: {str(e)}"


@mcp.resource("gemini://status")
async def gemini_status_resource() -> str:
    """Resource containing Gemini client status."""
    try:
        client = await get_gemini_client()
        
        status_info = [
            f"Current Model: {client.model}",
            f"Agent Available: {client.agent is not None}",
            f"API Key Configured: {bool(client.api_key)}"
        ]
        
        if hasattr(client, '_server_info'):
            status_info.append(f"Connected to MCP Server: {client._connected}")
        
        return "Gemini Client Status:\n" + "\n".join(status_info)
        
    except Exception as e:
        return f"Error getting status: {str(e)}"


@mcp.prompt()
def creative_writing_prompt(topic: str, style: str = "story") -> str:
    """
    Generate a creative writing prompt using Gemini.
    
    Args:
        topic: The topic for the creative writing
        style: The style of writing (story, poem, essay, etc.)
    """
    return f"Write a {style} about {topic}. Be creative and engaging."


@mcp.prompt()
def code_review_prompt(code: str, language: str = "python") -> str:
    """
    Generate a code review prompt for Gemini.
    
    Args:
        code: The code to review
        language: Programming language of the code
    """
    return f"Please review this {language} code and provide suggestions for improvement:\n\n{code}"


if __name__ == "__main__":
    # Run the MCP server
    # This will only output MCP JSON protocol messages to stdout
    mcp.run()
