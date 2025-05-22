"""
Example usage of the Gemini MCP Client
"""

import asyncio
import os
from gemini_mcp_client import MCPClient


async def main():
    """Example usage of the MCP Client."""
    
    # Initialize client
    client = MCPClient()
    
    try:
        # Example: Connect to a simple echo server
        server_path = "examples/echo_server.py"  # Replace with actual server path
        
        print("🔌 Connecting to MCP server...")
        await client.connect_to_server(server_path)
        
        # Get server information
        print("\n📋 Server Information:")
        server_info = await client.get_server_info()
        print(f"  Tools: {len(server_info['tools'])}")
        print(f"  Resources: {len(server_info['resources'])}")
        print(f"  Prompts: {len(server_info['prompts'])}")
        
        # Example interactions
        print("\n💬 Example Interactions:")
        
        # Ask about available tools
        response1 = await client.get_response("What tools are available?")
        print(f"User: What tools are available?")
        print(f"Assistant: {response1}")
        
        # Try using a tool (if available)
        if server_info['tools']:
            tool_name = server_info['tools'][0]['name']
            response2 = await client.get_response(f"Use the {tool_name} tool to say hello")
            print(f"\nUser: Use the {tool_name} tool to say hello")
            print(f"Assistant: {response2}")
        
        # Start interactive chat (commented out for example)
        # print("\n🚀 Starting interactive chat...")
        # await client.chat_loop()
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        await client.close()
        print("\n👋 Client closed")


if __name__ == "__main__":
    asyncio.run(main())
