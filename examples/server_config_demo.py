"""
Example demonstrating MCP server configuration management
"""

import asyncio
from mcp_gemini_client import MCPClient
from mcp_gemini_client.config import get_config_manager, ServerConfig, ServerType


async def demo_server_management():
    """Demonstrate server configuration management."""

    print("🚀 MCP Server Configuration Demo")
    print("=" * 50)

    # Get configuration manager
    config_manager = get_config_manager()

    # List current servers
    print("\n📚 Current Servers:")
    servers = config_manager.list_servers()
    for name, server in servers.items():
        status = "✅" if server.enabled else "❌"
        print(f"  {status} {name}: {server.description}")

    # Add a custom server
    print("\n➕ Adding Custom Server...")
    custom_server = ServerConfig(
        name="custom-calculator",
        description="Custom calculator server",
        server_type=ServerType.PYTHON,
        command="python",
        args=["examples/calculator_server.py"],
        tags=["math", "calculator", "custom"]
    )

    config_manager.add_server(custom_server)
    print(f"   Added: {custom_server.name}")

    # Export Claude Desktop configuration
    print("\n📤 Exporting Claude Desktop Configuration...")
    claude_config = config_manager.export_claude_desktop_config("demo_claude_config.json")

    print(f"   Exported {len(claude_config['mcpServers'])} servers")
    print("   Configuration saved to: demo_claude_config.json")

    # Show the exported configuration
    print("\n📋 Claude Desktop Configuration Preview:")
    for name, server_config in claude_config["mcpServers"].items():
        print(f"  {name}:")
        print(f"    command: {server_config['command']}")
        print(f"    args: {server_config['args']}")

    # Test connecting to a configured server
    print("\n🔌 Testing Connection to Echo Server...")
    try:
        client = MCPClient()
        await client.connect_to_server("examples/echo_server.py")

        # Get server info
        info = await client.get_server_info()
        print("   Connected successfully!")
        print(f"   Tools available: {len(info['tools'])}")
        print(f"   Model: {info['model']}")

        # Test a simple interaction
        response = await client.get_response("What tools are available?")
        print(f"   Response: {response}")

        await client.close()

    except Exception as e:
        print(f"   ❌ Connection failed: {e}")


def demo_cli_usage():
    """Demonstrate CLI usage for server management."""

    print("\n💻 CLI Usage Examples:")
    print("=" * 30)

    print("# List configured servers")
    print("mcp-gemini-client servers list")

    print("\n# Add a new server interactively")
    print("mcp-gemini-client servers add")

    print("\n# Connect to a configured server by name")
    print("mcp-gemini-client chat echo-server")

    print("\n# Enable/disable servers")
    print("mcp-gemini-client servers enable my-server")
    print("mcp-gemini-client servers disable my-server")

    print("\n# Export configuration for Claude Desktop")
    print("mcp-gemini-client servers export my_claude_config.json")

    print("\n# Import from existing Claude Desktop config")
    print("mcp-gemini-client servers import ~/.config/claude-desktop/claude_desktop_config.json")

    print("\n# Set default model")
    print("mcp-gemini-client config set default_model gemini-2.5-pro-preview-03-25")

    print("\n# Show current configuration")
    print("mcp-gemini-client config show")


def create_sample_servers():
    """Create sample server configurations."""

    print("\n🔧 Creating Sample Server Configurations...")

    config_manager = get_config_manager()

    # Database server
    db_server = ServerConfig(
        name="database-server",
        description="SQLite database server with custom schema",
        server_type=ServerType.PYTHON,
        command="uv",
        args=["run", "mcp-server-sqlite", "--db-path", "/path/to/my_database.db"],
        env={"DB_READONLY": "false"},
        tags=["database", "sql", "production"]
    )

    # Web scraper server
    scraper_server = ServerConfig(
        name="web-scraper",
        description="Web scraping server with BeautifulSoup",
        server_type=ServerType.PYTHON,
        command="python",
        args=["servers/web_scraper_server.py"],
        env={"USER_AGENT": "MCP-Client/1.0"},
        tags=["web", "scraping", "data"]
    )

    # File system server
    fs_server = ServerConfig(
        name="filesystem",
        description="Filesystem access server",
        server_type=ServerType.JAVASCRIPT,
        command="npx",
        args=["@modelcontextprotocol/server-filesystem", "/allowed/directory"],
        tags=["filesystem", "files", "storage"]
    )

    # API server
    api_server = ServerConfig(
        name="rest-api",
        description="REST API integration server",
        server_type=ServerType.PYTHON,
        command="python",
        args=["servers/rest_api_server.py"],
        env={
            "API_BASE_URL": "https://api.example.com",
            "API_KEY": "${API_KEY}"  # Will be resolved from environment
        },
        tags=["api", "rest", "integration"]
    )

    # Add all servers
    servers = [db_server, scraper_server, fs_server, api_server]

    for server in servers:
        if server.name not in config_manager.list_servers():
            config_manager.add_server(server)
            print(f"   ✅ Added: {server.name}")
        else:
            print(f"   ⚠️  Already exists: {server.name}")


async def main():
    """Main demo function."""

    await demo_server_management()

    print("\n" + "="*60)

    demo_cli_usage()

    print("\n" + "="*60)

    create_sample_servers()

    print("\n🎉 Demo Complete!")
    print("\nNext steps:")
    print("1. Run 'mcp-gemini-client servers list' to see all configured servers")
    print("2. Edit config/servers.json to customize server configurations")
    print("3. Use 'mcp-gemini-client servers export' to create Claude Desktop config")
    print("4. Test connections with 'mcp-gemini-client info <server-name>'")


if __name__ == "__main__":
    asyncio.run(main())
