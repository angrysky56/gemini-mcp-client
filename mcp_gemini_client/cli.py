"""
Command Line Interface for Gemini MCP Client
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

from .client import MCPClient
from .config import get_config_manager, ServerConfig, ServerType


async def chat_command(args: argparse.Namespace) -> None:
    """Handle the chat command."""
    config_manager = get_config_manager()
    
    # Use server from config if specified by name, otherwise use path
    if args.server_path in config_manager.list_servers():
        server_config = config_manager.get_server(args.server_path)
        if not server_config.enabled:
            print(f"⚠️  Server '{args.server_path}' is disabled. Enable it first with:")
            print(f"   gemini-mcp-client servers enable {args.server_path}")
            return
        
        print(f"🔌 Using configured server: {server_config.name}")
        print(f"   Description: {server_config.description}")
        
        # For configured servers, we need to handle the connection differently
        # This is a simplified version - in practice, you'd implement full server launching
        if server_config.server_type == ServerType.PYTHON and server_config.args:
            server_path = server_config.args[0]  # Assume first arg is the script path
        else:
            print(f"❌ Cannot directly connect to server type: {server_config.server_type}")
            print("   This server requires external process management")
            return
    else:
        server_path = args.server_path
    
    client = MCPClient(
        model=args.model or config_manager.client_config.default_model,
        log_level=args.log_level or config_manager.client_config.log_level
    )
    
    try:
        await client.connect_to_server(server_path)
        await client.chat_loop()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        await client.close()


async def info_command(args: argparse.Namespace) -> None:
    """Handle the info command."""
    config_manager = get_config_manager()
    
    # Use server from config if specified by name
    if args.server_path in config_manager.list_servers():
        server_config = config_manager.get_server(args.server_path)
        if not server_config.enabled:
            print(f"⚠️  Server '{args.server_path}' is disabled")
            return
        
        if server_config.server_type == ServerType.PYTHON and server_config.args:
            server_path = server_config.args[0]
        else:
            print(f"❌ Cannot directly connect to server type: {server_config.server_type}")
            return
    else:
        server_path = args.server_path
    
    client = MCPClient(
        model=args.model or config_manager.client_config.default_model,
        log_level=args.log_level or config_manager.client_config.log_level
    )
    
    try:
        await client.connect_to_server(server_path)
        server_info = await client.get_server_info()
        
        print("📋 Server Information")
        print("=" * 50)
        print(f"Connected: {server_info['connected']}")
        print(f"AI Agent Available: {server_info['agent_available']}")
        print(f"Current Model: {server_info['model']}")
        print(f"Gemini Package: {server_info['gemini_package']}")
        
        print(f"\n🤖 Available Models:")
        for model in server_info['available_models']:
            marker = "→" if model == server_info['model'] else " "
            print(f"  {marker} {model}")
        
        tools = server_info.get('tools', [])
        print(f"\n🛠️  Tools ({len(tools)}):")
        for tool in tools:
            print(f"  • {tool['name']}: {tool['description']}")
        
        resources = server_info.get('resources', [])
        print(f"\n📁 Resources ({len(resources)}):")
        for resource in resources:
            print(f"  • {resource['name']}: {resource['description']}")
        
        prompts = server_info.get('prompts', [])
        print(f"\n💬 Prompts ({len(prompts)}):")
        for prompt in prompts:
            print(f"  • {prompt['name']}: {prompt['description']}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        await client.close()


async def models_command(args: argparse.Namespace) -> None:
    """Handle the models command."""
    config_manager = get_config_manager()
    client = MCPClient(log_level=args.log_level or config_manager.client_config.log_level)
    
    print("🤖 Available Gemini Models:")
    print("=" * 40)
    
    models = client.get_available_models()
    current_default = config_manager.client_config.default_model
    
    for i, model in enumerate(models, 1):
        marker = "⭐" if "2.0-flash" in model else "🚀" if "2.5" in model else "📱"
        default_marker = " (default)" if model == current_default else ""
        print(f"{i:2d}. {marker} {model}{default_marker}")
    
    print("\n💡 Tips:")
    print("- Use 'gemini-2.0-flash' for fast responses")
    print("- Use 'gemini-2.5-pro-preview-03-25' for complex tasks")
    print("- Set default model with: gemini-mcp-client config set default_model <model>")


def servers_command(args: argparse.Namespace) -> None:
    """Handle the servers command."""
    config_manager = get_config_manager()
    
    if args.servers_action == "list":
        servers = config_manager.list_servers()
        if not servers:
            print("📭 No servers configured")
            return
        
        print("📚 Configured MCP Servers:")
        print("=" * 50)
        
        for name, server in servers.items():
            status = "✅" if server.enabled else "❌"
            tags = ", ".join(server.tags) if server.tags else "no tags"
            
            print(f"{status} {name}")
            print(f"   Description: {server.description}")
            print(f"   Type: {server.server_type.value}")
            print(f"   Command: {server.command} {' '.join(server.args)}")
            print(f"   Tags: {tags}")
            print()
    
    elif args.servers_action == "add":
        # Interactive server addition
        print("➕ Add New MCP Server")
        print("=" * 30)
        
        name = input("Server name: ").strip()
        if not name:
            print("❌ Server name is required")
            return
        
        if name in config_manager.list_servers():
            print(f"❌ Server '{name}' already exists")
            return
        
        description = input("Description: ").strip() or f"MCP Server: {name}"
        
        print("\nServer types:")
        print("1. Python (.py script)")
        print("2. JavaScript/Node.js")
        print("3. Executable/Other")
        
        try:
            type_choice = int(input("Select type (1-3): ").strip())
            type_map = {1: ServerType.PYTHON, 2: ServerType.JAVASCRIPT, 3: ServerType.EXECUTABLE}
            server_type = type_map.get(type_choice, ServerType.PYTHON)
        except ValueError:
            server_type = ServerType.PYTHON
        
        if server_type == ServerType.PYTHON:
            command = input("Command (default: python): ").strip() or "python"
            script_path = input("Script path: ").strip()
            if not script_path:
                print("❌ Script path is required")
                return
            args = [script_path]
        elif server_type == ServerType.JAVASCRIPT:
            command = input("Command (default: npx): ").strip() or "npx"
            package = input("Package name: ").strip()
            if not package:
                print("❌ Package name is required")
                return
            args = [package]
        else:
            command = input("Command: ").strip()
            if not command:
                print("❌ Command is required")
                return
            args_input = input("Arguments (space-separated): ").strip()
            args = args_input.split() if args_input else []
        
        tags_input = input("Tags (comma-separated, optional): ").strip()
        tags = [tag.strip() for tag in tags_input.split(",")] if tags_input else None
        
        server_config = ServerConfig(
            name=name,
            description=description,
            server_type=server_type,
            command=command,
            args=args,
            tags=tags
        )
        
        config_manager.add_server(server_config)
        print(f"✅ Added server: {name}")
    
    elif args.servers_action == "remove":
        if not args.server_name:
            print("❌ Server name is required for remove action")
            return
        
        if config_manager.remove_server(args.server_name):
            print(f"✅ Removed server: {args.server_name}")
        else:
            print(f"❌ Server not found: {args.server_name}")
    
    elif args.servers_action == "enable":
        if not args.server_name:
            print("❌ Server name is required for enable action")
            return
        
        if config_manager.enable_server(args.server_name):
            print(f"✅ Enabled server: {args.server_name}")
        else:
            print(f"❌ Server not found: {args.server_name}")
    
    elif args.servers_action == "disable":
        if not args.server_name:
            print("❌ Server name is required for disable action")
            return
        
        if config_manager.disable_server(args.server_name):
            print(f"✅ Disabled server: {args.server_name}")
        else:
            print(f"❌ Server not found: {args.server_name}")
    
    elif args.servers_action == "export":
        output_file = args.output_file or "claude_desktop_config.json"
        claude_config = config_manager.export_claude_desktop_config(output_file)
        
        print(f"📤 Exported Claude Desktop configuration to: {output_file}")
        print(f"   Exported {len(claude_config['mcpServers'])} enabled servers")
    
    elif args.servers_action == "import":
        if not args.config_file:
            print("❌ Configuration file is required for import action")
            return
        
        try:
            config_manager.import_claude_desktop_config(args.config_file)
            print(f"📥 Imported configuration from: {args.config_file}")
        except Exception as e:
            print(f"❌ Import failed: {e}")


def config_command(args: argparse.Namespace) -> None:
    """Handle the config command."""
    config_manager = get_config_manager()
    
    if args.config_action == "show":
        print("⚙️  Client Configuration:")
        print("=" * 30)
        print(f"Default Model: {config_manager.client_config.default_model}")
        print(f"Log Level: {config_manager.client_config.log_level}")
        print(f"API Key: {'Set' if config_manager.client_config.api_key else 'Not set'}")
        print(f"Connection Timeout: {config_manager.client_config.connection_timeout}s")
        print(f"Max Retries: {config_manager.client_config.max_retries}")
        print(f"Retry Delay: {config_manager.client_config.retry_delay}s")
    
    elif args.config_action == "set":
        if not args.key or args.value is None:
            print("❌ Both key and value are required for set action")
            return
        
        try:
            # Convert value to appropriate type
            if args.key in ["connection_timeout", "retry_delay"]:
                value = float(args.value)
            elif args.key in ["max_retries"]:
                value = int(args.value)
            else:
                value = args.value
            
            config_manager.update_client_config(**{args.key: value})
            print(f"✅ Set {args.key} = {value}")
        
        except Exception as e:
            print(f"❌ Failed to set configuration: {e}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Gemini-powered MCP Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  gemini-mcp-client chat server.py
  gemini-mcp-client chat echo-server  # Use configured server
  
  # Model selection
  gemini-mcp-client chat server.py --model gemini-2.5-pro-preview-03-25
  
  # Server management
  gemini-mcp-client servers list
  gemini-mcp-client servers add
  gemini-mcp-client servers enable my-server
  
  # Export for Claude Desktop
  gemini-mcp-client servers export claude_config.json
  
  # Configuration
  gemini-mcp-client config show
  gemini-mcp-client config set default_model gemini-2.5-pro-preview-03-25
        """
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Set logging level"
    )
    
    parser.add_argument(
        "--model",
        default=None,
        help="Gemini model to use (overrides config default)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat")
    chat_parser.add_argument(
        "server_path",
        help="Path to MCP server script or configured server name"
    )
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get server information")
    info_parser.add_argument(
        "server_path",
        help="Path to MCP server script or configured server name"
    )
    
    # Models command
    models_parser = subparsers.add_parser("models", help="List available models")
    
    # Servers command
    servers_parser = subparsers.add_parser("servers", help="Manage MCP servers")
    servers_subparsers = servers_parser.add_subparsers(dest="servers_action", help="Server actions")
    
    # Server subcommands
    servers_subparsers.add_parser("list", help="List configured servers")
    servers_subparsers.add_parser("add", help="Add new server (interactive)")
    
    remove_parser = servers_subparsers.add_parser("remove", help="Remove server")
    remove_parser.add_argument("server_name", help="Name of server to remove")
    
    enable_parser = servers_subparsers.add_parser("enable", help="Enable server")
    enable_parser.add_argument("server_name", help="Name of server to enable")
    
    disable_parser = servers_subparsers.add_parser("disable", help="Disable server")
    disable_parser.add_argument("server_name", help="Name of server to disable")
    
    export_parser = servers_subparsers.add_parser("export", help="Export Claude Desktop config")
    export_parser.add_argument("--output-file", "-o", help="Output file name")
    
    import_parser = servers_subparsers.add_parser("import", help="Import Claude Desktop config")
    import_parser.add_argument("config_file", help="Claude Desktop config file to import")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Manage client configuration")
    config_subparsers = config_parser.add_subparsers(dest="config_action", help="Config actions")
    
    config_subparsers.add_parser("show", help="Show current configuration")
    
    set_parser = config_subparsers.add_parser("set", help="Set configuration value")
    set_parser.add_argument("key", help="Configuration key")
    set_parser.add_argument("value", help="Configuration value")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Validate server path for commands that need it
    if hasattr(args, 'server_path'):
        # Check if it's a configured server name first
        config_manager = get_config_manager()
        if args.server_path not in config_manager.list_servers():
            # If not a configured server, check if it's a valid file path
            server_path = Path(args.server_path)
            if not server_path.exists():
                print(f"❌ Error: Server script not found and no configured server named: {args.server_path}")
                print("\nConfigured servers:")
                for name in config_manager.list_servers():
                    print(f"  - {name}")
                sys.exit(1)
    
    # Run the appropriate command
    if args.command == "chat":
        asyncio.run(chat_command(args))
    elif args.command == "info":
        asyncio.run(info_command(args))
    elif args.command == "models":
        asyncio.run(models_command(args))
    elif args.command == "servers":
        servers_command(args)
    elif args.command == "config":
        config_command(args)


if __name__ == "__main__":
    main()
