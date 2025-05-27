#!/usr/bin/env python3
"""
Simple JSON validator for Claude Desktop configuration
"""

import json
import sys

def validate_claude_config(config_path):
    """Validate Claude Desktop configuration file."""
    try:
        with open(config_path) as f:
            config = json.load(f)

        print("✅ JSON is valid!")

        # Check structure
        if "mcpServers" in config:
            servers = config["mcpServers"]
            print(f"📚 Found {len(servers)} MCP servers:")

            for name, server_config in servers.items():
                print(f"  • {name}")
                if "command" in server_config:
                    print(f"    Command: {server_config['command']}")
                if "args" in server_config:
                    print(f"    Args: {' '.join(server_config['args'])}")
                print()
        else:
            print("⚠️  No 'mcpServers' section found")

    except json.JSONDecodeError as e:
        print(f"❌ JSON Error: {e}")
        print(f"   Line {e.lineno}, Column {e.colno}")
        return False
    except FileNotFoundError:
        print(f"❌ File not found: {config_path}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    return True

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "~/.config/claude-desktop/claude_desktop_config.json"
    config_path = config_path.replace("~", "/home/ty")  # Expand ~ for this user

    print(f"🔍 Validating: {config_path}")
    validate_claude_config(config_path)
