"""
Configuration management for MCP servers and client settings
"""

from dataclasses import asdict, dataclass
from enum import Enum
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional


class ServerType(Enum):
    """Types of MCP servers."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    EXECUTABLE = "executable"


@dataclass
@dataclass
class ServerConfig:
    """Configuration for an MCP server."""
    name: str
    description: str
    server_type: ServerType
    command: str
    args: list[str]
    env: dict[str, str] | None = None
    working_directory: str | None = None
    enabled: bool = True
    tags: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["server_type"] = self.server_type.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ServerConfig":
        """Create from dictionary."""
        data = data.copy()
        if "server_type" in data:
            data["server_type"] = ServerType(data["server_type"])
        return cls(**data)

@dataclass
class ClientConfig:
    """Configuration for the MCP client."""
    default_model: str = "gemini-2.0-flash"
    log_level: str = "INFO"
    api_key: str | None = None
    connection_timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClientConfig":
        """Create from dictionary."""
        return cls(**data)


class ConfigManager:
    """Manages configuration for MCP servers and client settings."""

    def __init__(self, config_dir: str | Path | None = None):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory to store configuration files
        """
        if config_dir is None:
            # Default to project config directory
            project_root = Path(__file__).parent.parent
            config_dir = project_root / "config"

        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        self.servers_file = self.config_dir / "servers.json"
        self.client_file = self.config_dir / "client.json"

        self.logger = logging.getLogger(__name__)

        # Load configurations
        self.servers: dict[str, ServerConfig] = {}
        self.client_config = ClientConfig()

        self.load_configurations()
    def load_configurations(self) -> None:
        """Load all configuration files."""
        self.load_servers()
        self.load_client_config()

    def load_servers(self) -> None:
        """Load server configurations from file."""
        if self.servers_file.exists():
            try:
                with open(self.servers_file) as f:
                    data = json.load(f)

                self.servers = {}
                for name, server_data in data.items():
                    try:
                        self.servers[name] = ServerConfig.from_dict(server_data)
                    except Exception as e:
                        self.logger.error(f"Failed to load server config '{name}': {e}")

                self.logger.info(f"Loaded {len(self.servers)} server configurations")

            except Exception as e:
                self.logger.error(f"Failed to load servers configuration: {e}")
                self.servers = {}
        else:
            # Create default servers file
            self._create_default_servers()

    def load_client_config(self) -> None:
        """Load client configuration from file."""
        if self.client_file.exists():
            try:
                with open(self.client_file) as f:
                    data = json.load(f)

                self.client_config = ClientConfig.from_dict(data)
                self.logger.info("Loaded client configuration")

            except Exception as e:
                self.logger.error(f"Failed to load client configuration: {e}")
                self.client_config = ClientConfig()
        else:
            # Create default client config
            self._create_default_client_config()

    def save_servers(self) -> None:
        """Save server configurations to file."""
        try:
            data = {name: config.to_dict() for name, config in self.servers.items()}

            with open(self.servers_file, 'w') as f:
                json.dump(data, f, indent=2)

            self.logger.info(f"Saved {len(self.servers)} server configurations")

        except Exception as e:
            self.logger.error(f"Failed to save servers configuration: {e}")

    def save_client_config(self) -> None:
        """Save client configuration to file."""
        try:
            with open(self.client_file, 'w') as f:
                json.dump(self.client_config.to_dict(), f, indent=2)

            self.logger.info("Saved client configuration")

        except Exception as e:
            self.logger.error(f"Failed to save client configuration: {e}")

    def add_server(self, config: ServerConfig) -> None:
        """Add a new server configuration."""
        self.servers[config.name] = config
        self.save_servers()
        self.logger.info(f"Added server configuration: {config.name}")

    def remove_server(self, name: str) -> bool:
        """Remove a server configuration."""
        if name in self.servers:
            del self.servers[name]
            self.save_servers()
            self.logger.info(f"Removed server configuration: {name}")
            return True
        return False

    def get_server(self, name: str) -> ServerConfig | None:
        """Get a server configuration by name."""
        return self.servers.get(name)
    def list_servers(self, enabled_only: bool = False) -> dict[str, ServerConfig]:
        """List all server configurations."""
        if enabled_only:
            return {
                name: config
                for name, config in self.servers.items()
                if config.enabled
            }
        return self.servers.copy()
    def enable_server(self, name: str) -> bool:
        """Enable a server."""
        if name in self.servers:
            self.servers[name].enabled = True
            self.save_servers()
            return True
        return False

    def disable_server(self, name: str) -> bool:
        """Disable a server."""
        if name in self.servers:
            self.servers[name].enabled = False
            self.save_servers()
            return True
        return False

    def update_client_config(self, **kwargs: dict[str, Any]) -> None:
        """Update client configuration."""
        for key, value in kwargs.items():
            if hasattr(self.client_config, key):
                setattr(self.client_config, key, value)

        self.save_client_config()
        self.logger.info("Updated client configuration")

    def _create_default_servers(self) -> None:
        """Create default server configurations."""
        # Echo server example
        echo_server = ServerConfig(
            name="echo-server",
            description="Simple echo server for testing",
            server_type=ServerType.PYTHON,
            command="python",
            args=["examples/echo_server.py"],
            tags=["example", "testing"]
        )

        # Memory server example
        memory_server = ServerConfig(
            name="memory-server",
            description="Memory storage server",
            server_type=ServerType.JAVASCRIPT,
            command="npx",
            args=["@modelcontextprotocol/server-memory"],
            tags=["memory", "storage"]
        )

        # SQLite server example
        sqlite_server = ServerConfig(
            name="sqlite-server",
            description="SQLite database server",
            server_type=ServerType.PYTHON,
            command="uv",
            args=["run", "mcp-server-sqlite", "--db-path", "data/database.db"],
            tags=["database", "sql"],
            enabled=False  # Disabled by default since it needs setup
        )

        self.servers = {
            "echo-server": echo_server,
            "memory-server": memory_server,
            "sqlite-server": sqlite_server
        }

        self.save_servers()
        self.logger.info("Created default server configurations")

    def _create_default_client_config(self) -> None:
        """Create default client configuration."""
        # Load from environment variables if available
        self.client_config = ClientConfig(
            default_model=os.getenv("DEFAULT_MODEL", "gemini-2.0-flash"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            api_key=os.getenv("GEMINI_API_KEY"),
            connection_timeout=float(os.getenv("CONNECTION_TIMEOUT", "30.0")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("RETRY_DELAY", "1.0"))
        )

        self.save_client_config()
        self.logger.info("Created default client configuration")

    def export_claude_desktop_config(
        self, output_file: str | None = None
    ) -> dict[str, dict[str, dict[str, str | list[str] | dict[str, str]]]]:
        """
        Export server configurations in Claude Desktop format.

        Args:
            output_file: Optional file to write the configuration to

        Returns:
            Claude Desktop configuration dictionary
        """
        claude_config: dict[
            str, dict[str, dict[str, str | list[str] | dict[str, str]]]
        ] = {"mcpServers": {}}

        # Type annotation for server entries
        claude_server: dict[str, str | list[str] | dict[str, str]]

        for name, server_config in self.servers.items():
            if not server_config.enabled:
                continue

            claude_server = {
                "command": server_config.command,
                "args": server_config.args
            }

            if server_config.env:
                claude_server["env"] = server_config.env

            claude_config["mcpServers"][name] = claude_server

        if output_file:
            output_path = Path(output_file)
            with open(output_path, 'w') as f:
                json.dump(claude_config, f, indent=2)

            self.logger.info(f"Exported Claude Desktop configuration to: {output_path}")

        return claude_config

    def import_claude_desktop_config(self, config_file: str | Path) -> None:
        """
        Import server configurations from Claude Desktop format.

        Args:
            config_file: Path to Claude Desktop configuration file
        """
        config_path = Path(config_file)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            claude_config = json.load(f)

        if "mcpServers" not in claude_config:
            raise ValueError("Invalid Claude Desktop configuration: missing 'mcpServers'")

        imported_count = 0

        for name, server_data in claude_config["mcpServers"].items():
            try:
                # Determine server type based on command
                command = server_data.get("command", "")
                if command == "python" or any(".py" in arg for arg in server_data.get("args", [])):
                    server_type = ServerType.PYTHON
                elif command in ["node", "npm", "npx"] or any(".js" in arg for arg in server_data.get("args", [])):
                    server_type = ServerType.JAVASCRIPT
                else:
                    server_type = ServerType.EXECUTABLE

                server_config = ServerConfig(
                    name=name,
                    description=f"Imported from Claude Desktop: {name}",
                    server_type=server_type,
                    command=command,
                    args=server_data.get("args", []),
                    env=server_data.get("env"),
                    tags=["imported", "claude-desktop"]
                )

                self.servers[name] = server_config
                imported_count += 1

            except Exception as e:
                self.logger.error(f"Failed to import server '{name}': {e}")

        if imported_count > 0:
            self.save_servers()
            self.logger.info(f"Imported {imported_count} server configurations from Claude Desktop")
        else:
            self.logger.warning("No servers were imported")# Global configuration manager instance
_config_manager: ConfigManager | None = None

def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def reset_config_manager() -> None:
    """Reset the global configuration manager instance."""
    global _config_manager
    _config_manager = None
