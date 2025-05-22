# MCP Server Configuration Guide

## Overview

The Gemini MCP Client includes a powerful configuration system for managing MCP servers. This allows you to:

- **Store server configurations**: Define servers once and reuse them
- **Enable/disable servers**: Turn servers on or off without removing them
- **Export for Claude Desktop**: Generate configuration files for Claude Desktop
- **Import existing configs**: Import from Claude Desktop configurations
- **Model and client settings**: Configure default models and client behavior

## Configuration Files

The client stores configuration in JSON files:

- `config/servers.json` - MCP server configurations
- `config/client.json` - Client settings (model, timeouts, etc.)

## Managing Servers

### List Configured Servers

```bash
# List all servers
gemini-mcp-client servers list

# Example output:
📚 Configured MCP Servers:
==================================================
✅ echo-server
   Description: Simple echo server for testing
   Type: python
   Command: python examples/echo_server.py
   Tags: example, testing

❌ sqlite-server
   Description: SQLite database server
   Type: python
   Command: uv run mcp-server-sqlite --db-path data/database.db
   Tags: database, sql
```

### Add New Servers

#### Interactive Addition
```bash
gemini-mcp-client servers add
```

This will prompt you for:
- Server name
- Description
- Server type (Python/JavaScript/Executable)
- Command and arguments
- Tags

#### Programmatic Addition
```python
from gemini_mcp_client.config import get_config_manager, ServerConfig, ServerType

config_manager = get_config_manager()

server = ServerConfig(
    name="my-database",
    description="SQLite database with custom schema",
    server_type=ServerType.PYTHON,
    command="uv",
    args=["run", "mcp-server-sqlite", "--db-path", "/path/to/db.sqlite"],
    env={"DB_READONLY": "false"},
    tags=["database", "production"]
)

config_manager.add_server(server)
```

### Enable/Disable Servers

```bash
# Disable a server (keeps config but marks as disabled)
gemini-mcp-client servers disable sqlite-server

# Enable a server
gemini-mcp-client servers enable sqlite-server

# Remove a server completely
gemini-mcp-client servers remove old-server
```

### Using Configured Servers

Once configured, you can use servers by name instead of path:

```bash
# Connect to a configured server by name
gemini-mcp-client chat echo-server

# Get info about a configured server
gemini-mcp-client info memory-server
```

## Server Configuration Format

### Basic Server Structure

```json
{
  "server-name": {
    "name": "server-name",
    "description": "Human-readable description",
    "server_type": "python|javascript|executable",
    "command": "command-to-run",
    "args": ["arg1", "arg2"],
    "env": {
      "ENV_VAR": "value"
    },
    "working_directory": "/optional/working/dir",
    "enabled": true,
    "tags": ["tag1", "tag2"]
  }
}
```

### Server Types

#### Python Servers
```json
{
  "my-python-server": {
    "name": "my-python-server",
    "description": "Custom Python MCP server",
    "server_type": "python",
    "command": "python",
    "args": ["servers/my_server.py"],
    "enabled": true,
    "tags": ["python", "custom"]
  }
}
```

#### JavaScript/Node.js Servers
```json
{
  "memory-server": {
    "name": "memory-server",
    "description": "Memory storage server",
    "server_type": "javascript",
    "command": "npx",
    "args": ["@modelcontextprotocol/server-memory"],
    "enabled": true,
    "tags": ["memory", "javascript"]
  }
}
```

#### Executable Servers
```json
{
  "custom-binary": {
    "name": "custom-binary",
    "description": "Custom compiled MCP server",
    "server_type": "executable",
    "command": "/path/to/mcp-server-binary",
    "args": ["--config", "server.conf"],
    "enabled": true,
    "tags": ["compiled", "custom"]
  }
}
```

### Environment Variables

Servers can have custom environment variables:

```json
{
  "api-server": {
    "name": "api-server",
    "description": "REST API integration server",
    "server_type": "python",
    "command": "python",
    "args": ["servers/api_server.py"],
    "env": {
      "API_BASE_URL": "https://api.example.com",
      "API_KEY": "${API_KEY}",
      "DEBUG": "true"
    },
    "enabled": true,
    "tags": ["api", "integration"]
  }
}
```

## Claude Desktop Integration

### Export Configuration

Generate a configuration file for Claude Desktop:

```bash
# Export to default file
gemini-mcp-client servers export

# Export to specific file
gemini-mcp-client servers export my_claude_config.json
```

This creates a file compatible with Claude Desktop:

```json
{
  "mcpServers": {
    "echo-server": {
      "command": "python",
      "args": ["examples/echo_server.py"]
    },
    "memory-server": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-memory"]
    },
    "api-server": {
      "command": "python",
      "args": ["servers/api_server.py"],
      "env": {
        "API_BASE_URL": "https://api.example.com",
        "API_KEY": "${API_KEY}",
        "DEBUG": "true"
      }
    }
  }
}
```

### Import Existing Configuration

Import servers from an existing Claude Desktop configuration:

```bash
# Import from Claude Desktop config
gemini-mcp-client servers import ~/.config/claude-desktop/claude_desktop_config.json

# Import from any compatible config file
gemini-mcp-client servers import my_existing_config.json
```

## Client Configuration

### View Current Settings

```bash
gemini-mcp-client config show
```

Output:
```
⚙️  Client Configuration:
==============================
Default Model: gemini-2.0-flash
Log Level: INFO
API Key: Set
Connection Timeout: 30.0s
Max Retries: 3
Retry Delay: 1.0s
```

### Modify Settings

```bash
# Set default model
gemini-mcp-client config set default_model gemini-2.5-pro-preview-03-25

# Set log level
gemini-mcp-client config set log_level DEBUG

# Set connection timeout
gemini-mcp-client config set connection_timeout 60.0

# Set retry settings
gemini-mcp-client config set max_retries 5
gemini-mcp-client config set retry_delay 2.0
```

### Client Configuration File

The client configuration is stored in `config/client.json`:

```json
{
  "default_model": "gemini-2.0-flash",
  "log_level": "INFO",
  "api_key": null,
  "connection_timeout": 30.0,
  "max_retries": 3,
  "retry_delay": 1.0
}
```

## Common Server Examples

### Database Servers

#### SQLite Server
```json
{
  "my-database": {
    "name": "my-database",
    "description": "Project database server",
    "server_type": "python",
    "command": "uv",
    "args": ["run", "mcp-server-sqlite", "--db-path", "data/project.db"],
    "env": {
      "DB_READONLY": "false"
    },
    "enabled": true,
    "tags": ["database", "sqlite", "project"]
  }
}
```

#### PostgreSQL Server
```json
{
  "postgres-server": {
    "name": "postgres-server",
    "description": "PostgreSQL database server",
    "server_type": "python",
    "command": "python",
    "args": ["servers/postgres_server.py"],
    "env": {
      "DATABASE_URL": "postgresql://user:pass@localhost/db",
      "SCHEMA": "public"
    },
    "enabled": true,
    "tags": ["database", "postgresql", "production"]
  }
}
```

### File System Servers

#### Local Filesystem
```json
{
  "filesystem": {
    "name": "filesystem",
    "description": "Local filesystem access",
    "server_type": "javascript",
    "command": "npx",
    "args": ["@modelcontextprotocol/server-filesystem", "/home/user/documents"],
    "enabled": true,
    "tags": ["filesystem", "local", "documents"]
  }
}
```

#### Git Repository
```json
{
  "git-server": {
    "name": "git-server",
    "description": "Git repository access",
    "server_type": "python",
    "command": "python",
    "args": ["servers/git_server.py"],
    "env": {
      "REPO_PATH": "/path/to/repository",
      "BRANCH": "main"
    },
    "enabled": true,
    "tags": ["git", "repository", "version-control"]
  }
}
```

### API Integration Servers

#### REST API Server
```json
{
  "rest-api": {
    "name": "rest-api",
    "description": "Generic REST API integration",
    "server_type": "python",
    "command": "python",
    "args": ["servers/rest_api_server.py"],
    "env": {
      "API_BASE_URL": "https://api.example.com/v1",
      "API_KEY": "${REST_API_KEY}",
      "RATE_LIMIT": "100"
    },
    "enabled": true,
    "tags": ["api", "rest", "integration"]
  }
}
```

#### GraphQL Server
```json
{
  "graphql-server": {
    "name": "graphql-server",
    "description": "GraphQL API integration",
    "server_type": "python",
    "command": "python",
    "args": ["servers/graphql_server.py"],
    "env": {
      "GRAPHQL_ENDPOINT": "https://api.example.com/graphql",
      "AUTH_TOKEN": "${GRAPHQL_TOKEN}"
    },
    "enabled": true,
    "tags": ["api", "graphql", "integration"]
  }
}
```

### Utility Servers

#### Web Scraper
```json
{
  "web-scraper": {
    "name": "web-scraper",
    "description": "Web scraping with BeautifulSoup",
    "server_type": "python",
    "command": "python",
    "args": ["servers/web_scraper_server.py"],
    "env": {
      "USER_AGENT": "MCP-Client/1.0",
      "TIMEOUT": "30",
      "MAX_PAGES": "10"
    },
    "enabled": true,
    "tags": ["web", "scraping", "data"]
  }
}
```

#### Email Server
```json
{
  "email-server": {
    "name": "email-server",
    "description": "Email integration server",
    "server_type": "python",
    "command": "python",
    "args": ["servers/email_server.py"],
    "env": {
      "SMTP_HOST": "smtp.gmail.com",
      "SMTP_PORT": "587",
      "EMAIL_USER": "${EMAIL_USER}",
      "EMAIL_PASS": "${EMAIL_PASS}"
    },
    "enabled": false,
    "tags": ["email", "communication", "smtp"]
  }
}
```

## Best Practices

### Security
- **Environment Variables**: Use `${VAR_NAME}` syntax for sensitive values
- **File Paths**: Use absolute paths when possible
- **Permissions**: Ensure server scripts have appropriate permissions
- **API Keys**: Never hardcode API keys in configuration files

### Organization
- **Naming**: Use descriptive, kebab-case names for servers
- **Tags**: Use consistent tags for grouping and filtering
- **Descriptions**: Write clear descriptions for future reference
- **Enable/Disable**: Disable unused servers instead of removing them

### Development vs Production
```json
{
  "dev-database": {
    "name": "dev-database",
    "description": "Development database server",
    "server_type": "python",
    "command": "uv",
    "args": ["run", "mcp-server-sqlite", "--db-path", "dev.db"],
    "enabled": true,
    "tags": ["database", "development"]
  },
  "prod-database": {
    "name": "prod-database",
    "description": "Production database server",
    "server_type": "python",
    "command": "uv",
    "args": ["run", "mcp-server-sqlite", "--db-path", "/data/prod.db"],
    "env": {
      "DB_READONLY": "true"
    },
    "enabled": false,
    "tags": ["database", "production"]
  }
}
```

## Troubleshooting

### Common Issues

#### Server Not Found
```bash
❌ Error: Server script not found and no configured server named: my-server

Configured servers:
  - echo-server
  - memory-server
```

**Solution**: Check server name spelling or add the server configuration.

#### Server Disabled
```bash
⚠️  Server 'sqlite-server' is disabled. Enable it first with:
   gemini-mcp-client servers enable sqlite-server
```

**Solution**: Enable the server or check why it was disabled.

#### Permission Issues
```bash
❌ Connection failed: [Errno 13] Permission denied: '/path/to/server.py'
```

**Solution**: Check file permissions and ensure the script is executable.

### Debug Configuration

Enable debug logging to see detailed configuration loading:

```bash
gemini-mcp-client --log-level DEBUG chat echo-server
```

### Reset Configuration

To reset configuration to defaults:

```bash
# Remove configuration files
rm config/servers.json config/client.json

# Restart client to regenerate defaults
gemini-mcp-client servers list
```

## Integration Examples

### Development Workflow

1. **Set up development servers**:
```bash
gemini-mcp-client servers add
# Add local database, file system, and API servers
```

2. **Test servers**:
```bash
gemini-mcp-client info dev-database
gemini-mcp-client chat dev-database
```

3. **Export for team**:
```bash
gemini-mcp-client servers export team_config.json
# Share team_config.json with team members
```

### Production Deployment

1. **Configure production servers**:
```json
{
  "prod-api": {
    "name": "prod-api",
    "description": "Production API server",
    "server_type": "python",
    "command": "/opt/mcp/bin/api_server",
    "args": ["--config", "/etc/mcp/api.conf"],
    "env": {
      "ENVIRONMENT": "production",
      "LOG_LEVEL": "WARNING"
    },
    "enabled": true,
    "tags": ["production", "api"]
  }
}
```

2. **Export for Claude Desktop**:
```bash
gemini-mcp-client servers export /etc/claude/claude_desktop_config.json
```

3. **Monitor and maintain**:
```bash
# Check server status
gemini-mcp-client servers list

# Update configuration as needed
gemini-mcp-client config set connection_timeout 60.0
```
