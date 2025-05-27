# 🚨 FIXING CLAUDE DESKTOP MCP CONFIGURATION

## The Problem
The error shows that non-JSON content (emoji and formatted text) is being sent to stdout, corrupting the MCP protocol. The MCP protocol requires **only** valid JSON messages on stdout.

## ✅ Quick Fix

### Step 1: Use the Correct Configuration

Replace your Claude Desktop configuration with this **working** version:

```json
{
  "mcpServers": {
    "simple-test": {
      "command": "uv",
      "args": [
        "--directory",
        "/home/ty/Repositories/ai_workspace/gemini-mcp-client",
        "run",
        "python",
        "servers/simple_test_server.py"
      ]
    }
  }
}
```

### Step 2: Copy the Configuration

```bash
# Copy the working configuration
cp /home/ty/Repositories/ai_workspace/gemini-mcp-client/claude_desktop_config_correct.json ~/.config/claude-desktop/claude_desktop_config.json
```

### Step 3: Set Up Dependencies

```bash
cd /home/ty/Repositories/ai_workspace/gemini-mcp-client

# Install dependencies (if not already done)
uv sync

# Make sure mcp is installed
uv add mcp
```

### Step 4: Test the Server First

Before using with Claude Desktop, test the server directly:

```bash
cd /home/ty/Repositories/ai_workspace/gemini-mcp-client

# Test the simple server
uv run python servers/simple_test_server.py
```

This should start and wait for input. Press Ctrl+C to stop.

### Step 5: Restart Claude Desktop

After updating the configuration, restart Claude Desktop completely.

## 🎯 Working Configurations

### Option 1: Simple Test Server (No API Key Required)
```json
{
  "mcpServers": {
    "simple-test": {
      "command": "uv",
      "args": [
        "--directory",
        "/home/ty/Repositories/ai_workspace/gemini-mcp-client",
        "run",
        "python",
        "servers/simple_test_server.py"
      ]
    }
  }
}
```

### Option 2: Gemini AI Server (Requires API Key)
```json
{
  "mcpServers": {
    "gemini-ai": {
      "command": "uv",
      "args": [
        "--directory",
        "/home/ty/Repositories/ai_workspace/gemini-mcp-client",
        "run",
        "python",
        "servers/gemini_mcp_server.py"
      ],
      "env": {
        "GEMINI_API_KEY": "your_actual_api_key_here"
      }
    }
  }
}
```

### Option 3: Echo Server
```json
{
  "mcpServers": {
    "echo-server": {
      "command": "uv",
      "args": [
        "--directory",
        "/home/ty/Repositories/ai_workspace/gemini-mcp-client",
        "run",
        "python",
        "examples/echo_server.py"
      ]
    }
  }
}
```

## 🔍 What Was Wrong

The original issue was trying to use the **client** (`mcp-gemini-client chat`) as an MCP **server**. The client is designed for interactive use and outputs formatted text with emojis, which corrupts the MCP JSON protocol.

Instead, we need to use proper MCP **servers** that only output valid JSON messages to stdout.

## 🧪 Test the Fix

1. **Start with the simple server** (Option 1) since it has no external dependencies
2. **Check Claude Desktop logs** - you should see successful connection messages instead of JSON errors
3. **Try using the tools** in Claude Desktop - you should see tools like "echo_message" and "add_numbers"

## 🚀 Next Steps

Once the simple server works:

1. **Set up your Gemini API key** in `.env` file
2. **Switch to the Gemini AI server** configuration (Option 2)
3. **Test Gemini integration** with tools like "chat_with_gemini"

## 📝 Debugging

If you still have issues:

```bash
# Validate your JSON configuration
python /home/ty/Repositories/ai_workspace/gemini-mcp-client/validate_config.py ~/.config/claude-desktop/claude_desktop_config.json

# Test the server manually
cd /home/ty/Repositories/ai_workspace/gemini-mcp-client
uv run python servers/simple_test_server.py
```

The key insight is: **MCP clients are for interactive use, MCP servers are for Claude Desktop integration**.
