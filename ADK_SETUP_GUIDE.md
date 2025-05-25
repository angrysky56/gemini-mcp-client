# ADK Web Interface Setup for Gemini MCP Client

The ADK web interface is runnable at: **http://localhost:8080**

## Available Agents

You now have 5 specialized agents available in the web interface:

### 1. **SQLite Database Agent** (`sqlite_agent`)
- **Purpose**: Database querying and management
- **Capabilities**: Query algorithm platform database, explore structure, analyze data
- **MCP Server**: SQLite server with your algo.db database

### 2. **Multi-MCP Agent** (`multi_mcp_agent`)
- **Purpose**: Comprehensive assistant with ALL MCP servers
- **Capabilities**: All tools from all servers combined
- **MCP Servers**: SQLite, Docker, Desktop Commander, Chroma, ArXiv, Package Version, Code Executor

### 3. **Chroma Vector Agent** (`chroma_agent`)
- **Purpose**: Vector database and semantic search
- **Capabilities**: Document storage, embedding search, similarity queries
- **MCP Server**: Chroma vector database

### 4. **ArXiv Research Agent** (`arxiv_agent`)
- **Purpose**: Academic paper research and analysis
- **Capabilities**: Search papers, download, analyze research
- **MCP Server**: ArXiv paper server

### 5. **Code Executor Agent** (`code_executor_agent`)
- **Purpose**: Safe Python code execution
- **Capabilities**: Run Python code, test algorithms, debug
- **MCP Server**: Code execution environment

## Port Management & Cleanup

### 🚨 IMPORTANT: Always Clean Up Ports When Done

```bash
# Method 1: Stop the running ADK server gracefully
# In the terminal where ADK is running, press: Ctrl+C

# Method 2: Kill processes using specific ports
sudo lsof -ti:8081 | xargs kill -9
sudo lsof -ti:8080 | xargs kill -9

# Method 3: Find and kill ADK processes
pkill -f "adk web"
pkill -f "uvicorn"

# Method 4: Check what's using a port
sudo lsof -i :8081
sudo netstat -tulpn | grep :8081
```

### Port Management Script
```bash
#!/bin/bash
# Save as cleanup_ports.sh
echo "Cleaning up ADK and web server ports..."
sudo lsof -ti:8080 | xargs kill -9 2>/dev/null
sudo lsof -ti:8081 | xargs kill -9 2>/dev/null
sudo lsof -ti:8082 | xargs kill -9 2>/dev/null
pkill -f "adk web" 2>/dev/null
pkill -f "uvicorn" 2>/dev/null
echo "Ports cleaned up!"
```

## Optimal Gemini Models (Free/Experimental)

### Updated Agent Configurations for Free Models

Let me update all agents to use the most generous free models:

**Some Models to use (Limited but Free):**
1. `gemini-2.5-flash-preview-05-20` - Latest experimental flash 10 rpm 500 req/day (free)
2. `gemini-2.0-flash-lite` 30 rpm 1500 req/day (free)
2. `gemma-3-27b-it` - 30 rpm 14400 req/day (free)
3. `gemma-3n-e4b-it` - 30 rpm 14400 req/day (free)
4. `gemini-2.5-pro-preview-05-06` - 5 rpm 25 req/day (free)
5. `gemini-1.5-pro` - 2 rpm 50 req/day (free)
6. `gemini-2.0-flash-preview-image-generation` - 15 rpm 1500 req/day (free)