# 🚀 ADK Web Interface - Complete Setup Guide

## 🎯 Quick Start

```bash
# 1. Clean up any existing ports
./cleanup_ports.sh

# 2. Start ADK web interface
uv run adk web agents --port 8081 --host 0.0.0.0

# 3. Access in browser: http://localhost:8081
```

## 🤖 Optimized Gemini Models (Maximum Free Usage)

All agents now use the most generous experimental/preview models:

| Agent | Model | Why This Model |
|-------|-------|----------------|
| **Multi-MCP Agent** | `gemini-2.5-pro-exp-03-25` | Most powerful experimental - best for complex multi-tool tasks |
| **SQLite Agent** | `gemini-2.5-flash-preview-05-20` | Latest preview flash - generous limits for database queries |
| **Chroma Agent** | `gemini-2.5-flash-preview-05-20` | Latest preview - good for vector operations |
| **ArXiv Agent** | `gemini-2.5-pro-exp-03-25` | Experimental pro - best for research analysis |
| **Code Executor** | `gemini-2.0-flash-exp` | Experimental flash - efficient for code tasks |

### Model Hierarchy
1. 🥇 **`gemini-2.5-pro-exp-03-25`** - Highest capability, generous experimental limits
2. 🥈 **`gemini-2.5-flash-preview-05-20`** - Latest preview, very generous
3. 🥉 **`gemini-2.0-flash-exp`** - Experimental 2.0, good limits
4. 🎖️ **`gemini-1.5-flash-8b`** - Smallest/fastest, highest request limits

**Some Models to use (Limited but Free):**
1. `gemini-2.5-flash-preview-05-20` - Latest experimental flash 10 rpm 500 req/day (free)
2. `gemini-2.0-flash-lite` 30 rpm 1500 req/day (free)
2. `gemma-3-27b-it` - 30 rpm 14400 req/day (free)
3. `gemma-3n-e4b-it` - 30 rpm 14400 req/day (free)
4. `gemini-2.5-pro-preview-05-06` - 5 rpm 25 req/day (free)
5. `gemini-1.5-pro` - 2 rpm 50 req/day (free)
6. `gemini-2.0-flash-preview-image-generation` - 15 rpm 1500 req/day (free)

## 🧹 Port Management

### Always Clean Up When Done!
```bash
# Method 1: Use our cleanup script
./cleanup_ports.sh

# Method 2: Manual cleanup
sudo lsof -ti:8081 | xargs kill -9
pkill -f "adk web"

# Method 3: Check what's running
sudo lsof -i :8081
ps aux | grep adk
```

### Auto-cleanup on Exit
Add this to your `~/.bashrc`:
```bash
# ADK cleanup function
adk_start() {
    cd /home/ty/Repositories/ai_workspace/gemini-mcp-client
    ./cleanup_ports.sh
    uv run adk web agents --port 8081 --host 0.0.0.0
}

adk_stop() {
    cd /home/ty/Repositories/ai_workspace/gemini-mcp-client
    ./cleanup_ports.sh
}
```

## 🛠️ Available Agents & Their MCP Servers

### 1. Multi-MCP Agent (🎯 **Recommended for General Use**)
- **All 7 MCP servers combined**
- **Model**: `gemini-2.5-pro-exp-03-25` (most powerful)
- **Use for**: Complex tasks needing multiple tools

### 2. SQLite Database Agent
- **Database**: Algorithm platform (`algo.db`)
- **Tools**: Query, analyze, manage database
- **Model**: `gemini-2.5-flash-preview-05-20`

### 3. Chroma Vector Agent
- **Vector DB**: Semantic search, embeddings
- **Tools**: Document storage, similarity search
- **Model**: `gemini-2.5-flash-preview-05-20`

### 4. ArXiv Research Agent
- **Research**: Academic papers, analysis
- **Tools**: Search, download, analyze papers
- **Model**: `gemini-2.5-pro-exp-03-25`

### 5. Code Executor Agent
- **Python**: Safe code execution
- **Tools**: Run, test, debug code
- **Model**: `gemini-2.0-flash-exp`

## 🔧 Configuration Files

### MCP Server Status Check
```bash
# Test individual MCP servers
uv --directory /home/ty/Repositories/servers/src/sqlite run mcp-server-sqlite --version
uvx chroma-mcp --help
uvx docker-mcp --help
```

### Environment Variables (.env)
```env
GEMINI_API_KEY=  # ✅ Set
LOG_LEVEL=INFO
```

## 🚨 Troubleshooting

### Port Issues
```bash
# If port 8081 is busy, try others:
uv run adk web agents --port 8082 --host 0.0.0.0
uv run adk web agents --port 8083 --host 0.0.0.0
```

### MCP Server Issues
```bash
# Test if servers are accessible
ls -la /home/ty/Repositories/servers/src/sqlite/
ls -la /home/ty/Repositories/chroma-db/
ls -la /home/ty/Repositories/arxiv-mcp-server/
```

### Model Issues
If a model gives rate limit errors, the agent will fall back to simpler models automatically.

## 💡 Usage Tips

1. **Start with Multi-MCP Agent** - It has all tools available
2. **Use specific agents** for focused tasks (SQLite for DB, ArXiv for research)
3. **Always run cleanup script** when switching between sessions
4. **Monitor rate limits** - experimental models are generous but not unlimited
5. **Use different ports** if running multiple instances

## 📁 Project Structure
```
gemini-mcp-client/
├── agents/                  # ADK agent directory
│   ├── multi_mcp_agent/    # All MCP servers combined
│   ├── sqlite_agent/       # Database agent
│   ├── chroma_agent/       # Vector DB agent
│   ├── arxiv_agent/        # Research agent
│   └── code_executor_agent/ # Code execution agent
├── cleanup_ports.sh        # Port cleanup script
├── .env                    # Environment variables
└── ADK_SETUP_GUIDE.md     # This guide
```

## 🎉 Ready to Use!

Your ADK web interface is configured with:
- ✅ 5 specialized agents
- ✅ 7 MCP servers integrated
- ✅ Optimized free Gemini models
- ✅ Proper port management
- ✅ All your requested tools

**Start command:**
```bash
cd /home/ty/Repositories/ai_workspace/gemini-mcp-client
./cleanup_ports.sh && uv run adk web agents --port 8081 --host 0.0.0.0
```

Then open: **http://localhost:8081**
