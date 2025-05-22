# Troubleshooting Guide

## Common Issues and Solutions

### Connection Issues

#### "Server script not found"
**Problem**: The specified server script doesn't exist.
**Solution**: 
- Check the file path is correct
- Ensure the file exists and is readable
- Use absolute paths if relative paths aren't working

#### "Unsupported server script type"
**Problem**: Trying to connect to a non-.py or non-.js file.
**Solution**: 
- Only Python (.py) and JavaScript (.js) servers are supported
- Ensure your server file has the correct extension

#### "Connection failed"
**Problem**: Cannot establish connection to the server.
**Solution**:
- Check that the server script is valid MCP server
- Ensure required dependencies are installed
- Check server logs for startup errors
- Verify Python/Node.js is available in PATH

### API Key Issues

#### "No Gemini API key provided"
**Problem**: Missing or invalid Gemini API key.
**Solution**:
- Set `GEMINI_API_KEY` in your environment variables
- Add the key to your `.env` file
- Pass the key directly to `MCPClient(api_key="your-key")`

#### "Gemini agent failed to initialize"
**Problem**: API key is invalid or Gemini service is unavailable.
**Solution**:
- Verify your API key is correct
- Check your internet connection
- Ensure you have proper Gemini API access
- Try again later if service is temporarily unavailable

### Tool Execution Issues

#### "Tool execution failed"
**Problem**: A tool call failed to execute properly.
**Solution**:
- Check tool arguments are correct
- Verify the tool exists on the server
- Look at server logs for specific error messages
- Ensure the server is still running

#### "Tool call timeout"
**Problem**: Tool execution is taking too long.
**Solution**:
- Increase timeout settings if possible
- Check if the tool is designed for long-running operations
- Verify the server isn't overloaded
- Consider breaking down complex operations

### Installation Issues

#### "gemini-tool-agent not found"
**Problem**: The required package isn't installed.
**Solution**:
```bash
uv add gemini-tool-agent
```

#### "MCP package not found"
**Problem**: Core MCP package missing.
**Solution**:
```bash
uv add mcp
```

#### "Import errors"
**Problem**: Python can't find the modules.
**Solution**:
- Ensure virtual environment is activated
- Run `uv add .` to install in development mode
- Check Python path includes the project directory

### Runtime Issues

#### "Not connected to server"
**Problem**: Trying to use client before connecting.
**Solution**:
- Call `await client.connect_to_server()` first
- Check connection status with `client._connected`
- Handle connection errors properly

#### "Session initialization failed"
**Problem**: MCP session couldn't be established.
**Solution**:
- Verify server implements MCP protocol correctly
- Check server startup logs
- Ensure server is listening on stdio
- Try connecting to a known working server first

#### "Memory or resource leaks"
**Problem**: Client consuming too much memory over time.
**Solution**:
- Always call `await client.close()` when done
- Use context managers for automatic cleanup
- Monitor conversation history size
- Restart client periodically for long-running applications

### Debugging Tips

#### Enable Debug Logging
```python
client = MCPClient(log_level="DEBUG")
```

#### Check Log Files
```bash
tail -f mcp_client.log
```

#### Test with Simple Server
```bash
# Test with the provided echo server
gemini-mcp-client chat examples/echo_server.py
```

#### Manual Tool Testing
```python
# Test tools directly without AI
result = await client.call_tool_directly("tool_name", {"param": "value"})
```

### Performance Issues

#### "Slow responses"
**Problem**: Client takes too long to respond.
**Solution**:
- Check internet connection for Gemini API calls
- Reduce conversation history size
- Use simpler prompts
- Check server performance
- Consider using local models if available

#### "High memory usage"
**Problem**: Client using too much memory.
**Solution**:
- Clear conversation history periodically
- Close and recreate client for long sessions
- Monitor tool result sizes
- Use streaming responses if available

### Environment Issues

#### "Wrong Python version"
**Problem**: Code requires Python 3.12+.
**Solution**:
```bash
# Create environment with correct Python version
uv venv --python 3.12 --seed
source .venv/bin/activate
```

#### "Package conflicts"
**Problem**: Dependency version conflicts.
**Solution**:
```bash
# Recreate environment
rm -rf .venv
uv venv --python 3.12 --seed
source .venv/bin/activate
uv add .
```

### Getting Help

If you encounter issues not covered here:

1. Check the project logs (`mcp_client.log`)
2. Enable debug logging for more details
3. Test with the provided echo server
4. Check the MCP protocol documentation
5. Verify your server implementation
6. Test with a minimal example

### Emergency Recovery

If the client gets stuck or corrupted:

```bash
# Clean restart
rm -rf .venv mcp_client.log
uv venv --python 3.12 --seed
source .venv/bin/activate
uv add .
cp .env.example .env
# Edit .env with your API key
```
