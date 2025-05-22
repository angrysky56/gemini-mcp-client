# Best Practices for MCP Client Usage

## Development Best Practices

### 1. Project Structure
- Keep server scripts in dedicated folders
- Use clear, descriptive names for servers
- Maintain separate environments for different projects
- Document server capabilities and usage

### 2. Error Handling
```python
async def robust_client_usage():
    client = MCPClient()
    try:
        await client.connect_to_server("server.py")
        response = await client.get_response("user input")
        return response
    except ServerConnectionError as e:
        logger.error(f"Connection failed: {e}")
        # Handle connection errors
    except ToolExecutionError as e:
        logger.error(f"Tool execution failed: {e}")
        # Handle tool errors
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        # Handle other errors
    finally:
        await client.close()
```

### 3. Resource Management
```python
# Use async context managers for automatic cleanup
async def proper_resource_management():
    async with AsyncExitStack() as stack:
        client = MCPClient()
        await stack.enter_async_context(client)
        # Use client safely
        return await client.get_response("Hello")
```

### 4. Configuration Management
```python
# Load configuration from environment
import os
from dotenv import load_dotenv

load_dotenv()

client = MCPClient(
    api_key=os.getenv("GEMINI_API_KEY"),
    log_level=os.getenv("LOG_LEVEL", "INFO")
)
```

## Performance Best Practices

### 1. Connection Reuse
```python
# Reuse connections for multiple interactions
async def efficient_multiple_queries():
    client = MCPClient()
    await client.connect_to_server("server.py")
    
    try:
        # Multiple queries on same connection
        response1 = await client.get_response("First query")
        response2 = await client.get_response("Second query")
        response3 = await client.get_response("Third query")
        
        return [response1, response2, response3]
    finally:
        await client.close()
```

### 2. Async Best Practices
```python
# Use asyncio.gather for concurrent operations
async def concurrent_operations():
    client = MCPClient()
    await client.connect_to_server("server.py")
    
    try:
        # Run multiple operations concurrently
        tasks = [
            client.call_tool_directly("tool1", {"param": "value1"}),
            client.call_tool_directly("tool2", {"param": "value2"}),
            client.read_resource("resource://example")
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    finally:
        await client.close()
```

### 3. Memory Management
```python
# Clear conversation history periodically
async def manage_conversation_history():
    client = MCPClient()
    await client.connect_to_server("server.py")
    
    try:
        for i in range(100):
            response = await client.get_response(f"Query {i}")
            
            # Clear history every 20 interactions
            if i % 20 == 0 and client.agent:
                client.agent.history = client.agent.history[-10:]  # Keep last 10
        
    finally:
        await client.close()
```

## Security Best Practices

### 1. API Key Management
```python
# Never hardcode API keys
# ❌ Bad
client = MCPClient(api_key="actual-api-key-here")

# ✅ Good
client = MCPClient(api_key=os.getenv("GEMINI_API_KEY"))
```

### 2. Input Validation
```python
# Validate user inputs before processing
def validate_user_input(user_input: str) -> bool:
    # Check for malicious content
    dangerous_patterns = ["eval(", "exec(", "__import__"]
    return not any(pattern in user_input for pattern in dangerous_patterns)

async def safe_interaction():
    client = MCPClient()
    await client.connect_to_server("server.py")
    
    user_input = input("Enter your query: ")
    if validate_user_input(user_input):
        response = await client.get_response(user_input)
        print(response)
    else:
        print("Invalid input detected")
```

### 3. Server Validation
```python
# Validate server scripts before connecting
def validate_server_script(script_path: str) -> bool:
    """Basic validation of server script safety."""
    try:
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Check for suspicious patterns
        dangerous_imports = ["os.system", "subprocess.call", "eval"]
        return not any(pattern in content for pattern in dangerous_imports)
    except Exception:
        return False
```

## Testing Best Practices

### 1. Unit Testing
```python
import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_client_connection():
    """Test client connection functionality."""
    client = MCPClient()
    
    # Mock the connection
    with patch('mcp.client.stdio.stdio_client') as mock_stdio:
        mock_stdio.return_value = (AsyncMock(), AsyncMock())
        
        await client.connect_to_server("test_server.py")
        assert client._connected is True
```

### 2. Integration Testing
```python
@pytest.mark.asyncio
async def test_end_to_end_workflow():
    """Test complete workflow with real server."""
    client = MCPClient()
    
    try:
        await client.connect_to_server("examples/echo_server.py")
        
        # Test tool usage
        response = await client.get_response("Echo 'Hello World'")
        assert "Hello World" in response
        
        # Test direct tool call
        result = await client.call_tool_directly("echo", {"message": "Test"})
        assert result == "Echo: Test"
        
    finally:
        await client.close()
```

### 3. Mock Testing
```python
# Create mock servers for testing
class MockMCPServer:
    def __init__(self):
        self.tools = [{"name": "test_tool", "description": "Test tool"}]
    
    async def list_tools(self):
        return self.tools
    
    async def call_tool(self, name, args):
        return f"Mock result for {name} with {args}"
```

## Production Best Practices

### 1. Logging Configuration
```python
import logging
from logging.handlers import RotatingFileHandler

# Configure production logging
def setup_production_logging():
    handler = RotatingFileHandler(
        "mcp_client.log", 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    logger = logging.getLogger("gemini_mcp_client")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
```

### 2. Health Monitoring
```python
class HealthMonitor:
    def __init__(self, client: MCPClient):
        self.client = client
        self.last_successful_call = None
        self.failure_count = 0
    
    async def health_check(self) -> bool:
        """Check if client is healthy."""
        try:
            if not self.client._connected:
                return False
            
            # Try a simple operation
            await self.client.get_server_info()
            self.failure_count = 0
            self.last_successful_call = asyncio.get_event_loop().time()
            return True
            
        except Exception:
            self.failure_count += 1
            return False
```

### 3. Circuit Breaker Pattern
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e
```

## Code Quality Best Practices

### 1. Type Hints
```python
from typing import Optional, Dict, Any, List

async def process_queries(
    client: MCPClient,
    queries: List[str],
    timeout: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Process multiple queries with proper type hints."""
    results = []
    for query in queries:
        try:
            response = await client.get_response(query)
            results.append({"query": query, "response": response, "success": True})
        except Exception as e:
            results.append({"query": query, "error": str(e), "success": False})
    
    return results
```

### 2. Documentation
```python
class MCPClientWrapper:
    """
    High-level wrapper for MCP Client with additional features.
    
    This class provides a simplified interface for common MCP operations
    while handling connection management, error recovery, and resource cleanup.
    
    Example:
        async with MCPClientWrapper("server.py") as client:
            response = await client.ask("What tools are available?")
            print(response)
    """
    
    def __init__(self, server_path: str, api_key: Optional[str] = None):
        """
        Initialize the MCP client wrapper.
        
        Args:
            server_path: Path to the MCP server script
            api_key: Optional Gemini API key
        """
        self.server_path = server_path
        self.client = MCPClient(api_key=api_key)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.client.connect_to_server(self.server_path)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.close()
```

### 3. Configuration Classes
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class MCPClientConfig:
    """Configuration for MCP Client."""
    api_key: Optional[str] = None
    log_level: str = "INFO"
    connection_timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    
    @classmethod
    def from_env(cls) -> "MCPClientConfig":
        """Create configuration from environment variables."""
        return cls(
            api_key=os.getenv("GEMINI_API_KEY"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            connection_timeout=float(os.getenv("CONNECTION_TIMEOUT", "30.0")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("RETRY_DELAY", "1.0"))
        )
```

## monitoring and Observability

### 1. Metrics Collection
```python
import time
from collections import defaultdict

class MetricsCollector:
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def record_response_time(self, operation: str, duration: float):
        self.metrics[f"{operation}_response_time"].append(duration)
    
    def record_error(self, operation: str, error_type: str):
        self.metrics[f"{operation}_errors"].append(error_type)
    
    def get_stats(self) -> Dict[str, Any]:
        stats = {}
        for key, values in self.metrics.items():
            if "response_time" in key:
                stats[key] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
            else:
                stats[key] = {"count": len(values)}
        return stats
```

### 2. Structured Logging
```python
import structlog

logger = structlog.get_logger()

async def logged_operation(client: MCPClient, query: str):
    """Operation with structured logging."""
    start_time = time.time()
    
    try:
        logger.info("Starting operation", query=query)
        response = await client.get_response(query)
        
        duration = time.time() - start_time
        logger.info(
            "Operation completed successfully",
            query=query,
            duration=duration,
            response_length=len(response)
        )
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            "Operation failed",
            query=query,
            duration=duration,
            error=str(e),
            error_type=type(e).__name__
        )
        raise
```

These best practices will help you build robust, maintainable, and secure applications with the Gemini MCP Client.
