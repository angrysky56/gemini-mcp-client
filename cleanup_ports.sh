#!/bin/bash
# Port Cleanup Script for ADK Web Interface
# Save as cleanup_ports.sh and make executable: chmod +x cleanup_ports.sh

echo "🧹 Cleaning up ADK and web server ports..."

# Kill processes using common ADK ports
for port in 8080 8081 8082 8083 8084; do
    if sudo lsof -ti:$port >/dev/null 2>&1; then
        echo "  🔫 Killing processes on port $port"
        sudo lsof -ti:$port | xargs kill -9 2>/dev/null
    fi
done

# Kill ADK-related processes
echo "  🔫 Killing ADK processes"
pkill -f "adk web" 2>/dev/null
pkill -f "uvicorn" 2>/dev/null
pkill -f "fastapi" 2>/dev/null

# Wait a moment for cleanup
sleep 2

echo "✅ Port cleanup complete!"
echo ""
echo "Available ports: 8080, 8081, 8082, 8083, 8084"
echo "To start ADK: uv run adk web agents --port 8081 --host 0.0.0.0"
