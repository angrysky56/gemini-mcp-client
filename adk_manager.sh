#!/bin/bash
# ADK Manager Script - Easy start/stop for the web interface
# Usage: ./adk_manager.sh [start|stop|restart|status]

PROJECT_DIR="/home/ty/Repositories/ai_workspace/gemini-mcp-client"
PORT=8081

case "$1" in
    start)
        echo "🚀 Starting ADK Web Interface..."
        cd "$PROJECT_DIR"
        ./cleanup_ports.sh
        echo "Starting on port $PORT..."
        uv run adk web agents --port $PORT --host 0.0.0.0
        ;;
    stop)
        echo "🛑 Stopping ADK Web Interface..."
        cd "$PROJECT_DIR"
        ./cleanup_ports.sh
        echo "✅ ADK stopped and ports cleaned up"
        ;;
    restart)
        echo "🔄 Restarting ADK Web Interface..."
        cd "$PROJECT_DIR"
        ./cleanup_ports.sh
        sleep 2
        echo "Starting on port $PORT..."
        uv run adk web agents --port $PORT --host 0.0.0.0
        ;;
    status)
        echo "📊 ADK Status Check..."
        if sudo lsof -i :$PORT >/dev/null 2>&1; then
            echo "✅ ADK is running on port $PORT"
            echo "🌐 Access at: http://localhost:$PORT"
            sudo lsof -i :$PORT
        else
            echo "❌ ADK is not running"
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        echo ""
        echo "Commands:"
        echo "  start   - Clean ports and start ADK web interface"
        echo "  stop    - Stop ADK and clean up ports"
        echo "  restart - Stop, clean, and restart ADK"
        echo "  status  - Check if ADK is running"
        echo ""
        echo "After starting, access at: http://localhost:$PORT"
        exit 1
        ;;
esac
