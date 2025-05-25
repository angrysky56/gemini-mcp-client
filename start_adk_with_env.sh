#!/bin/bash
# ADK Startup Script with Environment Loading
# This script loads the .env file and starts ADK with proper environment variables

PROJECT_DIR="/home/ty/Repositories/ai_workspace/gemini-mcp-client"
cd "$PROJECT_DIR"

# Load environment variables from .env file
if [ -f ".env" ]; then
    echo "📁 Loading environment variables from .env file..."
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ GEMINI_API_KEY loaded: ${GEMINI_API_KEY:0:20}..."
else
    echo "❌ .env file not found!"
    exit 1
fi

# Verify API key is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ GEMINI_API_KEY not found in .env file!"
    exit 1
fi

# Clean up ports first
echo "🧹 Cleaning up ports..."
./cleanup_ports.sh

# Start ADK with loaded environment
echo "🚀 Starting ADK Web Interface with environment variables..."
uv run adk web agents --port 8081 --host 0.0.0.0
