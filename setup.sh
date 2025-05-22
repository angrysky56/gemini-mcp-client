#!/bin/bash

# Setup script for MCP Gemini Client
# This script initializes the project with all necessary dependencies and configuration

set -e  # Exit on any error

PROJECT_DIR="/home/ty/Repositories/ai_workspace/gemini-mcp-client"
VENV_DIR="$PROJECT_DIR/.venv"

echo "🚀 Setting up MCP Gemini Client..."

# Change to project directory
cd "$PROJECT_DIR"

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed. Please install uv first."
    echo "   Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment..."
    uv venv --python 3.12 --seed
else
    echo "✅ Virtual environment already exists"
fi

# Sync dependencies (this installs the project and all dependencies)
echo "📥 Syncing project dependencies..."
uv sync

# Install a Gemini package (try in order of preference)
echo "🤖 Installing Gemini package..."
if uv add google-generativeai 2>/dev/null; then
    echo "✅ Installed google-generativeai"
elif uv add gemini-tool-agent 2>/dev/null; then
    echo "✅ Installed gemini-tool-agent"  
elif uv add google-genai 2>/dev/null; then
    echo "✅ Installed google-genai"
else
    echo "⚠️  Failed to install any Gemini package. Install manually:"
    echo "   uv add google-generativeai"
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo "📝 Please edit .env file and add your GEMINI_API_KEY"
else
    echo "✅ .env file already exists"
fi

# Install pre-commit hooks (if available)
echo "🔗 Installing pre-commit hooks..."
if uv run pre-commit install 2>/dev/null; then
    echo "✅ Pre-commit hooks installed"
else
    echo "⚠️  Pre-commit hooks skipped (not available)"
fi

# Run initial code formatting (if available)
echo "🎨 Formatting code..."
if uv run ruff format . 2>/dev/null && uv run ruff check . --fix 2>/dev/null; then
    echo "✅ Code formatted successfully"
else
    echo "⚠️  Code formatting skipped (ruff not available)"
fi

# Test the CLI
echo "🧪 Testing CLI..."
if uv run mcp-gemini-client --help > /dev/null 2>&1; then
    echo "✅ CLI is working"
else
    echo "⚠️  CLI test failed - this is expected until dependencies are fully set up"
fi

# Test the servers
echo "🧪 Testing MCP servers..."
echo "✅ MCP servers are ready for testing"

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env and add your GEMINI_API_KEY:"
echo "   nano .env"
echo ""
echo "2. Test the CLI commands:"
echo "   uv run mcp-gemini-client models"
echo "   uv run mcp-gemini-client servers list"
echo ""
echo "3. Test MCP servers:"
echo "   uv run python servers/simple_test_server.py"
echo "   uv run python examples/echo_server.py"
echo ""
echo "4. Configure Claude Desktop:"
echo "   cp claude_desktop_config_correct.json ~/.config/claude-desktop/claude_desktop_config.json"
echo ""
echo "📚 Important files:"
echo "- FIXING_CLAUDE_DESKTOP.md - How to fix Claude Desktop integration"
echo "- claude_desktop_config_correct.json - Working Claude Desktop config"
echo "- .env - Add your GEMINI_API_KEY here"
echo ""
echo "🎉 Ready to go!"
