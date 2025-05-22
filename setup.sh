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

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install dependencies
echo "📥 Installing dependencies..."
uv add .

# Install a Gemini package (try in order of preference)
echo "🤖 Installing Gemini package..."
if uv add --optional-dependencies gemini-tool-agent 2>/dev/null; then
    echo "✅ Installed gemini-tool-agent"
elif uv add --optional-dependencies google-generativeai 2>/dev/null; then
    echo "✅ Installed google-generativeai"
elif uv add --optional-dependencies google-genai 2>/dev/null; then
    echo "✅ Installed google-genai"
else
    echo "⚠️  Failed to install any Gemini package. Install manually:"
    echo "   uv add --optional-dependencies google-generativeai"
fi

# Install development dependencies
echo "🛠️  Installing development dependencies..."
uv add --dev ".[dev]"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo "📝 Please edit .env and add your GEMINI_API_KEY"
else
    echo "✅ .env file already exists"
fi

# Install pre-commit hooks
echo "🔗 Installing pre-commit hooks..."
uv run pre-commit install

# Run initial code formatting
echo "🎨 Formatting code..."
uv run ruff format .
uv run ruff check . --fix

# Run type checking
echo "🔍 Running type checks..."
uv run pyright || echo "⚠️  Some type checking issues found - check output above"

# Run tests
echo "🧪 Running tests..."
uv run pytest || echo "⚠️  Some tests failed - check output above"

# Test the CLI
echo "🧪 Testing CLI..."
if uv run mcp-gemini-client --help > /dev/null 2>&1; then
    echo "✅ CLI is working"
else
    echo "⚠️  CLI test failed"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env and add your GEMINI_API_KEY"
echo "2. Activate the environment: source .venv/bin/activate"
echo "3. Test the client: uv run mcp-gemini-client info examples/echo_server.py"
echo "4. Start chatting: uv run mcp-gemini-client chat examples/echo_server.py"
echo "5. List available models: uv run mcp-gemini-client models"
echo "6. Manage servers: uv run mcp-gemini-client servers list"
echo ""
echo "📚 Documentation:"
echo "- Usage guide: prompts/usage_guide.md"
echo "- Server configuration: prompts/server_configuration_guide.md"
echo "- Troubleshooting: prompts/troubleshooting.md"
echo "- Best practices: prompts/best_practices.md"
echo ""
echo "📤 Export for Claude Desktop:"
echo "   uv run mcp-gemini-client servers export"
echo ""
echo "🎉 Happy coding!"
