#!/bin/bash

# Setup script for Gemini MCP Client
# This script initializes the project with all necessary dependencies and configuration

set -e  # Exit on any error

PROJECT_DIR="/home/ty/Repositories/ai_workspace/gemini-mcp-client"
VENV_DIR="$PROJECT_DIR/.venv"

echo "🚀 Setting up Gemini MCP Client..."

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

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env and add your GEMINI_API_KEY"
echo "2. Activate the environment: source .venv/bin/activate"
echo "3. Test the client: uv run gemini-mcp-client info examples/echo_server.py"
echo "4. Start chatting: uv run gemini-mcp-client chat examples/echo_server.py"
echo ""
echo "📚 Documentation:"
echo "- Usage guide: prompts/usage_guide.md"
echo "- Troubleshooting: prompts/troubleshooting.md"
echo "- Best practices: prompts/best_practices.md"
echo ""
echo "🎉 Happy coding!"
