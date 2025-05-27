#!/usr/bin/env python3
"""
ReasonFlow MVP Startup Script - PROPERLY FIXED VERSION
Uses environment variables and avoids async context issues
"""

import sys
import asyncio

print("🧠 ReasonFlow MVP - PROPERLY FIXED MCP Agent Integration")
print("=" * 50)
print("Now using properly configured MCP agents:")
print("  • Loads API key from .env file")
print("  • Uses unified agent (no switching)")
print("  • Avoids async context conflicts")
print("  • SQLite Agent: Database creation and queries")
print("  • Code Executor Agent: Python analysis algorithms")
print("  • ArXiv Agent: academic paper searches")
print()

# Import our fixed coordination system
try:
    from agents.reasonflow_mvp.reasonflow_mcp_integration_fixed_v2 import ReasonFlowCoordinator
    print("✅ ReasonFlow properly fixed coordinator loaded")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure reasonflow_mcp_integration_fixed_properly.py is in the current directory")
    sys.exit(1)

async def demo():
    """Demo using properly configured MCP agents"""
    coordinator = ReasonFlowCoordinator()
    await coordinator.initialize()

    print("\n🔄 Running FIXED MCP ReasonFlow demo...")
    print("This will:")
    print("  1. Load API key from .env")
    print("  2. Create SQLite database (unified agent)")
    print("  3. Execute Python code for analysis (unified agent)")
    print("  4. Show measurable argument improvements")
    print("  5. Avoid async context conflicts")
    print()

    await coordinator.run_simplified_demo()

async def setup_only():
    """Just setup the database"""
    coordinator = ReasonFlowCoordinator()
    await coordinator.initialize()
    print("🗄️  Setting up ReasonFlow database only...")
    success = await coordinator.setup_database()
    if success:
        print("✅ Database setup complete!")
        print(f"📍 Location: {coordinator.bridge.reasonflow_db_path}")
    else:
        print("❌ Database setup failed")

async def show_stats():
    """Show database statistics"""
    coordinator = ReasonFlowCoordinator()
    await coordinator.initialize()

    print("📊 ReasonFlow Database Statistics")
    print("-" * 30)

    # Query database stats
    stats_query = """
    SELECT
        COUNT(*) as total_arguments,
        AVG(fitness_score) as avg_fitness,
        MAX(generation) as max_generation,
        COUNT(CASE WHEN type = 'evolved_claim' THEN 1 END) as evolved_count
    FROM rf_arguments;
    """

    result = await coordinator.bridge.execute_sql_query(stats_query)
    print(f"Database stats: {result}")

    # Show recent arguments
    recent_query = "SELECT id, content, fitness_score FROM rf_arguments ORDER BY created_at DESC LIMIT 5;"
    recent = await coordinator.bridge.execute_sql_query(recent_query)
    print(f"\nRecent arguments: {recent}")

def show_help():
    print("""
💡 ReasonFlow PROPERLY FIXED Commands:

  python start_reasonflow.py demo     - Full demo with fixed agents
  python start_reasonflow.py setup    - Setup database only
  python start_reasonflow.py stats    - Show database statistics
  python start_reasonflow.py help     - Show this help

🎯 What this FIXED version does:
  • Loads API key from .env file automatically
  • Uses unified MCP agent (no dynamic switching)
  • Avoids async context conflicts that caused RuntimeError
  • Creates SQLite database for argument storage
  • Executes Python code for fallacy detection and evolution
  • Demonstrates computational evolution of reasoning
  • Uses your local MCP infrastructure reliably

🔧  FIXES Applied:
  • Environment variable loading from .env
  • Single unified agent instead of switching agents
  • Proper async context management
  • Simplified MCP connection lifecycle
  • Better error handling and recovery

🚀 This version should work without the async context errors!

🎬 Expected Demo Output:
  ✅ API key loaded from .env
  ✅ Database initialization via unified agent
  📝 Adding test arguments (with database IDs)
  🔍 Python-based argument analysis (via unified agent)
  📊 Measurable improvement statistics

📁 Files Created:
  • ReasonFlow SQLite database with your arguments
  • Analysis results from Python code execution
""")

if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "demo"

    if command == "demo":
        asyncio.run(demo())
    elif command == "help":
        show_help()
    elif command == "setup":
        asyncio.run(setup_only())
    elif command == "stats":
        asyncio.run(show_stats())
    else:
        print(f"Unknown command: {command}")
        show_help()
