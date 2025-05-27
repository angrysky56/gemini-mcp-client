#!/usr/bin/env python3
"""
ReasonFlow MVP Startup Script - MCP AGENT VERSION
Quick launcher that uses your MCP agents with enhanced error handling
"""

import asyncio
import os
import sys
from contextlib import suppress

print("🧠 ReasonFlow MVP - MCP Agent Integration")
print("=" * 50)
print("Now using your MCP agents for all operations:")
print("  • SQLite Agent: Database creation and queries")
print("  • Code Executor Agent: Python analysis algorithms")
print("  • ArXiv Agent: academic paper searches")
print()

# Configure asyncio to be more resilient to GeneratorExit exceptions
# This fixes issues in the stdio_client asyncio implementation
from functools import partial

# Patch asyncio.run to handle GeneratorExit exceptions more gracefully
original_run = asyncio.run

def patched_run(coro, *, debug=None, loop_factory=None):
    """Patched version of asyncio.run that's more tolerant of GeneratorExit exceptions"""
    try:
        return original_run(coro, debug=debug)
    except BaseExceptionGroup as e:
        # Filter out GeneratorExit exceptions
        filtered_exceptions = []
        for ex in e.exceptions:
            if not isinstance(ex, GeneratorExit) and not isinstance(ex.__cause__, GeneratorExit):
                filtered_exceptions.append(ex)

        if filtered_exceptions:
            if len(filtered_exceptions) == 1:
                raise filtered_exceptions[0] from e
            else:
                raise BaseExceptionGroup("filtered errors", filtered_exceptions) from e

        print("Note: All exceptions were GeneratorExit, which were suppressed")
        return None
    except GeneratorExit:
        print("Note: GeneratorExit exception caught at top level")
        return None
# Apply the patch
asyncio.run = patched_run

# Import our coordination system
try:
    from reasonflow_mcp_fixed import ReasonFlowCoordinator
    print("✅ ReasonFlow MCP coordinator loaded")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure reasonflow_mcp_integration.py is in the current directory")
    sys.exit(1)

async def demo():
    """Demo using Google AI with MCP tool enabled agents"""

    coordinator = ReasonFlowCoordinator()
    await coordinator.initialize()

    print("\n🔄 Running MCP ReasonFlow demo...")
    print("This will:")
    print("  1. Create SQLite database")
    print("  2. Execute Python code for analysis")
    print("  3. Perform ArXiv searches")
    print("  4. Show measurable argument improvements")
    print()

    await coordinator.run_evolution_demo()

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
    print(f"""
💡 ReasonFlow MCP Commands:

  python start_reasonflow.py demo     - Full evolution demo with agents
  python start_reasonflow.py setup    - Setup database only
  python start_reasonflow.py stats    - Show database statistics
  python start_reasonflow.py help     - Show this help

🎯 What this does:
  • Creates SQLite database for argument storage
  • Executes Python code for fallacy detection and evolution
  • Performs ArXiv searches for academic validation
  • Uses your local MCP infrastructure (saves API costs!)
  • Demonstrates computational evolution of reasoning

🔧  MCP agents being used:
  • SQLite Agent: Database operations at {os.path.dirname(os.path.abspath(__file__))}/reasonflow.db
  • Code Executor: Python analysis/evolution algorithms
  • ArXiv Agent: Academic validation searches

🚀 This proves the ReasonFlow concept works with infrastructure!

🎬 Expected Demo Output:
  ✅ Database initialization
  📝 Adding test arguments (with database IDs)
  🔍 Python-based argument analysis
  🧬 Algorithm-driven argument evolution
  🔬 Academic research validation
  📊 Measurable improvement statistics

📁 Files Created:
  • ReasonFlow SQLite database with your arguments
  • Evolution log tracking improvements over time
  • Fallacy detection results with confidence scores
""")
async def safe_demo():
    """Demo with enhanced error handling"""
    try:
        return await demo()
    except BaseExceptionGroup as e:
        # Filter out GeneratorExit exceptions
        filtered_exceptions = []
        for ex in e.exceptions:
            if not isinstance(ex, GeneratorExit) and not isinstance(ex.__cause__, GeneratorExit):
                filtered_exceptions.append(ex)

        if filtered_exceptions:
            print(f"❌ Demo errors: {filtered_exceptions}")
        else:
            print("✅ Demo completed (suppressed GeneratorExit exceptions)")
        return None
    except Exception as e:
        print(f"❌ Demo error: {e}")
        return None

# Similarly, wrap the other async functions
async def safe_setup_only():
    """Setup with enhanced error handling"""
    try:
        return await setup_only()
    except BaseExceptionGroup as e:
        # Filter out GeneratorExit exceptions
        filtered_exceptions = []
        for ex in e.exceptions:
            if not isinstance(ex, GeneratorExit) and not isinstance(ex.__cause__, GeneratorExit):
                filtered_exceptions.append(ex)

        if filtered_exceptions:
            print(f"❌ Setup errors: {filtered_exceptions}")
        else:
            print("✅ Setup completed (suppressed GeneratorExit exceptions)")
        return None
    except Exception as e:
        print(f"❌ Setup error: {e}")
        return None

async def safe_show_stats():
    """Stats with enhanced error handling"""
    try:
        return await show_stats()
    except BaseExceptionGroup as e:
        # Filter out GeneratorExit exceptions
        filtered_exceptions = []
        for ex in e.exceptions:
            if not isinstance(ex, GeneratorExit) and not isinstance(ex.__cause__, GeneratorExit):
                filtered_exceptions.append(ex)

        if filtered_exceptions:
            print(f"❌ Stats errors: {filtered_exceptions}")
        else:
            print("✅ Stats completed (suppressed GeneratorExit exceptions)")
        return None
    except Exception as e:
        print(f"❌ Stats error: {e}")
        return None

if __name__ == "__main__":
    # Configure for better error handling in asyncio
    import platform
    if platform.python_version_tuple()[0] >= '3' and platform.python_version_tuple()[1] >= '11':
        # Python 3.11+ has improved task group error handling
        print("✅ Using Python 3.11+ improved error handling")
    else:
        print("⚠️ For best results, consider using Python 3.11 or higher")

    command = sys.argv[1] if len(sys.argv) > 1 else "demo"

    try:
        if command == "demo":
            asyncio.run(safe_demo())
        elif command == "help":
            show_help()
        elif command == "setup":
            asyncio.run(safe_setup_only())
        elif command == "stats":
            asyncio.run(safe_show_stats())
        else:
            print(f"Unknown command: {command}")
            show_help()
    except KeyboardInterrupt:
        print("\n👋 Operation interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n✅ Cleanup complete")
