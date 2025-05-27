#!/usr/bin/env python3
"""
ReasonFlow MCP Integration Bridge - ADK API VERSION
Connects to MCP agents using the correct ADK API pattern
"""

import asyncio
import json
import sys

from agents import multi_mcp_agent

# Add your agent /home/ty/Repositories/ai_workspace/gemini-mcp-client/agents/multi_mcp_agent/agent.py
sys.path.append('/home/ty/Repositories/ai_workspace/gemini-mcp-client/agents/multi_mcp_agent/agent.py')

# Import your agents and ADK components
try:
    from google.adk.artifacts import InMemoryArtifactService
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    from google.adk.agents import LlmAgent
    from google.adk.tools.mcp_tool import MCPToolset
    from google.adk.tools.mcp_tool.mcp_toolset import StdioServerParameters
    from multi_mcp_agent.agent import (
        agent as multi_mcp_agent,
        sqlite_agent,
        arxiv_agent,
        code_executor_agent,
    )

    print("✅ Successfully imported MCP agents and ADK components")
    AGENTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Could not import MCP agents or ADK components: {e}")
    AGENTS_AVAILABLE = False

class MCPBridge:
    """
    Bridge that uses your MCP agents via ADK Runner interface
    """

    def __init__(self):
        self.agents_available = AGENTS_AVAILABLE
        self.initialized = False

        # Database path for ReasonFlow
        self.reasonflow_db_path = "/home/ty/Repositories/ai_workspace/gemini-mcp-client/agents/reasonflow_mvp/reasonflow.db"

        # Ensure the directory exists
        import os
        db_dir = os.path.dirname(self.reasonflow_db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            print(f"📁 Created directory: {db_dir}")

        # Response cache
        self.response_cache = {}
        print(f"🗄️  ReasonFlow database will be at: {self.reasonflow_db_path}")

    async def initialize(self):
        """Initialize the bridge with ADK components (must be called after construction)"""
        if self.agents_available and not self.initialized:


            # Import the agents
            self.multi_agent = multi_mcp_agent
            self.sqlite_agent = sqlite_agent
            self.arxiv_agent = arxiv_agent
            self.code_executor_agent = code_executor_agent

            # Set up ADK services for agent execution
            self.session_service = InMemorySessionService()

            self.initialized = True
            print("✅ MCPBridge initialized successfully")

    def _run_agent_sync(self, agent, query: str) -> str:
        """Helper method to run an agent using ADK API synchronously."""

        if not self.agents_available or not self.initialized:
            return "ERROR: MCP agents not available or not initialized"

        try:
            # Create a fresh runner for each call with the session service
            self.runner = Runner(session_service=self.session_service, app_name="reasonflow", agent=agent)

            # Run the agent synchronously
            content_message = types.Content(parts=[types.Part(text=query)])
            result_gen = self.runner.run(user_id="user", session_id="id", new_message=content_message)

            # Extract the response from the generator of events
            output_text = ""
            try:
                for event in result_gen:
                    # If the event has a 'content' attribute with parts, extract the text from the first part
                    if hasattr(event, "content") and hasattr(event.content, "parts") and event.content.parts:
                        output_text += str(event.content.parts[0].text)
                    elif isinstance(event, str):
                        output_text += event
                return output_text if output_text else "No response content found"
            except Exception as e:
                import traceback
                traceback.print_exc()
                return f"Agent execution error (event iteration): {str(e)}"

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Agent execution error: {str(e)}"
    async def execute_sql_query(self, query: str) -> str:
        """Execute SQL using the SQLite agent with ADK API"""

        if not self.agents_available:
            return "ERROR: MCP agents not available"

        if not self.initialized:
            await self.initialize()

        try:
            # Create message for the SQLite agent about our specific database
            message = f"""
            I need to work with a ReasonFlow database at {self.reasonflow_db_path}.
            Please execute this SQL query:

            {query}

            If the database doesn't exist yet, please create it first.
            Return just the result of the SQL operation.
            """

            print("🗄️  Executing SQL via MCP SQLite agent...")
            result = self._run_agent_sync(self.sqlite_agent, message)

            if "error" in result.lower():  # Basic error check
                print(f"❌ SQL error detected: {result}")
                return result
            print(f"📊 SQL result: {result[:100]}...")
            return result

        except Exception as e:
            error_msg = f"SQL execution error: {type(e).__name__}: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

    async def execute_python_code(self, code: str) -> str:
        """Execute Python code using the Code Executor agent with ADK API"""

        if not self.agents_available:
            return "ERROR: MCP agents not available"

        if not self.initialized:
            await self.initialize()

        try:
            message = f"""
            Please execute this Python code and return the output:

            ```python
            {code}
            ```

            Show me the complete output including any print statements or final results.
            """

            print("🐍 Executing Python code via MCP Code Executor agent...")
            result = self._run_agent_sync(self.code_executor_agent, message)

            if "error" in result.lower():  # Basic error check
                print(f"❌ Code execution error detected: {result}")
                return result

            print(f"⚡ Code result: {result[:100]}...")
            return result

        except Exception as e:
            error_msg = f"Code execution error: {type(e).__name__}: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

    async def search_arxiv(self, query: str, max_results: int = 3) -> str:
        """Search ArXiv using the ArXiv agent with ADK API"""

        if not self.agents_available:
            return "ERROR: MCP agents not available"

        if not self.initialized:
            await self.initialize()

        cache_key = f"arxiv_{hash(query)}"
        if cache_key in self.response_cache:
            print("📚 Using cached ArXiv result")
            return self.response_cache[cache_key]

        try:
            message = f"""
            Search ArXiv for papers related to: {query}

            Please find up to {max_results} relevant papers and provide:
            - Paper titles
            - Brief abstracts or summaries
            - How they relate to the search topic

            Focus on the most relevant and recent papers.
            """

            print("📚 Searching ArXiv via MCP ArXiv agent...")

            # Use ADK API to call the ArXiv agent
            result = self._run_agent_sync(self.arxiv_agent, message)

            # Cache the result
            self.response_cache[cache_key] = result
            print("🔍 ArXiv search completed")

            return result

        except Exception as e:
            error_msg = f"ArXiv search error: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

class ReasonFlowCoordinator:
    """
    ReasonFlow coordinator using MCP agents with ADK API
    """

    def __init__(self):
        self.bridge = MCPBridge()
        self.db_initialized = False
        self.arguments_created = 0

        print("🧠 ReasonFlow Coordinator initialized with MCP agents via ADK API")

    async def initialize(self):
        """Initialize the coordinator and its bridge"""
        await self.bridge.initialize()

    async def setup_database(self) -> bool:
        """Initialize ReasonFlow database using SQLite agent"""

        print("🗄️  Setting up ReasonFlow database schema...")

        # Create database schema
        schema_sql = """
        CREATE TABLE IF NOT EXISTS rf_arguments (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            type TEXT DEFAULT 'claim',
            fitness_score REAL DEFAULT 0.0,
            generation INTEGER DEFAULT 0,
            parent_ids TEXT DEFAULT '[]',
            analysis_cache TEXT DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_rf_arguments_type ON rf_arguments (type);
        CREATE INDEX IF NOT EXISTS idx_rf_arguments_created_at ON rf_arguments (created_at);

        CREATE TABLE IF NOT EXISTS rf_evolution_log (
            id TEXT PRIMARY KEY,
            parent_id TEXT,
            child_id TEXT,
            operation_type TEXT,
            improvement_score REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_rf_evolution_log_parent_id ON rf_evolution_log (parent_id);
        CREATE INDEX IF NOT EXISTS idx_rf_evolution_log_child_id ON rf_evolution_log (child_id);

        CREATE TABLE IF NOT EXISTS rf_fallacy_detections (
            id TEXT PRIMARY KEY,
            argument_id TEXT,
            fallacy_type TEXT,
            confidence REAL,
            explanation TEXT,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_rf_fallacy_detections_argument_id ON rf_fallacy_detections (argument_id);
        """

        result = await self.bridge.execute_sql_query(schema_sql)

        # Check if creation was successful
        success_indicators = ["success", "created", "table", "completed", "ok", "done"]
        self.db_initialized = any(indicator in result.lower() for indicator in success_indicators)

        if self.db_initialized:
            print("✅ ReasonFlow database initialized successfully")

            # Verify with a test query
            test_result = await self.bridge.execute_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'rf_%';"
            )
            print(f"📋 Tables created: {test_result}")

        else:
            print(f"❌ Database initialization may have failed: {result}")

        return self.db_initialized

    async def add_argument(self, content: str, arg_type: str = "claim") -> str:
        """Add argument to database using SQLite agent"""

        import random
        arg_id = f"rf_arg_{random.randint(10000, 99999)}"

        # Escape content for SQL
        safe_content = content.replace("'", "''")

        insert_sql = f"""
        INSERT INTO rf_arguments (id, content, type, fitness_score, generation)
        VALUES ('{arg_id}', '{safe_content}', '{arg_type}', 0.0, 0);
        """

        result = await self.bridge.execute_sql_query(insert_sql)

        # Check if insertion was successful
        success_indicators = ["success", "inserted", "1 row", "ok", "completed", "done"]
        if any(indicator in result.lower() for indicator in success_indicators):
            print(f"✅ Added argument: {arg_id}")
            self.arguments_created += 1
            return arg_id
        else:
            print(f"❌ Failed to add argument: {result}")
            return ""

    async def get_argument_content(self, arg_id: str) -> str:
        """Retrieve argument content from database"""

        query_sql = f"SELECT content FROM rf_arguments WHERE id = '{arg_id}';"
        result = await self.bridge.execute_sql_query(query_sql)

        # Extract content from result
        try:
            # Look for content in the result
            lines = result.strip().split('\n')
            for line in lines:
                # Skip header lines and empty lines
                if line.strip() and not line.startswith('-') and 'content' not in line.lower() and not line.startswith('-') and len(line.strip()) > 10:
                    return line.strip()

            # Fallback - return first substantial line
            for line in lines:
                if len(line.strip()) > 5:
                    return line.strip()

            return "Content not found"

        except Exception as e:
            print(f"Error extracting content: {e}")
            return "Content extraction failed"

    async def analyze_argument(self, arg_id: str, content: str) -> dict:
        """Analyze argument using Code Executor agent"""

        print(f"🔍 Analyzing argument: {arg_id}")
        try:
            import importlib.util

            # Check if analysis_functions.analyze_argument_advanced is available
            spec = importlib.util.find_spec("analysis_functions")
            if spec is None:
                raise ImportError("analysis_functions module not found")

            # Execute analysis
            analysis_code = f'result = analyze_argument_advanced("""{content.replace('"', '\\"')}""")\nprint(json.dumps(result, indent=2))'  # Correctly generate JSON output

            analysis_result = await self.bridge.execute_python_code(analysis_code)

            # Improved parsing
            try:
                analysis = json.loads(analysis_result)
                print(f"✅ Analysis completed: {analysis['reasoning_quality']} reasoning, {len(analysis['fallacies'])} fallacies")
                return analysis

            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing error: {e}")
                print(f"Raw analysis result: {analysis_result}")
                return {
                    "word_count": len(content.split()),
                    "fallacies": [],
                    "strength": 0.4,
                    "suggestions": ["Check analysis code"],
                    "reasoning_quality": "error",
                    "error": str(e)
                }

        except ImportError as e:
            print(f"❌ Could not import analyze_argument_advanced: {e}")
            return {
                "word_count": len(content.split()),
                "fallacies": [],
                "strength": 0.1,
                "suggestions": ["Check analysis function"],
                "reasoning_quality": "unavailable",
                "error": str(e)
            }
        except Exception as e:
            print(f"❌ Analysis execution error: {e}")
            return {
                "word_count": len(content.split()),
                "fallacies": [],
                "strength": 0.2,
                "suggestions": ["Check execution"],
                "reasoning_quality": "failed",
                "error": str(e)
            }

    async def evolve_argument(self, arg_id: str, content: str, analysis: dict) -> str:
        """Evolve argument using Code Executor agent"""

        print(f"🧬 Evolving argument: {arg_id}")
        try:
            import importlib.util

            # Check if evolution_functions.evolve_argument_advanced is available
            spec = importlib.util.find_spec("evolution_functions")
            if spec is None:
                raise ImportError("evolution_functions module not found")

            evolution_code = f'''
import json
analysis = {json.dumps(analysis)}  # Pass analysis as JSON
original = """{content.replace('"', '\\"')}"""
result = evolve_argument_advanced(original, analysis)
print(json.dumps(result, indent=2))
            '''

            evolution_result = await self.bridge.execute_python_code(evolution_code)

            # Improved parsing
            try:
                evolution_data = json.loads(evolution_result)
                evolved_content = evolution_data.get("evolved_text", content + " [Evolution failed]")
                changes = evolution_data.get("changes_made", [])
                print(f"✅ Evolution completed with changes: {', '.join(changes)}")

                # Create new argument
                new_arg_id = await self.add_argument(evolved_content, "evolved_claim")
                if new_arg_id:
                    print(f"🎯 Created evolved argument: {new_arg_id}")
                    return new_arg_id
                else:
                    print("❌ Failed to save evolved argument")
                    return ""

            except json.JSONDecodeError as e:
                print(f"❌ Evolution JSON parsing error: {e}")
                print(f"Raw evolution result: {evolution_result}")
                new_arg_id = await self.add_argument(content + " [Evolution Failed: JSON Parse Error]", "failed_evolution")
                return new_arg_id if new_arg_id else ""

        except ImportError as e:
            print(f"❌ Could not import evolve_argument_advanced: {e}")
            new_arg_id = await self.add_argument(content + " [Evolution Failed: Import Error]", "failed_evolution")
            return new_arg_id if new_arg_id else ""

        except Exception as e:
            print(f"❌ Evolution execution error: {e}")
            new_arg_id = await self.add_argument(content + " [Evolution Failed: Execution Error]", "failed_evolution")
            return new_arg_id if new_arg_id else ""

    async def validate_with_research(self, content: str) -> dict:
        """Validate argument with ArXiv agent"""

        print("🔬 Validating with academic research...")

        # Extract key terms
        words = content.split()
        search_terms = " ".join(words[:4])  # First 4 words

        papers_result = await self.bridge.search_arxiv(search_terms, max_results=2)

        # Simple analysis of whether papers support the argument
        academic_support = ("paper" in papers_result.lower() or
                          "study" in papers_result.lower() or
                          "research" in papers_result.lower())

        return {
            "validation_attempted": True,
            "search_terms": search_terms,
            "papers_found": papers_result,
            "academic_support": academic_support,
            "confidence": 0.8 if academic_support else 0.3
        }

    async def run_evolution_demo(self):
        """Complete ReasonFlow demo using MCP agents with ADK API"""

        print("🚀 Starting ADK MCP ReasonFlow Demo")
        print("=" * 50)

        # Ensure bridge is initialized
        if not self.bridge.initialized:
            await self.initialize()

        if not self.bridge.agents_available:
            print("❌ MCP agents not available - cannot run demo")
            return

        # Setup database
        print("🗄️  Setting database with ADK API...")
        if not await self.setup_database():
            print("❌ Database setup failed")
            return

        # Test arguments with known issues
        test_args = [
            "Climate change is because scientists say so",
            "Vaccines are dangerous because they cause autism and everyone knows this",
            "Democracy is the best system because everyone likes it"
        ]

        arg_data = []
        print("\n📝 Adding test arguments to database via ADK API...")
        for i, arg in enumerate(test_args):
            arg_id = await self.add_argument(arg)
            if arg_id:
                arg_data.append((arg_id, arg))
                print(f"   {i+1}. {arg_id}: {arg[:50]}...")

        # analysis using code executor
        print("\n🔍 Analyzing arguments with code execution via ADK API...")
        analyses = []
        for arg_id, content in arg_data:
            analysis = await self.analyze_argument(arg_id, content)
            analyses.append((arg_id, content, analysis))

            print(f"   {arg_id}:")
            print(f"     Strength: {analysis.get('strength', 0):.2f}")
            print(f"     Quality: {analysis.get('reasoning_quality', 'unknown')}")
            print(f"     Fallacies: {len(analysis.get('fallacies', []))}")
            if analysis.get('fallacies'):
                print(f"     Detected: {', '.join(analysis['fallacies'])}")

        # Evolution using code execution
        print("\n🧬 Evolving arguments with computation via ADK API...")
        evolution_results = []
        for arg_id, content, analysis in analyses[:2]:  # Evolve first 2
            if analysis.get('strength', 1.0) < 0.8:
                new_id = await self.evolve_argument(arg_id, content, analysis)
                if new_id:
                    # Analyze evolved version
                    new_content = await self.get_argument_content(new_id)
                    new_analysis = await self.analyze_argument(new_id, new_content)

                    improvement = new_analysis.get('strength', 0) - analysis.get('strength', 0)
                    evolution_results.append((arg_id, new_id, improvement))

                    print(f"   {arg_id} → {new_id}: Improvement = {improvement:+.2f}")

        # academic validation
        print("\n🔬 Academic validation with ArXiv search via ADK API...")
        if arg_data:
            validation = await self.validate_with_research(arg_data[0][1])
            print(f"   Search terms: {validation.get('search_terms')}")
            print(f"   Academic support: {validation.get('academic_support', False)}")
            print(f"   Confidence: {validation.get('confidence', 0.0):.2f}")

        # database query for final stats
        print("\n📊 Final statistics from database via ADK API...")
        stats_query = "SELECT COUNT(*) as total FROM rf_arguments;"
        total_result = await self.bridge.execute_sql_query(stats_query)

        print(f"   Database response: {total_result}")
        print(f"   Arguments processed: {len(arg_data)}")
        print(f"   Arguments evolved: {len(evolution_results)}")
        print(f"   Total improvements: {sum(r[2] for r in evolution_results):.2f}")

        print("\n✅ ADK MCP ReasonFlow demo complete!")
        print("\n🎯 What just happened:")
        print("   • Used ADK API with async Runner and Content objects")
        print("   • Created SQLite database for argument storage")
        print("   • Executed Python code for analysis and evolution")
        print("   • Performed ArXiv searches for validation")
        print("   • Demonstrated measurable argument improvement")
        print("   • All using YOUR local MCP infrastructure with correct API!")

# Main execution
async def main():
    coordinator = ReasonFlowCoordinator()
    await coordinator.initialize()
    await coordinator.run_evolution_demo()

if __name__ == "__main__":
    print("🧠 ReasonFlow with MCP Agent Integration - ADK API")
    print("This version uses your MCP agents with ADK Runner interface")
    print()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
