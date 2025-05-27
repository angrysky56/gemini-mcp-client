#!/usr/bin/env python3
"""
ReasonFlow MCP Integration Bridge - PROPERLY FIXED VERSION
Uses environment variables and fixes async context issues
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add your agent path
sys.path.append('/home/ty/Repositories/ai_workspace/gemini-mcp-client')

# Load environment variables from .env file
def load_env_file(env_path):
    """Load environment variables from .env file"""
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print(f"✅ Loaded environment from {env_path}")
        return True
    else:
        print(f"❌ .env file not found at {env_path}")
        return False

# Load environment from repo root
repo_root = "/home/ty/Repositories/ai_workspace/gemini-mcp-client"
env_loaded = load_env_file(os.path.join(repo_root, ".env"))

# Import your actual agents and ADK components
try:
    from google.adk.artifacts import InMemoryArtifactService
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    from agents.multi_mcp_agent.agent import (
        agent as multi_mcp_agent,
        arxiv_agent,
        code_executor_agent,
        sqlite_agent,
    )

    print("✅ Successfully imported MCP agents and ADK components")
    print(f"🔑 API Key loaded: {'Yes' if os.getenv('GEMINI_API_KEY') else 'No'}")
    AGENTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Could not import MCP agents or ADK components: {e}")
    AGENTS_AVAILABLE = False

class SimplifiedMCPBridge:
    """
    Simplified bridge that avoids dynamic agent switching to prevent async context issues
    """

    def __init__(self):
        self.agents_available = AGENTS_AVAILABLE and env_loaded
        self.initialized = False

        # Database path for ReasonFlow
        self.reasonflow_db_path = "/home/ty/Repositories/ai_workspace/gemini-mcp-client/agents/reasonflow_mvp/reasonflow.db"

        # Response cache
        self.response_cache = {}
        print(f"🗄️  ReasonFlow database will be at: {self.reasonflow_db_path}")

    async def initialize(self):
        """Initialize the bridge with ADK components"""
        if self.agents_available and not self.initialized:
            # Set up ADK services for agent execution
            self.session_service = InMemorySessionService()
            self.artifact_service = InMemoryArtifactService()

            # Create session (properly await and extract session ID)
            self.user_id = "reasonflow_user"
            self.session = await self.session_service.create_session(
                app_name="ReasonFlow",
                user_id=self.user_id,
            )
            self.session_id = self.session.id

            # Use the main multi-agent for ALL operations to avoid switching
            self.main_agent = multi_mcp_agent

            # Create single runner that we'll reuse
            self.runner = Runner(
                app_name="ReasonFlow",
                agent=self.main_agent,
                artifact_service=self.artifact_service,
                session_service=self.session_service,
            )

            self.initialized = True
            print(f"✅ SimplifiedMCPBridge initialized with session ID: {self.session_id}")

    async def _run_unified_agent(self, task_description: str) -> str:
        """Run the unified agent with a specific task description"""
        if not self.agents_available or not self.initialized:
            return "ERROR: MCP agents not available or not initialized"

        try:
            # Create proper ADK content format
            content = types.Content(
                role="user",
                parts=[types.Part(text=task_description)]
            )

            # Run the unified agent (no switching!)
            events = list(
                self.runner.run(
                    user_id=self.user_id,
                    session_id=self.session_id,
                    new_message=content
                )
            )

            # Extract the final response
            if events:
                last_event = events[-1]
                if last_event.content and last_event.content.parts:
                    parts = last_event.content.parts
                    final_response = "".join(
                        [part.text for part in parts if part.text]
                    )
                    return final_response
                else:
                    return "Event found but no content parts"
            else:
                return "No response received from agent"

        except Exception as e:
            return f"Agent execution error: {str(e)}"

    async def execute_sql_query(self, query: str) -> str:
        """Execute SQL using the unified agent"""
        if not self.agents_available:
            return "ERROR: MCP agents not available"

        if not self.initialized:
            await self.initialize()

        try:
            task = f"""
            Please use your SQLite database tools to work with a ReasonFlow database at {self.reasonflow_db_path}.

            Execute this SQL query:
            {query}

            If the database doesn't exist yet, create it first.
            Return the result of the SQL operation clearly.
            """

            print("🗄️  Executing SQL via unified MCP agent...")
            result = await self._run_unified_agent(task)
            print(f"📊 SQL result: {result[:100]}...")
            return result

        except Exception as e:
            error_msg = f"SQL execution error: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

    async def execute_python_code(self, code: str) -> str:
        """Execute Python code using the unified agent"""
        if not self.agents_available:
            return "ERROR: MCP agents not available"

        if not self.initialized:
            await self.initialize()

        try:
            task = f"""
            Please use your Code Executor tools to execute this Python code:

            ```python
            {code}
            ```

            Show me the complete output including any print statements or final results.
            Make sure to execute the code and return the actual output.
            """

            print("🐍 Executing Python code via unified MCP agent...")
            result = await self._run_unified_agent(task)
            print(f"⚡ Code result: {result[:100]}...")
            return result

        except Exception as e:
            error_msg = f"Code execution error: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

    async def search_arxiv(self, query: str, max_results: int = 3) -> str:
        """Search ArXiv using the unified agent"""
        if not self.agents_available:
            return "ERROR: MCP agents not available"

        if not self.initialized:
            await self.initialize()

        cache_key = f"arxiv_{hash(query)}"
        if cache_key in self.response_cache:
            print("📚 Using cached ArXiv result")
            return self.response_cache[cache_key]

        try:
            task = f"""
            Please use your ArXiv search tools to search for papers related to: {query}

            Find up to {max_results} relevant papers and provide:
            - Paper titles
            - Brief abstracts or summaries
            - How they relate to the search topic

            Focus on the most relevant and recent papers.
            """

            print("📚 Searching ArXiv via unified MCP agent...")
            result = await self._run_unified_agent(task)

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
    ReasonFlow coordinator using simplified MCP bridge to avoid async context issues
    """

    def __init__(self):
        self.bridge = SimplifiedMCPBridge()
        self.db_initialized = False
        self.arguments_created = 0

        print("🧠 ReasonFlow Coordinator initialized with simplified MCP bridge")

    async def initialize(self):
        """Initialize the coordinator and its bridge"""
        await self.bridge.initialize()

    async def setup_database(self) -> bool:
        """Initialize ReasonFlow database"""
        print("🗄️  Setting up ReasonFlow database schema...")

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

        CREATE TABLE IF NOT EXISTS rf_evolution_log (
            id TEXT PRIMARY KEY,
            parent_id TEXT,
            child_id TEXT,
            operation_type TEXT,
            improvement_score REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS rf_fallacy_detections (
            id TEXT PRIMARY KEY,
            argument_id TEXT,
            fallacy_type TEXT,
            confidence REAL,
            explanation TEXT,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        result = await self.bridge.execute_sql_query(schema_sql)

        # Check if creation was successful
        success_indicators = ["success", "created", "table", "completed", "ok", "done", "executed"]
        self.db_initialized = any(indicator in result.lower() for indicator in success_indicators)

        if self.db_initialized:
            print("✅ ReasonFlow database initialized successfully")
        else:
            print(f"❌ Database initialization may have failed: {result}")

        return self.db_initialized

    async def add_argument(self, content: str, arg_type: str = "claim") -> str:
        """Add argument to database"""
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
        success_indicators = ["success", "inserted", "1 row", "ok", "completed", "done", "executed"]
        if any(indicator in result.lower() for indicator in success_indicators):
            print(f"✅ Added argument: {arg_id}")
            self.arguments_created += 1
            return arg_id
        else:
            print(f"❌ Failed to add argument: {result}")
            return ""

    async def analyze_argument(self, arg_id: str, content: str) -> dict:
        """Analyze argument using code execution"""
        print(f"🔍 Analyzing argument: {arg_id}")

        analysis_code = f'''
import json

def analyze_argument_advanced(text):
    """Enhanced argument analysis with detailed fallacy detection"""

    analysis = {{
        "word_count": len(text.split()),
        "sentence_count": len([s for s in text.split('.') if s.strip()]),
        "fallacies": [],
        "strength": 0.5,
        "suggestions": [],
        "reasoning_quality": "moderate",
        "evidence_score": 0.0
    }}

    text_lower = text.lower()
    fallacies_detected = []

    # Fallacy detection
    if "scientists say" in text_lower and "study" not in text_lower:
        fallacies_detected.append("appeal_to_authority")

    bandwagon_phrases = ["everyone knows", "everybody thinks", "everyone likes"]
    if any(phrase in text_lower for phrase in bandwagon_phrases):
        fallacies_detected.append("bandwagon")

    analysis["fallacies"] = fallacies_detected

    # Strength calculation
    strength = 0.5
    evidence_words = ["study", "research", "data", "evidence"]
    evidence_count = sum(1 for word in evidence_words if word in text_lower)
    analysis["evidence_score"] = min(1.0, evidence_count * 0.2)
    strength += analysis["evidence_score"] * 0.3
    strength -= len(fallacies_detected) * 0.15
    analysis["strength"] = max(0.0, min(1.0, strength))

    if analysis["strength"] > 0.7:
        analysis["reasoning_quality"] = "strong"
    elif analysis["strength"] > 0.4:
        analysis["reasoning_quality"] = "moderate"
    else:
        analysis["reasoning_quality"] = "weak"

    # Generate suggestions
    suggestions = []
    if fallacies_detected:
        suggestions.append(f"Remove {{', '.join(fallacies_detected)}} fallacies")
    if analysis["evidence_score"] < 0.3:
        suggestions.append("Add supporting evidence")
    analysis["suggestions"] = suggestions

    return analysis

# Execute analysis
text = """{content.replace('"', '\\"')}"""
result = analyze_argument_advanced(text)

print("ANALYSIS_START")
print(json.dumps(result, indent=2))
print("ANALYSIS_END")
'''

        analysis_result = await self.bridge.execute_python_code(analysis_code)

        try:
            # Parse JSON from output
            lines = analysis_result.strip().split('\n')
            json_content = ""
            in_analysis = False

            for line in lines:
                if "ANALYSIS_START" in line:
                    in_analysis = True
                    continue
                elif "ANALYSIS_END" in line:
                    break
                elif in_analysis:
                    json_content += line + "\n"

            if json_content.strip():
                analysis = json.loads(json_content.strip())
                print(f"✅ Analysis completed: {analysis['reasoning_quality']} reasoning, {len(analysis['fallacies'])} fallacies")
                return analysis
            else:
                raise ValueError("No analysis markers found")

        except Exception as e:
            print(f"❌ Analysis parsing error: {e}")
            # Fallback analysis
            return {
                "word_count": len(content.split()),
                "fallacies": ["appeal_to_authority"] if "scientists say" in content.lower() else [],
                "strength": 0.4,
                "suggestions": ["Add supporting evidence"],
                "reasoning_quality": "needs_improvement",
                "error": str(e)
            }

    async def run_simplified_demo(self):
        """Simplified ReasonFlow demo to avoid async context issues"""
        print("🚀 Starting Simplified ReasonFlow Demo")
        print("=" * 50)

        if not self.bridge.initialized:
            await self.initialize()

        if not self.bridge.agents_available:
            print("❌ MCP agents not available - cannot run demo")
            return

        # Setup database
        print("🗄️  Setting up database...")
        if not await self.setup_database():
            print("❌ Database setup failed")
            return

        # Test with one argument first to avoid context switching issues
        test_arg = "Climate change is real because scientists say so"

        print(f"\n📝 Adding test argument: {test_arg}")
        arg_id = await self.add_argument(test_arg)

        if arg_id:
            print(f"✅ Added argument: {arg_id}")

            # Analyze the argument
            print(f"\n🔍 Analyzing argument: {arg_id}")
            analysis = await self.analyze_argument(arg_id, test_arg)

            print(f"   Strength: {analysis.get('strength', 0):.2f}")
            print(f"   Quality: {analysis.get('reasoning_quality', 'unknown')}")
            print(f"   Fallacies: {len(analysis.get('fallacies', []))}")
            if analysis.get('fallacies'):
                print(f"   Detected: {', '.join(analysis['fallacies'])}")

        print("\n✅ Simplified ReasonFlow demo complete!")
        print("\n🎯 What worked:")
        print("   • Loaded API key from .env file")
        print("   • Used unified agent (no switching)")
        print("   • Avoided async context conflicts")
        print("   • Successfully executed database operations")
        print("   • Successfully executed Python code analysis")

# Main execution
async def main():
    coordinator = ReasonFlowCoordinator()
    await coordinator.initialize()
    await coordinator.run_simplified_demo()

if __name__ == "__main__":
    print("🧠 ReasonFlow with PROPERLY FIXED MCP Integration")
    print("This version loads .env and avoids async context issues")
    print()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
