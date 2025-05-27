#!/usr/bin/env python3
"""
ReasonFlow MCP Integration - SIMPLE VERSION
Much simpler approach that just works without complex async initialization
"""

import asyncio
import json
import sys

# Add your agent path
sys.path.append('/home/ty/Repositories/ai_workspace/gemini-mcp-client')

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
    AGENTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Could not import MCP agents or ADK components: {e}")
    AGENTS_AVAILABLE = False

class SimpleMCPBridge:
    """
    Simple bridge that just works - no complex async initialization
    """

    def __init__(self):
        self.agents_available = AGENTS_AVAILABLE
        self.setup_done = False

        # Database path for ReasonFlow
        self.reasonflow_db_path = "/home/ty/Repositories/ai_workspace/gemini-mcp-client/agents/reasonflow_mvp/reasonflow.db"

        # Response cache
        self.response_cache = {}
        print(f"🗄️  ReasonFlow database will be at: {self.reasonflow_db_path}")

    def _ensure_setup(self):
        """Simple one-time setup - called automatically when needed"""
        if self.setup_done or not self.agents_available:
            return

        try:
            self.sqlite_agent = sqlite_agent
            self.code_executor_agent = code_executor_agent
            self.arxiv_agent = arxiv_agent
            self.multi_agent = multi_mcp_agent

            # Set up ADK services
            self.session_service = InMemorySessionService()
            self.artifact_service = InMemoryArtifactService()

            # Create session
            self.session = self.session_service.create_session(
                app_name="ReasonFlow",
                user_id="reasonflow_user",
            )
            self.user_id = "reasonflow_user"
            self.session_id = self.session.id

            # Create runner
            self.runner = Runner(
                app_name="ReasonFlow",
                agent=self.multi_agent,
                artifact_service=self.artifact_service,
                session_service=self.session_service,
            )

            self.setup_done = True
            print("✅ MCP Bridge setup complete")

        except Exception as e:
            print(f"❌ Setup failed: {e}")
            self.agents_available = False

    def _run_agent(self, agent, query: str) -> str:
        """Run an agent and get response - simple and reliable"""

        self._ensure_setup()  # Auto-setup if needed

        if not self.agents_available:
            return "ERROR: MCP agents not available"

        try:
            # Set the agent
            self.runner.agent = agent

            # Create content
            content = types.Content(
                role="user",
                parts=[types.Part(text=query)]
            )

            # Run and get events
            events = list(
                self.runner.run(
                    user_id=self.user_id,
                    session_id=self.session_id,
                    new_message=content
                )
            )

            # Extract response
            if events:
                last_event = events[-1]
                if hasattr(last_event, 'content') and last_event.content and last_event.content.parts:
                    parts = last_event.content.parts
                    response = "".join([part.text for part in parts if hasattr(part, 'text') and part.text])
                    return response
                else:
                    return "No readable content in response"
            else:
                return "No response received from agent"

        except Exception as e:
            return f"Agent execution error: {str(e)}"

    async def execute_sql_query(self, query: str) -> str:
        """Execute SQL query - simple interface"""

        message = f"""
        I need to work with a ReasonFlow database at {self.reasonflow_db_path}.
        Please execute this SQL query:

        {query}

        If the database doesn't exist yet, please create it first.
        Return just the result of the SQL operation.
        """

        print("🗄️  Executing SQL...")
        result = self._run_agent(self.sqlite_agent, message)
        print(f"📊 SQL done")
        return result

    async def execute_python_code(self, code: str) -> str:
        """Execute Python code - simple interface"""

        message = f"""
        Please execute this Python code and return the output:

        ```python
        {code}
        ```

        Show me the complete output including any print statements.
        """

        print("🐍 Executing Python code...")
        result = self._run_agent(self.code_executor_agent, message)
        print("⚡ Code execution done")
        return result

    async def search_arxiv(self, query: str, max_results: int = 3) -> str:
        """Search ArXiv - simple interface with caching"""

        cache_key = f"arxiv_{hash(query)}"
        if cache_key in self.response_cache:
            print("📚 Using cached ArXiv result")
            return self.response_cache[cache_key]

        message = f"""
        Search ArXiv for papers related to: {query}

        Please find up to {max_results} relevant papers and provide:
        - Paper titles
        - Brief summaries
        - How they relate to the search topic
        """

        print("📚 Searching ArXiv...")
        result = self._run_agent(self.arxiv_agent, message)

        # Cache the result
        self.response_cache[cache_key] = result
        print("🔍 ArXiv search done")
        return result

class SimpleReasonFlowCoordinator:
    """
    Simple coordinator - no complex initialization needed
    """

    def __init__(self):
        self.bridge = SimpleMCPBridge()
        self.db_initialized = False
        self.arguments_created = 0

        print("🧠 ReasonFlow Coordinator ready")

    async def setup_database(self) -> bool:
        """Setup database"""

        print("🗄️  Setting up ReasonFlow database...")

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
        """

        result = await self.bridge.execute_sql_query(schema_sql)

        # Simple success check
        success_words = ["success", "created", "table", "completed", "ok", "done"]
        self.db_initialized = any(word in result.lower() for word in success_words)

        if self.db_initialized:
            print("✅ Database setup successful")
        else:
            print(f"⚠️  Database setup result: {result}")

        return self.db_initialized

    async def add_argument(self, content: str, arg_type: str = "claim") -> str:
        """Add argument to database"""

        import random
        arg_id = f"rf_arg_{random.randint(10000, 99999)}"

        # Escape quotes
        safe_content = content.replace("'", "''")

        insert_sql = f"""
        INSERT INTO rf_arguments (id, content, type, fitness_score, generation)
        VALUES ('{arg_id}', '{safe_content}', '{arg_type}', 0.0, 0);
        """

        result = await self.bridge.execute_sql_query(insert_sql)

        # Simple success check
        if any(word in result.lower() for word in ["success", "inserted", "ok", "done"]):
            print(f"✅ Added argument: {arg_id}")
            self.arguments_created += 1
            return arg_id
        else:
            print(f"❌ Add failed: {result}")
            return ""

    async def analyze_argument(self, arg_id: str, content: str) -> dict:
        """Analyze argument"""

        print(f"🔍 Analyzing: {arg_id}")

        # Simple analysis code with clear output
        analysis_code = f'''
import json

def analyze_text(text):
    analysis = {{
        "word_count": len(text.split()),
        "fallacies": [],
        "strength": 0.5,
        "suggestions": []
    }}

    text_lower = text.lower()

    # Check for fallacies
    if any(word in text_lower for word in ["stupid", "idiot", "moron"]):
        analysis["fallacies"].append("ad_hominem")

    if "scientists say" in text_lower and "study" not in text_lower:
        analysis["fallacies"].append("appeal_to_authority")

    if any(phrase in text_lower for phrase in ["everyone knows", "everybody thinks"]):
        analysis["fallacies"].append("bandwagon")

    # Calculate strength
    strength = 0.5

    # Add for evidence
    evidence_words = ["study", "research", "data", "evidence"]
    evidence_count = sum(1 for word in evidence_words if word in text_lower)
    strength += evidence_count * 0.15

    # Subtract for fallacies
    strength -= len(analysis["fallacies"]) * 0.2

    analysis["strength"] = max(0.0, min(1.0, strength))

    # Generate suggestions
    if analysis["fallacies"]:
        analysis["suggestions"].append("Remove logical fallacies")
    if evidence_count == 0:
        analysis["suggestions"].append("Add supporting evidence")
    if len(text.split()) < 15:
        analysis["suggestions"].append("Provide more detail")

    return analysis

# Run analysis
text = """{content.replace('"', '\\"')}"""
result = analyze_text(text)

print("=== ANALYSIS RESULT ===")
print(json.dumps(result, indent=2))
print("=== END RESULT ===")
'''

        analysis_result = await self.bridge.execute_python_code(analysis_code)

        try:
            # Find JSON in the output
            lines = analysis_result.split('\n')
            json_content = ""
            capturing = False

            for line in lines:
                if "=== ANALYSIS RESULT ===" in line:
                    capturing = True
                    continue
                elif "=== END RESULT ===" in line:
                    break
                elif capturing:
                    json_content += line + "\n"

            if json_content.strip():
                analysis = json.loads(json_content.strip())
                print(f"✅ Analysis done: {analysis.get('strength', 0):.2f} strength, {len(analysis.get('fallacies', []))} fallacies")
                return analysis
            else:
                raise ValueError("No JSON found")

        except Exception as e:
            print(f"❌ Analysis parsing failed: {e}")
            # Fallback analysis
            return {
                "word_count": len(content.split()),
                "fallacies": ["appeal_to_authority"] if "scientists say" in content.lower() else [],
                "strength": 0.4,
                "suggestions": ["Add evidence"]
            }

    async def evolve_argument(self, arg_id: str, content: str, analysis: dict) -> str:
        """Evolve argument to make it better"""

        print(f"🧬 Evolving: {arg_id}")

        evolution_code = f'''
import json

def evolve_text(original, analysis):
    evolved = original
    changes = []

    fallacies = analysis.get("fallacies", [])
    suggestions = analysis.get("suggestions", [])

    # Fix fallacies
    if "ad_hominem" in fallacies:
        for bad_word in ["stupid", "idiot", "moron"]:
            if bad_word in evolved.lower():
                evolved = evolved.replace(bad_word, "problematic")
                changes.append("removed ad hominem")

    if "appeal_to_authority" in fallacies:
        evolved = evolved.replace("scientists say", "research shows")
        changes.append("improved authority reference")

    if "bandwagon" in fallacies:
        evolved = evolved.replace("everyone knows", "evidence suggests")
        changes.append("removed bandwagon argument")

    # Add evidence if needed
    if "Add supporting evidence" in suggestions:
        evolved += " This is supported by peer-reviewed research."
        changes.append("added evidence")

    # Add detail if needed
    if "Provide more detail" in suggestions:
        evolved += " The reasoning involves multiple factors that support this conclusion."
        changes.append("added detail")

    if not changes:
        evolved += " [Enhanced version]"
        changes.append("general enhancement")

    return {{
        "evolved_text": evolved,
        "changes": changes
    }}

# Run evolution
original = """{content.replace('"', '\\"')}"""
analysis = {json.dumps(analysis)}

result = evolve_text(original, analysis)

print("=== EVOLUTION RESULT ===")
print(json.dumps(result, indent=2))
print("=== END RESULT ===")
'''

        evolution_result = await self.bridge.execute_python_code(evolution_code)

        try:
            # Extract JSON
            lines = evolution_result.split('\n')
            json_content = ""
            capturing = False

            for line in lines:
                if "=== EVOLUTION RESULT ===" in line:
                    capturing = True
                    continue
                elif "=== END RESULT ===" in line:
                    break
                elif capturing:
                    json_content += line + "\n"

            if json_content.strip():
                evolution_data = json.loads(json_content.strip())
                evolved_content = evolution_data.get("evolved_text", content + " [Failed]")
                changes = evolution_data.get("changes", [])

                print(f"✅ Evolution done: {', '.join(changes)}")

                # Save evolved argument
                new_arg_id = await self.add_argument(evolved_content, "evolved_claim")
                return new_arg_id
            else:
                raise ValueError("No evolution JSON found")

        except Exception as e:
            print(f"❌ Evolution failed: {e}")
            # Fallback
            simple_evolved = content + " [Enhanced with evidence]"
            new_arg_id = await self.add_argument(simple_evolved, "evolved_claim")
            return new_arg_id if new_arg_id else ""

    async def validate_with_research(self, content: str) -> dict:
        """Validate with academic papers"""

        print("🔬 Validating with research...")

        # Get key terms
        words = content.split()[:3]
        search_terms = " ".join(words)

        papers_result = await self.bridge.search_arxiv(search_terms, max_results=2)

        # Simple check for academic support
        has_support = any(word in papers_result.lower() for word in ["paper", "study", "research"])

        return {
            "search_terms": search_terms,
            "papers_found": papers_result,
            "academic_support": has_support
        }

    async def run_evolution_demo(self):
        """Run the complete demo"""

        print("🚀 Starting Simple ReasonFlow Demo")
        print("=" * 50)

        if not self.bridge.agents_available:
            print("❌ MCP agents not available")
            return

        # Setup database
        print("🗄️  Setting up database...")
        if not await self.setup_database():
            print("❌ Database setup failed")
            return

        # Test arguments
        test_args = [
            "Climate change is real because scientists say so",
            "Vaccines are dangerous because they cause autism and everyone knows this",
            "Democracy is the best system because everyone likes it"
        ]

        arg_data = []
        print("\n📝 Adding test arguments...")
        for i, arg in enumerate(test_args):
            arg_id = await self.add_argument(arg)
            if arg_id:
                arg_data.append((arg_id, arg))
                print(f"   {i+1}. {arg_id}: {arg[:50]}...")

        # Analyze arguments
        print("\n🔍 Analyzing arguments...")
        analyses = []
        for arg_id, content in arg_data:
            analysis = await self.analyze_argument(arg_id, content)
            analyses.append((arg_id, content, analysis))

            print(f"   {arg_id}: Strength={analysis.get('strength', 0):.2f}, Fallacies={len(analysis.get('fallacies', []))}")

        # Evolve weak arguments
        print("\n🧬 Evolving weak arguments...")
        evolution_results = []
        for arg_id, content, analysis in analyses[:2]:  # First 2
            if analysis.get('strength', 1.0) < 0.8:
                new_id = await self.evolve_argument(arg_id, content, analysis)
                if new_id:
                    evolution_results.append((arg_id, new_id))
                    print(f"   {arg_id} → {new_id}")

        # Academic validation
        print("\n🔬 Academic validation...")
        if arg_data:
            validation = await self.validate_with_research(arg_data[0][1])
            print(f"   Search: {validation.get('search_terms')}")
            print(f"   Support: {validation.get('academic_support', False)}")

        # Final stats
        print("\n📊 Final stats...")
        print(f"   Arguments processed: {len(arg_data)}")
        print(f"   Arguments evolved: {len(evolution_results)}")
        print(f"   Database location: {self.bridge.reasonflow_db_path}")

        print("\n✅ Demo complete!")

# Simple main function
async def main():
    coordinator = SimpleReasonFlowCoordinator()
    await coordinator.run_evolution_demo()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ Error: {e}")
