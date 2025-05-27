#!/usr/bin/env python3
"""
ReasonFlow MCP Integration Bridge
Connects the A2A coordinator to your existing MCP agents
"""

import sys
import os
import asyncio
import json
from typing import Dict, Any

# Add your agent path
sys.path.append('/home/ty/Repositories/ai_workspace/gemini-mcp-client/agents/multi_mcp_agent')

try:
    from agent import (
        agent as multi_agent,
        sqlite_agent,
        code_executor_agent,
        arxiv_agent
    )
except ImportError as e:
    print(f"Could not import agents: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)

class MCPAgentBridge:
    """
    Bridge between ReasonFlow coordinator and your MCP agents
    Handles the communication protocol and response parsing
    """

    def __init__(self):
        self.sqlite = sqlite_agent
        self.code_executor = code_executor_agent
        self.arxiv = arxiv_agent
        self.multi = multi_agent

        # Response cache to minimize repeated calls
        self.response_cache = {}

    async def execute_sql_query(self, query: str) -> str:
        """Execute SQL via SQLite agent"""

        cache_key = f"sql_{hash(query)}"
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]

        try:
            # Construct proper message for SQLite agent
            message = f"Execute this SQL query: {query}"

            # This would use your agent's actual interface
            # Adapting to whatever message format your agents expect
            response = await self._call_agent(self.sqlite, message)

            self.response_cache[cache_key] = response
            return response

        except Exception as e:
            return f"SQL Error: {str(e)}"

    async def execute_python_code(self, code: str) -> str:
        """Execute Python code via Code Executor agent"""

        try:
            message = f"Execute this Python code:\n\n```python\n{code}\n```"
            response = await self._call_agent(self.code_executor, message)
            return response

        except Exception as e:
            return f"Code execution error: {str(e)}"

    async def search_arxiv(self, query: str, max_results: int = 3) -> str:
        """Search ArXiv via ArXiv agent"""

        cache_key = f"arxiv_{hash(query)}"
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]

        try:
            message = f"Search ArXiv for papers about: {query}. Limit to {max_results} results."
            response = await self._call_agent(self.arxiv, message)

            self.response_cache[cache_key] = response
            return response

        except Exception as e:
            return f"ArXiv search error: {str(e)}"

    async def _call_agent(self, agent, message: str) -> str:
        """
        Generic agent caller - adapt this to your agent interface
        This is a placeholder - you'll need to implement based on your actual agent API
        """

        # Your agents might use different calling conventions
        # This is a generic example - adapt to your actual interface

        try:
            if hasattr(agent, 'send_message'):
                response = await agent.send_message(message)
            elif hasattr(agent, 'query'):
                response = await agent.query(message)
            elif hasattr(agent, 'run'):
                response = await agent.run(message)
            else:
                # Fallback - try direct call
                response = await agent(message)

            return str(response)

        except Exception as e:
            return f"Agent call failed: {str(e)}"

class ReasonFlowMCPCoordinator:
    """
    ReasonFlow coordinator that uses your MCP agents
    """

    def __init__(self):
        self.bridge = MCPAgentBridge()
        self.db_initialized = False

    async def setup_database(self) -> bool:
        """Initialize ReasonFlow database schema"""

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
        self.db_initialized = "error" not in result.lower()

        if self.db_initialized:
            print("✅ ReasonFlow database initialized successfully")
        else:
            print(f"❌ Database initialization failed: {result}")

        return self.db_initialized

    async def add_argument(self, content: str, arg_type: str = "claim") -> str:
        """Add new argument to database"""

        import random
        arg_id = f"rf_arg_{random.randint(10000, 99999)}"

        # Escape single quotes for SQL
        safe_content = content.replace("'", "''")

        insert_sql = f"""
        INSERT INTO rf_arguments (id, content, type, fitness_score, generation)
        VALUES ('{arg_id}', '{safe_content}', '{arg_type}', 0.0, 0);
        """

        result = await self.bridge.execute_sql_query(insert_sql)

        if "error" not in result.lower():
            print(f"✅ Added argument: {arg_id}")
            return arg_id
        else:
            print(f"❌ Failed to add argument: {result}")
            return None

    async def analyze_argument(self, arg_id: str) -> Dict:
        """Analyze argument using local Python execution"""

        # First, get the argument content
        query_sql = f"SELECT content FROM rf_arguments WHERE id = '{arg_id}';"
        result = await self.bridge.execute_sql_query(query_sql)

        if "error" in result.lower():
            return {"error": "Could not retrieve argument"}

        # Extract content from SQL result (this parsing would need to be adjusted)
        # For demo purposes, assuming we can extract the content
        content = "extracted_content"  # Would parse from SQL result

        # Local analysis code
        analysis_code = f'''
import json
import re

def analyze_text(text):
    """Local argument analysis"""

    analysis = {{
        "word_count": len(text.split()),
        "sentence_count": len([s for s in text.split('.') if s.strip()]),
        "fallacies": [],
        "strength": 0.5,
        "suggestions": []
    }}

    text_lower = text.lower()

    # Basic fallacy detection
    if any(word in text_lower for word in ["stupid", "idiot", "moron"]):
        analysis["fallacies"].append("ad_hominem")

    if "because everyone" in text_lower or "everybody knows" in text_lower:
        analysis["fallacies"].append("bandwagon")

    if "scientists say" in text_lower and "study" not in text_lower:
        analysis["fallacies"].append("appeal_to_authority")

    # Strength calculation
    strength = 0.5

    # Boost for evidence words
    evidence_words = ["data", "study", "research", "evidence", "statistics", "analysis"]
    evidence_count = sum(1 for word in evidence_words if word in text_lower)
    strength += evidence_count * 0.1

    # Reduce for uncertainty words
    uncertainty_words = ["maybe", "possibly", "might", "could"]
    uncertainty_count = sum(1 for word in uncertainty_words if word in text_lower)
    strength -= uncertainty_count * 0.05

    # Reduce for fallacies
    strength -= len(analysis["fallacies"]) * 0.15

    analysis["strength"] = max(0.0, min(1.0, strength))

    # Generate suggestions
    if len(analysis["fallacies"]) > 0:
        analysis["suggestions"].append("Remove logical fallacies")

    if evidence_count == 0:
        analysis["suggestions"].append("Add supporting evidence")

    if analysis["word_count"] < 15:
        analysis["suggestions"].append("Provide more detailed reasoning")

    return analysis

# Execute analysis
text = "{content}"
result = analyze_text(text)
print(json.dumps(result, indent=2))
'''

        # Execute analysis
        analysis_result = await self.bridge.execute_python_code(analysis_code)

        try:
            # Parse JSON from code output
            analysis = json.loads(analysis_result.split('\n')[-2])  # Get JSON line

            # Update database with analysis
            update_sql = f"""
            UPDATE rf_arguments
            SET analysis_cache = '{json.dumps(analysis).replace("'", "''")}',
                fitness_score = {analysis.get('strength', 0.5)}
            WHERE id = '{arg_id}';
            """

            await self.bridge.execute_sql_query(update_sql)

            return analysis

        except Exception as e:
            return {"error": f"Analysis parsing failed: {str(e)}"}

    async def evolve_argument(self, arg_id: str) -> str:
        """Evolve an argument using local computation"""

        # Get argument and its analysis
        query_sql = f"""
        SELECT content, analysis_cache, fitness_score
        FROM rf_arguments
        WHERE id = '{arg_id}';
        """

        result = await self.bridge.execute_sql_query(query_sql)

        if "error" in result.lower():
            return None

        # For demo - in practice you'd parse the SQL result properly
        content = "original_content"
        analysis = {"suggestions": ["Add evidence"], "strength": 0.4}

        # Evolution code
        evolution_code = f'''
import json
import random

def evolve_argument(original, analysis):
    """Generate improved version of argument"""

    suggestions = analysis.get("suggestions", [])
    strength = analysis.get("strength", 0.5)

    evolved = original

    # Apply improvements based on analysis
    if "Add supporting evidence" in suggestions:
        evolved += " This is supported by peer-reviewed research and empirical data."

    if "Remove logical fallacies" in suggestions:
        # Simple fallacy removal
        fallacy_words = ["stupid", "idiot", "everyone knows", "everybody"]
        for word in fallacy_words:
            evolved = evolved.replace(word, "[removed]")

    if "more detailed reasoning" in suggestions:
        evolved += " The logical foundation for this claim rests on established principles and verifiable observations."

    # Ensure some change occurred
    if evolved == original:
        evolved += " [Evolutionarily enhanced version]"

    return evolved

# Execute evolution
original = "{content}"
analysis = {json.dumps(analysis)}

evolved = evolve_argument(original, analysis)
print(evolved)
'''

        # Execute evolution
        evolved_content = await self.bridge.execute_python_code(evolution_code)

        # Create new argument with evolved content
        new_arg_id = await self.add_argument(evolved_content, "evolved_claim")

        if new_arg_id:
            # Log the evolution
            import random
            log_id = f"evo_{random.randint(10000, 99999)}"
            log_sql = f"""
            INSERT INTO rf_evolution_log (id, parent_id, child_id, operation_type, improvement_score)
            VALUES ('{log_id}', '{arg_id}', '{new_arg_id}', 'evolution', 0.1);
            """

            await self.bridge.execute_sql_query(log_sql)
            print(f"✅ Evolved {arg_id} → {new_arg_id}")

            return new_arg_id

        return None

    async def validate_with_research(self, arg_id: str) -> Dict:
        """Validate argument against academic literature"""

        # Get argument content
        query_sql = f"SELECT content FROM rf_arguments WHERE id = '{arg_id}';"
        result = await self.bridge.execute_sql_query(query_sql)

        # Extract key terms for research (simplified)
        search_terms = "climate change"  # Would extract from content

        # Search ArXiv
        papers = await self.bridge.search_arxiv(search_terms, max_results=2)

        return {
            "validation_attempted": True,
            "search_terms": search_terms,
            "papers_found": papers,
            "academic_support": "paper" in papers.lower()
        }

    async def run_evolution_demo(self):
        """Demo the full ReasonFlow pipeline"""

        print("🚀 Starting ReasonFlow MCP Demo")
        print("=" * 50)

        # Setup
        if not await self.setup_database():
            return

        # Add test arguments
        test_args = [
            "Climate change is real because scientists say so",
            "Vaccines are dangerous because they cause autism",
            "Democracy is the best system because everyone likes it"
        ]

        arg_ids = []
        print("\n📝 Adding test arguments...")
        for arg in test_args:
            arg_id = await self.add_argument(arg)
            if arg_id:
                arg_ids.append(arg_id)

        # Analyze arguments
        print("\n🔍 Analyzing arguments...")
        for arg_id in arg_ids:
            analysis = await self.analyze_argument(arg_id)
            print(f"  {arg_id}: Strength={analysis.get('strength', 0):.2f}, "
                  f"Fallacies={len(analysis.get('fallacies', []))}")

        # Evolve arguments
        print("\n🧬 Evolving arguments...")
        for arg_id in arg_ids[:2]:  # Just evolve first 2 to save compute
            new_id = await self.evolve_argument(arg_id)
            if new_id:
                # Analyze evolved version
                new_analysis = await self.analyze_argument(new_id)
                print(f"  {arg_id} → {new_id}: New strength={new_analysis.get('strength', 0):.2f}")

        # Research validation (sparingly)
        print("\n🔬 Research validation...")
        if arg_ids:
            validation = await self.validate_with_research(arg_ids[0])
            print(f"  Validation result: {validation.get('academic_support', False)}")

        # Final stats
        stats_sql = "SELECT COUNT(*) as total_args FROM rf_arguments;"
        stats = await self.bridge.execute_sql_query(stats_sql)
        print(f"\n📊 Final stats: {stats}")

        print("\n✅ ReasonFlow demo complete!")

# Quick test runner
async def main():
    coordinator = ReasonFlowMCPCoordinator()
    await coordinator.run_evolution_demo()

if __name__ == "__main__":
    print("ReasonFlow MCP Integration Bridge")
    print("Make sure your MCP agents are running...")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")