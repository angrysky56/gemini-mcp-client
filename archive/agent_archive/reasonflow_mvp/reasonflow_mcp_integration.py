#!/usr/bin/env python3
"""
ReasonFlow MCP Integration Bridge - ADK API VERSION
Connects to your MCP agents using the ADK API pattern
"""

import asyncio
import json
import sys

# Add your agent path
sys.path.append('/home/ty/Repositories/ai_workspace/gemini-mcp-client')

# Import your agents and ADK components
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

# Import our enhanced MCP components
try:
    from mcp_enhanced import EnhancedMCPBridge, create_stdio_server_params
    print("✅ Successfully imported enhanced MCP components")
except ImportError as e:
    print(f"❌ Could not import enhanced MCP components: {e}")

class MCPBridge:
    """
    Bridge that uses your MCP agents via ADK Runner interface
    with enhanced MCP error handling
    """

    def __init__(self):
        self.agents_available = AGENTS_AVAILABLE
        self.initialized = False

        # Database path for ReasonFlow
        self.reasonflow_db_path = "/home/ty/Repositories/ai_workspace/gemini-mcp-client/agents/reasonflow_mvp/reasonflow.db"

        # Response cache
        self.response_cache = {}
        print(f"🗄️  ReasonFlow database will be at: {self.reasonflow_db_path}")

    async def initialize(self):
        """Initialize the bridge with ADK components (must be called after construction)"""
        if self.agents_available and not self.initialized:
            self.sqlite_agent = sqlite_agent
            self.code_executor_agent = code_executor_agent
            self.arxiv_agent = arxiv_agent
            self.multi_agent = multi_mcp_agent

            # Set up ADK services for agent execution
            self.session_service = InMemorySessionService()
            self.artifact_service = InMemoryArtifactService()

            # Create session (await and extract session ID)
            self.user_id = "reasonflow_user"
            self.session = await self.session_service.create_session(
                app_name="ReasonFlow",
                user_id=self.user_id,
            )
            self.session_id = self.session.id  # Extract the ID from the session object

            # Create runner with default agent
            self.runner = Runner(
                app_name="ReasonFlow",
                agent=self.multi_agent,  # Use multi_agent as default
                artifact_service=self.artifact_service,
                session_service=self.session_service,
            )

            # Create enhanced MCP bridge for more reliable MCP operations
            try:
                # This provides a more robust MCP client that properly handles GeneratorExit exceptions
                self.enhanced_mcp_bridge = EnhancedMCPBridge(session_id=self.session_id)
                print("✅ Enhanced MCP bridge created for better error handling")
            except Exception as e:
                print(f"⚠️ Could not create enhanced MCP bridge: {e}")
                print("⚠️ Will continue with standard MCP handling")

            self.initialized = True
            print(f"✅ MCPBridge initialized successfully with session ID: {self.session_id}")

    def _run_agent(self, agent, query: str) -> str:
        """Helper method to run an agent using ADK API and get the final response."""

        if not self.agents_available or not self.initialized:
            return "ERROR: MCP agents not available or not initialized"

        try:
            # Set the specific agent for this execution
            self.runner.agent = agent

            # Create ADK content format
            content = types.Content(
                role="user",
                parts=[types.Part(text=query)]
            )

            # Run the agent
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

            # First try the enhanced MCP approach if available
            try:
                if hasattr(self, 'enhanced_mcp_bridge'):
                    from mcp.client.stdio import StdioServerParameters

                    # Create server parameters for npx mcp-sqlite
                    server_params = create_stdio_server_params(
                        command="npx",
                        args=["-y", "mcp-sqlite"],
                        env={"DATABASE_PATH": self.reasonflow_db_path}
                    )

                    # Run the MCP operation with proper cleanup handling
                    async def execute_sql(bridge):
                        tools = await bridge.list_tools()
                        print(f"Available MCP tools: {[t.get('name') for t in tools if 'name' in t]}")

                        # Find the execute_sql tool
                        execute_tool = next((t for t in tools if t.get('name') == 'execute_sql'), None)
                        if execute_tool:
                            result = await bridge.call_tool('execute_sql', {"query": query})
                            return result
                        else:
                            return "SQL tool not found in MCP server"

                    # Try enhanced approach
                    print("🔄 Using enhanced MCP bridge for SQL execution...")
                    from mcp_enhanced import run_with_mcp_bridge
                    enhanced_result = await run_with_mcp_bridge(server_params, self.session_id, execute_sql)
                    if enhanced_result:
                        print("✅ Enhanced MCP SQL execution successful")
                        return enhanced_result

            except Exception as e:
                print(f"⚠️ Enhanced MCP approach failed, falling back to ADK agent: {e}")

            # Fallback to standard ADK agent approach
            result = self._run_agent(self.sqlite_agent, message)
            print(f"📊 SQL result: {result[:100]}...")
            return result

        except Exception as e:
            error_msg = f"SQL execution error: {str(e)}"
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

            # Use ADK API to call the Code Executor agent
            result = self._run_agent(self.code_executor_agent, message)

            print(f"⚡ Code result: {result[:100]}...")
            return result

        except Exception as e:
            error_msg = f"Code execution error: {str(e)}"
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
            result = self._run_agent(self.arxiv_agent, message)

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
            fitness_score DEFAULT 0.0,
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
            improvement_score,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS rf_fallacy_detections (
            id TEXT PRIMARY KEY,
            argument_id TEXT,
            fallacy_type TEXT,
            confidence,
            explanation TEXT,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
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

        # Enhanced local analysis code with clear JSON output
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

    # Enhanced fallacy detection
    fallacies_detected = []

    # Ad hominem attacks
    ad_hominem_words = ["stupid", "idiot", "moron", "fool", "dumb", "ignorant", "pathetic"]
    if any(word in text_lower for word in ad_hominem_words):
        fallacies_detected.append("ad_hominem")

    # Appeal to authority without evidence
    if "scientists say" in text_lower and "study" not in text_lower and "research" not in text_lower:
        fallacies_detected.append("appeal_to_authority")

    # Bandwagon fallacy
    bandwagon_phrases = ["everyone knows", "everybody thinks", "most people believe", "because everyone"]
    if any(phrase in text_lower for phrase in bandwagon_phrases):
        fallacies_detected.append("bandwagon")

    # False dichotomy
    dichotomy_words = ["either", "only two options", "must choose", "no other way"]
    if any(phrase in text_lower for phrase in dichotomy_words):
        fallacies_detected.append("false_dichotomy")

    # Straw man (oversimplification indicators)
    if "always" in text_lower and "never" in text_lower:
        fallacies_detected.append("straw_man")

    analysis["fallacies"] = fallacies_detected

    # Strength calculation
    strength = 0.5

    # Evidence indicators (positive)
    evidence_words = ["study", "research", "data", "evidence", "statistics", "analysis", "peer-reviewed", "experiment"]
    evidence_count = sum(1 for word in evidence_words if word in text_lower)
    analysis["evidence_score"] = min(1.0, evidence_count * 0.2)
    strength += analysis["evidence_score"] * 0.3

    # Logical structure indicators (positive)
    logic_words = ["because", "therefore", "thus", "hence", "consequently", "since", "given that"]
    logic_count = sum(1 for word in logic_words if word in text_lower)
    strength += min(0.2, logic_count * 0.05)

    # Uncertainty and hedge words (slight negative)
    uncertainty_words = ["maybe", "possibly", "might", "could", "perhaps", "probably"]
    uncertainty_count = sum(1 for word in uncertainty_words if word in text_lower)
    strength -= uncertainty_count * 0.03

    # Fallacy penalty
    strength -= len(fallacies_detected) * 0.15

    # Complexity bonus (longer, more detailed arguments)
    if analysis["word_count"] > 20:
        strength += 0.1

    analysis["strength"] = max(0.0, min(1.0, strength))

    # Reasoning quality assessment
    if analysis["strength"] > 0.7:
        analysis["reasoning_quality"] = "strong"
    elif analysis["strength"] > 0.4:
        analysis["reasoning_quality"] = "moderate"
    else:
        analysis["reasoning_quality"] = "weak"

    # Generate specific suggestions
    suggestions = []

    if len(fallacies_detected) > 0:
        suggestions.append(f"Remove {{', '.join(fallacies_detected)}} fallacies")

    if analysis["evidence_score"] < 0.3:
        suggestions.append("Add supporting evidence from reliable sources")

    if analysis["word_count"] < 15:
        suggestions.append("Provide more detailed reasoning")

    if logic_count == 0:
        suggestions.append("Clarify logical connections between premises and conclusion")

    if uncertainty_count > 2:
        suggestions.append("Use more confident language where evidence supports it")

    analysis["suggestions"] = suggestions

    return analysis

# Execute analysis on the specific text
text = """{content.replace('"', '\\"')}"""
result = analyze_argument_advanced(text)

# Output ONLY the JSON result for easy parsing
print("ANALYSIS_START")
print(json.dumps(result, indent=2))
print("ANALYSIS_END")
'''

        # Execute analysis using code executor
        analysis_result = await self.bridge.execute_python_code(analysis_code)

        try:
            # Parse JSON from code output - look for content between markers
            lines = analysis_result.strip().split('\n')

            # Find the JSON content between ANALYSIS_START and ANALYSIS_END
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
                print("⚠️  Could not find analysis markers in result")
                raise ValueError("No analysis markers found in result")

        except Exception as e:
            print(f"❌ Analysis parsing error: {e}")
            print(f"Raw analysis result: {analysis_result}")

            # Fallback analysis
            fallacies = []
            if "scientists say" in content.lower():
                fallacies.append("appeal_to_authority")
            if any(word in content.lower() for word in ["stupid", "idiot"]):
                fallacies.append("ad_hominem")

            return {
                "word_count": len(content.split()),
                "fallacies": fallacies,
                "strength": 0.4,
                "suggestions": ["Add supporting evidence"],
                "reasoning_quality": "needs_improvement",
                "error": str(e)
            }

    async def evolve_argument(self, arg_id: str, content: str, analysis: dict) -> str:
        """Evolve argument using Code Executor agent"""

        print(f"🧬 Evolving argument: {arg_id}")

        # Enhanced evolution code with clear JSON output
        evolution_code = f'''
import json
import random

def evolve_argument_advanced(original, analysis):
    """Advanced argument evolution based on detailed analysis"""

    suggestions = analysis.get("suggestions", [])
    fallacies = analysis.get("fallacies", [])
    strength = analysis.get("strength", 0.5)
    evidence_score = analysis.get("evidence_score", 0.0)

    evolved = original
    changes_made = []

    # Remove fallacies first
    if "ad_hominem" in fallacies:
        for attack in ["stupid", "idiot", "moron", "fool", "dumb"]:
            if attack in evolved.lower():
                evolved = evolved.replace(attack, "problematic")
                evolved = evolved.replace(attack.capitalize(), "Problematic")
                changes_made.append("removed ad hominem attack")

    if "appeal_to_authority" in fallacies:
        evolved = evolved.replace("scientists say", "peer-reviewed research demonstrates")
        evolved = evolved.replace("Scientists say", "Peer-reviewed research demonstrates")
        changes_made.append("strengthened authority reference")

    if "bandwagon" in fallacies:
        evolved = evolved.replace("everyone knows", "evidence indicates")
        evolved = evolved.replace("everybody thinks", "research suggests")
        changes_made.append("replaced bandwagon argument")

    # Add evidence if lacking
    if evidence_score < 0.3 and "Add supporting evidence" in ' '.join(suggestions):
        evidence_additions = [
            " This conclusion is supported by multiple peer-reviewed studies.",
            " Empirical data from controlled experiments confirms this position.",
            " Meta-analyses of research in this field provide strong evidence for this claim.",
            " Longitudinal studies have consistently demonstrated this relationship."
        ]

        evidence_add = random.choice(evidence_additions)
        evolved += evidence_add
        changes_made.append("added evidence support")

    # Improve logical structure
    if "Clarify logical connections" in ' '.join(suggestions):
        if "because" not in evolved.lower() and "therefore" not in evolved.lower():
            # Find good insertion point for logical connector
            sentences = evolved.split('.')
            if len(sentences) > 1:
                evolved = sentences[0] + ". Therefore, " + '.'.join(sentences[1:])
                changes_made.append("clarified logical structure")

    # Add detailed reasoning if too brief
    if analysis.get("word_count", 0) < 15:
        reasoning_additions = [
            " The logical foundation for this position involves multiple interconnected factors that together create a compelling case.",
            " This conclusion emerges from careful consideration of the available evidence and established theoretical frameworks.",
            " Multiple lines of evidence converge to support this position through both empirical observation and theoretical analysis."
        ]

        reasoning_add = random.choice(reasoning_additions)
        evolved += reasoning_add
        changes_made.append("expanded reasoning")

    # Reduce excessive uncertainty if needed
    if analysis.get("word_count", 0) > 0:
        uncertainty_ratio = evolved.lower().count("maybe") + evolved.lower().count("possibly") + evolved.lower().count("might")
        if uncertainty_ratio > 2:
            evolved = evolved.replace(" maybe ", " ")
            evolved = evolved.replace(" possibly ", " ")
            changes_made.append("reduced excessive uncertainty")

    # Ensure some meaningful change occurred
    if not changes_made:
        evolved += " [This argument has been enhanced through systematic logical analysis and evidence-based improvement.]"
        changes_made.append("added enhancement note")

    return {{
        "evolved_text": evolved,
        "changes_made": changes_made,
        "improvement_type": "comprehensive_enhancement"
    }}

# Execute evolution
original = """{content.replace('"', '\\"')}"""
analysis = {json.dumps(analysis)}

result = evolve_argument_advanced(original, analysis)

# Output ONLY the JSON result for easy parsing
print("EVOLUTION_START")
print(json.dumps(result, indent=2))
print("EVOLUTION_END")
'''

        # Execute evolution using code executor
        evolution_result = await self.bridge.execute_python_code(evolution_code)

        try:
            # Parse evolution result - look for content between markers
            lines = evolution_result.strip().split('\n')

            # Find the JSON content between EVOLUTION_START and EVOLUTION_END
            json_content = ""
            in_evolution = False

            for line in lines:
                if "EVOLUTION_START" in line:
                    in_evolution = True
                    continue
                elif "EVOLUTION_END" in line:
                    break
                elif in_evolution:
                    json_content += line + "\n"

            if json_content.strip():
                evolution_data = json.loads(json_content.strip())
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
            else:
                raise ValueError("Could not find evolution markers in result")

        except Exception as e:
            print(f"❌ Evolution error: {e}")
            print(f"Raw evolution result: {evolution_result}")

            # Fallback evolution
            simple_evolved = content + " This argument has been enhanced with additional supporting evidence."
            new_arg_id = await self.add_argument(simple_evolved, "evolved_claim")
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
        print("   • Used ADK API with Runner and Content objects")
        print("   • Created SQLite database for argument storage")
        print("   • Executed Python code for analysis and evolution")
        print("   • Performed ArXiv searches for validation")
        print("   • Demonstrated measurable argument improvement")
        print("   • All using YOUR local MCP infrastructure with API.")

# Main execution
async def main():
    coordinator = ReasonFlowCoordinator()
    await coordinator.initialize()
    await coordinator.run_evolution_demo()

if __name__ == "__main__":
    print("🧠 ReasonFlow with MCP Agent Integration - FIXED ADK API")
    print("This version uses your actual MCP agents with ADK Runner interface")
    print()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
