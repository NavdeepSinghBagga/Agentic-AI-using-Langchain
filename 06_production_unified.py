"""
Phase 6: Production Extras
06_production_unified.py - Retries, Timeouts, & Validation

This script demonstrates production-level reliability patterns:
1. Tool Retries (using tenacity) for handling temporary failures.
2. Logic-based Timeouts (simulation) for preventing hung agents.
3. Input Validation (using Pydantic) for grounding LLM outputs.
"""

import time
from typing import Optional
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed
from langchain_core.tools import tool
from agent_utils import create_llm, run_agent
from langchain.agents import create_agent

# ============================================================================
# 1. TOOL VALIDATION (Pydantic Schema)
# ============================================================================

class SearchInput(BaseModel):
    query: str = Field(description="The search query. Must be at least 3 characters.")
    year: Optional[int] = Field(None, description="Optional year filter.")

# ============================================================================
# 2. TOOL RETRIES & FAILURES
# ============================================================================

# Global counter for flaky tool demo
flaky_attempts = 0

@tool(args_schema=SearchInput)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def production_search_tool(query: str, year: Optional[int] = None) -> str:
    """A search tool that demonstrates reliability features."""
    global flaky_attempts
    
    # CASE 1: INPUT VALIDATION (Pydantic will catch if query < 3)
    # (Triggered automatically by LangChain if schema is passed)

    # CASE 2: RETRIES FOR FLAKY RESPONSE
    if "flaky" in query.lower():
        flaky_attempts += 1
        if flaky_attempts < 3:
            print(f"   [Tool] Simulated API Failure (Attempt {flaky_attempts})...")
            raise Exception("503 Service Unavailable")
        return f"SUCCESS! Recovered on attempt {flaky_attempts}. Results: {query}."

    # CASE 3: TIMEOUT SIMULATION
    if "slow" in query.lower():
        print("   [Tool] Detected potentially hung process...")
        # In a real system, we'd use a thread-safe timeout. 
        # Here we simulate the result of a timeout for the agent to reason about.
        time.sleep(2)
        return "ERROR: ToolExecutionTimeout - Service unresponsive after 5 seconds."

    return f"Standard results for '{query}'"

# ============================================================================
# AGENT SETUP
# ============================================================================

def create_production_agent():
    llm = create_llm()
    tools = [production_search_tool]
    
    system_prompt = """
    You are a production-grade research agent.
    If a tool returns an error (Timeout or Failure), explain it to the user.
    If a query seems too short (validation error), ask for more detail.
    """
    
    return create_agent(llm, tools=tools, system_prompt=system_prompt)

# ============================================================================
# DEMO EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("PHASE 6: PRODUCTION EXTRAS")
    print("=" * 80)
    
    agent = create_production_agent()
    
    # Demo 1: Retries
    print("\n[DEMO 1: AUTOMATIC RETRIES]")
    print("Feeding a query that contains 'flaky' - Watch the tool fail and recover.")
    run_agent(agent, "Please use the search_tool to look up exactly: 'flaky data about Mars'")
    
    # Demo 2: Timeouts
    print("\n" + "-" * 40)
    print("[DEMO 2: TIMEOUT HANDLING]")
    run_agent(agent, "Please use the search_tool to look up exactly: 'slow report on AI'")

    # Demo 3: Validation
    print("\n" + "-" * 40)
    print("[DEMO 3: VALIDATION ERRORS]")
    print("Giving a query that is too short...")
    # LLMs usually refuse to send short queries if told, but we can try to force it
    run_agent(agent, "Search for 'a'")

    print("\n✅ Production Extras Demo Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
