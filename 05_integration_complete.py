"""
Phase 5: Integration
05_integration_complete.py - Unified Agent Demo

The "Final Boss" of basic agentic AI. 
This script combines:
1. ReAct Reasoning Loop (Phase 1)
2. Complex Tool Registry - Weather, Wikipedia, News (Phase 2)
3. Persistent Long-term Memory with SQLite (Phase 4)
4. Robust agent_utils display system
"""

import os
import sqlite3
import importlib.util
import sys
from agent_utils import create_llm, run_agent
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver

# Helper to import digit-prefixed files
def import_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import Tool Registry from Phase 2
tool_registry = import_from_file("tool_registry", "./02_tool_registry.py")
get_all_tools = tool_registry.get_all_tools

# ============================================================================
# UNIFIED AGENT SETUP
# ============================================================================

def create_unified_agent(db_path="unified_agent.db"):
    llm = create_llm()
    tools = get_all_tools()
    
    # Persistent SQLite Memory
    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)
    
    system_prompt = """
    You are an advanced, stateful AI research assistant.
    You have access to Wikipedia, OpenWeatherMap, and Google News tools.
    You remember users across conversation sessions using a persistent database.
    
    GUIDELINES:
    1. Be concise but factual.
    2. Use your tools sequentially if a query requires multiple steps (e.g. check city weather, then find news).
    3. Always refer back to shared context if available (e.g. if the user previously mentioned their location).
    """
    
    return create_agent(
        llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=memory
    )

# ============================================================================
# DEMO EXECUTION
# ============================================================================

def main():
    db_file = "unified_agent.db"
    print("=" * 80)
    print("PHASE 5: UNIFIED AGENT INTEGRATION")
    print("=" * 80)
    print("Capabilities: ReAct + Multi-Tool Registry + SQLite Persistence")
    
    agent = create_unified_agent(db_file)
    config = {"configurable": {"thread_id": "final_boss_demo"}}
    
    # turn 1: Context & Persistence
    print("\n[TURN 1: PERSISTENCE & TOOL USAGE]")
    q1 = "Hi! I'm planning a move to Toronto. Can you tell me the current weather there and save my preference?"
    run_agent(agent, q1, config=config)
    
    print("\n" + "." * 80)
    print("SIMULATING SESSION RESTART...")
    print("." * 80)
    
    # turn 2: Recall & Different Tool Usage
    print("\n[TURN 2: CONTEXTUAL RECALL & NEWS SEARCH]")
    q2 = "Based on my move, what are the latest news headlines from my new city?"
    run_agent(agent, q2, config=config)
    
    print("\n" + "." * 80)
    print("SIMULATING ANOTHER RESTART...")
    print("." * 80)
    
    # turn 3: Deep Research
    print("\n[TURN 3: WIKIPEDIA RESEARCH]")
    q3 = "Forget news for a second. Can you find the main highlights of the city's history on Wikipedia?"
    run_agent(agent, q3, config=config)
    
    print("\n✅ Unified Agent Demo Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
