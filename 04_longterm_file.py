"""
Phase 4: Long-term Memory
04_longterm_file.py - JSON-based persistence

This script demonstrates a naive but functional way to achieve 
long-term memory by using a file-based checkpointer.
"""

import os
import json
import sqlite3
from agent_utils import create_llm, run_agent
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver

# ============================================================================
# AGENT SETUP WITH PERSISTENT DB
# ============================================================================

def create_persistent_agent(db_path="memory.db"):
    llm = create_llm()
    
    # Instead of MemorySaver (RAM), we use SqliteSaver (Disk)
    # This allows memory to survive script restarts!
    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)
    
    return create_agent(
        llm,
        tools=[],
        system_prompt="You are a helpful assistant with a permanent memory. You never forget a face (or a message)!",
        checkpointer=memory
    )

# ============================================================================
# DEMO EXECUTION
# ============================================================================

def main():
    db_file = "memory.db"
    print("=" * 80)
    print("Long-term Memory - SQLite Persistence")
    print("=" * 80)
    
    # Check if we have an existing memory
    first_time = not os.path.exists(db_file)
    
    if first_time:
        print("\n📝 No existing memory found. Starting a fresh conversation...")
    else:
        print("\n📂 Founding existing memory database! The agent should remember us.")

    agent = create_persistent_agent(db_file)
    config = {"configurable": {"thread_id": "persistent_user_456"}}

    if first_time:
        question = "Hi! My name is Alice and my favorite color is Blue. Please save this in your long-term memory."
        print(f"\n🗣️ Turn (Session 1): {question}")
        answer = run_agent(agent, question, config=config)
        print(f"\n✅ Answer: {answer}")
        print("\n💾 Memory has been saved to 'memory.db'.")
        print("👉 RUN THIS SCRIPT AGAIN to see the agent remember Alice!")
    else:
        question = "Do you remember me? What is my name and favorite color?"
        print(f"\n🗣️ Turn (Session 2): {question}")
        answer = run_agent(agent, question, config=config)
        print(f"\n✅ Answer: {answer}")
        
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
