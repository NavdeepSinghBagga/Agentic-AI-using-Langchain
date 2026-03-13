"""
Phase 3: Short-term Memory
03_memory_window.py - Window-based Memory

This script demonstrates 'ConversationBufferWindowMemory'.
Instead of keeping the entire chat history (which grows indefinitely
and consumes tokens), a windowed memory only keeps the last 'k' interactions.
"""

from agent_utils import create_llm, run_agent
from langchain_core.tools import tool
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from typing import Any

# ============================================================================
# SIMPLE TOOL
# ============================================================================

@tool
def get_world_capitals(country: str) -> str:
    """Returns the capital of a given country."""
    capitals = {
        "France": "Paris",
        "Japan": "Tokyo",
        "Brazil": "Brasilia",
        "Canada": "Ottawa"
    }
    return capitals.get(country, f"I don't know the capital of {country}.")

# ============================================================================
# STATE MODIFIER (Window Logic)
# ============================================================================

# We keep only the last 'k' human messages (and their corresponding AI responses).
# Since LangGraph stores memory as a raw list of messages, we use a 
# state_modifier function to trim the old messages out before passing to the LLM.

def create_windowed_agent(k=2):
    """
    Create a ReAct agent with a sliding window memory limit.
    k represents the number of recent Human-AI exchanges to keep.
    """
    llm = create_llm()
    tools = [get_world_capitals]
    memory = MemorySaver()
    
    system_prompt = """
    You are a helpful geography assistant.
    You have a limit to your memory—you only remember the last few things we discussed!
    
    CRITICAL INSTRUCTIONS:
    1. UNDER NO CIRCUMSTANCES output raw JSON dicts or representative strings of tool calls.
    2. Answer naturally in plain text. 
    3. If the user asks about something that has been trimmed from your memory window (meaning you 
       honestly don't see it in history), just say you don't recall. 
    4. ONLY use tools if explicitly needed to answer a geography fact.
    """
    
    # Use LangChain middleware strategy to trim messages
    @before_model
    def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        messages = state["messages"]
        # In a real conversation, each 'turn' is often a User + AI message pair.
        # We find the human messages.
        human_messages = [m for m in messages if m.type == "human"]
        
        if len(human_messages) <= k:
            return None # No trimming needed
            
        # Find the human message that starts our window
        cutoff_msg = human_messages[-k]
        cutoff_idx = next(i for i, m in enumerate(messages) if m.id == cutoff_msg.id)
        
        # Delete everything before the cutoff strictly by ID. 
        # We only remove messages that have a valid ID.
        messages_to_remove = [m for m in messages[:cutoff_idx] if hasattr(m, "id") and m.id]
        
        # We specifically keep the system prompt if it was the first message
        if messages_to_remove and messages_to_remove[0].type == "system":
            messages_to_remove = messages_to_remove[1:]
        
        removals = [RemoveMessage(id=m.id) for m in messages_to_remove]
        
        if not removals:
            return None
            
        return {"messages": removals}

    agent = create_agent(
        llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[trim_messages],
        checkpointer=memory
    )
    
    return agent

# ============================================================================
# DEMO EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("Short-term Memory - ConversationBufferWindowMemory")
    print("=" * 80)
    print("\nThis demo uses a custom state_modifier to implement a sliding window.")
    print("The agent will only remember the last 2 interactions (k=2).")
    print("Older messages are trimmed from the prompt to save tokens.\n")
    
    # Create agent with k=2 window
    print("🤖 Creating Agent with windowed memory (k=2)...")
    agent = create_windowed_agent(k=2)
    config = {"configurable": {"thread_id": "window_demo"}}
    
    print("✅ Agent created!\n")
    
    test_cases = [
        "Hi! My name is Charlie. Let's talk about countries.",
        "What is the capital of France?",
        "What is the capital of Japan?",
        "What is the capital of Canada?",
        "Do you remember my name?" 
    ]
    
    for i, question in enumerate(test_cases, 1):
        print(f"{'='*80}")
        print(f"🗣️ Turn {i}: {question}")
        if i == 5:
            print("   💭 Expected: Should FORGET the name (Turn 1 was trimed out!)")
        print(f"{'='*80}")
        
        answer = run_agent(agent, question, config=config)
        print(f"\n✅ Answer: {answer}\n")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
