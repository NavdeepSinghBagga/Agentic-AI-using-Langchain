"""
Phase 3: Short-term Memory
03_memory_basic.py - ConversationBufferMemory

This script demonstrates how to give an agent short-term memory so it can
remember the context of the conversation across multiple turns.
We use LangGraph's MemorySaver to persist the conversational state.
"""

from agent_utils import create_llm, run_agent, run_interactive_loop
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

# ============================================================================
# AGENT SETUP WITH MEMORY
# ============================================================================

def create_memory_agent():
    """Create a ReAct agent with conversational memory."""
    llm = create_llm()
    
    # We create an in-memory checkpointer. This is what stores the chat history.
    # It lives only as long as this Python script is running context.
    memory = MemorySaver()
    
    system_prompt = """
    You are a helpful assistant with a good memory.
    Answer the user's questions naturally.
    If the user tells you their name, remember it!
    
    CRITICAL INSTRUCTION:
    UNDER NO CIRCUMSTANCES should you output raw JSON dicts or strings representing tool calls.
    Answer the user naturally in plain text.
    """
    
    # Notice we pass `checkpointer=memory` to LangGraph
    agent = create_agent(
        llm,
        tools=[], # No tools, pure conversation memory demo
        system_prompt=system_prompt,
        checkpointer=memory
    )
    
    return agent

# ============================================================================
# DEMO EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("Short-term Memory - ConversationBufferMemory")
    print("=" * 80)
    print("\nThis demo uses LangGraph's MemorySaver to maintain chat history")
    print("across multiple questions within the same session.\n")
    
    print("🤖 Creating Agent with Memory...")
    agent = create_memory_agent()
    
    # We MUST provide a thread_id when using memory so LangGraph knows
    # which conversation history to look up!
    config = {"configurable": {"thread_id": "demo_session_1"}}
    
    print("✅ Agent created! Thread ID: demo_session_1\n")
    
    # This is a multi-turn conversation
    test_cases = [
        "Hi! My name is Alice, and I'm learning about AI agents.",
        "What is my name?",
        "What did I say I was learning about?"
    ]
    
    for i, question in enumerate(test_cases, 1):
        print(f"{'='*80}")
        print(f"🗣️ Turn {i}: {question}")
        print(f"{'='*80}")
        
        # We pass 'config' to run_agent so it sends the thread_id
        answer = run_agent(agent, question, config=config)
        print(f"\n✅ Answer: {answer}\n")

    print("\n" + "=" * 80)
    
    # To use interactive loop with memory, we'd need to update agent_utils.
    # For now, we just end the demo.
    print("Demo complete. To test interactively, run one of the previous scripts.")

if __name__ == "__main__":
    main()
