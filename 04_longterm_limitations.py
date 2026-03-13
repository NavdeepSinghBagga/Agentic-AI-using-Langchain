"""
Phase 4: Long-term Memory
04_longterm_limitations.py - Prompt Bloat & RAG Hype

This script demonstrates the "Prompt Bloat" problem: how adding too much 
history leads to context window exhaustion, and introduces the concept 
of RAG (Retrieval Augmented Generation) as the hype-worthy solution.
"""

from agent_utils import create_llm, run_agent
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

# ============================================================================
# AGENT SETUP
# ============================================================================

def create_scaling_agent():
    llm = create_llm()
    # MemorySaver is used here to show current state accumulation
    memory = MemorySaver()
    
    system_prompt = """
    You are an agent with a 'perfect' memory. 
    You are very verbose and detailed in your responses.
    """
    
    return create_agent(
        llm,
        tools=[],
        system_prompt=system_prompt,
        checkpointer=memory
    )

# ============================================================================
# DEMO EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("Long-term Memory - The Challenge of Scaling (Prompt Bloat)")
    print("=" * 80)
    
    agent = create_scaling_agent()
    config = {"configurable": {"thread_id": "scaling_demo"}}
    
    print("\n📦 SIMULATING A LONG CONVERSATION...")
    print("We are going to feed the agent some facts to 'bloat' the prompt.")
    
    facts = [
        "The Eiffel Tower can be 15 cm taller during the summer.",
        "A group of flamingos is called a 'flamboyance'.",
        "Octopuses have three hearts."
    ]
    
    for i, fact in enumerate(facts, 1):
        print(f"   Feeding Fact #{i}...")
        # Running normally now that silent is removed from agent_utils
        run_agent(agent, f"Tell me a long story about: {fact}", config=config)

    print("\n⚠️  The agent's memory 'bucket' is now filling up!")
    print("Every time we ask a question, ALL previous stories are sent back to the LLM.")
    
    # Show the current state size (messages count)
    state = agent.get_state(config)
    msg_count = len(state.values["messages"])
    print(f"📊 Current Message Count in History: {msg_count}")
    
    print("\n" + "!" * 40)
    print("THE PROBLEM: PROMPT BLOAT")
    print("1. Your token costs are skyrocketing.")
    print("2. Latency is increasing.")
    print("3. Soon, we will hit the Model's Context Window Limit (e.g. 128k tokens).")
    print("!" * 40)

    print("\n🚀 THE SOLUTION: RAG (Retrieval Augmented Generation)")
    print("-" * 40)
    print("Instead of sending EVERYTHING, we only search for relevant bits.")
    print("\nCOMING UP IN FUTURE SESSIONS:")
    print("🔹 Vector Databases (ChromaDB, Pinecone)")
    print("🔹 Embeddings (converting text to numbers)")
    print("🔹 Semantic Search (retrieving memories by meaning)")
    print("-" * 40)
    
    print("\n✅ Phase 4 Complete: We understand persistence AND its limits.")
    print("=" * 80)

if __name__ == "__main__":
    main()
