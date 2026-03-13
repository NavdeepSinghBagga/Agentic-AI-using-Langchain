"""
File: 01_react_advanced.py
Topic: ReAct Pattern - Advanced Implementation
Level: Advanced
Duration: 2 minutes in presentation

Description:
This file demonstrates the ReAct pattern with ERROR HANDLING and RECOVERY.
Tools can fail, and the agent must handle those failures gracefully — a
critical capability for production agents.

Prerequisites:
- Ollama installed and running
- qwen2.5:7b model pulled
- Virtual environment activated
- Completed: 01_react_basic.py, 01_react_intermediate.py

Learning Objectives:
1. Handle tool failures gracefully inside the agent loop
2. See how agents recover and try alternative approaches
3. Understand recursion_limit as a safety net against infinite loops

Usage:
    python 01_react_advanced.py
"""

# ============================================================================
# IMPORTS
# ============================================================================
import math
import random
from langchain_core.tools import tool
from langchain.agents import create_agent
from agent_utils import create_llm, run_agent, run_interactive_loop

# ============================================================================
# TOOL DEFINITIONS — Including tools that can FAIL
# ============================================================================
@tool
def reliable_calculator(expression: str) -> str:
    """
    A reliable calculator that always works.
    Use this for mathematical calculations.

    Args:
        expression: A valid Python math expression
    """
    try:
        # Import math module into the eval environment so functions like sqrt() work
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def unreliable_api(query: str) -> str:
    """
    Simulates an external API that sometimes fails.
    Use this to look up information. Note: this service can be unreliable.

    Args:
        query: The information to look up
    """
    # Simulate 50% failure rate
    if random.random() < 0.5:
        return "ERROR: API Service temporarily unavailable (503). Try using fallback_knowledge instead."

    responses = {
        "population": "The world population is approximately 8.1 billion people.",
        "python": "Python is a high-level programming language created by Guido van Rossum.",
        "ai": "Artificial Intelligence is a branch of computer science focused on creating intelligent machines.",
    }

    for key, response in responses.items():
        if key in query.lower():
            return response

    return f"Information about '{query}': This is simulated data for demonstration purposes."


@tool
def fallback_knowledge(topic: str) -> str:
    """
    A fallback knowledge base that always works but has limited information.
    Use this as a backup when unreliable_api fails or returns an error.

    Args:
        topic: The topic to look up
    """
    knowledge = {
        "population": "World population is estimated at around 8 billion (cached data).",
        "python": "Python is a popular programming language (from local cache).",
        "ai": "AI refers to machines that can perform tasks requiring human intelligence (from local cache).",
        "weather": "Weather data unavailable offline. Temperatures vary by season and location.",
    }

    for key, info in knowledge.items():
        if key in topic.lower():
            return f"[Fallback] {info}"

    return f"[Fallback] Limited information available about '{topic}'. Try a more specific query."


# ============================================================================
# AGENT SETUP
# ============================================================================
def create_advanced_react_agent():
    """
    Create a ReAct agent that handles errors gracefully.
    """
    agent = create_agent(
        create_llm(),
        tools=[reliable_calculator, unreliable_api, fallback_knowledge],
        system_prompt="""You are a resilient assistant that handles errors gracefully.

You have access to these tools:
- reliable_calculator: Always works. Use for math.
- unreliable_api: Can fail sometimes. Use for looking up information.
- fallback_knowledge: Always works but has limited info. Use as a BACKUP when unreliable_api fails.

IMPORTANT ERROR HANDLING STRATEGY:
1. Try unreliable_api first for information lookups
2. If it returns an ERROR message, try fallback_knowledge instead
3. Never give up — always try an alternative approach
4. If all tools fail, say so honestly but provide what you can

Always explain to the user if you had to use a fallback source.

Note: DO NOT output raw JSON logic to simulate a tool call. You MUST use your native tool calling interface only.
Note: Do not Hullucinate, do not try to answer the question if you don't have the answer.
""",
    )

    return agent


# ============================================================================
# DEMONSTRATION
# ============================================================================
def main():
    print("=" * 80)
    print("ReAct Pattern - Advanced: Error Handling & Recovery")
    print("=" * 80)
    print("\nThis demo shows how agents RECOVER from tool failures.")
    print("The 'unreliable_api' tool fails ~50% of the time.")
    print("The agent should fall back to 'fallback_knowledge' when it fails.")
    print("\n" + "=" * 80 + "\n")

    random.seed(42)

    print("🤖 Creating advanced ReAct agent with error handling...")
    agent = create_advanced_react_agent()
    print("✅ Agent created with reliable + unreliable tools!\n")

    test_cases = [
        {
            "question": "What is the world population?",
            "note": "Uses unreliable_api first, falls back if needed",
        },
        {
            "question": "Calculate the square root of 144 and then multiply by 5",
            "note": "Uses reliable_calculator — should always work",
        },
        {
            "question": "Tell me about artificial intelligence",
            "note": "Uses unreliable_api first, falls back if needed",
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"📝 Test {i}: {test['question']}")
        print(f"   💭 Expected: {test['note']}")
        print(f"{'='*80}")

        try:
            answer = run_agent(agent, test["question"])
            print(f"\n✅ Answer: {answer}\n")
        except Exception as e:
            print(f"\n❌ Agent failed completely: {str(e)}")
            print("   This demonstrates the need for robust error handling!\n")

    print("\n" + "=" * 80)
    print("\n💡 Key Observations:")
    print("   1. Agent tries the primary tool first")
    print("   2. On failure, agent falls back to alternatives")
    print("   3. The agent loop has a built-in recursion limit to prevent infinite loops")
    print("   4. Agent communicates failures honestly to the user")
    print("\n" + "=" * 80)

    # Launch interactive mode
    run_interactive_loop(agent)


# ============================================================================
# EXPECTED OUTPUT
# ============================================================================
"""
Expected Output (varies due to random failures):

📝 Test 1: What is the world population?
  ⚡ Step 1: Agent calls tool
     🔧 Tool:  unreliable_api
     📥 Input: {'query': 'world population'}
     📤 Output: ERROR: API Service temporarily unavailable (503)...
  ⚡ Step 2: Agent calls tool
     🔧 Tool:  fallback_knowledge
     📥 Input: {'topic': 'population'}
     📤 Output: [Fallback] World population is estimated at around 8 billion
  ✅ Answer: World population is approximately 8 billion (from fallback)

📝 Test 2: Calculate sqrt(144) * 5
  ⚡ Step 1: Agent calls tool
     🔧 Tool:  reliable_calculator
     📥 Input: {'expression': '144 ** 0.5 * 5'}
     📤 Output: Result: 60.0
  ✅ Answer: 60.0
"""

if __name__ == "__main__":
    main()
