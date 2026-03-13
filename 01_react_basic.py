"""
File: 01_react_basic.py
Topic: ReAct Pattern - Basic Implementation
Level: Basic
Duration: 5 minutes in presentation

Description:
This file demonstrates the fundamental ReAct (Reasoning + Acting) pattern with a single
tool. The agent follows a Thought → Action → Observation cycle to solve problems.

Prerequisites:
- Ollama installed and running
- qwen2.5:7b model pulled (or any tool-calling capable model)
- Virtual environment activated

Learning Objectives:
1. Understand the ReAct pattern (Thought → Action → Observation)
2. See how agents reason before taking actions
3. Observe the iterative problem-solving process

Usage:
    python 01_react_basic.py
"""

# ============================================================================
# IMPORTS
# ============================================================================
from langchain_core.tools import tool
from langchain.agents import create_agent
from agent_utils import create_llm, run_agent, run_interactive_loop

# ============================================================================
# TOOL DEFINITION
# ============================================================================
@tool
def calculator(expression: str) -> str:
    """
    Perform mathematical calculations.

    This tool evaluates mathematical expressions and returns the result.
    Use this when you need to perform arithmetic operations.

    Args:
        expression: A valid Python mathematical expression (e.g., "25 * 4 + 10")

    Returns:
        str: The result of the calculation or an error message

    Examples:
        calculator("25 * 4") -> "100"
        calculator("100 + 10") -> "110"
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


# ============================================================================
# AGENT SETUP
# ============================================================================
def create_basic_react_agent():
    """
    Create a basic ReAct agent with a single calculator tool.

    Uses LangChain's create_agent which creates a LangGraph-based agent
    that calls tools in a loop until the task is complete.
    """
    agent = create_agent(
        create_llm(),
        tools=[calculator],
        system_prompt="""You are a helpful assistant that solves problems step by step.

Think through the problem carefully and use the available tools when needed.
Break down complex calculations into simpler steps.

When you use a tool:
1. Think about what you need to do
2. Use the appropriate tool
3. Observe the result
4. Continue until you have the final answer""",
    )

    return agent


# ============================================================================
# DEMONSTRATION
# ============================================================================
def main():
    print("=" * 80)
    print("ReAct Pattern - Basic Implementation")
    print("=" * 80)
    print("\nThis demo shows how an agent uses the ReAct pattern:")
    print("1. Thought: Agent reasons about what to do")
    print("2. Action: Agent decides which tool to use")
    print("3. Observation: Agent sees the result")
    print("4. Repeat until problem is solved")
    print("\n" + "=" * 80 + "\n")

    print("🤖 Creating ReAct agent with calculator tool...")
    agent = create_basic_react_agent()
    print("✅ Agent created successfully!\n")

    question = "What is 25 * 4 + 10?"
    print(f"📝 Question: {question}")

    try:
        answer = run_agent(agent, question)
        print(f"\n✅ Final Answer: {answer}")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("\n💡 Key Observations:")
    print("   1. Agent breaks down the problem into steps")
    print("   2. Agent uses the calculator tool")
    print("   3. Each action builds on previous observations")
    print("   4. Agent reaches the correct answer: 110")
    print("\n" + "=" * 80)

    # Launch interactive mode
    run_interactive_loop(agent)


# ============================================================================
# EXPECTED OUTPUT
# ============================================================================
"""
Expected Output (stream mode):

📋 Agent Execution (streaming):
──────────────────────────────────────────────────────────

  ⚡ Step 1: Agent calls tool
     🔧 Tool:  calculator
     📥 Input: {'expression': '25 * 4 + 10'}
     📤 Output: Result: 110

──────────────────────────────────────────────────────────

✅ Final Answer: The result of the expression 25 * 4 + 10 is 110.
"""

if __name__ == "__main__":
    main()
