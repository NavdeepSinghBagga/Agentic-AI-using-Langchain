"""
Phase 2: Tool Registry
02_tool_single.py - Single Tool Definition

This script demonstrates how to create a highly structured LangChain tool
with Pydantic validation. It shows how the Python function is converted
into a JSON Schema that the LLM understands, proving that "prompting"
happens under the hood.

Concepts:
- @tool decorator
- Pydantic BaseModel for input validation
- JSON Schema extraction
"""

import json
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from agent_utils import create_llm, run_agent, run_interactive_loop
from langchain.agents import create_agent

# ============================================================================
# TOOL SCHEMAS AND DEFINITIONS
# ============================================================================

class CalculatorInput(BaseModel):
    """Input schema for the calculator tool."""
    expression: str = Field(
        description="A mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(144) * 5')"
    )

# The 'args_schema' connects the Pydantic validator to the tool
@tool(args_schema=CalculatorInput)
def safe_calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the result.
    This tool should be used for ANY math calculation to avoid hallucination.
    """
    try:
        # Import math module into a safe eval dictionary
        import math
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"

# ============================================================================
# AGENT SETUP
# ============================================================================

def create_single_tool_agent():
    """Create a ReAct agent injected with our structured schema tool."""
    llm = create_llm()
    tools = [safe_calculator]
    
    system_prompt = """
    You are a helpful assistant with access to a safe calculator.
    You MUST use the calculator for any mathematical questions.
    Do not guess the answer.
    """
    
    agent = create_agent(
        llm,
        tools=tools,
        system_prompt=system_prompt,
    )
    
    return agent

# ============================================================================
# DEMO EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("Tool Registry - Single Tool & Schema Validation")
    print("=" * 80)
    
    # 1. Look under the hood at the schema
    print("\n🔍 UNDER THE HOOD: HOW THE LLM SEES THIS TOOL")
    print("When LangChain boots up, it converts our Python code into this JSON Schema:\n")
    
    schema = safe_calculator.args_schema.model_json_schema()
    print(json.dumps(schema, indent=2))
    
    print("\nThis exact schema is injected directly into the LLM's system prompt!")
    print("=" * 80)
    
    # 2. Run the agent
    print("\n🤖 Creating Agent and executing math problem...")
    agent = create_single_tool_agent()
    
    question = "What is the square root of 65536 divided by 4?"
    print(f"\n📝 Question: {question}\n")
    
    print("📋 Agent Execution:")
    answer = run_agent(agent, question)
    print(f"\n✅ Answer: {answer}\n")

    print("\n" + "=" * 80)# 3. Interactive mode
    print("\n" + "=" * 80)
    run_interactive_loop(agent)

if __name__ == "__main__":
    main()
