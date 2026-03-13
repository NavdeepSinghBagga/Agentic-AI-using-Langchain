"""
File: 01_react_intermediate.py
Topic: ReAct Pattern - Intermediate Implementation
Level: Intermediate
Duration: 3 minutes in presentation

Description:
This file demonstrates the ReAct pattern with MULTIPLE tools. The agent must
choose which tool to use based on the user's question — showing intelligent
tool selection as part of the reasoning loop.

Prerequisites:
- Ollama installed and running
- qwen2.5:7b model pulled
- Virtual environment activated
- Completed: 01_react_basic.py

Learning Objectives:
1. See how agents choose between multiple tools
2. Understand tool selection reasoning
3. Observe multi-step problem solving with different tools

Usage:
    python 01_react_intermediate.py
"""

# ============================================================================
# IMPORTS
# ============================================================================
from langchain_core.tools import tool
from langchain.agents import create_agent
from agent_utils import create_llm, run_agent, run_interactive_loop

# ============================================================================
# TOOL DEFINITIONS — Multiple tools for the agent to choose from
# ============================================================================
@tool
def calculator(expression: str) -> str:
    """
    Perform mathematical calculations.
    Use this when the user asks a math question or needs arithmetic.

    Args:
        expression: A valid Python math expression (e.g., "25 * 4 + 10")
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def string_analyzer(text: str) -> str:
    """
    Analyze a text string and return useful statistics.
    Use this when the user asks about text properties like length, word count,
    or character analysis.

    Args:
        text: The text string to analyze
    """
    word_count = len(text.split())
    char_count = len(text)
    char_count_no_spaces = len(text.replace(" ", ""))
    sentence_count = text.count(".") + text.count("!") + text.count("?")
    uppercase_count = sum(1 for c in text if c.isupper())
    lowercase_count = sum(1 for c in text if c.islower())

    return (
        f"Text Analysis Results:\n"
        f"  - Word count: {word_count}\n"
        f"  - Character count (with spaces): {char_count}\n"
        f"  - Character count (without spaces): {char_count_no_spaces}\n"
        f"  - Sentence count: {sentence_count}\n"
        f"  - Uppercase letters: {uppercase_count}\n"
        f"  - Lowercase letters: {lowercase_count}"
    )


@tool
def unit_converter(conversion: str) -> str:
    """
    Convert between common units of measurement.
    Use this when the user asks to convert between units like km/miles,
    kg/pounds, celsius/fahrenheit, etc.

    Args:
        conversion: A string in the format "VALUE FROM_UNIT to TO_UNIT"
                    Examples: "100 km to miles", "32 fahrenheit to celsius"
    """
    try:
        parts = conversion.lower().strip().split()
        value = float(parts[0])
        from_unit = parts[1]
        to_unit = parts[3] if len(parts) > 3 else parts[-1]

        conversions = {
            ("km", "miles"): lambda v: v * 0.621371,
            ("miles", "km"): lambda v: v * 1.60934,
            ("kg", "pounds"): lambda v: v * 2.20462,
            ("pounds", "kg"): lambda v: v * 0.453592,
            ("celsius", "fahrenheit"): lambda v: (v * 9 / 5) + 32,
            ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
            ("meters", "feet"): lambda v: v * 3.28084,
            ("feet", "meters"): lambda v: v * 0.3048,
            ("liters", "gallons"): lambda v: v * 0.264172,
            ("gallons", "liters"): lambda v: v * 3.78541,
        }

        key = (from_unit, to_unit)
        if key in conversions:
            result = conversions[key](value)
            return f"{value} {from_unit} = {result:.2f} {to_unit}"
        else:
            return (f"Unsupported conversion: {from_unit} to {to_unit}. "
                    f"Supported: km/miles, kg/pounds, celsius/fahrenheit, meters/feet, liters/gallons")
    except (ValueError, IndexError) as e:
        return f"Conversion error: Use format 'VALUE FROM_UNIT to TO_UNIT'. Error: {str(e)}"


# ============================================================================
# AGENT SETUP
# ============================================================================
def create_intermediate_react_agent():
    """
    Create a ReAct agent with multiple tools.
    The agent must reason about WHICH tool to use for each question.
    """
    agent = create_agent(
        create_llm(),
        tools=[calculator, string_analyzer, unit_converter],
        system_prompt="""You are a helpful assistant with access to multiple tools.

For each question:
1. Think about which tool is most appropriate
2. Use that tool to get information
3. Provide a clear answer based on the result

Available tools:
- calculator: For math calculations
- string_analyzer: For analyzing text (word count, character stats, etc.)
- unit_converter: For converting between units (km/miles, kg/pounds, etc.)

Choose the right tool for each task. If a task doesn't need any tool, answer directly.""",
    )

    return agent


# ============================================================================
# DEMONSTRATION
# ============================================================================
def main():
    print("=" * 80)
    print("ReAct Pattern - Intermediate: Multi-Tool Agent")
    print("=" * 80)
    print("\nThis demo shows how an agent SELECTS the right tool from several options.")
    print("Tools available: calculator, string_analyzer, unit_converter")
    print("\n" + "=" * 80 + "\n")

    print("🤖 Creating multi-tool ReAct agent...")
    agent = create_intermediate_react_agent()
    print("✅ Agent created with 3 tools!\n")

    # Each question requires a DIFFERENT tool
    questions = [
        "What is 15% of 2500?",
        "Analyze this text: The quick brown fox jumps over the lazy dog.",
        "Convert 100 kilometers to miles.",
        "If I buy 3 items at $24.99 each, what's the total with 8% tax?",
        "Use the calculator to find the total cost of: 3 items of ABC at $24.99 each, plus 4 items of XYZ at $100 each, with an 18% tax on the entire order."
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"📝 Question {i}: {question}")
        print(f"{'='*80}")

        try:
            answer = run_agent(agent, question)
            print(f"\n✅ Answer: {answer}\n")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")

    print("=" * 80)
    print("\n💡 Key Observations:")
    print("   1. Agent chose calculator for math questions")
    print("   2. Agent chose string_analyzer for text analysis")
    print("   3. Agent chose unit_converter for conversions")
    print("   4. The SAME agent handled all types of queries")
    print("   5. Tool selection is part of the reasoning process")
    print("\n" + "=" * 80)

    # Launch interactive mode
    run_interactive_loop(agent)


# ============================================================================
# EXPECTED OUTPUT
# ============================================================================
"""
Expected Output (stream mode):

📝 Question 1: What is 15% of 2500?
  ⚡ Step 1: Agent calls tool
     🔧 Tool:  calculator
     📥 Input: {'expression': '2500 * 0.15'}
     📤 Output: Result: 375.0
  ✅ Answer: 15% of 2500 is 375.0

📝 Question 2: Analyze this text: The quick brown fox...
  ⚡ Step 1: Agent calls tool
     🔧 Tool:  string_analyzer
     📥 Input: {'text': 'The quick brown fox jumps over the lazy dog.'}
     📤 Output: Word count: 9, Characters: 44, ...

📝 Question 3: Convert 100 kilometers to miles.
  ⚡ Step 1: Agent calls tool
     🔧 Tool:  unit_converter
     📥 Input: {'conversion': '100 km to miles'}
     📤 Output: 100.0 km = 62.14 miles
"""
# If I buy 3 items of ABC at $24.99 each, and 4 items of XYZ at 100$ each with tax of 18% on both items, how much to pay? Think step by step

if __name__ == "__main__":
    main()
