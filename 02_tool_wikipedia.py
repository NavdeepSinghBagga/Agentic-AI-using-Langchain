"""
Phase 2: Tool Registry
02_tool_wikipedia.py - Wikipedia Integration

This script demonstrates how to integrate the 'wikipedia' Python package.
It highlights how to handle tool exceptions (like DisambiguationError) gracefully
and return them to the LLM so it can retry with a better query.

Prerequisites:
- pip install wikipedia (Already in requirements.txt)
"""

import wikipedia
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from agent_utils import create_llm, run_agent, run_interactive_loop
from langchain.agents import create_agent

# ============================================================================
# TOOL SCHEMAS AND DEFINITIONS
# ============================================================================

class WikipediaInput(BaseModel):
    """Input schema for the Wikipedia search tool."""
    query: str = Field(
        description="The specific topic, person, or concept to look up on Wikipedia."
    )

@tool(args_schema=WikipediaInput)
def search_wikipedia(query: str) -> str:
    """
    Looks up a topic on Wikipedia and returns a summary of the article.
    Use this to find factual information, history, and definitions.
    """
    try:
        # We limit the sentences to avoid overflowing the LLM's context window
        summary = wikipedia.summary(query, sentences=4, auto_suggest=False)
        return summary
        
    except wikipedia.exceptions.DisambiguationError as e:
        # If the query is too broad (e.g. "Apple"), Wikipedia suggests alternatives.
        # We catch this error and return the specific options back to the Agent!
        options = e.options[:5] # Only return top 5 to save context
        return f"Query '{query}' is too ambiguous. Did you mean one of these? {', '.join(options)}"
        
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for '{query}'. Try a different search term."
        
    except Exception as e:
        return f"Wikipedia Search Error: {str(e)}"

# ============================================================================
# AGENT SETUP
# ============================================================================

def create_wiki_agent():
    """Create a ReAct agent injected with our Wikipedia tool."""
    llm = create_llm()
    tools = [search_wikipedia]
    
    system_prompt = """
    You are a highly knowledgeable encyclopedic researcher.
    You have access to a Wikipedia search tool.
    When asked about any factual topic, ALWAYS use the `search_wikipedia` tool.
    
    CRITICAL: If the tool says the query is "too ambiguous" and provides a list of options,
    you MUST immediately call the tool again using one of the more specific options provided!
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
    print("Tool Registry - Wikipedia API Integration")
    print("=" * 80)
    print("\nThis demo highlights how agents handle Disambiguation errors.")
    print("If a query is too broad, the Python code catches the exception")
    print("and returns alternatives to the agent so it can try again automatically!\n")
    
    print("🤖 Creating Wikipedia Agent...")
    agent = create_wiki_agent()
    print("✅ Agent created with `search_wikipedia` tool!\n")
    
    test_cases = [
        "Tell me a brief summary of Marie Curie",
        "Look up 'Mercury'. I want to know about the planet, not the element.", # Forces Disambiguation handling
    ]
    
    for i, question in enumerate(test_cases, 1):
        print(f"{'='*80}")
        print(f"📝 Test {i}: {question}")
        print(f"{'='*80}")
        print("📋 Agent Execution:")
        answer = run_agent(agent, question)
        print(f"\n✅ Answer: {answer}\n")

    print("\n" + "=" * 80)
    run_interactive_loop(agent)

if __name__ == "__main__":
    main()
