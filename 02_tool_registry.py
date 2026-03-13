"""
Phase 2: Tool Registry
02_tool_registry.py - Complete Tool Registry

This script demonstrates giving the agent a comprehensive "toolbox" of all the tools
we've built so far. It shows how the LLM reasoning engine autonomously selects
the right tool for the right job based on the Pydantic schemas.

Prerequisites:
- OPENWEATHER_API_KEY set in .env
- NEWS_API_KEY set in .env
"""

import os
import math
import requests
import wikipedia
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from agent_utils import create_llm, run_agent, run_interactive_loop
from langchain.agents import create_agent

# ============================================================================
# TOOL 1: Calculator
# ============================================================================
class CalculatorInput(BaseModel):
    expression: str = Field(description="A VALID Python mathematical expression to evaluate (e.g., '0.15 * 1250'). Do NOT use %, currency symbols, or words.")

@tool(args_schema=CalculatorInput)
def safe_calculator(expression: str) -> str:
    """Evaluates a mathematical expression and returns the result. Use for ANY math."""
    try:
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"

# ============================================================================
# TOOL 2: Weather API
# ============================================================================
class WeatherInput(BaseModel):
    city: str = Field(description="The name of the city to get the weather for (e.g., 'London')")

@tool(args_schema=WeatherInput)
def get_weather(city: str) -> str:
    """Fetches the current live weather for a specified city using OpenWeatherMap."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key or api_key == "your_openweather_api_key_here":
        return "ERROR: OPENWEATHER_API_KEY is not set in .env."
        
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    try:
        response = requests.get(base_url, params={"q": city, "appid": api_key, "units": "metric"}, timeout=5)
        response.raise_for_status()
        data = response.json()
        return f"Weather in {city}: {data['weather'][0]['description']}, {data['main']['temp']}°C"
    except Exception as e:
        return f"Weather API Error: {str(e)}"

# ============================================================================
# TOOL 3: News API
# ============================================================================
class NewsInput(BaseModel):
    topic: str = Field(description="The topic to search for news about")
    max_results: int = Field(default=3, description="Maximum number of articles to return")

@tool(args_schema=NewsInput)
def get_recent_news(topic: str, max_results: int = 3) -> str:
    """Fetches the latest news headlines for a specific topic using NewsAPI."""
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key or api_key == "your_news_api_key_here":
        return "ERROR: NEWS_API_KEY is not set in .env."
        
    url = "https://newsapi.org/v2/everything"
    try:
        response = requests.get(url, params={"q": topic, "language": "en", "sortBy": "publishedAt", "pageSize": max_results, "apiKey": api_key}, timeout=5)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        if not articles: return f"No news found for '{topic}'"
        
        return "\n".join([f"- {a.get('title')} ({a.get('source', {}).get('name')})" for a in articles])
    except Exception as e:
        return f"News API Error: {str(e)}"

# ============================================================================
# TOOL 4: Wikipedia Search
# ============================================================================
class WikipediaInput(BaseModel):
    query: str = Field(description="The topic to look up on Wikipedia")

@tool(args_schema=WikipediaInput)
def search_wikipedia(query: str) -> str:
    """Looks up a topic on Wikipedia and returns a summary. Use for historical/factual info."""
    try:
        return wikipedia.summary(query, sentences=3, auto_suggest=False)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Ambiguous query. Try one of: {', '.join(e.options[:5])}"
    except Exception as e:
        return f"Wikipedia Error: {str(e)}"


def get_all_tools():
    """Returns the list of all tools in the registry."""
    return [
        safe_calculator, 
        get_weather, 
        get_recent_news, 
        search_wikipedia
    ]

def create_registry_agent():
    """Create a ReAct agent injected with ALL our tools."""
    llm = create_llm()
    
    # We pass the entire toolbox to the Agent
    tools = get_all_tools()
    
    system_prompt = """
    You are a powerful, multi-tool AI assistant.
    You have access to a calculator, live weather, live news, and Wikipedia.
    
    CRITICAL RULE: You MUST use your tools to answer the user's questions. 
    Do not guess math, weather, news, or factual history. 
    Select the BEST tool for the job based on the user's query.
    If a user asks a multi-part question, you can use MULTIPLE tools sequentially!
    
    NEVER output raw JSON blocks. ALWAYS use the native tool calling capability.
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
    print("Tool Registry - Comprehensive Tool Selection")
    print("=" * 80)
    print("\n🤖 Creating Multi-Tool Agent...")
    agent = create_registry_agent()
    print("✅ Agent created with 4 tools (Calculator, Weather, News, Wikipedia)!\n")
    
    test_cases = [
        "What is 15% of 1250?", # Should route to Calculator
        "Is it raining in Seattle right now?", # Should route to Weather
        "What is the history of the Eiffel Tower?", # Should route to Wikipedia
        "What is the current temperature in Paris, and what is the square root of that number?", # Multi-tool routing!
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
