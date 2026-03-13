"""
Phase 2: Tool Registry
02_tool_news.py - News API Integration

This script demonstrates how to integrate the NewsAPI to fetch recent articles.
It highlights using multiple parameters in the Pydantic schema (topic, max_results).

Prerequisites:
- NEWS_API_KEY set in .env
"""

import os
import requests
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from agent_utils import create_llm, run_agent, run_interactive_loop
from langchain.agents import create_agent

# ============================================================================
# TOOL SCHEMAS AND DEFINITIONS
# ============================================================================

class NewsInput(BaseModel):
    """Input schema for the news search tool."""
    topic: str = Field(
        description="The topic or keyword to search for news about (e.g., 'artificial intelligence', 'Apple')."
    )
    max_results: int = Field(
        default=3,
        description="Maximum number of articles to return. Keep between 1 and 5 to avoid context overflow."
    )

@tool(args_schema=NewsInput)
def get_recent_news(topic: str, max_results: int = 3) -> str:
    """
    Fetches the latest news headlines for a specific topic using NewsAPI.
    """
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key or api_key == "your_news_api_key_here":
        return "ERROR: NEWS_API_KEY is not set or is invalid in the .env file."
        
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": max_results,
        "apiKey": api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        articles = data.get("articles", [])
        if not articles:
            return f"No news articles found for topic: '{topic}'"
            
        result_lines = [f"Found {len(articles)} recent articles about '{topic}':\n"]
        for i, article in enumerate(articles, 1):
            title = article.get("title", "No Title")
            source = article.get("source", {}).get("name", "Unknown Source")
            desc = article.get("description", "No description available")
            url = article.get("url", "#")
            
            result_lines.append(f"{i}. {title} ({source})")
            result_lines.append(f"   Summary: {desc}")
            result_lines.append(f"   Link: {url}\n")
            
        return "\n".join(result_lines)
        
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        if status_code == 401:
            return "News ERROR: Unauthorized. Please check your NewsAPI key."
        elif status_code == 429:
            return "News ERROR: Rate limit exceeded for NewsAPI."
        return f"News API HTTP Error: {status_code}"
    except requests.exceptions.RequestException as e:
        return f"News API Connection Error: {str(e)}"

# ============================================================================
# AGENT SETUP
# ============================================================================

def create_news_agent():
    """Create a ReAct agent injected with our news tool."""
    llm = create_llm()
    tools = [get_recent_news]
    
    system_prompt = """
    You are a helpful news researcher.
    You have access to a live news search tool.
    When a user asks about current events, use the `get_recent_news` tool to fetch recent articles.
    Summarize the top articles neatly into a bulleted list for the user.
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
    print("Tool Registry - Live News API Integration")
    print("=" * 80)
    
    print("\n🤖 Creating News Agent...")
    agent = create_news_agent()
    print("✅ Agent created with `get_recent_news` tool!\n")
    
    test_cases = [
        "What are the latest developments in quantum computing? Keep it brief.",
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
