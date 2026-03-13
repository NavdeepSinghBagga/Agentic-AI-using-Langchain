"""
Phase 2: Tool Registry
02_tool_weather.py - Weather API Integration

This script demonstrates how to integrate a real REST API as a tool for the agent.
It wraps the OpenWeatherMap API output in a structured Pydantic schema
and properly handles common API errors (timeouts, missing keys, city not found).

Prerequisites:
- OPENWEATHER_API_KEY set in .env
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

class WeatherInput(BaseModel):
    """Input schema for the weather search tool."""
    city: str = Field(
        description="The name of the city to get the weather for (e.g., 'London', 'Tokyo, JP')"
    )

@tool(args_schema=WeatherInput)
def get_weather(city: str) -> str:
    """
    Fetches the current weather for a specified city using OpenWeatherMap.
    Returns temperature, description, and humidity.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key or api_key == "your_openweather_api_key_here":
        return "ERROR: OPENWEATHER_API_KEY is not set or is invalid in the .env file."
        
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric" # Output in Celsius
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # Parse the JSON response
        weather_desc = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        country = data["sys"]["country"]
        
        return f"Current weather in {city}, {country}: {weather_desc}, Temperature: {temp}°C, Humidity: {humidity}%"
        
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        if status_code == 404:
            return f"Weather ERROR: City '{city}' not found. Please try another city."
        elif status_code == 401:
            return "Weather ERROR: Unauthorized. Please check your OpenWeatherMap API key."
        return f"Weather API HTTP Error: {status_code}"
    except requests.exceptions.RequestException as e:
        return f"Weather API Connection Error: {str(e)}"

# ============================================================================
# AGENT SETUP
# ============================================================================

def create_weather_agent():
    """Create a ReAct agent injected with our weather tool."""
    llm = create_llm()
    tools = [get_weather]
    
    system_prompt = """
    You are a helpful meteorologist assistant.
    You have access to a live weather tool. 
    Always use the weather tool to get the CURRENT weather. Never guess or hallucinate the weather.
    If the weather tool returns an error, inform the user about the error.
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
    print("Tool Registry - Live Weather API Integration")
    print("=" * 80)
    
    print("\n🤖 Creating Weather Agent...")
    agent = create_weather_agent()
    print("✅ Agent created with `get_weather` tool!\n")
    
    test_cases = [
        "What is the current weather in London?",
        "How is the weather looking in Tokyo right now?",
        "What's the weather in FakeCityThatDoesNotExist123?", # Should trigger 404 handling
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
