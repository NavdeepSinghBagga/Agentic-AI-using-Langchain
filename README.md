# Agentic-AI-using-Langchain

An AI agent implementation using LangChain and LangGraph with multiple tools and memory capabilities.

## Features

- **ReAct Agent**: Basic, intermediate, and advanced implementations
- **Multiple Tools**: Weather, News, Wikipedia integration
- **Memory Management**: Basic and window-based memory systems
- **Long-term Memory**: File-based and database storage
- **Production Ready**: Complete integration with observability

## Requirements

- Python 3.10+
- Ollama (for local LLM)
- API Keys for OpenWeatherMap and NewsAPI

## Installation

1. Clone the repository:
```bash
git clone https://github.com/NavdeepSinghBagga/Agentic-AI-using-Langchain.git
cd Agentic-AI-using-Langchain
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your API keys

## Usage

Run the different agent implementations:

```bash
# Basic ReAct agent
python 01_react_basic.py

# Agent with tools
python 02_tool_weather.py

# Agent with memory
python 03_memory_basic.py

# Production-ready agent
python 06_production_unified.py
```

## Project Structure

- `01_react_*.py` - ReAct agent implementations
- `02_tool_*.py` - Tool integration examples
- `03_memory_*.py` - Memory management
- `04_longterm_*.py` - Long-term memory storage
- `05_integration_complete.py` - Complete integration
- `06_production_unified.py` - Production-ready implementation
- `agent_utils.py` - Utility functions

## License

MIT License