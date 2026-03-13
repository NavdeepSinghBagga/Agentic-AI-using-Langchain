"""
Shared Agent Utilities Library
===============================
Common utilities used across all demo files:
- Agent creation with standard config
- Two output modes: 'stream' (real-time) and 'summary' (post-execution)
- Configurable via AGENT_OUTPUT_MODE env variable

Usage:
    from agent_utils import create_llm, run_agent_with_display
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# ============================================================================
# Observability (Arize Phoenix)
# ============================================================================
PHOENIX_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
if PHOENIX_ENDPOINT:
    try:
        from phoenix.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor
        
        # Register Phoenix as the OpenTelemetry endpoint
        tracer_provider = register(
            project_name=os.getenv("PHOENIX_PROJECT_NAME", "agent_presentation"),
            endpoint=PHOENIX_ENDPOINT,
        )
        # Instrument LangChain
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        print("✅ Arize Phoenix tracing enabled!")
    except ImportError:
        print("⚠️ Phoenix not installed. Run: pip install arize-phoenix openinference-instrumentation-langchain")
        
from langchain_ollama import ChatOllama
# The following imports are not part of the original file, but are included in the provided snippet.
# To maintain syntactic correctness and match the provided snippet, they are added here.
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.prebuilt import create_react_agent

# ============================================================================
# CONFIGURATION (shared across all files)
# ============================================================================
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"

# Output mode: "stream" = real-time step-by-step, "summary" = post-execution flow
AGENT_OUTPUT_MODE = os.getenv("AGENT_OUTPUT_MODE", "stream")


def create_llm(temperature=0):
    """Create a ChatOllama instance with standard configuration."""
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
    )


# ============================================================================
# OUTPUT MODE 1: STREAM — Real-time step-by-step (great for live demos)
# ============================================================================
def _run_stream(agent, question, config=None):
    """Stream agent execution, printing each step as it happens."""
    print("\n📋 Agent Execution (streaming):")
    print("─" * 60)

    final_answer = None
    step_num = 0

    for step in agent.stream({"messages": [{"role": "user", "content": question}]}, config=config):
        for node_name, node_output in step.items():
            # Skip non-dict outputs or empty message blocks (common with middleware)
            if not isinstance(node_output, dict) or not node_output.get("messages"):
                continue
                
            for msg in node_output.get("messages", []):
                # Skip internal LangGraph utility messages like RemoveMessage
                if isinstance(msg, RemoveMessage):
                    continue
                    
                # Tool call by the model
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        step_num += 1
                        print(f"\n  ⚡ Step {step_num}: Agent calls tool")
                        print(f"     🔧 Tool:  {tc['name']}")
                        print(f"     📥 Input: {tc['args']}")

                # Tool result
                elif msg.type == "tool":
                    print(f"     📤 Output: {msg.content}")

                # Final AI answer
                elif msg.type == "ai" and msg.content:
                    final_answer = msg.content
                    # print(f"\n  🤖 Agent: {msg.content}")

    print("\n" + "─" * 60)
    return final_answer


# ============================================================================
# OUTPUT MODE 2: SUMMARY — Post-execution formatted flow
# ============================================================================
def _run_summary(agent, question, config=None):
    """Run agent and display a summary of the execution."""
    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config=config)
    messages = result.get("messages", [])

    print("\n📋 Agent Execution (summary):")
    print("─" * 60)

    step_num = 0
    final_answer = None

    for msg in messages:
        if msg.type == "human":
            print(f"\n  👤 User: {msg.content}")

        elif msg.type == "ai" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                step_num += 1
                print(f"\n  ⚡ Step {step_num}: Agent calls tool")
                print(f"     🔧 Tool:  {tc['name']}")
                print(f"     📥 Input: {tc['args']}")

        elif msg.type == "tool":
            print(f"     📤 Output: {msg.content}")

        elif msg.type == "ai" and msg.content:
            final_answer = msg.content
            # print(f"\n  🤖 Agent: {msg.content}")

    print("\n" + "─" * 60)
    return final_answer


# ============================================================================
# PUBLIC API
# ============================================================================
def run_agent(agent, question, mode=None, config=None):
    """
    Run an agent and display results using the configured output mode.

    Args:
        agent: A LangGraph agent created with create_agent()
        question: The user query string
        mode: Optional override for AGENT_OUTPUT_MODE ("stream" or "summary")
        config: Optional LangGraph config dictionary (e.g., for thread_id)
    """
    mode = mode or AGENT_OUTPUT_MODE
    
    if mode == "stream":
        return _run_stream(agent, question, config)
    elif mode == "summary":
        return _run_summary(agent, question, config)
    else:
        print(f"⚠️  Unknown AGENT_OUTPUT_MODE '{mode}', defaulting to 'stream'")
        return _run_stream(agent, question, config)


def run_interactive_loop(agent):
    """
    Runs an interactive chat loop with the given agent.
    Allows the presenter/audience to ask arbitrary questions.
    """
    print("\n" + "=" * 80)
    print("🎮 INTERACTIVE MODE: Your turn!")
    print("Type 'exit' or 'quit' to end the session.")
    print("=" * 80 + "\n")

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ['exit', 'quit']:
                print("\n👋 Ending interactive session. Goodbye!\n")
                break

            answer = run_agent(agent, user_input)
            print(f"\n✅ Answer: {answer}\n")

        except KeyboardInterrupt:
            print("\n👋 Ending interactive session. Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")
