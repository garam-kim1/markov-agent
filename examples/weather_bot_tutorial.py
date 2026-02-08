import asyncio
import os

from markov_agent import Agent, model_config

"""
Progressive Weather Bot Tutorial
--------------------------------
This tutorial demonstrates building a multi-agent team using Markov Agent's
ADK-compatible API.

Phases:
1. Single Agent with Tools
2. State and Memory
3. Multi-Agent Orchestration (Router Pattern)
4. Evaluation
"""

# --- Phase 1: Tools ---

def get_weather(city: str) -> str:
    """Returns weather data for a given city."""
    data = {"Tokyo": "sunny", "London": "rainy", "Seattle": "cloudy"}
    return data.get(city, "unknown")

# --- Phase 2: State and Dynamic Instructions ---

def weather_instruction_provider(context) -> str:
    """Example of a dynamic system prompt based on session state."""
    # context is an ADK ReadonlyContext
    user_name = context.session.state.get("user_name", "Explorer")
    return f"You are a helpful weather assistant. Greeting {user_name}. Provide concise updates."

# --- Phase 3: Multi-Agent Orchestration ---

# 1. Specialist Agent
weather_agent = Agent(
    name="weather_worker",
    model=model_config(name="gemini-1.5-flash"),
    system_prompt="You provide weather updates using the get_weather tool."
)
weather_agent.add_tool(get_weather)

# 2. Handoff Tool
def call_weather_agent(query: str) -> str:
    """Delegates weather-related questions to the specialist Weather Agent."""
    # This demonstrates agent-to-agent delegation
    response = weather_agent.run(query)
    return response.text

# 3. Router Agent
router_agent = Agent(
    name="router",
    model=model_config(name="gemini-1.5-flash"),
    system_prompt=(
        "You are a receptionist. If the user asks about weather, use the "
        "call_weather_agent tool. Otherwise, answer directly and politely."
    )
)
router_agent.add_tool(call_weather_agent)

# --- Phase 4: Evaluation ---

evaluator_agent = Agent(
    name="evaluator",
    model=model_config(name="gemini-1.5-flash"),
    system_prompt=(
        "You are a quality control agent. Evaluate if the agent's response "
        "accurately answers the user's question. Output 'PASS' or 'FAIL' "
        "followed by a brief reason."
    )
)

async def run_weather_pipeline(query: str, user_name: str = "Alice"):
    print(f"\n[User ({user_name})]: {query}")

    # We can set initial state for Phase 2 dynamic greeting
    # Note: Agent.run uses ADKController.run which creates a temporary session.
    # For state persistence across agents in a team, one would usually use a Graph.

    # Run the router
    response = router_agent.run(query)
    print(f"[Router]: {response.text}")

    # Evaluate the result
    eval_query = (
        f"Question: {query}\n"
        f"Response: {response.text}"
    )
    evaluation = evaluator_agent.run(eval_query)
    print(f"[Evaluator]: {evaluation.text}")

async def main():
    # Check for API keys
    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        print("Error: GOOGLE_API_KEY or GEMINI_API_KEY not set.")
        return

    print("--- Starting Progressive Weather Bot Demo ---")

    # Test 1: General Chat
    await run_weather_pipeline("Hello, who are you?")

    # Test 2: Weather Query (Triggers Delegation)
    await run_weather_pipeline("What is the weather in Tokyo?")

    # Test 3: Unknown City
    await run_weather_pipeline("How is the weather in Atlantis?")

if __name__ == "__main__":
    asyncio.run(main())
