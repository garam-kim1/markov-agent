import asyncio
import os

from markov_agent.core.state import BaseState
from markov_agent.engine.adk_wrapper import ADKConfig
from markov_agent.engine.ppu import ProbabilisticNode
from markov_agent.topology.graph import Graph

# --- 1. Define State ---


class SimpleState(BaseState):
    """A simple state to track the user query and the model's response."""

    query: str
    response: str | None = None


# --- 2. Define Node ---


class GeminiNode(ProbabilisticNode[SimpleState]):
    """A specialized node that uses Gemini to process the query."""

    def parse_result(self, state: SimpleState, result: str) -> SimpleState:
        # Standard implementation to store the LLM output in the state
        state.response = result
        state.record_step({"node": self.name, "output": "Query processed by Gemini"})
        return state


# --- 3. Main Execution ---


async def main():
    # Retrieve the API key from the environment variable GEMINI_API_KEY
    # as requested by the user.
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Usage: export GEMINI_API_KEY='your-google-api-key'")
        return

    # Configuration for the gemini-3-flash-preview model.
    # We pass the api_key explicitly to the ADKConfig.
    config = ADKConfig(
        model_name="gemini-3-flash-preview",
        api_key=api_key,
        temperature=0.7,
    )

    # Initialize the Probabilistic Processing Unit (PPU)
    node = GeminiNode(
        name="gemini_flash_ppu",
        adk_config=config,
        prompt_template="Explain the following concept simply: {{ query }}",
        state_type=SimpleState,
    )

    # Define the topology (a single-node graph for this example)
    graph = Graph(
        name="gemini_flash_graph",
        nodes={node.name: node},
        edges=[],
        entry_point=node.name,
        state_type=SimpleState,
    )

    # Define initial input
    initial_state = SimpleState(query="Markov Chains in AI")

    print(f"--- Running Markov Agent with {config.model_name} ---")

    try:
        # Run the graph
        final_state = await graph.run(initial_state)

        print("\n--- Final Output ---")
        print(f"Concept: {final_state.query}")
        print(f"Explanation:\n{final_state.response}")
        print(f"\nSteps taken: {len(final_state.history)}")

    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        print(
            "\nPlease ensure your GEMINI_API_KEY is valid and "
            "you have access to the requested model.",
        )


if __name__ == "__main__":
    # Ensure we run in an async environment
    asyncio.run(main())
