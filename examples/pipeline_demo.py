import asyncio

from markov_agent import Agent, BaseState, Workflow
from markov_agent.topology.node import FunctionalNode


# Optional: Define a state (or just use dicts)
class PipelineState(BaseState):
    topic: str = ""
    research: str = ""
    draft: str = ""
    status: str = "pending"


async def main():
    print("--- Markov Agent Workflow Demo ---")

    # 1. Create Agents
    researcher = Agent(
        name="researcher",
        model="gemini-3-flash-preview",
        system_prompt="You are a researcher. Given a topic, provide bullet points.",
        mock_responder=lambda p: "1. Quantum superposition\\n2. Entanglement",
        state_updater=lambda state, result: state.update(research=result),
    )

    writer = Agent(
        name="writer",
        model="gemini-3-flash-preview",
        system_prompt="You are a writer. Draft an article based on research.",
        mock_responder=lambda p: (
            "Quantum computing is fascinating. It uses superposition and entanglement."
        ),
        state_updater=lambda state, result: state.update(draft=result),
    )

    # 2. Create custom functional nodes to glue things together
    extract_topic = FunctionalNode(
        name="extract_topic",
        func=lambda state: state,  # Or just print
    )

    def do_extract(state: PipelineState) -> PipelineState:
        print(f"Extracting topic: {state.topic}")
        return state

    extract_topic.func = do_extract

    def do_publish(state: PipelineState) -> PipelineState:
        print(f"\\n--- PUBLISHED DRAFT ---\\n{state.draft}\\n-----------------------")
        return state.update(status="published")

    publish = FunctionalNode(name="publish", func=do_publish)

    # 3. Build the Workflow using the Fluent API
    # Extract -> Researcher -> Writer -> Publish
    pipeline_flow = extract_topic >> researcher >> writer >> publish

    # 4. Initialize and run
    workflow = Workflow(
        name="ArticlePipeline", flow=pipeline_flow, state_type=PipelineState
    )

    initial_state = PipelineState(topic="Quantum Computing")

    print("\\nRunning the workflow...")
    final_state = await workflow.run(initial_state)

    print(f"\\nFinal Status: {final_state.status}")


if __name__ == "__main__":
    asyncio.run(main())
