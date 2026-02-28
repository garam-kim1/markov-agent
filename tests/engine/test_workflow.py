import pytest

from markov_agent import Agent, BaseState, Workflow
from markov_agent.topology.node import FunctionalNode


class SimpleState(BaseState):
    query: str = ""
    result: str = ""


@pytest.mark.asyncio
async def test_workflow_from_chain():
    def node_1(state: SimpleState) -> SimpleState:
        return state.update(result="Node 1 executed")

    def node_2(state: SimpleState) -> SimpleState:
        return state.update(result=state.result + " -> Node 2 executed")

    workflow = Workflow.from_chain([node_1, node_2], state_type=SimpleState)
    assert len(workflow.nodes) == 2
    assert len(workflow.edges) == 1

    final_state = await workflow.run(SimpleState(query="test"))
    assert final_state.result == "Node 1 executed -> Node 2 executed"


@pytest.mark.asyncio
async def test_workflow_fluent_api():
    node_a = FunctionalNode(
        name="NodeA", func=lambda s: s.update(result="A"), state_type=SimpleState
    )
    node_b = FunctionalNode(
        name="NodeB",
        func=lambda s: s.update(result=s.result + "B"),
        state_type=SimpleState,
    )

    flow = node_a >> node_b
    workflow = Workflow(flow=flow, state_type=SimpleState)

    assert "NodeA" in workflow.nodes
    assert "NodeB" in workflow.nodes
    assert workflow.entry_point == "NodeA"

    final_state = await workflow.run(SimpleState())
    assert final_state.result == "AB"


@pytest.mark.asyncio
async def test_workflow_fluent_api_with_agents():
    agent_1 = Agent(name="agent1", model="mock-model")
    agent_2 = Agent(name="agent2", model="mock-model")

    # Mock the responder for predictable output
    agent_1.mock_responder = lambda _: "Agent1 Output"
    agent_2.mock_responder = lambda _: "Agent2 Output"

    flow = agent_1 >> agent_2
    workflow = Workflow(flow=flow)

    assert "agent1" in workflow.nodes
    assert "agent2" in workflow.nodes
    assert workflow.entry_point == "agent1"
