import pytest
from pydantic import Field

from markov_agent.core.state import BaseState
from markov_agent.topology.graph import Graph


class WorkflowState(BaseState):
    discussion_count: int = 0
    max_discussions: int = 2
    discussion_history: list[str] = Field(default_factory=list)
    a_and_b_done: bool = False
    c_result: str = ""
    needs_deep_thinking: bool = False


@pytest.mark.asyncio
async def test_complex_agent_workflow():
    # Setup the graph
    graph = Graph(name="ComplexWorkflow", state_type=WorkflowState)

    @graph.task
    def agent_a(state: WorkflowState) -> WorkflowState:
        history = list(state.discussion_history)
        history.append("Agent A thinks...")
        return state.update(discussion_history=history)

    @graph.task
    def agent_b(state: WorkflowState) -> WorkflowState:
        history = list(state.discussion_history)
        history.append("Agent B responds...")
        count = state.discussion_count + 1
        # Stop discussing when max discussions is reached
        done = count >= state.max_discussions
        return state.update(
            discussion_history=history, discussion_count=count, a_and_b_done=done
        )

    @graph.task
    def agent_c(state: WorkflowState) -> WorkflowState:
        return state.update(c_result="C summarized")

    @graph.task
    def agent_d(state: WorkflowState) -> WorkflowState:
        # Agent D acts as the orchestrator/reviewer
        # For this test, D will ask for deep thinking one time.
        # It asks for a retry if total discussion count < 4
        needs_deep = state.discussion_count < 4
        if needs_deep:
            # We must reset a_and_b_done so they can discuss again
            return state.update(needs_deep_thinking=True, a_and_b_done=False)
        return state.update(needs_deep_thinking=False)

    @graph.task
    def final_output(state: WorkflowState) -> WorkflowState:
        history = list(state.discussion_history)
        history.append("Final Output Reached")
        return state.update(discussion_history=history)

    # Topology definition
    graph.entry_point = "agent_a"

    # A -> B
    graph.add_transition("agent_a", "agent_b")

    # B -> A (if not done) or B -> C (if done)
    graph.add_transition("agent_b", "agent_a", condition=lambda s: not s.a_and_b_done)
    graph.add_transition("agent_b", "agent_c", condition=lambda s: s.a_and_b_done)

    # C -> D
    graph.add_transition("agent_c", "agent_d")

    # D -> A (if needs deep thinking) or D -> final_output (if not)
    graph.add_transition(
        "agent_d", "agent_a", condition=lambda s: s.needs_deep_thinking
    )
    graph.add_transition(
        "agent_d", "final_output", condition=lambda s: not s.needs_deep_thinking
    )

    initial_state = WorkflowState(max_discussions=2)
    final_state = await graph.run(initial_state)

    # Assertions
    # First loop: A -> B -> A -> B (count=2, done=True)
    # Then -> C -> D (needs_deep_thinking=True since count=2 < 4)
    # Reset a_and_b_done=False, goes to A
    # Second loop: A -> B -> A -> B (count=4, done=True)
    # Then -> C -> D (needs_deep_thinking=False since count=4 >= 4)
    # Then -> final_output

    assert final_state.discussion_count == 4
    assert final_state.c_result == "C summarized"
    assert "Final Output Reached" in final_state.discussion_history
    assert len(final_state.discussion_history) == 9  # 4 * 2 (A and B pairs) + 1 (final)
