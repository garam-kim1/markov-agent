import pytest

from markov_agent.core.state import BaseState
from markov_agent.simulation.twin import BaseDigitalTwin
from markov_agent.topology.evolution import TopologyOptimizer
from markov_agent.topology.graph import Graph


class ResourceState(BaseState):
    resources: int = 100
    consumed: int = 0
    task_completed: bool = False


class ResourceTwin(BaseDigitalTwin):
    async def validate_transition(
        self, current: ResourceState, proposed: ResourceState
    ) -> bool:
        # Rule: Cannot consume more resources than available
        if proposed.resources < 0:
            return False
        # Rule: Resources must decrease if consumed increases
        return not (
            proposed.consumed > current.consumed
            and proposed.resources >= current.resources
        )


@pytest.mark.asyncio
async def test_neuro_symbolic_shell_enforcement():
    """Test that the Digital Twin (Shell) prevents invalid state transitions."""
    graph = Graph(name="ResourceAgent", state_type=ResourceState)

    @graph.task
    def valid_task(state: ResourceState):
        return state.update(
            resources=state.resources - 10, consumed=state.consumed + 10
        )

    @graph.task
    def invalid_task(state: ResourceState):
        # Violates rule: consume more than available
        return state.update(resources=-10, consumed=200)

    graph.add_transition("valid_task", "invalid_task")

    # Set the twin
    graph.twin = ResourceTwin()

    initial_state = ResourceState()
    # Step 1: Execute valid_task
    # We'll use graph.run which uses the internal async implementation
    final_state = await graph.run(initial_state)

    # The execution should have stopped at invalid_task because it was reverted
    assert final_state.resources == 90
    assert final_state.consumed == 10
    # The invalid_task should have been reverted, so we remain at the state after valid_task
    # (or rather, the state before invalid_task execution)


@pytest.mark.asyncio
async def test_reward_based_evolution():
    """Test that TopologyOptimizer can update edge weights based on rewards."""
    graph = Graph(name="EvolutionAgent", state_type=BaseState)

    @graph.task
    def start(state: BaseState):
        return state

    @graph.task
    def path_a(state: BaseState):
        s = state.update()
        s.record_reward(1.0)
        return s

    @graph.task
    def path_b(state: BaseState):
        s = state.update()
        s.record_reward(-1.0)
        return s

    graph.add_transition("start", "path_a", weight=1.0)
    graph.add_transition("start", "path_b", weight=1.0)

    optimizer = TopologyOptimizer(graph)

    # Mock simulation results
    from markov_agent.simulation.runner import SimulationResult

    results = [
        SimulationResult(
            final_state=BaseState(reward=1.0),
            trajectory=[{"node": "start"}, {"node": "path_a"}],
            success=True,
        ),
        SimulationResult(
            final_state=BaseState(reward=-1.0),
            trajectory=[{"node": "start"}, {"node": "path_b"}],
            success=False,
        ),
    ]

    updates = optimizer.learn_from_rewards(results, learning_rate=0.5)

    # Path A should have increased weight (1.0 * (1 + 0.5 * 1.0) = 1.5)
    # Path B should have decreased weight (1.0 * (1 + 0.5 * -1.0) = 0.5)

    assert updates[("start", "path_a")] == 1.5
    assert updates[("start", "path_b")] == 0.5

    # Verify weights in graph
    edge_a = next(
        e for e in graph.edges if e.source == "start" and e.target == "path_a"
    )
    edge_b = next(
        e for e in graph.edges if e.source == "start" and e.target == "path_b"
    )

    assert edge_a.weight == 1.5
    assert edge_b.weight == 0.5
