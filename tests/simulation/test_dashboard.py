from unittest.mock import MagicMock, patch

import pytest

from markov_agent import BaseState, Graph
from markov_agent.simulation.dashboard import DashboardRunner


class SimpleState(BaseState):
    val: int = 0


@pytest.mark.asyncio
async def test_dashboard_runner():
    graph = Graph("TestGraph", state_type=SimpleState)

    # Use task decorator directly
    @graph.task
    def increment(state: SimpleState):
        return state.update(val=state.val + 1)

    graph.add_transition("increment", None)  # Terminal

    state = SimpleState()

    # Mock Live context manager to avoid actual rendering
    with patch("markov_agent.simulation.dashboard.Live") as mock_live:
        mock_live.return_value.__enter__.return_value = MagicMock()

        # Run with 0 delay for speed
        runner = DashboardRunner(graph, state, delay=0)
        final_state = await runner.run()

        assert final_state.val == 1
        assert runner.step_count > 0
        assert "Simulation complete." in runner.logs
