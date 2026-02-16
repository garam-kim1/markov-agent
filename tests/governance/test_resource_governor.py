from unittest.mock import patch

import pytest

from markov_agent.core.state import BaseState
from markov_agent.governance.resource import ResourceGovernor
from markov_agent.topology.graph import Graph


class SimpleState(BaseState):
    value: int = 0


@patch("psutil.virtual_memory")
def test_resource_governor_safe(mock_vm):
    # Mock safe memory usage (10%)
    mock_vm.return_value.percent = 10.0
    gov = ResourceGovernor(memory_threshold_percent=80.0)
    assert gov.check_safety() is True
    gov.enforce()  # Should not raise


@patch("psutil.virtual_memory")
def test_resource_governor_unsafe(mock_vm):
    # Mock unsafe memory usage (90%)
    mock_vm.return_value.percent = 90.0
    gov = ResourceGovernor(memory_threshold_percent=80.0)
    assert gov.check_safety() is False
    with pytest.raises(MemoryError, match="Resource limit exceeded"):
        gov.enforce()


@pytest.mark.asyncio
@patch("psutil.virtual_memory")
async def test_graph_enforces_governor(mock_vm):
    # Mock unsafe memory
    mock_vm.return_value.percent = 95.0
    gov = ResourceGovernor(memory_threshold_percent=80.0)

    graph = Graph(name="TestGraph", governor=gov)

    @graph.task(name="task1")
    def task1(state: SimpleState):
        return state.update(value=1)

    graph.entry_point = "task1"
    state = SimpleState()

    with pytest.raises(MemoryError):
        await graph.run(state)
