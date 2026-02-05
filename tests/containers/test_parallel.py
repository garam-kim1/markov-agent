import pytest

from markov_agent.containers.parallel import ParallelNode
from markov_agent.core.state import BaseState
from markov_agent.topology.node import BaseNode


class ParallelState(BaseState):
    val: int = 0
    list_a: list[str] = []
    list_b: list[str] = []


class SetterNode(BaseNode[ParallelState]):
    def __init__(
        self,
        name: str,
        val: int | None = None,
        append_a: str | None = None,
        append_b: str | None = None,
    ):
        super().__init__(name)
        self.val = val
        self.append_a = append_a
        self.append_b = append_b

    async def execute(self, state: ParallelState) -> ParallelState:
        updates = {}
        if self.val is not None:
            updates["val"] = self.val

        if self.append_a:
            updates["list_a"] = [*state.list_a, self.append_a]

        if self.append_b:
            updates["list_b"] = [*state.list_b, self.append_b]

        return state.update(**updates)


@pytest.mark.asyncio
async def test_parallel_independent():
    """Test parallel execution where nodes touch different parts of state."""
    node_a = SetterNode("A", append_a="A")
    node_b = SetterNode("B", append_b="B")

    parallel = ParallelNode("Par", nodes=[node_a, node_b])

    initial = ParallelState()
    result = await parallel.execute(initial)

    assert result.list_a == ["A"]
    assert result.list_b == ["B"]


@pytest.mark.asyncio
async def test_parallel_conflict():
    """Test conflict resolution (last node wins)."""
    node_a = SetterNode("A", val=10)
    node_b = SetterNode("B", val=20)

    # A then B -> B wins
    parallel = ParallelNode("Par", nodes=[node_a, node_b])

    initial = ParallelState(val=0)
    result = await parallel.execute(initial)

    assert result.val == 20

    # Swap order: B then A -> A wins
    parallel_rev = ParallelNode("ParRev", nodes=[node_b, node_a])
    result_rev = await parallel_rev.execute(initial)
    assert result_rev.val == 10
