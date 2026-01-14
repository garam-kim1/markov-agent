import pytest
from markov_agent.containers.parallel import ParallelNode
from markov_agent.containers.loop import LoopNode
from markov_agent.containers.sequential import SequentialNode
from markov_agent.containers.nested import NestedGraphNode
from markov_agent.containers.chain import Chain
from markov_agent.core.state import BaseState
from markov_agent.topology.node import BaseNode

class ParallelState(BaseState):
    branch_a: str = ""
    branch_b: str = ""
    shared: str = "start"

class AppendNode(BaseNode[ParallelState]):
    def __init__(self, name: str, field: str, value: str):
        super().__init__(name=name)
        self.field = field
        self.value = value

    async def execute(self, state: ParallelState) -> ParallelState:
        current = getattr(state, self.field)
        update_data = {self.field: current + self.value}
        return state.update(**update_data)

@pytest.mark.asyncio
async def test_nested_graph_node():
    node_a = AppendNode("A", "branch_a", "A")
    node_b = AppendNode("B", "branch_a", "B")
    
    # Chain is a Graph
    chain = Chain(nodes=[node_a, node_b])
    
    # Wrap Chain in NestedGraphNode
    nested = NestedGraphNode(name="NestedChain", graph=chain)
    
    initial = ParallelState()
    final = await nested.execute(initial)
    
    assert final.branch_a == "AB"

@pytest.mark.asyncio
async def test_sequential_node():
    node_a = AppendNode("A", "branch_a", "A")
    node_b = AppendNode("B", "branch_a", "B")
    
    seq = SequentialNode(name="Seq", nodes=[node_a, node_b])
    
    initial = ParallelState()
    final = await seq.execute(initial)
    
    assert final.branch_a == "AB"

@pytest.mark.asyncio
async def test_parallel_node():
    node_a = AppendNode("A", "branch_a", "A")
    node_b = AppendNode("B", "branch_b", "B")
    
    # Both run on the same initial state
    parallel = ParallelNode(name="Parallel", nodes=[node_a, node_b])
    
    initial = ParallelState()
    final = await parallel.execute(initial)
    
    assert final.branch_a == "A"
    assert final.branch_b == "B"
    assert final.shared == "start" # Should be unchanged

@pytest.mark.asyncio
async def test_loop_node():
    class LoopState(BaseState):
        count: int = 0
        
    class IncrementNode(BaseNode[LoopState]):
        async def execute(self, state: LoopState) -> LoopState:
            return state.update(count=state.count + 1)
            
    increment = IncrementNode("Inc")
    
    # Loop until count >= 5
    loop = LoopNode(
        name="Looper",
        body=increment,
        condition=lambda s: s.count >= 5,
        max_iterations=10
    )
    
    initial = LoopState(count=0)
    final = await loop.execute(initial)
    
    assert final.count == 5

@pytest.mark.asyncio
async def test_loop_max_iterations():
    class LoopState(BaseState):
        count: int = 0
        
    class IncrementNode(BaseNode[LoopState]):
        async def execute(self, state: LoopState) -> LoopState:
            return state.update(count=state.count + 1)
            
    increment = IncrementNode("Inc")
    
    # Loop max 3 times, condition never met
    loop = LoopNode(
        name="LooperMax",
        body=increment,
        condition=lambda s: False,
        max_iterations=3
    )
    
    initial = LoopState(count=0)
    final = await loop.execute(initial)
    
    assert final.count == 3
