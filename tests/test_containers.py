import pytest

from markov_agent.containers.chain import Chain
from markov_agent.containers.swarm import Swarm
from markov_agent.core.state import BaseState
from markov_agent.topology.node import BaseNode


class ContainerState(BaseState):
    value: str = ""
    next_worker: str = ""


class SimpleNode(BaseNode[ContainerState]):
    async def execute(self, state: ContainerState) -> ContainerState:
        return state.update(value=state.value + self.name)


@pytest.mark.asyncio
async def test_chain_container():
    node_a = SimpleNode(name="A")
    node_b = SimpleNode(name="B")
    node_c = SimpleNode(name="C")

    chain = Chain(nodes=[node_a, node_b, node_c])

    initial_state = ContainerState()
    final_state = await chain.run(initial_state)

    assert final_state.value == "ABC"


@pytest.mark.asyncio
async def test_swarm_container():
    class Supervisor(BaseNode[ContainerState]):
        async def execute(self, state: ContainerState) -> ContainerState:
            # Simple supervisor that just marks it was here
            return state.update(value=state.value + "S")

    supervisor = Supervisor(name="Supervisor")
    worker_a = SimpleNode(name="A")
    worker_b = SimpleNode(name="B")

    def router(state: ContainerState) -> str:
        if state.next_worker == "A":
            return "A"
        if state.next_worker == "B":
            return "B"
        return "END"  # Terminate

    swarm = Swarm(
        supervisor=supervisor,
        workers=[worker_a, worker_b],
        router_func=router,
        max_steps=10,
    )

    # 1. Start with A
    # Step 1: Supervisor executes -> value="S", router(S) -> "A"
    # Step 2: Worker A executes -> value="SA", router(A) -> "Supervisor"
    # Step 3: Supervisor executes -> value="SAS", router(SAS) -> "A" ...
    # Wait, the router needs to change the state to terminate.

    class SmartSupervisor(BaseNode[ContainerState]):
        async def execute(self, state: ContainerState) -> ContainerState:
            new_val = state.value + "S"
            # If we already have A, terminate
            if "A" in state.value:
                return state.update(value=new_val, next_worker="DONE")
            return state.update(value=new_val)

    swarm = Swarm(
        supervisor=SmartSupervisor(name="Supervisor"),
        workers=[worker_a, worker_b],
        router_func=router,
        max_steps=10,
    )

    final_state = await swarm.run(ContainerState(next_worker="A"))
    # S -> A -> S (done) -> terminate
    assert "SA" in final_state.value
    assert final_state.value == "SAS"  # Supervisor (S) -> A -> Supervisor (S)
