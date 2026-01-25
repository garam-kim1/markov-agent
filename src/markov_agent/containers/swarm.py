from collections.abc import Callable
from typing import Any

from markov_agent.topology.edge import Edge
from markov_agent.topology.graph import Graph
from markov_agent.topology.node import BaseNode


class Swarm(Graph):
    """
    Supervisor/Worker pattern.
    """

    def __init__(
        self,
        supervisor: BaseNode,
        workers: list[BaseNode],
        router_func: Callable[[Any], str],
        name: str = "Swarm",
        state_type: type | None = None,
        **kwargs,
    ):
        nodes = {node.name: node for node in [supervisor] + workers}

        edges = [Edge(source=supervisor.name, target_func=router_func)]

        for worker in workers:
            # Default return to supervisor
            edges.append(
                Edge(source=worker.name, target_func=lambda s, sup=supervisor.name: sup)
            )

        super().__init__(
            name=name,
            nodes=nodes,
            edges=edges,
            entry_point=supervisor.name,
            state_type=state_type,
            **kwargs,
        )
