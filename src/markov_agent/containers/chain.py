from typing import Any

from markov_agent.topology.edge import Edge
from markov_agent.topology.graph import Graph
from markov_agent.topology.node import BaseNode


class Chain(Graph):
    """Standard Linear Chain pattern A -> B -> C.

    Automatically creates edges between the provided list of nodes.
    """

    def __init__(
        self,
        nodes: list[BaseNode],
        name: str = "Chain",
        state_type: type | None = None,
        **kwargs: Any,
    ):
        node_dict = {node.name: node for node in nodes}
        edges = []

        for i in range(len(nodes) - 1):
            source = nodes[i].name
            target = nodes[i + 1].name
            # Linear router: always goes to the next node
            edges.append(Edge(source=source, target_func=lambda s, t=target: t))

        entry_point = nodes[0].name if nodes else ""

        super().__init__(
            name=name,
            nodes=node_dict,
            edges=edges,
            entry_point=entry_point,
            state_type=state_type,
            **kwargs,
        )
