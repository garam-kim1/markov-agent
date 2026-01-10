from collections.abc import Callable
from typing import TypeVar

from markov_agent.core.state import BaseState

StateT = TypeVar("StateT", bound=BaseState)

# Edge is a router function: func(state) -> next_node_id
RouterFunction = Callable[[StateT], str]


class Edge:
    """
    Defines a transition between nodes.
    Although often just a function, this class can hold metadata about the transition.
    """

    def __init__(self, source: str, target_func: RouterFunction):
        self.source = source
        self.target_func = target_func

    def route(self, state: StateT) -> str:
        return self.target_func(state)
