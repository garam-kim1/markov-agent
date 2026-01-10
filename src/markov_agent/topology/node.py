from abc import ABC, abstractmethod

from markov_agent.core.state import BaseState


class BaseNode[StateT: BaseState](ABC):
    """
    Abstract Base Node.
    Must define input_schema and output_schema.
    Receives State, performs work (potentially using ADK), and returns StateUpdate.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def execute(self, state: StateT) -> StateT:
        """
        Execute the node's logic.

        Args:
            state: The current global state.

        Returns:
            The updated state.
        """
        pass
