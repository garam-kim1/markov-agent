from collections.abc import AsyncGenerator, Callable
from typing import Any, TypeVar, cast

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types
from pydantic import Field

from markov_agent.core.state import BaseState
from markov_agent.engine.adk_wrapper import ADKConfig, ADKController, RetryPolicy
from markov_agent.topology.node import BaseNode

StateT = TypeVar("StateT", bound=BaseState)


class EntropyCheck(BaseNode[StateT]):
    """Calculates uncertainty (Entropy) of the current state/query.

    Transitions to clarification if uncertainty is high.
    """

    threshold: float = Field(default=0.5)
    adk_config: ADKConfig
    clarification_node: str = Field(default="clarification")
    execution_node: str = Field(default="execution")

    def __init__(
        self,
        name: str,
        adk_config: ADKConfig,
        threshold: float = 0.5,
        clarification_node: str = "clarification",
        execution_node: str = "execution",
        state_type: type[StateT] | None = None,
        mock_responder: Callable[[str], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, adk_config=adk_config, state_type=state_type, **kwargs
        )
        self.adk_config = adk_config
        self.threshold = threshold
        self.clarification_node = clarification_node
        self.execution_node = execution_node
        self.controller = ADKController(
            self.adk_config, RetryPolicy(), mock_responder=mock_responder
        )

    async def execute(self, state: StateT) -> StateT:
        """Estimate entropy and decide next step."""
        entropy = await self._calculate_entropy(state)

        # Transition logic based on threshold
        next_node = (
            self.clarification_node if entropy > self.threshold else self.execution_node
        )

        # Update state with entropy and decision
        # We assume the state might have these fields or use generic update
        if hasattr(state, "update"):
            return state.update(entropy_score=entropy, next_step=next_node)

        # Fallback if it's a dict or similar (though StateT is bound to BaseState)
        return state

    async def _calculate_entropy(self, state: StateT) -> float:
        """Use a lightweight prompt to estimate Shannon Entropy/Missing Information."""
        state_str = (
            str(state.model_dump()) if hasattr(state, "model_dump") else str(state)
        )

        prompt = f"""Analyze the following state/query for missing information or ambiguity:
'{state_str}'

Estimate the 'Missing Information' on a scale from 0.0 (Perfectly Clear) to 1.0 (Highly Ambiguous/Missing Info). Return only the number."""

        try:
            result = await self.controller.generate(prompt)
            # Basic parsing
            import re

            result_str = str(result)
            match = re.search(r"(\d+(\.\d+)?)", result_str)
            if match:
                return float(match.group(1))
        except Exception:
            return 0.5
        return 0.5

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        """ADK Runtime implementation."""
        state_dict = ctx.session.state
        if self.state_type:
            state_obj = self.state_type.model_validate(state_dict)
        else:
            state_obj = cast("StateT", state_dict)

        new_state = await self.execute(state_obj)

        # Update Session State
        if hasattr(new_state, "model_dump"):
            ctx.session.state.update(new_state.model_dump())
        elif isinstance(new_state, dict):
            ctx.session.state.update(new_state)

        entropy = getattr(new_state, "entropy_score", 0.0)
        next_step = getattr(new_state, "next_step", "unknown")

        yield Event(
            author=self.name,
            actions=EventActions(),
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text=f"Entropy Check: {entropy:.2f}. Recommended path: {next_step}"
                    )
                ],
            ),
        )
