from collections.abc import AsyncGenerator
from typing import TypeVar

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from pydantic import BaseModel

from markov_agent.core.state import BaseState
from markov_agent.topology.node import BaseNode

StateT = TypeVar("StateT", bound=BaseState)


class CritiqueResult(BaseModel):
    is_valid: bool
    feedback: str


class SelfCorrectionNode(BaseNode[StateT]):
    """Implements a self-correction (Reflexion) loop.

    1. Executes primary node.
    2. Executes critique node (which must output a CritiqueResult).
    3. If not valid, executes correction node (or repeats primary with feedback) and loops.
    """

    def __init__(
        self,
        name: str,
        primary: BaseNode,
        critique: BaseNode,
        max_retries: int = 3,
        state_type: type[StateT] | None = None,
        feedback_key: str = "critique_feedback",
    ):
        super().__init__(name=name, state_type=state_type)
        self.primary = primary
        self.critique = critique
        self.max_retries = max_retries
        self.feedback_key = feedback_key

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        for i in range(self.max_retries + 1):
            # 1. Run Primary
            async for event in self.primary._run_async_impl(ctx):
                yield event

            # 2. Run Critique
            # We need to capture the output of critique to decide whether to continue
            async for event in self.critique._run_async_impl(ctx):
                yield event

            # 3. Check State for CritiqueResult
            # We assume critique node updates the state with CritiqueResult-like fields
            # or we can look for it in ctx.session.state

            # If the state has an 'is_valid' field (smart mapping should have put it there)
            is_valid = ctx.session.state.get("is_valid", True)
            feedback = ctx.session.state.get("feedback", "")

            if is_valid:
                break

            if i < self.max_retries:
                # Inject feedback into state for the next primary run
                ctx.session.state[self.feedback_key] = (
                    f"Previous attempt failed. Feedback: {feedback}"
                )
            else:
                # Max retries reached
                break

    async def execute(self, state: StateT) -> StateT:
        # For legacy/direct execution support
        current_state = state
        for i in range(self.max_retries + 1):
            current_state = await self.primary.execute(current_state)
            critique_state = await self.critique.execute(current_state)

            # Extract is_valid from critique_state
            is_valid = getattr(critique_state, "is_valid", True)
            feedback = getattr(critique_state, "feedback", "")

            if is_valid:
                return critique_state

            if i < self.max_retries:
                # Inject feedback
                current_state = critique_state.update(
                    **{
                        self.feedback_key: f"Previous attempt failed. Feedback: {feedback}"
                    }
                )
            else:
                return critique_state

        return current_state
