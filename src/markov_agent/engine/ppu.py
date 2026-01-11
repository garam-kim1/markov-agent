from typing import Any, TypeVar

from pydantic import BaseModel

from markov_agent.core.state import BaseState
from markov_agent.engine.adk_wrapper import ADKConfig, ADKController, RetryPolicy
from markov_agent.engine.sampler import execute_parallel_sampling
from markov_agent.topology.node import BaseNode

StateT = TypeVar("StateT", bound=BaseState)


class ProbabilisticNode(BaseNode[StateT]):
    """
    A node that uses a Probabilistic Processing Unit (LLM via ADK)
    to determine the next state.
    """

    def __init__(
        self,
        name: str,
        adk_config: ADKConfig,
        prompt_template: str,
        output_schema: type[BaseModel] | None = None,
        samples: int = 1,
        retry_policy: RetryPolicy = None,
        mock_responder=None,
    ):
        super().__init__(name)
        self.adk_config = adk_config
        self.prompt_template = prompt_template
        self.output_schema = output_schema
        self.samples = samples
        self.retry_policy = retry_policy or RetryPolicy()

        self.controller = ADKController(
            self.adk_config, self.retry_policy, mock_responder=mock_responder
        )

    async def execute(self, state: StateT) -> StateT:
        """
        Executes the PPU logic:
        1. Render prompt from state.
        2. Sample k trajectories.
        3. Select best result (defaulting to first valid for now).
        4. Update state.
        """
        # 1. Render Prompt
        prompt = self._render_prompt(state)

        # 2. Define the generation function for the sampler
        async def generate_task():
            return await self.controller.generate(
                prompt, output_schema=self.output_schema
            )

        # 3. Execute Parallel Sampling
        result = await execute_parallel_sampling(
            generate_func=generate_task, k=self.samples
        )

        # 4. Update State
        return self.parse_result(state, result)

    def _render_prompt(self, state: StateT) -> str:
        # Simple format
        try:
            return self.prompt_template.format(**state.model_dump())
        except Exception:
            # Fallback if state format fails
            return self.prompt_template

    def parse_result(self, state: StateT, result: Any) -> StateT:
        """
        Default parser: appends result to history.
        """
        output_payload = result
        if isinstance(result, BaseModel):
            output_payload = result.model_dump()

        state.record_step({"node": self.name, "output": output_payload})
        return state
