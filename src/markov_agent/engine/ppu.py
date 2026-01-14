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
        state_updater=None,
    ):
        super().__init__(name)
        self.adk_config = adk_config
        self.prompt_template = prompt_template
        self.output_schema = output_schema
        self.samples = samples
        self.retry_policy = retry_policy or RetryPolicy()
        self.state_updater = state_updater

        # Inject native JSON support if schema is provided
        if self.output_schema:
            if self.adk_config.generation_config is None:
                self.adk_config.generation_config = {}

            # Configure native structured output
            self.adk_config.generation_config["response_mime_type"] = "application/json"
            # google-adk/genai often accepts the pydantic class or its schema
            self.adk_config.generation_config["response_schema"] = self.output_schema

        self.controller = ADKController(
            self.adk_config,
            self.retry_policy,
            mock_responder=mock_responder,
            output_schema=self.output_schema,
        )

    async def _run_async_impl(self, context: Any) -> Any:
        """
        Executes the PPU logic within the ADK runtime.
        """
        from google.adk.events import Event, EventActions
        
        # 1. Access State (Dict)
        # context.session.state is a dict
        state_dict = context.session.state
        
        # 2. Render Prompt
        # We need to render the prompt using the dictionary state
        try:
            prompt = self.prompt_template.format(**state_dict)
        except Exception:
            prompt = self.prompt_template

        # 3. Define generation task
        async def generate_task():
            return await self.controller.generate(
                prompt, output_schema=self.output_schema
            )
            
        # 4. Execute Parallel Sampling
        # This returns the best result (type depends on output_schema)
        result = await execute_parallel_sampling(
            generate_func=generate_task, k=self.samples
        )
        
        # 5. Update State
        # If result is a Pydantic model, dump it.
        output_payload = result
        if isinstance(result, BaseModel):
            output_payload = result.model_dump()
            
        # If we have a custom state_updater, it usually expects (StateT, Result) -> StateT
        # This is tricky because we only have a dict here. 
        # For now, we assume standard behavior: update keys in state based on output_key?
        # Or we rely on the user to provide a dict-compatible updater?
        # Let's support the 'state_updater' if it can handle dicts, otherwise fallback.
        
        if self.state_updater:
             # Attempt to invoke updater. If it expects StateT, this might fail or require Proxy.
             # We'll assume the updater for the "Wrapper" version of the lib handles dicts.
             updated_state = self.state_updater(state_dict, result)
             # Update session state with the new dictionary
             if isinstance(updated_state, BaseModel):
                 context.session.state.update(updated_state.model_dump())
             elif isinstance(updated_state, dict):
                 context.session.state.update(updated_state)
        else:
             # Default: append to history if exists, or set output key if we had one (we don't have output_key field yet)
             # But BaseState has 'history'.
             if "history" not in context.session.state:
                 context.session.state["history"] = []
             context.session.state["history"].append({"node": self.name, "output": output_payload})
             
             # Also merge the result into state if it's a dict/model?
             if isinstance(output_payload, dict):
                 context.session.state.update(output_payload)

        # 6. Emit Event
        yield Event(
            author=self.name,
            actions=EventActions(), # No special actions
            payload={"output": output_payload} # Optional payload usage
        )

    async def execute(self, state: StateT) -> StateT:
        """
        Legacy/Convenience wrapper. 
        Warning: This bypasses the ADK Runner mechanics and runs logic directly on the State object.
        """
        # Mimic the logic for standalone usage
        prompt = self._render_prompt(state)
        
        async def generate_task():
            return await self.controller.generate(
                prompt, output_schema=self.output_schema
            )

        result = await execute_parallel_sampling(
            generate_func=generate_task, k=self.samples
        )
        
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
        Parses result. If state_updater is provided, uses it.
        Otherwise, default parser appends result to history.
        """
        # If a custom updater is provided, use it
        if self.state_updater:
            # The updater should return a NEW state instance (immutability)
            return self.state_updater(state, result)

        # Default behavior
        output_payload = result
        if isinstance(result, BaseModel):
            output_payload = result.model_dump()

        state.record_step({"node": self.name, "output": output_payload})
        return state
