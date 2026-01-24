from collections.abc import AsyncGenerator, Callable
from typing import Any, TypeVar

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types
from pydantic import BaseModel

from markov_agent.core.state import BaseState
from markov_agent.engine.adk_wrapper import ADKConfig, ADKController, RetryPolicy
from markov_agent.engine.prompt import PromptEngine
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
        selector: Callable[[list[Any]], Any] = None,
        retry_policy: RetryPolicy = None,
        mock_responder=None,
        state_updater=None,
        state_type: type[StateT] | None = None,
    ):
        super().__init__(name, state_type=state_type)
        self.adk_config = adk_config
        self.prompt_template = prompt_template
        self.output_schema = output_schema
        self.samples = samples
        self.selector = selector
        self.retry_policy = retry_policy or RetryPolicy()
        self.state_updater = state_updater
        self.prompt_engine = PromptEngine()

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

    async def _run_async_impl(
        self, context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Executes the PPU logic within the ADK runtime.
        """

        # 1. Access State (Dict or Typed)
        # context.session.state is a dict
        state_dict = context.session.state

        # We try to use the typed state for prompt rendering if available
        state_obj = state_dict
        if self.state_type:
            try:
                state_obj = self.state_type.model_validate(state_dict)
            except Exception:
                # Fallback to construct if validation strictly fails (e.g. extra fields)
                # or just use the dict if we can't build the object.
                try:
                    state_obj = self.state_type.construct(**state_dict)
                except Exception:
                    pass

        # 2. Render Prompt
        render_kwargs = {}
        if isinstance(state_obj, BaseModel):
            # dict(model) iterates over fields and returns values as-is (preserving objects)
            # This is better than model_dump() which recursively converts to dicts
            render_kwargs.update(dict(state_obj))
            
            # Add method access if needed? Jinja2 can call methods if passed in context.
            # But dict(model) only gives fields.
            # We can pass the object itself as 'state' or 'self'?
            render_kwargs["state"] = state_obj
            # Also try to expose methods directly if possible, or user calls state.method()
        elif isinstance(state_dict, dict):
            render_kwargs.update(state_dict)

        try:
            prompt = self.prompt_engine.render(self.prompt_template, **render_kwargs)
        except Exception:
            # If formatting fails (e.g. missing key), we log or just use raw template
            # to avoid crashing, but usually we want to crash to alert the user.
            # We'll re-raise for visibility during dev
            raise

        # 3. Define generation task
        async def generate_task():
            return await self.controller.generate(
                prompt, output_schema=self.output_schema
            )

        # 4. Execute Parallel Sampling
        # This returns the best result (type depends on output_schema)
        result = await execute_parallel_sampling(
            generate_func=generate_task, k=self.samples, selector_func=self.selector
        )

        # 5. Update State
        output_payload = result
        if isinstance(result, BaseModel):
            output_payload = result.model_dump()

        if self.state_updater:
            # Use state_updater. It usually expects (StateT, Result) -> StateT
            # If we have a typed state object, use it.
            if self.state_type and isinstance(state_obj, BaseModel):
                updated_state = self.state_updater(state_obj, result)
                if isinstance(updated_state, BaseModel):
                    context.session.state.update(updated_state.model_dump())
                elif isinstance(updated_state, dict):
                    context.session.state.update(updated_state)
            else:
                # Fallback for dict
                updated_state = self.state_updater(state_dict, result)
                if isinstance(updated_state, dict):
                    context.session.state.update(updated_state)
                elif isinstance(updated_state, BaseModel):
                    context.session.state.update(updated_state.model_dump())
        else:
            # Try parse_result (for subclasses overriding it)
            # We only do this if we can form a valid state object, or if parse_result handles dicts?
            # Standard parse_result in ProbabilisticNode expects StateT.

            used_parse_result = False
            if self.state_type and isinstance(state_obj, BaseModel):
                try:
                    updated_state = self.parse_result(state_obj, result)
                    if isinstance(updated_state, BaseModel):
                        context.session.state.update(updated_state.model_dump())
                        used_parse_result = True
                except Exception:
                    # If parse_result fails (e.g. not implemented or type mismatch), fall through
                    pass

            if not used_parse_result:
                # Default: append to history if exists
                if "history" not in context.session.state:
                    context.session.state["history"] = []
                context.session.state["history"].append(
                    {"node": self.name, "output": output_payload}
                )

                # Also merge the result into state if it's a dict/model
                if isinstance(output_payload, dict):
                    context.session.state.update(output_payload)

        # 6. Emit Event
        # Populate content for ADK API Server compatibility
        content_text = ""
        if isinstance(output_payload, dict) or isinstance(output_payload, list):
             import json
             try:
                 content_text = json.dumps(output_payload, indent=2)
             except Exception:
                 content_text = str(output_payload)
        else:
             content_text = str(output_payload)

        yield Event(
            author=self.name,
            actions=EventActions(),
            content=types.Content(role="model", parts=[types.Part(text=content_text)])
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
            generate_func=generate_task, k=self.samples, selector_func=self.selector
        )

        return self.parse_result(state, result)

    def _render_prompt(self, state: StateT) -> str:
        # Simple format
        render_kwargs = {}
        if isinstance(state, BaseModel):
             render_kwargs.update(dict(state))
             render_kwargs["state"] = state
        else:
             render_kwargs.update(state)

        try:
            return self.prompt_engine.render(self.prompt_template, **render_kwargs)
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
