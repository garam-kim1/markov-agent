from collections.abc import AsyncGenerator, Callable
from typing import Any, TypeVar

from google.adk.agents.invocation_context import InvocationContext
from google.adk.artifacts import BaseArtifactService
from google.adk.events import Event, EventActions
from google.genai import types
from pydantic import BaseModel, Field

from markov_agent.core.state import BaseState
from markov_agent.engine.adk_wrapper import ADKConfig, ADKController, RetryPolicy
from markov_agent.engine.prompt import PromptEngine
from markov_agent.engine.sampler import (
    SamplingStrategy,
    execute_parallel_sampling,
    generate_varied_configs,
)
from markov_agent.topology.node import BaseNode

StateT = TypeVar("StateT", bound=BaseState)


class ProbabilisticNode(BaseNode[StateT]):
    """
    A node that uses a Probabilistic Processing Unit (LLM via ADK)
    to determine the next state.
    """

    adk_config: Any = Field(default=None)
    prompt_template: Any = Field(default="")
    output_schema: Any = Field(default=None)
    samples: Any = Field(default=1)
    sampling_strategy: SamplingStrategy = Field(default=SamplingStrategy.UNIFORM)
    selector: Any = Field(default=None)
    retry_policy: Any = Field(default=None)
    mock_responder: Any = Field(default=None)
    state_updater: Any = Field(default=None)
    artifact_service: BaseArtifactService | None = Field(default=None)

    def __init__(
        self,
        name: str,
        adk_config: ADKConfig,
        prompt_template: str,
        output_schema: type[BaseModel] | None = None,
        samples: int = 1,
        sampling_strategy: SamplingStrategy = SamplingStrategy.UNIFORM,
        selector: Callable[[list[Any]], Any] | None = None,
        retry_policy: RetryPolicy | None = None,
        mock_responder=None,
        state_updater=None,
        state_type: type[StateT] | None = None,
        artifact_service: BaseArtifactService | None = None,
    ):
        super().__init__(name, state_type=state_type)
        self.adk_config = adk_config
        self.prompt_template = prompt_template
        self.output_schema = output_schema
        self.samples = samples
        self.sampling_strategy = sampling_strategy
        self.selector = selector
        self.retry_policy = retry_policy or RetryPolicy()
        self.state_updater = state_updater
        self.prompt_engine = PromptEngine()
        self.artifact_service = artifact_service

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
            artifact_service=self.artifact_service,
        )

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Executes the PPU logic within the ADK runtime.
        """

        # 1. Access State (Dict or Typed)
        # ctx.session.state is a dict
        state_dict = ctx.session.state

        # We try to use the typed state for prompt rendering if available
        state_obj = state_dict
        if self.state_type:
            try:
                state_obj = self.state_type.model_validate(state_dict)
            except Exception:
                try:
                    state_obj = self.state_type.construct(**state_dict)
                except Exception:
                    pass

        # 2. Render Prompt
        render_kwargs = {}
        if isinstance(state_obj, BaseModel):
            render_kwargs.update(dict(state_obj))
            render_kwargs["state"] = state_obj
        elif isinstance(state_dict, dict):
            render_kwargs.update(state_dict)

        try:
            prompt = self.prompt_engine.render(self.prompt_template, **render_kwargs)
        except Exception:
            raise

        # 3. Generate Varied Configs (Explore/Exploit Strategy)
        base_gen_config = self.adk_config.generation_config or {}
        # Ensure temperature is in base config if set at top level
        if "temperature" not in base_gen_config:
            base_gen_config["temperature"] = self.adk_config.temperature
        # Ensure top_p is in base config if set at top level
        # (needed for DIVERSE strategy)
        if (
            "top_p" not in base_gen_config
            and getattr(self.adk_config, "top_p", None) is not None
        ):
            base_gen_config["top_p"] = self.adk_config.top_p

        varied_configs = generate_varied_configs(
            base_gen_config, self.samples, self.sampling_strategy
        )

        # 4. Create Generation Tasks (Factories)
        task_factories = []
        for cfg in varied_configs:
            # If strategy is uniform, we can reuse the main controller.
            # Otherwise, create a variant.
            if self.sampling_strategy == SamplingStrategy.UNIFORM:
                controller_to_use = self.controller
            else:
                controller_to_use = self.controller.create_variant(cfg)

            # Closure to capture the specific controller
            def make_task(c=controller_to_use):
                return c.generate(
                    prompt,
                    output_schema=self.output_schema,
                    initial_state=state_dict,
                    include_state=False,
                )

            task_factories.append(make_task)

        # 5. Execute Parallel Sampling
        result = await execute_parallel_sampling(
            generate_func=task_factories, k=self.samples, selector_func=self.selector
        )

        # 6. Update State
        if self.adk_config.output_key and isinstance(result, (str, BaseModel)):
            val = result if isinstance(result, str) else result.model_dump_json()
            ctx.session.state[self.adk_config.output_key] = val

        output_payload = result
        if isinstance(result, BaseModel):
            output_payload = result.model_dump()

        if self.state_updater:
            if self.state_type and isinstance(state_obj, BaseModel):
                updated_state = self.state_updater(state_obj, result)
                if isinstance(updated_state, BaseModel):
                    ctx.session.state.update(updated_state.model_dump())
                elif isinstance(updated_state, dict):
                    ctx.session.state.update(updated_state)
            else:
                updated_state = self.state_updater(state_dict, result)
                if isinstance(updated_state, dict):
                    ctx.session.state.update(updated_state)
                elif isinstance(updated_state, BaseModel):
                    ctx.session.state.update(updated_state.model_dump())
        else:
            used_parse_result = False
            if self.state_type and isinstance(state_obj, BaseModel):
                try:
                    updated_state = self.parse_result(state_obj, result)
                    if isinstance(updated_state, BaseModel):
                        ctx.session.state.update(updated_state.model_dump())
                        used_parse_result = True
                except Exception:
                    pass

            if not used_parse_result:
                if "history" not in ctx.session.state:
                    ctx.session.state["history"] = []
                ctx.session.state["history"].append(
                    {"node": self.name, "output": output_payload}
                )
                if isinstance(output_payload, dict):
                    ctx.session.state.update(output_payload)

        # 7. Emit Event
        content_text = ""
        if isinstance(output_payload, (dict, list)):
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
            content=types.Content(role="model", parts=[types.Part(text=content_text)]),
        )

    async def execute(self, state: StateT) -> StateT:
        """
        Legacy/Convenience wrapper.
        Runs logic directly on the State object, bypassing ADK runner.
        Respects SamplingStrategy.
        """
        prompt = self._render_prompt(state)
        state_dict = state.model_dump() if isinstance(state, BaseModel) else dict(state)

        # Duplicate logic for configs (refactor target?)
        base_gen_config = self.adk_config.generation_config or {}
        if "temperature" not in base_gen_config:
            base_gen_config["temperature"] = self.adk_config.temperature
        if (
            "top_p" not in base_gen_config
            and getattr(self.adk_config, "top_p", None) is not None
        ):
            base_gen_config["top_p"] = self.adk_config.top_p

        varied_configs = generate_varied_configs(
            base_gen_config, self.samples, self.sampling_strategy
        )

        task_factories = []
        for cfg in varied_configs:
            if self.sampling_strategy == SamplingStrategy.UNIFORM:
                controller_to_use = self.controller
            else:
                controller_to_use = self.controller.create_variant(cfg)

            def make_task(c=controller_to_use):
                return c.generate(
                    prompt,
                    output_schema=self.output_schema,
                    initial_state=state_dict,
                    include_state=False,
                )

            task_factories.append(make_task)

        result = await execute_parallel_sampling(
            generate_func=task_factories, k=self.samples, selector_func=self.selector
        )

        return self.parse_result(state, result)

    def _render_prompt(self, state: StateT) -> str:
        render_kwargs = {}
        if isinstance(state, BaseModel):
            render_kwargs.update(dict(state))
            render_kwargs["state"] = state
        else:
            render_kwargs.update(state)

        try:
            return self.prompt_engine.render(self.prompt_template, **render_kwargs)
        except Exception:
            return self.prompt_template

    def parse_result(self, state: StateT, result: Any) -> StateT:
        if self.state_updater:
            return self.state_updater(state, result)

        output_payload = result
        if isinstance(result, BaseModel):
            output_payload = result.model_dump()

        state.record_step({"node": self.name, "output": output_payload})
        return state
