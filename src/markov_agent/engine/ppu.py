import contextlib
import logging
from collections.abc import AsyncGenerator, Callable
from typing import Any, TypeVar, cast

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
logger = logging.getLogger(__name__)


class ProbabilisticNode(BaseNode[StateT]):
    """A node that uses a Probabilistic Processing Unit (LLM via ADK).

    Determines the next state by sampling from the model.
    """

    adk_config: Any = Field(default=None)
    prompt_template: Any = Field(default="")
    output_schema: Any = Field(default=None)
    samples: Any = Field(default=1)
    sampling_strategy: SamplingStrategy = Field(default=SamplingStrategy.UNIFORM)
    selector: Any = Field(default=None)
    verifier_node: Any = Field(default=None)
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
        selector: Callable[[list[Any]], Any] | str | None = None,
        verifier_node: BaseNode | None = None,
        retry_policy: RetryPolicy | None = None,
        mock_responder: Callable[[str], Any] | None = None,
        state_updater: Callable[[Any, Any], Any] | None = None,
        state_type: type[StateT] | None = None,
        artifact_service: BaseArtifactService | None = None,
    ) -> None:
        super().__init__(name, state_type=state_type)
        self.adk_config = adk_config
        self.prompt_template = prompt_template
        self.output_schema = output_schema
        self.samples = samples
        self.sampling_strategy = sampling_strategy

        # Resolve Selector
        if isinstance(selector, str):
            from markov_agent.engine.selectors import SELECTOR_REGISTRY

            if selector in SELECTOR_REGISTRY:
                # If it's a class/instance with a .select method, wrap it
                sel_obj = SELECTOR_REGISTRY[selector]
                if hasattr(sel_obj, "select"):
                    self.selector = sel_obj.select
                else:
                    self.selector = sel_obj
            else:
                msg = f"Selector alias '{selector}' not found in registry."
                raise ValueError(msg)
        else:
            self.selector = selector

        self.verifier_node = verifier_node
        self.retry_policy = retry_policy or RetryPolicy()
        self.state_updater = state_updater
        self.prompt_engine = PromptEngine()
        self.artifact_service = artifact_service
        self.mock_responder = mock_responder

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
            name=self.name,
        )

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        """Execute the PPU logic within the ADK runtime."""
        logger.info("Node '%s' starting execution.", self.name)
        # 1. Access State (Dict or Typed)
        # ctx.session.state is a dict
        state_dict = ctx.session.state

        # We try to use the typed state for prompt rendering if available
        state_obj = state_dict
        if self.state_type:
            try:
                state_obj = self.state_type.model_validate(state_dict)
            except Exception:
                with contextlib.suppress(Exception):
                    state_obj = self.state_type.construct(**state_dict)

        # 2. Render Prompt
        try:
            prompt = self._render_prompt(state_obj)
        except Exception:
            logger.exception("Node '%s' failed to render prompt", self.name)
            raise

        # 3. Generate Varied Configs (Explore/Exploit Strategy)
        base_gen_config = self._get_base_gen_config()
        varied_configs = generate_varied_configs(
            base_gen_config,
            self.samples,
            self.sampling_strategy,
        )

        # 4. Create Generation Tasks (Factories)
        task_factories = self._create_task_factories(prompt, state_dict, varied_configs)

        # 5. Execute Parallel Sampling
        logger.debug(
            "Node '%s' executing %s parallel samples.", self.name, self.samples
        )
        try:
            results = await execute_parallel_sampling(
                generate_func=task_factories,
                k=self.samples,
                selector_func=lambda x: x,  # Get all results to perform verification
            )
        except Exception:
            logger.exception("Node '%s' parallel sampling failed", self.name)
            raise

        # Calculate Sample Confidence (Ratio of selected result in samples)
        # This is useful for Majority Voting.
        # We perform selection and verification
        try:
            result = await self._verify_results(ctx, results)
        except Exception:
            logger.exception("Node '%s' verification/selection failed", self.name)
            raise

        # Determine selection confidence and distribution (for Entropy)
        selection_confidence = 1.0
        distribution = None
        if self.samples > 1 and results:
            # Count how many samples match each result to form a distribution
            try:

                def _norm(val: Any) -> Any:
                    if isinstance(val, BaseModel):
                        return val.model_dump_json()
                    if isinstance(val, (dict, list)):
                        import json

                        return json.dumps(val, sort_keys=True)
                    return val

                norm_results = [_norm(r) for r in results]
                from collections import Counter

                counts = Counter(norm_results)
                total = len(results)
                distribution = {str(k): v / total for k, v in counts.items()}

                target = _norm(result)
                selection_confidence = counts.get(target, 0) / total
            except Exception:
                selection_confidence = 1.0 / len(results)

        # 6. Update State
        if hasattr(state_obj, "record_probability") and callable(
            state_obj.record_probability
        ):
            # This records the confidence of the PPU in its own selection
            cast("Any", state_obj).record_probability(
                source=f"{self.name}_ppu",
                probability=selection_confidence,
                distribution=distribution,
            )
            # Sync back
            if hasattr(state_obj, "meta"):
                ctx.session.state["meta"] = cast("Any", state_obj).meta

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
                with contextlib.suppress(Exception):
                    updated_state = self.parse_result(state_obj, result)
                    if isinstance(updated_state, BaseModel):
                        ctx.session.state.update(updated_state.model_dump())
                        used_parse_result = True

            if not used_parse_result:
                if "history" not in ctx.session.state:
                    ctx.session.state["history"] = []
                ctx.session.state["history"].append(
                    {"node": self.name, "output": output_payload},
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
        """Legacy/Convenience wrapper.

        Runs logic directly on the State object, bypassing ADK runner.
        Respects SamplingStrategy.
        """
        return await self._execute_impl(state)

    async def _execute_impl(self, state: StateT) -> StateT:
        logger.info("Node '%s' starting execution (legacy).", self.name)
        try:
            prompt = self._render_prompt(state)
            state_dict = (
                state.model_dump() if isinstance(state, BaseModel) else dict(state)
            )

            base_gen_config = self._get_base_gen_config()
            varied_configs = generate_varied_configs(
                base_gen_config,
                self.samples,
                self.sampling_strategy,
            )

            task_factories = self._create_task_factories(
                prompt, state_dict, varied_configs
            )

            logger.debug(
                "Node '%s' executing %s parallel samples.", self.name, self.samples
            )
            result = await execute_parallel_sampling(
                generate_func=task_factories,
                k=self.samples,
                selector_func=self.selector,
            )

            return self.parse_result(state, result)
        except Exception:
            logger.exception("Node '%s' execution failed", self.name)
            raise

    async def deep(self, state: StateT) -> StateT:
        """System 2 reasoning (Think-then-Act).

        1. Generates internal reasoning/thought process.
        2. Executes the primary task with reasoning context.
        """
        # Phase 1: Reasoning
        base_prompt = self._render_prompt(state)
        thinking_prompt = (
            f"{base_prompt}\n\n"
            "SYSTEM INSTRUCTION: Before providing the final output, "
            "perform a deep step-by-step reasoning analysis. "
            "Identify constraints, edge cases, and the best strategy."
        )

        # Use a text-only controller for reasoning
        thought_ctrl = self.controller.create_variant(
            {"response_mime_type": "text/plain"}
        )
        reasoning = await thought_ctrl.generate(
            thinking_prompt,
            output_schema=None,
            initial_state=state.model_dump() if isinstance(state, BaseModel) else state,
        )

        # Phase 2: Action (with reasoning)
        # We inject reasoning into the prompt via a temporary state field
        if isinstance(state, BaseModel):
            rich_state = state.model_copy(update={"reasoning": reasoning})
        else:
            rich_state = {**state, "reasoning": reasoning}

        # Temporarily enrich prompt template
        original_template = self.prompt_template
        self.prompt_template = (
            f"{original_template}\n\n"
            "INTERNAL REASONING (System 2):\n"
            "{{ reasoning }}\n\n"
            "Now, generate the final output according to the requested schema."
        )

        try:
            return await self._execute_impl(rich_state)
        finally:
            self.prompt_template = original_template

    def _get_base_gen_config(self) -> dict[str, Any]:
        base_gen_config = self.adk_config.generation_config or {}
        if "temperature" not in base_gen_config:
            base_gen_config["temperature"] = self.adk_config.temperature
        if (
            "top_p" not in base_gen_config
            and getattr(self.adk_config, "top_p", None) is not None
        ):
            base_gen_config["top_p"] = self.adk_config.top_p
        return base_gen_config

    def _create_task_factories(
        self,
        prompt: str,
        state_dict: dict[str, Any],
        varied_configs: list[dict[str, Any]],
    ) -> list[Callable[[], Any]]:
        task_factories = []
        for cfg in varied_configs:
            if self.sampling_strategy == SamplingStrategy.UNIFORM:
                controller_to_use = self.controller
            else:
                controller_to_use = self.controller.create_variant(cfg)

            def make_task(c: ADKController = controller_to_use) -> Any:
                return c.generate(
                    prompt,
                    output_schema=self.output_schema,
                    initial_state=state_dict,
                    include_state=False,
                )

            task_factories.append(make_task)
        return task_factories

    async def _verify_results(self, ctx: InvocationContext, results: list[Any]) -> Any:
        if self.verifier_node and isinstance(results, list) and len(results) > 1:
            best_result = results[0]
            for candidate in results:
                temp_state = ctx.session.state.copy()
                temp_state["candidate"] = (
                    candidate.model_dump()
                    if isinstance(candidate, BaseModel)
                    else candidate
                )
                if hasattr(self.verifier_node, "execute"):
                    v_node: Any = self.verifier_node
                    try:
                        v_state = await v_node.execute(temp_state)
                        if v_state.get("verified", False):
                            best_result = candidate
                            break
                    except Exception:  # noqa: S112
                        continue
            return best_result
        if self.selector and isinstance(results, list):
            return self.selector(results)
        if isinstance(results, list):
            return results[0]
        return results

    def _render_prompt(self, state: StateT | dict[str, Any]) -> str:
        render_kwargs = {}

        # Try to convert dict to model if we have a state_type
        state_obj = state
        if isinstance(state, dict) and self.state_type:
            try:
                state_obj = self.state_type.model_validate(state)
            except Exception:
                with contextlib.suppress(Exception):
                    state_obj = self.state_type.construct(**state)

        if isinstance(state_obj, BaseModel):
            render_kwargs.update(dict(state_obj))
            render_kwargs["state"] = state_obj
        elif isinstance(state_obj, dict):
            render_kwargs.update(state_obj)

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

        # Record in history
        state.record_step({"node": self.name, "output": output_payload})

        # Smart Mapping Logic
        if isinstance(output_payload, dict):
            update_data = {}
            matched_any = False

            # If we have an output schema and state type, we can be smart
            if (
                self.output_schema
                and hasattr(self.output_schema, "model_fields")
                and self.state_type
                and hasattr(self.state_type, "model_fields")
            ):
                output_fields = self.output_schema.model_fields
                state_fields = self.state_type.model_fields

                for field_name in output_fields:
                    if field_name in state_fields:
                        update_data[field_name] = output_payload.get(field_name)
                        matched_any = True

                if not matched_any:
                    logger.warning(
                        "Node '%s': No matching fields found between output_schema and state_type, "
                        "and no state_updater was provided. Falling back to dict merge.",
                        self.name,
                    )
                    update_data = output_payload
            else:
                # Fallback to current behavior if schema info is missing
                update_data = output_payload

            # We use state.update which now handles 'append' behavior
            return state.update(**update_data)

        return state
