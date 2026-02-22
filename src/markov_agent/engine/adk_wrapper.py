import asyncio
import contextlib
import logging
import os
import uuid
from collections.abc import AsyncGenerator, Callable
from typing import Any, Self, TypeVar, override

from google.adk.agents import Agent, LiveRequestQueue
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.apps import App
from google.adk.artifacts import BaseArtifactService
from google.adk.events import Event
from google.adk.memory import BaseMemoryService
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import BaseSessionService
from google.adk.tools import load_memory as load_memory_tool
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.api_core import exceptions as google_exceptions
from google.genai import types
from json_repair import repair_json
from pydantic import BaseModel, ConfigDict, Field

from markov_agent.core.services import ServiceRegistry
from markov_agent.engine.plugins import BasePlugin
from markov_agent.engine.runtime import RunConfig
from markov_agent.engine.telemetry_plugin import MarkovBridgePlugin
from markov_agent.engine.token_utils import (
    ReductionStrategy,
    count_tokens,
    reduce_text_to_tokens,
)

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


class ADKConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: Any
    temperature: float = 0.7
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    max_tokens: int | None = None
    max_input_tokens: int | None = None
    reduction_strategy: ReductionStrategy = ReductionStrategy.GREEDY
    reduction_prompt: str | None = None
    reduction_model_name: str | None = None
    recency_weight: float = 2.0
    safety_settings: list[Any] = Field(default_factory=list)
    api_base: str | None = None
    api_key: str | None = None
    name: str | None = None
    tools: list[Any] = Field(default_factory=list)
    instruction: str | Callable[[ReadonlyContext], str] | None = None
    description: str | None = None
    generation_config: dict[str, Any] | None = None
    plugins: list[BasePlugin] = Field(default_factory=list)
    callbacks: list[Any] = Field(default_factory=list)
    mock_responder: Callable[[str], Any] | None = None
    use_litellm: bool = False
    output_key: str | None = None
    context_cache_config: Any | None = None
    events_compaction_config: Any | None = None
    enable_logging: bool = False
    enable_tracing: bool = False
    compress_state: bool = False
    session_service: BaseSessionService | None = None
    memory_service: BaseMemoryService | None = None
    artifact_service: BaseArtifactService | None = None
    enable_memory: bool = False
    enable_grounding: bool = False
    enable_code_execution: bool = False


class RetryPolicy(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_attempts: int = 3
    initial_delay: float = 0.5
    backoff_factor: float = 2.0


class MockLlm(BaseLlm):
    """Mock LLM implementation for testing."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    mock_responder: Callable[[str], Any] | None = None

    def __init__(self, mock_responder: Callable[[str], Any], model: str = "mock-model"):
        super().__init__(model=model)
        object.__setattr__(self, "mock_responder", mock_responder)

    @override
    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        # contents is likely LlmRequest object which has .contents list
        actual_contents = []
        if hasattr(llm_request, "contents"):
            actual_contents = llm_request.contents
        elif isinstance(llm_request, list):
            actual_contents = llm_request

        # Extract prompt from contents
        prompt = ""
        if actual_contents and actual_contents[-1].parts:
            prompt = "".join(p.text for p in actual_contents[-1].parts if p.text)

        assert self.mock_responder is not None
        result = self.mock_responder(prompt)
        if asyncio.iscoroutine(result):
            result = await result

        parts = []
        if isinstance(result, dict) and ("text" in result or "thought" in result):
            if "thought" in result:
                parts.append(types.Part(text=str(result["thought"]), thought=True))
            if "text" in result:
                parts.append(types.Part(text=str(result["text"])))
        else:
            parts.append(types.Part(text=str(result)))

        yield LlmResponse(content=types.Content(role="model", parts=parts))


class ResultWithReasoning(str):
    """A string that carries an optional reasoning/thought process."""

    __slots__ = ("_reasoning",)
    _reasoning: str | None

    def __new__(cls, value: str, reasoning: str | None = None) -> Self:
        instance = super().__new__(cls, value)
        object.__setattr__(instance, "_reasoning", reasoning)
        return instance

    @property
    def reasoning(self) -> str | None:
        return self._reasoning


class ADKController:
    """Wrapper around google_adk.Agent and Runner.

    Manages configuration, retries, and interaction with the underlying model.
    """

    def __init__(
        self,
        config: ADKConfig,
        retry_policy: RetryPolicy,
        mock_responder: Callable[[str], Any] | None = None,
        output_schema: Any | None = None,
        artifact_service: BaseArtifactService | None = None,
        name: str | None = None,
    ) -> None:
        self.config = config
        self.retry_policy = retry_policy
        self.mock_responder = mock_responder or config.mock_responder
        self.name = name or config.name or "agent"

        # Resolve Services using ServiceRegistry as fallback
        self.session_service = (
            config.session_service or ServiceRegistry.get_session_service()
        )
        self.memory_service = config.memory_service
        if self.memory_service is None and self.config.enable_memory:
            self.memory_service = ServiceRegistry.get_memory_service()

        self.artifact_service = (
            artifact_service
            or config.artifact_service
            or ServiceRegistry.get_artifact_service()
        )

        # Configure environment if needed
        api_key = (
            self.config.api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key

        # Prepare model_config from generation_config and top-level fields
        model_config = (self.config.generation_config or {}).copy()

        # Map basic fields
        if "temperature" not in model_config:
            model_config["temperature"] = self.config.temperature

        # Map other sampling parameters
        for field in [
            "top_p",
            "top_k",
            "min_p",
            "frequency_penalty",
            "presence_penalty",
        ]:
            val = getattr(self.config, field)
            if val is not None and field not in model_config:
                model_config[field] = val

        # Map max_tokens -> max_output_tokens
        # (Standardize on Google's name internally for Agent)
        if self.config.max_tokens is not None:
            if (
                "max_output_tokens" not in model_config
                and "max_tokens" not in model_config
            ):
                model_config["max_output_tokens"] = self.config.max_tokens
        elif "max_tokens" in model_config:
            # If user provided max_tokens in generation_config, rename it
            model_config["max_output_tokens"] = model_config.pop("max_tokens")

        # If response_schema made it into model_config (via PPU init), remove it
        # and use the explicit output_schema argument or the one from config
        if "response_schema" in model_config:
            if output_schema is None:
                output_schema = model_config.pop("response_schema")
            else:
                model_config.pop("response_schema")

        # Split config into Safe (for GenerateContentConfig) and Extra (for LiteLLM)
        # Based on google.genai.types.GenerateContentConfig and LiteLLM wrapper support
        safe_gen_config_keys = {
            "temperature",
            "top_p",
            "top_k",
            "candidate_count",
            "max_output_tokens",
            "stop_sequences",
            "presence_penalty",
            "frequency_penalty",
            "response_mime_type",
            "response_schema",
            "response_modalities",
            "speech_config",
            "seed",  # Usually supported by GenAI.
        }

        safe_config = {}
        extra_kwargs = {}

        for k, v in model_config.items():
            if k in safe_gen_config_keys:
                safe_config[k] = v
            else:
                extra_kwargs[k] = v

        # Model Initialization Logic
        model_instance = self.config.model_name
        use_litellm = self.config.use_litellm

        # Auto-detection for LiteLLM
        if (
            not use_litellm
            and isinstance(self.config.model_name, str)
            and (self.config.model_name.startswith("openai/") or self.config.api_base)
        ):
            use_litellm = True
            logger.info(
                "Auto-enabling LiteLLM for model '%s' (api_base: %s)",
                self.config.model_name,
                self.config.api_base,
            )

        if self.mock_responder:
            model_instance = MockLlm(self.mock_responder, model=self.config.model_name)
        elif use_litellm:
            from google.adk.models.lite_llm import LiteLlm

            # Setup environment for LiteLLM if api_base provided
            if self.config.api_base and isinstance(self.config.model_name, str):
                if self.config.model_name.startswith("openai/"):
                    # Assuming OpenAI-compatible local server if using openai/ prefix
                    os.environ["OPENAI_API_BASE"] = self.config.api_base
                else:
                    # For other models, we might need different env vars,
                    # but api_base is generic in ADKConfig
                    os.environ["LITELLM_API_BASE"] = self.config.api_base

                if self.config.api_key:
                    # Try to set appropriate key
                    if self.config.model_name.startswith("openai/"):
                        os.environ["OPENAI_API_KEY"] = self.config.api_key
                    else:
                        os.environ["LITELLM_API_KEY"] = self.config.api_key
                elif "OPENAI_API_KEY" not in os.environ:
                    os.environ["OPENAI_API_KEY"] = "dummy"

            # Pass extra_kwargs to LiteLlm constructor (e.g. min_p)
            model_instance = LiteLlm(model=self.config.model_name, **extra_kwargs)
        else:
            # For Google models, we ignore extra_kwargs or warn?
            # For now, we just drop them to prevent crashing GenerateContentConfig
            pass

        self.output_schema = output_schema

        tools = list(self.config.tools)
        if self.config.enable_memory:
            tools.append(load_memory_tool)

        if self.config.enable_grounding:
            tools.append(GoogleSearchTool())

        self.agent = Agent(
            name=self.name,
            model=model_instance,
            instruction=self.config.instruction or "",
            description=self.config.description or "",
            tools=tools,
            generate_content_config=types.GenerateContentConfig(**safe_config),
            output_schema=output_schema,
            output_key=self.config.output_key,
        )

        if self.config.enable_code_execution:
            # Inject Code Execution tool post-init to bypass LlmAgent validation
            # which might not support mixing server-side tools in constructor
            if self.agent.generate_content_config.tools is None:
                self.agent.generate_content_config.tools = []

            # Use ToolCodeExecution for server-side execution
            ce_tool = types.Tool(code_execution=types.ToolCodeExecution())
            self.agent.generate_content_config.tools.append(ce_tool)

        if self.config.enable_tracing:
            from markov_agent.engine.observability import configure_local_telemetry

            configure_local_telemetry()

        plugins = [MarkovBridgePlugin()]
        if self.config.enable_logging:
            from google.adk.plugins.logging_plugin import LoggingPlugin

            from markov_agent.engine.observability import configure_standard_logging

            configure_standard_logging()
            plugins.append(LoggingPlugin())

        if self.config.callbacks:
            from markov_agent.engine.callback_adapter import CallbackAdapterPlugin

            plugins.append(CallbackAdapterPlugin(self.config.callbacks))

        plugins.extend(self.config.plugins)

        self.app = App(
            name="markov_agent",
            root_agent=self.agent,
            plugins=plugins,
            context_cache_config=self.config.context_cache_config,
            events_compaction_config=self.config.events_compaction_config,
        )

        self.runner = Runner(
            app=self.app,
            session_service=self.session_service,
            artifact_service=self.artifact_service,
            memory_service=self.memory_service,
        )

    async def rewind(
        self,
        session_id: str,
        user_id: str = "system",
        rewind_before_invocation_id: str | None = None,
    ) -> None:
        """Rewind a session to a previous point."""
        kwargs = {
            "user_id": user_id,
            "session_id": session_id,
        }
        if rewind_before_invocation_id is not None:
            kwargs["rewind_before_invocation_id"] = rewind_before_invocation_id

        await self.runner.rewind_async(**kwargs)

    async def add_session_to_memory(
        self, session_id: str, user_id: str = "system"
    ) -> None:
        """Save a completed session into memory."""
        if not self.memory_service:
            msg = "Memory service not configured."
            raise RuntimeError(msg)

        session = await self.session_service.get_session(
            app_name="markov_agent",
            user_id=user_id,
            session_id=session_id,
        )
        if session:
            await self.memory_service.add_session_to_memory(session)

    def create_variant(
        self,
        generation_config_override: dict[str, Any],
        artifact_service: BaseArtifactService | None = None,
        session_service: BaseSessionService | None = None,
        memory_service: BaseMemoryService | None = None,
    ) -> "ADKController":
        """Create a new ADKController instance with specific overrides.

        Useful for adaptive sampling or sharing services across nodes.
        """
        new_config = self.config.model_copy(deep=True)
        if artifact_service:
            new_config.artifact_service = artifact_service
        if session_service:
            new_config.session_service = session_service
        if memory_service:
            new_config.memory_service = memory_service

        if new_config.generation_config is None:
            new_config.generation_config = {}

        new_config.generation_config.update(generation_config_override)

        # Sync top-level fields if they are in the override (convenience)
        # We iterate over all potential sampling fields
        for field in [
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "frequency_penalty",
            "presence_penalty",
            "max_tokens",
        ]:
            if field in generation_config_override:
                # Update the top-level config field to match the override
                # This ensures consistent state if checked elsewhere
                setattr(new_config, field, generation_config_override[field])

        return ADKController(
            config=new_config,
            retry_policy=self.retry_policy,
            mock_responder=self.mock_responder,
            output_schema=self.output_schema,
            artifact_service=artifact_service or self.artifact_service,
            name=self.name,
        )

    async def generate(
        self,
        prompt: str,
        output_schema: type[T] | None = None,
        initial_state: dict[str, Any] | None = None,
        *,
        include_state: bool = False,
        run_config: RunConfig | None = None,
        artifact_service: BaseArtifactService | None = None,
        session_service: BaseSessionService | None = None,
        memory_service: BaseMemoryService | None = None,
    ) -> Any | tuple[Any, dict[str, Any]]:
        """Generate content with retry logic using the ADK Runner.

        If output_schema is provided, attempts to generate and parse JSON.
        Returns the result, or a tuple of (result, updated_session_state)
        if include_state is True.
        """
        controller = self

        # Handle Context Reduction if max_input_tokens is set
        if self.config.max_input_tokens:
            prompt, initial_state = await self._reduce_context(
                prompt,
                initial_state,
                max_tokens=self.config.max_input_tokens,
                artifact_service=artifact_service or self.artifact_service,
                memory_service=memory_service or self.memory_service,
            )

        # Handle Overrides
        if (
            artifact_service
            or session_service
            or memory_service
            or (run_config and (run_config.model or run_config.tools))
        ):
            # Since create_variant logic is complex for top-level fields not in generation_config,
            # and we need to handle services, let's just create a new one if needed.
            new_config = self.config.model_copy(deep=True)
            if run_config and run_config.model:
                new_config.model_name = run_config.model
            if run_config and run_config.tools:
                new_config.tools = run_config.tools

            if artifact_service:
                new_config.artifact_service = artifact_service
            if session_service:
                new_config.session_service = session_service
            if memory_service:
                new_config.memory_service = memory_service

            controller = ADKController(
                config=new_config,
                retry_policy=self.retry_policy,
                mock_responder=self.mock_responder,
                output_schema=self.output_schema,
                artifact_service=artifact_service or self.artifact_service,
                name=self.name,
            )

        async def run_attempt() -> tuple[Any, dict[str, Any]]:
            final_prompt = prompt
            # Note: We do NOT manually inject the JSON schema into the prompt here.
            # The Agent is configured with output_schema, so the underlying model
            # should natively enforce the structure.

            session_id = f"gen_{uuid.uuid4().hex[:8]}"
            await controller.session_service.create_session(
                app_name="markov_agent",
                user_id=run_config.user_email
                if run_config and run_config.user_email
                else "system",
                session_id=session_id,
                state=initial_state or {},
            )

            content = types.Content(role="user", parts=[types.Part(text=final_prompt)])

            final_text = ""
            reasoning = ""
            adk_run_config = run_config.to_adk_run_config() if run_config else None
            async for event in controller.runner.run_async(
                user_id=run_config.user_email
                if run_config and run_config.user_email
                else "system",
                session_id=session_id,
                new_message=content,
                run_config=adk_run_config,
            ):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        parts = event.content.parts
                        # Separate thought parts (reasoning) from content
                        content_parts = [
                            p for p in parts if not getattr(p, "thought", False)
                        ]
                        thought_parts = [
                            p for p in parts if getattr(p, "thought", False)
                        ]

                        final_text = "".join(p.text for p in content_parts if p.text)
                        reasoning = "".join(p.text for p in thought_parts if p.text)
                    break

            # Retrieve final state
            final_session = await controller.session_service.get_session(
                app_name="markov_agent",
                user_id=run_config.user_email
                if run_config and run_config.user_email
                else "system",
                session_id=session_id,
            )
            final_state = final_session.state if final_session else {}

            # Inject reasoning into state meta if present
            if reasoning:
                if "meta" not in final_state:
                    final_state["meta"] = {}
                final_state["meta"]["reasoning"] = reasoning

            return final_text, final_state

        # Retry Loop
        attempt = 0
        current_delay = controller.retry_policy.initial_delay
        last_error = None

        while attempt < controller.retry_policy.max_attempts:
            try:
                if attempt > 0:
                    logger.info(
                        "Retrying generation (attempt %s/%s)...",
                        attempt + 1,
                        controller.retry_policy.max_attempts,
                    )
                raw_text, final_state = await run_attempt()

                # Parsing Logic
                result = raw_text
                if output_schema:
                    try:
                        cleaned_text = raw_text.strip()
                        # Use repair_json for more robustness
                        # We try with schema guidance first
                        try:
                            repaired_obj = repair_json(
                                cleaned_text.strip(),
                                return_objects=True,
                                schema=output_schema,
                            )
                        except Exception:
                            # Fallback to standard repair if guided fails
                            repaired_obj = repair_json(
                                cleaned_text.strip(),
                                return_objects=True,
                            )

                        if isinstance(repaired_obj, str) and repaired_obj == "":
                            # If it's still an empty string, try standard validation
                            # which might give a better error message or handle
                            # cases we didn't expect.
                            result = output_schema.model_validate_json(
                                cleaned_text.strip()
                            )
                        else:
                            result = output_schema.model_validate(repaired_obj)
                    except Exception:
                        logger.exception(
                            "Failed to parse model output as JSON (even with repair)"
                        )
                        logger.debug("Raw output: %s", raw_text)
                        raise

                # Attach reasoning to the result object
                reasoning = final_state.get("meta", {}).get("reasoning")
                if reasoning:
                    if isinstance(result, BaseModel):
                        with contextlib.suppress(Exception):
                            object.__setattr__(result, "_reasoning", reasoning)
                    elif isinstance(result, str):
                        result = ResultWithReasoning(result, reasoning)

                if include_state:
                    return result, final_state
                return result

            except (
                google_exceptions.ResourceExhausted,
                google_exceptions.ServiceUnavailable,
                google_exceptions.InternalServerError,
                google_exceptions.TooManyRequests,
            ) as e:
                last_error = e
                attempt += 1
                logger.warning(
                    "Generation attempt %s failed with transient error: %s: %s",
                    attempt,
                    type(e).__name__,
                    e,
                )
                if attempt < controller.retry_policy.max_attempts:
                    await asyncio.sleep(current_delay)
                    current_delay *= controller.retry_policy.backoff_factor

            except Exception as e:
                last_error = e
                attempt += 1
                logger.warning(
                    "Generation attempt %s failed: %s: %s",
                    attempt,
                    type(e).__name__,
                    e,
                )
                if attempt < controller.retry_policy.max_attempts:
                    await asyncio.sleep(current_delay)
                    current_delay *= controller.retry_policy.backoff_factor
            else:
                break

        msg = (
            f"Failed to generate after {controller.retry_policy.max_attempts} attempts. "
            f"Last error: {type(last_error).__name__}: {last_error}"
        )
        raise RuntimeError(msg) from last_error

    def run(self, prompt: str, config: RunConfig | None = None) -> Any:
        """Run the agent synchronously and block until finished."""
        return asyncio.run(self.generate(prompt, run_config=config))

    async def run_async(
        self,
        prompt: str,
        session_id: str | None = None,
        user_id: str = "system",
        initial_state: dict[str, Any] | None = None,
        run_config: RunConfig | None = None,
    ) -> AsyncGenerator[Event, None]:
        """Expose the underlying ADK event stream.

        This allows for real-time handling of tool calls, streaming responses, and more.
        """
        if session_id is None:
            session_id = f"stream_{uuid.uuid4().hex[:8]}"

        actual_user_id = (
            run_config.user_email if run_config and run_config.user_email else user_id
        )

        # Ensure session exists
        await self.session_service.create_session(
            app_name="markov_agent",
            user_id=actual_user_id,
            session_id=session_id,
            state=initial_state or {},
        )

        content = types.Content(role="user", parts=[types.Part(text=prompt)])

        adk_run_config = run_config.to_adk_run_config() if run_config else None
        async for event in self.runner.run_async(
            user_id=actual_user_id,
            session_id=session_id,
            new_message=content,
            run_config=adk_run_config,
        ):
            yield event

    async def run_live(
        self,
        live_request_queue: LiveRequestQueue,
        session_id: str | None = None,
        user_id: str = "system",
        initial_state: dict[str, Any] | None = None,
        run_config: RunConfig | None = None,
    ) -> AsyncGenerator[Event, None]:
        """Expose the underlying ADK live streaming (bidirectional) capability.

        This allows for real-time handling of text and audio interactions.
        """
        if session_id is None:
            session_id = f"live_{uuid.uuid4().hex[:8]}"

        actual_user_id = (
            run_config.user_email if run_config and run_config.user_email else user_id
        )

        # Ensure session exists
        session = await self.session_service.get_session(
            app_name="markov_agent",
            user_id=actual_user_id,
            session_id=session_id,
        )
        if not session:
            session = await self.session_service.create_session(
                app_name="markov_agent",
                user_id=actual_user_id,
                session_id=session_id,
                state=initial_state or {},
            )

        adk_run_config = run_config.to_adk_run_config() if run_config else None

        async for event in self.runner.run_live(
            session=session,
            live_request_queue=live_request_queue,
            run_config=adk_run_config,
        ):
            yield event

    async def get_session_events(
        self, session_id: str, user_id: str = "system"
    ) -> list[Event]:
        """Retrieve the history of events for a given session."""
        session = await self.session_service.get_session(
            app_name="markov_agent",
            user_id=user_id,
            session_id=session_id,
        )
        return session.events if session else []

    async def _reduce_context(
        self,
        prompt: str,
        initial_state: dict[str, Any] | None,
        max_tokens: int,
        artifact_service: BaseArtifactService | None = None,
        memory_service: BaseMemoryService | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        """Reduce the size of the context to fit within max_tokens."""
        model_name = str(self.config.model_name)

        instruction_text = ""
        if self.config.instruction and isinstance(self.config.instruction, str):
            instruction_text = self.config.instruction

        # 1. Calculate current totals
        instruction_tokens = count_tokens(instruction_text, model_name)
        prompt_tokens = count_tokens(prompt, model_name)

        state_text = ""
        if initial_state:
            import json

            try:
                state_text = json.dumps(initial_state)
            except Exception:
                state_text = str(initial_state)
        state_tokens = count_tokens(state_text, model_name)

        total_tokens = instruction_tokens + prompt_tokens + state_tokens

        if total_tokens <= max_tokens:
            return prompt, initial_state

        logger.info(
            "Context total tokens (%s) exceeds max_tokens (%s). Reducing...",
            total_tokens,
            max_tokens,
        )

        # 2. Budgeting (Dynamic)
        # We prioritize instruction, then prompt, then state
        remaining_budget = max_tokens

        # Keep up to 20% for instruction
        inst_limit = int(max_tokens * 0.20)
        if instruction_tokens > inst_limit:
            # But we can warn.
            logger.warning("System instruction is large and might exceed its budget.")

        remaining_budget -= min(instruction_tokens, inst_limit)

        # Allocate budget: ensure at least 30% for prompt if needed, but allow state to take what's left
        prompt_limit = max(int(max_tokens * 0.3), remaining_budget - 100)

        if prompt_tokens > prompt_limit:
            if (
                self.config.reduction_strategy == ReductionStrategy.LLM
                and self.config.reduction_prompt
            ):
                logger.info("Using LLM for prompt reduction")
                prompt = await self._run_reduction_llm(
                    prompt, self.config.reduction_prompt, prompt_limit
                )
            else:
                logger.info(
                    "Reducing prompt from %s to %s tokens using strategy %s",
                    prompt_tokens,
                    prompt_limit,
                    self.config.reduction_strategy,
                )
                prompt = reduce_text_to_tokens(
                    prompt,
                    prompt_limit,
                    model_name,
                    strategy=self.config.reduction_strategy,
                    recency_weight=self.config.recency_weight,
                )
            prompt_tokens = count_tokens(prompt, model_name)

        remaining_budget -= prompt_tokens

        # 3. Reduce State if still over
        if initial_state and remaining_budget > 0:
            import json

            new_state = initial_state.copy()
            state_json = json.dumps(new_state)
            state_tokens = count_tokens(state_json, model_name)

            # If we are STILL over budget, reduce state to fit the REMAINING budget
            if state_tokens > remaining_budget:
                # Strategy: If LLM reduction is enabled and the whole state is too big,
                # we might want to summarize the WHOLE state once instead of key-by-key.
                used_holistic = False
                if (
                    self.config.reduction_strategy == ReductionStrategy.LLM
                    and self.config.reduction_prompt
                ):
                    logger.info("Using LLM for holistic state reduction")
                    try:
                        summary = await self._run_reduction_llm(
                            state_json,
                            self.config.reduction_prompt,
                            remaining_budget,
                        )
                        # Try to parse as JSON if it looks like one
                        if summary.strip().startswith("{"):
                            with contextlib.suppress(Exception):
                                new_state = json.loads(summary)
                                return prompt, new_state

                        # If not JSON but LLM succeeded in giving a string summary, we could use it,
                        # but iterative reduction on original keys might be better.
                        # For now, let's treat non-JSON as a partial failure and proceed to iterative if it's too large.
                    except Exception as e:
                        logger.warning(
                            "Holistic LLM reduction failed, falling back: %s", e
                        )

                # Standard iterative reduction for all keys
                if not used_holistic:
                    keys = list(new_state.keys())
                    # Sort keys by value size to reduce largest first
                    keys.sort(key=lambda k: len(json.dumps(new_state[k])), reverse=True)

                    from markov_agent.engine.token_utils import (
                        reduce_dict_to_tokens,
                        reduce_list_to_tokens,
                    )

                    # We pre-calculate how much we need to reduce in total
                    # Use count_tokens on current new_state JSON to be accurate
                    current_state_tokens = count_tokens(
                        json.dumps(new_state), model_name
                    )
                    tokens_to_shed = current_state_tokens - remaining_budget

                    for k in keys:
                        if tokens_to_shed <= 0:
                            break

                        val = new_state[k]
                        try:
                            val_json = json.dumps(val)
                        except Exception:
                            val_json = str(val)
                        val_tokens = count_tokens(val_json, model_name)

                        # If this key is large, reduce it
                        if val_tokens > 10:
                            # Give it a target that helps us reach the overall goal
                            # Simple heuristic: try to reduce this key by its proportion of total excess
                            # but ensure we don't wipe it out completely unless necessary.
                            limit = max(10, val_tokens - tokens_to_shed)
                            old_val_tokens = val_tokens
                            if isinstance(val, str):
                                new_state[k] = reduce_text_to_tokens(
                                    val,
                                    limit,
                                    model_name,
                                    strategy=self.config.reduction_strategy,
                                )
                            elif isinstance(val, list):
                                new_state[k] = reduce_list_to_tokens(
                                    val,
                                    limit,
                                    model_name,
                                    strategy=self.config.reduction_strategy,
                                )
                            elif isinstance(val, dict):
                                new_state[k] = reduce_dict_to_tokens(
                                    val,
                                    limit,
                                    model_name,
                                    strategy=self.config.reduction_strategy,
                                )

                            new_val_tokens = count_tokens(
                                json.dumps(new_state[k]), model_name
                            )
                            tokens_to_shed -= old_val_tokens - new_val_tokens

                initial_state = new_state

        return prompt, initial_state

    async def _run_reduction_llm(self, text: str, prompt: str, max_tokens: int) -> str:
        """Use an LLM to reduce text according to a prompt."""
        # Use a cheaper model for reduction if specified, or current model
        model_to_use = self.config.reduction_model_name or self.config.model_name

        reduction_full_prompt = (
            f"{prompt}\n\n"
            f"TARGET TOKEN LIMIT: {max_tokens}\n"
            f"TEXT TO REDUCE:\n{text}\n\n"
            "Output only the reduced text, maintaining the original meaning and critical information."
        )

        # Create a transient controller for reduction to avoid recursing into _reduce_context
        # We don't set max_input_tokens on this variant to avoid loops
        reduction_cfg = self.config.model_copy(
            update={
                "model_name": model_to_use,
                "max_input_tokens": None,
                "reduction_prompt": None,
            }
        )
        reduction_ctrl = ADKController(
            config=reduction_cfg,
            retry_policy=self.retry_policy,
            mock_responder=self.mock_responder,
            artifact_service=self.artifact_service,
            name=f"{self.name}_reducer",
        )

        try:
            result = await reduction_ctrl.generate(reduction_full_prompt)
            return str(result)
        except Exception as e:
            logger.warning(
                "LLM reduction failed, falling back to %s strategy: %s",
                self.config.reduction_strategy,
                e,
            )
            return reduce_text_to_tokens(
                text,
                max_tokens,
                str(model_to_use),
                strategy=self.config.reduction_strategy,
                recency_weight=self.config.recency_weight,
            )


def model_config(
    name: str,
    temperature: float = 0.7,
    **kwargs: Any,
) -> ADKConfig:
    """Create an ADKConfig instance, mimicking google-adk's model_config."""
    return ADKConfig(model_name=name, temperature=temperature, **kwargs)
