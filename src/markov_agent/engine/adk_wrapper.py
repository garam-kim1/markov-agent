import asyncio
import os
import uuid
from collections.abc import Callable
from typing import Any, TypeVar

from google.adk.agents import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.apps import App
from google.adk.artifacts import BaseArtifactService, InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

from markov_agent.engine.telemetry_plugin import MarkovBridgePlugin

T = TypeVar("T", bound=BaseModel)


class ADKConfig(BaseModel):
    model_name: str
    temperature: float = 0.7
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    max_tokens: int | None = None
    safety_settings: list[Any] = Field(default_factory=list)
    api_base: str | None = None
    api_key: str | None = None
    tools: list[Any] = Field(default_factory=list)
    instruction: str | Callable[[ReadonlyContext], str] | None = None
    description: str | None = None
    generation_config: dict[str, Any] | None = None
    plugins: list[Any] = Field(default_factory=list)
    callbacks: list[Any] = Field(default_factory=list)
    use_litellm: bool = False
    output_key: str | None = None


class RetryPolicy(BaseModel):
    max_attempts: int = 3
    initial_delay: float = 0.5
    backoff_factor: float = 2.0


class ADKController:
    """
    Wrapper around google_adk.Agent and Runner.
    Manages configuration, retries, and interaction with the underlying model.
    """

    def __init__(
        self,
        config: ADKConfig,
        retry_policy: RetryPolicy,
        mock_responder=None,
        output_schema: Any | None = None,
        artifact_service: BaseArtifactService | None = None,
    ):
        self.config = config
        self.retry_policy = retry_policy
        self.mock_responder = mock_responder
        self.session_service = InMemorySessionService()
        self.artifact_service = artifact_service or InMemoryArtifactService()

        # Configure environment if needed
        if self.config.api_key:
            os.environ["GOOGLE_API_KEY"] = self.config.api_key

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
        SAFE_GEN_CONFIG_KEYS = {
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
            if k in SAFE_GEN_CONFIG_KEYS:
                safe_config[k] = v
            else:
                extra_kwargs[k] = v

        # Model Initialization Logic
        model_instance = self.config.model_name
        if self.config.use_litellm:
            from google.adk.models.lite_llm import LiteLlm

            # Setup environment for LiteLLM if api_base provided
            if self.config.api_base:
                # Assuming OpenAI-compatible local server if using openai/ prefix
                if self.config.model_name.startswith("openai/"):
                    os.environ["OPENAI_API_BASE"] = self.config.api_base
                    if self.config.api_key:
                        os.environ["OPENAI_API_KEY"] = self.config.api_key
                    elif "OPENAI_API_KEY" not in os.environ:
                        os.environ["OPENAI_API_KEY"] = "dummy"

            # Pass extra_kwargs to LiteLlm constructor (e.g. min_p)
            model_instance = LiteLlm(model=self.config.model_name, **extra_kwargs)
        else:
            # For Google models, we ignore extra_kwargs or warn?
            # For now, we just drop them to prevent crashing GenerateContentConfig
            pass

        self.output_schema = output_schema
        self.agent = Agent(
            name="markov_ppu_agent",
            model=model_instance,
            instruction=self.config.instruction
            or (
                "You are a probabilistic processing unit in a Markov Engine. "
                "Execute the requested task accurately."
            ),
            description=self.config.description
            or "Markov Agent PPU for stochastic processing.",
            tools=self.config.tools,
            generate_content_config=types.GenerateContentConfig(**safe_config),
            output_schema=output_schema,
            output_key=self.config.output_key,
        )

        plugins = [MarkovBridgePlugin()]
        if self.config.callbacks:
            from markov_agent.engine.callback_adapter import CallbackAdapterPlugin

            plugins.append(CallbackAdapterPlugin(self.config.callbacks))

        plugins.extend(self.config.plugins)

        self.app = App(
            name="markov_agent",
            root_agent=self.agent,
            plugins=plugins,
        )

        self.runner = Runner(
            app=self.app,
            session_service=self.session_service,
            artifact_service=self.artifact_service,
        )

    def create_variant(
        self, generation_config_override: dict[str, Any]
    ) -> "ADKController":
        """
        Creates a new ADKController instance with specific generation config overrides.
        Useful for adaptive sampling (changing temperature/top_p).
        """
        new_config = self.config.model_copy(deep=True)
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
            artifact_service=self.artifact_service,
        )

    async def generate(
        self,
        prompt: str,
        output_schema: type[T] | None = None,
        initial_state: dict[str, Any] | None = None,
        include_state: bool = False,
    ) -> Any | tuple[Any, dict[str, Any]]:
        """
        Generates content with retry logic using the ADK Runner.
        If output_schema is provided, attempts to generate and parse JSON.
        Returns the result, or a tuple of (result, updated_session_state)
        if include_state is True.
        """

        async def run_attempt():
            if self.mock_responder:
                res = self.mock_responder(prompt)
                if asyncio.iscoroutine(res):
                    res = await res
                return res, (initial_state or {})

            final_prompt = prompt
            # Note: We do NOT manually inject the JSON schema into the prompt here.
            # The Agent is configured with output_schema, so the underlying model
            # should natively enforce the structure.

            session_id = f"gen_{uuid.uuid4().hex[:8]}"
            await self.session_service.create_session(
                app_name="markov_agent",
                user_id="system",
                session_id=session_id,
                state=initial_state or {},
            )

            content = types.Content(role="user", parts=[types.Part(text=final_prompt)])

            final_text = ""
            async for event in self.runner.run_async(
                user_id="system", session_id=session_id, new_message=content
            ):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        final_text = "".join(
                            p.text for p in event.content.parts if p.text
                        )
                    break

            # Retrieve final state
            final_session = await self.session_service.get_session(
                app_name="markov_agent", user_id="system", session_id=session_id
            )
            final_state = final_session.state if final_session else {}

            return final_text, final_state

        # Retry Loop
        attempt = 0
        current_delay = self.retry_policy.initial_delay
        last_error = None

        while attempt < self.retry_policy.max_attempts:
            try:
                raw_text, final_state = await run_attempt()

                # Parsing Logic
                result = raw_text
                if output_schema:
                    cleaned_text = raw_text.strip()
                    # Some models might still wrap in markdown
                    # even with structured output enforced
                    if cleaned_text.startswith("```json"):
                        cleaned_text = cleaned_text.replace("```json", "", 1)
                        if cleaned_text.endswith("```"):
                            cleaned_text = cleaned_text[:-3]
                    elif cleaned_text.startswith("```"):
                        cleaned_text = cleaned_text.replace("```", "", 1)
                        if cleaned_text.endswith("```"):
                            cleaned_text = cleaned_text[:-3]

                    result = output_schema.model_validate_json(cleaned_text.strip())

                if include_state:
                    return result, final_state
                return result

            except Exception as e:
                last_error = e
                attempt += 1
                if attempt < self.retry_policy.max_attempts:
                    await asyncio.sleep(current_delay)
                    current_delay *= self.retry_policy.backoff_factor

        msg = f"Failed to generate after {self.retry_policy.max_attempts} attempts"
        raise RuntimeError(msg) from last_error
