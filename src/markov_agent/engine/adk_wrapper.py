import asyncio
import os
import uuid
from typing import Any, TypeVar

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

from markov_agent.engine.telemetry_plugin import MarkovBridgePlugin

T = TypeVar("T", bound=BaseModel)


class ADKConfig(BaseModel):
    model_name: str
    temperature: float = 0.7
    safety_settings: list[Any] = Field(default_factory=list)
    api_base: str | None = None
    api_key: str | None = None
    tools: list[Any] = Field(default_factory=list)
    instruction: str | None = None
    description: str | None = None
    generation_config: dict[str, Any] | None = None
    plugins: list[Any] = Field(default_factory=list)
    use_litellm: bool = False


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
    ):
        self.config = config
        self.retry_policy = retry_policy
        self.mock_responder = mock_responder
        self.session_service = InMemorySessionService()

        # Configure environment if needed
        if self.config.api_key:
            os.environ["GOOGLE_API_KEY"] = self.config.api_key

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

            model_instance = LiteLlm(model=self.config.model_name)

        # Prepare model_config from generation_config and temperature
        model_config = self.config.generation_config or {}
        if "temperature" not in model_config:
            model_config["temperature"] = self.config.temperature

        # If response_schema made it into model_config (via PPU init), remove it
        # and use the explicit output_schema argument or the one from config
        if "response_schema" in model_config:
            if output_schema is None:
                output_schema = model_config.pop("response_schema")
            else:
                model_config.pop("response_schema")

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
            generate_content_config=model_config,
            output_schema=output_schema,
        )

        self.runner = Runner(
            app_name="markov_agent",
            agent=self.agent,
            session_service=self.session_service,
            plugins=[MarkovBridgePlugin()] + self.config.plugins,
        )

    async def generate(self, prompt: str, output_schema: type[T] | None = None) -> Any:
        """
        Generates content with retry logic using the ADK Runner.
        If output_schema is provided, attempts to generate and parse JSON.
        """

        async def run_attempt():
            if self.mock_responder:
                return self.mock_responder(prompt)

            final_prompt = prompt
            # Note: We do NOT manually inject the JSON schema into the prompt here.
            # The Agent is configured with output_schema, so the underlying model
            # should natively enforce the structure.

            session_id = f"gen_{uuid.uuid4().hex[:8]}"
            await self.session_service.create_session(
                app_name="markov_agent", user_id="system", session_id=session_id
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

            return final_text

        # Retry Loop
        attempt = 0
        current_delay = self.retry_policy.initial_delay
        last_error = None

        while attempt < self.retry_policy.max_attempts:
            try:
                raw_text = await run_attempt()

                # Parsing Logic
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
                    
                    return output_schema.model_validate_json(cleaned_text.strip())

                return raw_text

            except Exception as e:
                last_error = e
                attempt += 1
                if attempt < self.retry_policy.max_attempts:
                    await asyncio.sleep(current_delay)
                    current_delay *= self.retry_policy.backoff_factor

        msg = f"Failed to generate after {self.retry_policy.max_attempts} attempts"
        raise RuntimeError(msg) from last_error
