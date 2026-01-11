import asyncio
import os
from typing import Any

from pydantic import BaseModel, Field

# Mocking google_adk for structure compliance
try:
    import google_adk

    try:
        from google.adk.models.lite_llm import LiteLlm
    except ImportError:
        LiteLlm = None
except ImportError:
    # Minimal mock for scaffolding purposes
    class google_adk:
        class Model:
            def __init__(self, model_name, safety_settings=None):
                self.model_name = model_name

            async def generate(self, prompt: str):
                # Simulate network delay
                await asyncio.sleep(0.01)
                return f"Mock response for: {prompt[:20]}..."

        class SafetySetting:
            BLOCK_NONE = "BLOCK_NONE"

    # Mock LiteLlm if google_adk is missing
    class LiteLlm:
        def __init__(self, model, **kwargs):
            self.model_name = model

        async def generate(self, prompt: str):
            await asyncio.sleep(0.01)
            # If the prompt asks for JSON (which our controller does),
            # return a minimal valid JSON
            prompt_up = prompt.upper()
            if "RETURN VALID JSON" in prompt_up or "JSON" in prompt_up:
                if "SUMMARY" in prompt_up:
                    return '{"summary": "Mocked summary of results"}'
                if "STEPS" in prompt_up:
                    return '{"steps": ["step 1", "step 2", "step 3"]}'
                if "ANSWER" in prompt_up:
                    return '{"answer": "Mocked answer content", "confidence": 0.95}'
                return "{}"
            return f"Mock LiteLLM response for: {prompt[:20]}..."


class ADKConfig(BaseModel):
    model_name: str
    temperature: float = 0.7
    safety_settings: list[Any] = Field(default_factory=list)
    api_base: str | None = None
    api_key: str | None = None


class RetryPolicy(BaseModel):
    max_attempts: int = 3
    initial_delay: float = 0.5
    backoff_factor: float = 2.0


class ADKController:
    """
    Wrapper around google_adk.Model (The PPU).
    Manages configuration, retries, and interaction with the underlying model.
    """

    def __init__(
        self, config: ADKConfig, retry_policy: RetryPolicy, mock_responder=None
    ):
        self.config = config
        self.retry_policy = retry_policy
        self.mock_responder = mock_responder

        if self.config.api_base or (LiteLlm and "openai/" in self.config.model_name):
            # Configure environment for LiteLLM if needed
            if self.config.api_base:
                os.environ["OPENAI_API_BASE"] = self.config.api_base
            if self.config.api_key:
                os.environ["OPENAI_API_KEY"] = self.config.api_key

            # Initialize LiteLlm
            # Ensure model name includes provider prefix if not present,
            # though usually handled by user
            self.model = LiteLlm(
                model=self.config.model_name,
                safety_settings=self.config.safety_settings,
            )
        else:
            self.model = google_adk.Model(
                model_name=config.model_name, safety_settings=config.safety_settings
            )

    async def generate(
        self, prompt: str, output_schema: type[BaseModel] | None = None
    ) -> Any:
        """
        Generates content with retry logic.
        If output_schema is provided, attempts to generate and parse JSON.
        """

        async def run_attempt():
            if self.mock_responder:
                return self.mock_responder(prompt)

            final_prompt = prompt
            if output_schema:
                # Basic prompt engineering to force JSON if the API doesn't support
                # strict mode
                final_prompt = (
                    f"{prompt}\n\nReturn valid JSON matching this schema: "
                    f"{output_schema.model_json_schema()}"
                )

            response = await self.model.generate(final_prompt)

            if hasattr(response, "text"):
                return response.text
            return str(response)

        # Retry Loop
        attempt = 0
        current_delay = self.retry_policy.initial_delay
        last_error = None

        while attempt < self.retry_policy.max_attempts:
            try:
                raw_text = await run_attempt()

                # Parsing Logic
                if output_schema:
                    # simplistic extraction of JSON from markdown
                    cleaned_text = raw_text.strip()
                    if cleaned_text.startswith("```json"):
                        cleaned_text = cleaned_text.replace("```json", "").replace(
                            "```", ""
                        )
                    elif cleaned_text.startswith("```"):
                        cleaned_text = cleaned_text.replace("```", "")

                    return output_schema.model_validate_json(cleaned_text.strip())

                return raw_text

            except Exception as e:
                # If using mock, don't retry unless we want to simulate flaky mocks?
                # For now let's assume if mock fails (or parsing fails) we retry
                # if it's not a mock-specific error?
                # Actually, if parsing fails, we might want to retry (self-correction).

                last_error = e
                # If mock, maybe fail immediately? Or respect retry policy?
                # Let's respect retry policy for parsing errors even with mocks.

                attempt += 1
                if attempt < self.retry_policy.max_attempts:
                    await asyncio.sleep(current_delay)
                    current_delay *= self.retry_policy.backoff_factor

        msg = f"Failed to generate after {self.retry_policy.max_attempts} attempts"
        raise RuntimeError(msg) from last_error
