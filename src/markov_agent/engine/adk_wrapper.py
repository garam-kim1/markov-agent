import asyncio
from typing import Any

from pydantic import BaseModel, Field

# Mocking google_adk for structure compliance
try:
    import google_adk
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


class ADKConfig(BaseModel):
    model_name: str
    temperature: float = 0.7
    safety_settings: list[Any] = Field(default_factory=list)


class RetryPolicy(BaseModel):
    max_attempts: int = 3
    initial_delay: float = 0.5
    backoff_factor: float = 2.0


class ADKController:
    """
    Wrapper around google_adk.Model (The PPU).
    Manages configuration, retries, and interaction with the underlying model.
    """

    def __init__(self, config: ADKConfig, retry_policy: RetryPolicy, mock_responder=None):
        self.config = config
        self.retry_policy = retry_policy
        self.mock_responder = mock_responder
        self.model = google_adk.Model(
            model_name=config.model_name, safety_settings=config.safety_settings
        )

    async def generate(self, prompt: str) -> str:
        """
        Generates content with retry logic.
        """
        attempt = 0
        current_delay = self.retry_policy.initial_delay
        last_error = None

        if self.mock_responder:
            # Bypass retry logic for mocks, or keep it if we want to test retries with mocks
            return self.mock_responder(prompt)

        while attempt < self.retry_policy.max_attempts:
            try:
                # Assuming the mock/wrapper returns a string or a response object
                response = await self.model.generate(prompt)

                # specific handling if the real ADK returns an object
                if hasattr(response, "text"):
                    return response.text
                return str(response)

            except Exception as e:
                last_error = e
                attempt += 1
                if attempt < self.retry_policy.max_attempts:
                    await asyncio.sleep(current_delay)
                    current_delay *= self.retry_policy.backoff_factor

        msg = f"Failed to generate after {self.retry_policy.max_attempts} attempts"
        raise RuntimeError(msg) from last_error
