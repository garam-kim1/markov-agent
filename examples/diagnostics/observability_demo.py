import asyncio
import os
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from markov_agent.engine.adk_wrapper import ADKConfig, ADKController, RetryPolicy


class MockLLM(BaseLlm):
    """A mock LLM for testing without API keys, inheriting from ADK BaseLlm."""

    model: str = "mock-model"
    response_text: str = "This is a mock response."

    async def generate_content_async(
        self, llm_request: Any, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        # Mocking the response structure
        response = LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=self.response_text)],
            ),
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=10,
                candidates_token_count=5,
                total_token_count=15,
            ),
            finish_reason=types.FinishReason.STOP,
        )
        yield response


async def main():
    print("=== Markov Agent Observability Demo ===")

    # 1. Initialize a mock model
    mock_llm = MockLLM(response_text="Hello! I am a mock agent with observability enabled.")

    # 2. Setup ADKConfig with observability enabled and mock model
    config = ADKConfig(
        model_name=mock_llm,
        enable_logging=True,
        enable_tracing=True,
    )

    # 3. Initialize ADKController
    controller = ADKController(
        config=config,
        retry_policy=RetryPolicy(max_attempts=1),
    )

    # 4. Run the agent
    print("\n--- Running Agent (should see logs and traces) ---")
    response = await controller.generate("Hello, can you help me?")

    print(f"\nFinal Response: {response}")


if __name__ == "__main__":
    # Ensure we don't actually call Google APIs
    os.environ["GOOGLE_API_KEY"] = "mock-key"

    try:
        asyncio.run(main())
    except Exception as e:
        # If it fails due to pydantic validation of ADKConfig.model_name,
        # we might need to adjust the type hint in ADKConfig.
        import traceback
        traceback.print_exc()
        print(f"\nDemo finished with error: {e}")
