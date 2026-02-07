import asyncio
import os

from markov_agent import (
    ADKConfig,
    ADKController,
    BasePlugin,
    CallbackContext,
    LlmRequest,
    LlmResponse,
    RetryPolicy,
)


class VerboseLoggingPlugin(BasePlugin):
    """A custom plugin that logs detailed information about LLM interactions."""

    def __init__(self):
        super().__init__(name="verbose_logger")

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        print(f"\n[Plugin] Sending request to model: {callback_context.agent_name}")
        for content in llm_request.contents:
            if not content.parts:
                continue
            for part in content.parts:
                if part.text:
                    print(f"  Prompt part: {part.text[:50]}...")
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        print("[Plugin] Received response from model")
        return None


async def main():
    # 1. Configure the agent with the custom plugin
    config = ADKConfig(
        model_name="gemini-3-flash-preview",
        instruction="You are a helpful assistant.",
        plugins=[VerboseLoggingPlugin()],
        enable_logging=False,  # Disable standard logging to see our plugin output clearly
    )

    # 2. Initialize the controller
    mock_resp = None
    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        print("No GOOGLE_API_KEY or GEMINI_API_KEY found, using mock responder.")

        def mock_resp(p):
            return "Why did the Markov chain cross the road? To get to the other state!"

    controller = ADKController(
        config=config,
        retry_policy=RetryPolicy(max_attempts=2),
        mock_responder=mock_resp,
    )

    # 3. Run the agent
    print("Starting agent run...")
    response = await controller.generate("Tell me a short joke about Markov chains.")
    print(f"\nFinal Response: {response}")


if __name__ == "__main__":
    asyncio.run(main())
