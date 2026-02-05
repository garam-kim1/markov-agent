import asyncio
import re

from markov_agent.engine.adk_wrapper import ADKConfig, ADKController, RetryPolicy
from markov_agent.engine.callbacks import (
    AfterModelCallback,
    BeforeAgentCallback,
    BeforeModelCallback,
    CallbackError,
)

# --- Callbacks (Same logic as complex_safety.py) ---


class AuditStartCallback(BeforeAgentCallback):
    def __call__(self, context, *args, **kwargs):
        print(f"[Audit] Session started: {context.invocation_id}")


class PIIScrubCallback(BeforeModelCallback):
    EMAIL_REGEX = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"

    def __call__(self, context, model_request):
        if not hasattr(model_request, "contents"):
            return model_request
        modified = False
        for content in model_request.contents:
            for part in content.parts:
                if part.text:
                    scrubbed = re.sub(self.EMAIL_REGEX, "[REDACTED_EMAIL]", part.text)
                    if scrubbed != part.text:
                        print("[Security] Redacting PII from prompt...")
                        part.text = scrubbed
                        modified = True
        return model_request if modified else None


class PolicyCheckCallback(AfterModelCallback):
    FORBIDDEN_TERMS = ["confidential_internal_project", "unspeakable_secret"]

    def __call__(self, context, model_response):
        text_content = ""
        if hasattr(model_response, "text") and isinstance(model_response.text, str):
            text_content = model_response.text
        elif hasattr(model_response, "candidates") and model_response.candidates:
            parts = model_response.candidates[0].content.parts
            text_content = "".join(
                [p.text for p in parts if hasattr(p, "text") and p.text]
            )

        for term in self.FORBIDDEN_TERMS:
            if term in text_content:
                print(f"[Policy] VIOLATION: Found '{term}'")
                raise CallbackError(f"Policy Violation: '{term}' detected.")


# --- Main Execution with Local LLM ---


async def main():
    print("--- Starting Local LLM Callback Example ---")

    config = ADKConfig(
        model_name="openai/Qwen3-0.6B-Q4_K_M.gguf",
        api_base="http://192.168.1.213:8080/v1",
        api_key="no-key",
        use_litellm=True,
        temperature=0.7,
        callbacks=[
            AuditStartCallback(),
            PIIScrubCallback(),
            PolicyCheckCallback(),
        ],
    )

    controller = ADKController(config=config, retry_policy=RetryPolicy(max_attempts=1))

    prompt = "Tell john.doe@example.com about the confidential_internal_project."
    print(f"\nUser Prompt: {prompt}")

    try:
        # This will trigger PIIScrubCallback (redacting email)
        # AND PolicyCheckCallback (detecting the forbidden project name in response, if generated)
        response = await controller.generate(prompt)
        print(f"\nFinal Response: {response}")
    except CallbackError as e:
        print(f"\nBlocked by Policy: {e}")
    except Exception as e:
        print(f"\nExecution Error (Connection/API): {e}")


if __name__ == "__main__":
    asyncio.run(main())
