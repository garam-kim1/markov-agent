import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from markov_agent.engine.adk_wrapper import ADKConfig, ADKController, RetryPolicy
from markov_agent.engine.callbacks import (
    AfterAgentCallback,
    AfterModelCallback,
    BeforeAgentCallback,
    BeforeModelCallback,
    CallbackError,
)


# --- 1. Audit Logger ---
class AuditLogCallback(BeforeAgentCallback, AfterAgentCallback):
    """Logs agent session start/end to a JSONL file."""

    def __init__(self, log_path: str = "audit_log.jsonl"):
        self.log_path = Path(log_path)

    def _log(self, event_type: str, context: Any, **kwargs):
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event_type,
            "agent_name": getattr(context, "agent_name", "unknown"),
            "session_id": getattr(context, "session_id", "unknown"),
            **kwargs,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def __call__(self, context, *args, **kwargs):
        # We implement both interfaces, but need to distinguish which one was called.
        # This is a bit tricky with the current single __call__ signature in the generic base.
        # But in Python, we can just define the specific methods if we wanted,
        # or relying on the fact that CallbackAdapter calls the instance.
        # Wait, the CallbackAdapter calls `cb(context, ...)` for all of them.
        # It doesn't tell us WHICH hook it is if we implement multiple interfaces in one class.
        #
        # Correct Pattern: Implement specific methods or separate classes.
        # The Abstract Base Classes (ABC) define `__call__`.
        # If a class implements multiple ABCs, `__call__` is shared.
        # This is a design limitation of the simple "Callable" interface if using one class for multiple hooks.
        #
        # Solution for this example: Use separate classes or inspect arguments?
        # Arguments for BeforeAgent and AfterAgent are similar.
        #
        # Better approach: The library expects `__call__`.
        # If I want one class to handle multiple phases, I should probably have `on_start` and `on_end` methods,
        # but the interface forces `__call__`.
        #
        # So I will define TWO separate callbacks for logging to be clean.
        pass


class AuditStartCallback(BeforeAgentCallback):
    def __init__(self, log_path: str = "audit_log.jsonl"):
        self.log_path = Path(log_path)

    def __call__(self, context, *args, **kwargs):
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": "agent_start",
            "agent_name": context.agent_name,
            "invocation_id": context.invocation_id,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[Audit] Session started: {context.invocation_id}")


class AuditEndCallback(AfterAgentCallback):
    def __init__(self, log_path: str = "audit_log.jsonl"):
        self.log_path = Path(log_path)

    def __call__(self, context, *args, **kwargs):
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": "agent_end",
            "agent_name": context.agent_name,
            "invocation_id": context.invocation_id,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[Audit] Session ended: {context.invocation_id}")


# --- 2. PII Scrubber ---
class PIIScrubCallback(BeforeModelCallback):
    """Redacts email addresses from the user prompt before sending to LLM."""

    EMAIL_REGEX: ClassVar[str] = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"

    def __call__(self, context, model_request):
        # We expect model_request to be google.adk.models.llm_request.LlmRequest
        if not hasattr(model_request, "contents"):
            return model_request

        modified = False
        # Iterate through contents (User/System messages)
        for content in model_request.contents:
            for part in content.parts:
                if part.text:
                    scrubbed_text = re.sub(
                        self.EMAIL_REGEX, "[REDACTED_EMAIL]", part.text
                    )
                    if scrubbed_text != part.text:
                        print("[Security] Redacting PII from prompt...")
                        part.text = scrubbed_text
                        modified = True

        return model_request if modified else None


# --- 3. Policy Enforcer ---
class PolicyCheckCallback(AfterModelCallback):
    """Ensures the model does not generate forbidden content."""

    FORBIDDEN_TERMS: ClassVar[list[str]] = [
        "confidential_internal_project",
        "unspeakable_secret",
    ]

    def __call__(self, context, model_response):
        # Inspect response. Structure depends on model but ADK normalizes some things?
        # Usually model_response is LlmResponse (or similar from ADK)
        # Let's try to find text candidates.

        text_content = ""

        # Heuristic to get text from various response types
        if hasattr(model_response, "text") and isinstance(model_response.text, str):
            text_content = model_response.text
        elif hasattr(model_response, "candidates"):
            # Google GenAI style
            if model_response.candidates:
                # Check if candidates is iterable and has content
                candidate = model_response.candidates[0]
                if hasattr(candidate, "content") and hasattr(
                    candidate.content, "parts"
                ):
                    parts = candidate.content.parts
                    text_content = "".join(
                        [p.text for p in parts if hasattr(p, "text") and p.text]
                    )

        if not text_content:
            return

        for term in self.FORBIDDEN_TERMS:
            if term in text_content:
                print(f"[Policy] VIOLATION DETECTED: Found forbidden term '{term}'")
                raise CallbackError(
                    f"Policy Violation: Generated content contains forbidden term '{term}'"
                )

        return  # No modification


# --- Main Execution Example ---
async def main():
    print("--- Starting Complex Callback Example ---")

    # 1. Configure Callbacks
    callbacks = [
        AuditStartCallback(),
        PIIScrubCallback(),
        PolicyCheckCallback(),
        AuditEndCallback(),
    ]

    # 2. Setup Agent
    # Note: In a real run, ensure GOOGLE_API_KEY is set.
    config = ADKConfig(
        model_name="gemini-3-flash-preview",
        callbacks=callbacks,
    )

    controller = ADKController(config=config, retry_policy=RetryPolicy(max_attempts=1))

    # 3. Trigger PII Scrubbing
    prompt_with_pii = "Please send the confidential report to boss@company.com ASAP."
    print(f"\nUser Prompt: {prompt_with_pii}")

    try:
        # We expect this to fail due to no API key, but we want to see the [Security] log first.
        await controller.generate(prompt_with_pii)
    except Exception as e:
        if (
            "API_KEY" in str(e)
            or "403" in str(e)
            or "401" in str(e)
            or "credentials" in str(e)
        ):
            print(
                "\n(Note: API call failed as expected in demo env, but callbacks should have fired)"
            )
        else:
            print(f"\nResult: {e}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
