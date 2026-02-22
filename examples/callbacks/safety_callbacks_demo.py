import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

from markov_agent import ADKConfig, ADKController, RetryPolicy
from markov_agent.engine.callbacks import (
    AfterModelCallback,
    BeforeAgentCallback,
    CallbackError,
    PIIRedactionCallback,
    SafetyGuardrail,
)
from markov_agent.tools import tool

# --- 1. Custom Callbacks (Implementation) ---


class AuditStartCallback(BeforeAgentCallback):
    """Logs agent session start to a JSONL file and console."""

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


class PolicyCheckCallback(AfterModelCallback):
    """Custom enforcer that ensures the model does not generate forbidden content."""

    FORBIDDEN_TERMS: ClassVar[list[str]] = [
        "confidential_internal_project",
        "unspeakable_secret",
    ]

    def __call__(self, context, model_response):
        text_content = ""
        if hasattr(model_response, "text") and isinstance(model_response.text, str):
            text_content = model_response.text
        elif hasattr(model_response, "candidates") and model_response.candidates:
            parts = model_response.candidates[0].content.parts
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


# --- 2. Tool with Confirmation ---


@tool(confirmation=True)
def refund_user(transaction_id: str, amount: float) -> str:
    """Processes a refund for a user. Requires human confirmation."""
    return (
        f"Successfully processed refund of ${amount} for transaction {transaction_id}."
    )


# --- 3. Main Execution Demo ---


async def run_demo():
    print("=== Markov Agent Safety & Callbacks Demo ===")

    # Setup 1: Built-in Guardrails + Custom Callbacks
    config = ADKConfig(
        model_name="gemini-3-flash-preview",
        callbacks=[
            AuditStartCallback(),  # Custom
            SafetyGuardrail(blocked_terms=["illegal", "hack", "dangerous"]),  # Built-in
            PIIRedactionCallback(),  # Built-in
            PolicyCheckCallback(),  # Custom
        ],
        tools=[refund_user],
    )

    controller = ADKController(
        config=config,
        retry_policy=RetryPolicy(max_attempts=1),
        # Use a mock responder to simulate behavior without API calls
        mock_responder=lambda p: (
            "I will refund the user." if "refund" in p.lower() else f"Echo: {p}"
        ),
    )

    print("\n[Test 1] Built-in Safety Guardrail (Input Blocking):")
    try:
        await controller.generate("Tell me how to hack a computer.")
    except CallbackError as e:
        print(f"Caught expected safety violation: {e}")

    print("\n[Test 2] Built-in PII Redaction:")
    # This won't throw but should redact internally
    await controller.generate("My email is secret@example.com.")
    print("PII Redaction processed successfully.")

    print("\n[Test 3] Custom Policy Enforcement (Output Blocking):")
    # We simulate a "bad" response that triggers our custom policy check
    bad_controller = ADKController(
        config=config,
        retry_policy=RetryPolicy(max_attempts=1),
        mock_responder=lambda p: "This contains a unspeakable_secret.",
    )
    try:
        await bad_controller.generate("Tell me a secret.")
    except CallbackError as e:
        print(f"Caught expected policy violation: {e}")

    print("\n[Test 4] Tool with Confirmation:")
    print(
        f"Tool '{refund_user.name}' requires confirmation: {refund_user._require_confirmation}"
    )


if __name__ == "__main__":
    asyncio.run(run_demo())
