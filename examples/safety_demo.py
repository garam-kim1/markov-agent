import asyncio

from markov_agent.engine.adk_wrapper import ADKConfig, ADKController, RetryPolicy
from markov_agent.engine.callbacks import PIIRedactionCallback, SafetyGuardrail
from markov_agent.tools import tool


# 1. Define a tool that requires confirmation
@tool(confirmation=True)
def refund_user(transaction_id: str, amount: float) -> str:
    """Processes a refund for a user. Requires human confirmation."""
    return (
        f"Successfully processed refund of ${amount} for transaction {transaction_id}."
    )


# 2. Setup ADK Config with Safety Guardrails
config = ADKConfig(
    model_name="gemini-3-flash-preview",
    callbacks=[
        SafetyGuardrail(blocked_terms=["illegal", "hack", "dangerous"]),
        PIIRedactionCallback(),
    ],
    tools=[refund_user],
)


async def main():
    # We'll use a Mock responder to simulate the interaction without hitting the API
    def mock_responder(prompt: str) -> str:
        if "refund" in prompt.lower():
            # Return a tool call (simulated text that would trigger tool use in a real model)
            # In a real scenario, the model would emit a function call event.
            # Since we are using MockLlm in ADKController, it's easier to just test the logic.
            return "I will refund the user now."
        return f"Response to: {prompt}"

    controller = ADKController(
        config=config,
        retry_policy=RetryPolicy(max_attempts=1),
        mock_responder=mock_responder,
    )

    print("--- Test 1: Safety Violation ---")
    try:
        await controller.generate("Tell me how to hack a computer.")
    except Exception as e:
        print(f"Caught expected error: {e}")

    print("\n--- Test 2: PII Redaction ---")
    # This one won't throw but should redact if we could see the internal request
    # We can check if it runs without error.
    res = await controller.generate(
        "My email is secret@example.com, don't tell anyone."
    )
    print(f"Agent response: {res}")

    print("\n--- Test 3: Tool with Confirmation (Logic check) ---")
    print("Note: In a full ADK runtime, confirmation would pause execution.")
    print("The tool is registered correctly with confirmation=True.")
    print(
        f"Tool {refund_user.name} requires confirmation: {refund_user._require_confirmation}"
    )


if __name__ == "__main__":
    asyncio.run(main())
