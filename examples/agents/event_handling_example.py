import asyncio
import os
from typing import Any, cast

from google.adk.agents.callback_context import CallbackContext

from markov_agent import ADKConfig, ADKController, RetryPolicy
from markov_agent.engine.callbacks import BeforeModelCallback


# 1. Using Callbacks for Event Control (Audit and Guard)
class AuditAndGuardCallback(BeforeModelCallback):
    """This callback runs before the request is sent to the LLM.
    It can be used for logging or guardrails.
    """

    def __call__(self, context: CallbackContext, model_request: Any) -> Any:
        # Inspect the request
        print(f"\n[Callback] Intercepted call for Agent: {context.agent_name}")

        # Implement Logic: Block prompts containing "secret"
        has_secret = False
        if hasattr(model_request, "contents"):
            for content in model_request.contents:
                for part in content.parts:
                    if (
                        hasattr(part, "text")
                        and part.text
                        and "secret" in part.text.lower()
                    ):
                        has_secret = True
                        break

        if has_secret:
            print(
                "[Callback] BLOCKING request due to security policy (contains 'secret')."
            )
            return None

        return None


# 2. The Event Loop: Consuming Events
async def run_agent_event_loop(
    controller: ADKController,
    user_input: str,
    session_id: str | None = None,
    user_id: str = "system",
):
    print(f"\n--- Running Event Loop for: '{user_input}' ---")

    # Using the new run_async method in ADKController
    async for event in controller.run_async(
        prompt=user_input, session_id=session_id, user_id=user_id
    ):
        # A. Check for Tool Calls (Agent requesting an action)
        function_calls = event.get_function_calls()
        if function_calls:
            for call in function_calls:
                print(f"[Event] Agent calling tool: {call.name} with args: {call.args}")

        # B. Check for Tool Results (Output from the tool)
        tool_responses = event.get_function_responses()
        if tool_responses:
            for response in tool_responses:
                print(f"[Event] Tool {response.name} returned result.")

        # C. Check for Streaming/Partial Text
        if hasattr(event, "partial") and event.partial:
            if event.content and event.content.parts:
                print(f"[Stream] {event.content.parts[0].text}", end="", flush=True)

        # D. Check for Final Response (Text to display to user)
        if event.is_final_response():
            if event.content and event.content.parts:
                print(f"\n[Event] Final Answer: {event.content.parts[0].text}")


async def main():
    # Setup configuration
    config = ADKConfig(
        model_name="gemini-3-flash-preview",
        callbacks=[AuditAndGuardCallback()],
    )

    # Initialize Controller
    controller = ADKController(
        config=config,
        retry_policy=RetryPolicy(max_attempts=1),
    )

    # Example 1: Normal Prompt
    await run_agent_event_loop(controller, "What is the capital of France?")

    # Example 2: History tracking in a specific session
    session_id = "history_demo_session"
    user_id = "user_123"

    print("\n--- Populating Session History ---")
    # We'll just use a loop to consume the events
    async for _ in controller.run_async(
        "Who is the president of the USA?", session_id=session_id, user_id=user_id
    ):
        pass

    # 3. Event Persistence: Retrieving History
    print("\n--- Retrieving Session History ---")
    events = await controller.get_session_events(session_id=session_id, user_id=user_id)
    print(f"Found {len(events)} events in session '{session_id}'")
    for event in events:
        if event.is_final_response() and event.content and event.content.parts:
            parts = cast("list", event.content.parts)
            print(f" - Saved Response: {parts[0].text[:50]}...")
        elif event.get_function_calls():
            print(f" - Saved Tool Call: {event.get_function_calls()[0].name}")


if __name__ == "__main__":
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = "dummy-key"

    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\nExecution ended (Expected if no real API key): {e}")
