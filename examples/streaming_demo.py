import asyncio
import os

from google.adk.agents import LiveRequestQueue
from google.genai import types

from markov_agent import ADKConfig, ADKController, RetryPolicy, RunConfig

# 1. Define the Agent/Controller Configuration
# We use a model that supports streaming/live API (e.g., gemini-3-flash-preview)
config = ADKConfig(
    model_name="gemini-3-flash-preview",
    instruction="You are a helpful voice assistant. Keep answers concise.",
    description="A helpful assistant that answers questions via voice or text.",
    enable_logging=True,
)


async def main():
    # 2. Initialize the Controller
    # The ADKController manages the Runner, SessionService, and Agent
    controller = ADKController(config=config, retry_policy=RetryPolicy(max_attempts=3))

    # 3. Initialize Communication Queue
    # This queue is used to SEND data (text/audio) TO the agent
    live_request_queue = LiveRequestQueue()

    # 4. Configure the Run
    # We specify bidirectional streaming and requested modalities
    run_config = RunConfig(
        streaming_mode="bidi",
        response_modalities=["AUDIO", "TEXT"],
        # Optional: Configure voice
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
            )
        ),
    )

    print("--- Starting Bidirectional Streaming Demo ---")
    print("Connecting to agent. Listening...")

    # 5. Start the Event Processing Loop
    # We use asyncio.gather to handle sending and receiving simultaneously
    try:
        await asyncio.gather(
            handle_receiving(controller, live_request_queue, run_config),
            handle_sending(live_request_queue),
        )
    except Exception as e:
        print(f"\nAn error occurred: {e}")


async def handle_receiving(
    controller: ADKController, queue: LiveRequestQueue, config: RunConfig
):
    """Listens to the agent's output stream."""

    # controller.run_live() returns an async generator of Events
    async for event in controller.run_live(live_request_queue=queue, run_config=config):
        # Process the event content
        if event.content and event.content.parts:
            for part in event.content.parts:
                # Handle Text Chunks
                if part.text:
                    print(f"Agent (Text): {part.text}", end="", flush=True)

                # Handle Audio Chunks (Bytes)
                if part.inline_data:
                    # In a real app, you would play these bytes or stream them to a client
                    # For this demo, we just acknowledge receiving them
                    pass

        # Handle Tool Calls (Automatically handled by Runner, but visible here)
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.function_call:
                    print(f"\n[System] Agent calling tool: {part.function_call.name}")


async def handle_sending(queue: LiveRequestQueue):
    """Simulates sending user input to the agent."""
    # Wait a bit for the connection to be established
    await asyncio.sleep(2)

    print("\n[User] Sending: 'Hello, who are you?'")
    # Send Text
    queue.send_content(
        types.Content(
            role="user", parts=[types.Part.from_text(text="Hello, who are you?")]
        )
    )

    # Wait for the response
    await asyncio.sleep(5)

    print("\n[User] Sending: 'Tell me a short joke.'")
    queue.send_content(
        types.Content(
            role="user", parts=[types.Part.from_text(text="Tell me a short joke.")]
        )
    )

    await asyncio.sleep(5)
    print("\n[System] Closing session...")
    queue.close()  # End the session


if __name__ == "__main__":
    # Ensure GEMINI_API_KEY is set
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set.")
    else:
        asyncio.run(main())
