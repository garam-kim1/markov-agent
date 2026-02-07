import asyncio
import os

from markov_agent import (
    ADKConfig,
    ADKController,
    AdkWebServer,
    RetryPolicy,
    RunConfig,
)


async def run_server_demo():
    # 1. Initialize the controller
    config = ADKConfig(
        model_name="gemini-3-flash-preview", instruction="You are a helpful assistant."
    )

    mock_resp = None
    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        print("No GOOGLE_API_KEY or GEMINI_API_KEY found, using mock responder.")

        def mock_resp(p):
            return "I am a Markov Agent running in a demo environment."

    controller = ADKController(
        config=config, retry_policy=RetryPolicy(), mock_responder=mock_resp
    )

    # 2. Start the Web Server
    # This exposes the agent at http://localhost:8080
    # In a real application, you would run this and interact via UI or API
    print("Initializing Web Server...")
    server = AdkWebServer(agent=controller)
    print(f"Server created: {server}")

    # Note: server.run() is blocking and uses uvicorn.run()
    # In a real app, you would call: server.run(host="0.0.0.0", port=8080)
    print("Server initialized successfully.")

    # 3. Demonstrate RunConfig usage programmatically
    print("\nDemonstrating RunConfig usage...")
    run_cfg = RunConfig(user_email="dev@example.com", streaming=False)

    response = await controller.generate("Who are you?", run_config=run_cfg)
    print(f"Response with user context: {response}")


if __name__ == "__main__":
    asyncio.run(run_server_demo())
