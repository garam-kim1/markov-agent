import asyncio

from markov_agent.engine.agent import VerifiedAgent


async def main():
    # Use a mock responder for the demo to avoid needing API keys
    def mock_responder(prompt: str) -> str:
        if "Research" in prompt:
            return (
                "Research Findings:\n"
                "1. The 'markov-agent' library uses 'google-adk' as its engine.\n"
                "2. It uses 'uv' for dependency management.\n"
                "3. The instruction to add a 'VerifiedAgent' makes sense as it improves reliability."
            )
        return (
            "Implementation Plan:\n"
            "1. Add VerifiedAgent to markov_agent.engine.agent.\n"
            "2. Include GoogleSearchTool in its initialization.\n"
            "3. Implement run_verified method.\n\n"
            "Result: Implementation completed successfully."
        )

    # Initialize the VerifiedAgent
    agent = VerifiedAgent(
        name="research_coder",
        model="gemini-3-flash-preview",
        mock_responder=mock_responder,
    )

    print("--- Verified Coding Agent Demo ---")
    instruction = "Improve the library by adding a VerifiedAgent that researches instructions first."
    print(f"Instruction: {instruction}\n")

    print("Running verified implementation...")
    result = await agent.run_verified(instruction)

    print("\n--- Final Result ---")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
