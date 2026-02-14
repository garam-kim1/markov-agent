import pytest

from markov_agent.engine.agent import VerifiedAgent


@pytest.mark.asyncio
async def test_verified_agent_run_verified():
    responses = {
        "Research the following instruction": "Research: The instruction is valid.",
        "Based on the following research": "Implementation: Done.",
    }

    def mock_responder(prompt: str) -> str:
        for key, val in responses.items():
            if key in prompt:
                return val
        return "Default response"

    agent = VerifiedAgent(name="verified_tester", mock_responder=mock_responder)

    result = await agent.run_verified("Test instruction")
    assert "Implementation: Done." in result


@pytest.mark.asyncio
async def test_verified_agent_hallucination_check():
    responses = {
        "Research the following instruction": "Research: This library doesn't exist. It is a hallucination.",
        "Based on the following research": "I cannot implement this because it is based on a hallucination.",
    }

    def mock_responder(prompt: str) -> str:
        for key, val in responses.items():
            if key in prompt:
                return val
        return "Default response"

    agent = VerifiedAgent(name="verified_tester", mock_responder=mock_responder)

    result = await agent.run_verified("Use the non-existent-lib")
    assert "cannot implement" in result or "hallucination" in result
