import asyncio
from collections.abc import Callable
from typing import Any

from markov_agent.engine.adk_wrapper import ADKConfig, ADKController
from markov_agent.engine.ppu import ProbabilisticNode
from markov_agent.tools.search import GoogleSearchTool


class Agent(ProbabilisticNode):
    """A simplified Agent class that mimics the google-adk Agent API.

    Inherits from ProbabilisticNode to remain compatible with Markov Engine topologies.
    """

    def __init__(
        self,
        name: str,
        model: ADKConfig | str | None = None,
        system_prompt: str | Callable[..., str] | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Handle string model name or ADKConfig
        if isinstance(model, str):
            adk_config = ADKConfig(model_name=model)
        elif isinstance(model, ADKConfig):
            adk_config = model
        else:
            # Default model if none provided
            adk_config = ADKConfig(model_name="gemini-1.5-flash")

        # Map 'system_prompt' to 'adk_config.instruction'
        if system_prompt:
            adk_config.instruction = system_prompt

        if tools:
            adk_config.tools = tools

        # Default prompt_template to just passing the query if not provided
        prompt_template = kwargs.pop("prompt_template", "{{ query }}")

        # Initialize ProbabilisticNode
        super().__init__(
            name=name,
            adk_config=adk_config,
            prompt_template=prompt_template,
            **kwargs,
        )

    @property
    def system_prompt(self) -> str | Callable[..., str] | None:
        return self.adk_config.instruction

    @system_prompt.setter
    def system_prompt(self, value: str | Callable[..., str]) -> None:
        self.adk_config.instruction = value
        self._refresh_controller()

    def add_tool(self, tool: Any) -> None:
        """Add a tool to the agent."""
        if self.adk_config.tools is None:
            self.adk_config.tools = []
        self.adk_config.tools.append(tool)
        self._refresh_controller()

    def _refresh_controller(self) -> None:
        """Refresh the underlying ADKController with updated config."""
        self.controller = ADKController(
            self.adk_config,
            self.retry_policy,
            mock_responder=self.mock_responder,
            output_schema=self.output_schema,
            artifact_service=self.artifact_service,
            name=self.name,
        )

    def run(self, query: str) -> Any:
        """Run the agent synchronously and return a response object with .text.

        Matches the google-adk Agent.run() pattern.
        """

        class AgentResponse:
            def __init__(self, text: str):
                self.text = text

            def __str__(self) -> str:
                return self.text

        try:
            asyncio.get_running_loop()
            # If we are here, there is a running loop.
            # We use a worker thread to run the async generate()
            # to avoid "RuntimeError: asyncio.run() cannot be called from a running event loop"
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.controller.generate(query))
                result = future.result()
        except RuntimeError:
            # No running loop, we can use the standard controller.run (which uses asyncio.run)
            result = self.controller.run(query)

        # If result is already a Response object or similar, handle it
        if hasattr(result, "text"):
            return result

        return AgentResponse(text=str(result))

    async def run_async_text(self, query: str) -> str:
        """Asynchronous version of run that returns text directly."""
        result = await self.controller.generate(query)
        return str(result)


class VerifiedAgent(Agent):
    """An Agent that automatically performs research to verify instructions before acting.

    It uses a research phase to ensure technical accuracy and minimize hallucinations.
    """

    def __init__(
        self,
        name: str,
        model: ADKConfig | str | None = None,
        system_prompt: str | Callable[..., str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, model, system_prompt, **kwargs)
        self.add_tool(GoogleSearchTool().as_tool_list()[0])

    async def run_verified(self, query: str) -> str:
        """Run the agent with an explicit verification/research step."""
        research_prompt = (
            f"Research the following instruction to verify its technical accuracy and feasibility: {query}\n"
            "Search for official documentation, recent community discussions, and potential pitfalls. "
            "Identify if any part of the instruction seems like a hallucination or is outdated."
        )
        research_result = await self.run_async_text(research_prompt)

        implementation_prompt = (
            f"Based on the following research:\n{research_result}\n\n"
            f"Implement the original instruction: {query}\n"
            "If the research suggests the instruction is invalid or harmful, explain why instead of implementing."
        )
        return await self.run_async_text(implementation_prompt)
