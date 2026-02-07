from typing import Any

from google.adk.tools import agent_tool

from markov_agent.topology.node import BaseNode


class AgentAsTool:
    """Wraps a Markov Agent Node (BaseNode) as a tool for use by other agents.

    Allows for Hierarchical Task Decomposition and Explicit Invocation patterns.
    """

    def __init__(
        self,
        node: BaseNode,
        *,
        skip_summarization: bool = False,
        include_plugins: bool = True,
    ):
        self.node = node
        # ADK's AgentTool wraps a BaseAgent.
        # Since BaseNode inherits from BaseAgent, this works directly.
        self._tool = agent_tool.AgentTool(
            agent=node,
            skip_summarization=skip_summarization,
            include_plugins=include_plugins,
        )

    def as_tool_list(self) -> list[Any]:
        """Return the wrapped ADK AgentTool in a list."""
        return [self._tool]
