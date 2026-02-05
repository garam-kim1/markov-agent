from typing import Any

from google.adk.tools import agent_tool

from markov_agent.topology.node import BaseNode


class AgentAsTool:
    """Wraps a Markov Agent Node (BaseNode) as a tool for use by other agents.
    Allows for Hierarchical Task Decomposition and Explicit Invocation patterns.
    """

    def __init__(self, node: BaseNode):
        self.node = node
        # ADK's AgentTool wraps a BaseAgent.
        # Since BaseNode inherits from BaseAgent, this works directly.
        self._tool = agent_tool.AgentTool(agent=node)

    def as_tool_list(self) -> list[Any]:
        """Returns the wrapped ADK AgentTool in a list."""
        return [self._tool]
