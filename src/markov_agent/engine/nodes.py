from markov_agent.engine.adk_wrapper import ADKConfig
from markov_agent.engine.ppu import ProbabilisticNode
from markov_agent.tools.search import GoogleSearchTool


class SearchNode(ProbabilisticNode):
    """
    A ProbabilisticNode pre-configured with Google Search capabilities.
    Uses the native Google ADK GoogleSearchTool.
    """

    def __init__(
        self, name: str, adk_config: ADKConfig, prompt_template: str, **kwargs
    ):
        # Ensure tools list exists
        if adk_config.tools is None:
            adk_config.tools = []

        # Add Google Search Tool if not present
        # Note: We instantiate the wrapper from markov_agent.tools.search
        # which returns the native tool via as_tool_list()
        search_tool_wrapper = GoogleSearchTool()
        adk_config.tools.extend(search_tool_wrapper.as_tool_list())

        super().__init__(
            name=name, adk_config=adk_config, prompt_template=prompt_template, **kwargs
        )
