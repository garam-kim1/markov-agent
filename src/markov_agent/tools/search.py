from google.adk.tools.google_search_tool import GoogleSearchTool as AdkGoogleSearchTool
from pydantic import BaseModel


class GoogleSearchTool:
    """
    A wrapper around Google ADK's native GoogleSearchTool.
    """

    def __init__(self):
        self._tool = AdkGoogleSearchTool()

    def as_tool_list(self) -> list:
        return [self._tool]
