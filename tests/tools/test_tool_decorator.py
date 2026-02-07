from google.adk.tools.function_tool import FunctionTool

from markov_agent.tools import tool


def test_tool_decorator_basic():
    @tool
    def my_tool(x: int) -> int:
        """My tool description."""
        return x + 1

    assert isinstance(my_tool, FunctionTool)
    assert my_tool.name == "my_tool"
    assert "My tool description" in my_tool.description


def test_tool_decorator_with_confirmation():
    @tool(confirmation=True)
    def sensitive_tool(data: str) -> str:
        return f"Processed {data}"

    assert isinstance(sensitive_tool, FunctionTool)
    assert sensitive_tool._require_confirmation is True


def test_tool_decorator_no_parens():
    @tool
    def another_tool(s: str) -> str:
        return s.upper()

    assert isinstance(another_tool, FunctionTool)
    assert another_tool.name == "another_tool"


def test_tool_decorator_with_parens_no_args():
    @tool()
    def yet_another_tool(s: str) -> str:
        return s.lower()

    assert isinstance(yet_another_tool, FunctionTool)
    assert yet_another_tool.name == "yet_another_tool"
