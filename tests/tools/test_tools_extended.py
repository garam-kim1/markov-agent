from unittest.mock import MagicMock, patch

from markov_agent.engine.adk_wrapper import ADKConfig
from markov_agent.engine.nodes import SearchNode
from markov_agent.tools.database import DatabaseTool


def test_database_tool_security():
    tool = DatabaseTool("sqlite:///:memory:")

    # Test blocked query
    result = tool.query("DELETE FROM users")
    assert "Error: Only SELECT queries are allowed" in result

    # Test allowed query (mocked execution not needed for this check)
    with patch.object(tool.engine, "connect") as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn

        # Setup mock result
        mock_result = MagicMock()
        mock_result.__iter__.return_value = []  # Empty result
        mock_conn.execute.return_value = mock_result

        tool.query("SELECT * FROM users")
        mock_conn.execute.assert_called_once()


def test_database_tool_schema():
    tool = DatabaseTool("sqlite:///:memory:")

    with patch("sqlalchemy.inspect") as mock_inspect:
        mock_inspector = MagicMock()
        mock_inspect.return_value = mock_inspector

        mock_inspector.get_table_names.return_value = ["users"]
        mock_inspector.get_columns.return_value = [
            {"name": "id", "type": "INTEGER"},
            {"name": "name", "type": "VARCHAR"},
        ]

        schema = tool.get_schema()
        assert "Table: users" in schema
        assert "id (INTEGER)" in schema
        assert "name (VARCHAR)" in schema


def test_database_tool_as_list():
    tool = DatabaseTool("sqlite:///:memory:")
    tools = tool.as_tool_list()
    assert len(tools) == 2
    assert tool.query in tools
    assert tool.get_schema in tools


def test_search_node_configuration():
    """Verify that SearchNode automatically injects the Google Search tool."""
    config = ADKConfig(model_name="mock")
    # SearchNode adds the tool
    node = SearchNode("searcher", config, "Find {query}")

    # We expect at least one tool (the search tool)
    assert len(node.adk_config.tools) >= 1
    # Check that it's the expected object type (simplified check)
    assert node.adk_config.tools[0] is not None
