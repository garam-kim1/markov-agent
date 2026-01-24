from typing import Any, Optional

from google.adk.tools import ToolContext
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


class DatabaseTool:
    """
    A tool for executing SQL queries against a database.
    Compatible with Google ADK tool registration.
    """

    def __init__(self, connection_string: str):
        self.engine: Engine = create_engine(connection_string)

    def query(self, sql_query: str, tool_context: Optional[ToolContext] = None) -> str:
        """
        Executes a read-only SQL query and returns the results.

        Args:
            sql_query: The SQL query to execute.
            tool_context: Optional ADK ToolContext for access to session/invocation details.

        Returns:
            String representation of the results.
        """
        # Example of using tool_context if available
        if tool_context:
            # We could log the invocation ID or check permissions here
            pass

        # Security check: Simple safeguard against obvious non-SELECTs
        # In production, use a read-only user!
        if not sql_query.strip().lower().startswith("select"):
            return "Error: Only SELECT queries are allowed."

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql_query))
                rows = [dict(row._mapping) for row in result]
                return str(rows)
        except Exception as e:
            return f"Database Error: {e}"

    def get_schema(self) -> str:
        """
        Returns the schema of the database to help the agent understand tables.
        """
        # This is a simplified schema dumper.
        # For complex DBs, use reflection.
        try:
            from sqlalchemy import inspect

            inspector = inspect(self.engine)
            schema_info = []
            for table_name in inspector.get_table_names():
                columns = inspector.get_columns(table_name)
                cols_desc = ", ".join([f"{c['name']} ({c['type']})" for c in columns])
                schema_info.append(f"Table: {table_name} | Columns: {cols_desc}")
            return "\n".join(schema_info)
        except Exception as e:
            return f"Schema Error: {e}"

    def as_tool_list(self) -> list[Any]:
        """
        Returns the methods as a list of callables for ADK.
        """
        return [self.query, self.get_schema]
