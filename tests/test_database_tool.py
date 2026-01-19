import pytest
from markov_agent.tools.database import DatabaseTool
from sqlalchemy import create_engine, text

@pytest.fixture
def db_tool():
    # Setup in-memory DB
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE users (id INTEGER, name TEXT)"))
        conn.execute(text("INSERT INTO users VALUES (1, 'Alice')"))
        conn.execute(text("INSERT INTO users VALUES (2, 'Bob')"))
        conn.commit()
    
    # Pass the same connection string? 
    # SQLite memory DBs are unique per connection if url is simple.
    # To share, we need to pass the *creator* or use a shared url with cache=shared?
    # Or just let the tool create its own engine on a file-based sqlite for test?
    # Easier: "sqlite://" creates a new memory db.
    # If the tool creates its own engine, it won't see the data we inserted if we used a different engine.
    
    # Workaround: Use a temp file db for the test to ensure persistence across connections
    pass

def test_database_tool_safeguard():
    tool = DatabaseTool("sqlite:///:memory:")
    result = tool.query("DELETE FROM users")
    assert "Error: Only SELECT queries are allowed" in result

def test_database_tool_query(tmp_path):
    # Use a file-based DB so the tool can open it
    db_file = tmp_path / "test.db"
    conn_str = f"sqlite:///{db_file}"
    
    # Pre-populate
    engine = create_engine(conn_str)
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE users (id INTEGER, name TEXT)"))
        conn.execute(text("INSERT INTO users VALUES (1, 'Alice')"))
        conn.commit()
    
    tool = DatabaseTool(conn_str)
    result = tool.query("SELECT * FROM users")
    assert "Alice" in result
    assert "1" in result

def test_database_tool_schema(tmp_path):
    db_file = tmp_path / "schema_test.db"
    conn_str = f"sqlite:///{db_file}"
    
    engine = create_engine(conn_str)
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE items (id INTEGER, price REAL)"))
        conn.commit()
        
    tool = DatabaseTool(conn_str)
    schema = tool.get_schema()
    
    assert "Table: items" in schema
    assert "price (FLOAT)" in schema or "price (REAL)" in schema or "price (DOUBLE)" in schema
    # Type representation varies by sqlalchemy version/dialect, so check relaxed.
