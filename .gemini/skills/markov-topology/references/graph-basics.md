# Markov Graph Construction

## Graph Basics
A `Graph` is a directed graph where nodes process state and edges define transitions. It inherits from `google.adk.agents.BaseAgent`.

## State Definition
Every `Graph` needs a `StateT` that inherits from `BaseState`.
```python
from markov_agent.core.state import BaseState

class MyState(BaseState):
    input_text: str = ""
    output_text: str = ""
    retry_count: int = 0
```

## Graph Construction
```python
from markov_agent.topology.graph import Graph

graph = Graph[MyState](name="my-graph")
graph.add_node(my_node)
graph.add_edge(edge1)
```

## Running the Graph
```python
result = await graph.run(MyState(input_text="hello"))
```
