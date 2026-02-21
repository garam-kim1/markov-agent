# Edge Routing and Flow Logic

## `Edge`
Connects two nodes. $f(S) 	o (next\_id, p)$.

## `Flow`
High-level routing logic.
```python
from markov_agent.topology.edge import Edge, Flow

# Simple static edge
edge = Edge(from_node="start", to_node="end")

# Conditional edge (Flow)
edge = Edge(
    from_node="router",
    to_node="success",
    condition=lambda s: s.retry_count < 3
)
```

## Transition Probabilities
Transitions can be probabilistic (especially in simulation).
```python
edge = Edge(from_node="A", to_node="B", p=0.7)
```
