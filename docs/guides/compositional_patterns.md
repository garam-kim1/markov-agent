# Compositional Patterns in Markov Agent

The true power of the `markov-agent` framework lies not in individual prompts, but in **Topology**. By composing Nodes into deterministic structures, you can build systems that are significantly more reliable than their individual parts.

The `markov_agent.containers` module provides high-level abstractions to assemble these topologies quickly.

## 1. The Chain (Linear Pipeline)

The simplest pattern. Data flows sequentially from one node to the next.

**Use Case:** Sequential data transformation (e.g., Extract -> Transform -> Load).

```python
from markov_agent.containers.chain import Chain

pipeline = Chain(
    name="etl_pipeline",
    nodes=[extractor_node, transformer_node, loader_node]
)
# Execution: extractor -> transformer -> loader
```

## 2. The Swarm (Supervisor/Worker)

A "Hub and Spoke" architecture. A Supervisor node decides which Worker node to delegate to, and Workers report back to the Supervisor.

**Use Case:** Multi-domain assistants where a central brain manages specialized sub-agents (e.g., a Customer Service bot delegating to "Billing", "Tech Support", or "Sales").

```python
from markov_agent.containers.swarm import Swarm

swarm = Swarm(
    name="support_swarm",
    supervisor=triage_node,
    workers=[billing_agent, tech_agent, sales_agent],
    router_func=my_router_logic  # Function mapping State -> Worker Name
)
```

**How it works:**
1.  **Supervisor** executes.
2.  **Router Function** checks the Supervisor's output state and picks a Worker.
3.  **Worker** executes.
4.  **Loop:** Control returns to the Supervisor to decide the next step (or finish).

## 3. Parallel (Map-Reduce)

Executes multiple nodes concurrently and merges their results.

**Use Case:** Ensemble voting, generating multiple drafts simultaneously, or parallel research on different sub-topics.

```python
from markov_agent.containers.parallel import ParallelNode

ensemble = ParallelNode(
    name="writers_room",
    nodes=[creative_writer, technical_writer, editor_writer]
)
```

**State Merging:**
Since `State` is immutable, the `ParallelNode` runs each sub-node on a *copy* of the state. It then identifies changes in each branch and merges them back. Conflict resolution is currently "last-write-wins" or additive, depending on the implementation details of the specific merge strategy used.

## 4. Loop (Iterative Refinement)

Repeats a Node (or a sub-graph) until a condition is met.

**Use Case:** Self-correction, Code refinement loops (Code -> Test -> Fix -> Test), or recursive searching.

```python
from markov_agent.containers.loop import Loop

optimizer = Loop(
    node=refinement_chain,
    condition=lambda state: state.quality_score < 90,
    max_iterations=5
)
```

## 5. Nested (Fractal Architecture)

A `Graph` is valid `Node`. This means you can use an entire Agent as a single step within a larger Agent.

**Use Case:** Building complex systems from modular, tested components.

```python
from markov_agent.containers.nested import NestedNode

# Define a sub-agent
research_agent = Graph(name="researcher", ...)

# Use it as a node in a larger flow
main_flow = Chain(
    nodes=[
        briefing_node,
        NestedNode(graph=research_agent), # Treats the whole graph as one atomic step
        summary_node
    ]
)
```
