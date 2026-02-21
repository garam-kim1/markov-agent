---
name: markov-topology
description: Expert in designing and implementing Markovian Finite State Machines (FSM) using markov-agent. Use for building deterministic topologies with stochastic (LLM) transitions, PPU (Probabilistic Processing Units), and graph-based routing in Python.
---

# Markov Topology Architect

## Philosophy & Architecture
`markov-agent` wraps `google-adk` to provide a topology-based control system. It shifts Prompt Engineering into Markov Engineering by using a Directed Graph where nodes are processing units (Probabilistic or Functional) and edges are transition rules.

## Core Components
1. **`Graph`**: The main execution engine. It's an ADK-compatible `BaseAgent`.
2. **`BaseState`**: Pydantic model with history and Markov view.
3. **`ProbabilisticNode` (PPU)**: LLM-driven nodes that determine the next state by sampling.
4. **`FunctionalNode`**: Python-based deterministic logic nodes.
5. **`Edge`**: Routing logic $f(S) 	o (next\_id, p)$.

## Execution Strategy
1. **Define the State**: Create a subclass of `BaseState` with all necessary fields.
2. **Define Nodes**:
   - For LLM tasks (reasoning/extraction), use `ProbabilisticNode`.
   - For deterministic tasks (math/database), use `FunctionalNode`.
3. **Define Edges**:
   - Use `Edge` to connect nodes.
   - Use `Flow` for conditional routing or probabilistic transitions.
4. **Construct the Graph**:
   - Add all nodes and edges to the `Graph`.
   - Start the graph by calling `await graph.run(...)`.

## Context Loading
- If working on **Graph/Topology construction**, read `references/graph-basics.md`.
- If working on **PPU/LLM Node configuration**, read `references/ppu-nodes.md`.
- If working on **Edge routing and Flow logic**, read `references/edge-routing.md`.

## Best Practices
- Use **Parallel Sampling** in `ProbabilisticNode` for increased reliability.
- Favor `BaseState` history for long-term memory instead of giant context windows.
- Visualize the graph with `graph.to_mermaid()` to debug the flow.
