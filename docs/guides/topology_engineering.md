# Topology Engineering: Structural Architectures for Logic

Topology Engineering is the practice of enforcing business logic through formal graph structures. By selecting an optimized architectural "Skeleton," we ensure the system's reasoning is constrained by the requirements of the task.

## The Fallacy of Simplicity

Organizations often assume simple tasks (scheduling, FAQ bots) do not require complex architecture. However, in an enterprise context, simple tasks carry high consequences and hidden entropy. Sophisticated topology is required not because the task is hard, but because the **reliability requirement is absolute**.

---

## Architectural Topologies

### 1. The Linear Chain (The Pipeline)
Focused on one-way flow ($A \to B \to C$), this topology is reserved for low-complexity, deterministic tasks like data extraction or summarization where a retry mechanism is unnecessary.

```python
from markov_agent.containers.chain import Chain
pipeline = Chain(nodes=[extractor, transformer, loader])
```

### 2. The Cyclic Graph (The Loop)
Specifically designed for high $pass@k$ tasks, this structure incorporates "Critic" nodes. If a task fails verification, the logic loops back to a previous state, utilizing $pass@k$ principles to enable self-correction.

```python
from markov_agent.containers.loop import Loop
optimizer = Loop(
    node=generator_chain,
    condition=lambda state: state.quality_score < 90
)
```

### 3. The Hierarchical Swarm (The Org Chart)
This architecture utilizes a **Supervisor** node to manage global state and delegate specialized sub-tasks to **Worker** nodes. The Supervisor acts as the arbiter of the State Schema Registry, ensuring separation of concerns.

```python
from markov_agent.containers.swarm import Swarm

swarm = Swarm(
    name="enterprise_controller",
    supervisor=triage_node,
    workers=[billing_agent, tech_agent, sales_agent]
)
```

**Strategic Advantage:** The Hierarchical Swarm ensures that complex enterprise resource planning is broken down into manageable, highly-reliable sub-trajectories.

---

## Pattern Library

### Parallel (Map-Reduce)
Executes multiple nodes concurrently and merges results. Used for ensemble voting or parallel research.

### Nested (Fractal Modularity)
A `Graph` is a valid `Node`. Build complex systems from modular, tested sub-graphs.
