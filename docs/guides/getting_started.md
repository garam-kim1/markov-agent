# Getting Started with Markov Agent

`markov-agent` is a **Probabilistic Control System** designed to bring deterministic control to stochastic LLM processes through the **Markov Engineering** paradigm.

## Core Concepts

*   **PPU (Probabilistic Processing Unit):** We treat the LLM as a stochastic CPU.
*   **Topology:** The structural "skeleton" that enforces business logic.
*   **The Markov Workbench:** The internal platform for engineering high-reliability systems via simulation and trajectory recording.

## Prerequisites
...

*   **Python:** Version 3.12 or higher.
*   **Package Manager:** `uv` (by Astral). This project strictly uses `uv` for dependency management.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd markov-agent
    ```

2.  **Install dependencies:**
    ```bash
    uv sync
    ```
    This will create a virtual environment and install all required packages defined in `pyproject.toml`.

## Running Tests

We use `pytest` for testing. To run the suite:

```bash
uv run pytest
```

## Basic Usage

To run a simple agent topology, you can use the examples provided in `examples/`.

For instance, to run the Code Improver Agent:

```bash
uv run examples/code_improver_agent.py
```

## Developing New Agents

1.  **Define State:** Subclass `BaseState` in `src/markov_agent/core/state.py` (or your own file) to define your agent's memory.
2.  **Create Nodes:** Subclass `ProbabilisticNode` or `Node` to implement specific steps (e.g., "Draft", "Review").
3.  **Compose Topology:**
    *   **Option A (Fluent Decorators):** Use the `@graph.node` and `@graph.task` decorators for rapid prototyping.
    *   **Option B (Connect & Route):** Use `g.connect(a >> b >> c)` and `g.route(src, targets)` for clear control flow.
    *   **Option C (Containers):** Use high-level patterns like `Chain`, `Loop`, or `Parallel` for rapid assembly.

    ```python
    from markov_agent.topology.graph import Graph

    g = Graph("Agent", state_type=MyState)

    @g.node()
    async def brainstorm(state: MyState):
        """Generate 3 ideas for {{ topic }}"""

    @g.node()
    async def select(state: MyState):
        """Pick the best idea from: {{ ideas }}"""

    # Linear connection
    g.connect(brainstorm >> select)

    # Visualization
    g.visualize()
    ```
4.  **Simulate:** Use `g.simulate()` to verify your agent's reliability before deployment. "If you didn't test it 50 times, it doesn't work."
    ```python
    results = await g.simulate(dataset=my_dataset, n_runs=50)
    ```
