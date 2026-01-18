# Getting Started with Markov Agent

## Prerequisites

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
    *   **Option A (Manual):** Use `Graph` to connect your nodes with edges.
    *   **Option B (Containers):** Use high-level patterns like `Chain`, `Loop`, or `Parallel` for rapid assembly.
    ```python
    from markov_agent.containers.chain import Chain
    agent = Chain(nodes=[draft_node, review_node])
    ```
4.  **Simulate:** Use `MonteCarloRunner` to verify your agent's reliability before deployment.
