# Markov Agent Architecture

## Philosophy: Markov Engineering

`markov-agent` rejects the "Prompt Engineering" paradigm in favor of **Markov Engineering**. We treat Large Language Models (LLMs) not as creative partners, but as **Probabilistic Processing Units (PPUs)**—stochastic components within a deterministic system.

Our goal is to build reliable control systems where:
*   **State ($S$)** is immutable and strongly typed.
*   **Transitions ($T$)** are defined by a directed graph (Topology).
*   **Execution** is bounded and observable.

## System Components

### 1. The Core (`src/markov_agent/core`)
*   **State:** The single source of truth. All nodes receive and return a `BaseState` object.
*   **Events:** An event bus that broadcasts every state change, node execution, and error for observability.

### 2. The Topology (`src/markov_agent/topology`)
*   **Graph:** The execution engine. It manages the flow of control, ensuring `max_steps` are respected to avoid halting problems.
*   **Nodes:** The units of work. A Node transforms State $S_t \to S_{t+1}$.
*   **Edges:** The routing logic. Deterministic functions that decide the next Node based on the current State.

### 3. The Engine (`src/markov_agent/engine`)
*   **PPU (Probabilistic Processing Unit):** The wrapper around the LLM. It handles:
    *   **Parallel Trajectory Generation ($pass@k$):** Running the model $k$ times to explore different paths.
    *   **Validation:** Ensuring outputs match the expected schema.
    *   **Retries:** Automatically recovering from stochastic failures.
*   **ADK Wrapper:** Integrates `google-adk` to provide a unified interface for various models.

### 4. Simulation (`src/markov_agent/simulation`)
*   **Monte Carlo Runner:** A workbench for running a topology $N$ times to statistically verify its reliability.
*   **Metrics:** Tools to calculate Pass@1 and Pass@k accuracy.

### 5. Containers (`src/markov_agent/containers`)
High-level architectural patterns that compose Nodes into common structures:
*   **`Chain`:** A linear sequence of nodes ($A \to B \to C$) where state is passed forward.
*   **`Sequential`:** Strict ordered execution, similar to Chain but with more rigid control.
*   **`Parallel`:** Executes multiple nodes concurrently, merging their state updates (Map-Reduce pattern).
*   **`Loop`:** Iterates a node or subgraph until a condition is met (e.g., "Reviewer score > 8").
*   **`Nested`:** Encapsulates an entire `Graph` as a single `Node`, allowing for fractal architectures.
*   **`Swarm`:** A Supervisor/Worker pattern for multi-agent delegation.

### 6. Tools (`src/markov_agent/tools`)
Production-ready integrations for real-world capabilities:
*   **`AgentAsTool`:** The recursive primitive. Wraps any Markov Node/Agent as a tool, allowing agents to call other agents.
*   **`DatabaseTool`:** Safe SQL execution via SQLAlchemy.
*   **`MCPTool`:** Integration with Model Context Protocol servers for dynamic tool discovery.

## Data Flow

1.  **Initialization:** A `Graph` is instantiated with a set of `Nodes` and `Edges`.
2.  **Input:** An initial `State` is injected.
3.  **Cycle:**
    *   The Graph identifies the current Node.
    *   The Node executes (potentially using a PPU) and returns a new State.
    *   The Graph uses Edges to determine the next Node.
4.  **Termination:** The Graph stops when a terminal Node is reached or `max_steps` is exceeded.
