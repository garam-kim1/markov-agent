# Markov Agent Architecture

## Philosophy: Markov Engineering

`markov-agent` is a **specialized wrapper for the Google Agent Development Kit (ADK)** that implements the **Markov Engineering** paradigm. We treat Large Language Models (LLMs) as **Probabilistic Processing Units (PPUs)**—stochastic components within a deterministic system built on top of the Google ADK.

## The Agent as a Markov Decision Process (MDP)

The strategic advantage of treating agent behavior as a formal MDP lies in moving away from simple text generation and toward a structured state-transition model.

*   **State ($S$):** The formal context, including the state schema, memory, and task completion progress. Managed via Pydantic for strict data contracts.
*   **Transition ($T$):** The logic graph that defines the probability of moving from the current state to a valid subsequent state.

This framework enables **Trajectory Optimization**, balancing **Exploration** (maximizing $pass@k$ via parallel sampling) with **Guardrails** (maximizing $pass\wedge k$ via strict State Space constraints).

## The Cognitive Kernel: Infrastructure for Reliability

The library provides the "Markov Workbench"—infrastructure dedicated to reliability governance through four core components:

| Component | Technical Function | Business Imperative |
| :--- | :--- | :--- |
| **State Schema Registry** | Enforces strict Pydantic contracts between nodes. | **Preventing Downstream Corruption:** Guarantees agent outputs are compatible with legacy systems. |
| **Pass@k Simulation Engine** | Executes parallel scenario testing against "Golden Datasets". | **Confidence in Deployment:** Moves from subjective "trust" to mathematical proof. |
| **Trajectory Recorder** | A "Black Box" flight recorder logging $\Delta S$ at every traversal. | **Auditability:** Enables "Time-Travel Debugging" and compliance records. |
| **Latency & Cost Governor** | Middleware for task complexity estimation and routing. | **Unit Economics:** Ensures compute cost does not exceed problem value. |

## System Components

### 1. The Core (`src/markov_agent/core`)
*   **State:** The single source of truth. All nodes receive and return a `BaseState` object.
*   **Events:** An event bus that broadcasts every state change, node execution, and error.

### 2. The Topology (`src/markov_agent/topology`)
*   **Graph:** The execution engine (the "Skeleton"). It manages the flow of control.
*   **Nodes:** The units of work. A Node transforms State $S_t \to S_{t+1}$.
*   **Edges:** Routing functions $T(s, a) \to s'$.

### 3. The Engine (`src/markov_agent/engine`)
*   **PPU (Probabilistic Processing Unit):** Wraps `google-adk` to provide:
    *   **Parallel Trajectory Generation ($pass@k$):** Running $k$ independent paths.
    *   **Verification:** Using deterministic critics to select successful outcomes.
    *   **Retry Policy:** Recovering from stochastic failures.

## Strategic Horizon: Auto-Topology

The future of the Markov Engine involves a shift from manual graph engineering to automated meta-learning via **Genetic Logic Optimization (GLO)**. 

GLO treats prompts and nodes as "genes" in an evolutionary algorithm, optimizing the agent variation that maximizes a multi-variable reward function:
$$R = \alpha \cdot \text{Accuracy} - \beta \cdot \text{TokenCost} - \gamma \cdot \text{Latency}$$

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
