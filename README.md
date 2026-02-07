# Markov Engine 🧠📐

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![uv](https://img.shields.io/badge/managed%20by-uv-arc.svg)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Moving from "Prompt Engineering" to "Markov Engineering".**

**Markov Engine** (`markov-agent`) is a **Probabilistic Control System** that serves as a high-reliability execution layer over the **Google Agent Development Kit (ADK)**. It treats Large Language Models (LLMs) as **Probabilistic Processing Units (PPUs)**—stochastic components nested within deterministic topologies.

---

## 🏗️ The Paradigm Shift

We are transitioning from **Generative AI** (stateless, creative) to **Agentic AI** (stateful, reliable). Markov Engine provides the formal mathematical treatment required to bridge the "GenAI Paradox" where pilots fail to reach production due to probabilistic decay.

| Feature | Classic "Chatbot" | Markov Engine |
| :--- | :--- | :--- |
| **Mental Model** | Conversation / Magic Box | **Stochastic CPU (PPU)** |
| **Control Flow** | Linear / Scripted | **Directed Graphs (Topology)** |
| **Reliability** | "Hope it works" | **$P(Success) = 1 - (1 - p)^k$** |
| **State** | Mutable / Messy | **Immutable / Strongly Typed** |
| **Verification** | Human-in-the-loop | **Monte Carlo Simulations** |

---

## 📐 Mathematical Foundations

Markov Engine formalizes agentic failure to enable rigorous engineering.

### The Joint Probability Trap
In multi-step workflows, success probability ($P_{total}$) decays exponentially:
$$P_{total} = \prod_{i=1}^{n} p_i$$
A 5-step process with 90% accuracy per step yields only **~59% total reliability**. We solve this through:

*   **Accuracy ($pass@k$):** $1 - (1 - p)^k$ — Parallel verification to transform low-probability reasoning into high-reliability output.
*   **Consistency ($pass\wedge k$):** $p^k$ — Validating stability across batches to ensure enterprise-grade production readiness.

---

## 🚀 Key Features

*   **PPU Design Pattern:** Treat ADK Models as CPUs that sometimes lie. Enforce reliability via parallel sampling and deterministic verification.
*   **Topology Engineering:** Select optimized "Skeletons" (Linear, Cyclic, Hierarchical Swarm) to constrain reasoning.
*   **Cognitive Kernel:** Infrastructure for reliability assurance, including a **State Schema Registry** (Pydantic) and **Trajectory Recorder**.
*   **Advanced Deliberative Logic:** Support for **System 2** reasoning using **MCTS** and **Bayesian Information Gain**.
*   **Deep ADK Integration:** Native compatibility with Google ADK agents, tools, and servers.
*   **Multi-Provider (LiteLLM):** Swap between Gemini, OpenAI, Anthropic, or Local models (Qwen/Llama) with a single config flag.

---

## 📦 Installation

Markov Engine requires **Python 3.12+** and is optimized for the **`uv`** package manager.

```bash
# Clone and setup
git clone https://github.com/yourusername/markov_agent.git
cd markov_agent
uv sync
```

---

## ⚡ Quick Start: The "Markovian" Way

Define a strongly-typed state and a deterministic node (or a `ProbabilisticNode` for LLMs).

```python
import asyncio
from markov_agent.core.state import BaseState
from markov_agent.topology.node import BaseNode
from markov_agent.containers.chain import Chain

# 1. Define Immutable, Strongly-Typed State
class MyState(BaseState):
    input_text: str
    processed_text: str = ""

# 2. Define a Node (The Transition Logic)
class UpperCaseNode(BaseNode[MyState]):
    async def execute(self, state: MyState) -> MyState:
        return state.update(processed_text=state.input_text.upper())

# 3. Create the Topology (The Skeleton)
agent = Chain(nodes=[UpperCaseNode(name="worker")])

# 4. Run with Full Observability
async def main():
    final_state = await agent.run(MyState(input_text="hello world"))
    print(f"Result: {final_state.processed_text}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 📂 Project Structure

*   **`core/`**: Immutable State management and the Event Bus.
*   **`topology/`**: Graph engine, Nodes, and Routing logic.
*   **`engine/`**: The "Cognitive Kernel" & PPU implementation.
*   **`containers/`**: High-level patterns (Chain, Swarm, Loop, Parallel).
*   **`simulation/`**: Reliability workbench (Monte Carlo Runner).
*   **`tools/`**: ADK-native tool wrappers (Database, MCP, Agent-as-Tool).

---

## 📚 Documentation Index

*   [**Building Coding Agents**](docs/guides/building_coding_agents.md): Constructing self-correcting software agents.
*   [**Topology Engineering**](docs/guides/topology_engineering.md): Designing structural architectures for logic.
*   [**Reliability Engineering**](docs/guides/reliability_engineering.md): Quantifying uncertainty with $pass@k$.
*   [**Advanced Deliberative Logic**](docs/guides/deliberative_logic.md): "System 2" reasoning and MCTS.
*   [**Architecture Deep Dive**](docs/architecture/overview.md): MDP formalization and the Cognitive Kernel.

---

## 🛠️ Development & Quality

We enforce strict typing and formatting standards.

```bash
uv run pytest             # Run the test suite
uv run ruff check . --fix # Lint and fix
uv run ruff format .      # Format code
```

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.