# Markov Engine

> **From Prompt Engineering to Markov Engineering.**

**Markov Engine** is a **Probabilistic Control System** that treats Large Language Models (LLMs) not as chatbots, but as **Probabilistic Processing Units (PPUs)** within a deterministic topology. 

It provides the mathematical and architectural scaffolding to build reliable, self-correcting AI systems by constraining the stochastic nature of LLMs within rigorous state machines.

## 🧠 The Paradigm Shift

| Feature | Classic "Chatbot" | Markov Engine |
| :--- | :--- | :--- |
| **Mental Model** | Conversation / Magic Box | Stochastic CPU (PPU) |
| **Control Flow** | Linear / Scripted | Directed Acyclic (or Cyclic) Graphs |
| **Reliability** | "Hope it works" | $P(Success) = 1 - (1 - p)^k$ |
| **State** | Mutable / Messy | Immutable / Time-Travel Ready |
| **Debugging** | Print statements | Monte Carlo Simulations |

## 🚀 Key Concepts

### 1. The PPU (Probabilistic Processing Unit)
We wrap the Google Agent Development Kit (ADK) in a `ProbabilisticNode`. Instead of a single generation, we support **Parallel Trajectory Generation ($pass@k$)**. By sampling multiple futures and selecting the best one (via a lightweight critic or heuristic), we mathematically increase system reliability.

### 2. Topology as Code
Define your application logic as a Graph. Nodes perform work (deterministically or probabilistically), and Edges act as routers, deciding the next step based on the current State.

### 3. Simulation Workbench
Don't guess if your prompt works. Run it 50 times. The built-in `MonteCarloRunner` executes your graph against a dataset to calculate **Accuracy** and **Consistency** metrics before you deploy.

## 📦 Installation

This project is managed by `uv`, a distinctively fast Python package manager.

```bash
# Clone the repository
git clone https://github.com/yourusername/markov_agent.git
cd markov_agent

# Install dependencies
uv sync
```

## ⚡ Quick Start

Here is a simple example of a linear chain using a mock PPU:

```python
import asyncio
from markov_agent.core.state import BaseState
from markov_agent.topology.node import BaseNode
from markov_agent.containers.chain import Chain

# 1. Define your State
class MyState(BaseState):
    input_text: str
    processed_text: str = ""

# 2. Define a Deterministic Node (or use ProbabilisticNode for LLMs)
class UpperCaseNode(BaseNode[MyState]):
    async def execute(self, state: MyState) -> MyState:
        return state.update(processed_text=state.input_text.upper())

# 3. Create the Topology
chain = Chain(nodes=[UpperCaseNode(name="uppercase_worker")])

# 4. Run it
async def main():
    initial = MyState(input_text="hello world")
    final = await chain.run(initial)
    print(f"Result: {final.processed_text}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🏗️ Architecture

*   **`core/`**: The immutable `State` registry and `EventBus` for observability.
*   **`topology/`**: The graph engine (`Graph`, `Node`, `Edge`). It handles routing and cyclic safeguards.
*   **`engine/`**: The "Cognitive Kernel". Wraps the ADK and implements the $pass@k$ parallel sampler.
*   **`containers/`**: High-level patterns like `Chain` (Linear) and `Swarm` (Supervisor/Worker).
*   **`simulation/`**: The reliability lab. Contains `MonteCarloRunner` and metric calculators.

## 🛠️ Development

We enforce strict typing and code quality standards.

```bash
# Run tests
uv run pytest

# Format and Lint
uv run ruff check . --fix
uv run ruff format .
```

## 📜 License

MIT License.