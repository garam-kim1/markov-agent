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

## 🧠 The Markovian Philosophy

Markov Engineering is built on the belief that **stochasticity is not a bug, but a property** that must be governed. Our philosophy rests on three pillars:

1.  **Deterministic Skeleton, Probabilistic Meat:** The business logic (Topology) must be a hard-coded, verifiable graph. The reasoning (PPUs) is allowed to be probabilistic, but only within the constraints of the skeleton.
2.  **State as the Single Source of Truth:** We treat agent behavior as a **Markov Decision Process (MDP)**. Every transition $S_t \to S_{t+1}$ is explicitly logged, immutable, and verifiable.
3.  **The 50x Rule:** "If you haven't simulated your agent 50 times against a golden dataset, you don't have an agent; you have a demo." Reliability is a statistical proof, not a feeling.

---

## 📐 Mathematical Foundations

Markov Engine formalizes agentic failure to enable rigorous engineering.

### The Joint Probability Trap
In multi-step workflows, success probability ($P_{total}$) decays exponentially:
$$P_{total} = \prod_{i=1}^{n} p_i$$
A 5-step process with 90% accuracy per step yields only **~59% total reliability**. We solve this through:

*   **Accuracy ($pass@k$):** $1 - (1 - p)^k$ — Running $k$ parallel trajectories to transform low-probability reasoning into high-reliability output.
*   **Consistency ($pass\wedge k$):** $p^k$ — Validating that the same input yields the same output across batches, ensuring stability.

---

## 🚀 How to Use Guide

Building with Markov Engine follows a strict, engineering-first workflow.

### 1. Define the State Contract
Agents must have a strongly-typed boundary. We use Pydantic for strict data validation.

```python
from markov_agent.core.state import BaseState

class CodingState(BaseState):
    requirement: str
    code: str = ""
    unit_tests: str = ""
    is_valid: bool = False
```

### 2. Implement a PPU (Probabilistic Processing Unit)
A PPU is an LLM wrapper that transforms state. It uses **Google ADK** under the hood.

```python
from markov_agent.engine.ppu import ProbabilisticNode
from markov_agent.engine.adk_wrapper import ADKConfig

coder_node = ProbabilisticNode(
    name="programmer",
    adk_config=ADKConfig(model_name="gemini-1.5-pro", temperature=0.2),
    prompt_template="Write Python code for: {{ requirement }}",
    state_type=CodingState
)
```

### 3. Assemble the Topology
Use **Containers** for common patterns (Chain, Loop, Parallel) or a manual **Graph** for complex FSMs.

```python
from markov_agent.containers.self_correction import SelfCorrectionLoop

# A loop that runs 'coder' then 'reviewer' until 'is_valid' is True
agent = SelfCorrectionLoop(
    worker=coder_node,
    reviewer=reviewer_node,
    max_iterations=3
)
```

### 4. Engineer Reliability (Simulation)
Never deploy a raw agent. Run it through the **MonteCarloRunner** to calculate its $pass@k$ metrics.

```python
from markov_agent.simulation.runner import MonteCarloRunner

runner = MonteCarloRunner(
    graph=agent,
    dataset=[CodingState(requirement="Sort a list"), ...],
    n_runs=10,
    success_criteria=lambda s: s.is_valid
)

results = await runner.run_simulation()
print(f"Reliability: {sum(r.success for r in results) / len(results):.2%}")
```

---

## ⚡ Quick Start: The Hello World

```python
import asyncio
from markov_agent.core.state import BaseState
from markov_agent.containers.chain import Chain
from markov_agent.topology.node import BaseNode

class MyState(BaseState):
    val: str

class UpperNode(BaseNode[MyState]):
    async def execute(self, state: MyState) -> MyState:
        return state.update(val=state.val.upper())

async def main():
    agent = Chain(nodes=[UpperNode(name="u")])
    final = await agent.run(MyState(val="hello"))
    print(final.val) # HELLO

asyncio.run(main())
```

---

## 🚀 Key Features

*   **PPU Design Pattern:** Treat ADK Models as CPUs that sometimes lie. Enforce reliability via parallel sampling and deterministic verification.
*   **Topology Engineering:** Select optimized "Skeletons" (Linear, Cyclic, Hierarchical Swarm) to constrain reasoning.
*   **Cognitive Kernel:** Infrastructure for reliability assurance, including a **State Schema Registry** (Pydantic) and **Trajectory Recorder**.
*   **Advanced Deliberative Logic:** Support for **System 2** reasoning using **MCTS** and **Bayesian Information Gain**.
*   **Deep ADK Integration:** Native compatibility with Google ADK agents, tools, and servers.
*   **Multi-Provider (LiteLLM):** Swap between Gemini, OpenAI, Anthropic, or Local models (Qwen/Llama) with a single config flag.

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
uvx ty check              # Type check
```

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.
