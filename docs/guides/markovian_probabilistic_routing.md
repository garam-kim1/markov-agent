# Markovian Probabilistic Routing

In the `markov-agent` library, we treat the agent's execution path as a **Discrete-Time Markov Chain (DTMC)**. This allows us to move beyond simple "if/else" logic and into the realm of probabilistic behavior management.

## The Core Equation

The probability of the agent being in state $j$ at time $t+1$, given it was in state $i$ at time $t$, is defined by the transition probability $P_{ij}$:

$$P(S_{t+1} = j \mid S_t = i) = P_{ij}$$

In our implementation, this transition can be:
1.  **Deterministic:** $P_{ij} \in \{0, 1\}$.
2.  **Conditional:** $P_{ij}$ depends on a boolean function of the state.
3.  **Probabilistic:** $P_{ij}$ is determined by a distribution returned by an LLM or a scoring function.

## Probabilistic Nodes and Edges

### 1. Probabilistic Node (PPU)
A `ProbabilisticNode` uses parallel sampling ($k > 1$) to generate multiple possible outcomes. This creates a distribution of potential next states within the node itself.

### 2. Probabilistic Edge
An `Edge` can return a `TransitionDistribution` (a dictionary of `node_name: probability`). 

```python
# Example of a probabilistic router
def router(state: MyState):
    if state.confidence > 0.9:
        return "deploy"
    return {"refine": 0.8, "fail": 0.2}

g.add_transition("reviewer", router)
```

## Entropy and Uncertainty

We measure the uncertainty of a transition using **Shannon Entropy ($H$)**:

$$H(S) = -\sum_{i=1}^{n} p_i \log_2 p_i$$

*   **Low Entropy:** The agent is "certain" of its next move.
*   **High Entropy:** The agent is confused or the routing logic is ambiguous. 

The `Graph` execution engine automatically logs high-entropy events (typically $H > 1.5$), signaling that the topology may need refinement or that the prompt is producing inconsistent results.

## Stationary Distribution and Analysis

By analyzing the **Transition Matrix ($P$)** of your graph, you can predict its long-term behavior.

### Using TopologyAnalyzer

```python
analyzer = g.analyze()
matrix = analyzer.extract_matrix()

# Find where the agent spends most of its time
stationary = analyzer.calculate_stationary_distribution(matrix)

# Identify terminal/absorbing states
terminal_nodes = analyzer.detect_absorbing_states(matrix)
```

The stationary distribution $\pi$ satisfies $\pi P = \pi$. If a node has a high value in the stationary distribution, it means your agent might be getting "stuck" there (e.g., an infinite loop in a self-correction cycle).

## Ergodicity and Mixing Time

*   **Ergodicity:** Is it possible to reach every state from every other state? If your graph isn't ergodic, you might have "dead ends".
*   **Mixing Time:** How many steps does it take for the agent to reach a "stable" behavior?

These metrics help you optimize your topology for speed and reliability, ensuring that the agent converges on a solution as efficiently as possible.
