# Reliability Engineering with Markov Agent

> "If you didn't test it 50 times, it doesn't work."

In traditional software, `assert 2 + 2 == 4` is binary. In AI engineering, `assert llm.add(2, 2) == 4` is probabilistic. It might be true 99% of the time, or 60% of the time depending on the model, prompt, and phase of the moon.

**Markov Engineering** is the discipline of quantifying and managing this uncertainty.

## The Simulation Workbench

The `markov_agent.simulation` module provides tools to treat your Agent like a stochastic system that needs statistical verification.

### 1. The Monte Carlo Runner

The `MonteCarloRunner` executes your graph against a dataset $N$ times per case.

```python
from markov_agent.simulation.runner import MonteCarloRunner

runner = MonteCarloRunner(
    graph=my_agent,
    dataset=[test_case_1, test_case_2, ...],
    n_runs=20,  # Run each case 20 times
    success_criteria=lambda state: state.result == "expected_value"
)

results = await runner.run_simulation()
```

### 2. Success Criteria

You must define a programmatic way to verify if a run was successful.
*   **Exact Match:** `state.answer == "42"`
*   **Fuzzy Match:** `"error" not in state.response.lower()`
*   **Unit Tests:** For coding agents, compiling and running the generated code.

## Interpreting Metrics

We measure three key dimensions of reliability:

### Accuracy (Pass@1)
*   **Definition:** The raw probability that a single run is correct.
*   **Formula:** $\frac{\text{Total Successes}}{\text{Total Runs}}$
*   **Goal:** High accuracy is good, but often expensive to guarantee with just prompting.

### Reliability (Pass@k)
*   **Definition:** The probability that *at least one* result is correct given $k$ attempts.
*   **Formula:** $1 - (1 - p)^k$ (simplified).
*   **Insight:** If your agent has 50% accuracy (Pass@1), running it 5 times (Pass@5) gives you a theoretical 96.8% reliability. This is why we use `ParallelNode` and Voting!

### Consistency (Pass^k)
*   **Definition:** The probability that *all* $k$ runs are correct.
*   **Insight:** Important for user trust. If an agent works 9 times but hallucinates the 10th time, it has low consistency.

## The Engineering Cycle

1.  **Prototype:** Build your Graph using `containers`.
2.  **Baseline:** Run a Simulation with `n_runs=10`.
3.  **Analyze:**
    *   Low **Accuracy**? Improve your prompts or add context.
    *   Low **Consistency**? Reduce Temperature or add strict validation nodes.
    *   Good Reliability but Low Accuracy? Use **Parallel Execution** ($k > 1$) and a Critic/Voter node to pick the winner.
4.  **Refine:** Update the topology and repeat.
