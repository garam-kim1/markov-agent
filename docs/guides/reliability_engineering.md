# Reliability Engineering: pass@k and pass∧k

Industrial AI systems require a dual-metric approach to distinguish between a system's **capacity for reasoning** and its **operational consistency**.

## Metric A: pass@k (Solving for Accuracy)

To mitigate probabilistic decay in complex tasks, we implement **Parallel Verification**. This mechanism generates $k$ independent execution paths and uses a deterministic "Verifier" (unit test or critic) to select the successful outcome.

**Formula for Accuracy:**
$$P(\text{Accuracy}) = 1 - (1 - p)^k$$

By increasing $k$, we transform low-probability reasoning into high-reliability output. 
*   **Example:** If a model has a 20% success rate ($p=0.2$) on a difficult logic task, increasing attempts to $k=10$ raises the probability of at least one correct answer to **89%**.

## Metric B: pass∧k (Solving for Consistency)

Enterprise stability demands **Strict Consistency**, or $pass\wedge k$. This metric validates that a system can perform a task $k$ times with zero failures.

**Formula for Stability:**
$$P(\text{Stability}) = p^k$$

This metric exposes the "Risk Scenario" in batch processing. 
*   **Example:** If an agent is 99% accurate ($p=0.99$) but must process a batch of 100 tasks ($k=100$), the probability of completing the entire batch without a single error drops to **~36%**.

$pass\wedge k$ serves as the definitive **Gatekeeper Metric** for production deployment.

---

## The Simulation Workbench

The `markov_agent.simulation` module provides tools to treat your Agent like a stochastic system that needs statistical verification.

### 1. The MonteCarloRunner

The `MonteCarloRunner` executes your graph against a dataset $N$ times per case to calculate these metrics.

```python
from markov_agent.simulation.runner import MonteCarloRunner

runner = MonteCarloRunner(
    graph=my_agent,
    dataset=[test_case_1, test_case_2, ...],
    n_runs=50,  # "If you didn't test it 50 times, it doesn't work."
    success_criteria=lambda state: state.result == "expected_value"
)

results = await runner.run_simulation()
```

## The Engineering Cycle

1.  **Prototype:** Build your Graph using `containers`.
2.  **Baseline:** Run a Simulation with `n_runs=10`.
3.  **Analyze:**
    *   Low **Accuracy**? Improve your prompts or add context.
    *   Low **Consistency**? Reduce Temperature or add strict validation nodes.
    *   Good Reliability but Low Accuracy? Use **Parallel Execution** ($k > 1$) and a Critic/Voter node to pick the winner.
4.  **Refine:** Update the topology and repeat.
