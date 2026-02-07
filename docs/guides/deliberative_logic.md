# Advanced Deliberative Logic: "System 2" Reasoning

High-stakes decision-making in financial or strategic domains requires **System 2** behavior—slow, deliberative reasoning that moves beyond reflexive model responses.

## 1. Inference-Time Search (MCTS)

Utilizing **Monte Carlo Tree Search**, the agent follows an Expand-Simulate-Select process. It generates multiple plans, simulates outcomes via a critic model, and selects the path that maximizes the **Value Function**:

$$V(s) = \mathbb{E} \left[ \sum_{t=0}^{T} \gamma^t r(s_t, a_t) \right]$$


Here, the discount factor $\gamma$ is critical, as it allows the system to prioritize immediate task success over speculative future states.

## 2. Test-Time Compute

This involves trading processing speed for higher accuracy. The Markov Engine allows you to allocate additional compute cycles for:
*   **Internal Monologue:** Thinking steps before final output.
*   **Multi-step Verification:** Self-correction loops before state commitment.
*   **Parallel Trajectories:** Generating $k$ paths simultaneously.

## 3. Bayesian Information Gain

To manage uncertainty, agents utilize **Expected Information Gain (EIG)** to determine if a clarifying question is mathematically necessary. 

**Formula:**
$$EIG(q, d) = H(Y) - H(Y|q, a)$$

This prevents redundant interactions by ensuring the agent only asks questions that significantly reduce entropy ($H$). If $EIG$ is below a threshold, the agent proceeds with the best available transition.
