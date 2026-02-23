# MARKOV-AGENT TECHNICAL CONTEXT (LLM-ONLY)

## ESSENCE
PPU-based FSM wrapper for `google-adk`. Paradigm: Deterministic Topology (Graph) + Stochastic Transitions (LLM Nodes). Focus: Shifting Prompt Eng to Markov/Reliability Engineering.

## TECH STACK
* **Runtime:** Python 3.12+ (Strict Type Hints).
* **Package/Env:** `uv` (FORBIDDEN: `pip`, `poetry`).
* **Lint/Format:** `ruff` (Strict compliance).
* **Static Analysis:** `ty` (pyright via `uvx ty`).
* **Test:** `pytest` (Async-heavy).
* **Core:** `google-adk`, `pydantic` v2, `asyncio`, `tenacity`, `jinja2`, `litellm`, `numpy`, `pandas`, `sqlalchemy`, `mcp`, `rich`.
* **Verify Cmd:** `uvx ty check && uv run ruff check --fix && uv run ruff format && uv run pytest`.

## ARCHITECTURE (TOPOLOGY)
* **S (State):** `BaseState`. Pydantic model with immutable `history` and `meta` (confidence/entropy).
* **G (Graph):** `Graph`. ADK-compatible FSM runner. Handles node transitions and Mermaid exports.
* **T (Node):** `BaseNode` -> `ProbabilisticNode` (PPU), `RouterNode` (Semantic Router), or `FunctionalNode`. $S_t \to S_{t+1}$.
* **E (Edge):** `Edge`. Router logic $f(S) \to (next\_id, p)$. Supports condition-based or probabilistic routing.

## DIRECTORY MAP (`src/markov_agent/`)
* **`core/`**: Foundation
    * `state.py`: `BaseState` with history tracking & Markov view.
    * `events.py`: `EventBus` for async observability/telemetry.
    * `probability.py`: Log-space math, Shannon entropy, distributions.
    * `registry.py`: Component/plugin registration system.
    * `services.py`: Shared service management (Session, Memory, Artifact).
    * `logging.py`, `monitoring.py`: Observability foundation.
* **`topology/`**: FSM Structure
    * `graph.py`: Main execution engine & Mermaid generator.
    * `node.py`: Base abstractions for all graph nodes.
    * `router.py`: Semantic Router Node (LLM-based routing).
    * `edge.py`: Routing logic & transition probability.
    * `gate.py`: Conditional branching & multiplexing gates.
    * `analysis.py`, `evolution.py`: Graph analysis and self-evolving topologies.
* **`engine/`**: PPU & ADK Bridge
    * `ppu.py`: `ProbabilisticNode` for LLM-driven transitions.
    * `adk_wrapper.py`: `ADKConfig`, `ADKController` for GenAI integration. Supports LiteLLM.
    * `sampler.py`: Parallel $k$-sampling strategies (Uniform, etc.).
    * `prompt.py`: Jinja2-based `PromptEngine`.
    * `runtime.py`, `agent.py`: ADK runtime/session and agent abstractions.
    * `selectors.py`, `trajectory.py`: Result selection and path tracking.
    * `observability.py`, `plugins.py`, `callbacks.py`: Extension system.
* **`containers/`**: FSM Patterns
    * `chain.py`, `loop.py`, `parallel.py`, `self_correction.py`, `swarm.py`, `nested.py`, `sequential.py`.
* **`governance/`**: Guardrails & Constraints
    * `cost.py`: Budget tracking and spending limits.
    * `resource.py`: Token usage and rate limiting.
* **`tools/`**: External Integration
    * `mcp.py`: Model Context Protocol support.
    * `search.py`, `database.py`, `agent_tool.py`: Standard tool implementations.
* **`simulation/`**: Reliability Engineering
    * `dashboard.py`: Interactive CLI dashboard for real-time observation.
    * `runner.py`: Monte Carlo simulation for topology verification.
    * `metrics.py`, `analysis.py`: Reliability, accuracy, and latency tracking.
    * `scenario.py`, `twin.py`: Digital twin and evaluation scenarios.

## CORE EXPORTS
* **Topology:** `Graph`, `ProbabilisticNode`, `RouterNode`, `TopologyOptimizer`.
* **Runtime:** `ADKConfig`, `ADKController`, `Agent`, `AdkWebServer`.
* **State:** `BaseState`.
* **Simulation:** `DigitalTwin`, `WorldModel`.

## CODING RULES
* **Initialization:** Explicit `ADKConfig` and `RetryPolicy` preferred.
    * **Note:** `ProbabilisticNode` defaults to `gemini-3-flash-preview` if no config/model is provided, but explicit configuration is recommended for production.
    * **Gemini Example:**
      ```python
      ADKConfig(model_name="gemini-3-flash-preview", api_key=os.environ.get("GEMINI_API_KEY"))
      ```
    * **Local/LiteLLM Example:**
      ```python
      ADKConfig(
          model_name="openai/Qwen3-0.6B-Q4_K_M.gguf",
          api_base="http://192.168.1.213:8080/v1",
          api_key="no-key",
          use_litellm=True,
      )
      ```
* **Imports:** Use `markov_agent` top-level imports (`Graph`, `BaseState`, `ADKConfig`).
* **Dashboard:** Use `graph.run_with_dashboard(state)` for interactive debugging.
* **PPU Logic:** Favor parallel sampling (`execute_parallel_sampling`) for reliability.
* **Async:** All LLM and Graph operations MUST be `async`.
* **Events:** Emit via `event_bus`. Use `rich` for CLI observability.
* **Typing:** Strict `typing.Any` avoidance where possible. Use `TypeVar(bound=BaseState)`.
* **Testing:** Use `MockLLM` for unit tests. No live API calls in CI.
* **Forbidden:**
    1. `print()` -> Use `rich.console` or `logging`.
    2. Magic Strings -> Use constants or templates.
    3. Global Mutable State -> State must live in `BaseState` or `InvocationContext`.
    4. Sync LLM calls -> Use `await` and `asyncio`.
    5. Direct `python/pip` -> Use `uv run` and `uv add`.

## RELIABILITY MATH
* $P(Success) = 1 - (1 - p)^k$ where $k$ is sample count.
* $H = -\sum p_i \log_2 p_i$ (Entropy). High $H \implies$ Graph needs refinement or clarification.
* $D_{KL}(P || Q) = \sum p_i \log(p_i / q_i)$ (KL Divergence). Measures drift from reference policy.
* $JSD(P || Q) = \frac{1}{2} D_{KL}(P || M) + \frac{1}{2} D_{KL}(Q || M)$ where $M = \frac{1}{2}(P+Q)$. Symmetric metric.
