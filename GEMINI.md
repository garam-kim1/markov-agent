# MARKOV-AGENT TECHNICAL CONTEXT (LLM-ONLY)

## ESSENCE
PPU-based FSM wrapper for `google-adk`. Paradigm: Deterministic Topology (Graph) + Stochastic Transitions (LLM Nodes). Focus: Shifting Prompt Eng to Markov/Reliability Engineering.

## TECH STACK
* **Runtime:** Python 3.12+ (Strict Type Hints).
* **Package/Env:** `uv` (FORBIDDEN: `pip`, `poetry`).
* **Lint/Format:** `ruff` (Strict compliance).
* **Static Analysis:** `ty` (pyright via `uvx ty`).
* **Test:** `pytest` (Async-heavy).
* **Core:** `google-adk`, `pydantic` v2, `asyncio`, `tenacity`, `jinja2`.
* **Verify Cmd:** `uvx ty check && uv run ruff check --fix && uv run ruff format && uv run pytest`.

## ARCHITECTURE (TOPOLOGY)
* **S (State):** `BaseState`. Pydantic model with immutable `history` and `meta` (confidence/entropy).
* **G (Graph):** `Graph`. ADK-compatible FSM runner. Handles node transitions and Mermaid exports.
* **T (Node):** `BaseNode` -> `ProbabilisticNode` (PPU) or `FunctionalNode`. $S_t \to S_{t+1}$.
* **E (Edge):** `Edge`. Router logic $f(S) \to (next\_id, p)$. Supports condition-based or probabilistic routing.

## DIRECTORY MAP (`src/markov_agent/`)
* **`core/`**: Foundation
    * `state.py`: `BaseState` with history tracking & Markov view.
    * `events.py`: `EventBus` for async observability/telemetry.
    * `probability.py`: Log-space math, Shannon entropy, distributions.
    * `registry.py`: Component/plugin registration system.
* **`topology/`**: FSM Structure
    * `graph.py`: Main execution engine & Mermaid generator.
    * `node.py`: Base abstractions for all graph nodes.
    * `edge.py`: Routing logic & transition probability.
    * `gate.py`: Conditional branching & multiplexing gates.
* **`engine/`**: PPU & ADK Bridge
    * `ppu.py`: `ProbabilisticNode` for LLM-driven transitions.
    * `adk_wrapper.py`: `ADKConfig`, `ADKController` for GenAI integration.
    * `sampler.py`: Parallel $k$-sampling strategies (Uniform, etc.).
    * `prompt.py`: Jinja2-based `PromptEngine`.
    * `runtime.py`: ADK runtime/session integration.
    * `selectors.py`: Result selection logic (Majority Vote, etc.).
* **`containers/`**: FSM Patterns
    * `chain.py`, `loop.py`, `parallel.py`, `self_correction.py`, `swarm.py`: Common topology patterns.
* **`tools/`**: External Integration
    * `mcp.py`, `search.py`, `database.py`: MCP and external tool wrappers.
* **`simulation/`**: Reliability Engineering
    * `runner.py`: Monte Carlo simulation for topology verification.
    * `metrics.py`: Reliability, accuracy, and latency tracking.

## CODING RULES
* **Initialization:** Explicit `ADKConfig` and `RetryPolicy`. No magic defaults.
    * **Gemini Example:**
      ```python
      ADKConfig(model_name="gemini-3-flash-preview", api_key=os.environ.get("GEMINI_API_KEY"))
      ```
    * **Local LLM Example:**
      ```python
      ADKConfig(
          model_name="openai/Qwen3-0.6B-Q4_K_M.gguf",
          api_base="http://192.168.1.213:8080/v1",
          api_key="no-key",
          use_litellm=True,
      )
      ```
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
