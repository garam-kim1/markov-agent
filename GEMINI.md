## 1. IDENTITY & DIRECTIVE
**Core:** Wrapper for `google-adk` treating LLMs as Stochastic PPUs (Probabilistic Processing Units) in deterministic Topology.
**Goal:** Move "Prompt Eng" -> "Markov Eng" (Reliable/Math/Control Sys).

## 2. STACK & CONSTRAINTS (STRICT)
* **Lang:** Python 3.12+ (Strict Type Hints).
* **Mgr:** `uv` (Astral). **NO** `pip`/`poetry`. **NO** direct `python`.
* **Run:** `uv run <script>.py`.
* **Lint:** `ruff`. **NO** `black`/`isort`/`flake8`.
    * `uv run ruff check`
    * `uv run ruff format`
* **Type:** `ty`.
    * `uvx ty check`
* **Test:** `pytest`.
    * `uv run pytest`
* **Workflow:** ALWAYS run verification after any code modification:
    1. `uvx ty check` (Type checking)
    2. `uv run ruff check` (Linting)
    3. `uv run pytest` (Tests)
* **Async:** `asyncio` required (I/O & parallel trajectories).
* **Validation:** `pydantic` v2.x (Models ONLY, NO unstructured dicts).
* **Engine:** `google-adk` (Model/Tool/Agent primitives).
    * *LiteLLM:* `ADKConfig(use_litellm=True)`.
    * *Keys:* `GOOGLE_API_KEY`, `GEMINI_API_KEY`.
    * *Preferred Model:* `gemini-3-flash-preview` for all Gemini-based examples and nodes.

## 3. ARCHITECTURE (TOPOLOGY)
**Paradigm:** FSM (Finite State Machine).
* **State ($S$):** `markov_agent.core.state.BaseState`. Immutable history. Single Source of Truth.
* **Graph:** `markov_agent.topology.graph.Graph`. Runner. Respects `max_steps`.
* **Node ($T$):** `markov_agent.topology.node.BaseNode`. Transform $S_t \to S_{t+1}$.
    * *PPU:* `ProbabilisticNode`. LLM wrapper.
* **Edge:** `markov_agent.topology.edge.Edge`. Router func: $f(S) \to next\_node\_id$.

## 4. PPU DESIGN (PROBABILISTIC NODE)
**Class:** `markov_agent.engine.ppu.ProbabilisticNode`
**Logic:**
1.  **Input:** Typed $S$.
2.  **Process:** Parallel sampling ($k$ times) via `execute_parallel_sampling`.
3.  **Output:** Validated JSON (ADK struct gen).
4.  **Fail:** `RetryPolicy` (loop), never crash.
**Math:** $P(Success) = 1 - (1 - p)^k$.

## 5. DIRECTORY MAP
`markov_agent/`
├── `pyproject.toml` (Managed by `uv`)
└── `src/markov_agent/`
    ├── `core/` {state.py (BaseState), events.py (Bus)}
    ├── `topology/` {graph.py, node.py, edge.py}
    ├── `engine/` {ppu.py, adk_wrapper.py, runtime.py, sampler.py, prompt.py}
    ├── `tools/` {agent_tool.py, database.py, mcp.py, search.py}
    ├── `containers/` {chain.py, loop.py, nested.py, parallel.py, swarm.py}
    └── `simulation/` {runner.py (MonteCarlo), metrics.py}

## 6. CODING PATTERNS
* **Strict Constructor:** NO defaults. Explicit `ADKConfig`.
    ```python
    node = ProbabilisticNode(
        name="reasoner",
        adk_config=ADKConfig(model_name="gemini-3-flash-preview", temperature=0.7),
        retry_policy=RetryPolicy(max_attempts=3),
        prompt_template="{query}"
    )
    ```
* **Local LLM:**
    ```python
    ADKConfig(
        model_name="openai/Qwen3-0.6B-Q4_K_M.gguf",
        api_base="[http://192.168.1.213:8080/v1](http://192.168.1.213:8080/v1)",
        use_litellm=True
    )
    ```
* **Observability:** Emit via `markov_agent.core.events.event_bus`.
* **Logging:** Use `rich` (console.log). **NO** `print()`.
* **Iteration:** Track `iteration_count` in State. Prevent infinite loops.

## 7. SIMULATION & RELIABILITY
* **Workflow:** Prompt -> Topology -> `MonteCarloRunner` (N=50) -> Deploy.
* **Metrics:** Track $pass@1$ (Acc) and $pass@k$ (Reliability).

## 8. FORBIDDEN
1.  `print()` debugging.
2.  Magic strings (Use templates/constants).
3.  Global mutable state (Pass `state` obj).
4.  Sync LLM calls (Blocker).
5.  `black`/`isort` (Use `ruff`).
6.  Direct `python` exec (Use `uv run`).

## 9. MATH VARS
* $S$: State Vector.
* $T(s, a)$: Transition/Logic.
* $V(s)$: Critic Score.
* $H$: Entropy (Ambiguity). *If High $H$: Trigger Clarification Loop.*
