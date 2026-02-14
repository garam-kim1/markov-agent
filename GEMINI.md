# MARKOV-AGENT CONTEXT (LLM ONLY)

## IDENTITY
PPU-based FSM wrapper for `google-adk`. Paradigm: Deterministic Topology + Stochastic LLM nodes. Goal: Prompt Eng -> Markov/Reliability Eng.

## STACK
* **Lang:** Python 3.12+ (Strict Type Hints).
* **Pkg:** `uv` (NO pip/poetry).
* **Lint/Fmt:** `ruff`.
* **Types:** `ty` (pyright).
* **Test:** `pytest`.
* **Core:** `google-adk`, `pydantic` v2, `asyncio`, `tenacity`.
* **Verification:** `uvx ty check && uv run ruff check --fix && uv run ruff format && uv run pytest`.

## ARCHITECTURE (TOPOLOGY)
* **S (State):** `BaseState`. Immutable history.
* **G (Graph):** `Graph`. FSM runner.
* **T (Node):** `BaseNode` / `ProbabilisticNode` (PPU). $S_t \to S_{t+1}$.
* **E (Edge):** `Edge`. Router func $f(S) \to next\_id$.

## DIRECTORY MAP
* `src/markov_agent/`
  * `core/`: {state (BaseState), events (Bus), registry, probability}
  * `topology/`: {graph (Runner), node (T), edge (E), gate (Branch), analysis}
  * `engine/`: {ppu (PPU), adk_wrapper, runtime, sampler, prompt, nodes, selectors, mcts, eig}
  * `containers/`: {chain, loop, nested, parallel, swarm, sequential, self_correction}
  * `tools/`: {agent_tool, database, mcp, search}
  * `governance/`: {cost (Governor)}
  * `simulation/`: {runner (MonteCarlo), batch, metrics, analysis}

## CODING PATTERNS
* **ADKConfig:** Mandatory for PPU. `model_name="gemini-3-flash-preview"` preferred.
* **PPU Logic:** Parallel sampling ($k$ times) -> `execute_parallel_sampling`.
* **Strict Construction:** No defaults. Explicit `ADKConfig`, `RetryPolicy`.
* **Bridge Plugins:** `MarkovBridgePlugin` requires `name` in `super().__init__`.
* **Testing:** Use `MockLLM` for CI/unit tests to bypass API keys.
* **Events:** Emit via `event_bus`. Observability via `rich`.

## FORBIDDEN
1. `print()` (Use `rich` or logging).
2. Magic strings (Use templates/consts).
3. Global mutable state.
4. Sync LLM calls (Blocking).
5. Direct `python`/`pip` (Use `uv run`).

## MATH
* $S$: State Vector.
* $T(s, a) \to s'$ : Transition.
* $P(Success) = 1 - (1 - p)^k$.
* $H$: Entropy. High $H \to$ Clarification loop.
