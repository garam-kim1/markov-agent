# MARKOV-AGENT TECH CONTEXT

## ESSENCE
PPU-based FSM wrapper for `google-adk`. Graph(Deterministic) + Nodes(Stochastic/LLM). Goal: Prompt Eng -> Reliability Eng.

## STACK
* **Env:** `uv` (NO pip/poetry). Py 3.12+.
* **Lint:** `ruff` (Strict), `ty` (`uvx ty` -> pyright).
* **Core:** `google-adk`, `pydantic` v2, `asyncio`, `tenacity`, `jinja2`, `litellm`, `rich`.
* **Verify:** `uvx ty check && uv run ruff check --fix && uv run ruff format && uv run pytest`.

## ARCHITECTURE (TOPOLOGY)
* **S (State):** `BaseState` (immut history/meta). Use `AppendList`/`AppendString`.
* **G (Graph):** `Graph`. FSM runner, Mermaid exp.
* **T (Node):** `BaseNode` -> `ProbabilisticNode` (PPU), `RouterNode` (Semantic), `FunctionalNode`.
* **E (Edge):** `Edge` (Router logic). `Switch` (Fluent logic `>>`).

## DIR MAP (`src/markov_agent/`)
* **`core/`**: Fndtn.
    * `state.py`: `BaseState` (immut), `AppendList/String`.
    * `events.py`: `EventBus` (Async telemet).
    * `probability.py`: Log-space math, Shannon entropy.
    * `registry.py`: Plugin discov.
    * `services.py`: Infra (Session/Mem/Artifact).
    * `logging.py`: Base obs.
* **`topology/`**: Graph struct.
    * `graph.py`: Core FSM eng, trans, Mermaid.
    * `node.py`: `BaseNode`, `FunctionalNode`.
    * `router.py`: `RouterNode` (LLM-based routing).
    * `edge.py`: `Edge` logic, `Switch`.
    * `gate.py`: Logic gates (AND/OR).
    * `analysis.py`: Topo optimization.
* **`engine/`**: Exec/PPU.
    * `ppu.py`: `ProbabilisticNode` (LLM).
    * `adk_wrapper.py`: `ADKConfig`, `ADKController` (Gemini/LiteLLM).
    * `sampler.py`: Parallel $k$-sampling (Vote, Best-N).
    * `prompt.py`: `PromptEngine` (Jinja2).
    * `runtime.py`: Exec env, session lifecycle.
    * `agent.py`: High-lvl `Agent`.
    * `selectors.py`: Result select logic.
    * `trajectory.py`: Exec path tracking.
    * `observability.py`: Plugin sys.
* **`containers/`**: Patterns.
    * `chain`, `loop`, `parallel`, `swarm`, `nested`.
* **`governance/`**: Constraints.
    * `cost.py`: Budgets.
    * `resource.py`: Token/Rate limits.
* **`tools/`**: Ext.
    * `mcp.py`: MCP integ.
    * `search.py`, `database.py`: Std tools.
* **`simulation/`**: Reliability.
    * `dashboard.py`: CLI dash (`rich`).
    * `runner.py`: Monte Carlo verif.
    * `metrics.py`: Reliab/Acc metrics.
    * `twin.py`: Digital Twin.

## RULES
* **Init:** `ADKConfig` w/ `RetryPolicy`. Explicit cfg rec (Prod).
    * `ADKConfig(model_name="gemini-3-flash-preview", api_key=...)`
* **Imp:** `markov_agent` top-level only.
* **Debug:** `graph.run_with_dashboard(state)`.
* **PPU:** `execute_parallel_sampling` > single.
* **Async:** ALL operations. `await` req.
* **Events:** `event_bus` req. `rich` for obs.
* **Type:** No `Any`. `TypeVar(bound=BaseState)`.
* **Tst:** `MockLLM` only. NO live API in CI.
* **Forbid:** `print()`, Magic Strings, Global Mutable, Sync LLM, `pip`.

## MATH
* $P(Succ) = 1 - (1 - p)^k$
* $H = -\sum p_i \log_2 p_i$ (Entropy).
* $D_{KL}(P || Q)$ (Drift).
