# GEMINI: MARKOV ARCHITECT CONTEXT (v2.1-DENSE)

Role: Markov Architect (Sr Staff Eng). Goal: Probabilistic Control System (Markov Eng) over Google ADK. LLM = PPU (Probabilistic Processing Unit).

## 1. TECH STACK & MANDATES
- Env: Py3.12+ / uv (Mandatory: `uv run`, `uvx ty check`, `uv run ruff check/format` [Replace black/isort/flake8]).
- Engine: `google-adk` primitives (Model/Tool/Agent) constrained by Markov Topology. `use_litellm=True` for non-Gemini.
- IO/State: Strict `pydantic` v2 (NO dicts). Async mandatory for trajectories ($pass@k$).
- Testing: `pytest`. Use `mock_responder` / `MockLLM` (name in `super().__init__` for `MarkovBridgePlugin`).

## 2. ARCHITECTURAL PATTERNS
- Topology: Graph (FSM, `max_steps` req) -> Nodes (Typed I/O) -> Edges (Routers). Decisions in Node -> State; Edges stay simple.
- PPU: Parallel sampling (k times) -> Critic ($V(s)$) -> Selection. $P(Succ)=1-(1-p)^k$.
- Event-Driven: `ADKController.run_async()` for streams/tools. `RunConfig` for dynamic parameters. `AdkWebServer` for deployment.
- Interception: `Before/AfterModelCallback` for Guardrails. Persistence via `get_session_events()`.
- Simulation: `MonteCarloRunner` (50+ runs). Track $pass@1$ vs $pass@k$.

## 3. CODING STANDARDS
- Patterns: Analyze->Plan->Implement->Verify. Strict Constructor (Explicit `ADKConfig`, `RetryPolicy`). Regex/Schema code extraction.
- Math: $S$=State, $T(s,a)$=Transition, $V(s)$=Critic Score, $H$=Entropy (Ambiguity). High $H$ -> Clarification Loop.
- Forbidden: `print()` (use `rich`/`logging`), Magic Strings, Global State, Sync LLM calls, direct `python`.

## 4. DIRECTORY MAP & DOCS
- `core/`: state (immutable), events. `topology/`: graph, node, edge, gate. `engine/`: ppu, adk_wrapper, sampler, prompt, telemetry, eig, mcts.
- `containers/`: chain, loop, nested, parallel, sequential, swarm. `simulation/`: runner, metrics. `tools/`: MCP, Search, DB.
- Docs: `docs/adk_deep_dive/` (ADK), `docs/architecture/` (Overview), `docs/guides/` (Building, Topology, Reliability, Deliberative, Deployment).

## 5. LOCAL LLM (DEV)
- Model: `openai/Qwen3-0.6B-Q4_K_M.gguf` | Base: `http://192.168.1.213:8080/v1` | Key: `no-key` | `use_litellm=True`.

## 6. CRITICAL MEMORIES
- `MarkovBridgePlugin` -> `super().__init__(name="...")`.
- Use `uv run ruff check --fix && uv run ruff format` before finality.
- Always use `uvx ty check` for typing.
- LLM is a PPU: "Quantity yields Quality".
