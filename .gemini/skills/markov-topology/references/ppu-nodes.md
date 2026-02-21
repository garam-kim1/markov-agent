# Probabilistic Processing Units (PPU)

## `ProbabilisticNode`
The core LLM-driven node in `markov-agent`. It determine the next state by sampling from a Gemini or LiteLLM model.

## Configuration
- `adk_config`: `ADKConfig` instance.
- `prompt_template`: Jinja2 template.
- `samples`: Number of parallel samples ($k$).
- `sampling_strategy`: `SamplingStrategy.UNIFORM` (default).

## Example
```python
from markov_agent.engine.ppu import ProbabilisticNode
from markov_agent.engine.adk_wrapper import ADKConfig

node = ProbabilisticNode.from_prompt(
    name="research-node",
    prompt="Research {{ input_text }} and return JSON.",
    adk_config=ADKConfig(model_name="gemini-1.5-flash"),
    samples=3
)
```

## Functional Nodes
For deterministic logic:
```python
from markov_agent.topology.node import FunctionalNode

def process_logic(state: MyState) -> MyState:
    state.output_text = state.input_text.upper()
    return state

node = FunctionalNode(name="upper-node", func=process_logic)
```
