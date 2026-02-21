# Custom Agents (Python)

For complex logic not covered by standard workflows, inherit from `BaseAgent`.

## Core Concept
Override `_run_async_impl` to define custom orchestration. You must yield `Event` objects.

## Template

```python
from typing import AsyncGenerator
from google.adk.agents import BaseAgent
from google.adk.models import Event
from google.adk.types import InvocationContext

class MyCustomAgent(BaseAgent):
    def __init__(self, name: str, sub_agent: BaseAgent):
        # Register sub-agents for visibility
        super().__init__(name=name, agents=[sub_agent])
        self.sub_agent = sub_agent

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # 1. Pre-processing
        # Access state: ctx.session.state
        input_data = ctx.session.state.get("input")

        if not input_data:
             yield Event(source=self.name, type="error", data="No input")
             return

        # 2. Run sub-agent
        # Delegate execution and stream events up
        async for event in self.sub_agent.run_async(ctx):
            yield event

        # 3. Post-processing
        result = ctx.session.state.get("output")
        if result == "retry":
             # Conditional logic: Run again if needed
             async for event in self.sub_agent.run_async(ctx):
                 yield event
```

## Critical Rules
1. **Async Generator**: Must use `async for` and `yield`.
2. **State Access**: Use `ctx.session.state` (dict-like) for data sharing.
3. **Event Propagation**: Always yield events from sub-agents unless you specifically want to suppress them.
4. **Registration**: Pass all sub-agents to `super().__init__(agents=[...])` so the framework can manage their lifecycle.
