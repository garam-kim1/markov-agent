# ADK Observability & Logging Spec (Python)

## Trace Logic
Enable measurement of an agent's internal reasoning, tool calls, and model outputs.

## Core Tools
1. **Dev UI (`adk web`)**: Trace tab for detailed interactive inspection.
   - **Event**: Raw JSON payload.
   - **Request/Response**: Direct Model I/O.
   - **Graph**: Visual logic flow.
2. **Logging Plugin**: Detailed JSON logs at each callback point.

## Advanced Integrations
- **BigQuery Agent Analytics**: Long-term log storage and SQL-based analysis.
- **Observability Libraries**: AgentOps, Phoenix, Weave, MLflow (via ADK Integrations).

## Pattern
```python
# In Runner config
runner = Runner(
    agent=root_agent,
    plugins=[LoggingPlugin()] # Built-in plugin
)
```

## Success Criteria
- Reasoning traces correctly captured.
- Token usage monitored via structured logs.
- Performance trends identified using BigQuery.
