# ADK Observability & Logging Spec (Python)

## 1. Logging Strategy
ADK uses Python's standard `logging` module.

### Configuration
- **CLI**: `adk run agent.py --log_level=DEBUG`
- **Programmatic**:
  ```python
  import logging
  logging.basicConfig(level=logging.DEBUG)
  ```

### Structured Logging (Cloud)
To emit JSON logs compatible with Google Cloud Logging:
```python
from google.cloud.logging.handlers import CloudLoggingHandler
import google.cloud.logging

client = google.cloud.logging.Client()
handler = CloudLoggingHandler(client)
logging.getLogger().addHandler(handler)
```

## 2. Plugins for Debugging

### `LoggingPlugin`
Prints formatted execution details to the console.
```python
from google.adk.plugins import LoggingPlugin

runner = Runner(
    agent=my_agent,
    plugins=[LoggingPlugin()]
)
```

### `DebugLoggingPlugin`
Captures exhaustive interaction data (LLM prompts, tool inputs, session state) to a YAML file.
```python
from google.adk.plugins import DebugLoggingPlugin

runner = Runner(
    agent=my_agent,
    plugins=[DebugLoggingPlugin(output_path="debug_trace.yaml")]
)
```

## 3. Tracing (OpenTelemetry)
ADK supports OpenTelemetry (OTEL) for distributed tracing.

### Monocle Integration
Monocle provides automatic instrumentation for ADK agents.
```bash
pip install monocle3
```
```python
from monocle3 import Monocle
Monocle.instrument()
```

### Trace View (Local)
Use the ADK Dev UI to visualize traces locally.
```bash
adk web .
```
Navigate to the "Trace" tab to see:
- **Timeline**: Agent execution flow.
- **Spans**: Duration of tools and LLM calls.
- **Payloads**: Inspect full request/response bodies.

## 4. BigQuery Analytics
For production, export logs to BigQuery for SQL-based analysis of:
- Token usage trends.
- Latency distribution.
- Tool error rates.
