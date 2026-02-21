# ADK Integrations Catalog (Python)

## Google Cloud Tools
- `google-search`: Real-time web facts.
- `bigquery`: Data analytics and logging.
- `vertex-ai-search`: Enterprise document retrieval.
- `pubsub`: Event-driven messaging.
- `spanner`: Distributed database.

## 3rd Party Connectors
- `github` / `gitlab`: SCM integration.
- `slack` / `mailgun`: Messaging.
- `stripe` / `paypal`: Financial transactions.
- `atlassian` / `asana` / `linear`: Project management.

## Observability & Ops
- `agentops` / `phoenix` / `weave`: Advanced tracing.
- `agent-engine`: Vertex AI managed runtime.

## Implementation Pattern
Integrate via the `tools` or `plugins` list in `LlmAgent` or `Runner`.
```python
from google.adk.integrations import google_search
agent = Agent(tools=[google_search.search])
```
