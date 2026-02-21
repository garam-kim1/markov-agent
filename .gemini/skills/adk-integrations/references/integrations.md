# ADK Integrations Catalog (Python)

## 1. Google Cloud Integrations
Native support via `google-genai` and ADK tools.
- **BigQuery**: Use `BigQueryTool` for SQL analytics.
- **Pub/Sub**: Trigger agents from event streams.

## 2. LangChain Interop
Use LangChain tools within ADK agents.

```python
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from google.adk.integrations.langchain import LangChainTool

# Wrap LangChain tool for ADK
wiki_tool = LangChainTool(
    tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
)

agent = LlmAgent(tools=[wiki_tool])
```

## 3. AgentOps Observability
Full-stack monitoring (cost, latency, session replays).

### Configuration
1. Set `AGENTOPS_API_KEY` in environment.
2. Initialize at start of script.

```python
import agentops
agentops.init() # Auto-instruments ADK

# ... run your agent ...

agentops.end_session("Success")
```

## 4. Common Third-Party APIs
- **GitHub**: Use `PyGithub` wrapped in `BaseTool`.
- **Slack**: Use `slack_sdk` for chatops.
- **Linear/Jira**: Use standard REST APIs wrapped in ADK tools.
