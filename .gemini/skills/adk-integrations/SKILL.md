---
name: adk-integrations
description: Expert in ADK pre-built tools and integrations. Use for connecting agents to Google Search, BigQuery, GitHub, Jira, and third-party observability libraries.
---

# ADK Integration Specialist (Python Edition)

## Philosophy & Architecture
Integrations are pre-packaged tools, plugins, and services that extend ADK agent capabilities.

## Key Categories
1. **Google Cloud**:
   - `BigQueryTool` for analytics.
   - `PubSubTool` for event-driven architectures.
2. **LangChain**:
   - Wrap existing LangChain tools using `LangChainTool` for ADK compatibility.
3. **Observability**:
   - **AgentOps**: Auto-instrumentation via `agentops.init()`.
   - **Phoenix/Weave**: Advanced tracing integration.
4. **Third-Party APIs**:
   - GitHub, Slack, Linear, Asana (via REST wrappers).

## Implementation
- Reference pre-built tool functions in the `LlmAgent` tools list.
- Read `references/integrations.md` for the full catalog and configuration patterns.

## Success Criteria
- Valid configuration of `LangChainTool` wrappers.
- Successful authentication to third-party services.
- Correct initialization of `agentops` for monitoring.
