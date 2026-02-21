---
name: adk-integrations
description: Expert in ADK pre-built tools and integrations. Use for connecting agents to Google Search, BigQuery, GitHub, Jira, and third-party observability libraries.
---

# ADK Integration Specialist

## Philosophy & Architecture
Integrations are pre-packaged tools, plugins, and services that extend ADK agent capabilities.

## Key Categories
1. **Google Cloud**:
   - Google Search, BigQuery, Vertex AI Search, Spanner, Pub/Sub.
2. **Third-Party Services**:
   - GitHub, GitLab, Jira, Slack, Notion, Stripe, PayPal, Linear, Asana.
3. **Observability & Ops**:
   - AgentOps, Phoenix, Weave, MLflow, Arize-AX, Cloud Trace.
4. **LLM Infrastructure**:
   - Hugging Face, MCP Toolbox for Databases.

## Implementation
- Reference pre-built tool functions in the `LlmAgent` tools list.
- Read `references/integrations.md` for the full catalog of available integrations.

## Success Criteria
- Valid configuration of pre-built integrations.
- Successful authentication to third-party services.
- Correct integration of observability libraries in the `Runner`.
