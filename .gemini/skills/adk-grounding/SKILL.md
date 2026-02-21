---
name: adk-grounding
description: Expert in grounding ADK agents with external data (Google Search, Vertex AI Search, RAG) in Python. Use for implementing verifiable responses, real-time web access, and private knowledge retrieval.
---

# ADK Grounding Specialist (Python Edition)

## Philosophy & Architecture
Grounding connects agents to external sources to reduce hallucinations and provide citations.

## Grounding Patterns
1. **Google Search (Real-Time)**:
   - Use `GoogleSearchTool` for news, weather, and up-to-the-minute facts.
2. **Vertex AI Search (Enterprise)**:
   - Use `VertexAiSearchTool` to query private PDFs, docs, and intranets.
   - Requires a configured Data Store in Google Cloud.
3. **Agentic RAG**:
   - Dynamic query construction and metadata filtering (e.g., Vector Search 2.0).

## Implementation
- Configure grounding via the `LlmAgent` tools list.
- Ensure the agent is instructed to use grounding for specific types of queries.
- Read `references/grounding.md` for tool configuration details.

## Success Criteria
- Valid citation metadata in agent responses.
- Correct configuration of Vertex AI Search `data_store_id`.
- High grounding accuracy as measured by `hallucinations_v1` metrics.
