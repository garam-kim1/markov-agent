---
name: adk-grounding
description: Expert in grounding ADK agents with external data (Google Search, Vertex AI Search, RAG) in Python. Use for implementing verifiable responses, real-time web access, and private knowledge retrieval.
---

# ADK Grounding Specialist

## Philosophy & Architecture
Grounding connects agents to external sources to reduce hallucinations and provide citations.

## Grounding Patterns
1. **Google Search**:
   - Access real-time web facts, news, and weather.
   - Built-in tool integration in `LlmAgent`.
2. **Vertex AI Search**:
   - Query indexed private documents and enterprise datastores.
3. **Agentic RAG**:
   - Dynamic query construction and metadata filtering (e.g., Vector Search 2.0).

## Implementation
- Configure grounding in the `LlmAgent` tools or model config.
- Ensure the agent is instructed to use grounding for specific types of queries.
- Read `references/grounding.md` for tool configuration details.

## Success Criteria
- Valid citation handling in agent responses.
- Correct configuration of Vertex AI Search datastores.
- High grounding accuracy as measured by `hallucinations_v1` metrics.
