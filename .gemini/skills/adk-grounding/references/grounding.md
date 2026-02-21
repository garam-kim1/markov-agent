# ADK Grounding Spec (Python)

## Grounding Patterns
1. **Google Search**: Real-time web info (news, weather).
2. **Vertex AI Search**: Private enterprise docs/datastores.
3. **Agentic RAG**: Dynamic query construction (Vector Search 2.0).

## Core Mechanisms
- **Source Attribution**: Citations formatted as URLs or document references.
- **Verification**: `hallucinations_v1` metric for groundedness.

## Implementation Pattern
```python
# LlmAgent with Google Search tool
agent = LlmAgent(
    model="gemini-2.0-flash",
    name="search_agent",
    tools=[google_search_tool],
    instruction="Use Google Search for real-time news."
)
```

## Advanced Patterns
- **User Simulation**: Test grounding via AI-driven mock users (`user-sim.md`).
- **Deep Search**: Two-phase Research -> Composition workflow.
- **RAG Engine**: Upload/Search documents via Vertex AI RAG Engine.
