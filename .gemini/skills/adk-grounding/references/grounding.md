# ADK Grounding Spec (Python)

## 1. Google Search Grounding (Real-Time)
Enable agents to search the live web for facts, news, and weather.

### Configuration
Use the `GoogleSearchTool` included in ADK.
```python
from google.adk import LlmAgent
from google.adk.tools import GoogleSearchTool

agent = LlmAgent(
    model="gemini-2.0-flash-exp",
    tools=[GoogleSearchTool()],
    instruction="Always use Google Search for events after 2023."
)
```

### Citations
The model will return `grounding_metadata` in the response. ADK automatically parses this, but you can inspect it in the `Event` object.

## 2. Vertex AI Search (Enterprise Data)
Ground responses in private company documents (PDFs, Intranet, Confluence) indexed in Vertex AI.

### Prerequisite
- Create a Data Store in Google Cloud Console.
- Copy the `DATA_STORE_ID`.

### Configuration
```python
from google.adk.tools import VertexAiSearchTool

enterprise_tool = VertexAiSearchTool(
    project_id="my-project",
    location="global",
    data_store_id="my-datastore-id"
)

agent = LlmAgent(
    model="gemini-1.5-pro",
    tools=[enterprise_tool],
    instruction="Answer questions using the internal policy documents."
)
```

## 3. RAG Engine & Vector Search
For granular control over retrieval:
- **Vertex AI RAG Engine**: Managed RAG pipeline.
- **Vector Search 2.0**: High-scale similarity search with hybrid keyword support.

### Implementation Tip
For complex RAG, wrap the retrieval logic in a custom `BaseTool` or `FunctionalNode` to pre-process queries before hitting the vector database.
