# ADK Deployment Spec (Python)

## 1. Vertex AI Agent Engine
Fully managed, auto-scaling runtime for ADK agents.

### Prerequisites
- Google Cloud Project with Billing enabled.
- APIs: `Vertex AI API`, `Cloud Storage API`.
- Service Account with `Vertex AI User` role.

### Deployment CLI
Deploys the current directory as an agent service.
```bash
uv run adk deploy --project_id=my-gcp-project --location=us-central1
```

### Configuration
Ensure `agent.py` exposes a `root_agent` variable.
```python
# agent.py
from google.adk import LlmAgent

root_agent = LlmAgent(...)
```

## 2. Cloud Run (Containerized)
Serverless container deployment. Best for custom runtime requirements.

### Command
Builds a Docker image and deploys to Cloud Run.
```bash
uv run adk deploy cloud_run \
    --project_id=my-gcp-project \
    --region=us-central1 \
    --service_name=my-agent-service
```

### Artifact Registry
Images are stored in Google Artifact Registry. Ensure `Artifact Registry API` is enabled.

## 3. Deployment Checklist
1. **Dependencies**: `requirements.txt` or `pyproject.toml` must exist.
2. **Secrets**: Use Secret Manager for API keys (`GEMINI_API_KEY`).
3. **Identity**: Assign a Service Account to the Cloud Run service for GCP resource access.
