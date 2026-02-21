# ADK Deployment Spec (Python)

## Vertex AI Agent Engine (Recommended)
Managed auto-scaling for ADK. No API server logic required.
- **Commands**: `adk deploy` (CLI).
- **Environment**: Runtime provides ADK libraries. Declare custom deps in `requirements.txt`.
- **Identity**: Requires Service Account with Vertex AI permissions.

## Cloud Run (Custom/Container)
- **Scaffold**: FastAPI entry point + Dockerfile.
- **Scale**: Auto-scales to zero.
- **Deployment**: `gcloud run deploy`.

## GKE (High-Control)
- **Pattern**: Sidecar architecture with Open Models (vLLM/Ollama).
- **Orchestration**: Managed Kubernetes (GKE).

## Checklist
1. Declare dependencies in `pyproject.toml` or `requirements.txt`.
2. Configure `Runner` with persistent `SessionService` (SQL/VertexAi).
3. Set `GOOGLE_APPLICATION_CREDENTIALS` for local testing against cloud.
