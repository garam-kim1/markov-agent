---
name: adk-deploy
description: Expert in ADK agent deployment to Vertex AI and Cloud Run. Use for productionizing agents, configuring managed infrastructure, and CLI deployment.
---

# ADK Deployment Architect

## Philosophy
Deployment moves your agent from local development to a scalable environment. ADK prioritizes managed services.

## Logic Flow
1. **Target Identification**:
   - For managed auto-scaling -> **Vertex AI Agent Engine**.
   - For custom containers -> **Cloud Run**.
   - For high-control K8s -> **GKE**.
2. **Context Loading**:
   - Load `references/deploy.md` for CLI commands and architecture diagrams.
3. **Execution**:
   - Run `adk deploy` (CLI).
   - Configure `Runner` for persistent `SessionService`.

## Output Standards
- Valid `requirements.txt` or `pyproject.toml` generation.
- Correct Service Account configuration.
- Successful container build commands.
