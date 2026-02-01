# ADK Agent Deployment Implementation Guide (Python)

**Objective:** Implement and deploy a **Markov Agent** (using Google ADK) to one of three targets: **Cloud Run**, **Vertex AI Agent Engine**, or **Google Kubernetes Engine (GKE)**.

This guide assumes you have built a `markov_agent.topology.graph.Graph` agent and want to deploy it as a service.

**Prerequisites:**

* Python 3.10+
* Google Cloud Project with billing enabled.
* `gcloud` CLI installed and authenticated (`gcloud auth login`).
* `markov-agent` library installed (or available in your environment).

---

## 1. Universal Project Structure

We recommend the following file structure for your deployed agent. An example implementation is available in `examples/deployment/`.

```text
my-agent-project/
├── app/
│   ├── __init__.py
│   └── agent.py          # Core Agent Definition (Markov Graph)
├── main.py               # FastAPI Entry Point (Server)
├── requirements.txt      # Python Dependencies
└── Dockerfile            # Container definition
```

### 1.1 Core Agent Code (`app/agent.py`)

Implement the agent using `markov_agent`. Since `Graph` inherits from `google.adk.agents.Agent`, it is fully compatible with ADK deployment tools.

```python
# app/agent.py
from markov_agent.topology.graph import Graph
from markov_agent.engine.nodes import ProbabilisticNode
# Import your custom nodes and state...

# ... Define nodes and edges ...

agent = Graph(
    name="my_markov_agent",
    nodes=nodes,
    edges=edges,
    entry_point="start_node",
    state_type=MyState
)
```

### 1.2 Server Entry Point (`main.py`)

Wrap the agent in a FastAPI server using ADK's helper. This is required for Cloud Run and GKE.

```python
# main.py
import os
import uvicorn
from google.adk.cli.fast_api import get_fast_api_app
from app.agent import agent

# Create the FastAPI app from the Markov Graph (which is an ADK Agent)
app = get_fast_api_app(agent)

if __name__ == "__main__":
    # Use PORT environment variable for Cloud Run compatibility
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

### 1.3 Dependencies (`requirements.txt`)

```text
google-adk>=0.1.0
markov-agent>=0.1.0  # Ensure this is available
uvicorn
fastapi
```

### 1.4 Container Definition (`Dockerfile`)

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment variables
ENV PORT=8080

# Run the web server
CMD ["python", "main.py"]
```

---

## 2. Deployment Option A: Cloud Run (Serverless)

*Best for: Rapid deployment, auto-scaling, low maintenance.*

**Implementation Steps:**

1. **Build and Deploy:**
Run the following `gcloud` command from the project root.
```bash
gcloud run deploy my-markov-agent \
  --source . \
  --port 8080 \
  --allow-unauthenticated \
  --region us-central1
```

2. **Verification:**
The command will output a Service URL (e.g., `https://my-markov-agent-xyz.a.run.app`).
Test the health endpoint:
```bash
curl https://<YOUR_SERVICE_URL>/health
```

---

## 3. Deployment Option B: Vertex AI Agent Engine

*Best for: Fully managed agent infrastructure, state management, and strict enterprise governance.*

This method does **not** use the `Dockerfile` or `main.py`. It deploys the `agent` object directly using the Vertex AI SDK.

**Implementation Script (`deploy_vertex.py`):**

```python
# deploy_vertex.py
from vertexai.preview import agent_engines
from app.agent import agent

# 1. Initialize Agent Engine Client
# Ensure you have "Vertex AI Agent Engine API" enabled
client = agent_engines.AgentEngineClient()

# 2. Deploy
print("Deploying agent to Vertex AI Agent Engine...")
operation = client.create_agent_engine(
    agent=agent,
    display_name="my-production-agent",
    # Define runtime requirements directly here
    requirements=[
        "google-adk>=0.1.0",
        "markov-agent>=0.1.0"
    ]
)

# 3. Wait for completion
remote_agent = operation.result()
print(f"Agent deployed! Resource Name: {remote_agent.name}")
```

**Run Deployment:**

```bash
pip install google-cloud-aiplatform
python deploy_vertex.py
```

---

## 4. Deployment Option C: Google Kubernetes Engine (GKE)

*Best for: Full control over infrastructure, custom networking, VPC integration.*

**Implementation Steps:**

1. **Create Artifact Registry Repo:**
```bash
gcloud artifacts repositories create adk-repo \
    --repository-format=docker \
    --location=us-central1
```

2. **Build and Push Image:**
```bash
gcloud builds submit --tag us-central1-docker.pkg.dev/PROJECT_ID/adk-repo/my-agent:v1 .
```

3. **Define Kubernetes Manifest (`deployment.yaml`):**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: markov-agent
spec:
  replicas: 1
  selector:
    matchLabels:
      app: markov-agent
  template:
    metadata:
      labels:
        app: markov-agent
    spec:
      containers:
      - name: markov-agent
        image: us-central1-docker.pkg.dev/PROJECT_ID/adk-repo/my-agent:v1
        ports:
        - containerPort: 8080
        env:
        - name: PORT
          value: "8080"
---
apiVersion: v1
kind: Service
metadata:
  name: markov-agent-service
spec:
  selector:
    app: markov-agent
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
  type: LoadBalancer
```

4. **Apply to Cluster:**
```bash
kubectl apply -f deployment.yaml
```
