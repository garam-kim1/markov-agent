# deploy_agent_engine.py
from vertexai.preview import agent_engines  # type: ignore

from examples.deployment.app.agent import agent

# 1. Initialize Agent Engine Client
# Ensure you have "Vertex AI Agent Engine API" enabled
# and have run `gcloud auth login` and `gcloud config set project <PROJECT_ID>`
client = agent_engines.AgentEngineClient()

# 2. Deploy
print("Deploying markov-agent to Vertex AI Agent Engine...")
operation = client.create_agent_engine(
    agent=agent,
    display_name="markov-production-agent",
    # Define runtime requirements directly here
    requirements=[
        "google-adk>=0.1.0",
        # "markov-agent>=0.1.0" # Add this if published
    ],
)

# 3. Wait for completion
remote_agent = operation.result()
print(f"Agent deployed! Resource Name: {remote_agent.name}")
