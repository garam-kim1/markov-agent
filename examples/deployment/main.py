import os

import uvicorn
from app.agent import agent
from google.adk.cli.fast_api import get_fast_api_app

# Create the FastAPI app from the ADK agent (Graph)
app = get_fast_api_app(agent)

if __name__ == "__main__":
    # Use PORT environment variable for Cloud Run compatibility
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
