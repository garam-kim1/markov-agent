from typing import Any

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.run_config import RunConfig as ADKRunConfig
from google.adk.apps.app import App
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.auth.credential_service.in_memory_credential_service import (
    InMemoryCredentialService,
)
from google.adk.cli.adk_web_server import AdkWebServer as InternalAdkWebServer
from google.adk.cli.utils.base_agent_loader import BaseAgentLoader
from google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager
from google.adk.evaluation.local_eval_set_results_manager import (
    LocalEvalSetResultsManager,
)
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from pydantic import BaseModel, ConfigDict, Field


class RunConfig(BaseModel):
    """Runtime configuration for an agent run.

    Allows overriding model, tools, and providing user context for a specific execution.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any | None = None
    """The specific LLM model instance to use for this run."""

    tools: list[Any] = Field(default_factory=list)
    """A list of tool instances available to the agent for this run."""

    user_email: str | None = None
    """The email of the user invoking the agent."""

    history: list[Any] | None = None
    """Pre-existing chat history or session state to initialize the agent with."""

    streaming: bool = False
    """Whether to use streaming mode for this run."""

    def to_adk_run_config(self) -> ADKRunConfig:
        """Convert to the underlying ADK RunConfig."""
        from google.adk.agents.run_config import StreamingMode

        return ADKRunConfig(
            streaming_mode=(
                StreamingMode.SSE if self.streaming else StreamingMode.NONE
            )
        )


class SimpleAgentLoader(BaseAgentLoader):
    """Simple agent loader that returns a pre-initialized agent or app."""

    def __init__(self, agent_or_app: Any):
        self.agent_or_app = agent_or_app

    def load_agent(self, agent_name: str) -> BaseAgent | App:
        return self.agent_or_app

    def list_agents(self) -> list[str]:
        return ["default"]

    def list_agents_detailed(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "default",
                "root_agent_name": getattr(self.agent_or_app, "name", "agent"),
                "description": "Markov Agent",
                "language": "python",
            }
        ]


class AdkWebServer:
    """Web server wrapper for ADK agents.

    Exposes agent logic as a REST/WebSocket API.
    """

    def __init__(self, agent: Any):
        from markov_agent.engine.adk_wrapper import ADKController

        self.agent_instance = agent

        # Determine the actual ADK App or Agent
        if isinstance(agent, ADKController):
            self.app_or_agent = agent.app
        else:
            self.app_or_agent = agent

        # Setup internal ADK Web Server with in-memory services
        self.internal_server = InternalAdkWebServer(
            agent_loader=SimpleAgentLoader(self.app_or_agent),
            session_service=getattr(agent, "session_service", InMemorySessionService()),
            memory_service=getattr(agent, "memory_service", InMemoryMemoryService()),
            artifact_service=getattr(
                agent, "artifact_service", InMemoryArtifactService()
            ),
            credential_service=InMemoryCredentialService(),
            eval_sets_manager=InMemorySetsManager(),
            eval_set_results_manager=LocalEvalSetResultsManager(agents_dir="."),
            agents_dir=".",
        )

    def run(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        """Start the web server."""
        import uvicorn

        app = self.internal_server.get_fast_api_app()
        uvicorn.run(app, host=host, port=port)


class InMemorySetsManager(InMemoryEvalSetsManager):
    """Bridge for EvalSetsManager to avoid complex setup."""

    def list_eval_sets(self, app_name: str) -> list[str]:
        return []


__all__ = ["AdkWebServer", "RunConfig"]
