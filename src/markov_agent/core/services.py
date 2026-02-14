from google.adk.artifacts import BaseArtifactService, InMemoryArtifactService
from google.adk.memory import BaseMemoryService, InMemoryMemoryService
from google.adk.sessions import BaseSessionService, InMemorySessionService


class ServiceRegistry:
    """Global registry for ADK services to enable sharing across agents and nodes."""

    _artifact_service: BaseArtifactService | None = None
    _memory_service: BaseMemoryService | None = None
    _session_service: BaseSessionService | None = None

    @classmethod
    def set_artifact_service(cls, service: BaseArtifactService) -> None:
        """Set the global artifact service."""
        cls._artifact_service = service

    @classmethod
    def get_artifact_service(cls) -> BaseArtifactService:
        """Get the global artifact service, creating an in-memory one if none exists."""
        if cls._artifact_service is None:
            cls._artifact_service = InMemoryArtifactService()
        return cls._artifact_service

    @classmethod
    def set_memory_service(cls, service: BaseMemoryService) -> None:
        """Set the global memory service."""
        cls._memory_service = service

    @classmethod
    def get_memory_service(cls) -> BaseMemoryService:
        """Get the global memory service, creating an in-memory one if none exists."""
        if cls._memory_service is None:
            cls._memory_service = InMemoryMemoryService()
        return cls._memory_service

    @classmethod
    def set_session_service(cls, service: BaseSessionService) -> None:
        """Set the global session service."""
        cls._session_service = service

    @classmethod
    def get_session_service(cls) -> BaseSessionService:
        """Get the global session service, creating an in-memory one if none exists."""
        if cls._session_service is None:
            cls._session_service = InMemorySessionService()
        return cls._session_service

    @classmethod
    def use_shared_services(cls) -> None:
        """Force initialization of all shared services."""
        cls.get_artifact_service()
        cls.get_memory_service()
        cls.get_session_service()
