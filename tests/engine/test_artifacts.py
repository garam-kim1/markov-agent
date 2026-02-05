import pytest
from google.adk.agents.invocation_context import InvocationContext
from google.adk.artifacts import InMemoryArtifactService
from google.genai import types

from markov_agent.engine.adk_wrapper import ADKConfig
from markov_agent.engine.ppu import ProbabilisticNode
from markov_agent.topology.graph import Graph


class MockArtifactService(InMemoryArtifactService):
    pass


@pytest.fixture
def mock_service():
    return MockArtifactService()


def test_artifact_service_wiring_node_init(mock_service):
    """Verifies that if we pass an artifact service to ProbabilisticNode init,
    it is correctly wired into the controller and runner.
    """
    config = ADKConfig(model_name="mock")
    node = ProbabilisticNode(
        name="test_node",
        adk_config=config,
        prompt_template="foo",
        artifact_service=mock_service,
    )

    assert node.artifact_service is mock_service
    assert node.controller.artifact_service is mock_service
    # Runner is created inside controller
    assert node.controller.runner.artifact_service is mock_service


@pytest.mark.asyncio
async def test_artifact_creation_manual():
    """Verifies that we can manually save and load artifacts using the service,
    simulating what a tool would do.
    """
    service = InMemoryArtifactService()

    # Simulate saving an artifact (e.g. from a tool)
    artifact_content = types.Part(text="Hello Artifact World")

    # Use internal API as discovered
    await service.save_artifact(
        app_name="markov_agent",
        user_id="test_user",
        filename="hello.txt",
        artifact=artifact_content,
        session_id="session_123",
    )

    # Verify we can list it
    keys = await service.list_artifact_keys(
        app_name="markov_agent",
        user_id="test_user",
        session_id="session_123",
    )
    assert "hello.txt" in keys

    # Verify we can load it
    loaded_part = await service.load_artifact(
        app_name="markov_agent",
        user_id="test_user",
        filename="hello.txt",
        session_id="session_123",
    )
    assert loaded_part is not None
    assert loaded_part.text == "Hello Artifact World"


@pytest.mark.asyncio
async def test_graph_propagates_service_to_context():
    """Verifies that Graph.run passes the artifact service to the invocation context.
    Note: This does NOT verify that nodes use it (they might ignore it),
    but it checks the Graph's contract.
    """
    mock_service = MockArtifactService()

    # Create a simple graph
    graph = Graph(
        name="test_graph",
        nodes={},
        edges=[],
        entry_point="start",
    )

    # We mock _run_async_impl to capture the context
    # original_impl = graph._run_async_impl
    captured_ctx = None

    async def mock_run(ctx: InvocationContext):
        nonlocal captured_ctx
        captured_ctx = ctx
        # Make it an empty async generator
        return
        yield

    graph._run_async_impl = mock_run  # type: ignore

    from markov_agent.core.state import BaseState

    state = BaseState()

    await graph.run(state=state, artifact_service=mock_service)

    assert captured_ctx is not None
    assert captured_ctx.artifact_service is mock_service
