from typing import Any

import pytest
from google.adk.artifacts import InMemoryArtifactService

from markov_agent.core.services import ServiceRegistry
from markov_agent.engine.adk_wrapper import ADKConfig, ADKController, RetryPolicy


@pytest.mark.asyncio
async def test_shared_artifact_service():
    # 1. Setup shared service
    shared_artifact = InMemoryArtifactService()
    ServiceRegistry.set_artifact_service(shared_artifact)

    # 2. Create two controllers
    config1 = ADKConfig(model_name="mock-model")
    config2 = ADKConfig(model_name="mock-model")

    ctrl1 = ADKController(config1, RetryPolicy())
    ctrl2 = ADKController(config2, RetryPolicy())

    # 3. Verify they use the same service
    assert ctrl1.artifact_service is shared_artifact
    assert ctrl2.artifact_service is shared_artifact

    # 4. Save artifact in one, list in another
    # Note: save_artifact is async
    from google.genai.types import Part

    part = Part(text="hello world")

    await ctrl1.artifact_service.save_artifact(
        app_name="test",
        user_id="user",
        session_id="session",
        filename="test.txt",
        artifact=part,
    )

    keys = await ctrl2.artifact_service.list_artifact_keys(
        app_name="test", user_id="user", session_id="session"
    )
    assert "test.txt" in keys


@pytest.mark.asyncio
async def test_shared_memory_service():
    # 1. Setup shared service
    from google.adk.memory import InMemoryMemoryService

    shared_memory = InMemoryMemoryService()
    ServiceRegistry.set_memory_service(shared_memory)

    # 2. Create two controllers with memory enabled
    config1 = ADKConfig(model_name="mock-model", enable_memory=True)
    config2 = ADKConfig(model_name="mock-model", enable_memory=True)

    ctrl1 = ADKController(config1, RetryPolicy())
    ctrl2 = ADKController(config2, RetryPolicy())

    # 3. Verify they use the same service
    assert ctrl1.memory_service is shared_memory
    assert ctrl2.memory_service is shared_memory


@pytest.mark.asyncio
async def test_graph_node_service_inheritance():
    import uuid

    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.sessions import InMemorySessionService, Session

    from markov_agent.engine.ppu import ProbabilisticNode
    from markov_agent.topology.graph import Graph

    # 1. Setup shared artifact service
    shared_artifact = InMemoryArtifactService()
    ServiceRegistry.set_artifact_service(shared_artifact)

    # 2. Create Node
    node = ProbabilisticNode(name="test_node", prompt_template="hello")

    # 3. Create Graph
    graph = Graph(name="test_graph")
    graph.add_node(node)

    # 4. Mock InvocationContext with shared service
    session = Session(id="s1", app_name="app", user_id="u1", state={})
    _ = InvocationContext(
        session=session,
        session_service=InMemorySessionService(),
        invocation_id=str(uuid.uuid4()),
        agent=graph,
        artifact_service=shared_artifact,
    )

    # 5. Run node implementation directly to check if it uses ctx services
    # We use a mock generate that we can spy on if possible,
    # but let's just check if create_variant was called with correct artifact_service

    # Actually, we can check if the task factories created use the shared_artifact
    # We need to render prompt first to call _create_task_factories
    prompt = "test prompt"
    state_dict = {}
    varied_configs = [{}]

    factories = node._create_task_factories(
        prompt, state_dict, varied_configs, artifact_service=shared_artifact
    )

    # Inspect the closure of the first factory
    # This is a bit hacky but works for verification
    task: Any = factories[0]
    # task is 'make_task' inside _create_task_factories
    # it has 'c' (controller_to_use) as a default argument
    controller = task.__defaults__[0]  # type: ignore
    assert controller.artifact_service is shared_artifact
