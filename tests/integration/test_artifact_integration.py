import uuid

import pytest
from google.adk.agents.invocation_context import InvocationContext
from google.adk.artifacts import InMemoryArtifactService
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
from google.genai import types


# Define a custom tool that uses artifacts
class ArtifactGenTool:
    def generate_report(self, filename: str, content: str, tool_context: ToolContext | None = None) -> str:
        """
        Generates a report and saves it as an artifact.
        """
        if not tool_context:
            return "Error: No tool context"
            
        # Create artifact part
        # part = types.Part(text=content)
        
        # Save using the helper method on ToolContext (which calls artifact_service)
        # Note: ToolContext.save_artifact is a wrapper that usually delegates 
        # to context.artifact_service.save_artifact or similar.
        # Based on inspection, ToolContext has save_artifact.
        
        # We need to await it if it's async. Inspection showed it is a method on CallbackContext.
        # But wait, is ToolContext.save_artifact async?
        # Telemetry plugin inspection showed callbacks are async.
        # Let's assume usage pattern:
        
        # Actually, let's verify if save_artifact is async in the test itself or check inspection again.
        # The inspection showed: 
        # save_artifact: <function CallbackContext.save_artifact at ...>
        # It didn't explicitly say "async function" but usually ADK I/O is async.
        # InMemoryArtifactService.save_artifact IS async (we awaited it in previous test).
        # So ToolContext.save_artifact likely is too.
        
        # However, tools are often synchronous functions. 
        # If the tool is sync, it can't await.
        # But ADK supports async tools.
        
        return "Dummy report"
        # We will implement the logic in the test function to handle async nature if needed,
        # or implement this method as async.

    async def generate_report_async(self, filename: str, content: str, tool_context: ToolContext | None = None) -> str:
        if not tool_context:
            return "Error: No tool context"
            
        part = types.Part(text=content)
        
        # We assume tool_context.save_artifact is available.
        # We need to pass app_name etc?
        # CallbackContext.save_artifact signature from inspect_artifacts.py (on service) was:
        # (app_name, user_id, filename, artifact, session_id)
        
        # CallbackContext.save_artifact might simplify this by using context info.
        # Let's inspect signature of CallbackContext.save_artifact if possible.
        # But we saw it in inspect_tool_context.py.
        
        await tool_context.save_artifact(filename=filename, artifact=part)
        return f"Saved {filename}"

@pytest.mark.asyncio
async def test_tool_artifact_integration():
    # 1. Setup Services
    artifact_service = InMemoryArtifactService()
    session_service = InMemorySessionService()
    
    # 2. Setup Session
    session_id = str(uuid.uuid4())
    user_id = "test_user"
    app_name = "markov_agent"
    
    # Create session explicitly so it exists
    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state={}
    )
    
    session = await session_service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id
    )
    assert session is not None
    
    # 3. Setup Invocation Context
    # We need a dummy agent
    from google.adk.agents import Agent
    # Mock agent
    agent = Agent(name="test_agent", model="mock")
    
    context = InvocationContext(
        session=session,
        session_service=session_service,
        invocation_id="inv_123",
        agent=agent,
        artifact_service=artifact_service
    )
    
    # 4. Setup ToolContext
    tool_context = ToolContext(invocation_context=context)
    
    # 5. Execute Tool Logic
    tool = ArtifactGenTool()
    filename = "report.txt"
    content = "This is a generated report."
    
    result = await tool.generate_report_async(filename, content, tool_context)
    assert result == f"Saved {filename}"
    
    # 6. Verify Artifact exists in service
    # We check the service directly
    keys = await artifact_service.list_artifact_keys(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id
    )
    assert filename in keys
    
    loaded = await artifact_service.load_artifact(
        app_name=app_name,
        user_id=user_id,
        filename=filename,
        session_id=session_id
    )
    assert loaded is not None
    assert loaded.text == content

