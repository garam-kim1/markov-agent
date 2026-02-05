from pydantic import BaseModel, Field

from markov_agent.engine.adk_wrapper import ADKConfig, ADKController, RetryPolicy
from markov_agent.engine.prompt import PromptEngine


class SimulatorState(BaseModel):
    persona: str
    goal: str
    history: list[dict[str, str]] = Field(default_factory=list)
    last_agent_response: str | None = None


class UserSimulator:
    """Simulates a user in a conversation loop."""

    def __init__(
        self,
        persona: str,
        goal: str,
        adk_config: ADKConfig,
        termination_signal: str = "[GOAL_MET]",
    ):
        self.state = SimulatorState(persona=persona, goal=goal)
        self.adk_config = adk_config
        self.termination_signal = termination_signal
        self.prompt_engine = PromptEngine()

        # We use ADKController directly to drive the simulator LLM
        self.controller = ADKController(
            config=self.adk_config,
            retry_policy=RetryPolicy(max_attempts=3),
        )

        self.system_prompt_template = """
You are playing the role of a user in a conversation with an AI agent.
Your Persona: {{ persona }}
Your Goal: {{ goal }}

Instructions:
1. Act according to your persona.
2. Interact with the agent to achieve your goal.
3. If the agent successfully satisfies your goal, output exactly:
   {{ termination_signal }}
4. Do not break character.
5. Provide realistic inputs, including potential ambiguity if consistent with persona.

Conversation History:
{% for turn in history %}
{{ turn.role }}: {{ turn.content }}
{% endfor %}

Agent's Last Response:
{{ last_agent_response }}

Your Response:
"""

    async def generate_next_turn(self, agent_response: str) -> str:
        """Generates the next user input based on the agent's response."""
        # Update history with agent's response
        if agent_response:
            self.state.history.append({"role": "Agent", "content": agent_response})

        self.state.last_agent_response = agent_response

        # Render prompt
        prompt = self.prompt_engine.render(
            self.system_prompt_template,
            persona=self.state.persona,
            goal=self.state.goal,
            history=self.state.history,
            last_agent_response=agent_response,
            termination_signal=self.termination_signal,
        )

        # Generate response
        response = await self.controller.generate(prompt=prompt)

        # Extract text content from response
        # ADKController returns a ModelResponse object or str depending on config.
        # Assuming string or simple object for now.
        content = ""
        if isinstance(response, str):
            content = response
        elif hasattr(response, "text"):
            content = response.text
        elif isinstance(response, BaseModel):
            # Try to dump if it's a structured model
            content = response.model_dump_json()
        else:
            content = str(response)

        # Clean up response
        if not isinstance(content, str):
            content = str(content)
        content = content.strip()

        # Update history with user's response (unless it's the termination signal,
        # but usually we want to record that too or handle it in the runner)
        # We record it here for context continuity
        if content != self.termination_signal:
            self.state.history.append({"role": "User", "content": content})

        return content
