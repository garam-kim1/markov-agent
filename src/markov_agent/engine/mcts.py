import math
from collections.abc import AsyncGenerator, Callable
from typing import Any, TypeVar, cast

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types
from pydantic import BaseModel, Field

from markov_agent.core.state import BaseState
from markov_agent.engine.adk_wrapper import ADKConfig, ADKController, RetryPolicy
from markov_agent.topology.node import BaseNode

StateT = TypeVar("StateT", bound=BaseState)


class MCTSInternalNode(BaseModel):
    """An internal node in the MCTS tree."""

    state: Any
    parent: Any | None = None
    children: list["MCTSInternalNode"] = Field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    action: Any | None = None

    def uct(self, exploration_weight: float) -> float:
        if self.visits == 0:
            return float("inf")
        if self.parent is None:
            return self.value / self.visits
        return (self.value / self.visits) + exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )


class MCTSNode(BaseNode[StateT]):
    """A node that performs MCTS-based Inference-Time Search.

    Implements System 2 Deliberation with a formal Value Function and Discounting.
    """

    adk_config: ADKConfig
    max_rollouts: int = Field(default=10)
    exploration_weight: float = Field(default=1.414)
    value_estimator: Callable[[Any], float] | None = Field(default=None)
    expansion_k: int = Field(default=3)
    simulation_config: ADKConfig | None = Field(default=None)
    gamma: float = Field(default=0.9, description="Discount factor for future rewards.")

    def __init__(
        self,
        name: str,
        adk_config: ADKConfig,
        max_rollouts: int = 10,
        exploration_weight: float = 1.414,
        value_estimator: Callable[[Any], float] | None = None,
        expansion_k: int = 3,
        simulation_config: ADKConfig | None = None,
        state_type: type[StateT] | None = None,
        mock_responder: Callable[[str], Any] | None = None,
        gamma: float = 0.9,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, adk_config=adk_config, state_type=state_type, **kwargs
        )
        self.adk_config = adk_config
        self.max_rollouts = max_rollouts
        self.exploration_weight = exploration_weight
        self.value_estimator = value_estimator
        self.expansion_k = expansion_k
        self.simulation_config = simulation_config
        self.gamma = gamma
        self.controller = ADKController(
            self.adk_config, RetryPolicy(), mock_responder=mock_responder
        )
        if self.simulation_config:
            self.simulation_controller = ADKController(
                self.simulation_config, RetryPolicy(), mock_responder=mock_responder
            )
        else:
            self.simulation_controller = self.controller

    async def execute(self, state: StateT) -> StateT:
        """Perform MCTS to find the best next state."""
        root = MCTSInternalNode(state=state)

        for _ in range(self.max_rollouts):
            # 1. Selection
            leaf = await self._select(root)

            # 2. Expansion
            await self._expand(leaf)

            if leaf.children:
                child = leaf.children[0]  # Simple: select first new child
                # 3. Simulation (Rollout)
                reward = await self._simulate(child)
                # 4. Backpropagation
                self._backpropagate(child, reward)
            else:
                # Terminal or expansion failed
                reward = await self._simulate(leaf)
                self._backpropagate(leaf, reward)

        # Selection best child based on visits
        if not root.children:
            return state

        best_child = max(root.children, key=lambda c: c.visits)
        # Update state with the best action/result
        # This assumes the action is a partial state update or the new state itself
        if isinstance(best_child.action, dict):
            return state.update(**best_child.action)
        if isinstance(best_child.action, BaseState):
            return best_child.action

        return best_child.state

    async def _select(self, node: MCTSInternalNode) -> MCTSInternalNode:
        while node.children:
            node = max(node.children, key=lambda c: c.uct(self.exploration_weight))
        return node

    async def _expand(self, node: MCTSInternalNode) -> None:
        """Generate k possible next steps."""
        prompt = f"""Given the current state: {node.state}
Generate {self.expansion_k} diverse possible next steps or actions to take. Return them as a JSON list of objects, each representing a partial state update."""

        try:
            results = await self.controller.generate(
                prompt,
                output_schema=None,  # We'll parse it manually or expect JSON
            )
            # Minimal parsing for demo purposes - in real system use output_schema
            import json

            # Clean markdown
            if isinstance(results, str):
                cleaned = results.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned.split("```json")[1].split("```")[0].strip()
                elif cleaned.startswith("```"):
                    cleaned = cleaned.split("```")[1].split("```")[0].strip()

                actions = json.loads(cleaned)
                if isinstance(actions, list):
                    for action in actions[: self.expansion_k]:
                        new_state = (
                            node.state.update(**action)
                            if hasattr(node.state, "update")
                            else {**node.state, **action}
                        )
                        child = MCTSInternalNode(
                            state=new_state, parent=node, action=action
                        )
                        node.children.append(child)
        except Exception:  # noqa: S110
            # Expansion might fail (e.g. terminal state)
            pass

    async def _simulate(self, node: MCTSInternalNode) -> float:
        """Rollout the state to estimate its value."""
        if self.value_estimator:
            return self.value_estimator(node.state)

        # Default simulation: Ask the LLM to score the state
        prompt = f"""Evaluate the quality of this state: {node.state}
Rate it from 0.0 (Fail) to 1.0 (Success). Return only the number."""
        try:
            score_result = await self.simulation_controller.generate(prompt)
            score_text = str(score_result)
            return float(score_text.strip())
        except Exception:
            return 0.5

    def _backpropagate(self, node: MCTSInternalNode, reward: float) -> None:
        curr = node
        depth = 0
        while curr is not None:
            curr.visits += 1
            # Apply discount factor: V(s) = reward * gamma^depth
            discounted_reward = reward * (self.gamma**depth)
            curr.value += discounted_reward
            curr = curr.parent
            depth += 1

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        # Implementation for ADK runtime
        state_dict = ctx.session.state
        if self.state_type:
            state_obj = self.state_type.model_validate(state_dict)
        else:
            state_obj = cast("StateT", state_dict)

        new_state = await self.execute(state_obj)

        if hasattr(new_state, "model_dump"):
            ctx.session.state.update(new_state.model_dump())
        elif isinstance(new_state, dict):
            ctx.session.state.update(new_state)

        yield Event(
            author=self.name,
            actions=EventActions(),
            content=types.Content(
                role="model", parts=[types.Part(text=f"MCTS completed for {self.name}")]
            ),
        )
