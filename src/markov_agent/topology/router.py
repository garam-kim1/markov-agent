from typing import Any

# ruff: noqa: S608
from pydantic import BaseModel, Field, create_model

from markov_agent.core.state import BaseState
from markov_agent.engine.ppu import ProbabilisticNode


class RoutingDecision(BaseModel):
    """The standard output format for a semantic router."""

    target: str = Field(..., description="The name of the selected target node.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)."
    )
    reasoning: str = Field(
        ..., description="Brief justification for the routing decision."
    )


class RouterNode(ProbabilisticNode):
    """A specialized ProbabilisticNode that acts as a Semantic Router.

    It analyzes the state and selects the best next node from a set of options
    based on their natural language descriptions.
    """

    routes: dict[str, str] = Field(default_factory=dict)

    def __init__(
        self,
        name: str,
        routes: dict[str, str],
        state_type: type[BaseState] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the RouterNode.

        Args:
            name: The name of this router node.
            routes: A dictionary mapping target node names to their descriptions.
                    Example: {"search_tool": "If the user asks for information",
                              "end_conversation": "If the user says goodbye"}
            state_type: The state class used by the graph.
            **kwargs: Additional arguments for ProbabilisticNode.

        """
        options = list(routes.keys())

        from enum import Enum

        # 1. Dynamically create a Pydantic model with an Enum for valid targets
        # This constrains the LLM to only output valid node names.
        # Inherit from str to make it JSON serializable and string-comparable
        TargetEnum = Enum(f"TargetEnum_{name}", {k: k for k in options}, type=str)

        # We need a unique name for the model to avoid registry conflicts
        model_name = f"RoutingDecision_{name}"

        decision_model = create_model(
            model_name,
            target=(TargetEnum, Field(..., description="The selected target node.")),
            confidence=(float, Field(..., ge=0.0, le=1.0)),
            reasoning=(str, Field(..., description="Justification for the decision.")),
            __base__=BaseModel,
        )
        # Workaround to mark it as a dict proxy if needed, though ADK usually handles Pydantic models fine.
        # But ProbabilisticNode logic checks _is_dict_proxy for some things.
        # getattr(decision_model, "_is_dict_proxy", False) is used in PPU.

        # 2. Construct the Routing Prompt
        route_list = "\n".join([f"- '{k}': {v}" for k, v in routes.items()])
        prompt_template = (
            "You are a semantic routing system. Analyze the current state and "
            "select the most appropriate next step from the following options:\n\n"
            f"{route_list}\n\n"
            "Current State:\n"
            "{{ state }}\n\n"
            "Determine the target, your confidence (0.0-1.0), and the reasoning."
        )
        # 3. Initialize Parent
        # We pass routes here so Pydantic sets the field
        super().__init__(
            name=name,
            prompt_template=prompt_template,
            output_schema=decision_model,
            state_type=state_type,
            routes=routes,
            **kwargs,
        )

    def parse_result(self, state: BaseState, result: Any) -> BaseState:
        """Parse the routing decision and update the state's metadata.

        We don't overwrite the main state fields; instead, we store the
        routing decision in `state.meta['routing']`. This allows downstream
        edges to read it.
        """
        # Ensure result is a model or dict
        if isinstance(result, BaseModel):
            # Use JSON mode to ensure Enums are serialized to values (str)
            data = result.model_dump(mode="json")
        else:
            data = result

        target = data.get("target")

        confidence = data.get("confidence", 1.0)
        reasoning = data.get("reasoning", "")

        # Store decision in meta for edges to consume
        if "routing" not in state.meta:
            state.meta["routing"] = {}

        state.meta["routing"][self.name] = {
            "target": target,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        # Also update global reasoning if present
        if reasoning:
            state.meta["reasoning"] = reasoning

        # Record probability for entropy tracking
        # Ideally, we'd have the full distribution from the LLM (logprobs),
        # but here we at least record the point estimate.
        state.record_probability(
            source=self.name, target=target, probability=confidence
        )

        # Record history
        state.record_step(
            {
                "node": self.name,
                "decision": target,
                "confidence": confidence,
                "reasoning": reasoning,
            }
        )

        return state
